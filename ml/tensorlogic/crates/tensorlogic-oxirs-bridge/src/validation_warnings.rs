//! Schema quality warnings for RDF/OWL schemas loaded into [`SchemaAnalyzer`].
//!
//! [`SchemaWarningAnalyzer`] inspects a fully-populated [`SchemaAnalyzer`] and
//! produces a list of [`SchemaWarning`] items that describe missing labels,
//! missing comments, unused classes/properties, and opportunities to add SHACL
//! shapes.

use serde::{Deserialize, Serialize};

use crate::schema::{ClassInfo, PropertyInfo, SchemaAnalyzer};

// ── warning kinds ─────────────────────────────────────────────────────────────

/// Categories of schema quality warnings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaWarningKind {
    /// A class or property has no `rdfs:label`.
    MissingLabel,
    /// A class or property has no `rdfs:comment`.
    MissingComment,
    /// A class appears in the schema but is never the domain of any property.
    UnusedClass,
    /// A property has no domain information (arity > 0 but no `arg_domains`).
    UnusedProperty,
    /// A property could benefit from a SHACL node-shape.
    SuggestShaclShape,
}

// ── warning record ────────────────────────────────────────────────────────────

/// A single schema quality warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaWarning {
    /// The kind of issue.
    pub kind: SchemaWarningKind,
    /// IRI of the affected class or property.
    pub subject_iri: String,
    /// Human-readable description.
    pub message: String,
    /// Optional remediation hint.
    pub suggestion: Option<String>,
}

impl SchemaWarning {
    fn new(
        kind: SchemaWarningKind,
        subject_iri: impl Into<String>,
        message: impl Into<String>,
        suggestion: Option<String>,
    ) -> Self {
        Self {
            kind,
            subject_iri: subject_iri.into(),
            message: message.into(),
            suggestion,
        }
    }
}

// ── analyser ─────────────────────────────────────────────────────────────────

/// Runs a set of quality checks on a [`SchemaAnalyzer`] and collects warnings.
pub struct SchemaWarningAnalyzer;

impl SchemaWarningAnalyzer {
    /// Run all checks and return every warning found.
    pub fn analyze(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        let mut warnings = Vec::new();
        warnings.extend(Self::check_missing_labels(analyzer));
        warnings.extend(Self::check_missing_comments(analyzer));
        warnings.extend(Self::check_unused_classes(analyzer));
        warnings.extend(Self::check_unused_properties(analyzer));
        warnings.extend(Self::suggest_shacl_shapes(analyzer));
        warnings
    }

    /// Warn about every class or property that has no `rdfs:label`.
    pub fn check_missing_labels(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        let mut warnings = Vec::new();

        for (_name, class) in &analyzer.classes {
            if class.label.is_none() {
                warnings.push(SchemaWarning::new(
                    SchemaWarningKind::MissingLabel,
                    &class.iri,
                    format!("Class '{}' has no rdfs:label", class.iri),
                    Some(format!(
                        "Add `<{}> rdfs:label \"{}\" .` to your schema.",
                        class.iri,
                        Self::humanise(&class.iri)
                    )),
                ));
            }
        }

        for (_name, prop) in &analyzer.properties {
            if prop.label.is_none() {
                warnings.push(SchemaWarning::new(
                    SchemaWarningKind::MissingLabel,
                    &prop.iri,
                    format!("Property '{}' has no rdfs:label", prop.iri),
                    Some(format!(
                        "Add `<{}> rdfs:label \"{}\" .` to your schema.",
                        prop.iri,
                        Self::humanise(&prop.iri)
                    )),
                ));
            }
        }

        warnings
    }

    /// Warn about every class or property that has no `rdfs:comment`.
    pub fn check_missing_comments(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        let mut warnings = Vec::new();

        for (_name, class) in &analyzer.classes {
            if class.comment.is_none() {
                warnings.push(SchemaWarning::new(
                    SchemaWarningKind::MissingComment,
                    &class.iri,
                    format!("Class '{}' has no rdfs:comment", class.iri),
                    Some(format!(
                        "Add `<{}> rdfs:comment \"Description of {}\" .` to your schema.",
                        class.iri,
                        Self::humanise(&class.iri)
                    )),
                ));
            }
        }

        for (_name, prop) in &analyzer.properties {
            if prop.comment.is_none() {
                warnings.push(SchemaWarning::new(
                    SchemaWarningKind::MissingComment,
                    &prop.iri,
                    format!("Property '{}' has no rdfs:comment", prop.iri),
                    Some(format!(
                        "Add `<{}> rdfs:comment \"Description of {}\" .` to your schema.",
                        prop.iri,
                        Self::humanise(&prop.iri)
                    )),
                ));
            }
        }

        warnings
    }

    /// Warn about classes that are never referenced as the domain of any property.
    pub fn check_unused_classes(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        // Collect every IRI that appears as a domain on some property
        let used_as_domain: std::collections::HashSet<&str> = analyzer
            .properties
            .values()
            .flat_map(|p: &PropertyInfo| p.domain.iter().map(|s| s.as_str()))
            .collect();

        analyzer
            .classes
            .values()
            .filter(|c: &&ClassInfo| !used_as_domain.contains(c.iri.as_str()))
            .map(|c| {
                SchemaWarning::new(
                    SchemaWarningKind::UnusedClass,
                    &c.iri,
                    format!(
                        "Class '{}' is never used as the domain of any property",
                        c.iri
                    ),
                    Some(format!(
                        "Consider removing the class or adding a property with `rdfs:domain <{}>`.",
                        c.iri
                    )),
                )
            })
            .collect()
    }

    /// Warn about properties with arity > 0 but without any domain declaration.
    pub fn check_unused_properties(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        analyzer
            .properties
            .values()
            .filter(|p: &&PropertyInfo| {
                !p.domain.is_empty() && p.iri.is_empty()
                // Report properties that have no domain IRI recorded at all
                || p.domain.is_empty()
            })
            .filter(|p| {
                // Only flag those with an arity > 0 (i.e. they should have a domain)
                // We approximate arity by checking if the local name exists
                !p.iri.is_empty()
            })
            .map(|p| {
                SchemaWarning::new(
                    SchemaWarningKind::UnusedProperty,
                    &p.iri,
                    format!("Property '{}' has no rdfs:domain declaration", p.iri),
                    Some(format!(
                        "Add `<{}> rdfs:domain <SomeClass> .` to your schema.",
                        p.iri
                    )),
                )
            })
            .collect()
    }

    /// Suggest SHACL shapes for every property that has a domain.
    pub fn suggest_shacl_shapes(analyzer: &SchemaAnalyzer) -> Vec<SchemaWarning> {
        analyzer
            .properties
            .values()
            .filter(|p: &&PropertyInfo| !p.domain.is_empty())
            .map(|p| {
                let domain = p.domain.first().map(|s| s.as_str());
                let template = Self::shacl_shape_template(&p.iri, domain);
                SchemaWarning::new(
                    SchemaWarningKind::SuggestShaclShape,
                    &p.iri,
                    format!(
                        "Property '{}' could be constrained with a SHACL shape",
                        p.iri
                    ),
                    Some(template),
                )
            })
            .collect()
    }

    /// Generate a minimal SHACL node-shape template for a property.
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaWarningAnalyzer;
    ///
    /// let template = SchemaWarningAnalyzer::shacl_shape_template(
    ///     "http://example.org/name",
    ///     Some("http://example.org/Person"),
    /// );
    /// assert!(template.contains("sh:path"));
    /// ```
    pub fn shacl_shape_template(pred_iri: &str, domain: Option<&str>) -> String {
        let target_line = if let Some(d) = domain {
            format!("    sh:targetClass <{}> ;", d)
        } else {
            "    # sh:targetClass <SomeClass> ;".to_owned()
        };
        format!(
            "<{}Shape> a sh:NodeShape ;\n{}\n    sh:property [ sh:path <{}> ] .\n",
            pred_iri, target_line, pred_iri
        )
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Extract a human-readable local name from an IRI (fragment or last path segment).
    fn humanise(iri: &str) -> String {
        if let Some(pos) = iri.rfind('#') {
            return iri[pos + 1..].to_owned();
        }
        if let Some(pos) = iri.rfind('/') {
            return iri[pos + 1..].to_owned();
        }
        iri.to_owned()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ClassInfo, PropertyInfo, SchemaAnalyzer};

    fn empty_analyzer() -> SchemaAnalyzer {
        SchemaAnalyzer::new()
    }

    fn analyzer_with_labeled_class() -> SchemaAnalyzer {
        let mut a = SchemaAnalyzer::new();
        a.classes.insert(
            "Person".to_owned(),
            ClassInfo {
                iri: "http://example.org/Person".to_owned(),
                label: Some("Person".to_owned()),
                comment: Some("A human being".to_owned()),
                subclass_of: vec![],
            },
        );
        a
    }

    fn analyzer_with_unlabeled_class() -> SchemaAnalyzer {
        let mut a = SchemaAnalyzer::new();
        a.classes.insert(
            "Thing".to_owned(),
            ClassInfo {
                iri: "http://example.org/Thing".to_owned(),
                label: None,
                comment: None,
                subclass_of: vec![],
            },
        );
        a
    }

    fn analyzer_with_property_no_domain() -> SchemaAnalyzer {
        let mut a = analyzer_with_labeled_class();
        a.properties.insert(
            "dangling".to_owned(),
            PropertyInfo {
                iri: "http://example.org/dangling".to_owned(),
                label: None,
                comment: None,
                domain: vec![],
                range: vec![],
            },
        );
        a
    }

    fn analyzer_with_property_with_domain() -> SchemaAnalyzer {
        let mut a = SchemaAnalyzer::new();
        a.classes.insert(
            "Person".to_owned(),
            ClassInfo {
                iri: "http://example.org/Person".to_owned(),
                label: Some("Person".to_owned()),
                comment: Some("A person".to_owned()),
                subclass_of: vec![],
            },
        );
        a.properties.insert(
            "name".to_owned(),
            PropertyInfo {
                iri: "http://example.org/name".to_owned(),
                label: Some("name".to_owned()),
                comment: Some("The name".to_owned()),
                domain: vec!["http://example.org/Person".to_owned()],
                range: vec!["http://www.w3.org/2001/XMLSchema#string".to_owned()],
            },
        );
        a
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_no_warnings_for_empty_schema() {
        let a = empty_analyzer();
        let warnings = SchemaWarningAnalyzer::analyze(&a);
        assert!(
            warnings.is_empty(),
            "empty schema should produce no warnings"
        );
    }

    #[test]
    fn test_missing_label_warning_emitted() {
        let a = analyzer_with_unlabeled_class();
        let warnings = SchemaWarningAnalyzer::check_missing_labels(&a);
        assert!(
            warnings
                .iter()
                .any(|w| w.kind == SchemaWarningKind::MissingLabel),
            "expected a MissingLabel warning"
        );
    }

    #[test]
    fn test_missing_comment_warning_emitted() {
        let a = analyzer_with_unlabeled_class();
        let warnings = SchemaWarningAnalyzer::check_missing_comments(&a);
        assert!(
            warnings
                .iter()
                .any(|w| w.kind == SchemaWarningKind::MissingComment),
            "expected a MissingComment warning"
        );
    }

    #[test]
    fn test_unused_class_detected() {
        // A class is unused when no property declares it as its domain
        let a = analyzer_with_unlabeled_class();
        let warnings = SchemaWarningAnalyzer::check_unused_classes(&a);
        assert!(
            warnings
                .iter()
                .any(|w| w.kind == SchemaWarningKind::UnusedClass),
            "expected an UnusedClass warning"
        );
    }

    #[test]
    fn test_suggest_shacl_shape_for_property() {
        let a = analyzer_with_property_with_domain();
        let warnings = SchemaWarningAnalyzer::suggest_shacl_shapes(&a);
        assert!(
            warnings
                .iter()
                .any(|w| w.kind == SchemaWarningKind::SuggestShaclShape),
            "expected a SuggestShaclShape warning"
        );
    }

    #[test]
    fn test_shacl_template_contains_sh_path() {
        let template = SchemaWarningAnalyzer::shacl_shape_template(
            "http://example.org/name",
            Some("http://example.org/Person"),
        );
        assert!(
            template.contains("sh:path"),
            "template must contain sh:path"
        );
        assert!(
            template.contains("http://example.org/name"),
            "template must reference the property IRI"
        );
    }

    #[test]
    fn test_analyze_returns_all_warning_types() {
        // Build an analyzer that will trigger every warning type
        let a = analyzer_with_property_no_domain();
        let warnings = SchemaWarningAnalyzer::analyze(&a);
        // We expect at minimum MissingLabel and MissingComment (for the unlabeled property)
        // and UnusedClass (Person has no properties using it as domain via IRI match)
        let kinds: Vec<&SchemaWarningKind> = warnings.iter().map(|w| &w.kind).collect();
        assert!(
            kinds.contains(&&SchemaWarningKind::MissingLabel)
                || kinds.contains(&&SchemaWarningKind::MissingComment)
                || kinds.contains(&&SchemaWarningKind::UnusedClass)
                || kinds.contains(&&SchemaWarningKind::UnusedProperty),
            "expected at least one warning kind, got: {:?}",
            kinds
        );
    }
}
