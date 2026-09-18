//! The *schema representation constraints* that are purely structural: which
//! attributes an element of the schema vocabulary may carry, which values
//! those attributes may take, and in which order and how often its children
//! may appear.
//!
//! [XML Schema Part 1 §3](https://www.w3.org/TR/xmlschema-1/) states these as
//! the "schema for schemas" plus a handful of prose rules. Checking them up
//! front means the component builders can read attributes without re-checking
//! that they exist, and it catches the large family of schema documents that
//! are wrong before any component is built at all.
//!
//! The table itself lives in [`rules`]; the attribute-value grammars live in
//! [`values`].

mod rules;
mod values;

use std::collections::BTreeSet;

use oxixml_dom::{Document, NodeId};

use crate::error::SchemaError;
use crate::names::{XSD_NS, XsdVersion};
use crate::parse::{child_elements, node_path, xsd_name};

use rules::{Context, ONLY_1_1, Rule, matches, rule};
use values::AttrType;

/// The attributes XML Schema 1.1 added, which a 1.0 processor must reject.
///
/// [XML Schema 1.1 Part 1 §2.5](https://www.w3.org/TR/xmlschema11-1/) lists
/// these under the features that are new in 1.1; a 1.0 schema document that
/// carries one is in error by the 1.0 schema for schemas, which simply has no
/// such attribute.
const ONLY_1_1_ATTRIBUTES: &[(&str, &str)] = &[
    ("schema", "defaultAttributes"),
    ("schema", "xpathDefaultNamespace"),
    ("element", "targetNamespace"),
    ("attribute", "targetNamespace"),
    ("attribute", "inheritable"),
    ("complexType", "defaultAttributesApply"),
    ("any", "notNamespace"),
    ("any", "notQName"),
    ("anyAttribute", "notNamespace"),
    ("anyAttribute", "notQName"),
    ("selector", "xpathDefaultNamespace"),
    ("field", "xpathDefaultNamespace"),
    ("unique", "ref"),
    ("key", "ref"),
    ("keyref", "ref"),
];

/// Checks the whole of one schema document.
///
/// # Errors
///
/// Returns the first `src-…` violation found, in document order.
pub(crate) fn check_document(
    doc: &Document,
    root: NodeId,
    version: XsdVersion,
) -> Result<(), SchemaError> {
    let mut identifiers = BTreeSet::new();
    check_element(doc, root, Context::TopLevel, version, &mut identifiers)
}

/// The context an element of the vocabulary gives its children.
fn child_context(name: &str) -> Context {
    match name {
        "schema" | "redefine" | "override" => Context::TopLevel,
        "simpleType" => Context::SimpleType,
        "simpleContent" => Context::SimpleContent,
        "complexContent" => Context::ComplexContent,
        _ => Context::Local,
    }
}

fn check_element(
    doc: &Document,
    node: NodeId,
    context: Context,
    version: XsdVersion,
    identifiers: &mut BTreeSet<String>,
) -> Result<(), SchemaError> {
    let Some(name) = xsd_name(doc, node) else {
        return Ok(());
    };
    if !version.is_1_1() && ONLY_1_1.contains(&name) {
        return Err(SchemaError::new(
            "src-element",
            format!("`xs:{name}` is an XML Schema 1.1 element"),
        )
        .at(node_path(doc, node)));
    }
    let Some(rule) = rule(name, context) else {
        return Err(SchemaError::new(
            "src-element",
            format!("`xs:{name}` is not part of the schema vocabulary here"),
        )
        .at(node_path(doc, node)));
    };

    check_attributes(doc, node, name, &rule, version, identifiers)?;

    // §3.8.6 *cos-all-limited.1*: under 1.0 an `xs:all` group occurs exactly
    // once (or not at all); 1.1 relaxes `maxOccurs` to `0` as well
    // (`msData/modelGroups/mgO001`).
    if name == "all"
        && !version.is_1_1()
        && !matches!(
            occurrence(doc, node, "maxOccurs").as_deref(),
            None | Some("1")
        )
    {
        return Err(SchemaError::new(
            "cos-all-limited.1",
            "an `xs:all` group must have `maxOccurs=\"1\"`",
        )
        .at(node_path(doc, node)));
    }

    // The content of `appinfo` and `documentation` is unconstrained.
    if matches!(name, "appinfo" | "documentation") {
        return Ok(());
    }

    check_children(doc, node, name, context, &rule, version, identifiers)
}

fn check_attributes(
    doc: &Document,
    node: NodeId,
    name: &str,
    rule: &Rule,
    version: XsdVersion,
    identifiers: &mut BTreeSet<String>,
) -> Result<(), SchemaError> {
    for attribute in doc.attributes(node) {
        let Some(data) = doc.attr(attribute) else {
            continue;
        };
        match data.namespace_uri() {
            // Attributes from another namespace are allowed everywhere; the
            // namespace declarations themselves are not attributes of the
            // vocabulary at all.
            Some(XSD_NS) => {
                return Err(SchemaError::new(
                    "src-element",
                    format!(
                        "`{}` may not be qualified with the XML Schema namespace",
                        data.qualified_name()
                    ),
                )
                .at(node_path(doc, node)));
            }
            Some(_) => continue,
            None => {}
        }
        let local = data.local_name();
        if local == "xmlns" {
            continue;
        }
        if local == "id" {
            check_id(doc, node, data.value(), identifiers)?;
            continue;
        }
        let Some((_, kind)) = rule
            .attributes
            .iter()
            .find(|(candidate, _)| *candidate == local)
        else {
            return Err(SchemaError::new(
                "src-element",
                format!("`xs:{name}` has no attribute `{local}`"),
            )
            .at(node_path(doc, node)));
        };
        if !version.is_1_1() && ONLY_1_1_ATTRIBUTES.contains(&(name, local)) {
            return Err(SchemaError::new(
                "src-element",
                format!("`{local}` on `xs:{name}` is an XML Schema 1.1 attribute"),
            )
            .at(node_path(doc, node)));
        }
        // XSD 1.1 §3.16.2 added `extension` to the `final` of a simple type
        // definition, which 1.0 restricted to `list`, `union` and
        // `restriction`.
        let kind = if version.is_1_1() && name == "simpleType" && local == "final" {
            values::AttrType::DerivationSet(&["list", "union", "restriction", "extension"])
        } else {
            *kind
        };
        if let Some(reason) = values::check(kind, data.value(), version) {
            return Err(SchemaError::new(
                "src-attribute-value",
                format!("`{local}` on `xs:{name}`: {reason}"),
            )
            .at(node_path(doc, node)));
        }
        // `substitutionGroup` became a *list* of QNames only in 1.1.
        if !version.is_1_1()
            && local == "substitutionGroup"
            && crate::value::collapse(data.value()).contains(' ')
        {
            return Err(SchemaError::new(
                "src-attribute-value",
                "`substitutionGroup` names more than one head, which needs XML Schema 1.1",
            )
            .at(node_path(doc, node)));
        }
    }
    for required in rule.required {
        if doc.get_attribute_ns(node, None, required).is_none() {
            return Err(SchemaError::new(
                "src-element",
                format!("`xs:{name}` needs a `{required}` attribute"),
            )
            .at(node_path(doc, node)));
        }
    }
    Ok(())
}

/// `id` is `xs:ID`: an `NCName`, and unique across the schema document.
fn check_id(
    doc: &Document,
    node: NodeId,
    raw: &str,
    identifiers: &mut BTreeSet<String>,
) -> Result<(), SchemaError> {
    if let Some(reason) = values::check(AttrType::Id, raw, XsdVersion::V1_0) {
        return Err(
            SchemaError::new("src-attribute-value", format!("`id`: {reason}"))
                .at(node_path(doc, node)),
        );
    }
    let value = crate::value::collapse(raw);
    if !identifiers.insert(value.clone()) {
        return Err(SchemaError::new(
            "src-attribute-value",
            format!("the `id` value `{value}` is used more than once in this schema document"),
        )
        .at(node_path(doc, node)));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_children(
    doc: &Document,
    node: NodeId,
    name: &str,
    own_context: Context,
    rule: &Rule,
    version: XsdVersion,
    identifiers: &mut BTreeSet<String>,
) -> Result<(), SchemaError> {
    // `xs:schema`, `xs:redefine` and `xs:override` allow annotations anywhere
    // among their children; everywhere else exactly one may lead.
    let annotations_anywhere = matches!(name, "schema" | "redefine" | "override");
    let context = child_context(name);
    let mut seen_non_annotation = false;
    let mut annotations = 0;
    let mut sequence: Vec<&str> = Vec::new();
    for child in child_elements(doc, node) {
        let Some(child_name) = xsd_name(doc, child) else {
            return Err(SchemaError::new(
                "src-element",
                "an element from another namespace may only appear inside `xs:appinfo` or \
                 `xs:documentation`",
            )
            .at(node_path(doc, child)));
        };
        if child_name == "annotation" && name != "annotation" {
            annotations += 1;
            if !annotations_anywhere && (seen_non_annotation || annotations > 1) {
                return Err(SchemaError::new(
                    "src-annotation",
                    "`xs:annotation` may appear only once, as the first child",
                )
                .at(node_path(doc, child)));
            }
            check_element(doc, child, Context::Local, version, identifiers)?;
            continue;
        }
        seen_non_annotation = true;
        sequence.push(child_name);
        // 1.0 restricts `xs:all` to element particles; 1.1 lifts that.
        if name == "all" && !version.is_1_1() && child_name != "element" {
            return Err(SchemaError::new(
                "src-element",
                format!("`xs:{child_name}` may appear inside `xs:all` only in XML Schema 1.1"),
            )
            .at(node_path(doc, child)));
        }
        // §3.8.6 *cos-all-limited.2*: under 1.0 every member of an `xs:all`
        // occurs at most once (`msData/additional/addB095`).
        if name == "all"
            && !version.is_1_1()
            && !matches!(
                occurrence(doc, child, "maxOccurs").as_deref(),
                None | Some("0") | Some("1")
            )
        {
            return Err(SchemaError::new(
                "cos-all-limited.2",
                "a member of an `xs:all` group may occur at most once",
            )
            .at(node_path(doc, child)));
        }
        // The schema for schemas types the model group inside a *named*
        // `xs:group` as `simpleExplicitGroup`, which fixes both occurrence
        // attributes to 1 (`msData/additional/addB053`).
        if name == "group"
            && own_context == Context::TopLevel
            && (occurrence(doc, child, "minOccurs").is_some()
                || occurrence(doc, child, "maxOccurs").is_some())
        {
            return Err(SchemaError::new(
                "src-element",
                "the model group of a named `xs:group` may not carry `minOccurs` or `maxOccurs`",
            )
            .at(node_path(doc, child)));
        }
        check_element(doc, child, context, version, identifiers)?;
    }
    if let Some(offender) = matches(rule.content, &sequence) {
        return Err(SchemaError::new(
            "src-element",
            format!("`xs:{offender}` may not appear here inside `xs:{name}`"),
        )
        .at(node_path(doc, node)));
    }
    Ok(())
}

/// The written value of `minOccurs`/`maxOccurs` on a particle, collapsed.
fn occurrence(doc: &Document, node: NodeId, name: &str) -> Option<String> {
    doc.get_attribute_ns(node, None, name)
        .map(crate::value::collapse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(text: &str) -> Result<(), SchemaError> {
        let doc = oxixml_dom::build::parse_str(text).expect("well-formed");
        let root = doc.document_element().expect("a root");
        check_document(&doc, root, XsdVersion::V1_0)
    }

    #[test]
    fn a_well_formed_schema_passes() -> Result<(), SchemaError> {
        check(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                 <xs:annotation><xs:documentation>hi<b/></xs:documentation></xs:annotation>
                 <xs:element name="a" type="xs:string"/>
               </xs:schema>"#,
        )
    }

    #[test]
    fn unknown_attributes_are_rejected() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:element name="a" bogus="1"/>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn foreign_attributes_are_allowed() -> Result<(), SchemaError> {
        check(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:p="urn:x">
                 <xs:element name="a" p:note="1"/>
               </xs:schema>"#,
        )
    }

    #[test]
    fn top_level_elements_have_no_occurrence_attributes() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:element name="a" minOccurs="0"/>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn annotation_must_come_first() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:element name="a">
                       <xs:complexType/>
                       <xs:annotation/>
                     </xs:element>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn version_1_1_elements_are_rejected_under_1_0() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:complexType name="t"><xs:assert test="true()"/></xs:complexType>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn identifiers_are_unique_across_the_document() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:attribute id="dup" name="a"/>
                     <xs:complexType id="dup" name="t"/>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn identifiers_must_be_ncnames() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:complexType name="t"><xs:sequence id=""/></xs:complexType>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn attributes_may_not_precede_the_model_group() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:complexType name="t">
                       <xs:attribute name="a"/>
                       <xs:sequence/>
                     </xs:complexType>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn one_content_model_per_complex_type() {
        assert!(
            check(
                r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                     <xs:complexType name="t">
                       <xs:simpleContent><xs:extension base="xs:string"/></xs:simpleContent>
                       <xs:simpleContent><xs:extension base="xs:string"/></xs:simpleContent>
                     </xs:complexType>
                   </xs:schema>"#,
            )
            .is_err()
        );
    }

    #[test]
    fn wildcard_namespace_lists_are_uris() {
        assert!(
            check(&format!(
                "<xs:schema xmlns:xs=\"http://www.w3.org/2001/XMLSchema\">\
                       <xs:complexType name=\"t\">\
                         <xs:anyAttribute namespace=\"{}anyAttribute\"/>\
                       </xs:complexType>\
                     </xs:schema>",
                "##"
            ),)
            .is_err()
        );
    }
}
