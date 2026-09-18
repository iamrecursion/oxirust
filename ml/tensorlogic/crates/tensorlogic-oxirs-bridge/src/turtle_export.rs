//! Turtle (.ttl) export for TensorLogic SymbolTable.
//!
//! Converts a [`SymbolTable`] into a valid Turtle serialization, emitting
//! `rdfs:Class` declarations for every domain and `rdf:Property` declarations
//! for every predicate.

use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;

use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};

use crate::error::BridgeError;

/// Exports a [`SymbolTable`] to Turtle (`.ttl`) format.
///
/// # Examples
///
/// ```
/// use tensorlogic_oxirs_bridge::TurtleExporter;
/// use tensorlogic_adapters::SymbolTable;
///
/// let exporter = TurtleExporter::new("http://example.org/");
/// let table = SymbolTable::new();
/// let ttl = exporter.export_symbol_table(&table).unwrap();
/// assert!(ttl.contains("@prefix rdf:"));
/// ```
pub struct TurtleExporter {
    /// Base IRI used to mint IRIs for domains and predicates.
    pub base_iri: String,
    /// Extra prefix declarations `(prefix_name, namespace_iri)`.
    pub prefixes: Vec<(String, String)>,
}

impl TurtleExporter {
    /// Create a new exporter with the given base IRI.
    pub fn new(base_iri: impl Into<String>) -> Self {
        Self {
            base_iri: base_iri.into(),
            prefixes: Vec::new(),
        }
    }

    /// Attach an additional prefix declaration.
    pub fn with_prefix(mut self, prefix: &str, namespace: &str) -> Self {
        self.prefixes
            .push((prefix.to_owned(), namespace.to_owned()));
        self
    }

    /// Export `table` as a Turtle string.
    ///
    /// The output always includes the standard `rdf:`, `rdfs:`, `xsd:`, and
    /// `owl:` prefix declarations followed by class and property stanzas.
    pub fn export_symbol_table(&self, table: &SymbolTable) -> Result<String, BridgeError> {
        let mut out = String::new();

        self.write_prefixes(&mut out);

        for (name, info) in &table.domains {
            self.write_class(name, info, &mut out);
        }

        for (name, info) in &table.predicates {
            self.write_property(name, info, &mut out);
        }

        Ok(out)
    }

    /// Serialise `table` and write it to `path`, creating the file if needed.
    pub fn write_to_file(
        &self,
        table: &SymbolTable,
        path: &std::path::Path,
    ) -> Result<(), BridgeError> {
        let content = self.export_symbol_table(table)?;
        let mut file = std::fs::File::create(path).map_err(|e| {
            BridgeError::InvalidSchema(format!("Cannot create file {}: {}", path.display(), e))
        })?;
        file.write_all(content.as_bytes()).map_err(|e| {
            BridgeError::InvalidSchema(format!("Cannot write to file {}: {}", path.display(), e))
        })?;
        Ok(())
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Wrap `iri` in angle-brackets, escaping characters that are illegal
    /// inside an IRI reference.
    fn escape_iri(iri: &str) -> String {
        let escaped = iri
            .replace('\\', "\\\\")
            .replace('<', "%3C")
            .replace('>', "%3E")
            .replace('"', "%22")
            .replace('{', "%7B")
            .replace('}', "%7D")
            .replace('|', "%7C")
            .replace('^', "%5E")
            .replace('`', "%60")
            .replace(' ', "%20");
        format!("<{}>", escaped)
    }

    /// Wrap `s` in double-quotes, escaping backslashes and double-quote chars.
    fn escape_literal(s: &str) -> String {
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{}\"", escaped)
    }

    /// Emit the standard prefix block plus any caller-supplied extras.
    fn write_prefixes(&self, out: &mut String) {
        // Standard prefixes
        writeln!(
            out,
            "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        )
        .ok();
        writeln!(
            out,
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ."
        )
        .ok();
        writeln!(out, "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .").ok();
        writeln!(out, "@prefix owl:  <http://www.w3.org/2002/07/owl#> .").ok();

        // Caller-supplied extras
        for (prefix, ns) in &self.prefixes {
            writeln!(out, "@prefix {}: {} .", prefix, Self::escape_iri(ns)).ok();
        }

        writeln!(out).ok(); // blank line separator
    }

    /// Emit a `rdfs:Class` stanza for `class_name`.
    fn write_class(&self, class_name: &str, _info: &DomainInfo, out: &mut String) {
        let iri = Self::escape_iri(&format!("{}{}", self.base_iri, class_name));
        let label = Self::escape_literal(class_name);
        writeln!(out, "{} a rdfs:Class ;", iri).ok();
        writeln!(out, "    rdfs:label {} .", label).ok();
        writeln!(out).ok();
    }

    /// Emit a `rdf:Property` stanza for `pred_name`.
    fn write_property(&self, pred_name: &str, info: &PredicateInfo, out: &mut String) {
        let iri = Self::escape_iri(&format!("{}{}", self.base_iri, pred_name));
        let label = Self::escape_literal(pred_name);

        writeln!(out, "{} a rdf:Property ;", iri).ok();
        writeln!(out, "    rdfs:label {} ;", label).ok();

        if let Some(domain_name) = info.arg_domains.first() {
            let domain_iri = Self::escape_iri(&format!("{}{}", self.base_iri, domain_name));
            writeln!(out, "    rdfs:domain {} ;", domain_iri).ok();
        }

        writeln!(out, "    rdfs:range xsd:string .").ok();
        writeln!(out).ok();
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_adapters::{DomainInfo, PredicateInfo};

    fn make_table_empty() -> SymbolTable {
        SymbolTable::new()
    }

    fn make_table_with_domain() -> SymbolTable {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Person", 0)).unwrap();
        t
    }

    fn make_table_with_predicate() -> SymbolTable {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Person", 0)).unwrap();
        t.add_predicate(PredicateInfo::new("name", vec!["Person".to_owned()]))
            .unwrap();
        t
    }

    #[test]
    fn test_export_empty_symbol_table_has_prefixes() {
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp.export_symbol_table(&make_table_empty()).unwrap();
        assert!(ttl.contains("@prefix rdf:"), "missing rdf prefix");
        assert!(ttl.contains("@prefix rdfs:"), "missing rdfs prefix");
        assert!(ttl.contains("@prefix xsd:"), "missing xsd prefix");
        assert!(ttl.contains("@prefix owl:"), "missing owl prefix");
    }

    #[test]
    fn test_export_single_domain_as_class() {
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp.export_symbol_table(&make_table_with_domain()).unwrap();
        assert!(ttl.contains("rdfs:Class"), "expected rdfs:Class");
        assert!(ttl.contains("Person"), "expected class name");
    }

    #[test]
    fn test_export_single_predicate_as_property() {
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp
            .export_symbol_table(&make_table_with_predicate())
            .unwrap();
        assert!(ttl.contains("rdf:Property"), "expected rdf:Property");
        assert!(ttl.contains("name"), "expected predicate name");
    }

    #[test]
    fn test_export_property_with_domain() {
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp
            .export_symbol_table(&make_table_with_predicate())
            .unwrap();
        assert!(ttl.contains("rdfs:domain"), "expected rdfs:domain");
        assert!(ttl.contains("Person"), "expected domain IRI fragment");
    }

    #[test]
    fn test_escape_literal_special_chars() {
        let escaped = TurtleExporter::escape_literal("say \"hello\" and \\goodbye");
        assert!(escaped.contains("\\\""), "quote not escaped");
        assert!(escaped.contains("\\\\"), "backslash not escaped");
    }

    #[test]
    fn test_export_multiple_domains() {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Dog", 0)).unwrap();
        t.add_domain(DomainInfo::new("Cat", 0)).unwrap();
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp.export_symbol_table(&t).unwrap();
        assert!(ttl.contains("Dog"), "expected Dog");
        assert!(ttl.contains("Cat"), "expected Cat");
    }

    #[test]
    fn test_export_multiple_predicates() {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Entity", 0)).unwrap();
        t.add_predicate(PredicateInfo::new("likes", vec!["Entity".to_owned()]))
            .unwrap();
        t.add_predicate(PredicateInfo::new("dislikes", vec!["Entity".to_owned()]))
            .unwrap();
        let exp = TurtleExporter::new("http://example.org/");
        let ttl = exp.export_symbol_table(&t).unwrap();
        assert!(ttl.contains("likes"), "expected predicate 'likes'");
        assert!(ttl.contains("dislikes"), "expected predicate 'dislikes'");
    }

    #[test]
    fn test_output_contains_base_iri() {
        let base = "http://mybase.org/onto/";
        let exp = TurtleExporter::new(base);
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Widget", 0)).unwrap();
        let ttl = exp.export_symbol_table(&t).unwrap();
        assert!(
            ttl.contains(base),
            "output should contain the configured base IRI"
        );
    }

    #[test]
    fn test_write_to_file_creates_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_turtle_export_output.ttl");
        let exp = TurtleExporter::new("http://example.org/");
        exp.write_to_file(&make_table_with_domain(), &path).unwrap();
        assert!(path.exists(), "file should have been created");
        // cleanup
        let _ = std::fs::remove_file(&path);
    }
}
