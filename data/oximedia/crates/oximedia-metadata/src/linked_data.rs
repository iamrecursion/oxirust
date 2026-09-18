//! Linked data / RDF-like concepts: subject-predicate-object triples,
//! namespace prefixes, and simple graph traversal.

#![allow(dead_code)]

use std::collections::HashMap;

/// A namespace prefix binding (e.g. "dc" -> "<http://purl.org/dc/elements/1.1/>").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceBinding {
    /// Short prefix (e.g. "dc").
    pub prefix: String,
    /// Full namespace URI (e.g. "<http://purl.org/dc/elements/1.1/>").
    pub uri: String,
}

impl NamespaceBinding {
    /// Create a new namespace binding.
    #[must_use]
    pub fn new(prefix: &str, uri: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            uri: uri.to_string(),
        }
    }
}

/// Registry of namespace prefix bindings.
#[derive(Debug, Default, Clone)]
pub struct NamespaceRegistry {
    bindings: HashMap<String, String>,
}

impl NamespaceRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Register a namespace prefix.
    pub fn register(&mut self, binding: NamespaceBinding) {
        self.bindings.insert(binding.prefix, binding.uri);
    }

    /// Register a prefix/uri pair directly.
    pub fn register_prefix(&mut self, prefix: &str, uri: &str) {
        self.bindings.insert(prefix.to_string(), uri.to_string());
    }

    /// Resolve a prefixed name like "dc:title" to its full URI.
    /// Returns `None` if the prefix is not registered.
    #[must_use]
    pub fn resolve(&self, prefixed: &str) -> Option<String> {
        let (prefix, local) = prefixed.split_once(':')?;
        let ns = self.bindings.get(prefix)?;
        Some(format!("{}{}", ns, local))
    }

    /// Compact a full URI back to its prefixed form.
    /// Returns `None` if no matching prefix is registered.
    #[must_use]
    pub fn compact(&self, uri: &str) -> Option<String> {
        for (prefix, ns) in &self.bindings {
            if let Some(local) = uri.strip_prefix(ns.as_str()) {
                return Some(format!("{}:{}", prefix, local));
            }
        }
        None
    }

    /// Return the number of registered bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Return `true` if no bindings are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// The object part of an RDF-like triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripleObject {
    /// A URI (IRI) resource reference.
    Uri(String),
    /// A plain string literal.
    Literal(String),
    /// A typed literal with datatype URI.
    TypedLiteral { value: String, datatype: String },
    /// A language-tagged literal.
    LangLiteral { value: String, lang: String },
}

impl TripleObject {
    /// Create a URI object.
    #[must_use]
    pub fn uri(uri: &str) -> Self {
        Self::Uri(uri.to_string())
    }

    /// Create a plain literal object.
    #[must_use]
    pub fn literal(value: &str) -> Self {
        Self::Literal(value.to_string())
    }

    /// Create a typed literal object.
    #[must_use]
    pub fn typed(value: &str, datatype: &str) -> Self {
        Self::TypedLiteral {
            value: value.to_string(),
            datatype: datatype.to_string(),
        }
    }

    /// Create a language-tagged literal object.
    #[must_use]
    pub fn lang(value: &str, lang: &str) -> Self {
        Self::LangLiteral {
            value: value.to_string(),
            lang: lang.to_string(),
        }
    }

    /// Return the string value regardless of variant.
    #[must_use]
    pub fn value(&self) -> &str {
        match self {
            Self::Uri(u) => u,
            Self::Literal(l) => l,
            Self::TypedLiteral { value, .. } => value,
            Self::LangLiteral { value, .. } => value,
        }
    }

    /// Return `true` if this is a URI resource.
    #[must_use]
    pub fn is_uri(&self) -> bool {
        matches!(self, Self::Uri(_))
    }

    /// Return `true` if this is any kind of literal.
    #[must_use]
    pub fn is_literal(&self) -> bool {
        !self.is_uri()
    }
}

/// An RDF-like triple: subject–predicate–object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triple {
    /// Subject URI.
    pub subject: String,
    /// Predicate URI.
    pub predicate: String,
    /// Object value.
    pub object: TripleObject,
}

impl Triple {
    /// Create a new triple.
    #[must_use]
    pub fn new(subject: &str, predicate: &str, object: TripleObject) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object,
        }
    }
}

/// A simple in-memory RDF-like graph storing triples.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    triples: Vec<Triple>,
}

impl Graph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    /// Add a triple to the graph.
    pub fn add(&mut self, triple: Triple) {
        self.triples.push(triple);
    }

    /// Add a shorthand triple (subject, predicate, literal object).
    pub fn add_literal(&mut self, subject: &str, predicate: &str, value: &str) {
        self.add(Triple::new(
            subject,
            predicate,
            TripleObject::literal(value),
        ));
    }

    /// Return the number of triples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the graph contains no triples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Find all triples with the given subject.
    #[must_use]
    pub fn triples_for_subject<'a>(&'a self, subject: &str) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject)
            .collect()
    }

    /// Find all triples with the given predicate.
    #[must_use]
    pub fn triples_for_predicate<'a>(&'a self, predicate: &str) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| t.predicate == predicate)
            .collect()
    }

    /// Find all object values for a given subject/predicate pair.
    #[must_use]
    pub fn objects_for(&self, subject: &str, predicate: &str) -> Vec<&TripleObject> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject && t.predicate == predicate)
            .map(|t| &t.object)
            .collect()
    }

    /// Remove all triples matching the subject. Returns removal count.
    pub fn remove_subject(&mut self, subject: &str) -> usize {
        let before = self.triples.len();
        self.triples.retain(|t| t.subject != subject);
        before - self.triples.len()
    }

    /// Traverse: return all unique subjects that have `predicate` pointing to `object_uri`.
    #[must_use]
    pub fn subjects_with_object_uri(&self, predicate: &str, object_uri: &str) -> Vec<&str> {
        self.triples
            .iter()
            .filter(|t| {
                t.predicate == predicate
                    && matches!(&t.object, TripleObject::Uri(u) if u == object_uri)
            })
            .map(|t| t.subject.as_str())
            .collect()
    }

    /// Return all triples as a slice.
    #[must_use]
    pub fn all_triples(&self) -> &[Triple] {
        &self.triples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_register_and_resolve() {
        let mut reg = NamespaceRegistry::new();
        reg.register_prefix("dc", "http://purl.org/dc/elements/1.1/");
        let uri = reg.resolve("dc:title").expect("should succeed in test");
        assert_eq!(uri, "http://purl.org/dc/elements/1.1/title");
    }

    #[test]
    fn test_namespace_resolve_unknown_prefix() {
        let reg = NamespaceRegistry::new();
        assert!(reg.resolve("dc:title").is_none());
    }

    #[test]
    fn test_namespace_compact() {
        let mut reg = NamespaceRegistry::new();
        reg.register_prefix("dc", "http://purl.org/dc/");
        let compact = reg
            .compact("http://purl.org/dc/title")
            .expect("should succeed in test");
        assert_eq!(compact, "dc:title");
    }

    #[test]
    fn test_namespace_compact_no_match() {
        let reg = NamespaceRegistry::new();
        assert!(reg.compact("http://example.com/foo").is_none());
    }

    #[test]
    fn test_namespace_len_and_is_empty() {
        let mut reg = NamespaceRegistry::new();
        assert!(reg.is_empty());
        reg.register_prefix("dc", "http://purl.org/dc/");
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_triple_object_value() {
        assert_eq!(
            TripleObject::uri("http://ex.com/").value(),
            "http://ex.com/"
        );
        assert_eq!(TripleObject::literal("hello").value(), "hello");
        assert_eq!(TripleObject::typed("42", "xsd:int").value(), "42");
        assert_eq!(TripleObject::lang("Bonjour", "fr").value(), "Bonjour");
    }

    #[test]
    fn test_triple_object_is_uri() {
        assert!(TripleObject::uri("u").is_uri());
        assert!(!TripleObject::literal("l").is_uri());
        assert!(!TripleObject::typed("v", "t").is_uri());
        assert!(!TripleObject::lang("v", "en").is_uri());
    }

    #[test]
    fn test_triple_object_is_literal() {
        assert!(TripleObject::literal("x").is_literal());
        assert!(!TripleObject::uri("x").is_literal());
    }

    #[test]
    fn test_graph_add_and_len() {
        let mut g = Graph::new();
        assert!(g.is_empty());
        g.add_literal("s", "p", "o");
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_graph_triples_for_subject() {
        let mut g = Graph::new();
        g.add_literal("s1", "p", "o1");
        g.add_literal("s1", "p2", "o2");
        g.add_literal("s2", "p", "o3");
        let found = g.triples_for_subject("s1");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_graph_triples_for_predicate() {
        let mut g = Graph::new();
        g.add_literal("s1", "dc:title", "Title 1");
        g.add_literal("s2", "dc:title", "Title 2");
        g.add_literal("s3", "dc:creator", "Author");
        let found = g.triples_for_predicate("dc:title");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_graph_objects_for() {
        let mut g = Graph::new();
        g.add_literal("s", "pred", "value1");
        g.add_literal("s", "pred", "value2");
        let objs = g.objects_for("s", "pred");
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn test_graph_remove_subject() {
        let mut g = Graph::new();
        g.add_literal("s1", "p", "o");
        g.add_literal("s1", "p2", "o2");
        g.add_literal("s2", "p", "o3");
        let removed = g.remove_subject("s1");
        assert_eq!(removed, 2);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_graph_subjects_with_object_uri() {
        let mut g = Graph::new();
        g.add(Triple::new("doc1", "type", TripleObject::uri("MediaFile")));
        g.add(Triple::new("doc2", "type", TripleObject::uri("MediaFile")));
        g.add(Triple::new("doc3", "type", TripleObject::uri("Image")));
        let subjects = g.subjects_with_object_uri("type", "MediaFile");
        assert_eq!(subjects.len(), 2);
        assert!(subjects.contains(&"doc1"));
        assert!(subjects.contains(&"doc2"));
    }

    #[test]
    fn test_namespace_binding_new() {
        let binding = NamespaceBinding::new("xmp", "http://ns.adobe.com/xap/1.0/");
        assert_eq!(binding.prefix, "xmp");
        assert_eq!(binding.uri, "http://ns.adobe.com/xap/1.0/");
    }

    #[test]
    fn test_graph_all_triples() {
        let mut g = Graph::new();
        g.add_literal("s", "p", "o");
        assert_eq!(g.all_triples().len(), 1);
    }
}
