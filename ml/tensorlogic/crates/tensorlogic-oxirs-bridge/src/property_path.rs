//! SPARQL 1.1 property path query support.
//!
//! Implements property path expressions for traversing RDF graphs with
//! sequence, alternative, closure (zero-or-more, one-or-more), and
//! inverse path patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Errors from property path evaluation.
#[derive(Debug, Error)]
pub enum PathError {
    #[error("Max depth {0} exceeded for closure path")]
    MaxDepthExceeded(usize),
    #[error("Empty path")]
    EmptyPath,
    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

/// A SPARQL 1.1 property path expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyPath {
    /// A simple IRI predicate: `foaf:knows`
    Iri(String),
    /// Sequence path: `a / b` (a then b)
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative path: `a | b` (a or b)
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Zero or more: `a*`
    ZeroOrMore(Box<PropertyPath>),
    /// One or more: `a+`
    OneOrMore(Box<PropertyPath>),
    /// Zero or one: `a?`
    ZeroOrOne(Box<PropertyPath>),
    /// Inverse path: `^a` (reverse direction)
    Inverse(Box<PropertyPath>),
}

impl PropertyPath {
    /// Create a simple IRI property path.
    pub fn iri(s: impl Into<String>) -> Self {
        PropertyPath::Iri(s.into())
    }

    /// Create a sequence path: `a / b`.
    pub fn seq(a: PropertyPath, b: PropertyPath) -> Self {
        PropertyPath::Sequence(Box::new(a), Box::new(b))
    }

    /// Create an alternative path: `a | b`.
    pub fn alt(a: PropertyPath, b: PropertyPath) -> Self {
        PropertyPath::Alternative(Box::new(a), Box::new(b))
    }

    /// Create a zero-or-more closure path: `a*`.
    pub fn zero_or_more(p: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(p))
    }

    /// Create a one-or-more closure path: `a+`.
    pub fn one_or_more(p: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(p))
    }

    /// Create a zero-or-one optional path: `a?`.
    pub fn zero_or_one(p: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(p))
    }

    /// Create an inverse path: `^a`.
    pub fn inverse(p: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(p))
    }

    /// Nesting depth of the path expression.
    pub fn depth(&self) -> usize {
        match self {
            PropertyPath::Iri(_) => 1,
            PropertyPath::Sequence(a, b) | PropertyPath::Alternative(a, b) => {
                1 + a.depth().max(b.depth())
            }
            PropertyPath::ZeroOrMore(p)
            | PropertyPath::OneOrMore(p)
            | PropertyPath::ZeroOrOne(p)
            | PropertyPath::Inverse(p) => 1 + p.depth(),
        }
    }

    /// Whether this path contains a closure operator (`*` or `+`).
    pub fn contains_closure(&self) -> bool {
        match self {
            PropertyPath::ZeroOrMore(_) | PropertyPath::OneOrMore(_) => true,
            PropertyPath::Sequence(a, b) | PropertyPath::Alternative(a, b) => {
                a.contains_closure() || b.contains_closure()
            }
            PropertyPath::ZeroOrOne(p) | PropertyPath::Inverse(p) => p.contains_closure(),
            PropertyPath::Iri(_) => false,
        }
    }

    /// Whether this path is a simple IRI (no operators).
    pub fn is_simple(&self) -> bool {
        matches!(self, PropertyPath::Iri(_))
    }
}

impl std::fmt::Display for PropertyPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyPath::Iri(s) => write!(f, "{}", s),
            PropertyPath::Sequence(a, b) => write!(f, "{}/{}", a, b),
            PropertyPath::Alternative(a, b) => write!(f, "{}|{}", a, b),
            PropertyPath::ZeroOrMore(p) => write!(f, "{}*", p),
            PropertyPath::OneOrMore(p) => write!(f, "{}+", p),
            PropertyPath::ZeroOrOne(p) => write!(f, "{}?", p),
            PropertyPath::Inverse(p) => write!(f, "^{}", p),
        }
    }
}

/// Simple in-memory triple store for property path evaluation.
#[derive(Debug, Clone, Default)]
pub struct TripleStore {
    triples: Vec<(String, String, String)>,
}

impl TripleStore {
    /// Create an empty triple store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple (subject, predicate, object) to the store.
    pub fn add(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) {
        self.triples
            .push((subject.into(), predicate.into(), object.into()));
    }

    /// Get all (subject, object) pairs for a given predicate.
    pub fn triples_with_predicate(&self, pred: &str) -> Vec<(&str, &str)> {
        self.triples
            .iter()
            .filter(|(_, p, _)| p == pred)
            .map(|(s, _, o)| (s.as_str(), o.as_str()))
            .collect()
    }

    /// Get all objects reachable from a subject via a predicate.
    pub fn objects_from(&self, subject: &str, predicate: &str) -> Vec<&str> {
        self.triples
            .iter()
            .filter(|(s, p, _)| s == subject && p == predicate)
            .map(|(_, _, o)| o.as_str())
            .collect()
    }

    /// Get all subjects that reach an object via a predicate (for inverse).
    pub fn subjects_to(&self, object: &str, predicate: &str) -> Vec<&str> {
        self.triples
            .iter()
            .filter(|(_, p, o)| o == object && p == predicate)
            .map(|(s, _, _)| s.as_str())
            .collect()
    }

    /// Number of triples in the store.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// All unique subjects in the store.
    pub fn all_subjects(&self) -> HashSet<&str> {
        self.triples.iter().map(|(s, _, _)| s.as_str()).collect()
    }

    /// All unique nodes (subjects and objects) in the store.
    pub fn all_nodes(&self) -> HashSet<&str> {
        self.triples
            .iter()
            .flat_map(|(s, _, o)| [s.as_str(), o.as_str()])
            .collect()
    }
}

/// Expands property path expressions against a triple store.
pub struct PropertyPathExpander {
    max_depth: usize,
}

impl PropertyPathExpander {
    /// Create a new expander with the given maximum recursion depth.
    pub fn new(max_depth: usize) -> Self {
        PropertyPathExpander {
            max_depth: max_depth.max(1),
        }
    }

    /// Expand a property path from a starting node, returning all reachable objects.
    pub fn expand(
        &self,
        start: &str,
        path: &PropertyPath,
        store: &TripleStore,
    ) -> Result<Vec<String>, PathError> {
        self.expand_inner(start, path, store, 0)
    }

    fn expand_inner(
        &self,
        start: &str,
        path: &PropertyPath,
        store: &TripleStore,
        depth: usize,
    ) -> Result<Vec<String>, PathError> {
        if depth > self.max_depth {
            return Err(PathError::MaxDepthExceeded(self.max_depth));
        }
        match path {
            PropertyPath::Iri(pred) => Ok(store
                .objects_from(start, pred)
                .into_iter()
                .map(String::from)
                .collect()),
            PropertyPath::Inverse(inner) => self.expand_inverse(start, inner, store, depth),
            PropertyPath::Sequence(a, b) => {
                let intermediate = self.expand_inner(start, a, store, depth + 1)?;
                let mut results = Vec::new();
                for mid in &intermediate {
                    let next = self.expand_inner(mid, b, store, depth + 1)?;
                    results.extend(next);
                }
                Ok(results)
            }
            PropertyPath::Alternative(a, b) => {
                let mut results = self.expand_inner(start, a, store, depth + 1)?;
                results.extend(self.expand_inner(start, b, store, depth + 1)?);
                // Deduplicate while preserving deterministic order
                let mut seen = HashSet::new();
                results.retain(|item| seen.insert(item.clone()));
                Ok(results)
            }
            PropertyPath::ZeroOrMore(inner) => {
                let mut visited = HashSet::new();
                visited.insert(start.to_string()); // zero hops: include start
                self.closure(start, inner, store, &mut visited, depth)?;
                let mut result: Vec<String> = visited.into_iter().collect();
                result.sort();
                Ok(result)
            }
            PropertyPath::OneOrMore(inner) => {
                let mut visited = HashSet::new();
                // One or more: start NOT included unless reachable via cycle
                let first = self.expand_inner(start, inner, store, depth + 1)?;
                for node in &first {
                    if visited.insert(node.clone()) {
                        self.closure(node, inner, store, &mut visited, depth)?;
                    }
                }
                let mut result: Vec<String> = visited.into_iter().collect();
                result.sort();
                Ok(result)
            }
            PropertyPath::ZeroOrOne(inner) => {
                let mut results = HashSet::new();
                results.insert(start.to_string()); // zero hops
                for obj in self.expand_inner(start, inner, store, depth + 1)? {
                    results.insert(obj);
                }
                let mut result: Vec<String> = results.into_iter().collect();
                result.sort();
                Ok(result)
            }
        }
    }

    fn expand_inverse(
        &self,
        start: &str,
        inner: &PropertyPath,
        store: &TripleStore,
        depth: usize,
    ) -> Result<Vec<String>, PathError> {
        match inner {
            PropertyPath::Iri(pred) => Ok(store
                .subjects_to(start, pred)
                .into_iter()
                .map(String::from)
                .collect()),
            _ => {
                // For complex inverse paths, check all nodes as potential sources
                let mut results = Vec::new();
                for subj in store.all_nodes() {
                    let reachable = self.expand_inner(subj, inner, store, depth + 1)?;
                    if reachable.iter().any(|r| r == start) {
                        results.push(subj.to_string());
                    }
                }
                results.sort();
                results.dedup();
                Ok(results)
            }
        }
    }

    /// Fixed-point closure: keep expanding until no new nodes are found.
    fn closure(
        &self,
        start: &str,
        path: &PropertyPath,
        store: &TripleStore,
        visited: &mut HashSet<String>,
        depth: usize,
    ) -> Result<(), PathError> {
        if depth > self.max_depth {
            return Err(PathError::MaxDepthExceeded(self.max_depth));
        }
        let next = self.expand_inner(start, path, store, depth + 1)?;
        for node in next {
            if visited.insert(node.clone()) {
                // New node discovered, recurse
                self.closure(&node, path, store, visited, depth + 1)?;
            }
        }
        Ok(())
    }

    /// Expand a property path for ALL subjects, returning (subject, object) pairs.
    pub fn expand_all(
        &self,
        path: &PropertyPath,
        store: &TripleStore,
    ) -> Result<Vec<(String, String)>, PathError> {
        let subjects = store.all_subjects();
        let mut results = Vec::new();
        for subj in subjects {
            let objects = self.expand(subj, path, store)?;
            for obj in objects {
                results.push((subj.to_string(), obj));
            }
        }
        results.sort();
        results.dedup();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_iri_construction() {
        let path = PropertyPath::iri("knows");
        assert_eq!(path, PropertyPath::Iri("knows".to_string()));
    }

    #[test]
    fn test_path_display() {
        let seq = PropertyPath::seq(PropertyPath::iri("a"), PropertyPath::iri("b"));
        assert_eq!(format!("{}", seq), "a/b");

        let alt = PropertyPath::alt(PropertyPath::iri("a"), PropertyPath::iri("b"));
        assert_eq!(format!("{}", alt), "a|b");

        let star = PropertyPath::zero_or_more(PropertyPath::iri("a"));
        assert_eq!(format!("{}", star), "a*");

        let plus = PropertyPath::one_or_more(PropertyPath::iri("a"));
        assert_eq!(format!("{}", plus), "a+");

        let opt = PropertyPath::zero_or_one(PropertyPath::iri("a"));
        assert_eq!(format!("{}", opt), "a?");

        let inv = PropertyPath::inverse(PropertyPath::iri("a"));
        assert_eq!(format!("{}", inv), "^a");
    }

    #[test]
    fn test_path_depth_simple() {
        let path = PropertyPath::iri("knows");
        assert_eq!(path.depth(), 1);
    }

    #[test]
    fn test_path_depth_nested() {
        // seq(iri, seq(iri, iri)) => depth 3
        let inner = PropertyPath::seq(PropertyPath::iri("b"), PropertyPath::iri("c"));
        let path = PropertyPath::seq(PropertyPath::iri("a"), inner);
        assert_eq!(path.depth(), 3);
    }

    #[test]
    fn test_path_contains_closure_true() {
        let path = PropertyPath::zero_or_more(PropertyPath::iri("knows"));
        assert!(path.contains_closure());

        let path2 = PropertyPath::one_or_more(PropertyPath::iri("knows"));
        assert!(path2.contains_closure());

        // Nested inside a sequence
        let path3 = PropertyPath::seq(
            PropertyPath::iri("a"),
            PropertyPath::zero_or_more(PropertyPath::iri("b")),
        );
        assert!(path3.contains_closure());
    }

    #[test]
    fn test_path_contains_closure_false() {
        let path = PropertyPath::iri("knows");
        assert!(!path.contains_closure());

        let path2 = PropertyPath::seq(PropertyPath::iri("a"), PropertyPath::iri("b"));
        assert!(!path2.contains_closure());

        let path3 = PropertyPath::zero_or_one(PropertyPath::iri("a"));
        assert!(!path3.contains_closure());
    }

    #[test]
    fn test_path_is_simple() {
        assert!(PropertyPath::iri("knows").is_simple());
        assert!(!PropertyPath::seq(PropertyPath::iri("a"), PropertyPath::iri("b")).is_simple());
        assert!(!PropertyPath::zero_or_more(PropertyPath::iri("a")).is_simple());
    }

    #[test]
    fn test_triple_store_add_and_len() {
        let mut store = TripleStore::new();
        assert!(store.is_empty());
        store.add("A", "knows", "B");
        store.add("B", "knows", "C");
        store.add("A", "likes", "D");
        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_triple_store_predicate_filter() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("B", "knows", "C");
        store.add("A", "likes", "D");

        let knows = store.triples_with_predicate("knows");
        assert_eq!(knows.len(), 2);

        let likes = store.triples_with_predicate("likes");
        assert_eq!(likes.len(), 1);
        assert_eq!(likes[0], ("A", "D"));
    }

    #[test]
    fn test_expander_simple_iri() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("A", "knows", "C");

        let expander = PropertyPathExpander::new(10);
        let result = expander
            .expand("A", &PropertyPath::iri("knows"), &store)
            .expect("expansion should succeed");
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"B".to_string()));
        assert!(result.contains(&"C".to_string()));
    }

    #[test]
    fn test_expander_sequence() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("B", "likes", "C");

        let expander = PropertyPathExpander::new(10);
        let path = PropertyPath::seq(PropertyPath::iri("knows"), PropertyPath::iri("likes"));
        let result = expander
            .expand("A", &path, &store)
            .expect("expansion should succeed");
        assert_eq!(result, vec!["C".to_string()]);
    }

    #[test]
    fn test_expander_alternative() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("A", "likes", "C");

        let expander = PropertyPathExpander::new(10);
        let path = PropertyPath::alt(PropertyPath::iri("knows"), PropertyPath::iri("likes"));
        let mut result = expander
            .expand("A", &path, &store)
            .expect("expansion should succeed");
        result.sort();
        assert_eq!(result, vec!["B".to_string(), "C".to_string()]);
    }

    #[test]
    fn test_expander_one_or_more() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("B", "knows", "C");
        store.add("C", "knows", "D");

        let expander = PropertyPathExpander::new(20);
        let path = PropertyPath::one_or_more(PropertyPath::iri("knows"));
        let mut result = expander
            .expand("A", &path, &store)
            .expect("expansion should succeed");
        result.sort();
        assert_eq!(
            result,
            vec!["B".to_string(), "C".to_string(), "D".to_string()]
        );
        // Start node A should NOT be included (no cycle back)
        assert!(!result.contains(&"A".to_string()));
    }

    #[test]
    fn test_expander_zero_or_more() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("B", "knows", "C");

        let expander = PropertyPathExpander::new(20);
        let path = PropertyPath::zero_or_more(PropertyPath::iri("knows"));
        let mut result = expander
            .expand("A", &path, &store)
            .expect("expansion should succeed");
        result.sort();
        // Zero-or-more includes start node
        assert_eq!(
            result,
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
    }

    #[test]
    fn test_expander_inverse() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("C", "knows", "B");

        let expander = PropertyPathExpander::new(10);
        let path = PropertyPath::inverse(PropertyPath::iri("knows"));
        let mut result = expander
            .expand("B", &path, &store)
            .expect("expansion should succeed");
        result.sort();
        assert_eq!(result, vec!["A".to_string(), "C".to_string()]);
    }

    #[test]
    fn test_expander_cycle_safe() {
        let mut store = TripleStore::new();
        store.add("A", "knows", "B");
        store.add("B", "knows", "A");

        let expander = PropertyPathExpander::new(50);
        let path = PropertyPath::one_or_more(PropertyPath::iri("knows"));
        let mut result = expander
            .expand("A", &path, &store)
            .expect("expansion should not infinite loop");
        result.sort();
        // A->B->A cycle: from A, one_or_more should find B, then from B find A
        assert_eq!(result, vec!["A".to_string(), "B".to_string()]);
    }
}
