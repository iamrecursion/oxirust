//! Blank-node management: mint stable IRIs for anonymous RDF nodes.
//!
//! In RDF, blank nodes are local identifiers (prefixed with `_:`) that have
//! no global identity. This module provides [`BlankNodeManager`] which
//! deterministically replaces blank-node identifiers with minted IRIs so that
//! downstream processing can treat every node uniformly.
//!
//! # Examples
//!
//! ```
//! use tensorlogic_oxirs_bridge::BlankNodeManager;
//!
//! let mut mgr = BlankNodeManager::new("http://example.org/blank/");
//! let iri = mgr.mint("_:b0");
//! assert!(iri.starts_with("http://example.org/blank/"));
//! assert_eq!(mgr.resolve("_:b0"), iri); // idempotent for same blank id
//! assert_eq!(mgr.resolve("http://named/"), "http://named/"); // named → unchanged
//! ```

use std::collections::HashMap;

/// Converts blank-node identifiers (`_:…`) into stable, unique IRIs.
pub struct BlankNodeManager {
    counter: u64,
    base_iri: String,
    /// Maps a blank-node id such as `"_:b0"` to the minted IRI.
    pub mapping: HashMap<String, String>,
}

impl BlankNodeManager {
    /// Create a new manager.  Every minted IRI will start with `base_iri`.
    pub fn new(base_iri: impl Into<String>) -> Self {
        Self {
            counter: 0,
            base_iri: base_iri.into(),
            mapping: HashMap::new(),
        }
    }

    /// Mint a fresh IRI for `blank_id` and remember the mapping.
    ///
    /// If the same `blank_id` has already been minted, the existing IRI is
    /// returned without incrementing the counter.
    pub fn mint(&mut self, blank_id: &str) -> String {
        if let Some(existing) = self.mapping.get(blank_id) {
            return existing.clone();
        }
        let iri = format!("{}_blank_{}", self.base_iri, self.counter);
        self.counter += 1;
        self.mapping.insert(blank_id.to_owned(), iri.clone());
        iri
    }

    /// If `s` is a blank-node identifier return (or create) its minted IRI;
    /// otherwise return `s` unchanged.
    pub fn resolve(&mut self, s: &str) -> String {
        if Self::is_blank(s) {
            self.mint(s)
        } else {
            s.to_owned()
        }
    }

    /// Return `true` when `s` starts with the conventional `_:` prefix.
    pub fn is_blank(s: &str) -> bool {
        s.starts_with("_:")
    }

    /// Total number of unique blank nodes minted so far.
    pub fn mapping_count(&self) -> usize {
        self.mapping.len()
    }

    /// Look up a previously minted IRI without mutating state.
    pub fn get_minted(&self, blank_id: &str) -> Option<&str> {
        self.mapping.get(blank_id).map(|s| s.as_str())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_blank_node_detection() {
        assert!(BlankNodeManager::is_blank("_:b0"));
        assert!(BlankNodeManager::is_blank("_:xyz"));
        assert!(!BlankNodeManager::is_blank("http://example.org/"));
        assert!(!BlankNodeManager::is_blank(""));
        assert!(!BlankNodeManager::is_blank("notblank"));
    }

    #[test]
    fn test_mint_returns_unique_iris() {
        let mut mgr = BlankNodeManager::new("http://base/");
        let a = mgr.mint("_:a");
        let b = mgr.mint("_:b");
        assert_ne!(a, b, "different blank ids must yield different IRIs");
    }

    #[test]
    fn test_mint_increments_counter() {
        let mut mgr = BlankNodeManager::new("http://base/");
        mgr.mint("_:x");
        mgr.mint("_:y");
        // Counter must have advanced to 2 after two distinct mints
        assert_eq!(mgr.mapping_count(), 2);
    }

    #[test]
    fn test_resolve_blank_returns_minted() {
        let mut mgr = BlankNodeManager::new("http://base/");
        let first = mgr.resolve("_:node1");
        let second = mgr.resolve("_:node1");
        assert_eq!(first, second, "resolve must be idempotent");
        assert!(first.starts_with("http://base/"));
    }

    #[test]
    fn test_resolve_named_unchanged() {
        let mut mgr = BlankNodeManager::new("http://base/");
        let named = "http://example.org/Alice";
        assert_eq!(mgr.resolve(named), named);
        assert_eq!(mgr.mapping_count(), 0, "no blank node should be recorded");
    }

    #[test]
    fn test_mapping_stored() {
        let mut mgr = BlankNodeManager::new("http://base/");
        let iri = mgr.mint("_:b42");
        assert!(mgr.mapping.contains_key("_:b42"));
        assert_eq!(mgr.mapping["_:b42"], iri);
    }

    #[test]
    fn test_get_minted_returns_correct_iri() {
        let mut mgr = BlankNodeManager::new("http://base/");
        let iri = mgr.mint("_:foo");
        assert_eq!(mgr.get_minted("_:foo"), Some(iri.as_str()));
        assert_eq!(mgr.get_minted("_:unknown"), None);
    }
}
