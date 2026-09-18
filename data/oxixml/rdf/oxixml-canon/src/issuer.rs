//! The [blank node identifier issuer](https://www.w3.org/TR/rdf-canon/#blank-node-identifier-issuer-state)
//! state and the [Issue Identifier algorithm](https://www.w3.org/TR/rdf-canon/#issue-identifier).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// An [RDFC-1.0 identifier issuer](https://www.w3.org/TR/rdf-canon/#dfn-identifier-issuer).
///
/// It bundles an *identifier prefix*, an *identifier counter* and an *issued identifiers map*.
/// Two issuers are used by the algorithm: the long-lived *canonical issuer* (prefix `c14n`) and
/// the short-lived *temporary issuers* (prefix `b`) created while exploring gossip paths.
///
/// The insertion order of the issued identifiers map is preserved in [`Self::issued_order`] so
/// that step 5.3.1 of the canonicalization algorithm can replay it "in the same order".
#[derive(Clone, Debug)]
pub(crate) struct IdentifierIssuer {
    identifier_prefix: &'static str,
    identifier_counter: u64,
    /// Maps an original blank-node identifier to the identifier issued for it.
    issued: HashMap<String, String>,
    /// The original blank-node identifiers, in order of issuance.
    order: Vec<String>,
}

impl IdentifierIssuer {
    /// Creates an issuer with the given identifier prefix and a zeroed identifier counter.
    pub(crate) fn new(identifier_prefix: &'static str) -> Self {
        Self {
            identifier_prefix,
            identifier_counter: 0,
            issued: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// The [Issue Identifier algorithm](https://www.w3.org/TR/rdf-canon/#issue-identifier).
    ///
    /// Returns the identifier already issued for `blank_node`, or issues (and records) a fresh
    /// one built from the prefix and the current counter.
    pub(crate) fn issue(&mut self, blank_node: &str) -> &str {
        match self.issued.entry(blank_node.to_owned()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let issued_identifier =
                    format!("{}{}", self.identifier_prefix, self.identifier_counter);
                self.identifier_counter += 1;
                self.order.push(blank_node.to_owned());
                entry.insert(issued_identifier)
            }
        }
    }

    /// Returns the identifier issued for `blank_node`, if any.
    pub(crate) fn issued(&self, blank_node: &str) -> Option<&str> {
        self.issued.get(blank_node).map(String::as_str)
    }

    /// Returns `true` if an identifier has already been issued for `blank_node`.
    pub(crate) fn has_issued(&self, blank_node: &str) -> bool {
        self.issued.contains_key(blank_node)
    }

    /// The original blank-node identifiers this issuer has issued identifiers for, in order of
    /// issuance.
    pub(crate) fn issued_order(&self) -> &[String] {
        &self.order
    }

    /// The full issued identifiers map, as `(original, issued)` pairs.
    pub(crate) fn issued_entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.issued
            .iter()
            .map(|(original, issued)| (original.as_str(), issued.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::IdentifierIssuer;

    #[test]
    fn issues_sequential_identifiers() {
        let mut issuer = IdentifierIssuer::new("c14n");
        assert_eq!(issuer.issue("e0"), "c14n0");
        assert_eq!(issuer.issue("e1"), "c14n1");
        assert_eq!(issuer.issue("e0"), "c14n0");
        assert_eq!(issuer.issued("e1"), Some("c14n1"));
        assert_eq!(issuer.issued("nope"), None);
        assert!(issuer.has_issued("e0"));
        assert_eq!(issuer.issued_order(), ["e0".to_owned(), "e1".to_owned()]);
    }
}
