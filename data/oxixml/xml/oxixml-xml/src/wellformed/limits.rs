//! Resource limits applied while general entities are expanded.
//!
//! The [No Recursion] constraint only forbids an entity that refers to *itself*.
//! It says nothing about a document in which every entity has a distinct name
//! but each one references the previous one several times — the classic
//! "billion laughs" bomb, whose expanded size grows exponentially while its
//! nesting depth stays small. A conforming non-validating processor is free to
//! stop expanding such a document instead of running out of time or memory
//! (section 4.4.3 leaves the treatment of unexpanded entities to the processor),
//! so the well-formedness checker charges every expansion against a budget that
//! is shared by the whole document.
//!
//! [No Recursion]: https://www.w3.org/TR/xml/#norecursion

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::errors::WellFormedError;

/// Upper bound on how deeply general entities may nest before the document is
/// rejected.
pub(crate) const MAX_ENTITY_DEPTH: usize = 40;

/// Upper bound on the total number of general entity expansions performed while
/// checking one document.
pub(crate) const MAX_ENTITY_EXPANSIONS: usize = 10_000;

/// Upper bound on the total size, in bytes, of all replacement text expanded
/// while checking one document.
pub(crate) const MAX_EXPANDED_BYTES: usize = 10 * 1024 * 1024;

/// A budget for general entity expansion, shared by every checker that works on
/// the same document (the document entity and all nested entity checkers).
#[derive(Debug, Default)]
pub(crate) struct Budget {
    /// Number of expansions performed so far.
    expansions: AtomicUsize,
    /// Total size of the replacement text expanded so far, in bytes.
    bytes: AtomicUsize,
}

impl Budget {
    /// Creates a fresh, shared budget.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Charges one expansion of `entity`, whose replacement text is `len` bytes
    /// long, against the budget.
    ///
    /// # Errors
    ///
    /// Returns [`WellFormedError::EntityExpansionLimit`] once either limit is
    /// exhausted. The budget stays exhausted afterwards, so the error is
    /// reported again for every further expansion attempt.
    pub(crate) fn charge(&self, entity: &str, len: usize) -> Result<(), WellFormedError> {
        let expansions = self.expansions.fetch_add(1, Ordering::Relaxed) + 1;
        if expansions > MAX_ENTITY_EXPANSIONS {
            return Err(WellFormedError::EntityExpansionLimit {
                entity: entity.to_owned(),
                limit: "limit on the number of entity expansions",
            });
        }
        let bytes = self.bytes.fetch_add(len, Ordering::Relaxed) + len;
        if bytes > MAX_EXPANDED_BYTES {
            return Err(WellFormedError::EntityExpansionLimit {
                entity: entity.to_owned(),
                limit: "limit on the total size of expanded entities",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_expansions() {
        let budget = Budget::new();
        for _ in 0..MAX_ENTITY_EXPANSIONS {
            assert!(budget.charge("e", 1).is_ok());
        }
        assert!(matches!(
            budget.charge("e", 1),
            Err(WellFormedError::EntityExpansionLimit { .. })
        ));
    }

    #[test]
    fn counts_bytes() {
        let budget = Budget::new();
        assert!(budget.charge("e", MAX_EXPANDED_BYTES).is_ok());
        assert!(matches!(
            budget.charge("e", 1),
            Err(WellFormedError::EntityExpansionLimit { .. })
        ));
    }

    #[test]
    fn is_shared_between_clones() {
        let budget = Budget::new();
        let clone = Arc::clone(&budget);
        assert!(clone.charge("e", MAX_EXPANDED_BYTES).is_ok());
        assert!(matches!(
            budget.charge("e", 1),
            Err(WellFormedError::EntityExpansionLimit { .. })
        ));
    }
}
