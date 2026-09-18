//! The [`CanonicalizationResult`] returned by a successful canonicalization.

use oxixml_model::Dataset;
use std::collections::BTreeMap;

/// The outcome of canonicalizing an RDF dataset with [RDFC-1.0](https://www.w3.org/TR/rdf-canon/).
///
/// It bundles the three artifacts a caller typically needs:
///
/// * the [canonical N-Quads](Self::canonical_n_quads) serialization — the sorted, blank-node
///   relabeled document whose bytes are identical for every
///   [isomorphic](https://www.w3.org/TR/rdf11-concepts/#dfn-dataset-isomorphism) input;
/// * the [issued identifier map](Self::issued_map) — each original blank-node identifier mapped to
///   its canonical `c14n`-prefixed label; and
/// * the [canonical dataset](Self::canonical_dataset) — the same graph with its blank nodes
///   relabeled to the canonical identifiers, ready for further in-memory processing.
#[derive(Debug, Clone)]
pub struct CanonicalizationResult {
    canonical_n_quads: String,
    issued_map: BTreeMap<String, String>,
    canonical_dataset: Dataset,
}

impl CanonicalizationResult {
    pub(crate) fn new(
        canonical_n_quads: String,
        issued_map: BTreeMap<String, String>,
        canonical_dataset: Dataset,
    ) -> Self {
        Self {
            canonical_n_quads,
            issued_map,
            canonical_dataset,
        }
    }

    /// The canonical [N-Quads](https://www.w3.org/TR/n-quads/) serialization.
    ///
    /// Lines are sorted and each is terminated by `" .\n"`, matching the RDFC-1.0
    /// [serialization step](https://www.w3.org/TR/rdf-canon/#canon-algo-algo).
    #[must_use]
    pub fn canonical_n_quads(&self) -> &str {
        &self.canonical_n_quads
    }

    /// The map from each original blank-node identifier to its canonical identifier
    /// (for example `"e0" -> "c14n0"`), ordered by original identifier.
    #[must_use]
    pub fn issued_map(&self) -> &BTreeMap<String, String> {
        &self.issued_map
    }

    /// The input dataset with every blank node relabeled to its canonical identifier.
    #[must_use]
    pub fn canonical_dataset(&self) -> &Dataset {
        &self.canonical_dataset
    }

    /// Consumes the result, returning the canonical N-Quads string.
    #[must_use]
    pub fn into_canonical_n_quads(self) -> String {
        self.canonical_n_quads
    }

    /// Consumes the result, returning its three parts:
    /// `(canonical N-Quads, issued map, canonical dataset)`.
    #[must_use]
    pub fn into_parts(self) -> (String, BTreeMap<String, String>, Dataset) {
        (
            self.canonical_n_quads,
            self.issued_map,
            self.canonical_dataset,
        )
    }
}
