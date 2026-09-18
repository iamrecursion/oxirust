//! `oxixml-canon` — [RDF Dataset Canonicalization (RDFC-1.0)](https://www.w3.org/TR/rdf-canon/).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing.
//!
//! RDFC-1.0 (also known as URDNA2015) deterministically relabels the blank nodes of an RDF dataset
//! so that any two [isomorphic](https://www.w3.org/TR/rdf11-concepts/#dfn-dataset-isomorphism)
//! datasets produce byte-for-byte identical [canonical N-Quads](https://www.w3.org/TR/n-quads/).
//! This is the basis for dataset diffing, digital signatures, and content-addressed storage.
//!
//! # Overview
//!
//! * [`canonicalize`] takes a borrowed [`Dataset`] and returns a [`CanonicalizationResult`]
//!   exposing the [canonical N-Quads](CanonicalizationResult::canonical_n_quads), the
//!   [issued identifier map](CanonicalizationResult::issued_map), and a
//!   [relabeled dataset](CanonicalizationResult::canonical_dataset).
//! * [`canonicalize_quads`] is the streaming entry point: it accepts any
//!   `IntoIterator<Item = Quad>` (deduplicating as it goes).
//! * The `*_with` variants ([`canonicalize_with`], [`canonicalize_quads_with`]) accept
//!   [`CanonicalizationOptions`] to select the [`HashAlgorithm`] (SHA-256 by default, SHA-384 per
//!   the spec's hash-algorithm parameter) and the [`WorkBudget`].
//!
//! The hash functions come from [`oxixml_support::hash`]; this crate pulls in no external
//! dependencies.
//!
//! # Poison graphs
//!
//! The [hash-n-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) step can, on
//! adversarial input, explore a factorial number of "gossip paths". As the spec's
//! [security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations) recommend,
//! every run is bounded by a configurable [`WorkBudget`]; exhausting it yields
//! [`CanonicalizationError::WorkBudgetExceeded`] instead of an unbounded computation.
//!
//! # Example
//!
//! ```
//! use oxixml_canon::canonicalize;
//! use oxixml_model::{BlankNode, Dataset, GraphNameRef, NamedNode, QuadRef};
//!
//! let p = NamedNode::new("http://example.com/p")?;
//! let q = NamedNode::new("http://example.com/q")?;
//!
//! // Two blank nodes, distinguishable only by structure.
//! let mut dataset = Dataset::new();
//! let a = BlankNode::default();
//! let b = BlankNode::default();
//! dataset.insert(QuadRef::new(&p, &q, &a, GraphNameRef::DefaultGraph));
//! dataset.insert(QuadRef::new(&a, &p, &b, GraphNameRef::DefaultGraph));
//!
//! let result = canonicalize(&dataset)?;
//!
//! // The blank nodes are relabeled to stable `c14n` identifiers.
//! assert_eq!(result.issued_map().len(), 2);
//! assert!(result.canonical_n_quads().contains("_:c14n0"));
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

mod engine;
mod error;
mod issuer;
mod options;
mod permutations;
mod result;
mod terms;

pub use crate::error::{CanonicalizationError, Result};
pub use crate::options::{CanonicalizationOptions, DEFAULT_WORK_BUDGET, HashAlgorithm, WorkBudget};
pub use crate::result::CanonicalizationResult;

use oxixml_model::{Dataset, Quad, QuadRef};
use std::collections::HashSet;

/// Canonicalizes `dataset` with the default [`CanonicalizationOptions`] (SHA-256 and the default
/// [`WorkBudget`]).
///
/// See the [crate-level documentation](crate) for an example.
///
/// # Errors
///
/// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the dataset is adversarial enough to
/// exhaust the work budget.
pub fn canonicalize(dataset: &Dataset) -> Result<CanonicalizationResult> {
    canonicalize_with(dataset, CanonicalizationOptions::default())
}

/// Canonicalizes `dataset` with explicit [`CanonicalizationOptions`].
///
/// # Errors
///
/// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the configured [`WorkBudget`] is
/// exhausted before canonicalization completes.
pub fn canonicalize_with(
    dataset: &Dataset,
    options: CanonicalizationOptions,
) -> Result<CanonicalizationResult> {
    engine::canonicalize(dataset.iter().map(QuadRef::into_owned).collect(), options)
}

/// Canonicalizes a stream of [`Quad`]s with the default [`CanonicalizationOptions`].
///
/// Duplicate quads are removed, so the input need not be a set.
///
/// # Errors
///
/// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the input is adversarial enough to
/// exhaust the work budget.
///
/// # Example
///
/// ```
/// use oxixml_canon::canonicalize_quads;
/// use oxixml_model::{BlankNode, GraphName, NamedNode, Quad};
///
/// let p = NamedNode::new("http://example.com/p")?;
/// let a = BlankNode::new("e0")?;
/// // The same quad twice: duplicates are removed, so the input need not be a set.
/// let quad = Quad::new(a.clone(), p, a, GraphName::DefaultGraph);
/// let result = canonicalize_quads([quad.clone(), quad])?;
///
/// assert_eq!(result.issued_map()["e0"], "c14n0");
/// assert_eq!(
///     result.canonical_n_quads(),
///     "_:c14n0 <http://example.com/p> _:c14n0 .\n"
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
pub fn canonicalize_quads(quads: impl IntoIterator<Item = Quad>) -> Result<CanonicalizationResult> {
    canonicalize_quads_with(quads, CanonicalizationOptions::default())
}

/// Canonicalizes a stream of [`Quad`]s with explicit [`CanonicalizationOptions`].
///
/// Duplicate quads are removed, so the input need not be a set.
///
/// # Errors
///
/// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the configured [`WorkBudget`] is
/// exhausted before canonicalization completes.
pub fn canonicalize_quads_with(
    quads: impl IntoIterator<Item = Quad>,
    options: CanonicalizationOptions,
) -> Result<CanonicalizationResult> {
    let mut seen = HashSet::new();
    let deduplicated = quads
        .into_iter()
        .filter(|quad| seen.insert(quad.clone()))
        .collect();
    engine::canonicalize(deduplicated, options)
}
