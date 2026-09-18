//! [RDF Dataset Canonicalization](https://www.w3.org/TR/rdf-canon/) for [`Dataset`].
//!
//! This module implements the blank-node renaming that gives isomorphic datasets identical
//! serializations. Two algorithm families are provided:
//!
//! * the default `Unstable`/`UnstableHashedIds` variants, which use the standard library's
//!   [`DefaultHasher`] and require no extra dependencies, and
//! * the standard [RDFC-1.0](https://www.w3.org/TR/rdf-canon/#dfn-rdfc-1-0) algorithm behind the
//!   `rdfc-10` feature, using SHA-256/SHA-384 provided by `oxixml-support`.
//!
//! All three variants share the same [hash-n-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads)
//! recursion, which the [security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations)
//! warn can be driven into super-polynomial work by an adversarial ("poison") input. `canonicalize`
//! and `canonicalize_blank_nodes` run that recursion under an effectively unlimited budget, for
//! parity with `oxrdf`'s infallible signatures; `try_canonicalize` and `try_canonicalize_blank_nodes`
//! run the identical algorithm under a configurable work budget and report
//! [`CanonicalizationError::WorkBudgetExceeded`] instead of running unboundedly. Prefer the `try_`
//! forms when canonicalizing untrusted input.
#![allow(clippy::type_complexity)]

use super::Dataset;
use crate::interning::*;
use crate::*;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::take;

#[cfg(feature = "rdfc-10")]
use oxixml_support::hash::{sha256, sha384};

type QuadsPerBlankNode<'a> = HashMap<
    InternedBlankNode,
    Vec<&'a (
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
        InternedGraphName,
    )>,
>;

/// The default work budget used by [`Dataset::try_canonicalize`] and
/// [`Dataset::try_canonicalize_blank_nodes`], in abstract work units: 262 144.
///
/// One work unit is charged for every gossip-path permutation examined, and for every
/// `hash-n-degree-quads` invocation no deeper than its internal safe-recursion-depth ceiling of
/// 256 — a call deeper than that charges the entire remaining budget instead, guaranteeing
/// rejection regardless of this value's own magnitude. See that function's docs for why a charge
/// that stayed flat at every depth would not be enough to keep native recursion depth bounded, and
/// [`CanonicalizationError::WorkBudgetExceeded`] for the resulting error. The heaviest
/// non-adversarial case in the W3C `rdf-canon` conformance suite (including its `poison – evil`
/// cases, which are explicitly *computable given defined limits*) needs at most 3 348 work units
/// (`test044c`), measured by running the full suite through this implementation — unchanged from
/// before `hash-n-degree-quads` started tracking depth at all, since no legitimate case in the
/// suite recurses anywhere near that ceiling; this default leaves nearly two orders of
/// magnitude of headroom above that while still rejecting the suite's one deliberately intractable
/// case (a fully connected 10-node blank-node clique) in close to a second in an optimized
/// (`--release`) build — an unoptimized build takes tens of seconds, since hashing and
/// string-building costs dominate the wall-clock time per work unit.
pub const DEFAULT_CANONICALIZATION_WORK_BUDGET: u64 = 1 << 18;

/// The maximum `hash-n-degree-quads` recursion depth any *bounded* work budget (anything less
/// than `u64::MAX`) ever lets that recursion reach: a call at this depth or shallower charges its
/// usual one work unit, exactly as before this crate tracked depth at all; a call any deeper
/// charges the caller's entire remaining budget in one shot, so it is always rejected, regardless
/// of how large that budget is. This number matches this workspace's usual structural-nesting
/// ceiling (see the crate-external policy it is drawn from: element/DOM trees, DER structures, and
/// RDF/HTML/SVG document nesting all use the same 256) — chosen because it comfortably covers
/// every legitimate case this crate has measured (see
/// [`DEFAULT_CANONICALIZATION_WORK_BUDGET`]'s doc comment) while still being a depth any thread's
/// native call stack survives many times over. See `hash_n_degree_quads`'s doc comment for why the
/// cutoff is a hard one rather than a gradually escalating charge. This is an internal pricing
/// knob, not a user-facing limit in its own right: callers who need more headroom for legitimately
/// deep *combinatorial* (permutation) work tune [`DEFAULT_CANONICALIZATION_WORK_BUDGET`] itself via
/// [`try_canonicalize_with_budget`](Dataset::try_canonicalize_with_budget); raising it can never
/// raise this ceiling.
const SAFE_RECURSION_DEPTH: usize = 256;

/// An error raised while canonicalizing a [`Dataset`] or [`Graph`](crate::Graph) under a bounded
/// work budget.
///
/// See [`Dataset::try_canonicalize`] and the module-level docs for why a budget is needed at
/// all: [RDFC-1.0's security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations)
/// warn that an adversarial ("poison") dataset can drive the underlying
/// [hash-n-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) recursion — shared by
/// every [`CanonicalizationAlgorithm`] variant, not just RDFC-1.0 — into super-polynomial (up to
/// factorial) work.
///
/// <div class="warning">Adding a future variant here that can be constructed other than by
/// exhausting a finite work budget would need its own review: <code>Dataset::canonicalize</code>
/// assumes, and asserts via <code>expect</code>, that an unlimited (<code>u64::MAX</code>) budget
/// can never produce an error. That assumption holds today because
/// <code>WorkBudgetExceeded</code> is the only variant and budget exhaustion is its only
/// construction site; keep it true, or stop relying on <code>expect</code> in
/// <code>canonicalize</code>.</div>
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CanonicalizationError {
    /// The configured work budget was exhausted before a canonical form could be produced.
    ///
    /// This almost always indicates a deliberately adversarial input graph. Callers who trust
    /// the input may retry with a larger budget through
    /// [`try_canonicalize_with_budget`](Dataset::try_canonicalize_with_budget), or use
    /// [`canonicalize`](Dataset::canonicalize) directly, which never rejects on budget grounds.
    WorkBudgetExceeded {
        /// The budget (in abstract work units) that was exceeded.
        budget: u64,
    },
}

impl fmt::Display for CanonicalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkBudgetExceeded { budget } => write!(
                f,
                "the RDF dataset canonicalization work budget of {budget} unit(s) was exceeded; \
                 the input is likely an adversarial (poison) dataset"
            ),
        }
    }
}

impl std::error::Error for CanonicalizationError {}

/// An algorithm used to canonicalize graphs and datasets.
///
/// See [`Graph::canonicalize`](crate::Graph::canonicalize) and [`Dataset::canonicalize`].
#[derive(Default, Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CanonicalizationAlgorithm {
    /// The algorithm preferred by this crate.
    ///
    /// <div class="warning">The canonicalization algorithm is not stable and canonical blank node
    /// ids might change between versions.</div>
    #[default]
    Unstable,
    /// The algorithm preferred by this crate but outputting ids based on hashes.
    ///
    /// This enables the blank node ids to be used for diffing: additions or deletions of triples
    /// affect fewer blank node ids.
    ///
    /// <div class="warning">The canonicalization algorithm is not stable and canonical blank node
    /// ids might change between versions.</div>
    UnstableHashedIds,
    /// The [RDF Canonicalization algorithm version 1.0](https://www.w3.org/TR/rdf-canon/#dfn-rdfc-1-0)
    /// parametrized with its [`CanonicalizationHashAlgorithm`].
    ///
    /// <div class="warning">Note that the algorithm does not support RDF 1.2, this
    /// implementation's behavior on triple terms is not part of the standard and might
    /// change.</div>
    #[cfg(feature = "rdfc-10")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdfc-10")))]
    Rdfc10 {
        /// The hash algorithm used by RDFC-1.0.
        hash_algorithm: CanonicalizationHashAlgorithm,
    },
}

/// The hash function used to canonicalize graphs and datasets.
///
/// See [`Graph::canonicalize`](crate::Graph::canonicalize) and [`Dataset::canonicalize`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CanonicalizationHashAlgorithm {
    /// SHA-256.
    #[cfg(feature = "rdfc-10")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdfc-10")))]
    Sha256,
    /// SHA-384.
    #[cfg(feature = "rdfc-10")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rdfc-10")))]
    Sha384,
}

impl Dataset {
    /// Canonicalizes the dataset by renaming blank nodes.
    ///
    /// Usage example ([Dataset isomorphism](https://www.w3.org/TR/rdf11-concepts/#dfn-dataset-isomorphism)):
    /// ```
    /// use oxixml_model::dataset::CanonicalizationAlgorithm;
    /// use oxixml_model::*;
    ///
    /// let iri = NamedNodeRef::new("http://example.com")?;
    ///
    /// let mut graph1 = Graph::new();
    /// let bnode1 = BlankNode::default();
    /// graph1.insert(TripleRef::new(iri, iri, &bnode1));
    /// graph1.insert(TripleRef::new(&bnode1, iri, iri));
    ///
    /// let mut graph2 = Graph::new();
    /// let bnode2 = BlankNode::default();
    /// graph2.insert(TripleRef::new(iri, iri, &bnode2));
    /// graph2.insert(TripleRef::new(&bnode2, iri, iri));
    ///
    /// assert_ne!(graph1, graph2);
    /// graph1.canonicalize(CanonicalizationAlgorithm::Unstable);
    /// graph2.canonicalize(CanonicalizationAlgorithm::Unstable);
    /// assert_eq!(graph1, graph2);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// It supports the [RDF Dataset Canonicalization](https://www.w3.org/TR/rdf-canon/) standard
    /// algorithm, which requires the `rdfc-10` feature to be enabled.
    ///
    /// <div class="warning">Blank node ids depend on the current shape of the graph. Adding a new
    /// quad might change the ids of a lot of blank nodes. Hence, this canonicalization might not
    /// be suitable for diffs.</div>
    ///
    /// <div class="warning">This implementation's worst-case complexity is exponential with
    /// respect to the number of blank nodes in the input dataset: an adversarially crafted
    /// dataset can make this method run for a very long time. Prefer
    /// [`try_canonicalize`](Self::try_canonicalize), which runs under a bounded work budget and
    /// reports [`CanonicalizationError::WorkBudgetExceeded`] instead, when the input is not
    /// trusted.</div>
    pub fn canonicalize(&mut self, algorithm: CanonicalizationAlgorithm) {
        // An unlimited (`u64::MAX`) work budget can never be exceeded: `charge` only ever grows a
        // counter that starts at `0` through saturating addition, so it can never end up greater
        // than `u64::MAX` itself. This keeps `canonicalize` infallible — matching its historical
        // signature, which mirrors upstream `oxrdf` — while sharing its implementation, budget
        // bookkeeping included, with the bounded `try_canonicalize`.
        self.try_canonicalize_with_budget(algorithm, u64::MAX)
            .expect("an unlimited work budget can never be exceeded");
    }

    /// Canonicalizes the dataset by renaming blank nodes, like [`canonicalize`](Self::canonicalize),
    /// but rejects the input instead of running unboundedly if doing so would exceed the
    /// [`DEFAULT_CANONICALIZATION_WORK_BUDGET`].
    ///
    /// Prefer this over [`canonicalize`](Self::canonicalize) when the input dataset is not
    /// trusted: see the module-level docs for why an adversarial ("poison") dataset can otherwise
    /// make canonicalization run for a very long time. See
    /// [`try_canonicalize_with_budget`](Self::try_canonicalize_with_budget) to use a different
    /// budget.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the work budget is exhausted
    /// before a canonical form could be produced. The dataset is left unmodified in that case.
    pub fn try_canonicalize(
        &mut self,
        algorithm: CanonicalizationAlgorithm,
    ) -> Result<(), CanonicalizationError> {
        self.try_canonicalize_with_budget(algorithm, DEFAULT_CANONICALIZATION_WORK_BUDGET)
    }

    /// Like [`try_canonicalize`](Self::try_canonicalize), with an explicit work budget (in
    /// abstract work units) instead of [`DEFAULT_CANONICALIZATION_WORK_BUDGET`]. Pass `u64::MAX`
    /// for an effectively unlimited budget (as [`canonicalize`](Self::canonicalize) does).
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if `work_budget` is exhausted before
    /// a canonical form could be produced. The dataset is left unmodified in that case.
    pub fn try_canonicalize_with_budget(
        &mut self,
        algorithm: CanonicalizationAlgorithm,
        work_budget: u64,
    ) -> Result<(), CanonicalizationError> {
        let bnode_mapping = self.canonicalize_interned_blank_nodes(algorithm, work_budget)?;
        let new_quads = self.map_blank_nodes(&bnode_mapping);
        self.clear();
        for quad in new_quads {
            self.insert_encoded(quad);
        }
        Ok(())
    }

    /// Returns a map between each current dataset blank node and its canonicalized blank node.
    ///
    /// See also [`canonicalize`](Self::canonicalize).
    ///
    /// <div class="warning">This implementation's worst-case complexity is exponential with
    /// respect to the number of blank nodes in the input dataset. Prefer
    /// [`try_canonicalize_blank_nodes`](Self::try_canonicalize_blank_nodes) for untrusted
    /// input.</div>
    pub fn canonicalize_blank_nodes(
        &self,
        algorithm: CanonicalizationAlgorithm,
    ) -> HashMap<BlankNodeRef<'_>, BlankNode> {
        // See `canonicalize` for why an unlimited budget can never fail.
        self.try_canonicalize_blank_nodes_with_budget(algorithm, u64::MAX)
            .expect("an unlimited work budget can never be exceeded")
    }

    /// Like [`canonicalize_blank_nodes`](Self::canonicalize_blank_nodes), but rejects the input
    /// instead of running unboundedly if doing so would exceed the
    /// [`DEFAULT_CANONICALIZATION_WORK_BUDGET`]. See [`try_canonicalize`](Self::try_canonicalize)
    /// for when to prefer this over the unbounded form.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if the work budget is exhausted
    /// before a canonical form could be produced.
    pub fn try_canonicalize_blank_nodes(
        &self,
        algorithm: CanonicalizationAlgorithm,
    ) -> Result<HashMap<BlankNodeRef<'_>, BlankNode>, CanonicalizationError> {
        self.try_canonicalize_blank_nodes_with_budget(
            algorithm,
            DEFAULT_CANONICALIZATION_WORK_BUDGET,
        )
    }

    /// Like [`try_canonicalize_blank_nodes`](Self::try_canonicalize_blank_nodes), with an
    /// explicit work budget (in abstract work units) instead of
    /// [`DEFAULT_CANONICALIZATION_WORK_BUDGET`]. Pass `u64::MAX` for an effectively unlimited
    /// budget.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] if `work_budget` is exhausted before
    /// a canonical form could be produced.
    pub fn try_canonicalize_blank_nodes_with_budget(
        &self,
        algorithm: CanonicalizationAlgorithm,
        work_budget: u64,
    ) -> Result<HashMap<BlankNodeRef<'_>, BlankNode>, CanonicalizationError> {
        Ok(self
            .canonicalize_interned_blank_nodes(algorithm, work_budget)?
            .into_iter()
            .map(|(from, to)| (from.decode_from(&self.interner), to))
            .collect())
    }

    fn canonicalize_interned_blank_nodes(
        &self,
        algorithm: CanonicalizationAlgorithm,
        work_budget: u64,
    ) -> Result<HashMap<InternedBlankNode, BlankNode>, CanonicalizationError> {
        let hash_algorithm = match algorithm {
            CanonicalizationAlgorithm::Unstable | CanonicalizationAlgorithm::UnstableHashedIds => {
                None
            }
            #[cfg(feature = "rdfc-10")]
            CanonicalizationAlgorithm::Rdfc10 { hash_algorithm } => Some(hash_algorithm),
        };
        // https://www.w3.org/TR/rdf-canon/#canon-algo-algo
        // 1)
        let mut canonicalization_state = CanonicalizationState {
            blank_node_to_quads_map: QuadsPerBlankNode::new(),
            hash_to_blank_nodes_map: BTreeMap::new(),
            canonical_issuer: IdentifierIssuer::new("c14n"),
            work_budget,
            work: Cell::new(0),
        };
        // 2)
        for quad in &self.spog {
            if let InternedNamedOrBlankNode::BlankNode(bnode) = quad.0 {
                Self::add_quad_to_blank_node_to_quads_map_for_blank_node(
                    bnode,
                    quad,
                    &mut canonicalization_state.blank_node_to_quads_map,
                );
            }
            if let InternedTerm::BlankNode(bnode) = &quad.2 {
                Self::add_quad_to_blank_node_to_quads_map_for_blank_node(
                    *bnode,
                    quad,
                    &mut canonicalization_state.blank_node_to_quads_map,
                );
            }
            #[cfg(feature = "rdf-12")]
            if let InternedTerm::Triple(t) = &quad.2 {
                Self::add_quad_to_blank_node_to_quads_map_based_on_triple(
                    t,
                    quad,
                    &mut canonicalization_state.blank_node_to_quads_map,
                );
            }
            if let InternedGraphName::BlankNode(bnode) = &quad.3 {
                Self::add_quad_to_blank_node_to_quads_map_for_blank_node(
                    *bnode,
                    quad,
                    &mut canonicalization_state.blank_node_to_quads_map,
                );
            }
        }
        // 3)
        for n in canonicalization_state.blank_node_to_quads_map.keys() {
            // 3.1)
            let hash = self.hash_first_degree_quads(&canonicalization_state, *n, hash_algorithm);
            // 3.2)
            canonicalization_state
                .hash_to_blank_nodes_map
                .entry(hash)
                .or_default()
                .push(*n);
        }
        // 4)
        canonicalization_state.hash_to_blank_nodes_map = canonicalization_state
            .hash_to_blank_nodes_map
            .into_iter()
            .filter(|(hash, identifier_list)| {
                match identifier_list.len() {
                    0 => unreachable!(),
                    // 4.1)
                    2.. => true,
                    1 => {
                        // 4.2)
                        Self::issue_identifier(
                            &mut canonicalization_state.canonical_issuer,
                            identifier_list[0],
                            algorithm,
                            hash,
                        );
                        // 4.3)
                        false
                    }
                }
            })
            .collect::<BTreeMap<_, _>>();
        // 5)
        for (hash, identifier_list) in take(&mut canonicalization_state.hash_to_blank_nodes_map) {
            // 5.1)
            let mut hash_path_list = Vec::new();
            // 5.2)
            for n in identifier_list {
                // 5.2.1)
                if canonicalization_state
                    .canonical_issuer
                    .issued_identifier_map
                    .contains_key(&n)
                {
                    continue;
                }
                // 5.2.2)
                let mut temporary_issuer = IdentifierIssuer::new("b");
                // 5.2.3)
                Self::issue_identifier(&mut temporary_issuer, n, algorithm, &hash);
                // 5.2.4)
                hash_path_list.push(self.hash_n_degree_quads(
                    &canonicalization_state,
                    n,
                    &temporary_issuer,
                    algorithm,
                    hash_algorithm,
                    1,
                )?)
            }
            // 5.3)
            hash_path_list.sort_unstable_by(|(_, hl), (_, hr)| hl.cmp(hr));
            for (result_identifier_issuer, _) in hash_path_list {
                // 5.3.1)
                for existing_identifier in result_identifier_issuer.issued_identifier_order {
                    Self::issue_identifier(
                        &mut canonicalization_state.canonical_issuer,
                        existing_identifier,
                        algorithm,
                        &hash,
                    );
                }
            }
        }
        // 6)
        Ok(canonicalization_state
            .canonical_issuer
            .issued_identifier_map)
    }

    #[cfg(feature = "rdf-12")]
    fn add_quad_to_blank_node_to_quads_map_based_on_triple<'a>(
        triple: &InternedTriple,
        quad: &'a (
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
        blank_node_to_quads_map: &mut QuadsPerBlankNode<'a>,
    ) {
        if let InternedNamedOrBlankNode::BlankNode(bnode) = triple.subject {
            Self::add_quad_to_blank_node_to_quads_map_for_blank_node(
                bnode,
                quad,
                blank_node_to_quads_map,
            );
        }
        if let InternedTerm::BlankNode(bnode) = &triple.object {
            Self::add_quad_to_blank_node_to_quads_map_for_blank_node(
                *bnode,
                quad,
                blank_node_to_quads_map,
            );
        } else if let InternedTerm::Triple(t) = &triple.object {
            Self::add_quad_to_blank_node_to_quads_map_based_on_triple(
                t,
                quad,
                blank_node_to_quads_map,
            );
        }
    }

    fn add_quad_to_blank_node_to_quads_map_for_blank_node<'a>(
        bnode: InternedBlankNode,
        quad: &'a (
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
        blank_node_to_quads_map: &mut QuadsPerBlankNode<'a>,
    ) {
        let entry = blank_node_to_quads_map.entry(bnode).or_default();
        if !entry.ends_with(&[quad]) {
            entry.push(quad);
        }
    }

    /// RDFC [Issue Identifier Algorithm](https://www.w3.org/TR/rdf-canon/#issue-identifier).
    fn issue_identifier(
        issuer: &mut IdentifierIssuer,
        blank_node: InternedBlankNode,
        algorithm: CanonicalizationAlgorithm,
        hash: &str,
    ) -> BlankNode {
        match issuer.issued_identifier_map.entry(blank_node) {
            // 1)
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                if algorithm == CanonicalizationAlgorithm::UnstableHashedIds {
                    let mut issued_identifier = BlankNode::new_unchecked(OxString::new_owned(hash));
                    let mut i = 0;
                    while !issuer
                        .already_issued_identifiers
                        .insert(issued_identifier.clone())
                    {
                        i += 1;
                        issued_identifier =
                            BlankNode::new_unchecked(OxString::new_owned(&format!("{hash}{i}")));
                    }
                    entry.insert(issued_identifier.clone());
                    issuer.issued_identifier_order.push(blank_node);
                    issued_identifier
                } else {
                    // 2)
                    let issued_identifier = BlankNode::new_unchecked(OxString::new_owned(
                        &format!("{}{}", issuer.identifier_prefix, issuer.identifier_counter),
                    ));
                    // 3)
                    entry.insert(issued_identifier.clone());
                    issuer.issued_identifier_order.push(blank_node);
                    // 4)
                    issuer.identifier_counter += 1;
                    // 5)
                    issued_identifier
                }
            }
        }
    }

    /// RDFC [Hash First Degree Quads](https://www.w3.org/TR/rdf-canon/#hash-1d-quads).
    fn hash_first_degree_quads(
        &self,
        canonicalization_state: &CanonicalizationState<'_>,
        reference_blank_node_identifier: InternedBlankNode,
        hash_algorithm: Option<CanonicalizationHashAlgorithm>,
    ) -> String {
        // 1)
        let mut nquads = Vec::new();
        // 2)
        let quads =
            &canonicalization_state.blank_node_to_quads_map[&reference_blank_node_identifier];
        // 3)
        for (subject, predicate, object, graph_name) in quads {
            // 3.1)
            let subject = self.hash_first_degree_quads_decode_named_or_blank_node(
                subject,
                &reference_blank_node_identifier,
            );
            let predicate = predicate.decode_from(&self.interner);
            let object =
                self.hash_first_degree_quads_decode_term(object, &reference_blank_node_identifier);
            let graph_name = self.hash_first_degree_quads_decode_graph_name(
                graph_name,
                &reference_blank_node_identifier,
            );
            nquads.push(if graph_name.is_default_graph() {
                format!("{subject} {predicate} {object} .\n")
            } else {
                format!("{subject} {predicate} {object} {graph_name} .\n")
            });
        }
        // 3)
        nquads.sort();
        // 4)
        Self::hash_function(&nquads.join(""), hash_algorithm)
    }

    fn hash_first_degree_quads_decode_named_or_blank_node(
        &self,
        term: &InternedNamedOrBlankNode,
        reference_blank_node_identifier: &InternedBlankNode,
    ) -> NamedOrBlankNode {
        match term {
            InternedNamedOrBlankNode::NamedNode(t) => t.decode_from(&self.interner).into(),
            InternedNamedOrBlankNode::BlankNode(t) => {
                BlankNodeRef::new_unchecked(if t == reference_blank_node_identifier {
                    "a"
                } else {
                    "z"
                })
                .into()
            }
        }
    }

    fn hash_first_degree_quads_decode_term(
        &self,
        term: &InternedTerm,
        reference_blank_node_identifier: &InternedBlankNode,
    ) -> Term {
        match term {
            InternedTerm::NamedNode(t) => t.decode_from(&self.interner).into(),
            InternedTerm::BlankNode(t) => {
                BlankNodeRef::new_unchecked(if t == reference_blank_node_identifier {
                    "a"
                } else {
                    "z"
                })
                .into()
            }
            InternedTerm::Literal(t) => t.decode_from(&self.interner).into(),
            #[cfg(feature = "rdf-12")]
            InternedTerm::Triple(t) => Triple::new(
                self.hash_first_degree_quads_decode_named_or_blank_node(
                    &t.subject,
                    reference_blank_node_identifier,
                ),
                t.predicate.decode_from(&self.interner),
                self.hash_first_degree_quads_decode_term(
                    &t.object,
                    reference_blank_node_identifier,
                ),
            )
            .into(),
        }
    }

    fn hash_first_degree_quads_decode_graph_name(
        &self,
        term: &InternedGraphName,
        reference_blank_node_identifier: &InternedBlankNode,
    ) -> GraphName {
        match term {
            InternedGraphName::NamedNode(t) => t.decode_from(&self.interner).into(),
            InternedGraphName::BlankNode(t) => {
                BlankNodeRef::new_unchecked(if t == reference_blank_node_identifier {
                    "a"
                } else {
                    "z"
                })
                .into()
            }
            InternedGraphName::DefaultGraph => GraphName::DefaultGraph,
        }
    }

    /// RDFC [Hash Related Blank Node](https://www.w3.org/TR/rdf-canon/#hash-related-blank-node).
    fn hash_related_blank_node(
        &self,
        canonicalization_state: &CanonicalizationState<'_>,
        related: InternedBlankNode,
        quad: &(
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
        issuer: &IdentifierIssuer,
        position: &str,
        hash_algorithm: Option<CanonicalizationHashAlgorithm>,
    ) -> String {
        // 1)
        let mut input = position.to_owned();
        // 2)
        if position != "g" {
            input.push('<');
            input.push_str(quad.1.decode_from(&self.interner).as_str());
            input.push('>');
        }
        // 3)
        if let Some(id) = canonicalization_state
            .canonical_issuer
            .issued_identifier_map
            .get(&related)
            .or_else(|| issuer.issued_identifier_map.get(&related))
        {
            input.push_str("_:");
            input.push_str(id.as_str());
        } else {
            // 4)
            input.push_str(&self.hash_first_degree_quads(
                canonicalization_state,
                related,
                hash_algorithm,
            ));
        }
        // 5)
        Self::hash_function(&input, hash_algorithm)
    }

    /// RDFC [Hash N Degree Quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads).
    ///
    /// `depth` is this call's own nesting depth (the top-level caller passes `1`; each recursive
    /// call at step 5.4.5.1 passes its caller's `depth + 1`) — see the charging note below for why
    /// it is threaded through at all.
    ///
    /// # Errors
    ///
    /// Returns [`CanonicalizationError::WorkBudgetExceeded`] once `canonicalization_state`'s work
    /// budget is exhausted. Every gossip-path permutation examined in step 5.4 charges one work
    /// unit, as before; every invocation of this recursion at a `depth` no greater than
    /// [`SAFE_RECURSION_DEPTH`] also charges one, exactly as before this function started tracking
    /// depth at all — but a call any deeper than that charges the *entire remaining budget* in one
    /// shot, guaranteeing rejection. That is not a cosmetic difference: a *chain*-shaped input
    /// (each blank node related to only the next one, so no permutation ever grows past a handful
    /// of choices) does not trip the permutation charge at all, yet this function's own recursion
    /// at step 5.4.5.1 still descends one native stack frame per link in the chain — a per-call
    /// charge that stayed flat regardless of depth would let that native recursion run to a depth
    /// in the hundreds of thousands, well past what any thread stack survives (this crate's own
    /// debug-build test suite overflows its stack at a native `hash_n_degree_quads` depth under
    /// 1 000; an optimized build tolerates a great deal more, but nothing in the public API lets a
    /// caller promise their consumer only ever builds optimized binaries), before flat charging
    /// could ever catch up with even a generous budget.
    ///
    /// Charging is unaffected below [`SAFE_RECURSION_DEPTH`], rather than growing from the very
    /// first call, so that every legitimate case shallower than that ceiling (which is every case
    /// this crate has measured, by a wide margin — see [`DEFAULT_CANONICALIZATION_WORK_BUDGET`])
    /// is charged exactly as before and none of them newly hit the budget. An earlier attempt at
    /// this fix charged every call's own `depth` unconditionally, which is simpler but wrong: over
    /// a hundred *fully interchangeable* blank nodes chained together — not a contrived case, RDF
    /// lists of identical items produce exactly this shape — measurably exceeded the default
    /// budget despite needing nowhere near a stack-unsafe recursion depth to resolve. A second
    /// attempt scaled the per-call charge up gradually past the ceiling instead of all at once,
    /// which keeps native recursion depth *roughly* bounded but ties the exact bound to the
    /// budget's own magnitude (a caller-supplied larger budget, meant only to allow more
    /// *permutation* slack, would also — surprisingly — permit deeper native recursion). Charging
    /// the whole remaining budget the instant `depth` exceeds the ceiling instead makes the
    /// deepest native frame this function can ever legitimately reach exactly
    /// `SAFE_RECURSION_DEPTH + 1`, independent of `work_budget`'s magnitude. `u64::MAX` (the budget
    /// [`Dataset::canonicalize`] uses) still can never be exceeded, by the same saturating-addition
    /// argument as [`CanonicalizationState::charge`] — this is still the *same* work-budget
    /// mechanism, not a second, independently-failable limit; it just charges a different amount.
    fn hash_n_degree_quads(
        &self,
        canonicalization_state: &CanonicalizationState<'_>,
        identifier: InternedBlankNode,
        issuer: &IdentifierIssuer,
        algorithm: CanonicalizationAlgorithm,
        hash_algorithm: Option<CanonicalizationHashAlgorithm>,
        depth: usize,
    ) -> Result<(IdentifierIssuer, String), CanonicalizationError> {
        // See the doc comment above for why this is `1` up to `SAFE_RECURSION_DEPTH` and the
        // entire remaining budget beyond it, rather than either a flat `1` or an unconditional
        // `depth`.
        let depth_charge = if depth <= SAFE_RECURSION_DEPTH {
            1
        } else {
            u64::MAX
        };
        canonicalization_state.charge(depth_charge)?;
        let mut issuer = issuer.clone();
        // 1)
        let mut h_n = BTreeMap::<_, HashSet<_>>::new();
        // 2)
        let quads = &canonicalization_state.blank_node_to_quads_map[&identifier];
        // 3)
        for quad in quads {
            // 3.1)
            if let InternedNamedOrBlankNode::BlankNode(component) = quad.0 {
                self.hash_related_blank_node_on_possible_component(
                    canonicalization_state,
                    component,
                    identifier,
                    quad,
                    &issuer,
                    "s",
                    &mut h_n,
                    hash_algorithm,
                );
            }
            if let InternedTerm::BlankNode(component) = quad.2 {
                self.hash_related_blank_node_on_possible_component(
                    canonicalization_state,
                    component,
                    identifier,
                    quad,
                    &issuer,
                    "o",
                    &mut h_n,
                    hash_algorithm,
                );
            }
            #[cfg(feature = "rdf-12")]
            if let InternedTerm::Triple(t) = &quad.2 {
                self.hash_related_blank_node_on_possible_triple(
                    canonicalization_state,
                    t,
                    identifier,
                    quad,
                    &issuer,
                    &mut h_n,
                    hash_algorithm,
                );
            }
            if let InternedGraphName::BlankNode(component) = quad.3 {
                self.hash_related_blank_node_on_possible_component(
                    canonicalization_state,
                    component,
                    identifier,
                    quad,
                    &issuer,
                    "g",
                    &mut h_n,
                    hash_algorithm,
                );
            }
        }
        // 4)
        let mut data_to_hash = String::new();
        // 5)
        for (related_hash, blank_node_list) in h_n {
            // 5.1)
            data_to_hash.push_str(&related_hash);
            // 5.2)
            let mut chosen_path = String::new();
            // 5.3)
            let mut chosen_issuer = IdentifierIssuer::new("");
            // 5.4)
            'perm: for p in Permutations::new(blank_node_list) {
                canonicalization_state.charge(1)?;
                // 5.4.1)
                let mut issuer_copy = issuer.clone();
                // 5.4.2)
                let mut path = String::new();
                // 5.4.3)
                let mut recursion_list = Vec::new();
                // 5.4.4)
                for related in p {
                    // 5.4.4.1)
                    if let Some(id) = canonicalization_state
                        .canonical_issuer
                        .issued_identifier_map
                        .get(&related)
                    {
                        path.push_str("_:");
                        path.push_str(id.as_str());
                    } else {
                        // 5.4.4.2)
                        // 5.4.4.2.1)
                        if !issuer_copy.issued_identifier_map.contains_key(&related) {
                            recursion_list.push(related);
                        }
                        // 5.4.4.2.2)
                        let id = Self::issue_identifier(
                            &mut issuer_copy,
                            related,
                            algorithm,
                            &related_hash,
                        );
                        path.push_str("_:");
                        path.push_str(id.as_str());
                    }
                    // 5.4.4.3)
                    if !chosen_path.is_empty()
                        && path.len() >= chosen_path.len()
                        && path > chosen_path
                    {
                        continue 'perm;
                    }
                }
                // 5.4.5)
                for related in recursion_list {
                    // 5.4.5.1)
                    let (result_identifier_issuer, result_hash) = self.hash_n_degree_quads(
                        canonicalization_state,
                        related,
                        &issuer_copy,
                        algorithm,
                        hash_algorithm,
                        depth.saturating_add(1),
                    )?;
                    // 5.4.5.2)
                    let id =
                        Self::issue_identifier(&mut issuer_copy, related, algorithm, &result_hash);
                    path.push_str("_:");
                    path.push_str(id.as_str());
                    // 5.4.5.3)
                    path.push('<');
                    path.push_str(&result_hash);
                    path.push('>');
                    // 5.4.5.4)
                    issuer_copy = result_identifier_issuer;
                    // 5.4.5.5)
                    if !chosen_path.is_empty()
                        && path.len() >= chosen_path.len()
                        && path > chosen_path
                    {
                        continue 'perm;
                    }
                }
                // 5.4.6)
                if chosen_path.is_empty() || path < chosen_path {
                    chosen_path = path;
                    chosen_issuer = issuer_copy;
                }
            }
            // 5.5)
            data_to_hash.push_str(&chosen_path);
            // 5.6)
            issuer = chosen_issuer;
        }
        // 6)
        Ok((issuer, Self::hash_function(&data_to_hash, hash_algorithm)))
    }

    #[cfg(feature = "rdf-12")]
    #[allow(clippy::too_many_arguments)]
    fn hash_related_blank_node_on_possible_triple(
        &self,
        canonicalization_state: &CanonicalizationState<'_>,
        triple: &InternedTriple,
        identifier: InternedBlankNode,
        quad: &(
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
        issuer: &IdentifierIssuer,
        h_n: &mut BTreeMap<String, HashSet<InternedBlankNode>>,
        hash_algorithm: Option<CanonicalizationHashAlgorithm>,
    ) {
        if let InternedNamedOrBlankNode::BlankNode(component) = triple.subject {
            self.hash_related_blank_node_on_possible_component(
                canonicalization_state,
                component,
                identifier,
                quad,
                issuer,
                "os",
                h_n,
                hash_algorithm,
            );
        }
        if let InternedTerm::BlankNode(component) = &triple.object {
            self.hash_related_blank_node_on_possible_component(
                canonicalization_state,
                *component,
                identifier,
                quad,
                issuer,
                "oo",
                h_n,
                hash_algorithm,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn hash_related_blank_node_on_possible_component(
        &self,
        canonicalization_state: &CanonicalizationState<'_>,
        component: InternedBlankNode,
        identifier: InternedBlankNode,
        quad: &(
            InternedNamedOrBlankNode,
            InternedNamedNode,
            InternedTerm,
            InternedGraphName,
        ),
        issuer: &IdentifierIssuer,
        position: &str,
        h_n: &mut BTreeMap<String, HashSet<InternedBlankNode>>,
        hash_algorithm: Option<CanonicalizationHashAlgorithm>,
    ) {
        if component != identifier {
            // 3.1.1)
            let hash = self.hash_related_blank_node(
                canonicalization_state,
                component,
                quad,
                issuer,
                position,
                hash_algorithm,
            );
            // 3.1.2)
            h_n.entry(hash).or_default().insert(component);
        }
    }

    fn hash_function(input: &str, hash_algorithm: Option<CanonicalizationHashAlgorithm>) -> String {
        match hash_algorithm {
            #[cfg(feature = "rdfc-10")]
            Some(CanonicalizationHashAlgorithm::Sha256) => to_hex(&sha256(input.as_bytes())),
            #[cfg(feature = "rdfc-10")]
            Some(CanonicalizationHashAlgorithm::Sha384) => to_hex(&sha384(input.as_bytes())),
            None => {
                let mut hasher = DefaultHasher::new();
                input.hash(&mut hasher);
                hasher.finish().to_string()
            }
        }
    }

    #[allow(clippy::needless_collect)] // The collect releases the borrow on `self.spog`.
    fn map_blank_nodes(
        &mut self,
        bnode_mapping: &HashMap<InternedBlankNode, BlankNode>,
    ) -> Vec<(
        InternedNamedOrBlankNode,
        InternedNamedNode,
        InternedTerm,
        InternedGraphName,
    )> {
        let old_quads: Vec<_> = self.spog.iter().cloned().collect();
        old_quads
            .into_iter()
            .map(|(s, p, o, g)| {
                (
                    match s {
                        InternedNamedOrBlankNode::NamedNode(_) => s,
                        InternedNamedOrBlankNode::BlankNode(bnode) => {
                            InternedNamedOrBlankNode::BlankNode(InternedBlankNode::encoded_into(
                                bnode_mapping[&bnode].as_ref(),
                                &mut self.interner,
                            ))
                        }
                    },
                    p,
                    match o {
                        InternedTerm::NamedNode(_) | InternedTerm::Literal(_) => o,
                        InternedTerm::BlankNode(bnode) => {
                            InternedTerm::BlankNode(InternedBlankNode::encoded_into(
                                bnode_mapping[&bnode].as_ref(),
                                &mut self.interner,
                            ))
                        }
                        #[cfg(feature = "rdf-12")]
                        InternedTerm::Triple(triple) => {
                            let mapped = self.map_triple_blank_nodes(&triple, bnode_mapping);
                            InternedTerm::Triple(Box::new(InternedTriple::encoded_into(
                                mapped.as_ref(),
                                &mut self.interner,
                            )))
                        }
                    },
                    match g {
                        InternedGraphName::NamedNode(_) | InternedGraphName::DefaultGraph => g,
                        InternedGraphName::BlankNode(bnode) => {
                            InternedGraphName::BlankNode(InternedBlankNode::encoded_into(
                                bnode_mapping[&bnode].as_ref(),
                                &mut self.interner,
                            ))
                        }
                    },
                )
            })
            .collect()
    }

    #[cfg(feature = "rdf-12")]
    fn map_triple_blank_nodes(
        &mut self,
        triple: &InternedTriple,
        bnode_mapping: &HashMap<InternedBlankNode, BlankNode>,
    ) -> Triple {
        Triple {
            subject: if let InternedNamedOrBlankNode::BlankNode(bnode) = &triple.subject {
                bnode_mapping[bnode].clone().into()
            } else {
                triple.subject.decode_from(&self.interner).into_owned()
            },
            predicate: triple.predicate.decode_from(&self.interner).into_owned(),
            object: if let InternedTerm::BlankNode(bnode) = &triple.object {
                bnode_mapping[bnode].clone().into()
            } else if let InternedTerm::Triple(t) = &triple.object {
                self.map_triple_blank_nodes(t, bnode_mapping).into()
            } else {
                triple.object.decode_from(&self.interner).into_owned()
            },
        }
    }
}

/// A RDFC [canonicalization state](https://www.w3.org/TR/rdf-canon/#canon-state).
struct CanonicalizationState<'a> {
    blank_node_to_quads_map: QuadsPerBlankNode<'a>,
    hash_to_blank_nodes_map: BTreeMap<String, Vec<InternedBlankNode>>,
    canonical_issuer: IdentifierIssuer,
    /// The maximum number of abstract work units `hash_n_degree_quads` may consume across this
    /// canonicalization run. `u64::MAX` behaves as an unlimited budget (see `charge`).
    work_budget: u64,
    /// The work units consumed so far.
    work: Cell<u64>,
}

impl CanonicalizationState<'_> {
    /// Charges `amount` work units against `work_budget`, failing once it is exhausted.
    ///
    /// A `work_budget` of `u64::MAX` can never be exceeded: `work` only ever grows through
    /// saturating addition, so it can never end up greater than `u64::MAX` itself. This is what
    /// lets `Dataset::canonicalize` share this bookkeeping with `Dataset::try_canonicalize`
    /// while staying infallible.
    fn charge(&self, amount: u64) -> Result<(), CanonicalizationError> {
        let used = self.work.get().saturating_add(amount);
        self.work.set(used);
        if used > self.work_budget {
            Err(CanonicalizationError::WorkBudgetExceeded {
                budget: self.work_budget,
            })
        } else {
            Ok(())
        }
    }
}

/// A RDFC [identifier issuer](https://www.w3.org/TR/rdf-canon/#dfn-identifier-issuer).
#[derive(Clone)]
struct IdentifierIssuer {
    identifier_prefix: &'static str,
    identifier_counter: u32,
    issued_identifier_map: HashMap<InternedBlankNode, BlankNode>,
    issued_identifier_order: Vec<InternedBlankNode>, // Tracks insertion order in the map.
    already_issued_identifiers: HashSet<BlankNode>,
}

impl IdentifierIssuer {
    fn new(identifier_prefix: &'static str) -> Self {
        Self {
            identifier_prefix,
            identifier_counter: 0,
            issued_identifier_map: HashMap::new(),
            issued_identifier_order: Vec::new(),
            already_issued_identifiers: HashSet::new(),
        }
    }
}

/// A lazily generated iterator over every distinct permutation of a list, in ascending code
/// point order, used by [hash-n-degree-quads step
/// 5.4](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) to explore every gossip path.
///
/// The permutations are produced one at a time rather than all up front, so that the caller can
/// charge the work budget *before* each one is examined: materializing all `n!` permutations of
/// a large related-blank-node list eagerly would allocate a factorial amount of memory before a
/// single budget check could ever fire, which is itself a denial-of-service vector independent
/// of how much CPU time is bounded.
///
/// The input is sorted on construction, so the first permutation yielded is the smallest and the
/// last is the largest. Equal elements are interchangeable, so only *distinct* permutations of
/// the (possibly repeating) input multiset are produced — a list such as `[a, a]` yields the
/// single permutation `[a, a]` rather than two identical ones.
struct Permutations<T> {
    items: Vec<T>,
    started: bool,
    exhausted: bool,
}

impl<T: Ord> Permutations<T> {
    /// Creates an iterator over every distinct permutation of `items`.
    fn new(items: impl IntoIterator<Item = T>) -> Self {
        let mut items: Vec<T> = items.into_iter().collect();
        items.sort_unstable();
        Self {
            items,
            started: false,
            exhausted: false,
        }
    }
}

impl<T: Ord + Copy> Iterator for Permutations<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.items.clone());
        }
        if next_permutation(&mut self.items) {
            Some(self.items.clone())
        } else {
            self.exhausted = true;
            None
        }
    }
}

/// Rewrites `items` with the next permutation in ascending order, returning `false` (and leaving
/// `items` sorted ascending) when the current permutation is already the largest one.
fn next_permutation<T: Ord>(items: &mut [T]) -> bool {
    if items.len() < 2 {
        return false;
    }
    // Find the longest non-increasing suffix; `pivot` is the element just before it.
    let mut pivot = items.len() - 1;
    while pivot > 0 && items[pivot - 1] >= items[pivot] {
        pivot -= 1;
    }
    if pivot == 0 {
        items.reverse();
        return false;
    }
    let pivot = pivot - 1;
    // Find the rightmost successor of the pivot in the suffix and swap it in.
    let mut successor = items.len() - 1;
    while items[successor] <= items[pivot] {
        successor -= 1;
    }
    items.swap(pivot, successor);
    items[pivot + 1..].reverse();
    true
}

#[cfg(feature = "rdfc-10")]
fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0F)] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rdfc-10")]
    #[test]
    fn test_canon() {
        let p = NamedNode::new_unchecked("http://example.com/#p");
        let q = NamedNode::new_unchecked("http://example.com/#q");
        let r = NamedNode::new_unchecked("http://example.com/#r");

        let mut dataset = Dataset::new();
        let e0 = BlankNode::new_unchecked("e0");
        let e1 = BlankNode::new_unchecked("e1");
        let e2 = BlankNode::new_unchecked("e2");
        let e3 = BlankNode::new_unchecked("e3");
        dataset.insert(QuadRef::new(&p, &q, &e0, GraphNameRef::DefaultGraph));
        dataset.insert(QuadRef::new(&p, &q, &e1, GraphNameRef::DefaultGraph));
        dataset.insert(QuadRef::new(&e0, &p, &e2, GraphNameRef::DefaultGraph));
        dataset.insert(QuadRef::new(&e1, &p, &e3, GraphNameRef::DefaultGraph));
        dataset.insert(QuadRef::new(&e2, &r, &e3, GraphNameRef::DefaultGraph));
        dataset.canonicalize(CanonicalizationAlgorithm::Rdfc10 {
            hash_algorithm: CanonicalizationHashAlgorithm::Sha256,
        });

        let mut expected = Dataset::new();
        let c14n0 = BlankNode::new_unchecked("c14n0");
        let c14n1 = BlankNode::new_unchecked("c14n1");
        let c14n2 = BlankNode::new_unchecked("c14n2");
        let c14n3 = BlankNode::new_unchecked("c14n3");
        expected.insert(QuadRef::new(&p, &q, &c14n2, GraphNameRef::DefaultGraph));
        expected.insert(QuadRef::new(&p, &q, &c14n3, GraphNameRef::DefaultGraph));
        expected.insert(QuadRef::new(&c14n0, &r, &c14n1, GraphNameRef::DefaultGraph));
        expected.insert(QuadRef::new(&c14n2, &p, &c14n1, GraphNameRef::DefaultGraph));
        expected.insert(QuadRef::new(&c14n3, &p, &c14n0, GraphNameRef::DefaultGraph));
        assert_eq!(dataset, expected);
    }

    /// Builds the "poison – Clique Graph" of the W3C `rdf-canon` negative test suite
    /// (`rdfc10/test074`): a fully connected clique of `n` blank nodes. A conformant
    /// implementation must be able to reject this input rather than process it for a
    /// super-polynomial amount of time — see the module-level docs.
    fn poison_clique(n: usize) -> Dataset {
        let p = NamedNode::new_unchecked("http://example.com/p");
        let mut dataset = Dataset::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let subject = BlankNode::new_unchecked(format!("e{i}"));
                    let object = BlankNode::new_unchecked(format!("e{j}"));
                    dataset.insert(QuadRef::new(
                        &subject,
                        &p,
                        &object,
                        GraphNameRef::DefaultGraph,
                    ));
                }
            }
        }
        dataset
    }

    /// Builds a chain of `n` blank nodes (`_:e0` related to `_:e1`, `_:e1` to `_:e2`, ...), all
    /// through the same predicate: a shape with *no* combinatorial blowup ([`poison_clique`]'s
    /// attack vector — every gossip-path permutation here is trivial, one candidate wide) but
    /// which drives `hash_n_degree_quads`'s own *native recursion* to a depth proportional to `n`,
    /// since every interior node is structurally indistinguishable from every other and so must be
    /// disambiguated by that recursion (see its doc comment). Before `hash_n_degree_quads` charged
    /// anything past [`SAFE_RECURSION_DEPTH`], a long enough chain reached this crate's own public,
    /// "safe for untrusted input" [`Dataset::try_canonicalize`] and overflowed the native stack —
    /// aborting the process — long before the flat, depth-blind charge caught up with any budget
    /// worth shipping as a default.
    fn blank_node_chain(n: usize) -> Dataset {
        let p = NamedNode::new_unchecked("http://example.com/p");
        let mut dataset = Dataset::new();
        for i in 0..n {
            let subject = BlankNode::new_unchecked(format!("e{i}"));
            let object = BlankNode::new_unchecked(format!("e{}", i + 1));
            dataset.insert(QuadRef::new(
                &subject,
                &p,
                &object,
                GraphNameRef::DefaultGraph,
            ));
        }
        dataset
    }

    #[test]
    fn short_chain_stays_within_the_default_budget() {
        // A chain shallower than `SAFE_RECURSION_DEPTH` must canonicalize exactly as it did before
        // `hash_n_degree_quads` started charging anything past that depth — this is the regression
        // guard for the charging *refinement* itself: an earlier version of this fix charged every
        // call's raw depth unconditionally from call 1, which also protected against
        // `long_chain_is_rejected_not_crashed` below but, measured against this exact case,
        // wrongly rejected a chain of only about a hundred *fully interchangeable* blank nodes —
        // a shape [`blank_node_chain`]'s doc comment explains is not contrived, and nowhere near a
        // stack-unsafe recursion depth on its own.
        let mut dataset = blank_node_chain(200);
        dataset
            .try_canonicalize(CanonicalizationAlgorithm::Unstable)
            .expect("a 200-node chain is far shallower than SAFE_RECURSION_DEPTH (256)");
    }

    #[test]
    fn long_chain_is_rejected_not_crashed() {
        // Before `hash_n_degree_quads` charged depth at all, this canonicalized successfully in a
        // release build (a stack overflow needs a much longer chain than this to reproduce
        // reliably — this repository's standalone reproduction used 300 000 blank nodes) but
        // demonstrates the same defect class: charging that never scales with native recursion
        // depth cannot bound native recursion depth. This is a small enough chain to run quickly
        // under the *shipped* `DEFAULT_CANONICALIZATION_WORK_BUDGET`, not just a reduced test
        // budget: rejection now happens at a native recursion depth of exactly
        // `SAFE_RECURSION_DEPTH + 1`, regardless of `n` or of `work_budget`'s own magnitude (see
        // `hash_n_degree_quads`'s doc comment) — long before any thread stack would be at risk.
        let mut dataset = blank_node_chain(2_000);
        let error = dataset
            .try_canonicalize(CanonicalizationAlgorithm::Unstable)
            .expect_err("a 2 000-node indistinguishable chain must be rejected, not hang or crash");
        assert_eq!(
            error,
            CanonicalizationError::WorkBudgetExceeded {
                budget: DEFAULT_CANONICALIZATION_WORK_BUDGET,
            }
        );
    }

    #[cfg(feature = "rdfc-10")]
    #[test]
    fn long_chain_is_rejected_not_crashed_rdfc10() {
        // The RDFC-1.0 variant of `long_chain_is_rejected_not_crashed`: the crate's own W3C
        // RDFC-1.0 poison-graph regression (`poison_clique_is_rejected_well_within_budget_rdfc10`)
        // covers the combinatorial attack; this covers the same public entry point against the
        // chain-shaped, native-recursion-depth attack.
        let mut dataset = blank_node_chain(2_000);
        let error = dataset
            .try_canonicalize(CanonicalizationAlgorithm::Rdfc10 {
                hash_algorithm: CanonicalizationHashAlgorithm::Sha256,
            })
            .expect_err("a 2 000-node indistinguishable chain must be rejected, not hang or crash");
        assert!(matches!(
            error,
            CanonicalizationError::WorkBudgetExceeded { .. }
        ));
    }

    /// A budget well above every legitimate case measured in the W3C conformance suite (whose
    /// heaviest case needs 3 348 units) but far below [`DEFAULT_CANONICALIZATION_WORK_BUDGET`],
    /// so the tests below reject the poison clique in well under a second instead of paying the
    /// full default's cost on every run. `poison_clique_exceeds_the_default_budget` (below,
    /// `#[ignore]`d) separately locks in that the clique exceeds the *shipped default* too, so
    /// this smaller budget is a speed optimization, not a weaker claim.
    const FAST_TEST_BUDGET: u64 = 10_000;

    #[test]
    fn poison_clique_is_rejected_well_within_budget() {
        // The bug this guards against is not specific to RDFC-1.0: every `CanonicalizationAlgorithm`
        // shares the same `hash_n_degree_quads` recursion, so `Unstable` is equally exposed and
        // needs no extra feature to exercise here.
        let mut dataset = poison_clique(10);
        let error = dataset
            .try_canonicalize_with_budget(CanonicalizationAlgorithm::Unstable, FAST_TEST_BUDGET)
            .expect_err("a 10-node fully-connected blank-node clique must be rejected, not hang");
        assert!(matches!(
            error,
            CanonicalizationError::WorkBudgetExceeded { .. }
        ));
    }

    #[cfg(feature = "rdfc-10")]
    #[test]
    fn poison_clique_is_rejected_well_within_budget_rdfc10() {
        // The W3C `rdf-canon` test suite's `test074` negative case ("poison - Clique Graph"):
        // RDFC-1.0 implementations must be able to reject this input rather than run unboundedly
        // (https://www.w3.org/TR/rdf-canon/#security-considerations).
        let mut dataset = poison_clique(10);
        let error = dataset
            .try_canonicalize_with_budget(
                CanonicalizationAlgorithm::Rdfc10 {
                    hash_algorithm: CanonicalizationHashAlgorithm::Sha256,
                },
                FAST_TEST_BUDGET,
            )
            .expect_err("the W3C poison clique (test074) must be rejected, not hang");
        assert!(matches!(
            error,
            CanonicalizationError::WorkBudgetExceeded { .. }
        ));
    }

    #[test]
    fn default_work_budget_has_not_silently_drifted() {
        // Locks in the constant so a future edit cannot casually shrink or grow it without
        // updating the measurement this value is based on (see its doc comment).
        assert_eq!(DEFAULT_CANONICALIZATION_WORK_BUDGET, 1 << 18);
    }

    /// Slow (tens of seconds in an unoptimized build): asserts the property the fast
    /// `FAST_TEST_BUDGET`-based tests above only take on trust — that the W3C poison clique
    /// (`test074`) exceeds [`DEFAULT_CANONICALIZATION_WORK_BUDGET`] itself, so
    /// [`Dataset::try_canonicalize`] (using the *shipped default*, not a smaller test budget)
    /// rejects it. Without this test, a future change that shrank the clique's charged work below
    /// the default (e.g. restructuring charge points, or lowering the default) could silently
    /// stop rejecting this input while every other test here kept passing.
    ///
    /// Run explicitly: `cargo test -p oxixml-model --all-features -- --ignored`.
    #[ignore = "slow: exercises the full DEFAULT_CANONICALIZATION_WORK_BUDGET, not a fast test budget"]
    #[test]
    fn poison_clique_exceeds_the_default_budget() {
        let mut dataset = poison_clique(10);
        let error = dataset
            .try_canonicalize(CanonicalizationAlgorithm::Unstable)
            .expect_err("the W3C poison clique must exceed the shipped default work budget");
        assert_eq!(
            error,
            CanonicalizationError::WorkBudgetExceeded {
                budget: DEFAULT_CANONICALIZATION_WORK_BUDGET,
            }
        );
    }

    #[test]
    fn try_canonicalize_agrees_with_canonicalize_on_ordinary_input()
    -> Result<(), CanonicalizationError> {
        // A sanity check that the bounded and unbounded entry points agree on non-adversarial
        // input: `canonicalize` is `try_canonicalize_with_budget(algorithm, u64::MAX)` in
        // everything but name.
        let p = NamedNode::new_unchecked("http://example.com/p");
        let build = || {
            let mut dataset = Dataset::new();
            dataset.insert(QuadRef::new(
                &BlankNode::new_unchecked("e0"),
                &p,
                &BlankNode::new_unchecked("e1"),
                GraphNameRef::DefaultGraph,
            ));
            dataset
        };

        let mut via_try = build();
        via_try.try_canonicalize(CanonicalizationAlgorithm::Unstable)?;

        let mut via_infallible = build();
        via_infallible.canonicalize(CanonicalizationAlgorithm::Unstable);

        assert_eq!(via_try, via_infallible);
        Ok(())
    }

    #[test]
    fn permutations_of_empty_and_singleton_lists() {
        assert_eq!(
            Permutations::new(Vec::<u8>::new()).collect::<Vec<_>>(),
            vec![Vec::<u8>::new()]
        );
        assert_eq!(
            Permutations::new(vec![7_u8]).collect::<Vec<_>>(),
            vec![vec![7_u8]]
        );
    }

    #[test]
    fn permutations_in_ascending_order() {
        let permutations = Permutations::new(vec![2_u8, 1, 3]).collect::<Vec<_>>();
        assert_eq!(
            permutations,
            vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
                vec![2, 1, 3],
                vec![2, 3, 1],
                vec![3, 1, 2],
                vec![3, 2, 1],
            ]
        );
    }

    #[test]
    fn permutations_collapse_equal_elements() {
        assert_eq!(
            Permutations::new(vec![1_u8, 1]).collect::<Vec<_>>(),
            vec![vec![1, 1]]
        );
        assert_eq!(
            Permutations::new(vec![1_u8, 1, 2]).collect::<Vec<_>>(),
            vec![vec![1, 1, 2], vec![1, 2, 1], vec![2, 1, 1]]
        );
    }

    #[test]
    fn permutation_count_is_factorial() {
        assert_eq!(Permutations::new(vec![1_u8, 2, 3, 4, 5]).count(), 120);
    }
}
