//! The RDFC-1.0 canonicalization engine.
//!
//! This is a direct, self-contained transcription of the
//! [RDF Dataset Canonicalization (RDFC-1.0)](https://www.w3.org/TR/rdf-canon/) algorithm operating
//! on owned [`oxixml_model`] terms, augmented with a configurable
//! [`WorkBudget`](crate::WorkBudget) that guards the
//! [hash-n-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) gossip-path recursion
//! against adversarial input (see the spec's
//! [security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations)).
//!
//! Step numbers in the comments refer to the numbered steps of the specification.

use crate::error::{CanonicalizationError, Result};
use crate::issuer::IdentifierIssuer;
use crate::options::{CanonicalizationOptions, HashAlgorithm};
use crate::permutations::Permutations;
use crate::result::CanonicalizationResult;
use crate::terms::{for_each_blank_node_in_quad, relabel_quad};
use oxixml_model::{Dataset, GraphName, NamedOrBlankNode, Quad, Term};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The identifier prefix of the [canonical issuer](https://www.w3.org/TR/rdf-canon/#dfn-canonical-issuer).
const CANONICAL_PREFIX: &str = "c14n";

/// The identifier prefix of the temporary issuers created in step 5.2.2.
const TEMPORARY_PREFIX: &str = "b";

/// The maximum [`Normalizer::hash_n_degree_quads`] recursion depth any *bounded* [`WorkBudget`]
/// (crate::WorkBudget) (anything less than [`WorkBudget::unlimited`](crate::WorkBudget::unlimited))
/// ever lets that recursion reach. See that method's doc comment for the full reasoning; in short,
/// a call at this depth or shallower charges its usual one work unit, exactly as before this
/// engine tracked depth at all, while a call any deeper charges the caller's entire remaining
/// budget in one shot, guaranteeing rejection regardless of how large the configured budget is.
/// This number matches this workspace's usual structural-nesting ceiling (element/DOM trees, DER
/// structures, and RDF/HTML/SVG document nesting all use the same 256) — chosen because it
/// comfortably covers every legitimate case this crate has measured (see
/// [`DEFAULT_WORK_BUDGET`](crate::DEFAULT_WORK_BUDGET)'s doc comment) while still being a depth any
/// thread's native call stack survives many times over.
const SAFE_RECURSION_DEPTH: usize = 256;

/// Canonicalizes `quads` with the given options, returning the full [`CanonicalizationResult`].
///
/// `quads` must already be duplicate-free — an RDF dataset is a *set* of quads.
pub(crate) fn canonicalize(
    quads: Vec<Quad>,
    options: CanonicalizationOptions,
) -> Result<CanonicalizationResult> {
    Normalizer::new(quads, options).run()
}

/// The [canonicalization state](https://www.w3.org/TR/rdf-canon/#canonicalization-state) plus the
/// immutable input and the work counter.
struct Normalizer {
    /// The (deduplicated) input quads.
    quads: Vec<Quad>,
    /// Maps each blank-node identifier to the indices (into [`Self::quads`]) of the quads that
    /// mention it — the
    /// [blank node to quads map](https://www.w3.org/TR/rdf-canon/#dfn-blank-node-to-quads-map).
    blank_node_to_quads: HashMap<String, Vec<usize>>,
    /// The blank-node identifiers of [`Self::blank_node_to_quads`], in code point order.
    blank_nodes: Vec<String>,
    /// The memoized [hash-first-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-1d-quads)
    /// result for every blank node, as the spec recommends.
    first_degree_hashes: HashMap<String, String>,
    hash_algorithm: HashAlgorithm,
    /// The maximum number of work units this run may consume.
    budget: u64,
    /// The work units consumed so far.
    work: Cell<u64>,
}

impl Normalizer {
    fn new(quads: Vec<Quad>, options: CanonicalizationOptions) -> Self {
        // Steps 1 and 2: build the blank node to quads map.
        let mut blank_node_to_quads: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, quad) in quads.iter().enumerate() {
            let mut seen = HashSet::new();
            for_each_blank_node_in_quad(quad, &mut |id| {
                if seen.insert(id.to_owned()) {
                    blank_node_to_quads
                        .entry(id.to_owned())
                        .or_default()
                        .push(index);
                }
            });
        }
        let mut blank_nodes: Vec<String> = blank_node_to_quads.keys().cloned().collect();
        blank_nodes.sort_unstable();

        let mut normalizer = Self {
            quads,
            blank_node_to_quads,
            blank_nodes,
            first_degree_hashes: HashMap::new(),
            hash_algorithm: options.hash_algorithm(),
            budget: options.work_budget().max_work_units(),
            work: Cell::new(0),
        };
        // Step 3: hash the first degree quads of every blank node exactly once.
        normalizer.first_degree_hashes = normalizer
            .blank_nodes
            .iter()
            .map(|identifier| {
                (
                    identifier.clone(),
                    normalizer.compute_first_degree_hash(identifier),
                )
            })
            .collect();
        normalizer
    }

    /// Charges `amount` work units against the budget, failing once it is exhausted.
    fn charge(&self, amount: u64) -> Result<()> {
        let used = self.work.get().saturating_add(amount);
        self.work.set(used);
        if used > self.budget {
            Err(CanonicalizationError::WorkBudgetExceeded {
                budget: self.budget,
            })
        } else {
            Ok(())
        }
    }

    /// Runs the [canonicalization algorithm](https://www.w3.org/TR/rdf-canon/#canon-algo-algo).
    fn run(&self) -> Result<CanonicalizationResult> {
        let mut canonical_issuer = IdentifierIssuer::new(CANONICAL_PREFIX);

        // Step 3: group the blank nodes by their first degree hash.
        let mut hash_to_blank_nodes: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for identifier in &self.blank_nodes {
            hash_to_blank_nodes
                .entry(self.first_degree_hash(identifier))
                .or_default()
                .push(identifier);
        }

        // Step 4: issue canonical identifiers to the blank nodes with a unique first degree hash
        // and drop them from the map, in code point order of the hash.
        hash_to_blank_nodes.retain(|_, identifier_list| match identifier_list.as_slice() {
            [single] => {
                canonical_issuer.issue(single);
                false
            }
            _ => true,
        });

        // Step 5: resolve the remaining (hash-colliding) blank nodes, in code point order of the
        // hash, via hash-n-degree-quads.
        for identifier_list in hash_to_blank_nodes.into_values() {
            // Step 5.1.
            let mut hash_path_list = Vec::new();
            // Step 5.2.
            for identifier in identifier_list {
                // Step 5.2.1: already assigned during an earlier gossip path.
                if canonical_issuer.has_issued(identifier) {
                    continue;
                }
                // Steps 5.2.2 and 5.2.3.
                let mut temporary_issuer = IdentifierIssuer::new(TEMPORARY_PREFIX);
                temporary_issuer.issue(identifier);
                // Step 5.2.4.
                hash_path_list.push(self.hash_n_degree_quads(
                    &canonical_issuer,
                    identifier,
                    &temporary_issuer,
                    1,
                )?);
            }
            // Step 5.3: code point ordered by hash; ties keep the step 5.2 order.
            hash_path_list.sort_by(|(_, left), (_, right)| left.cmp(right));
            for (result_issuer, _) in hash_path_list {
                // Step 5.3.1.
                for existing_identifier in result_issuer.issued_order() {
                    canonical_issuer.issue(existing_identifier);
                }
            }
        }

        // Step 6: serialize and relabel using the canonical issuer's issued identifiers map.
        Ok(self.build_result(&canonical_issuer))
    }

    /// The memoized [hash-first-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-1d-quads)
    /// value of `reference`, which must be a blank node of the input dataset.
    fn first_degree_hash(&self, reference: &str) -> &str {
        self.first_degree_hashes
            .get(reference)
            .map_or("", String::as_str)
    }

    /// [Hash First Degree Quads](https://www.w3.org/TR/rdf-canon/#hash-1d-quads).
    fn compute_first_degree_hash(&self, reference: &str) -> String {
        // Steps 1 to 3: serialize each mention with `_:a` for the reference node and `_:z` for
        // every other blank node.
        let mut nquads = Vec::new();
        for &index in self.mentions(reference) {
            let relabeled = relabel_quad(&self.quads[index], &|id| {
                if id == reference {
                    "a".to_owned()
                } else {
                    "z".to_owned()
                }
            });
            nquads.push(format!("{relabeled} .\n"));
        }
        // Steps 4 and 5: sort in code point order and hash the concatenation.
        nquads.sort_unstable();
        self.hash_algorithm.hex_digest(&nquads.concat())
    }

    /// The [mention set](https://www.w3.org/TR/rdf-canon/#dfn-mention-set) of `identifier`, as
    /// indices into [`Self::quads`].
    fn mentions(&self, identifier: &str) -> &[usize] {
        self.blank_node_to_quads
            .get(identifier)
            .map_or(&[], Vec::as_slice)
    }

    /// [Hash Related Blank Node](https://www.w3.org/TR/rdf-canon/#hash-related-blank-node).
    fn hash_related_blank_node(
        &self,
        canonical_issuer: &IdentifierIssuer,
        related: &str,
        quad: &Quad,
        issuer: &IdentifierIssuer,
        position: Position,
    ) -> String {
        // Step 1.
        let mut input = position.as_str().to_owned();
        // Step 2.
        if position != Position::GraphName {
            input.push('<');
            input.push_str(quad.predicate.as_str());
            input.push('>');
        }
        // Step 3: prefer the canonical identifier, then the temporary one.
        if let Some(id) = canonical_issuer
            .issued(related)
            .or_else(|| issuer.issued(related))
        {
            input.push_str("_:");
            input.push_str(id);
        } else {
            // Step 4: otherwise fall back to the first degree hash (without a `_:` prefix).
            input.push_str(self.first_degree_hash(related));
        }
        // Step 5.
        self.hash_algorithm.hex_digest(&input)
    }

    /// [Hash N-Degree Quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads).
    ///
    /// Returns the issuer that resulted from exploring the shortest gossip path together with the
    /// N-degree hash of `identifier`. `depth` is this call's own nesting depth: the top-level
    /// caller in [`Self::run`] passes `1`, and the recursive call in step 5.4.5.1 below passes its
    /// caller's `depth + 1`.
    ///
    /// A call charges one work unit as long as `depth` is no greater than
    /// [`SAFE_RECURSION_DEPTH`]; a call any deeper charges the *entire remaining budget*, which
    /// guarantees rejection under any bounded [`WorkBudget`](crate::WorkBudget) no matter its
    /// magnitude. A flat one-unit charge at every depth (this engine's original behavior) bounds
    /// only the *combinatorial* cost of the gossip-path search — the permutation charge in step
    /// 5.4 below already does that — but says nothing about how deep this function's own
    /// recursion at step 5.4.5.1 may go natively: a *chain*-shaped dataset (each blank node
    /// related to only the next one, so no permutation ever grows past a handful of choices, and
    /// the permutation charge never fires) drives native recursion depth directly with the length
    /// of the chain, and a long enough chain overflows the stack — aborting the process — long
    /// before a flat charge could ever catch up with any budget worth shipping as a default. See
    /// [`SAFE_RECURSION_DEPTH`] for why the cutoff is a hard one (charging the whole remaining
    /// budget) rather than a charge that merely grows with depth: the latter still ties the exact
    /// maximum native depth to the budget's own magnitude, so a caller raising the budget purely
    /// for more *permutation* slack would, surprisingly, also permit deeper native recursion.
    /// [`WorkBudget::unlimited`](crate::WorkBudget::unlimited) still can never be exceeded by any
    /// charge, by the same saturating-addition argument as [`Self::charge`] — this is still the
    /// *same* work-budget mechanism, not a second, independently-failable limit; it just charges a
    /// different amount depending on depth.
    fn hash_n_degree_quads(
        &self,
        canonical_issuer: &IdentifierIssuer,
        identifier: &str,
        issuer: &IdentifierIssuer,
        depth: usize,
    ) -> Result<(IdentifierIssuer, String)> {
        self.charge(if depth <= SAFE_RECURSION_DEPTH {
            1
        } else {
            u64::MAX
        })?;
        let mut issuer = issuer.clone();

        // Steps 1 to 3: relate hashes to the blank nodes reachable in one mention.
        let mut hash_to_related: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for &index in self.mentions(identifier) {
            let quad = &self.quads[index];
            self.accumulate_related(
                canonical_issuer,
                identifier,
                quad,
                &issuer,
                &mut hash_to_related,
            );
        }

        // Step 4.
        let mut data_to_hash = String::new();
        // Step 5: code point ordered by related hash.
        for (related_hash, blank_node_list) in hash_to_related {
            // Step 5.1.
            data_to_hash.push_str(&related_hash);
            // Step 5.2.
            let mut chosen_path = String::new();
            // Step 5.3.
            let mut chosen_issuer = issuer.clone();
            // Step 5.4: explore every gossip path.
            'permutation: for permutation in Permutations::new(blank_node_list) {
                self.charge(1)?;
                // Step 5.4.1.
                let mut issuer_copy = issuer.clone();
                // Step 5.4.2.
                let mut path = String::new();
                // Step 5.4.3.
                let mut recursion_list = Vec::new();
                // Step 5.4.4.
                for related in &permutation {
                    // Step 5.4.4.1.
                    if let Some(id) = canonical_issuer.issued(related) {
                        path.push_str("_:");
                        path.push_str(id);
                    } else {
                        // Step 5.4.4.2.
                        if !issuer_copy.has_issued(related) {
                            recursion_list.push(related.clone());
                        }
                        path.push_str("_:");
                        let issued = issuer_copy.issue(related).to_owned();
                        path.push_str(&issued);
                    }
                    // Step 5.4.4.3.
                    if is_worse_than(&path, &chosen_path) {
                        continue 'permutation;
                    }
                }
                // Step 5.4.5.
                for related in recursion_list {
                    // Step 5.4.5.1.
                    let (result_issuer, result_hash) = self.hash_n_degree_quads(
                        canonical_issuer,
                        &related,
                        &issuer_copy,
                        depth.saturating_add(1),
                    )?;
                    // Step 5.4.5.2.
                    path.push_str("_:");
                    let issued = issuer_copy.issue(&related).to_owned();
                    path.push_str(&issued);
                    // Step 5.4.5.3.
                    path.push('<');
                    path.push_str(&result_hash);
                    path.push('>');
                    // Step 5.4.5.4.
                    issuer_copy = result_issuer;
                    // Step 5.4.5.5.
                    if is_worse_than(&path, &chosen_path) {
                        continue 'permutation;
                    }
                }
                // Step 5.4.6.
                if chosen_path.is_empty() || path < chosen_path {
                    chosen_path = path;
                    chosen_issuer = issuer_copy;
                }
            }
            // Step 5.5.
            data_to_hash.push_str(&chosen_path);
            // Step 5.6.
            issuer = chosen_issuer;
        }

        // Step 6.
        Ok((issuer, self.hash_algorithm.hex_digest(&data_to_hash)))
    }

    /// Step 3 of hash-n-degree-quads: for every blank-node component of `quad` other than
    /// `identifier`, compute its related hash and record the component under that hash.
    ///
    /// Repetitions are preserved: the spec records a related blank node "once for every mention
    /// with related hash x".
    fn accumulate_related(
        &self,
        canonical_issuer: &IdentifierIssuer,
        identifier: &str,
        quad: &Quad,
        issuer: &IdentifierIssuer,
        hash_to_related: &mut BTreeMap<String, Vec<String>>,
    ) {
        if let NamedOrBlankNode::BlankNode(component) = &quad.subject {
            self.record_related(
                canonical_issuer,
                component.as_str(),
                identifier,
                quad,
                issuer,
                Position::Subject,
                hash_to_related,
            );
        }
        self.accumulate_related_in_term(
            canonical_issuer,
            &quad.object,
            identifier,
            quad,
            issuer,
            Position::Object,
            hash_to_related,
        );
        if let GraphName::BlankNode(component) = &quad.graph_name {
            self.record_related(
                canonical_issuer,
                component.as_str(),
                identifier,
                quad,
                issuer,
                Position::GraphName,
                hash_to_related,
            );
        }
    }

    /// Handles the object position, descending into RDF 1.2 triple terms when present.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_related_in_term(
        &self,
        canonical_issuer: &IdentifierIssuer,
        term: &Term,
        identifier: &str,
        quad: &Quad,
        issuer: &IdentifierIssuer,
        position: Position,
        hash_to_related: &mut BTreeMap<String, Vec<String>>,
    ) {
        match term {
            Term::BlankNode(component) => self.record_related(
                canonical_issuer,
                component.as_str(),
                identifier,
                quad,
                issuer,
                position,
                hash_to_related,
            ),
            #[cfg(feature = "rdf-12")]
            Term::Triple(triple) => {
                if let NamedOrBlankNode::BlankNode(component) = &triple.subject {
                    self.record_related(
                        canonical_issuer,
                        component.as_str(),
                        identifier,
                        quad,
                        issuer,
                        Position::TripleTermSubject,
                        hash_to_related,
                    );
                }
                self.accumulate_related_in_term(
                    canonical_issuer,
                    &triple.object,
                    identifier,
                    quad,
                    issuer,
                    Position::TripleTermObject,
                    hash_to_related,
                );
            }
            Term::NamedNode(_) | Term::Literal(_) => {}
            // Keeps the match exhaustive when `oxixml-model/rdf-12` is switched on elsewhere in
            // the build (Cargo feature unification) while this crate's `rdf-12` feature is off.
            #[allow(
                unreachable_patterns,
                reason = "only reachable when oxixml-model/rdf-12 is enabled elsewhere in the build"
            )]
            _ => {}
        }
    }

    /// Steps 3.1.1 and 3.1.2 of hash-n-degree-quads for a single component.
    #[allow(clippy::too_many_arguments)]
    fn record_related(
        &self,
        canonical_issuer: &IdentifierIssuer,
        component: &str,
        identifier: &str,
        quad: &Quad,
        issuer: &IdentifierIssuer,
        position: Position,
        hash_to_related: &mut BTreeMap<String, Vec<String>>,
    ) {
        if component != identifier {
            let hash =
                self.hash_related_blank_node(canonical_issuer, component, quad, issuer, position);
            hash_to_related
                .entry(hash)
                .or_default()
                .push(component.to_owned());
        }
    }

    /// Assembles the [`CanonicalizationResult`] from the completed canonical issuer.
    fn build_result(&self, canonical_issuer: &IdentifierIssuer) -> CanonicalizationResult {
        let map = |id: &str| {
            canonical_issuer
                .issued(id)
                .map_or_else(|| id.to_owned(), ToOwned::to_owned)
        };

        let mut canonical_dataset = Dataset::new();
        let mut lines = Vec::with_capacity(self.quads.len());
        for quad in &self.quads {
            let relabeled = relabel_quad(quad, &map);
            lines.push(format!("{relabeled} .\n"));
            canonical_dataset.insert(&relabeled);
        }
        lines.sort_unstable();
        lines.dedup();

        let issued_map: BTreeMap<String, String> = canonical_issuer
            .issued_entries()
            .map(|(original, canonical)| (original.to_owned(), canonical.to_owned()))
            .collect();

        CanonicalizationResult::new(lines.concat(), issued_map, canonical_dataset)
    }
}

/// The position a related blank node occupies within a quad, as passed to
/// [Hash Related Blank Node](https://www.w3.org/TR/rdf-canon/#hash-related-blank-node).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Position {
    /// The subject position, `s`.
    Subject,
    /// The object position, `o`.
    Object,
    /// The graph name position, `g`.
    GraphName,
    /// The subject of an RDF 1.2 triple term in the object position.
    #[cfg(feature = "rdf-12")]
    TripleTermSubject,
    /// The object of an RDF 1.2 triple term in the object position.
    #[cfg(feature = "rdf-12")]
    TripleTermObject,
}

impl Position {
    fn as_str(self) -> &'static str {
        match self {
            Self::Subject => "s",
            Self::Object => "o",
            Self::GraphName => "g",
            #[cfg(feature = "rdf-12")]
            Self::TripleTermSubject => "os",
            #[cfg(feature = "rdf-12")]
            Self::TripleTermObject => "oo",
        }
    }
}

/// Steps 5.4.4.3 and 5.4.5.5: is `path` already known to be no better than `chosen_path`?
///
/// A gossip path is "shorter" when its string is shorter, or equal in length but smaller in code
/// point order. An empty `chosen_path` means no path has been chosen yet.
fn is_worse_than(path: &str, chosen_path: &str) -> bool {
    !chosen_path.is_empty() && path.len() >= chosen_path.len() && path > chosen_path
}

#[cfg(test)]
mod tests {
    use super::is_worse_than;

    #[test]
    fn shortest_path_comparison() {
        assert!(!is_worse_than("anything", ""));
        assert!(!is_worse_than("aa", "ab"));
        assert!(is_worse_than("ab", "aa"));
        // Shorter paths always win, even when lexicographically greater.
        assert!(!is_worse_than("b", "aa"));
        assert!(is_worse_than("bb", "aa"));
    }
}
