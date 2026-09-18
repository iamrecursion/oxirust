//! Self-contained behavioural tests for [RDFC-1.0](https://www.w3.org/TR/rdf-canon/)
//! canonicalization: known answers, isomorphism invariance, hash-algorithm selection and the
//! poison-dataset work budget.
//!
//! These tests need no external data; the full W3C suite lives in `tests/rdf_canon.rs`.

use oxixml_canon::{
    CanonicalizationError, CanonicalizationOptions, HashAlgorithm, WorkBudget, canonicalize,
    canonicalize_quads, canonicalize_quads_with, canonicalize_with,
};
use oxixml_model::{Dataset, Quad};
use oxixml_turtle::NQuadsParser;
use std::collections::BTreeMap;
use std::error::Error;

/// The "blank node - diamond" case (`test020` of the W3C suite): two blank nodes that share a
/// first degree hash and are separated only by hash-n-degree-quads.
const DIAMOND: &str = concat!(
    "<http://example.org/vocab#test> <http://example.org/vocab#A> _:e0 .\n",
    "<http://example.org/vocab#test> <http://example.org/vocab#B> _:e1 .\n",
    "_:e0 <http://example.org/vocab#next> _:e2 .\n",
    "_:e1 <http://example.org/vocab#next> _:e2 .\n",
);

/// The spec's "double circle" example: two blank nodes doubly connected in opposite directions.
/// They share a first degree hash, so hash-n-degree-quads must run to separate them.
const DOUBLE_CIRCLE: &str = concat!(
    "_:e0 <http://example.org/vocab#next> _:e1 .\n",
    "_:e0 <http://example.org/vocab#prev> _:e1 .\n",
    "_:e1 <http://example.org/vocab#next> _:e0 .\n",
    "_:e1 <http://example.org/vocab#prev> _:e0 .\n",
);

fn parse(document: &str) -> Result<Vec<Quad>, Box<dyn Error>> {
    let mut quads = Vec::new();
    for quad in NQuadsParser::new().for_slice(document) {
        quads.push(quad?);
    }
    Ok(quads)
}

fn dataset(document: &str) -> Result<Dataset, Box<dyn Error>> {
    let mut dataset = Dataset::new();
    for quad in parse(document)? {
        dataset.insert(&quad);
    }
    Ok(dataset)
}

fn issued(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(original, canonical)| ((*original).to_owned(), (*canonical).to_owned()))
        .collect()
}

#[test]
fn canonicalizes_a_single_blank_node() -> Result<(), Box<dyn Error>> {
    let input = "_:e0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> \
                 <http://example.org/vocab#Foo> .\n";
    let result = canonicalize(&dataset(input)?)?;
    assert_eq!(
        result.canonical_n_quads(),
        "_:c14n0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> \
         <http://example.org/vocab#Foo> .\n"
    );
    assert_eq!(*result.issued_map(), issued(&[("e0", "c14n0")]));
    Ok(())
}

#[test]
fn canonicalizes_the_diamond_with_sha256() -> Result<(), Box<dyn Error>> {
    let result = canonicalize(&dataset(DIAMOND)?)?;
    assert_eq!(
        result.canonical_n_quads(),
        concat!(
            "<http://example.org/vocab#test> <http://example.org/vocab#A> _:c14n2 .\n",
            "<http://example.org/vocab#test> <http://example.org/vocab#B> _:c14n0 .\n",
            "_:c14n0 <http://example.org/vocab#next> _:c14n1 .\n",
            "_:c14n2 <http://example.org/vocab#next> _:c14n1 .\n",
        )
    );
    assert_eq!(
        *result.issued_map(),
        issued(&[("e0", "c14n2"), ("e1", "c14n0"), ("e2", "c14n1")])
    );
    Ok(())
}

#[test]
fn canonicalizes_the_diamond_with_sha384() -> Result<(), Box<dyn Error>> {
    let options = CanonicalizationOptions::default().with_hash_algorithm(HashAlgorithm::Sha384);
    let result = canonicalize_with(&dataset(DIAMOND)?, options)?;
    assert_eq!(
        result.canonical_n_quads(),
        concat!(
            "<http://example.org/vocab#test> <http://example.org/vocab#A> _:c14n0 .\n",
            "<http://example.org/vocab#test> <http://example.org/vocab#B> _:c14n2 .\n",
            "_:c14n0 <http://example.org/vocab#next> _:c14n1 .\n",
            "_:c14n2 <http://example.org/vocab#next> _:c14n1 .\n",
        )
    );
    assert_eq!(
        *result.issued_map(),
        issued(&[("e0", "c14n0"), ("e1", "c14n2"), ("e2", "c14n1")])
    );
    Ok(())
}

#[test]
fn canonical_dataset_matches_the_canonical_n_quads() -> Result<(), Box<dyn Error>> {
    let result = canonicalize(&dataset(DIAMOND)?)?;
    let mut lines: Vec<String> = result
        .canonical_dataset()
        .iter()
        .map(|quad| format!("{quad} .\n"))
        .collect();
    lines.sort();
    assert_eq!(lines.concat(), result.canonical_n_quads());
    Ok(())
}

#[test]
fn is_invariant_under_blank_node_relabeling() -> Result<(), Box<dyn Error>> {
    let relabeled = DIAMOND
        .replace("_:e0", "_:zzz")
        .replace("_:e1", "_:aaa")
        .replace("_:e2", "_:mmm");
    let original = canonicalize(&dataset(DIAMOND)?)?;
    let renamed = canonicalize(&dataset(&relabeled)?)?;
    assert_eq!(original.canonical_n_quads(), renamed.canonical_n_quads());
    assert_eq!(
        *renamed.issued_map(),
        issued(&[("aaa", "c14n0"), ("mmm", "c14n1"), ("zzz", "c14n2")])
    );
    Ok(())
}

#[test]
fn is_invariant_under_quad_reordering() -> Result<(), Box<dyn Error>> {
    let forwards = canonicalize_quads(parse(DIAMOND)?)?;
    let mut reversed = parse(DIAMOND)?;
    reversed.reverse();
    let backwards = canonicalize_quads(reversed)?;
    assert_eq!(forwards.canonical_n_quads(), backwards.canonical_n_quads());
    assert_eq!(forwards.issued_map(), backwards.issued_map());
    Ok(())
}

#[test]
fn streaming_input_is_deduplicated() -> Result<(), Box<dyn Error>> {
    let once = parse(DIAMOND)?;
    let twice: Vec<Quad> = once.iter().cloned().chain(once.iter().cloned()).collect();
    let result = canonicalize_quads(twice)?;
    assert_eq!(result.canonical_n_quads().lines().count(), 4);
    assert_eq!(
        result.canonical_n_quads(),
        canonicalize_quads(once)?.canonical_n_quads()
    );
    Ok(())
}

#[test]
fn canonicalizes_blank_node_graph_names() -> Result<(), Box<dyn Error>> {
    let input = concat!(
        "_:e0 <http://example.com/p1> _:e1 .\n",
        "_:e1 <http://example.com/p2> \"Foo\" .\n",
        "_:e1 <http://example.com/p3> _:g0 .\n",
        "_:e0 <http://example.com/p1> _:e1 _:g0 .\n",
        "_:e1 <http://example.com/p2> \"Bar\" _:g0 .\n",
    );
    let result = canonicalize(&dataset(input)?)?;
    assert_eq!(result.issued_map().len(), 3);
    assert!(result.canonical_n_quads().contains(" _:c14n"));
    // The relabeling is a bijection onto `c14n0..c14n2`.
    let mut canonical: Vec<&str> = result.issued_map().values().map(String::as_str).collect();
    canonical.sort_unstable();
    assert_eq!(canonical, ["c14n0", "c14n1", "c14n2"]);
    Ok(())
}

#[test]
fn empty_datasets_canonicalize_to_the_empty_document() -> Result<(), Box<dyn Error>> {
    let result = canonicalize(&Dataset::new())?;
    assert_eq!(result.canonical_n_quads(), "");
    assert!(result.issued_map().is_empty());
    assert!(result.canonical_dataset().is_empty());
    let (n_quads, map, dataset) = result.into_parts();
    assert_eq!(n_quads, "");
    assert!(map.is_empty());
    assert!(dataset.is_empty());
    Ok(())
}

#[test]
fn ground_datasets_need_no_identifiers() -> Result<(), Box<dyn Error>> {
    let input = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n";
    let result = canonicalize(&dataset(input)?)?;
    assert!(result.issued_map().is_empty());
    assert_eq!(result.into_canonical_n_quads(), input);
    Ok(())
}

#[test]
fn a_poison_clique_exhausts_the_default_budget() -> Result<(), Box<dyn Error>> {
    // A fully connected clique of ten blank nodes: the W3C suite's `test074` negative test.
    let mut document = String::new();
    for i in 0..10 {
        for j in 0..10 {
            if i != j {
                document.push_str(&format!("_:e{i} <http://example.com/p> _:e{j} .\n"));
            }
        }
    }
    let error = canonicalize_quads(parse(&document)?)
        .err()
        .ok_or("the poison clique was expected to be rejected")?;
    assert!(matches!(
        error,
        CanonicalizationError::WorkBudgetExceeded { .. }
    ));
    assert!(error.to_string().contains("work budget"));
    Ok(())
}

// Builds an N-Quads document chaining `n` blank nodes together through the same predicate
// (`_:e0` related to `_:e1`, `_:e1` to `_:e2`, ...). Unlike the poison clique above, no
// gossip-path permutation here ever grows past a handful of choices — every interior node has
// exactly one predecessor and one successor — so this shape triggers no combinatorial blowup at
// all. It does, however, drive `hash_n_degree_quads`'s own native recursion to a depth
// proportional to `n`, since every interior node is structurally indistinguishable from every
// other and so can only be disambiguated by that recursion.
fn chain_document(n: usize) -> String {
    let mut document = String::new();
    for i in 0..n {
        document.push_str(&format!("_:e{i} <http://example.com/p> _:e{} .\n", i + 1));
    }
    document
}

#[test]
fn a_short_chain_stays_within_the_default_budget() -> Result<(), Box<dyn Error>> {
    // A chain shallower than the engine's safe-recursion-depth ceiling must canonicalize
    // normally: this guards against a charging scheme that penalizes depth starting from the
    // very first call, which would wrongly reject long *legitimate* chains (RDF lists of
    // indistinguishable items produce exactly this shape) even though they pose no stack-safety
    // risk whatsoever.
    canonicalize_quads(parse(&chain_document(200))?)?;
    Ok(())
}

#[test]
fn a_long_chain_is_rejected_not_crashed() -> Result<(), Box<dyn Error>> {
    // Before the engine charged anything based on recursion depth, canonicalizing a long enough
    // chain overflowed the native stack and aborted the process well before the flat, depth-blind
    // work charge ever caught up with the default budget — this is the same defect class as
    // `a_poison_clique_exhausts_the_default_budget` (an unbounded-compute DoS reachable from a
    // public API on adversarial-but-syntactically-valid input) via a different graph shape that
    // does not trip the permutation charge at all. 2 000 nodes is far more than needed to exceed
    // the *shipped* default budget (rejection happens at a fixed, small native recursion depth
    // regardless of chain length or budget size — see `SAFE_RECURSION_DEPTH` in `engine.rs`) but
    // stays fast to build and parse.
    let error = canonicalize_quads(parse(&chain_document(2_000))?)
        .err()
        .ok_or("a 2 000-node indistinguishable chain was expected to be rejected")?;
    assert!(matches!(
        error,
        CanonicalizationError::WorkBudgetExceeded { .. }
    ));
    Ok(())
}

#[test]
fn the_double_circle_needs_hash_n_degree_quads() -> Result<(), Box<dyn Error>> {
    // A one-unit budget covers the single outer hash-n-degree-quads call but not the gossip-path
    // permutation that follows it.
    let options = CanonicalizationOptions::default().with_work_budget(WorkBudget::new(1));
    let error = canonicalize_quads_with(parse(DOUBLE_CIRCLE)?, options)
        .err()
        .ok_or("a one-unit budget was expected to be exhausted")?;
    assert_eq!(
        error,
        CanonicalizationError::WorkBudgetExceeded { budget: 1 }
    );

    // A budget that is merely generous is enough for the real thing.
    let options = CanonicalizationOptions::default().with_work_budget(WorkBudget::new(1_000));
    let result = canonicalize_quads_with(parse(DOUBLE_CIRCLE)?, options)?;
    assert_eq!(
        *result.issued_map(),
        issued(&[("e0", "c14n0"), ("e1", "c14n1")])
    );
    Ok(())
}

#[test]
fn a_diamond_is_resolved_by_first_degree_hashes_alone() -> Result<(), Box<dyn Error>> {
    // Every blank node of the diamond has a distinct first degree hash, so not a single work
    // unit is spent: step 5 of the algorithm is never entered.
    let options = CanonicalizationOptions::default().with_work_budget(WorkBudget::new(0));
    let result = canonicalize_quads_with(parse(DIAMOND)?, options)?;
    assert_eq!(result.issued_map().len(), 3);
    Ok(())
}

#[test]
fn an_unlimited_budget_still_canonicalizes() -> Result<(), Box<dyn Error>> {
    let options = CanonicalizationOptions::default().with_work_budget(WorkBudget::unlimited());
    assert_eq!(options.work_budget().max_work_units(), u64::MAX);
    let result = canonicalize_quads_with(parse(DOUBLE_CIRCLE)?, options)?;
    assert_eq!(result.issued_map().len(), 2);
    Ok(())
}

#[test]
fn literals_keep_their_canonical_n_quads_form() -> Result<(), Box<dyn Error>> {
    let input = concat!(
        "_:e0 <http://example.com/p> \"plain\" .\n",
        "_:e0 <http://example.com/p> \"typed\"^^<http://www.w3.org/2001/XMLSchema#string> .\n",
        "_:e0 <http://example.com/p> \"tagged\"@en .\n",
        "_:e0 <http://example.com/p> \"quote \\\" tab \\t\" .\n",
    );
    let result = canonicalize(&dataset(input)?)?;
    let canonical = result.canonical_n_quads();
    // `xsd:string` is dropped and the two lexically identical literals collapse to one quad.
    assert!(canonical.contains("\"typed\" ."));
    assert!(canonical.contains("\"tagged\"@en ."));
    assert!(canonical.contains("\"quote \\\" tab \\t\" ."));
    assert_eq!(canonical.lines().count(), 4);
    Ok(())
}

/// RDF 1.2 ([RDF-star](https://www.w3.org/TR/rdf12-concepts/)) coverage: blank nodes nested inside
/// triple terms participate in canonicalization just like top-level ones.
#[cfg(feature = "rdf-12")]
mod rdf_12 {
    use super::{canonicalize_quads, issued};
    use oxixml_model::{BlankNode, GraphName, NamedNode, Quad, Term, Triple};
    use std::error::Error;

    fn triple_term_quads(
        outer: &'static str,
        inner: &'static str,
    ) -> Result<Vec<Quad>, Box<dyn Error>> {
        let p = NamedNode::new("http://example.com/p")?;
        let q = NamedNode::new("http://example.com/q")?;
        let outer = BlankNode::new(outer)?;
        let inner = BlankNode::new(inner)?;
        Ok(vec![
            Quad::new(
                outer.clone(),
                q,
                Term::Triple(Box::new(Triple::new(inner.clone(), p.clone(), inner))),
                GraphName::DefaultGraph,
            ),
            Quad::new(
                outer,
                p,
                NamedNode::new("http://example.com/o")?,
                GraphName::DefaultGraph,
            ),
        ])
    }

    #[test]
    fn canonicalizes_blank_nodes_inside_triple_terms() -> Result<(), Box<dyn Error>> {
        let result = canonicalize_quads(triple_term_quads("e0", "e1")?)?;
        assert_eq!(
            *result.issued_map(),
            issued(&[("e0", "c14n1"), ("e1", "c14n0")])
        );
        assert!(result.canonical_n_quads().contains("_:c14n0"));
        Ok(())
    }

    #[test]
    fn triple_terms_are_invariant_under_relabeling() -> Result<(), Box<dyn Error>> {
        let original = canonicalize_quads(triple_term_quads("e0", "e1")?)?;
        let renamed = canonicalize_quads(triple_term_quads("zzz", "aaa")?)?;
        assert_eq!(original.canonical_n_quads(), renamed.canonical_n_quads());
        Ok(())
    }
}
