//! End-to-end round-trip tests: `write -> read -> same triple set`, over deterministic,
//! seeded corpora (including shared subject/objects, blank nodes, and Unicode literals).

use std::collections::HashSet;

use oxixml_hdt::{Hdt, HdtWriter};
use oxixml_model::vocab::xsd;
use oxixml_model::{BlankNode, Literal, NamedNode, NamedOrBlankNode, Term, Triple};
use oxixml_support::rng::{Pcg32, Rng};

/// Unicode literal values, deliberately spanning several scripts and an emoji, to exercise
/// front-coding and UTF-8 handling in the dictionary.
const UNICODE_WORDS: &[&str] = &[
    "日本語のテキスト",
    "café naïve",
    "Ελληνικά γράμματα",
    "Русский текст",
    "emoji party: 🎉🚀✨",
    "Zürich",
    "\u{0}\u{1}control-ish",
    "",
];

/// Builds a deterministic, seeded corpus of `num_triples` distinct triples, drawing subjects,
/// predicates, and objects from a small shared pool (so a good number of IRIs necessarily
/// appear as both subject and object -- exercising the dictionary's "shared" section) plus a
/// mix of blank nodes and Unicode literals (simple, language-tagged, and typed).
fn seeded_corpus(seed: u64, num_triples: usize) -> Vec<Triple> {
    let mut rng = Pcg32::new(seed);

    let entities: Vec<NamedNode> = (0..40)
        .map(|i| NamedNode::new(format!("http://example.org/entity/{i}")).expect("valid iri"))
        .collect();
    let predicates: Vec<NamedNode> = (0..8)
        .map(|i| NamedNode::new(format!("http://example.org/prop/{i}")).expect("valid iri"))
        .collect();
    let blanks: Vec<BlankNode> = (0..10)
        .map(|i| BlankNode::new(format!("b{i}")).expect("valid blank node id"))
        .collect();

    let pick = |rng: &mut Pcg32, len: usize| -> usize { rng.gen_range(0..len as u64) as usize };

    let mut seen = HashSet::new();
    let mut triples = Vec::with_capacity(num_triples);
    let mut attempts = 0;
    while triples.len() < num_triples && attempts < num_triples * 100 {
        attempts += 1;

        let subject: NamedOrBlankNode = if pick(&mut rng, 10) < 8 {
            let index = pick(&mut rng, entities.len());
            entities[index].clone().into()
        } else {
            let index = pick(&mut rng, blanks.len());
            blanks[index].clone().into()
        };

        let predicate_index = pick(&mut rng, predicates.len());
        let predicate = predicates[predicate_index].clone();

        let object: Term = match pick(&mut rng, 10) {
            0..=4 => {
                // Re-uses the very same entity pool as subjects, guaranteeing overlap.
                let index = pick(&mut rng, entities.len());
                entities[index].clone().into()
            }
            5..=6 => {
                let index = pick(&mut rng, blanks.len());
                blanks[index].clone().into()
            }
            7 => {
                let index = pick(&mut rng, UNICODE_WORDS.len());
                Literal::new_simple_literal(UNICODE_WORDS[index]).into()
            }
            8 => {
                let index = pick(&mut rng, UNICODE_WORDS.len());
                Literal::new_language_tagged_literal(UNICODE_WORDS[index], "ja")
                    .expect("valid language tag")
                    .into()
            }
            _ => {
                let value = rng.gen_range(0..1_000_000);
                Literal::new_typed_literal(value.to_string(), xsd::INTEGER).into()
            }
        };

        let triple = Triple::new(subject, predicate, object);
        if seen.insert(triple.to_string()) {
            triples.push(triple);
        }
    }
    triples
}

/// Compares two triple sets for equality, ignoring order (HDT is a set of triples, not a
/// sequence).
fn assert_same_triple_set(expected: &[Triple], actual: &[Triple]) {
    let expected: HashSet<String> = expected.iter().map(Triple::to_string).collect();
    let actual: HashSet<String> = actual.iter().map(Triple::to_string).collect();
    assert_eq!(expected, actual);
}

fn roundtrip_corpus(seed: u64, num_triples: usize) {
    let triples = seeded_corpus(seed, num_triples);
    assert_eq!(
        triples.len(),
        num_triples,
        "corpus generation should reach the target size"
    );

    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, triples.clone())
        .expect("writing should succeed");

    let hdt = Hdt::from_bytes(&bytes).expect("reading should succeed");
    assert_eq!(hdt.len(), num_triples as u64);

    let roundtripped: Vec<Triple> = hdt
        .triples()
        .collect::<Result<_, _>>()
        .expect("every resolved triple should be well-formed");
    assert_same_triple_set(&triples, &roundtripped);
}

#[test]
fn roundtrip_small_corpus() {
    roundtrip_corpus(0xC0FFEE, 25);
}

#[test]
fn roundtrip_medium_corpus() {
    roundtrip_corpus(0x5EED_1234, 500);
}

#[test]
fn roundtrip_is_stable_across_multiple_seeds() {
    for seed in [1u64, 2, 42, 1_000_003, u64::MAX / 7] {
        roundtrip_corpus(seed, 60);
    }
}

#[test]
fn roundtrip_single_triple() {
    roundtrip_corpus(7, 1);
}

#[test]
fn roundtrip_empty_dataset() {
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, Vec::<Triple>::new())
        .expect("writing an empty dataset should succeed");
    let hdt = Hdt::from_bytes(&bytes).expect("reading should succeed");
    assert!(hdt.is_empty());
    assert_eq!(hdt.triples().count(), 0);
}

#[test]
fn pattern_search_matches_full_scan_baseline() {
    let triples = seeded_corpus(0xABCD, 300);
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, triples.clone())
        .expect("write");
    let hdt = Hdt::from_bytes(&bytes).expect("read");

    // For every subject present, `triples_matching(Some(s), None, None)` must equal the
    // triples with that subject found by a naive linear scan of the original corpus.
    let mut subjects: Vec<NamedOrBlankNode> = triples.iter().map(|t| t.subject.clone()).collect();
    subjects.sort_by_key(ToString::to_string);
    subjects.dedup_by_key(|s| s.to_string());

    for subject in subjects.iter().take(15) {
        let expected: Vec<Triple> = triples
            .iter()
            .filter(|t| &t.subject == subject)
            .cloned()
            .collect();
        let actual: Vec<Triple> = hdt
            .triples_matching(Some(subject.as_ref()), None, None)
            .collect::<Result<_, _>>()
            .expect("resolves cleanly");
        assert_same_triple_set(&expected, &actual);
    }
}

/// A naive, wholly independent oracle for a triple pattern: linear-scan `triples` itself,
/// keeping each bound field's equality check separate from anything [`Hdt`] does internally.
fn naive_pattern_scan(
    triples: &[Triple],
    subject: Option<&NamedOrBlankNode>,
    predicate: Option<&NamedNode>,
    object: Option<&Term>,
) -> Vec<Triple> {
    triples
        .iter()
        .filter(|t| {
            subject.is_none_or(|expected| &t.subject == expected)
                && predicate.is_none_or(|expected| &t.predicate == expected)
                && object.is_none_or(|expected| &t.object == expected)
        })
        .cloned()
        .collect()
}

/// Every one of the eight triple-pattern bound/unbound shapes -- including the `?P?`, `??O`, and
/// `?PO` shapes that [`Hdt`] now resolves via a lazily-built PSO/OSP secondary index rather than
/// a full scan -- must return exactly the same result set as [`naive_pattern_scan`], a plain
/// linear scan over the original (pre-HDT) corpus kept as an oracle right here in the test.
///
/// The corpus is large enough, and dense enough in shared entities (see [`seeded_corpus`]), that
/// an index probe and a full scan would disagree on at least one of these patterns if the
/// PSO/OSP routing were wrong -- this is a correctness check, not a timing one.
#[test]
fn pattern_search_all_eight_shapes_match_full_scan_baseline() {
    let triples = seeded_corpus(0xFACE_B00C, 400);
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, triples.clone())
        .expect("write");
    let hdt = Hdt::from_bytes(&bytes).expect("read");

    let assert_pattern = |subject: Option<&NamedOrBlankNode>,
                          predicate: Option<&NamedNode>,
                          object: Option<&Term>| {
        let expected = naive_pattern_scan(&triples, subject, predicate, object);
        let actual: Vec<Triple> = hdt
            .triples_matching(
                subject.map(NamedOrBlankNode::as_ref),
                predicate.map(NamedNode::as_ref),
                object.map(Term::as_ref),
            )
            .collect::<Result<_, _>>()
            .expect("resolves cleanly");
        assert_same_triple_set(&expected, &actual);
    };

    let mut subjects: Vec<NamedOrBlankNode> = triples.iter().map(|t| t.subject.clone()).collect();
    subjects.sort_by_key(ToString::to_string);
    subjects.dedup_by_key(|s| s.to_string());
    let mut predicates: Vec<NamedNode> = triples.iter().map(|t| t.predicate.clone()).collect();
    predicates.sort_by_key(ToString::to_string);
    predicates.dedup_by_key(|p| p.to_string());
    let mut objects: Vec<Term> = triples.iter().map(|t| t.object.clone()).collect();
    objects.sort_by_key(ToString::to_string);
    objects.dedup_by_key(|o| o.to_string());

    // (_, _, _): nothing bound, still the full-scan path.
    assert_pattern(None, None, None);

    // (S, _, _), (S, P, _), (S, _, O): unchanged SPO subject-bound path.
    for subject in subjects.iter().take(6) {
        assert_pattern(Some(subject), None, None);
        for predicate in predicates.iter().take(3) {
            assert_pattern(Some(subject), Some(predicate), None);
        }
        for object in objects.iter().take(3) {
            assert_pattern(Some(subject), None, Some(object));
        }
    }

    // (_, P, _), (_, P, O): newly index-routed via PSO.
    for predicate in predicates.iter().take(6) {
        assert_pattern(None, Some(predicate), None);
        for object in objects.iter().take(3) {
            assert_pattern(None, Some(predicate), Some(object));
        }
    }

    // (_, _, O): newly index-routed via OSP.
    for object in objects.iter().take(6) {
        assert_pattern(None, None, Some(object));
    }

    // (S, P, O): fully bound, still the SPO subject-bound path.
    for subject in subjects.iter().take(3) {
        for predicate in predicates.iter().take(2) {
            for object in objects.iter().take(2) {
                assert_pattern(Some(subject), Some(predicate), Some(object));
            }
        }
    }

    // Bound-but-absent ids for every newly-routed shape must yield an empty result, not a
    // panic or a stray match from an unrelated group.
    let unknown_predicate =
        NamedNode::new("http://example.org/prop/does-not-exist").expect("valid iri");
    let unknown_object: Term = NamedNode::new("http://example.org/entity/does-not-exist")
        .expect("valid iri")
        .into();
    assert_pattern(None, Some(&unknown_predicate), None);
    assert_pattern(None, None, Some(&unknown_object));
    assert_pattern(None, Some(&unknown_predicate), Some(&unknown_object));
}

#[test]
fn round_trips_a_header_payload() {
    let header = b"<http://example.org/> <http://purl.org/dc/terms/format> \"hdt\" .\n";
    let mut bytes = Vec::new();
    HdtWriter::with_header(&header[..])
        .write(&mut bytes, seeded_corpus(9, 5))
        .expect("write");
    let hdt = Hdt::from_bytes(&bytes).expect("read");
    assert_eq!(hdt.header(), header);
}
