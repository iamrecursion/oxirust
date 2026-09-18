//! Byte-stability regression test for a small, fully deterministic HDT fixture.
//!
//! `tests/data/golden.hdt` was generated once by this crate's own [`HdtWriter`] from the
//! fixed triple set below (see the `#[ignore]`d `regenerate_golden_fixture` test) and checked
//! into the repository. [`golden_file_matches_current_writer_output`] re-serializes the same
//! triples and asserts the output is **byte-for-byte identical** to the checked-in file,
//! guarding against accidental wire-format drift (field reordering, a changed default block
//! size, a different property serialization, ...) in future refactors.
//!
//! To intentionally update the fixture after a deliberate, documented format change, run:
//! `cargo test -p oxixml-hdt --test golden -- --ignored regenerate_golden_fixture`.

use oxixml_hdt::{Hdt, HdtWriter};
use oxixml_model::vocab::xsd;
use oxixml_model::{BlankNode, Literal, NamedNode, Triple};

const GOLDEN_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/golden.hdt");

/// The fixed triple set backing the golden fixture. Never change this function's output
/// without deliberately regenerating `tests/data/golden.hdt` in the same commit.
fn golden_triples() -> Vec<Triple> {
    let ex = |s: &str| NamedNode::new(format!("http://example.org/{s}")).expect("valid iri");
    vec![
        Triple::new(ex("alice"), ex("knows"), ex("bob")),
        Triple::new(ex("bob"), ex("knows"), ex("alice")),
        Triple::new(ex("alice"), ex("knows"), ex("carol")),
        Triple::new(
            ex("alice"),
            ex("name"),
            Literal::new_language_tagged_literal("Alice", "en").expect("valid language tag"),
        ),
        Triple::new(
            ex("alice"),
            ex("age"),
            Literal::new_typed_literal("30", xsd::INTEGER),
        ),
        Triple::new(
            BlankNode::new("anon1").expect("valid blank node id"),
            ex("knows"),
            ex("alice"),
        ),
        Triple::new(
            ex("bob"),
            ex("nickname"),
            Literal::new_simple_literal("日本語のニックネーム"),
        ),
        Triple::new(ex("carol"), ex("knows"), ex("bob")),
    ]
}

#[test]
fn golden_file_matches_current_writer_output() {
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, golden_triples())
        .expect("writing the golden fixture should succeed");

    let expected = std::fs::read(GOLDEN_PATH).expect(
        "tests/data/golden.hdt should be checked into the repository; \
         regenerate it with `cargo test -p oxixml-hdt --test golden -- --ignored regenerate_golden_fixture`",
    );
    assert_eq!(
        bytes, expected,
        "the HDT wire format produced by HdtWriter has drifted from the checked-in golden fixture"
    );
    assert!(bytes.len() < 100_000, "golden fixture must stay small");

    // The fixture must also still be a valid, correctly round-tripping HDT file.
    let hdt = Hdt::from_bytes(&expected).expect("golden fixture must parse");
    assert_eq!(hdt.len(), golden_triples().len() as u64);
}

/// Regenerates `tests/data/golden.hdt`. Not run by default -- this test's only purpose is to
/// be invoked explicitly (with `--ignored`) when a deliberate wire-format change requires
/// refreshing the fixture.
#[test]
#[ignore = "only run explicitly to intentionally regenerate the golden fixture"]
fn regenerate_golden_fixture() {
    let mut bytes = Vec::new();
    HdtWriter::new()
        .write(&mut bytes, golden_triples())
        .expect("writing the golden fixture should succeed");
    std::fs::write(GOLDEN_PATH, &bytes).expect("writing tests/data/golden.hdt should succeed");
}
