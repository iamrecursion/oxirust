//! Regression test for the W3C `json-ld-api` **toRdf** conformance entry `#te036` ("Expanding
//! `@index`"), which is shared byte-for-byte (modulo the input's key order) with the
//! `json-ld-streaming` **toRdf** suite's own `#te036`.
//!
//! `toRdf/e036-in.jsonld` puts a native JSON `true` (and `false`) under two different keys of an
//! `@index`-container map (`"A"` and `"C"`). Both keys' arrays independently contain a bare
//! `true`/`false`, and the [Deserialize JSON-LD to RDF
//! algorithm](https://www.w3.org/TR/json-ld-api/#deserialize-json-ld-to-rdf-algorithm) must
//! produce one quad *per source value encountered*, not one quad per distinct RDF term: the
//! W3C-authored `toRdf/e036-out.nq` expectation literally repeats the
//! `<...> <http://example.com/container> "true"^^xsd:boolean .` line (and the `"false"` one)
//! twice, for a 25-quad total. A converter that deduplicates its output --- e.g. by collecting
//! into a `HashSet<Quad>` before returning --- silently drops exactly those two repeats,
//! producing 23 quads instead of 25.
//!
//! This test bypasses the suite harness's [`common::datasets_isomorphic`], which deliberately
//! deduplicates *both* sides before comparing (a reasonable general-purpose fuzzy comparator, but
//! one that would mask this exact defect: with both sides deduplicated to 23 quads, the harness
//! sees content matching content, however the *production* output silently lost two real quads
//! to get there). Instead it sorts both sides' N-Quads text and asserts the full 25-line
//! multisets are equal, exactly the way the W3C reference test runners compare `toRdf` output.
//!
//! Skipped (not failed) when `OXIXML_TESTSUITE_DIR` is unset or the `json-ld-api` checkout under
//! it is missing, so the crate builds and tests cleanly without the external suite checkout.

mod common;

use common::{build_options, file_loader, load_json, load_manifest, testsuite_dir};
use oxixml_jsonld::to_rdf;
use oxixml_turtle::NQuadsParser;
use std::collections::BTreeMap;
use std::path::Path;

/// Loads `#te036` from a `toRdf`-shaped manifest directory (shared by `json-ld-api/tests` and
/// `json-ld-streaming/tests`), runs it through [`to_rdf`], and asserts the produced quads are
/// exactly the expected 25-quad multiset (order-independent, duplicates significant).
fn run_te036(tests_dir: &Path, manifest_name: &str, suite_label: &str) -> Result<(), String> {
    let manifest = load_manifest(&tests_dir.join(manifest_name))?;
    let test = manifest
        .sequence
        .iter()
        .find(|t| t.id == "#te036")
        .ok_or_else(|| format!("{suite_label}: #te036 not found in {manifest_name}"))?;
    let loader = file_loader(&manifest.base_iri, tests_dir);
    let options = build_options(test, &loader, &manifest)?;
    let input = load_json(&manifest.dir.join(&test.input))?;

    let quads = to_rdf(input, &options).map_err(|e| format!("{suite_label}: toRdf failed: {e}"))?;

    let expect = test
        .expect
        .as_ref()
        .ok_or_else(|| format!("{suite_label}: #te036 declares no `expect`"))?;
    let expected_data =
        std::fs::read(manifest.dir.join(expect)).map_err(|e| format!("{suite_label}: {e}"))?;
    let expected: Vec<_> = NQuadsParser::new()
        .for_slice(&expected_data)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{suite_label}: {e}"))?;

    // Sanity check on the fixture itself: if this ever drifts, the assertion below would give a
    // confusing signal, so fail loudly and specifically instead.
    assert_eq!(
        expected.len(),
        25,
        "{suite_label}: fixture drifted, e036-out.nq no longer has 25 quads"
    );

    let mut got: Vec<String> = quads.iter().map(ToString::to_string).collect();
    let mut want: Vec<String> = expected.iter().map(ToString::to_string).collect();
    got.sort();
    want.sort();

    if got == want {
        return Ok(());
    }

    // Multiset diff (want-count vs got-count per distinct line) so a failure names exactly which
    // JSON node(s) lost a quad, rather than just reporting two differently-ordered Vecs.
    let mut got_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for g in &got {
        *got_counts.entry(g.as_str()).or_insert(0) += 1;
    }
    let mut want_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for w in &want {
        *want_counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut lines = Vec::new();
    for (line, want_n) in &want_counts {
        let got_n = got_counts.get(line).copied().unwrap_or(0);
        if got_n != *want_n {
            lines.push(format!("  want={want_n} got={got_n}  {line}"));
        }
    }
    for (line, got_n) in &got_counts {
        if !want_counts.contains_key(line) {
            lines.push(format!("  want=0 got={got_n}  {line} (unexpected)"));
        }
    }
    Err(format!(
        "{suite_label}: toRdf(#te036) produced {} quads, expected {} ({} distinct mismatched lines):\n{}",
        got.len(),
        want.len(),
        lines.len(),
        lines.join("\n")
    ))
}

#[test]
fn te036_index_container_produces_all_25_quads_batch() -> Result<(), String> {
    let Some(dir) = testsuite_dir() else {
        println!("te036 (batch toRdf): skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let tests_dir = dir.join("json-ld-api").join("tests");
    if !tests_dir.is_dir() {
        println!("te036 (batch toRdf): skipped (json-ld-api checkout missing)");
        return Ok(());
    }
    run_te036(&tests_dir, "toRdf-manifest.jsonld", "json-ld-api toRdf")
}

#[test]
fn te036_index_container_produces_all_25_quads_streaming() -> Result<(), String> {
    let Some(dir) = testsuite_dir() else {
        println!("te036 (streaming toRdf): skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let tests_dir = dir.join("json-ld-streaming").join("tests");
    if !tests_dir.is_dir() {
        println!("te036 (streaming toRdf): skipped (json-ld-streaming checkout missing)");
        return Ok(());
    }
    run_te036(
        &tests_dir,
        "stream-toRdf-manifest.jsonld",
        "json-ld-streaming toRdf",
    )
}
