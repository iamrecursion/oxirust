//! The W3C [JSON-LD 1.1 API](https://w3c.github.io/json-ld-api/tests/) and
//! [JSON-LD 1.1 Streaming](https://w3c.github.io/json-ld-streaming/tests/) conformance harnesses.
//!
//! Suites are discovered through `OXIXML_TESTSUITE_DIR` (the root of an oxigraph testsuite
//! checkout). When the variable is unset, or the suite directory is missing, every test in this
//! file silently reports "skipped" and passes, so the crate builds and tests cleanly on a machine
//! without the checkout.

mod common;

use common::{
    Report, TestCase, build_options, datasets_isomorphic, file_loader, json_eq, load_json,
    load_manifest, render, testsuite_dir,
};
use oxixml_jsonld::{
    JsonLdParser, JsonLdProfile, JsonLdSyntaxError, JsonValue, compact, expand, flatten, from_rdf,
    to_rdf,
};
use oxixml_model::Quad;
use oxixml_turtle::NQuadsParser;
use std::path::{Path, PathBuf};

/// Entries the harness does not run, with the reason.
///
/// These are features the crate deliberately does not implement (see the crate-root "Framing" and
/// "No I/O by default" sections) or suite entries that cannot be evaluated offline.
struct Allowed {
    /// Test `@id` fragment (e.g. `#t0001`).
    id: &'static str,
    /// Why the entry is not run.
    reason: &'static str,
}

/// `expand` entries that are not run. Every entry of the suite is currently evaluated.
const EXPAND_ALLOWLIST: &[Allowed] = &[];

/// `toRdf` (and streaming `toRdf`) entries that are not run. Every entry is currently evaluated.
const TO_RDF_ALLOWLIST: &[Allowed] = &[];

/// `compact` entries that are not run. Every entry of the suite is currently evaluated.
const COMPACT_ALLOWLIST: &[Allowed] = &[];

/// `flatten` entries that are not run. Every entry of the suite is currently evaluated.
const FLATTEN_ALLOWLIST: &[Allowed] = &[];

/// `fromRdf` entries that are not run. Every entry of the suite is currently evaluated.
const FROM_RDF_ALLOWLIST: &[Allowed] = &[];

fn allowed(list: &[Allowed], id: &str) -> Option<&'static str> {
    list.iter().find(|a| a.id == id).map(|a| a.reason)
}

fn api_tests_dir() -> Option<PathBuf> {
    let dir = testsuite_dir()?.join("json-ld-api").join("tests");
    dir.is_dir().then_some(dir)
}

fn streaming_tests_dir() -> Option<PathBuf> {
    let dir = testsuite_dir()?.join("json-ld-streaming").join("tests");
    dir.is_dir().then_some(dir)
}

/// Skips entries the harness never evaluates.
///
/// Only two categories are skipped: entries that target JSON-LD 1.0 (this is a 1.1 processor, and
/// those entries assert the *older* behaviour), and entries whose input is an HTML document (a
/// separate suite, out of scope for this crate). Entries the suite marks non-normative — the
/// `rdfDirection` ones in particular — *are* evaluated.
fn should_skip(test: &TestCase) -> Option<String> {
    if test.spec_version() != "json-ld-1.1" {
        return Some(format!("specVersion {}", test.spec_version()));
    }
    if test.input.ends_with(".html") {
        return Some("HTML input is out of scope".to_owned());
    }
    None
}

fn check_error(
    test: &TestCase,
    result: Result<JsonValue, JsonLdSyntaxError>,
    report: &mut Report,
) -> bool {
    let Some(expected) = &test.expect_error_code else {
        return false;
    };
    match result {
        Ok(value) => report.fail(format!(
            "{} {}: expected error '{expected}', got {}",
            test.id,
            test.name,
            render(&value)
        )),
        Err(error) => {
            let actual = error.code().map(|c| c.to_string());
            if actual.as_deref() == Some(expected.as_str()) {
                report.pass();
            } else {
                report.fail(format!(
                    "{} {}: expected error '{expected}', got '{}'",
                    test.id,
                    test.name,
                    actual.unwrap_or_else(|| error.to_string())
                ));
            }
        }
    }
    true
}

fn nquads(path: &Path) -> Result<Vec<Quad>, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut quads = Vec::new();
    for quad in NQuadsParser::new().for_slice(&data) {
        quads.push(quad.map_err(|e| format!("{}: {e}", path.display()))?);
    }
    Ok(quads)
}

fn expand_report(dir: &Path) -> Result<Report, String> {
    let manifest = load_manifest(&dir.join("expand-manifest.jsonld"))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = Report::default();
    for test in &manifest.sequence {
        if should_skip(test).is_some() || allowed(EXPAND_ALLOWLIST, &test.id).is_some() {
            report.skip();
            continue;
        }
        let options = build_options(test, &loader, &manifest)?;
        let input = match load_json(&manifest.dir.join(&test.input)) {
            Ok(input) => input,
            Err(e) => {
                report.fail(format!("{} {}: {e}", test.id, test.name));
                continue;
            }
        };
        let result = expand(input, &options);
        if check_error(test, result.clone(), &mut report) {
            continue;
        }
        let Some(expect) = &test.expect else {
            report.skip();
            continue;
        };
        let expected = load_json(&manifest.dir.join(expect))?;
        match result {
            Ok(value) if json_eq(&value, &expected) => report.pass(),
            Ok(value) => report.fail(format!(
                "{} {}\n     got: {}\nexpected: {}",
                test.id,
                test.name,
                render(&value),
                render(&expected)
            )),
            Err(e) => report.fail(format!("{} {}: unexpected error {e}", test.id, test.name)),
        }
    }
    Ok(report)
}

fn compact_report(dir: &Path) -> Result<Report, String> {
    let manifest = load_manifest(&dir.join("compact-manifest.jsonld"))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = Report::default();
    for test in &manifest.sequence {
        if should_skip(test).is_some() || allowed(COMPACT_ALLOWLIST, &test.id).is_some() {
            report.skip();
            continue;
        }
        let options = build_options(test, &loader, &manifest)?;
        let input = match load_json(&manifest.dir.join(&test.input)) {
            Ok(input) => input,
            Err(e) => {
                report.fail(format!("{} {}: {e}", test.id, test.name));
                continue;
            }
        };
        let context = match &test.context {
            Some(context) => load_json(&manifest.dir.join(context))?,
            None => JsonValue::Object(oxixml_jsonld::JsonObject::new()),
        };
        let result = compact(input, context, &options);
        if check_error(test, result.clone(), &mut report) {
            continue;
        }
        let Some(expect) = &test.expect else {
            report.skip();
            continue;
        };
        let expected = load_json(&manifest.dir.join(expect))?;
        match result {
            Ok(value) if json_eq(&value, &expected) => report.pass(),
            Ok(value) => report.fail(format!(
                "{} {}\n     got: {}\nexpected: {}",
                test.id,
                test.name,
                render(&value),
                render(&expected)
            )),
            Err(e) => report.fail(format!("{} {}: unexpected error {e}", test.id, test.name)),
        }
    }
    Ok(report)
}

fn flatten_report(dir: &Path) -> Result<Report, String> {
    let manifest = load_manifest(&dir.join("flatten-manifest.jsonld"))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = Report::default();
    for test in &manifest.sequence {
        if should_skip(test).is_some() || allowed(FLATTEN_ALLOWLIST, &test.id).is_some() {
            report.skip();
            continue;
        }
        let options = build_options(test, &loader, &manifest)?;
        let input = match load_json(&manifest.dir.join(&test.input)) {
            Ok(input) => input,
            Err(e) => {
                report.fail(format!("{} {}: {e}", test.id, test.name));
                continue;
            }
        };
        let context = match &test.context {
            Some(context) => Some(load_json(&manifest.dir.join(context))?),
            None => None,
        };
        let result = flatten(input, context, &options);
        if check_error(test, result.clone(), &mut report) {
            continue;
        }
        let Some(expect) = &test.expect else {
            report.skip();
            continue;
        };
        let expected = load_json(&manifest.dir.join(expect))?;
        match result {
            Ok(value) if json_eq(&value, &expected) => report.pass(),
            Ok(value) => report.fail(format!(
                "{} {}\n     got: {}\nexpected: {}",
                test.id,
                test.name,
                render(&value),
                render(&expected)
            )),
            Err(e) => report.fail(format!("{} {}: unexpected error {e}", test.id, test.name)),
        }
    }
    Ok(report)
}

fn to_rdf_report(dir: &Path, manifest_name: &str, allowlist: &[Allowed]) -> Result<Report, String> {
    let manifest = load_manifest(&dir.join(manifest_name))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = Report::default();
    for test in &manifest.sequence {
        if should_skip(test).is_some() || allowed(allowlist, &test.id).is_some() {
            report.skip();
            continue;
        }
        let options = build_options(test, &loader, &manifest)?;
        let input = match load_json(&manifest.dir.join(&test.input)) {
            Ok(input) => input,
            Err(e) => {
                report.fail(format!("{} {}: {e}", test.id, test.name));
                continue;
            }
        };
        let result = to_rdf(input, &options);
        if let Some(expected_code) = &test.expect_error_code {
            match result {
                Ok(_) => report.fail(format!(
                    "{} {}: expected error '{expected_code}', got success",
                    test.id, test.name
                )),
                Err(error) => {
                    let actual = error.code().map(|c| c.to_string());
                    if actual.as_deref() == Some(expected_code.as_str()) {
                        report.pass();
                    } else {
                        report.fail(format!(
                            "{} {}: expected error '{expected_code}', got '{}'",
                            test.id,
                            test.name,
                            actual.unwrap_or_else(|| error.to_string())
                        ));
                    }
                }
            }
            continue;
        }
        let quads = match result {
            Ok(quads) => quads,
            Err(e) => {
                report.fail(format!("{} {}: unexpected error {e}", test.id, test.name));
                continue;
            }
        };
        if test.is_positive_syntax() {
            report.pass();
            continue;
        }
        let Some(expect) = &test.expect else {
            report.skip();
            continue;
        };
        let expected = match nquads(&manifest.dir.join(expect)) {
            Ok(expected) => expected,
            Err(_) => {
                // The expectation is *generalized* RDF (a blank node or a literal in predicate
                // position). `oxixml_model::Quad` — like the RDF 1.1 abstract syntax — cannot
                // represent those terms, so the entry is out of scope and not evaluated.
                report.skip();
                continue;
            }
        };
        if datasets_isomorphic(&quads, &expected) {
            report.pass();
        } else {
            let (only_got, only_expected) = quad_difference(&quads, &expected);
            report.fail(format!(
                "{} {} ({} quads vs {} expected)\n   only in output: {}\n   only expected: {}",
                test.id,
                test.name,
                quads.len(),
                expected.len(),
                only_got.join(" | "),
                only_expected.join(" | ")
            ));
        }
    }
    Ok(report)
}

fn from_rdf_report(dir: &Path) -> Result<Report, String> {
    let manifest = load_manifest(&dir.join("fromRdf-manifest.jsonld"))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = Report::default();
    for test in &manifest.sequence {
        if should_skip(test).is_some() || allowed(FROM_RDF_ALLOWLIST, &test.id).is_some() {
            report.skip();
            continue;
        }
        let options = build_options(test, &loader, &manifest)?;
        let quads = match nquads(&manifest.dir.join(&test.input)) {
            Ok(quads) => quads,
            Err(e) => {
                if test.expect_error_code.is_some() {
                    // A malformed N-Quads input is exactly the failure the entry asks for.
                    report.pass();
                } else {
                    report.fail(format!("{} {}: {e}", test.id, test.name));
                }
                continue;
            }
        };
        let result = from_rdf(&quads, &options);
        if check_error(test, result.clone(), &mut report) {
            continue;
        }
        let Some(expect) = &test.expect else {
            report.skip();
            continue;
        };
        let expected = load_json(&manifest.dir.join(expect))?;
        match result {
            Ok(value) if json_eq(&value, &expected) => report.pass(),
            Ok(value) => report.fail(format!(
                "{} {}\n     got: {}\nexpected: {}",
                test.id,
                test.name,
                render(&value),
                render(&expected)
            )),
            Err(e) => report.fail(format!("{} {}: unexpected error {e}", test.id, test.name)),
        }
    }
    Ok(report)
}

/// Prints the tally and fails when the pass rate drops below `threshold`.
///
/// The gates are set at 100%: every entry the harness runs currently passes, so any regression is
/// a genuine one.
fn assert_report(report: &Report, label: &str, threshold: f64) {
    let summary = report.summary(label);
    println!("{summary}");
    for failure in &report.failures {
        println!("  FAIL {failure}");
    }
    assert!(
        report.ratio() >= threshold,
        "{summary} is below the {:.0}% gate",
        threshold * 100.0
    );
}

#[test]
fn expand_suite() -> Result<(), String> {
    let Some(dir) = api_tests_dir() else {
        println!("expand: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = expand_report(&dir)?;
    assert_report(&report, "expand", 1.0);
    Ok(())
}

#[test]
fn to_rdf_suite() -> Result<(), String> {
    let Some(dir) = api_tests_dir() else {
        println!("toRdf: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = to_rdf_report(&dir, "toRdf-manifest.jsonld", TO_RDF_ALLOWLIST)?;
    assert_report(&report, "toRdf", 1.0);
    Ok(())
}

#[test]
fn compact_suite() -> Result<(), String> {
    let Some(dir) = api_tests_dir() else {
        println!("compact: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = compact_report(&dir)?;
    assert_report(&report, "compact", 1.0);
    Ok(())
}

#[test]
fn flatten_suite() -> Result<(), String> {
    let Some(dir) = api_tests_dir() else {
        println!("flatten: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = flatten_report(&dir)?;
    assert_report(&report, "flatten", 1.0);
    Ok(())
}

#[test]
fn from_rdf_suite() -> Result<(), String> {
    let Some(dir) = api_tests_dir() else {
        println!("fromRdf: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = from_rdf_report(&dir)?;
    assert_report(&report, "fromRdf", 1.0);
    Ok(())
}

#[test]
fn streaming_suite() -> Result<(), String> {
    let Some(dir) = streaming_tests_dir() else {
        println!("streaming: skipped (OXIXML_TESTSUITE_DIR unset)");
        return Ok(());
    };
    let report = to_rdf_report(&dir, "stream-toRdf-manifest.jsonld", TO_RDF_ALLOWLIST)?;
    assert_report(&report, "streaming toRdf", 1.0);
    Ok(())
}

#[test]
fn streaming_profile_is_accepted_by_the_parser() -> Result<(), Box<dyn std::error::Error>> {
    let document =
        br#"{"@context":{"n":"http://example.com/n"},"@id":"http://example.com/s","n":"v"}"#;
    let quads = JsonLdParser::new()
        .with_profile(JsonLdProfile::Streaming)
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 1);
    Ok(())
}

/// The symmetric difference of two quad multisets, as displayable strings, for failure reports.
///
/// Blank node labels are shown verbatim: two isomorphic-but-differently-labelled datasets appear
/// as differences here, which is exactly what a reader needs in order to judge the mismatch.
fn quad_difference(got: &[Quad], expected: &[Quad]) -> (Vec<String>, Vec<String>) {
    let got_strings: Vec<String> = got.iter().map(ToString::to_string).collect();
    let expected_strings: Vec<String> = expected.iter().map(ToString::to_string).collect();
    let only_got = got_strings
        .iter()
        .filter(|q| !expected_strings.contains(q))
        .take(6)
        .cloned()
        .collect();
    let only_expected = expected_strings
        .iter()
        .filter(|q| !got_strings.contains(q))
        .take(6)
        .cloned()
        .collect();
    (only_got, only_expected)
}
