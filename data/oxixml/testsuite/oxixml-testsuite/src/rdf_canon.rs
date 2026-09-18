//! W3C [`rdf-canon`](https://w3c.github.io/rdf-canon/tests/) (RDFC-1.0) conformance runner.
//!
//! Every `rdfc:RDFC10EvalTest`, `rdfc:RDFC10MapTest` and `rdfc:RDFC10NegativeEvalTest` entry of
//! `rdf-canon/tests/manifest.ttl` is executed against [`oxixml_canon::canonicalize_quads_with`]:
//!
//! * eval tests compare the canonical N-Quads output byte for byte against the expected file;
//! * map tests compare the issued blank-node-identifier map against the expected flat JSON
//!   object (parsed with [`oxixml_jsonld`], which this crate already depends on for the
//!   [`crate::json_ld`] runner, rather than a second hand-rolled JSON reader);
//! * negative eval tests require canonicalization to be rejected (the work-budget mitigation).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_canon::{CanonicalizationOptions, HashAlgorithm, canonicalize_quads_with};
use oxixml_jsonld::JsonValue;
use oxixml_model::Quad;
use oxixml_turtle::NQuadsParser;

use crate::env;
use crate::manifest::{ManifestIndex, strip_base};
use crate::report::SuiteReport;

/// The suite name reported for this runner.
pub const SUITE: &str = "rdf-canon";

/// The manifest, relative to the `OXIXML_TESTSUITE_DIR` root.
const MANIFEST_RELATIVE_PATH: &str = "rdf-canon/tests/manifest.ttl";

/// A synthetic base under which the manifest is parsed; relative IRIs resolve under
/// [`BASE_DIR_IRI`], and stripping that prefix back off recovers a testsuite-relative path.
const BASE_IRI: &str = "http://oxixml.test/rdf-canon/tests/manifest.ttl";
const BASE_DIR_IRI: &str = "http://oxixml.test/rdf-canon/tests/";

const MF: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#";
const RDFC: &str = "https://w3c.github.io/rdf-canon/tests/vocab#";

/// What a manifest entry asks the implementation to prove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestKind {
    Eval,
    Map,
    NegativeEval,
}

fn test_kind(type_iri: &str) -> Option<TestKind> {
    match type_iri.strip_prefix(RDFC)? {
        "RDFC10EvalTest" => Some(TestKind::Eval),
        "RDFC10MapTest" => Some(TestKind::Map),
        "RDFC10NegativeEvalTest" => Some(TestKind::NegativeEval),
        _ => None,
    }
}

/// Runs the RDFC-1.0 test suite.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(testsuite_root) = env::testsuite_dir() else {
        return vec![SuiteReport::unavailable(
            SUITE,
            env::missing(env::TESTSUITE_DIR_VAR),
        )];
    };
    let manifest_path = testsuite_root.join(MANIFEST_RELATIVE_PATH);
    if !manifest_path.is_file() {
        return vec![SuiteReport::unavailable(
            SUITE,
            env::not_found(env::TESTSUITE_DIR_VAR, MANIFEST_RELATIVE_PATH),
        )];
    }

    let mut report = SuiteReport::new(SUITE);
    let Ok(bytes) = fs::read(&manifest_path) else {
        report.record_skip("manifest unreadable");
        return vec![report];
    };
    let Ok(index) = ManifestIndex::parse(&bytes, BASE_IRI) else {
        report.record_skip("manifest did not parse as Turtle");
        return vec![report];
    };
    let tests_dir = testsuite_root.join("rdf-canon").join("tests");

    for subject in index.subjects() {
        let Some(kind) = index.types(subject).find_map(test_kind) else {
            continue;
        };
        let id = subject.trim_start_matches(BASE_DIR_IRI).to_owned();
        let Some(action) = index
            .first_iri(subject, &format!("{MF}action"))
            .and_then(|iri| resolve(&tests_dir, iri))
        else {
            report.record_fail(&id, "mf:action is missing or outside the suite directory");
            continue;
        };
        let result = index
            .first_iri(subject, &format!("{MF}result"))
            .and_then(|iri| resolve(&tests_dir, iri));
        let hash_algorithm = match index.first_literal(subject, &format!("{RDFC}hashAlgorithm")) {
            Some("SHA384") => HashAlgorithm::Sha384,
            _ => HashAlgorithm::Sha256,
        };

        match run_case(kind, &action, result.as_deref(), hash_algorithm) {
            Ok(()) => report.record_pass(),
            Err(message) => report.record_fail(&id, message),
        }
    }
    vec![report]
}

fn run_case(
    kind: TestKind,
    action: &Path,
    result: Option<&Path>,
    hash_algorithm: HashAlgorithm,
) -> Result<(), String> {
    let quads = load_quads(action)?;
    let options = CanonicalizationOptions::default().with_hash_algorithm(hash_algorithm);
    let outcome = canonicalize_quads_with(quads, options);

    match kind {
        TestKind::NegativeEval => match outcome {
            Ok(canonicalized) => Err(format!(
                "expected canonicalization to be rejected, got {} line(s)",
                canonicalized.canonical_n_quads().lines().count()
            )),
            Err(_) => Ok(()),
        },
        TestKind::Eval => {
            let canonicalized = outcome.map_err(|error| error.to_string())?;
            let expected_path = result.ok_or("an eval test must have an mf:result")?;
            let expected = fs::read_to_string(expected_path)
                .map_err(|error| format!("{}: {error}", expected_path.display()))?;
            if canonicalized.canonical_n_quads() == expected {
                Ok(())
            } else {
                Err(format!(
                    "canonical N-Quads mismatch\n--- expected ---\n{expected}--- actual ---\n{}",
                    canonicalized.canonical_n_quads()
                ))
            }
        }
        TestKind::Map => {
            let canonicalized = outcome.map_err(|error| error.to_string())?;
            let expected_path = result.ok_or("a map test must have an mf:result")?;
            let expected_text = fs::read_to_string(expected_path)
                .map_err(|error| format!("{}: {error}", expected_path.display()))?;
            let expected = parse_flat_string_map(&expected_text)?;
            if *canonicalized.issued_map() == expected {
                Ok(())
            } else {
                Err(format!(
                    "issued map mismatch\n--- expected ---\n{expected:?}\n--- actual ---\n{:?}",
                    canonicalized.issued_map()
                ))
            }
        }
    }
}

/// Parses an N-Quads document into quads.
fn load_quads(path: &Path) -> Result<Vec<Quad>, String> {
    let data = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut quads = Vec::new();
    for quad in NQuadsParser::new().for_slice(&data) {
        quads.push(quad.map_err(|error| error.to_string())?);
    }
    Ok(quads)
}

/// Turns a manifest IRI resolved against [`BASE_DIR_IRI`] back into a path below `tests_dir`.
fn resolve(tests_dir: &Path, iri: &str) -> Option<PathBuf> {
    let relative = strip_base(iri, BASE_DIR_IRI)?;
    let mut path = tests_dir.to_path_buf();
    for segment in relative.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return None;
        }
        path.push(segment);
    }
    Some(path)
}

/// Parses the `{"key": "value", ...}` shape of a `*-rdfc10map.json` reference file into a
/// `BTreeMap`, matching the shape [`oxixml_canon::CanonicalizationResult::issued_map`] returns.
fn parse_flat_string_map(text: &str) -> Result<BTreeMap<String, String>, String> {
    let value = JsonValue::from_slice(text.as_bytes()).map_err(|error| error.to_string())?;
    let object = value.as_object().ok_or("expected a JSON object")?;
    let mut map = BTreeMap::new();
    for (key, value) in object.iter() {
        let value = value
            .as_str()
            .ok_or_else(|| format!("expected a string value for `{key}`"))?;
        map.insert(key.to_owned(), value.to_owned());
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_recognizes_the_three_rdfc_types() {
        assert_eq!(
            test_kind(&format!("{RDFC}RDFC10EvalTest")),
            Some(TestKind::Eval)
        );
        assert_eq!(
            test_kind(&format!("{RDFC}RDFC10MapTest")),
            Some(TestKind::Map)
        );
        assert_eq!(
            test_kind(&format!("{RDFC}RDFC10NegativeEvalTest")),
            Some(TestKind::NegativeEval)
        );
        assert_eq!(test_kind("http://example.com/Other"), None);
    }

    #[test]
    fn resolve_rejects_traversal_and_cross_suite_iris() {
        let root = Path::new("/suite/rdf-canon/tests");
        assert_eq!(
            resolve(root, &format!("{BASE_DIR_IRI}foo/bar.nq")),
            Some(root.join("foo").join("bar.nq"))
        );
        assert_eq!(resolve(root, &format!("{BASE_DIR_IRI}../escape.nq")), None);
        assert_eq!(resolve(root, "http://elsewhere/x.nq"), None);
    }

    #[test]
    fn parse_flat_string_map_matches_the_reference_shape() {
        let parsed = parse_flat_string_map("{\n  \"e0\": \"c14n0\",\n  \"e1\": \"c14n1\"\n}\n")
            .expect("valid flat JSON object");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["e0"], "c14n0");
        assert_eq!(parsed["e1"], "c14n1");
        assert!(
            parse_flat_string_map("{}")
                .expect("empty object")
                .is_empty()
        );
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::TESTSUITE_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].is_available());
    }
}
