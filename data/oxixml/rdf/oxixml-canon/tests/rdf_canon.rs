//! W3C [`rdf-canon`](https://w3c.github.io/rdf-canon/tests/) (RDFC-1.0) test suite runner.
//!
//! The suite is discovered through the `OXIXML_TESTSUITE_DIR` environment variable, which must
//! point at a checkout of the oxigraph testsuite root (the directory containing `rdf-canon`).
//! When the variable is unset the test silently succeeds, so the crate stays testable without the
//! external data.
//!
//! ```sh
//! export OXIXML_TESTSUITE_DIR="$HOME/work/refs/oxigraph/testsuite"
//! cargo test -p oxixml-canon --test rdf_canon
//! ```
//!
//! Every `rdfc:RDFC10EvalTest`, `rdfc:RDFC10MapTest` and `rdfc:RDFC10NegativeEvalTest` entry of
//! the manifest is executed:
//!
//! * eval tests compare the canonical N-Quads byte for byte,
//! * map tests compare the issued identifiers map against the reference JSON, and
//! * negative eval tests require canonicalization to abort (the work budget mitigation).

use oxixml_canon::{CanonicalizationOptions, HashAlgorithm, canonicalize_quads_with};
use oxixml_model::{NamedOrBlankNode, Quad, Term};
use oxixml_turtle::{NQuadsParser, TurtleParser};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// The environment variable pointing at the oxigraph testsuite root.
const TESTSUITE_DIR_VAR: &str = "OXIXML_TESTSUITE_DIR";

/// The manifest, relative to the testsuite root.
const MANIFEST_PATH: &str = "rdf-canon/tests/manifest.ttl";

/// A synthetic base IRI: the manifest uses relative IRIs, and stripping this prefix back off
/// turns the resolved IRIs into testsuite-relative paths again.
const BASE_IRI: &str = "http://oxixml.test/rdf-canon/tests/manifest.ttl";

/// The prefix produced by resolving a relative reference against [`BASE_IRI`].
const BASE_DIR_IRI: &str = "http://oxixml.test/rdf-canon/tests/";

const MF: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#";
const RDFC: &str = "https://w3c.github.io/rdf-canon/tests/vocab#";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// What a manifest entry asks the implementation to prove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestKind {
    /// The canonical N-Quads must equal the result file byte for byte.
    Eval,
    /// The issued identifiers map must equal the result JSON.
    Map,
    /// Canonicalization must be rejected (poison dataset).
    NegativeEval,
}

/// One manifest entry.
#[derive(Debug)]
struct TestCase {
    id: String,
    name: String,
    kind: TestKind,
    action: PathBuf,
    result: Option<PathBuf>,
    hash_algorithm: HashAlgorithm,
}

#[test]
fn rdfc10_test_suite() -> Result<(), Box<dyn Error>> {
    let Some(root) = testsuite_root() else {
        eprintln!("SKIP: {TESTSUITE_DIR_VAR} is not set; the W3C rdf-canon suite is unavailable");
        return Ok(());
    };
    let manifest_path = root.join(MANIFEST_PATH);
    if !manifest_path.is_file() {
        eprintln!("SKIP: {} does not exist", manifest_path.display());
        return Ok(());
    }

    let cases = load_manifest(&manifest_path, &root)?;
    assert!(
        !cases.is_empty(),
        "the rdf-canon manifest yielded no test cases"
    );

    let mut passed = 0_usize;
    let mut failures = Vec::new();
    for case in &cases {
        match run_case(case) {
            Ok(()) => passed += 1,
            Err(error) => failures.push(format!("{} ({}): {error}", case.id, case.name)),
        }
    }

    println!(
        "rdf-canon RDFC-1.0: {passed}/{} passed, {} failed",
        cases.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "failing tests:\n  {}",
        failures.join("\n  ")
    );
    Ok(())
}

/// Returns the testsuite root, or `None` when the environment variable is unset or empty.
fn testsuite_root() -> Option<PathBuf> {
    let value = env::var_os(TESTSUITE_DIR_VAR)?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

/// Executes a single manifest entry.
fn run_case(case: &TestCase) -> Result<(), Box<dyn Error>> {
    let quads = load_quads(&case.action)?;
    let options = CanonicalizationOptions::default().with_hash_algorithm(case.hash_algorithm);
    let outcome = canonicalize_quads_with(quads, options);

    match case.kind {
        TestKind::NegativeEval => {
            if let Ok(result) = outcome {
                return Err(format!(
                    "expected canonicalization to be rejected, got {} line(s)",
                    result.canonical_n_quads().lines().count()
                )
                .into());
            }
            Ok(())
        }
        TestKind::Eval => {
            let result = outcome?;
            let expected_path = case
                .result
                .as_ref()
                .ok_or("an eval test must have an mf:result")?;
            let expected = fs::read_to_string(expected_path)?;
            if result.canonical_n_quads() == expected {
                Ok(())
            } else {
                Err(format!(
                    "canonical N-Quads mismatch\n--- expected ---\n{expected}--- actual ---\n{}",
                    result.canonical_n_quads()
                )
                .into())
            }
        }
        TestKind::Map => {
            let result = outcome?;
            let expected_path = case
                .result
                .as_ref()
                .ok_or("a map test must have an mf:result")?;
            let expected = parse_flat_json_object(&fs::read_to_string(expected_path)?)?;
            if *result.issued_map() == expected {
                Ok(())
            } else {
                Err(format!(
                    "issued map mismatch\n--- expected ---\n{expected:?}\n--- actual ---\n{:?}",
                    result.issued_map()
                )
                .into())
            }
        }
    }
}

/// Parses an N-Quads document into quads.
fn load_quads(path: &Path) -> Result<Vec<Quad>, Box<dyn Error>> {
    let data = fs::read(path)?;
    let mut quads = Vec::new();
    for quad in NQuadsParser::new().for_slice(&data) {
        quads.push(quad?);
    }
    Ok(quads)
}

/// Reads the Turtle manifest and turns it into [`TestCase`]s.
fn load_manifest(manifest_path: &Path, root: &Path) -> Result<Vec<TestCase>, Box<dyn Error>> {
    let data = fs::read(manifest_path)?;
    // subject -> predicate -> objects
    let mut statements: BTreeMap<String, BTreeMap<String, Vec<Term>>> = BTreeMap::new();
    for triple in TurtleParser::new()
        .with_base_iri(BASE_IRI)?
        .for_slice(&data)
    {
        let triple = triple?;
        let subject = match &triple.subject {
            NamedOrBlankNode::NamedNode(node) => node.as_str().to_owned(),
            NamedOrBlankNode::BlankNode(node) => format!("_:{}", node.as_str()),
        };
        statements
            .entry(subject)
            .or_default()
            .entry(triple.predicate.as_str().to_owned())
            .or_default()
            .push(triple.object);
    }

    let mut cases = Vec::new();
    for (subject, predicates) in &statements {
        let Some(kind) = predicates
            .get(RDF_TYPE)
            .and_then(|types| types.iter().find_map(test_kind))
        else {
            continue;
        };
        let action = predicates
            .get(&format!("{MF}action"))
            .and_then(|objects| objects.first())
            .and_then(iri_of)
            .ok_or_else(|| format!("{subject} has no mf:action"))?;
        let result = predicates
            .get(&format!("{MF}result"))
            .and_then(|objects| objects.first())
            .and_then(iri_of);
        let name = predicates
            .get(&format!("{MF}name"))
            .and_then(|objects| objects.first())
            .and_then(literal_of)
            .unwrap_or_default();
        let hash_algorithm = match predicates
            .get(&format!("{RDFC}hashAlgorithm"))
            .and_then(|objects| objects.first())
            .and_then(literal_of)
            .as_deref()
        {
            Some("SHA384") => HashAlgorithm::Sha384,
            _ => HashAlgorithm::Sha256,
        };

        cases.push(TestCase {
            id: subject.clone(),
            name,
            kind,
            action: resolve(root, &action)?,
            result: result.map(|iri| resolve(root, &iri)).transpose()?,
            hash_algorithm,
        });
    }
    Ok(cases)
}

/// Maps an `rdf:type` object onto the kind of test it denotes, if any.
fn test_kind(term: &Term) -> Option<TestKind> {
    let iri = iri_of(term)?;
    match iri.strip_prefix(RDFC)? {
        "RDFC10EvalTest" => Some(TestKind::Eval),
        "RDFC10MapTest" => Some(TestKind::Map),
        "RDFC10NegativeEvalTest" => Some(TestKind::NegativeEval),
        _ => None,
    }
}

fn iri_of(term: &Term) -> Option<String> {
    match term {
        Term::NamedNode(node) => Some(node.as_str().to_owned()),
        _ => None,
    }
}

fn literal_of(term: &Term) -> Option<String> {
    match term {
        Term::Literal(literal) => Some(literal.value().to_owned()),
        _ => None,
    }
}

/// Turns a manifest IRI back into a path below the testsuite root.
fn resolve(root: &Path, iri: &str) -> Result<PathBuf, Box<dyn Error>> {
    let relative = iri
        .strip_prefix(BASE_DIR_IRI)
        .ok_or_else(|| format!("unexpected manifest IRI {iri}"))?;
    let mut path = root.join("rdf-canon").join("tests");
    for segment in relative.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(format!("rejected traversal in manifest IRI {iri}").into());
        }
        path.push(segment);
    }
    Ok(path)
}

/// Parses a flat `{"key": "value", ...}` JSON object — the shape of the `*-rdfc10map.json`
/// reference files. Deliberately minimal so that the test needs no JSON dependency.
fn parse_flat_json_object(input: &str) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut map = BTreeMap::new();
    let mut chars = input.chars().peekable();
    skip_whitespace(&mut chars);
    if chars.next() != Some('{') {
        return Err("expected `{` at the start of the JSON object".into());
    }
    loop {
        skip_whitespace(&mut chars);
        match chars.peek() {
            Some('}') => {
                chars.next();
                break;
            }
            Some(',') => {
                chars.next();
                continue;
            }
            Some('"') => {}
            other => return Err(format!("unexpected JSON token {other:?}").into()),
        }
        let key = parse_json_string(&mut chars)?;
        skip_whitespace(&mut chars);
        if chars.next() != Some(':') {
            return Err("expected `:` after a JSON key".into());
        }
        skip_whitespace(&mut chars);
        if chars.peek() != Some(&'"') {
            return Err("expected a JSON string value".into());
        }
        let value = parse_json_string(&mut chars)?;
        map.insert(key, value);
    }
    Ok(map)
}

fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while chars.peek().is_some_and(|c| c.is_ascii_whitespace()) {
        chars.next();
    }
}

fn parse_json_string(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<String, Box<dyn Error>> {
    if chars.next() != Some('"') {
        return Err("expected `\"` at the start of a JSON string".into());
    }
    let mut out = String::new();
    loop {
        let c = chars.next().ok_or("unterminated JSON string")?;
        match c {
            '"' => return Ok(out),
            '\\' => {
                let escape = chars.next().ok_or("unterminated JSON escape")?;
                match escape {
                    '"' | '\\' | '/' => out.push(escape),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{C}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let mut code = String::new();
                        for _ in 0..4 {
                            code.push(chars.next().ok_or("truncated \\u escape")?);
                        }
                        let value = u32::from_str_radix(&code, 16)?;
                        out.push(char::from_u32(value).ok_or("invalid \\u escape")?);
                    }
                    other => {
                        let mut message = String::new();
                        write!(message, "unsupported JSON escape \\{other}")?;
                        return Err(message.into());
                    }
                }
            }
            _ => out.push(c),
        }
    }
}

#[test]
fn flat_json_object_parser() -> Result<(), Box<dyn Error>> {
    let parsed = parse_flat_json_object("{\n  \"e0\": \"c14n0\",\n  \"e1\": \"c14n1\"\n}\n")?;
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed["e0"], "c14n0");
    assert_eq!(parsed["e1"], "c14n1");
    assert!(parse_flat_json_object("{}")?.is_empty());
    assert_eq!(parse_flat_json_object("{\"a\":\"\\u0041\"}")?["a"], "A");
    Ok(())
}
