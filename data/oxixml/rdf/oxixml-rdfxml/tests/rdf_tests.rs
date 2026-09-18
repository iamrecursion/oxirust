//! Runs the W3C [RDF 1.1 RDF/XML syntax test suite].
//!
//! The suite is not vendored: point `OXIXML_TESTSUITE_DIR` at a checkout of the oxigraph
//! testsuite (the directory that contains `rdf-tests/`) to enable these tests. When the variable
//! is unset the tests silently pass, so a bare `cargo test` stays green.
//!
//! ```sh
//! OXIXML_TESTSUITE_DIR=$HOME/work/refs/oxigraph/testsuite cargo test -p oxixml-rdfxml
//! ```
//!
//! [RDF 1.1 RDF/XML syntax test suite]: https://w3c.github.io/rdf-tests/rdf/rdf11/rdf-xml/

#![expect(clippy::print_stdout, reason = "test harness reports its own counts")]

use oxixml_model::{BlankNode, NamedOrBlankNode, Term, Triple};
use oxixml_rdfxml::RdfXmlParser;
use oxixml_turtle::{NTriplesParser, TurtleParser};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// The IRI the manifest declares as `mf:assumedTestBase`; every action file is parsed as if it
/// had been retrieved from there.
const ASSUMED_TEST_BASE: &str = "https://w3c.github.io/rdf-tests/rdf/rdf11/rdf-xml/";

/// The path of the manifest inside the testsuite checkout.
const MANIFEST_RELATIVE_PATH: &str = "rdf-tests/rdf/rdf11/rdf-xml/manifest.ttl";

/// `rdf:type`.
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
/// `mf:action`.
const MF_ACTION: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#action";
/// `mf:result`.
const MF_RESULT: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#result";
/// `mf:name`.
const MF_NAME: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#name";
/// The test type whose action must parse into the graph its result file describes.
const RDFT_TEST_XML_EVAL: &str = "http://www.w3.org/ns/rdftest#TestXMLEval";
/// The test type whose action must be rejected.
const RDFT_TEST_XML_NEGATIVE_SYNTAX: &str = "http://www.w3.org/ns/rdftest#TestXMLNegativeSyntax";

/// One entry of the manifest.
struct TestCase {
    /// The human-readable `mf:name`.
    name: String,
    /// Whether the action must parse (`TestXMLEval`) or must fail (`TestXMLNegativeSyntax`).
    positive: bool,
    /// The IRI of the RDF/XML document, which doubles as its base IRI.
    action_iri: String,
    /// The IRI of the expected N-Triples document, for positive tests.
    result_iri: Option<String>,
}

#[test]
fn rdf11_rdf_xml_suite() -> Result<(), Box<dyn Error>> {
    let Some(suite_dir) = testsuite_dir() else {
        println!("OXIXML_TESTSUITE_DIR is not set, skipping the W3C RDF/XML test suite");
        return Ok(());
    };
    let manifest_path = suite_dir.join(MANIFEST_RELATIVE_PATH);
    if !manifest_path.is_file() {
        println!("{} not found, skipping", manifest_path.display());
        return Ok(());
    }
    let base_dir = manifest_path
        .parent()
        .ok_or("the manifest should have a parent directory")?
        .to_path_buf();

    let cases = load_manifest(&manifest_path)?;
    assert!(!cases.is_empty(), "the manifest should declare test cases");

    let mut passed = 0_usize;
    let mut failures = Vec::new();
    for case in &cases {
        match run_case(case, &base_dir) {
            Ok(()) => passed += 1,
            Err(error) => failures.push(format!("{}: {error}", case.name)),
        }
    }

    println!(
        "RDF 1.1 RDF/XML suite: {passed}/{} passed ({} failed)",
        cases.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} test(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Reads `OXIXML_TESTSUITE_DIR`, returning `None` when it is unset or empty.
fn testsuite_dir() -> Option<PathBuf> {
    let value = std::env::var_os("OXIXML_TESTSUITE_DIR")?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

/// Parses the Turtle manifest into the list of test cases it declares.
fn load_manifest(manifest_path: &Path) -> Result<Vec<TestCase>, Box<dyn Error>> {
    let manifest = fs::read_to_string(manifest_path)?;
    let mut types: HashMap<String, String> = HashMap::new();
    let mut actions: HashMap<String, String> = HashMap::new();
    let mut results: HashMap<String, String> = HashMap::new();
    let mut names: HashMap<String, String> = HashMap::new();

    for triple in TurtleParser::new()
        .with_base_iri(ASSUMED_TEST_BASE)?
        .for_slice(&manifest)
    {
        let triple = triple?;
        let NamedOrBlankNode::NamedNode(subject) = &triple.subject else {
            continue;
        };
        let subject = subject.as_str().to_owned();
        match triple.predicate.as_str() {
            RDF_TYPE => {
                if let Term::NamedNode(object) = &triple.object {
                    types.insert(subject, object.as_str().to_owned());
                }
            }
            MF_ACTION => {
                if let Term::NamedNode(object) = &triple.object {
                    actions.insert(subject, object.as_str().to_owned());
                }
            }
            MF_RESULT => {
                if let Term::NamedNode(object) = &triple.object {
                    results.insert(subject, object.as_str().to_owned());
                }
            }
            MF_NAME => {
                if let Term::Literal(object) = &triple.object {
                    names.insert(subject, object.value().to_owned());
                }
            }
            _ => (),
        }
    }

    let mut cases = Vec::new();
    for (subject, test_type) in types {
        let positive = match test_type.as_str() {
            RDFT_TEST_XML_EVAL => true,
            RDFT_TEST_XML_NEGATIVE_SYNTAX => false,
            _ => continue,
        };
        let Some(action_iri) = actions.get(&subject) else {
            continue;
        };
        cases.push(TestCase {
            name: names
                .get(&subject)
                .cloned()
                .unwrap_or_else(|| subject.clone()),
            positive,
            action_iri: action_iri.clone(),
            result_iri: results.get(&subject).cloned(),
        });
    }
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(cases)
}

/// Runs one manifest entry, returning a human-readable message on failure.
fn run_case(case: &TestCase, base_dir: &Path) -> Result<(), String> {
    let action_path = iri_to_path(&case.action_iri, base_dir)
        .ok_or_else(|| format!("cannot map {} to a local file", case.action_iri))?;
    let action = fs::read(&action_path).map_err(|e| format!("{}: {e}", action_path.display()))?;

    let parsed = RdfXmlParser::new()
        .with_base_iri(&case.action_iri)
        .map_err(|e| format!("invalid base IRI {}: {e}", case.action_iri))?
        .for_slice(&action)
        .collect::<Result<Vec<Triple>, _>>();

    if !case.positive {
        return match parsed {
            Ok(triples) => Err(format!(
                "expected a syntax error but parsed {} triple(s)",
                triples.len()
            )),
            Err(_) => Ok(()),
        };
    }

    let actual = parsed.map_err(|e| format!("unexpected parse error: {e}"))?;
    let Some(result_iri) = &case.result_iri else {
        return Ok(()); // Nothing to compare against
    };
    let result_path = iri_to_path(result_iri, base_dir)
        .ok_or_else(|| format!("cannot map {result_iri} to a local file"))?;
    let expected_text =
        fs::read(&result_path).map_err(|e| format!("{}: {e}", result_path.display()))?;
    let expected = NTriplesParser::new()
        .for_slice(&expected_text)
        .collect::<Result<Vec<Triple>, _>>()
        .map_err(|e| format!("cannot parse {}: {e}", result_path.display()))?;

    if are_isomorphic(&actual, &expected) {
        Ok(())
    } else {
        Err(format!(
            "graphs differ.\n  actual ({}):\n{}\n  expected ({}):\n{}",
            actual.len(),
            format_graph(&actual),
            expected.len(),
            format_graph(&expected)
        ))
    }
}

/// Maps a test IRI back to the local file it was loaded from.
fn iri_to_path(iri: &str, base_dir: &Path) -> Option<PathBuf> {
    let relative = iri.strip_prefix(ASSUMED_TEST_BASE)?;
    Some(base_dir.join(relative))
}

/// Renders a graph as indented N-Triples, for failure messages.
fn format_graph(graph: &[Triple]) -> String {
    let mut lines = graph
        .iter()
        .map(|t| format!("    {t} ."))
        .collect::<Vec<_>>();
    lines.sort();
    lines.join("\n")
}

/// Returns `true` if the two graphs are equal up to a bijective renaming of blank nodes.
fn are_isomorphic(left: &[Triple], right: &[Triple]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    // Triples with the fewest blank nodes constrain the mapping most, so try them first.
    let mut ordered = left.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|t| blank_node_count(t));
    let mut used = vec![false; right.len()];
    let mut mapping = HashMap::new();
    matches_from(&ordered, right, &mut used, &mut mapping, 0)
}

/// Backtracking search over the pairings of `left[index..]` with the unused triples of `right`.
fn matches_from(
    left: &[&Triple],
    right: &[Triple],
    used: &mut Vec<bool>,
    mapping: &mut HashMap<BlankNode, BlankNode>,
    index: usize,
) -> bool {
    let Some(candidate) = left.get(index) else {
        return true;
    };
    for (position, target) in right.iter().enumerate() {
        if used[position] {
            continue;
        }
        let saved = mapping.clone();
        if unify_triple(candidate, target, mapping) {
            used[position] = true;
            if matches_from(left, right, used, mapping, index + 1) {
                return true;
            }
            used[position] = false;
        }
        *mapping = saved;
    }
    false
}

/// Extends `mapping` so that `left` and `right` become equal, or returns `false`.
fn unify_triple(
    left: &Triple,
    right: &Triple,
    mapping: &mut HashMap<BlankNode, BlankNode>,
) -> bool {
    if left.predicate != right.predicate {
        return false;
    }
    unify_subject(&left.subject, &right.subject, mapping)
        && unify_object(&left.object, &right.object, mapping)
}

/// Unifies two subjects.
fn unify_subject(
    left: &NamedOrBlankNode,
    right: &NamedOrBlankNode,
    mapping: &mut HashMap<BlankNode, BlankNode>,
) -> bool {
    match (left, right) {
        (NamedOrBlankNode::NamedNode(left), NamedOrBlankNode::NamedNode(right)) => left == right,
        (NamedOrBlankNode::BlankNode(left), NamedOrBlankNode::BlankNode(right)) => {
            unify_blank_node(left, right, mapping)
        }
        _ => false,
    }
}

/// Unifies two objects.
fn unify_object(left: &Term, right: &Term, mapping: &mut HashMap<BlankNode, BlankNode>) -> bool {
    match (left, right) {
        (Term::NamedNode(left), Term::NamedNode(right)) => left == right,
        (Term::Literal(left), Term::Literal(right)) => left == right,
        (Term::BlankNode(left), Term::BlankNode(right)) => unify_blank_node(left, right, mapping),
        _ => false,
    }
}

/// Records or checks one blank-node correspondence, keeping the mapping injective.
fn unify_blank_node(
    left: &BlankNode,
    right: &BlankNode,
    mapping: &mut HashMap<BlankNode, BlankNode>,
) -> bool {
    if let Some(bound) = mapping.get(left) {
        return bound == right;
    }
    if mapping.values().any(|bound| bound == right) {
        return false; // Another blank node already claims this one
    }
    mapping.insert(left.clone(), right.clone());
    true
}

/// Counts the blank-node positions of a triple.
fn blank_node_count(triple: &Triple) -> usize {
    usize::from(matches!(triple.subject, NamedOrBlankNode::BlankNode(_)))
        + usize::from(matches!(triple.object, Term::BlankNode(_)))
}

#[test]
fn isomorphism_helper_is_sound() -> Result<(), Box<dyn Error>> {
    let left = NTriplesParser::new()
        .for_slice(
            "_:a <http://example.com/p> _:b .\n_:b <http://example.com/p> <http://example.com/o> .",
        )
        .collect::<Result<Vec<_>, _>>()?;
    let right = NTriplesParser::new()
        .for_slice(
            "_:x <http://example.com/p> _:y .\n_:y <http://example.com/p> <http://example.com/o> .",
        )
        .collect::<Result<Vec<_>, _>>()?;
    let different = NTriplesParser::new()
        .for_slice(
            "_:x <http://example.com/p> _:y .\n_:x <http://example.com/p> <http://example.com/o> .",
        )
        .collect::<Result<Vec<_>, _>>()?;
    assert!(are_isomorphic(&left, &right));
    assert!(!are_isomorphic(&left, &different));
    Ok(())
}
