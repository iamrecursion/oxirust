//! W3C RDF test-suite harness for `oxixml-turtle`.
//!
//! This suite bootstraps on the crate's own [`TurtleParser`] to read each `manifest.ttl`
//! (there is no external RDF library involved), then executes every positive-syntax,
//! negative-syntax, negative-eval and eval test it discovers.
//!
//! Eval tests compare the triples/quads produced by the parser under test against the
//! expected N-Triples/N-Quads output using the blank-node bijection checker implemented in
//! [`isomorphic`]; ground statements must match exactly and blank nodes must admit a
//! consistent one-to-one mapping between the two graphs.
//!
//! The testsuite root is discovered from the `OXIXML_TESTSUITE_DIR` environment variable and
//! the whole harness silently succeeds (skips) when it is unset, so the crate builds and
//! tests cleanly without the data checked out.

use oxixml_model::{GraphName, NamedOrBlankNode, Quad, Term, Triple};
use oxixml_turtle::{NQuadsParser, NTriplesParser, TriGParser, TurtleParser};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const MF_ACTION: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#action";
const MF_RESULT: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#result";
const MF_ASSUMED_BASE: &str =
    "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#assumedTestBase";
const MF_INCLUDE: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#include";
const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
const RDFT: &str = "http://www.w3.org/ns/rdftest#";
/// Absolute base used only so the manifest's relative references resolve; its authority is
/// stripped back off to recover paths relative to the manifest directory.
const DUMMY_BASE: &str = "http://oxixml.invalid/";

/// The RDF concrete syntax exercised by a suite.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    NTriples,
    NQuads,
    Turtle,
    TriG,
}

/// A single test extracted from a manifest.
struct TestCase {
    name: String,
    kind: String,
    action: PathBuf,
    result: Option<PathBuf>,
    base: String,
}

/// Aggregate outcome of running one suite.
struct SuiteReport {
    passed: usize,
    failures: Vec<String>,
}

fn testsuite_root() -> Option<PathBuf> {
    let dir = std::env::var_os("OXIXML_TESTSUITE_DIR")?;
    Some(PathBuf::from(dir).join("rdf-tests").join("rdf"))
}

/// Returns the last `/`-separated segment of an (absolute) IRI.
fn last_segment(iri: &str) -> &str {
    iri.rsplit('/').next().unwrap_or(iri)
}

/// Recovers a manifest-directory-relative path from an IRI resolved against [`DUMMY_BASE`].
///
/// Returns `None` when the IRI resolved outside the dummy authority (e.g. a cross-suite
/// include that normalised above the root).
fn rel_path(iri: &str) -> Option<String> {
    iri.strip_prefix(DUMMY_BASE).map(str::to_owned)
}

/// The index key of a term, or `None` for terms that cannot be list nodes.
fn term_key(term: &Term) -> Option<String> {
    match term {
        Term::NamedNode(node) => Some(node.as_str().to_owned()),
        Term::BlankNode(node) => Some(format!("_:{}", node.as_str())),
        _ => None,
    }
}

/// Walks an RDF collection starting at `head`, returning its members in order.
fn walk_list(index: &HashMap<String, HashMap<String, Vec<Term>>>, head: &Term) -> Vec<Term> {
    let mut members = Vec::new();
    let mut current = head.clone();
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 100_000 {
            break; // defend against a malformed cyclic list
        }
        let key = match &current {
            Term::NamedNode(node) if node.as_str() == RDF_NIL => break,
            other => match term_key(other) {
                Some(k) => k,
                None => break,
            },
        };
        let Some(preds) = index.get(&key) else {
            break;
        };
        if let Some(first) = preds.get(RDF_FIRST).and_then(|v| v.first()) {
            members.push(first.clone());
        }
        match preds.get(RDF_REST).and_then(|v| v.first()) {
            Some(rest) => current = rest.clone(),
            None => break,
        }
    }
    members
}

/// The stable key of a subject: the IRI for a named node, `_:id` for a blank node.
fn subject_key(subject: &NamedOrBlankNode) -> String {
    match subject {
        NamedOrBlankNode::NamedNode(node) => node.as_str().to_owned(),
        NamedOrBlankNode::BlankNode(node) => format!("_:{}", node.as_str()),
    }
}

/// Parses the suite's `manifest.ttl` (following `mf:include`) and returns all discovered tests.
fn parse_manifest(dir: &Path) -> Result<Vec<TestCase>, String> {
    let mut tests = Vec::new();
    let mut visited = HashSet::new();
    collect_from_manifest(&dir.join("manifest.ttl"), None, &mut tests, &mut visited)?;
    tests.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tests)
}

/// Reads one manifest file, appends its tests, and recurses into every `mf:include`d manifest
/// that exists on disk (cross-suite includes that point elsewhere are silently skipped).
fn collect_from_manifest(
    manifest_path: &Path,
    inherited_base: Option<&str>,
    tests: &mut Vec<TestCase>,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    let canonical = manifest_path
        .canonicalize()
        .unwrap_or_else(|_| manifest_path.to_path_buf());
    if !visited.insert(canonical) {
        return Ok(());
    }
    let dir = manifest_path.parent().unwrap_or(Path::new("."));
    let Ok(bytes) = std::fs::read(manifest_path) else {
        return Ok(()); // missing include target — skip
    };

    let parser = TurtleParser::new()
        .with_base_iri(DUMMY_BASE)
        .map_err(|e| format!("invalid dummy base: {e}"))?;
    let mut triples = Vec::new();
    for triple in parser.for_slice(&bytes) {
        triples
            .push(triple.map_err(|e| {
                format!("manifest parse error in {}: {e}", manifest_path.display())
            })?);
    }

    // Index: subject-key -> predicate-IRI -> list of object terms.
    let mut index: HashMap<String, HashMap<String, Vec<Term>>> = HashMap::new();
    let mut assumed_base: Option<String> = None;
    let mut include_heads: Vec<Term> = Vec::new();
    for Triple {
        subject,
        predicate,
        object,
    } in &triples
    {
        if predicate.as_str() == MF_ASSUMED_BASE {
            if let Term::NamedNode(node) = object {
                assumed_base = Some(node.as_str().to_owned());
            }
        }
        if predicate.as_str() == MF_INCLUDE {
            include_heads.push(object.clone());
        }
        index
            .entry(subject_key(subject))
            .or_default()
            .entry(predicate.as_str().to_owned())
            .or_default()
            .push(object.clone());
    }

    // Resolve the `mf:include` RDF collections into manifest IRIs.
    let mut includes: Vec<String> = Vec::new();
    for head in &include_heads {
        for member in walk_list(&index, head) {
            if let Term::NamedNode(node) = member {
                includes.push(node.as_str().to_owned());
            }
        }
    }

    let base_prefix = assumed_base
        .clone()
        .or_else(|| inherited_base.map(str::to_owned))
        .unwrap_or_else(|| DUMMY_BASE.to_owned());

    for (subject, preds) in &index {
        let Some(types) = preds.get(RDF_TYPE) else {
            continue;
        };
        let Some(kind) = types.iter().find_map(|t| match t {
            Term::NamedNode(node) if node.as_str().starts_with(RDFT) => {
                Some(node.as_str()[RDFT.len()..].to_owned())
            }
            _ => None,
        }) else {
            continue;
        };

        let action_iri = preds.get(MF_ACTION).and_then(|v| v.first()).and_then(|t| {
            if let Term::NamedNode(node) = t {
                Some(node.as_str().to_owned())
            } else {
                None
            }
        });
        let Some(action_iri) = action_iri else {
            continue;
        };
        let Some(action_rel) = rel_path(&action_iri) else {
            continue;
        };
        let result = preds
            .get(MF_RESULT)
            .and_then(|v| v.first())
            .and_then(|t| {
                if let Term::NamedNode(node) = t {
                    rel_path(node.as_str())
                } else {
                    None
                }
            })
            .map(|rel| dir.join(rel));

        tests.push(TestCase {
            name: subject.clone(),
            kind,
            action: dir.join(&action_rel),
            result,
            base: format!("{base_prefix}{}", last_segment(&action_iri)),
        });
    }

    for include in includes {
        if let Some(rel) = rel_path(&include) {
            let child = dir.join(rel);
            collect_from_manifest(&child, Some(&base_prefix), tests, visited)?;
        }
    }
    Ok(())
}

fn parse_triples(bytes: &[u8], base: &str, format: Format) -> Result<Vec<Triple>, String> {
    let mut out = Vec::new();
    match format {
        Format::NTriples => {
            for t in NTriplesParser::new().for_slice(bytes) {
                out.push(t.map_err(|e| e.to_string())?);
            }
        }
        Format::Turtle => {
            let parser = TurtleParser::new()
                .with_base_iri(base)
                .map_err(|e| e.to_string())?;
            for t in parser.for_slice(bytes) {
                out.push(t.map_err(|e| e.to_string())?);
            }
        }
        Format::NQuads | Format::TriG => {
            return Err("parse_triples called for a quad format".to_owned());
        }
    }
    Ok(out)
}

fn parse_quads(bytes: &[u8], base: &str, format: Format) -> Result<Vec<Quad>, String> {
    let mut out = Vec::new();
    match format {
        Format::NQuads => {
            for q in NQuadsParser::new().for_slice(bytes) {
                out.push(q.map_err(|e| e.to_string())?);
            }
        }
        Format::TriG => {
            let parser = TriGParser::new()
                .with_base_iri(base)
                .map_err(|e| e.to_string())?;
            for q in parser.for_slice(bytes) {
                out.push(q.map_err(|e| e.to_string())?);
            }
        }
        Format::NTriples | Format::Turtle => {
            return Err("parse_quads called for a triple format".to_owned());
        }
    }
    Ok(out)
}

// --- Blank-node bijection checker ---------------------------------------------------------

/// A term reduced to a form suitable for isomorphism: a fixed ground value, a blank-node id,
/// or (under RDF 1.2) a nested triple term whose components are themselves [`Node`]s.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Node {
    Ground(String),
    Blank(String),
    // Only constructed for RDF 1.2 triple terms; the checker still matches on it either way.
    #[cfg_attr(not(feature = "rdf-12"), allow(dead_code))]
    Triple(Box<[Node; 3]>),
}

fn subject_node(subject: &NamedOrBlankNode) -> Node {
    match subject {
        NamedOrBlankNode::NamedNode(node) => Node::Ground(node.to_string()),
        NamedOrBlankNode::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
    }
}

fn object_node(object: &Term) -> Node {
    match object {
        Term::NamedNode(node) => Node::Ground(node.to_string()),
        Term::Literal(literal) => Node::Ground(literal.to_string()),
        Term::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
        #[cfg(feature = "rdf-12")]
        Term::Triple(inner) => Node::Triple(Box::new([
            subject_node(&inner.subject),
            Node::Ground(inner.predicate.to_string()),
            object_node(&inner.object),
        ])),
    }
}

fn graph_node(graph: &GraphName) -> Node {
    match graph {
        GraphName::DefaultGraph => Node::Ground(String::new()),
        GraphName::NamedNode(node) => Node::Ground(node.to_string()),
        GraphName::BlankNode(node) => Node::Blank(node.as_str().to_owned()),
    }
}

/// A statement reduced to four positional nodes (subject, predicate, object, graph).
type Statement = [Node; 4];

fn triple_statement(triple: &Triple) -> Statement {
    [
        subject_node(&triple.subject),
        Node::Ground(triple.predicate.to_string()),
        object_node(&triple.object),
        Node::Ground(String::new()),
    ]
}

fn quad_statement(quad: &Quad) -> Statement {
    [
        subject_node(&quad.subject),
        Node::Ground(quad.predicate.to_string()),
        object_node(&quad.object),
        graph_node(&quad.graph_name),
    ]
}

fn collect_blanks(node: &Node, set: &mut HashSet<String>) {
    match node {
        Node::Ground(_) => {}
        Node::Blank(id) => {
            set.insert(id.clone());
        }
        Node::Triple(inner) => {
            for child in inner.iter() {
                collect_blanks(child, set);
            }
        }
    }
}

fn blank_ids(statements: &[Statement]) -> Vec<String> {
    let mut set = HashSet::new();
    for stmt in statements {
        for node in stmt {
            collect_blanks(node, &mut set);
        }
    }
    let mut ids: Vec<String> = set.into_iter().collect();
    ids.sort();
    ids
}

/// Whether a node (recursively) contains no blank node.
fn is_ground(node: &Node) -> bool {
    match node {
        Node::Ground(_) => true,
        Node::Blank(_) => false,
        Node::Triple(inner) => inner.iter().all(is_ground),
    }
}

/// Renders a node into a key fragment under `mapping`; `None` if a blank is unmapped.
fn render_node(node: &Node, mapping: &HashMap<String, String>, out: &mut String) -> Option<()> {
    match node {
        Node::Ground(value) => {
            out.push('=');
            out.push_str(value);
        }
        Node::Blank(id) => {
            out.push('~');
            out.push_str(mapping.get(id)?);
        }
        Node::Triple(inner) => {
            out.push('<');
            for child in inner.iter() {
                render_node(child, mapping, out)?;
                out.push('\u{2}');
            }
            out.push('>');
        }
    }
    Some(())
}

/// Renders a statement into its canonical string key under the current blank-node mapping.
///
/// Returns `None` if any blank node in the statement is not yet mapped.
fn render(stmt: &Statement, mapping: &HashMap<String, String>) -> Option<String> {
    let mut key = String::new();
    for node in stmt {
        render_node(node, mapping, &mut key)?;
        key.push('\u{1}');
    }
    Some(key)
}

/// Renders a statement using each blank node's own id (the target-graph reference keys).
fn render_identity(stmt: &Statement) -> String {
    // An identity mapping over all blank ids present in the statement.
    let mut ids = HashSet::new();
    for node in stmt {
        collect_blanks(node, &mut ids);
    }
    let identity: HashMap<String, String> = ids.into_iter().map(|id| (id.clone(), id)).collect();
    render(stmt, &identity).unwrap_or_default()
}

/// Tests whether two sets of statements are isomorphic modulo blank-node renaming.
fn statements_isomorphic(left: &[Statement], right: &[Statement]) -> bool {
    let left_ids = blank_ids(left);
    let right_ids = blank_ids(right);
    if left_ids.len() != right_ids.len() {
        return false;
    }

    // Ground statements (no blank node) must coincide exactly.
    let left_ground: HashSet<&Statement> =
        left.iter().filter(|s| s.iter().all(is_ground)).collect();
    let right_ground: HashSet<&Statement> =
        right.iter().filter(|s| s.iter().all(is_ground)).collect();
    if left_ground != right_ground {
        return false;
    }

    if left_ids.is_empty() {
        return left.len() == right.len();
    }

    let right_keys: HashSet<String> = right.iter().map(render_identity).collect();

    let mut mapping = HashMap::new();
    let mut used = HashSet::new();
    backtrack(
        left,
        &left_ids,
        &right_ids,
        &right_keys,
        &mut mapping,
        &mut used,
    )
}

fn backtrack(
    left: &[Statement],
    left_ids: &[String],
    right_ids: &[String],
    right_keys: &HashSet<String>,
    mapping: &mut HashMap<String, String>,
    used: &mut HashSet<String>,
) -> bool {
    if mapping.len() == left_ids.len() {
        let left_keys: HashSet<String> = left
            .iter()
            .map(|s| render(s, mapping).unwrap_or_default())
            .collect();
        return left_keys == *right_keys && left_keys.len() == right_keys.len();
    }

    // Choose the next unmapped blank id (fixed order; pruning keeps this cheap).
    let Some(next) = left_ids.iter().find(|id| !mapping.contains_key(*id)) else {
        return false;
    };

    for candidate in right_ids {
        if used.contains(candidate) {
            continue;
        }
        mapping.insert(next.clone(), candidate.clone());
        used.insert(candidate.clone());
        if partial_consistent(left, mapping, right_keys)
            && backtrack(left, left_ids, right_ids, right_keys, mapping, used)
        {
            return true;
        }
        used.remove(candidate);
        mapping.remove(next);
    }
    false
}

/// Every fully-determined `left` statement (all blanks mapped) must exist in `right`.
fn partial_consistent(
    left: &[Statement],
    mapping: &HashMap<String, String>,
    right_keys: &HashSet<String>,
) -> bool {
    for stmt in left {
        if let Some(key) = render(stmt, mapping) {
            if !right_keys.contains(&key) {
                return false;
            }
        }
    }
    true
}

fn graphs_isomorphic(left: &[Triple], right: &[Triple]) -> bool {
    let l: Vec<Statement> = left.iter().map(triple_statement).collect();
    let r: Vec<Statement> = right.iter().map(triple_statement).collect();
    statements_isomorphic(&l, &r)
}

fn datasets_isomorphic(left: &[Quad], right: &[Quad]) -> bool {
    let l: Vec<Statement> = left.iter().map(quad_statement).collect();
    let r: Vec<Statement> = right.iter().map(quad_statement).collect();
    statements_isomorphic(&l, &r)
}

// --- Test execution -----------------------------------------------------------------------

fn is_positive_syntax(kind: &str) -> bool {
    // `PositiveC14N` tests carry a canonicalization expectation that belongs to `oxixml-canon`;
    // here we only assert that this crate accepts the (valid) input syntactically.
    kind.ends_with("PositiveSyntax") || kind.ends_with("PositiveC14N")
}

fn is_negative(kind: &str) -> bool {
    kind.ends_with("NegativeSyntax") || kind.ends_with("NegativeEval")
}

fn is_eval(kind: &str) -> bool {
    kind.ends_with("Eval") && !kind.ends_with("NegativeEval")
}

fn run_test(test: &TestCase, format: Format) -> Result<(), String> {
    let action = std::fs::read(&test.action)
        .map_err(|e| format!("cannot read action {}: {e}", test.action.display()))?;

    if is_positive_syntax(&test.kind) {
        return match format {
            Format::NTriples | Format::Turtle => {
                parse_triples(&action, &test.base, format).map(|_| ())
            }
            Format::NQuads | Format::TriG => parse_quads(&action, &test.base, format).map(|_| ()),
        };
    }

    if is_negative(&test.kind) {
        let parsed_ok = match format {
            Format::NTriples | Format::Turtle => parse_triples(&action, &test.base, format).is_ok(),
            Format::NQuads | Format::TriG => parse_quads(&action, &test.base, format).is_ok(),
        };
        return if parsed_ok {
            Err("expected a parse error but parsing succeeded".to_owned())
        } else {
            Ok(())
        };
    }

    if is_eval(&test.kind) {
        let Some(result_path) = &test.result else {
            return Err("eval test without a result file".to_owned());
        };
        let expected = std::fs::read(result_path)
            .map_err(|e| format!("cannot read result {}: {e}", result_path.display()))?;
        return match format {
            Format::Turtle => {
                let got = parse_triples(&action, &test.base, Format::Turtle)?;
                let want = parse_triples(&expected, &test.base, Format::NTriples)?;
                if graphs_isomorphic(&got, &want) {
                    Ok(())
                } else {
                    Err(format!(
                        "graphs differ: {} parsed vs {} expected triples",
                        got.len(),
                        want.len()
                    ))
                }
            }
            Format::TriG => {
                let got = parse_quads(&action, &test.base, Format::TriG)?;
                let want = parse_quads(&expected, &test.base, Format::NQuads)?;
                if datasets_isomorphic(&got, &want) {
                    Ok(())
                } else {
                    Err(format!(
                        "datasets differ: {} parsed vs {} expected quads",
                        got.len(),
                        want.len()
                    ))
                }
            }
            Format::NTriples | Format::NQuads => Err("line format has no eval tests".to_owned()),
        };
    }

    Err(format!("unknown test kind: {}", test.kind))
}

fn run_suite(root: &Path, suite: &str, format: Format) -> Result<SuiteReport, String> {
    let dir = root.join(suite);
    let tests = parse_manifest(&dir)?;
    let mut report = SuiteReport {
        passed: 0,
        failures: Vec::new(),
    };
    for test in &tests {
        match run_test(test, format) {
            Ok(()) => report.passed += 1,
            Err(reason) => report.failures.push(format!(
                "{} [{}]: {reason}",
                last_segment(&test.name),
                test.kind
            )),
        }
    }
    Ok(report)
}

/// Runs an rdf11 suite and asserts a perfect score.
fn assert_rdf11(suite: &str, format: Format) {
    let Some(root) = testsuite_root() else {
        eprintln!("OXIXML_TESTSUITE_DIR unset — skipping {suite}");
        return;
    };
    if !root.join(suite).join("manifest.ttl").exists() {
        eprintln!("manifest missing — skipping {suite}");
        return;
    }
    let report = run_suite(&root, suite, format).expect("suite run failed");
    let total = report.passed + report.failures.len();
    eprintln!("{suite}: {}/{total} passed", report.passed);
    assert!(
        report.failures.is_empty(),
        "{suite}: {} failure(s):\n{}",
        report.failures.len(),
        report.failures.join("\n")
    );
}

#[test]
fn rdf11_ntriples() {
    assert_rdf11("rdf11/rdf-n-triples", Format::NTriples);
}

#[test]
fn rdf11_nquads() {
    assert_rdf11("rdf11/rdf-n-quads", Format::NQuads);
}

#[test]
fn rdf11_turtle() {
    assert_rdf11("rdf11/rdf-turtle", Format::Turtle);
}

#[test]
fn rdf11_trig() {
    assert_rdf11("rdf11/rdf-trig", Format::TriG);
}

/// Runs the rdf12 suites and reports counts without asserting a perfect score.
///
/// RDF 1.2 exercises features (triple terms, base direction) whose grammar level depends on
/// the vendored parser; unsupported cases are expected and merely reported here.
#[test]
fn rdf12_report() {
    let Some(root) = testsuite_root() else {
        eprintln!("OXIXML_TESTSUITE_DIR unset — skipping rdf12 report");
        return;
    };
    let suites = [
        ("rdf12/rdf-n-triples", Format::NTriples),
        ("rdf12/rdf-n-quads", Format::NQuads),
        ("rdf12/rdf-turtle", Format::Turtle),
        ("rdf12/rdf-trig", Format::TriG),
    ];
    for (suite, format) in suites {
        if !root.join(suite).join("manifest.ttl").exists() {
            eprintln!("rdf12 report: {suite} manifest missing — skipped");
            continue;
        }
        match run_suite(&root, suite, format) {
            Ok(report) => {
                let total = report.passed + report.failures.len();
                eprintln!("rdf12 report: {suite}: {}/{total} passed", report.passed);
                for failure in &report.failures {
                    eprintln!("    unsupported/failing: {failure}");
                }
            }
            Err(e) => eprintln!("rdf12 report: {suite}: harness error: {e}"),
        }
    }
}
