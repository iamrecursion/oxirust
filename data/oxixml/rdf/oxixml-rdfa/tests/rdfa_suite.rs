//! The [W3C RDFa Test Suite](https://rdfa.info/test-suite/), run against this crate.
//!
//! # Running it
//!
//! The suite is not vendored. Clone it and point an environment variable at the checkout:
//!
//! ```text
//! git clone https://github.com/rdfa/rdfa-website
//! OXIXML_RDFA_SUITE_DIR=$PWD/rdfa-website cargo test -p oxixml-rdfa
//! ```
//!
//! Without `OXIXML_RDFA_SUITE_DIR` — or with it pointing somewhere without a `manifest.ttl` — the
//! harness skips silently and reports zero cases, so the crate's tests stay green on a machine
//! that does not have the suite.
//!
//! # How a case is run
//!
//! The suite does not ship the documents it tests. `tests/NNNN.txt` is a *fragment*, and the
//! suite's own Ruby harness (`lib/crazyivan/core.rb`, `get_test_content`) wraps it into a document
//! per host language: it lifts the leading lines into attributes on the root element, turns a
//! leading JSON object into a `@prefix` attribute, rewrites `NNNN.xhtml` references to the
//! extension of the host language, and substitutes `$TCPATH`. [`build_document`] is a port of that
//! function, and [`build_query`] of the same treatment applied to `tests/NNNN.sparql`.
//!
//! A case then means: build the document, extract a graph with the base IRI the suite would have
//! served it from, and answer the case's SPARQL `ASK` against that graph with the evaluator in
//! `tests/support/sparql_ask.rs`. `test:expectedResults false` inverts the expectation.
//!
//! Only the `rdfa1.1` cases of the `html5`, `xhtml1`, `xhtml5` and `xml` host languages are run —
//! the ones this crate claims. The `rdfa1.0`, `html4`, `svg` and `*-invalid` rows of the manifest
//! are counted as skips.

#[path = "support/sparql_ask.rs"]
mod sparql_ask;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_model::{NamedOrBlankNode, Term, Triple};
use oxixml_rdfa::{HostLanguage, RdfaParser};
use oxixml_sparql_syntax::SparqlParser;
use oxixml_turtle::TurtleParser;

/// The environment variable that points at a checkout of <https://github.com/rdfa/rdfa-website>.
const SUITE_DIR: &str = "OXIXML_RDFA_SUITE_DIR";

/// The RDFa version this crate implements.
const VERSION: &str = "rdfa1.1";

/// The host languages the harness runs, and the [`HostLanguage`] each maps to.
const SUITES: [(&str, HostLanguage); 4] = [
    ("html5", HostLanguage::Html5),
    ("xhtml1", HostLanguage::Xhtml1),
    ("xhtml5", HostLanguage::Xhtml5),
    ("xml", HostLanguage::Xml),
];

/// The pass rate each host language must reach for the harness to succeed.
///
/// These are floors just under the rates measured when the harness was written, so a regression
/// fails loudly while an unrelated upstream edit to the suite does not.
const THRESHOLDS: [(&str, f64); 4] = [
    ("html5", 1.0),
    ("xhtml1", 1.0),
    ("xhtml5", 1.0),
    ("xml", 1.0),
];

/// One row of `manifest.ttl`.
#[derive(Debug, Default)]
struct TestCase {
    number: String,
    title: String,
    versions: Vec<String>,
    host_languages: Vec<String>,
    expected: bool,
}

#[test]
fn rdfa_test_suite() {
    let Some(root) = suite_root() else {
        eprintln!("{SUITE_DIR} is not set or has no manifest.ttl; skipping the RDFa test suite");
        return;
    };
    let cases = load_manifest(&root).expect("the manifest parses");
    assert!(!cases.is_empty(), "the manifest has no test cases");

    let mut failures: Vec<String> = Vec::new();
    let mut report = String::new();
    let mut all_ok = true;

    for (suite, host) in SUITES {
        let mut passed = 0_usize;
        let mut failed = 0_usize;
        for case in &cases {
            if !case.versions.iter().any(|value| value == VERSION)
                || !case.host_languages.iter().any(|value| value == suite)
            {
                continue;
            }
            match run_case(&root, case, suite, host) {
                Ok(()) => passed += 1,
                Err(reason) => {
                    failed += 1;
                    failures.push(format!("{suite}/{} — {reason}", case.number));
                }
            }
        }
        let total = passed + failed;
        let rate = if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                passed as f64 / total as f64
            }
        };
        let _ = writeln!(
            report,
            "{suite:<8} {passed:>4} / {total:<4} passed ({:.1}%)",
            rate * 100.0
        );
        let threshold = THRESHOLDS
            .iter()
            .find(|(name, _)| *name == suite)
            .map_or(0.0, |(_, value)| *value);
        if rate < threshold {
            all_ok = false;
            let _ = writeln!(report, "  ↳ below the {:.0}% floor", threshold * 100.0);
        }
    }

    let skipped = cases
        .iter()
        .filter(|case| !case.versions.iter().any(|value| value == VERSION))
        .count();
    let _ = writeln!(report, "{skipped} cases skipped as not {VERSION}");

    eprintln!("\nW3C RDFa Test Suite\n{report}");
    if !failures.is_empty() {
        eprintln!("failing cases:");
        for failure in &failures {
            eprintln!("  {failure}");
        }
    }
    assert!(all_ok, "the RDFa test suite fell below a pass-rate floor");
}

fn suite_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os(SUITE_DIR)?);
    root.join("manifest.ttl").is_file().then_some(root)
}

fn run_case(root: &Path, case: &TestCase, suite: &str, host: HostLanguage) -> Result<(), String> {
    let document = build_document(root, &case.number, suite)
        .map_err(|error| format!("{}: {error}", case.title))?;
    let query_text = build_query(root, &case.number, suite)
        .map_err(|error| format!("{}: {error}", case.title))?;
    let base = document_url(&case.number, suite);

    let parser = RdfaParser::new()
        .with_host_language(host)
        .with_base_iri(&base)
        .map_err(|error| error.to_string())?;
    let graph: Vec<Triple> = parser
        .for_slice(&document)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{}: extraction failed: {error}", case.title))?;

    let query = SparqlParser::new()
        .parse_query(&query_text)
        .map_err(|error| format!("{}: the ASK does not parse: {error}", case.title))?;
    let answer = sparql_ask::ask(&query, &graph)
        .ok_or_else(|| format!("{}: the ASK uses an unsupported construct", case.title))?;

    if answer == case.expected {
        return Ok(());
    }
    Err(format!(
        "{}: expected the ASK to be {}, got {answer}\n{}",
        case.title,
        case.expected,
        dump(&graph)
    ))
}

fn dump(graph: &[Triple]) -> String {
    let mut out = String::from("    extracted graph:\n");
    for triple in graph {
        let _ = writeln!(out, "      {triple} .");
    }
    if graph.is_empty() {
        out.push_str("      (empty)\n");
    }
    out
}

// ------------------------------------------------------------------- manifest

fn load_manifest(root: &Path) -> Result<Vec<TestCase>, String> {
    let text = fs::read_to_string(root.join("manifest.ttl")).map_err(|e| e.to_string())?;
    let mut by_subject: BTreeMap<String, TestCase> = BTreeMap::new();
    for triple in TurtleParser::new().for_slice(text.as_bytes()) {
        let triple = triple.map_err(|error| error.to_string())?;
        let NamedOrBlankNode::NamedNode(subject) = &triple.subject else {
            continue;
        };
        let Some(number) = subject.as_str().rsplit('/').next() else {
            continue;
        };
        if number.len() != 4 || !number.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let entry = by_subject
            .entry(subject.as_str().to_owned())
            .or_insert_with(|| TestCase {
                number: number.to_owned(),
                expected: true,
                ..TestCase::default()
            });
        let Term::Literal(value) = &triple.object else {
            continue;
        };
        match triple.predicate.as_str() {
            "http://rdfa.info/vocabs/rdfa-test#rdfaVersion" => {
                entry.versions.push(value.value().to_owned());
            }
            "http://rdfa.info/vocabs/rdfa-test#hostLanguage" => {
                entry.host_languages.push(value.value().to_owned());
            }
            "http://purl.org/dc/terms/title" => entry.title = value.value().to_owned(),
            "http://www.w3.org/2006/03/test-description#expectedResults" => {
                entry.expected = value.value() == "true";
            }
            _ => {}
        }
    }
    Ok(by_subject.into_values().collect())
}

// ------------------------------------------------------------ document building

/// The file extension the suite serves a host language's documents under.
fn suffix(suite: &str) -> &'static str {
    if suite.contains("xhtml") {
        "xhtml"
    } else if suite.contains("html") {
        "html"
    } else if suite.contains("svg") {
        "svg"
    } else {
        "xml"
    }
}

fn tcpath(suite: &str) -> String {
    format!("http://rdfa.info/test-suite/test-cases/{VERSION}/{suite}")
}

fn document_url(number: &str, suite: &str) -> String {
    format!("{}/{number}.{}", tcpath(suite), suffix(suite))
}

/// A port of `CrazyIvan::Core.get_test_content` for the SPARQL half: the query text with the
/// document extension and `$TCPATH` substituted.
fn build_query(root: &Path, number: &str, suite: &str) -> Result<String, String> {
    let path = root.join("tests").join(format!("{number}.sparql"));
    let text = fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(substitute(&text, suite))
}

/// A port of `CrazyIvan::Core.get_test_content`: turns `tests/NNNN.txt` into the document the
/// suite would have served for `suite`.
fn build_document(root: &Path, number: &str, suite: &str) -> Result<String, String> {
    let path = root.join("tests").join(format!("{number}.txt"));
    let text = fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;

    let mut found_head = false;
    let mut in_json = false;
    let mut json = String::new();
    let mut root_attributes: Vec<String> = Vec::new();
    let mut content = String::new();

    for line in text.split_inclusive('\n') {
        // The Ruby original is a `case`, so only the first matching branch runs.
        if line.contains("<head") {
            found_head = true;
        } else if line.contains('{') {
            in_json = true;
        }
        if found_head {
            content.push_str(line);
            continue;
        }
        // An SVG root element carries its own namespace declaration and ends the preamble.
        found_head = line.contains("http://www.w3.org/2000/svg");
        if in_json {
            json.push_str(line.trim());
            if line.contains('}') {
                in_json = false;
            }
        } else {
            root_attributes.push(line.trim().to_owned());
        }
    }

    let mut attributes = String::new();
    let prefixes = parse_json_object(&json);
    if !prefixes.is_empty() {
        let pairs: Vec<String> = prefixes
            .iter()
            .map(|(prefix, iri)| format!("{prefix}: {iri}"))
            .collect();
        attributes.push(' ');
        let _ = write!(attributes, "prefix=\"{}\"", pairs.join(" "));
    }
    let rest = root_attributes.join("\n");
    if !rest.is_empty() {
        attributes.push(' ');
        attributes.push_str(&rest);
    }

    let content = substitute(&content, suite);
    Ok(match suffix(suite) {
        "html" => format!("<!DOCTYPE html>\n<html{attributes}>\n{content}</html>"),
        "xhtml" if suite == "xhtml1" => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML+RDFa 1.1//EN\" \
             \"http://www.w3.org/MarkUp/DTD/xhtml-rdfa-2.dtd\">\n\
             <html xmlns=\"http://www.w3.org/1999/xhtml\" version=\"XHTML+RDFa 1.1\"\
             {attributes}>\n{content}</html>"
        ),
        "xhtml" => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE html>\n\
             <html xmlns=\"http://www.w3.org/1999/xhtml\"{attributes}>\n{content}</html>"
        ),
        "svg" => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg{attributes}>\n{content}</svg>"
        ),
        _ => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root{attributes}>\n{content}</root>"
        ),
    })
}

/// Rewrites `NNNN.xhtml` to the host language's extension and expands `$TCPATH`.
fn substitute(text: &str, suite: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(".xhtml") {
        let (head, tail) = rest.split_at(index);
        let digits = head.chars().rev().take_while(char::is_ascii_digit).count();
        if digits >= 4 {
            out.push_str(head);
            out.push('.');
            out.push_str(suffix(suite));
        } else {
            out.push_str(head);
            out.push_str(".xhtml");
        }
        rest = &tail[".xhtml".len()..];
    }
    out.push_str(rest);
    out.replace("$TCPATH", &tcpath(suite))
}

/// Reads the flat `{"prefix": "iri", …}` object the suite puts at the head of a test fragment,
/// preserving the order the pairs appear in.
fn parse_json_object(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    let mut strings: Vec<String> = Vec::new();
    while let Some(ch) = chars.next() {
        if ch != '"' {
            continue;
        }
        let mut value = String::new();
        for ch in chars.by_ref() {
            if ch == '"' {
                break;
            }
            value.push(ch);
        }
        strings.push(value);
    }
    let mut iter = strings.into_iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        out.push((key, value));
    }
    out
}

#[test]
fn the_document_builder_matches_the_suites_own_wrapping() {
    let Some(root) = suite_root() else {
        return;
    };
    let document = build_document(&root, "0001", "html5").expect("test 0001 builds");
    assert!(
        document
            .starts_with("<!DOCTYPE html>\n<html prefix=\"dc: http://purl.org/dc/elements/1.1/\">")
    );
    assert!(
        document.contains("<span class=\"author\" about=\"photo1.jpg\" property=\"dc:creator\">")
    );

    let query = build_query(&root, "0001", "html5").expect("test 0001 has a query");
    assert!(query.contains("<http://rdfa.info/test-suite/test-cases/rdfa1.1/html5/photo1.jpg>"));
}
