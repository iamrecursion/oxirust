//! [W3C RDFa Test Suite](https://rdfa.info/test-suite/) runner, driven against [`oxixml_rdfa`].
//!
//! `oxixml_rdfa` is a complete RDFa 1.1 extractor (see its crate documentation) with its own
//! in-crate copy of this exact suite (`oxixml-rdfa/tests/rdfa_suite.rs`), reporting 100% on all
//! four host languages it implements. This module re-drives the same suite at the workspace
//! level, through the crate's *published* API only ([`RdfaParser`]) — never anything from that
//! `tests/` directory, which (being test-only code) is not something a sibling crate can depend
//! on in the first place.
//!
//! # Suite shape
//!
//! The suite is not a `mf:`-vocabulary manifest like every other RDF-family runner in this crate;
//! it is [its own small vocabulary](http://rdfa.info/vocabs/rdfa-test#) naming, per test number,
//! which RDFa version(s) and host language(s) a case applies to and whether its `ASK` query is
//! expected to hold. [`crate::manifest::ManifestIndex`] still does the Turtle indexing — it is a
//! generic subject/predicate index, not tied to the `mf:` namespace — only the predicate names
//! and the case-selection rules below are specific to this suite.
//!
//! A case's *document* is not shipped by the suite either: `tests/NNNN.txt` is a fragment that
//! the suite's own Ruby harness (`CrazyIvan::Core.get_test_content`) wraps into a full document
//! per host language, and `tests/NNNN.sparql` gets the same `$TCPATH`/`.xhtml`-extension
//! substitution. `build_document` and `build_query` are a port of that wrapping — there is no
//! more "idiomatic" way to write them, they simply have to reproduce what the suite's own server
//! would have sent an implementation under test.
//!
//! Only `rdfa1.1` cases of the `html5`, `xhtml1`, `xhtml5` and `xml` host languages are run — the
//! ones [`oxixml_rdfa`] claims (see [`HostLanguage`]). A case not declaring one of those four host
//! languages at all is simply not part of that host language's report (the same way
//! [`crate::xslt30`] silently excludes test sets outside its own allowlist); a case that does
//! declare the host language but targets `rdfa1.0` counts as skipped.
//!
//! # Evaluating the expectation
//!
//! Every case states its expectation as a SPARQL `ASK`. `sparql_ask` is a small, private
//! evaluator for exactly the fragment of the algebra those queries use — see its own
//! documentation for what that covers and why this crate cannot instead depend on a real SPARQL
//! engine for it.

mod sparql_ask;

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_model::Triple;
use oxixml_rdfa::{HostLanguage, RdfaParser};
use oxixml_sparql_syntax::SparqlParser;

use crate::env;
use crate::manifest::{ManifestIndex, term_literal};
use crate::report::SuiteReport;

/// The `rdfa-test:` vocabulary namespace the manifest's `rdfaVersion`/`hostLanguage` predicates
/// live under.
const RDFA_TEST_NS: &str = "http://rdfa.info/vocabs/rdfa-test#";
/// `dc:title`, giving a case's human-readable name.
const DC_TITLE: &str = "http://purl.org/dc/terms/title";
/// `test:expectedResults`, `"true"` (the default when absent) or `"false"`.
const TEST_EXPECTED_RESULTS: &str = "http://www.w3.org/2006/03/test-description#expectedResults";
/// The only RDFa version [`oxixml_rdfa`] implements; every other version's cases are skipped.
const VERSION: &str = "rdfa1.1";
/// A synthetic base the manifest is parsed under; every subject in the real manifest is an
/// absolute IRI, so nothing ever actually resolves against this — see [`ManifestIndex::parse`].
const MANIFEST_BASE_IRI: &str = "http://oxixml.invalid/rdfa-manifest/";

/// The host languages this runner exercises, and the [`HostLanguage`] each one maps to. Matches
/// the suite names [`oxixml_rdfa`]'s own harness runs — `html4` and `svg` cases are out of scope
/// because the crate does not implement either host language.
const HOST_LANGUAGES: [(&str, HostLanguage); 4] = [
    ("html5", HostLanguage::Html5),
    ("xhtml1", HostLanguage::Xhtml1),
    ("xhtml5", HostLanguage::Xhtml5),
    ("xml", HostLanguage::Xml),
];

/// One row of the manifest: a test number and the facts about it this runner needs.
struct Case {
    number: String,
    title: String,
    versions: Vec<String>,
    host_languages: Vec<String>,
    expected: bool,
}

/// Runs the W3C RDFa Test Suite against [`oxixml_rdfa`], one [`SuiteReport`] per host language.
///
/// Reports as unavailable — one row per host language, so [`crate::all_suites`]'s table always
/// shows the same four `rdfa/*` rows regardless of whether the suite is provisioned — when
/// `OXIXML_RDFA_SUITE_DIR` is unset or has no `manifest.ttl` under it.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(root) = suite_root() else {
        let reason = match env::rdfa_suite_dir() {
            None => env::missing(env::RDFA_SUITE_DIR_VAR),
            Some(_) => env::not_found(env::RDFA_SUITE_DIR_VAR, "manifest.ttl"),
        };
        return HOST_LANGUAGES
            .iter()
            .map(|(name, _)| SuiteReport::unavailable(format!("rdfa/{name}"), reason.clone()))
            .collect();
    };

    let cases = match load_cases(&root) {
        Ok(cases) => cases,
        Err(message) => {
            return HOST_LANGUAGES
                .iter()
                .map(|(name, _)| {
                    let mut report = SuiteReport::new(format!("rdfa/{name}"));
                    report.record_skip(message.clone());
                    report
                })
                .collect();
        }
    };

    HOST_LANGUAGES
        .iter()
        .map(|(name, host)| run_host_language(&root, &cases, name, *host))
        .collect()
}

fn suite_root() -> Option<PathBuf> {
    let root = env::rdfa_suite_dir()?;
    root.join("manifest.ttl").is_file().then_some(root)
}

fn run_host_language(
    root: &Path,
    cases: &[Case],
    suite_name: &str,
    host: HostLanguage,
) -> SuiteReport {
    let mut report = SuiteReport::new(format!("rdfa/{suite_name}"));
    for case in cases {
        if !case.host_languages.iter().any(|value| value == suite_name) {
            continue; // this case is not part of this host language's suite at all
        }
        if !case.versions.iter().any(|value| value == VERSION) {
            report.record_skip("targets an RDFa version other than 1.1");
            continue;
        }
        run_case(root, case, suite_name, host, &mut report);
    }
    report
}

fn run_case(root: &Path, case: &Case, suite: &str, host: HostLanguage, report: &mut SuiteReport) {
    let Some(document) = build_document(root, &case.number, suite) else {
        report.record_skip("test fragment file is missing or unreadable");
        return;
    };
    let Some(query_text) = build_query(root, &case.number, suite) else {
        report.record_skip("query file is missing or unreadable");
        return;
    };
    let base = document_url(&case.number, suite);

    let parser = match RdfaParser::new()
        .with_host_language(host)
        .with_base_iri(&base)
    {
        Ok(parser) => parser,
        Err(error) => {
            report.record_fail(
                &case.number,
                format!("{}: invalid base IRI: {error}", case.title),
            );
            return;
        }
    };
    let graph: Vec<Triple> = match parser.for_slice(&document).collect::<Result<Vec<_>, _>>() {
        Ok(graph) => graph,
        Err(error) => {
            report.record_fail(
                &case.number,
                format!("{}: extraction failed: {error}", case.title),
            );
            return;
        }
    };

    let Ok(query) = SparqlParser::new().parse_query(&query_text) else {
        report.record_skip("the case's ASK query does not parse");
        return;
    };
    let Some(answer) = sparql_ask::ask(&query, &graph) else {
        report.record_skip("the ASK uses a construct this harness does not evaluate");
        return;
    };

    if answer == case.expected {
        report.record_pass();
    } else {
        report.record_fail(
            &case.number,
            format!(
                "{}: expected the ASK to be {}, got {answer} ({} triple(s) extracted)",
                case.title,
                case.expected,
                graph.len()
            ),
        );
    }
}

// --- manifest ----------------------------------------------------------------------------------

fn load_cases(root: &Path) -> Result<Vec<Case>, String> {
    let bytes = fs::read(root.join("manifest.ttl")).map_err(|error| error.to_string())?;
    let index = ManifestIndex::parse(&bytes, MANIFEST_BASE_IRI)?;

    let mut cases: Vec<Case> = index
        .subjects()
        .filter_map(|subject| {
            let number = case_number(subject)?;
            let versions = index
                .objects(subject, &format!("{RDFA_TEST_NS}rdfaVersion"))
                .iter()
                .filter_map(term_literal)
                .map(str::to_owned)
                .collect();
            let host_languages = index
                .objects(subject, &format!("{RDFA_TEST_NS}hostLanguage"))
                .iter()
                .filter_map(term_literal)
                .map(str::to_owned)
                .collect();
            let title = index
                .first_literal(subject, DC_TITLE)
                .unwrap_or(subject)
                .to_owned();
            let expected = index
                .first_literal(subject, TEST_EXPECTED_RESULTS)
                .is_none_or(|value| value == "true");
            Some(Case {
                number,
                title,
                versions,
                host_languages,
                expected,
            })
        })
        .collect();
    cases.sort_by(|a, b| a.number.cmp(&b.number));
    Ok(cases)
}

/// The manifest keys each case's subject by an absolute IRI ending in a four-digit test number
/// (`.../test-cases/0001`); anything else in the manifest (there is nothing else today, but
/// nothing here should assume that) is not a case.
fn case_number(subject: &str) -> Option<String> {
    let number = subject.rsplit('/').next()?;
    (number.len() == 4 && number.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| number.to_owned())
}

// --- document/query building ------------------------------------------------------------------
//
// A port of `CrazyIvan::Core.get_test_content`, the suite's own Ruby harness function that turns
// a `tests/NNNN.txt` fragment (plus `tests/NNNN.sparql`) into what the suite would have served an
// implementation under test. See the module documentation for why this cannot be written any more
// "idiomatically" — it has to match an external, fixed wrapping convention exactly.

/// The file extension the suite serves a host language's documents under.
fn suffix(suite: &str) -> &'static str {
    if suite.contains("xhtml") {
        "xhtml"
    } else if suite.contains("html") {
        "html"
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

/// The query text with the document extension and `$TCPATH` substituted.
fn build_query(root: &Path, number: &str, suite: &str) -> Option<String> {
    let text = fs::read_to_string(root.join("tests").join(format!("{number}.sparql"))).ok()?;
    Some(substitute(&text, suite))
}

/// Turns `tests/NNNN.txt` into the document the suite would have served for `suite`: everything
/// up to the first line containing `<head` (or declaring the SVG namespace) is either a leading
/// `{"prefix": "iri", …}` JSON object — folded into a `prefix="…"` attribute on the root element —
/// or extra attribute text for the root element; everything from that line on is copied through
/// as the document's content, inside the host language's own root-element/DOCTYPE wrapping.
fn build_document(root: &Path, number: &str, suite: &str) -> Option<String> {
    let text = fs::read_to_string(root.join("tests").join(format!("{number}.txt"))).ok()?;

    let mut found_head = false;
    let mut in_json = false;
    let mut json = String::new();
    let mut root_attributes: Vec<String> = Vec::new();
    let mut content = String::new();

    for line in text.split_inclusive('\n') {
        if line.contains("<head") {
            found_head = true;
        } else if line.contains('{') {
            in_json = true;
        }
        if found_head {
            content.push_str(line);
            continue;
        }
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
        attributes.push_str(&format!(" prefix=\"{}\"", pairs.join(" ")));
    }
    let rest = root_attributes.join("\n");
    if !rest.is_empty() {
        attributes.push(' ');
        attributes.push_str(&rest);
    }

    let content = substitute(&content, suite);
    Some(match suffix(suite) {
        "html" => format!("<!DOCTYPE html>\n<html{attributes}>\n{content}</html>"),
        "xhtml" if suite == "xhtml1" => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML+RDFa 1.1//EN\" \
             \"http://www.w3.org/MarkUp/DTD/xhtml-rdfa-2.dtd\">\n\
             <html xmlns=\"http://www.w3.org/1999/xhtml\" version=\"XHTML+RDFa 1.1\"{attributes}>\n{content}</html>"
        ),
        "xhtml" => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE html>\n\
             <html xmlns=\"http://www.w3.org/1999/xhtml\"{attributes}>\n{content}</html>"
        ),
        _ => format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root{attributes}>\n{content}</root>"
        ),
    })
}

/// Rewrites a `NNNN.xhtml` reference to the host language's own extension, and expands `$TCPATH`
/// to this host language's base test-case URL.
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
/// preserving encounter order. Not a general JSON reader — this only ever sees quoted strings
/// alternating key/value, which is all the suite's fragments use it for.
fn parse_json_object(text: &str) -> Vec<(String, String)> {
    let mut strings: Vec<String> = Vec::new();
    let mut chars = text.chars();
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
    let mut pairs = Vec::new();
    let mut iter = strings.into_iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        pairs.push((key, value));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_unavailable_without_a_configured_suite_root() {
        if std::env::var_os(env::RDFA_SUITE_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), HOST_LANGUAGES.len());
        for (report, (name, _)) in reports.iter().zip(HOST_LANGUAGES) {
            assert!(!report.is_available());
            assert_eq!(report.suite, format!("rdfa/{name}"));
        }
    }

    #[test]
    fn case_number_accepts_only_a_four_digit_final_segment() {
        assert_eq!(
            case_number("http://rdfa.info/test-suite/test-cases/0001"),
            Some("0001".to_owned())
        );
        assert_eq!(
            case_number("http://rdfa.info/test-suite/test-cases/12345"),
            None
        );
        assert_eq!(
            case_number("http://rdfa.info/test-suite/test-cases/abcd"),
            None
        );
    }

    #[test]
    fn suffix_and_document_url_match_the_suites_url_scheme() {
        assert_eq!(suffix("html5"), "html");
        assert_eq!(suffix("xhtml1"), "xhtml");
        assert_eq!(suffix("xhtml5"), "xhtml");
        assert_eq!(suffix("xml"), "xml");
        assert_eq!(
            document_url("0001", "html5"),
            "http://rdfa.info/test-suite/test-cases/rdfa1.1/html5/0001.html"
        );
    }

    #[test]
    fn substitute_rewrites_the_extension_and_tcpath() {
        assert_eq!(substitute("see 0002.xhtml", "html5"), "see 0002.html");
        // Fewer than four digits before `.xhtml` is not a test-case reference.
        assert_eq!(substitute("a12.xhtml", "html5"), "a12.xhtml");
        assert_eq!(
            substitute("$TCPATH/img.png", "xml"),
            "http://rdfa.info/test-suite/test-cases/rdfa1.1/xml/img.png"
        );
    }

    #[test]
    fn parse_json_object_reads_ordered_string_pairs() {
        assert_eq!(
            parse_json_object(
                r#"{"dc": "http://purl.org/dc/terms/", "foaf": "http://xmlns.com/foaf/0.1/"}"#
            ),
            vec![
                ("dc".to_owned(), "http://purl.org/dc/terms/".to_owned()),
                ("foaf".to_owned(), "http://xmlns.com/foaf/0.1/".to_owned()),
            ]
        );
        assert!(parse_json_object("").is_empty());
    }

    /// An end-to-end run over a tiny, hand-built suite checkout: one manifest entry, its fragment
    /// and its query, laid out exactly like a real `OXIXML_RDFA_SUITE_DIR` checkout. This is what
    /// stands in for the real suite in this crate's own tests (see the module documentation for
    /// why the real suite is not vendored).
    #[test]
    fn end_to_end_over_a_tiny_embedded_suite() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-rdfa-fixture-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("tests")).expect("scratch dir");

        fs::write(
            dir.join("manifest.ttl"),
            br#"
                @prefix rdfatest: <http://rdfa.info/vocabs/rdfa-test#> .
                @prefix dc: <http://purl.org/dc/terms/> .
                @prefix test: <http://www.w3.org/2006/03/test-description#> .

                <http://rdfa.info/test-suite/test-cases/0001>
                    rdfatest:rdfaVersion "rdfa1.1" ;
                    rdfatest:hostLanguage "html5" , "xhtml1" ;
                    dc:title "a simple property" ;
                    test:expectedResults "true" .
            "#,
        )
        .expect("scratch file");
        fs::write(
            dir.join("tests").join("0001.txt"),
            br#"{"dc": "http://purl.org/dc/terms/"}
<head><title>Test 0001</title></head>
<body>
<p about="http://example.com/thing" property="dc:title">Kokoro</p>
</body>
"#,
        )
        .expect("scratch file");
        fs::write(
            dir.join("tests").join("0001.sparql"),
            b"ASK WHERE { <http://example.com/thing> <http://purl.org/dc/terms/title> \"Kokoro\" }",
        )
        .expect("scratch file");

        let cases = load_cases(&dir).expect("the fixture manifest parses");
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].number, "0001");
        assert!(cases[0].expected);

        let html5 = run_host_language(&dir, &cases, "html5", HostLanguage::Html5);
        assert_eq!(html5.passed, 1, "failures: {:?}", html5.failures);
        assert_eq!(html5.failed, 0);

        let xhtml1 = run_host_language(&dir, &cases, "xhtml1", HostLanguage::Xhtml1);
        assert_eq!(xhtml1.passed, 1, "failures: {:?}", xhtml1.failures);

        // `xhtml5` and `xml` are not in this case's `rdfatest:hostLanguage` list, so they must
        // report neither a pass nor a fail — the case simply is not part of those suites.
        let xhtml5 = run_host_language(&dir, &cases, "xhtml5", HostLanguage::Xhtml5);
        assert_eq!(xhtml5.total, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_case_targeting_only_rdfa10_is_skipped_not_failed() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-rdfa-fixture-10-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("tests")).expect("scratch dir");
        fs::write(
            dir.join("manifest.ttl"),
            br#"
                @prefix rdfatest: <http://rdfa.info/vocabs/rdfa-test#> .
                <http://rdfa.info/test-suite/test-cases/0002>
                    rdfatest:rdfaVersion "rdfa1.0" ;
                    rdfatest:hostLanguage "html5" .
            "#,
        )
        .expect("scratch file");

        let cases = load_cases(&dir).expect("the fixture manifest parses");
        let html5 = run_host_language(&dir, &cases, "html5", HostLanguage::Html5);
        assert_eq!(html5.skipped, 1);
        assert_eq!(html5.passed, 0);
        assert_eq!(html5.failed, 0);

        let _ = fs::remove_dir_all(&dir);
    }
}
