//! W3C `rdf-tests` RDF 1.1 concrete-syntax conformance runner.
//!
//! Drives the `rdf-tests/rdf/rdf11/{rdf-n-triples,rdf-n-quads,rdf-turtle,rdf-trig,rdf-xml}`
//! manifests — five families, one per concrete syntax — against this workspace's own parsers.
//! Every manifest in this tree is a single flat file with no `mf:include`, so this runner (unlike
//! [`crate::sparql`]) never needs to walk an include graph.
//!
//! Three shapes of test recur across the `rdft:` vocabulary, told apart by the suffix of the
//! `rdf:type` local name (`PositiveSyntax`, `NegativeSyntax`/`NegativeEval`, or a plain `Eval`):
//!
//! * **positive syntax** — the action document must parse;
//! * **negative** (syntax or eval) — the action document must be rejected;
//! * **eval** — the action document must parse *and* the resulting graph/dataset must be
//!   [isomorphic](crate::isomorphism) to the `mf:result` document (N-Triples for a triple
//!   family, N-Quads for a quad family).
//!
//! RDF/XML's `rdft:TestXMLEval`/`rdft:TestXMLNegativeSyntax` follow the same eval/negative
//! shapes under different names and are driven by `run_rdfxml_family`.
//!
//! **Scope**: only the RDF 1.1 families are run. The `rdf12/*` manifests `mf:include` both a
//! version-specific sub-manifest *and*, with a `../../rdf11/...` reference, the corresponding
//! RDF 1.1 manifest itself — a cross-directory include graph this flat-manifest runner does not
//! walk. RDF 1.2 conformance is exercised by each producer crate's own `rdf12_report`-style test
//! instead (see e.g. `rdf/oxixml-turtle/tests/rdf_tests.rs`).

use std::fs;
use std::path::Path;

use oxixml_model::{Quad, Triple};
use oxixml_rdfxml::RdfXmlParser;
use oxixml_turtle::{NQuadsParser, NTriplesParser, TriGParser, TurtleParser};

use crate::env;
use crate::isomorphism::{quads_isomorphic, triples_isomorphic};
use crate::manifest::{MF, ManifestIndex, strip_base};
use crate::report::SuiteReport;

/// `mf:assumedTestBase`.
const MF_ASSUMED_BASE: &str =
    "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#assumedTestBase";

/// The RDF concrete syntax a family exercises.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    NTriples,
    NQuads,
    Turtle,
    TriG,
}

impl Format {
    const fn is_quad_format(self) -> bool {
        matches!(self, Self::NQuads | Self::TriG)
    }
}

/// One concrete-syntax family: where it lives under `rdf-tests/rdf`, which parser drives it, and
/// what it is reported as.
struct Family {
    path: &'static str,
    format: Format,
    suite: &'static str,
}

const FAMILIES: &[Family] = &[
    Family {
        path: "rdf11/rdf-n-triples",
        format: Format::NTriples,
        suite: "rdf-tests/n-triples",
    },
    Family {
        path: "rdf11/rdf-n-quads",
        format: Format::NQuads,
        suite: "rdf-tests/n-quads",
    },
    Family {
        path: "rdf11/rdf-turtle",
        format: Format::Turtle,
        suite: "rdf-tests/turtle",
    },
    Family {
        path: "rdf11/rdf-trig",
        format: Format::TriG,
        suite: "rdf-tests/trig",
    },
];

/// Runs every RDF 1.1 concrete-syntax family.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let mut reports: Vec<SuiteReport> = FAMILIES.iter().map(run_family).collect();
    reports.push(run_rdfxml_family());
    reports
}

fn run_family(family: &Family) -> SuiteReport {
    let Some(testsuite_root) = env::testsuite_dir() else {
        return SuiteReport::unavailable(family.suite, env::missing(env::TESTSUITE_DIR_VAR));
    };
    let root = testsuite_root
        .join("rdf-tests")
        .join("rdf")
        .join(family.path);
    let manifest_path = root.join("manifest.ttl");
    if !manifest_path.is_file() {
        return SuiteReport::unavailable(
            family.suite,
            env::not_found(
                env::TESTSUITE_DIR_VAR,
                &format!("rdf-tests/rdf/{}/manifest.ttl", family.path),
            ),
        );
    }

    let mut report = SuiteReport::new(family.suite);
    let Ok(bytes) = fs::read(&manifest_path) else {
        report.record_skip("manifest unreadable");
        return report;
    };
    let base_iri = format!("http://oxixml.invalid/rdf-tests/{}/", family.path);
    let Ok(index) = ManifestIndex::parse(&bytes, &base_iri) else {
        report.record_skip("manifest did not parse as Turtle");
        return report;
    };
    let base_prefix = index
        .first_iri(&base_iri, MF_ASSUMED_BASE)
        .unwrap_or(&base_iri)
        .to_owned();

    let context = FamilyContext {
        format: family.format,
        root: &root,
        base_iri: &base_iri,
        base_prefix: &base_prefix,
    };
    for subject in index.subjects() {
        let Some(kind) = index
            .types(subject)
            .find(|iri| iri.starts_with(RDFT))
            .map(str::to_owned)
        else {
            continue;
        };
        run_case(&context, &index, subject, &kind, &mut report);
    }
    report
}

/// The `rdftest#` namespace every family's `rdf:type` local name lives under.
const RDFT: &str = "http://www.w3.org/ns/rdftest#";

/// The parts of one [`run_family`] pass that stay the same across every case in it, bundled so
/// [`run_case`] takes one context parameter instead of four (`clippy::too_many_arguments`).
struct FamilyContext<'a> {
    format: Format,
    root: &'a Path,
    base_iri: &'a str,
    base_prefix: &'a str,
}

fn run_case(
    context: &FamilyContext<'_>,
    index: &ManifestIndex,
    subject: &str,
    kind: &str,
    report: &mut SuiteReport,
) {
    let id = format!(
        "{}: {}",
        subject.trim_start_matches(context.base_iri),
        local_name(kind)
    );
    let Some(action_iri) = index.first_iri(subject, &format!("{MF}action")) else {
        return;
    };
    let Some(action_rel) = strip_base(action_iri, context.base_iri) else {
        report.record_skip("mf:action outside the family directory");
        return;
    };
    let Ok(action_bytes) = fs::read(context.root.join(action_rel)) else {
        report.record_fail(&id, format!("cannot read action {action_rel}"));
        return;
    };
    let parse_base = format!("{}{action_rel}", context.base_prefix);

    if local_name(kind).ends_with("PositiveSyntax") {
        match parse_only(context.format, &action_bytes, &parse_base) {
            Ok(()) => report.record_pass(),
            Err(error) => report.record_fail(&id, error),
        }
        return;
    }
    if local_name(kind).ends_with("NegativeSyntax") || local_name(kind).ends_with("NegativeEval") {
        match parse_only(context.format, &action_bytes, &parse_base) {
            Ok(()) => report.record_fail(&id, "expected a parse error, but parsing succeeded"),
            Err(_) => report.record_pass(),
        }
        return;
    }
    if local_name(kind).ends_with("Eval") {
        let Some(result_iri) = index.first_iri(subject, &format!("{MF}result")) else {
            report.record_fail(&id, "eval test without mf:result");
            return;
        };
        let Some(result_rel) = strip_base(result_iri, context.base_iri) else {
            report.record_skip("mf:result outside the family directory");
            return;
        };
        let Ok(expected_bytes) = fs::read(context.root.join(result_rel)) else {
            report.record_fail(&id, format!("cannot read result {result_rel}"));
            return;
        };
        match eval_case(context.format, &action_bytes, &parse_base, &expected_bytes) {
            Ok(()) => report.record_pass(),
            Err(error) => report.record_fail(&id, error),
        }
        return;
    }
    report.record_skip(format!("unrecognised test type {kind}"));
}

fn local_name(iri: &str) -> &str {
    iri.rsplit('#').next().unwrap_or(iri)
}

fn parse_only(format: Format, bytes: &[u8], base: &str) -> Result<(), String> {
    if format.is_quad_format() {
        parse_quads(format, bytes, base).map(|_| ())
    } else {
        parse_triples(format, bytes, base).map(|_| ())
    }
}

fn eval_case(format: Format, action: &[u8], base: &str, expected: &[u8]) -> Result<(), String> {
    if format.is_quad_format() {
        let actual = parse_quads(format, action, base)?;
        let expected = NQuadsParser::new()
            .for_slice(expected)
            .collect::<Result<Vec<Quad>, _>>()
            .map_err(|error| format!("expected result did not parse as N-Quads: {error}"))?;
        if quads_isomorphic(&actual, &expected) {
            Ok(())
        } else {
            Err(format!(
                "datasets differ: {} parsed vs {} expected quads",
                actual.len(),
                expected.len()
            ))
        }
    } else {
        let actual = parse_triples(format, action, base)?;
        let expected = NTriplesParser::new()
            .for_slice(expected)
            .collect::<Result<Vec<Triple>, _>>()
            .map_err(|error| format!("expected result did not parse as N-Triples: {error}"))?;
        if triples_isomorphic(&actual, &expected) {
            Ok(())
        } else {
            Err(format!(
                "graphs differ: {} parsed vs {} expected triples",
                actual.len(),
                expected.len()
            ))
        }
    }
}

fn parse_triples(format: Format, bytes: &[u8], base: &str) -> Result<Vec<Triple>, String> {
    match format {
        Format::NTriples => NTriplesParser::new()
            .for_slice(bytes)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string()),
        Format::Turtle => {
            let parser = TurtleParser::new()
                .with_base_iri(base)
                .map_err(|error| error.to_string())?;
            parser
                .for_slice(bytes)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())
        }
        Format::NQuads | Format::TriG => Err("internal error: not a triple format".to_owned()),
    }
}

fn parse_quads(format: Format, bytes: &[u8], base: &str) -> Result<Vec<Quad>, String> {
    match format {
        Format::NQuads => NQuadsParser::new()
            .for_slice(bytes)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string()),
        Format::TriG => {
            let parser = TriGParser::new()
                .with_base_iri(base)
                .map_err(|error| error.to_string())?;
            parser
                .for_slice(bytes)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())
        }
        Format::NTriples | Format::Turtle => Err("internal error: not a quad format".to_owned()),
    }
}

// --- RDF/XML -------------------------------------------------------------------------------

fn run_rdfxml_family() -> SuiteReport {
    const SUITE: &str = "rdf-tests/rdfxml";
    let Some(testsuite_root) = env::testsuite_dir() else {
        return SuiteReport::unavailable(SUITE, env::missing(env::TESTSUITE_DIR_VAR));
    };
    let root = testsuite_root
        .join("rdf-tests")
        .join("rdf")
        .join("rdf11")
        .join("rdf-xml");
    let manifest_path = root.join("manifest.ttl");
    if !manifest_path.is_file() {
        return SuiteReport::unavailable(
            SUITE,
            env::not_found(
                env::TESTSUITE_DIR_VAR,
                "rdf-tests/rdf/rdf11/rdf-xml/manifest.ttl",
            ),
        );
    }

    let mut report = SuiteReport::new(SUITE);
    let Ok(bytes) = fs::read(&manifest_path) else {
        report.record_skip("manifest unreadable");
        return report;
    };
    let base_iri = "http://oxixml.invalid/rdf-tests/rdf11/rdf-xml/";
    let Ok(index) = ManifestIndex::parse(&bytes, base_iri) else {
        report.record_skip("manifest did not parse as Turtle");
        return report;
    };
    let base_prefix = index
        .first_iri(base_iri, MF_ASSUMED_BASE)
        .unwrap_or(base_iri)
        .to_owned();

    for subject in index.subjects() {
        let Some(kind) = index
            .types(subject)
            .find(|iri| iri.starts_with(RDFT))
            .map(str::to_owned)
        else {
            continue;
        };
        run_rdfxml_case(
            &index,
            subject,
            &kind,
            &root,
            base_iri,
            &base_prefix,
            &mut report,
        );
    }
    report
}

fn run_rdfxml_case(
    index: &ManifestIndex,
    subject: &str,
    kind: &str,
    root: &Path,
    base_iri: &str,
    base_prefix: &str,
    report: &mut SuiteReport,
) {
    let id = subject.trim_start_matches(base_iri).to_owned();
    let Some(action_iri) = index.first_iri(subject, &format!("{MF}action")) else {
        return;
    };
    let Some(action_rel) = strip_base(action_iri, base_iri) else {
        report.record_skip("mf:action outside the family directory");
        return;
    };
    let Ok(action_bytes) = fs::read(root.join(action_rel)) else {
        report.record_fail(&id, format!("cannot read action {action_rel}"));
        return;
    };
    let document_iri = format!("{base_prefix}{action_rel}");
    let positive = local_name(kind) == "TestXMLEval";

    let parsed: Result<Vec<Triple>, String> = RdfXmlParser::new()
        .with_base_iri(&document_iri)
        .map_err(|error| error.to_string())
        .and_then(|parser| {
            parser
                .for_slice(&action_bytes)
                .collect::<Result<Vec<Triple>, _>>()
                .map_err(|error| error.to_string())
        });

    if !positive {
        match parsed {
            Ok(triples) => {
                report.record_fail(
                    &id,
                    format!("expected a syntax error, got {} triple(s)", triples.len()),
                );
            }
            Err(_) => report.record_pass(),
        }
        return;
    }

    let actual = match parsed {
        Ok(triples) => triples,
        Err(error) => {
            report.record_fail(&id, format!("unexpected parse error: {error}"));
            return;
        }
    };
    let Some(result_iri) = index.first_iri(subject, &format!("{MF}result")) else {
        report.record_fail(&id, "eval test without mf:result");
        return;
    };
    let Some(result_rel) = strip_base(result_iri, base_iri) else {
        report.record_skip("mf:result outside the family directory");
        return;
    };
    let Ok(expected_bytes) = fs::read(root.join(result_rel)) else {
        report.record_fail(&id, format!("cannot read result {result_rel}"));
        return;
    };
    let expected = match NTriplesParser::new()
        .for_slice(&expected_bytes)
        .collect::<Result<Vec<Triple>, _>>()
    {
        Ok(triples) => triples,
        Err(error) => {
            report.record_fail(
                &id,
                format!("expected result did not parse as N-Triples: {error}"),
            );
            return;
        }
    };
    if triples_isomorphic(&actual, &expected) {
        report.record_pass();
    } else {
        report.record_fail(
            &id,
            format!(
                "graphs differ: {} parsed vs {} expected triples",
                actual.len(),
                expected.len()
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_name_strips_the_fragment() {
        assert_eq!(
            local_name(&format!("{RDFT}TestTurtleEval")),
            "TestTurtleEval"
        );
        assert_eq!(local_name("no-fragment"), "no-fragment");
    }

    #[test]
    fn end_to_end_over_a_tiny_embedded_turtle_family() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-rdf-tests-fixture-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        fs::write(
            dir.join("pos.ttl"),
            b"<http://e/s> <http://e/p> <http://e/o> .",
        )
        .expect("scratch file");
        fs::write(
            dir.join("pos.nt"),
            b"<http://e/s> <http://e/p> <http://e/o> .\n",
        )
        .expect("scratch file");
        fs::write(dir.join("neg.ttl"), b"this is not turtle at all {{{").expect("scratch file");
        let manifest = dir.join("manifest.ttl");
        fs::write(
            &manifest,
            br#"
                @prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
                @prefix rdft: <http://www.w3.org/ns/rdftest#> .
                <#pos> a rdft:TestTurtleEval ; mf:name "pos" ; mf:action <pos.ttl> ; mf:result <pos.nt> .
                <#neg> a rdft:TestTurtleNegativeSyntax ; mf:name "neg" ; mf:action <neg.ttl> .
            "#,
        )
        .expect("scratch file");

        let base_iri = "http://oxixml.invalid/rdf-tests-fixture/";
        let bytes = fs::read(&manifest).expect("manifest readable");
        let index = ManifestIndex::parse(&bytes, base_iri).expect("valid Turtle");
        let mut report = SuiteReport::new("fixture");
        let context = FamilyContext {
            format: Format::Turtle,
            root: &dir,
            base_iri,
            base_prefix: base_iri,
        };
        for subject in index.subjects() {
            let Some(kind) = index
                .types(subject)
                .find(|iri| iri.starts_with(RDFT))
                .map(str::to_owned)
            else {
                continue;
            };
            run_case(&context, &index, subject, &kind, &mut report);
        }
        assert_eq!(report.failed, 0, "failures: {:?}", report.failures);
        assert_eq!(report.passed, 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::TESTSUITE_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), FAMILIES.len() + 1);
        assert!(reports.iter().all(|report| !report.is_available()));
    }
}
