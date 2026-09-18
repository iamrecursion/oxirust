//! SPARQL query/update syntax conformance runner.
//!
//! Drives every `manifest.ttl` found under the `rdf-tests/sparql/sparql10` and
//! `rdf-tests/sparql/sparql11` trees (and, behind this crate's `rdf-12` feature, `sparql12` too —
//! see `run_sparql12` below) of an [`oxixml_testsuite`](crate)-discovered `$OXIXML_TESTSUITE_DIR`
//! checkout against [`oxixml_sparql_syntax::SparqlParser`]. Each of the four syntax-test types in
//! the [`mf:`][mf] vocabulary is checked:
//!
//! * `mf:PositiveSyntaxTest[11]` — the query must parse, and printing the parsed algebra and
//!   re-parsing it must reproduce the exact same algebra (a round-trip check the manifest itself
//!   does not ask for, but which is nearly free once the query already parses once);
//! * `mf:NegativeSyntaxTest[11]` — the query must be rejected;
//! * `mf:Positive/NegativeUpdateSyntaxTest[11]` — the same two shapes, for `Update`.
//!
//! `sparql12`'s manifests reuse the same (unsuffixed) type names as `sparql10`, so `Kind` (below)
//! needs no SPARQL-1.2-specific variant of its own.
//!
//! Evaluation tests (the ones that also carry expected query results) are out of scope: this
//! runner only exercises the syntax-to-algebra half of the crate, which is all `oxixml-sparql-syntax`
//! implements.
//!
//! [mf]: http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_sparql_syntax::SparqlParser;

use crate::env;
use crate::manifest::{MF, ManifestIndex, strip_base};
use crate::report::SuiteReport;

/// A synthetic base under which every manifest is parsed, so that relative `mf:action`
/// references resolve predictably and [`strip_base`] can recover the on-disk path.
const DUMMY_BASE: &str = "http://oxixml.invalid/sparql-tests/";

/// The base IRI queries and updates themselves are parsed against; the suite's syntax tests do
/// not depend on its value, so any stable IRI will do.
const QUERY_BASE: &str = "http://example.com/base/";

/// What a manifest entry asks of the parser.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    PositiveQuery,
    NegativeQuery,
    PositiveUpdate,
    NegativeUpdate,
}

impl Kind {
    fn from_type_iri(iri: &str) -> Option<Self> {
        match iri.strip_prefix(MF)? {
            "PositiveSyntaxTest" | "PositiveSyntaxTest11" => Some(Self::PositiveQuery),
            "NegativeSyntaxTest" | "NegativeSyntaxTest11" => Some(Self::NegativeQuery),
            "PositiveUpdateSyntaxTest" | "PositiveUpdateSyntaxTest11" => Some(Self::PositiveUpdate),
            "NegativeUpdateSyntaxTest" | "NegativeUpdateSyntaxTest11" => Some(Self::NegativeUpdate),
            _ => None,
        }
    }

    const fn is_positive(self) -> bool {
        matches!(self, Self::PositiveQuery | Self::PositiveUpdate)
    }

    const fn is_update(self) -> bool {
        matches!(self, Self::PositiveUpdate | Self::NegativeUpdate)
    }
}

/// One runnable manifest entry.
struct Entry {
    id: String,
    kind: Kind,
    path: PathBuf,
}

/// Runs the SPARQL 1.0 and SPARQL 1.1 syntax suites, plus SPARQL 1.2 when this crate is built
/// with the `rdf-12` feature (see `run_sparql12` below).
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let mut reports = vec![
        run_family("sparql10", "sparql/sparql10"),
        run_family("sparql11", "sparql/sparql11"),
    ];
    reports.extend(run_sparql12());
    reports
}

/// The SPARQL 1.2 syntax suite (triple terms, reifiers, annotation blocks, `VERSION`), present
/// only when this crate's `rdf-12` feature is enabled.
///
/// `oxixml_sparql_syntax`'s SPARQL 1.2 grammar productions live in a module gated behind that
/// crate's own `sparql-12` feature (see `oxixml_sparql_syntax::parser::star`'s doc comment), which
/// this crate's `Cargo.toml` forwards `rdf-12` to, the same way every other RDF-1.2-shaped runner
/// here threads it through (see [`crate::isomorphism`]). Without the feature, `SparqlParser`
/// simply does not implement the sparql12 grammar, so every case would report an "unexpected
/// parse error" or "wrongly accepted" rather than a documented gap; returning no report instead
/// (an empty [`Option`], so the suite is absent from the table entirely) keeps a default,
/// feature-bare build honest about what it can attempt, the same way an unset environment
/// variable produces no row rather than a failing one.
///
/// Returns [`Option`] rather than pushing directly so [`run`] never needs a conditionally-`mut`
/// `Vec` — `Option<SuiteReport>` already implements [`IntoIterator`] with 0 or 1 items, which
/// `Vec::extend` consumes either way.
#[cfg(feature = "rdf-12")]
fn run_sparql12() -> Option<SuiteReport> {
    Some(run_family("sparql12", "sparql/sparql12"))
}

/// See the `rdf-12`-enabled [`run_sparql12`] above.
#[cfg(not(feature = "rdf-12"))]
fn run_sparql12() -> Option<SuiteReport> {
    None
}

fn run_family(directory: &str, suite_name: &str) -> SuiteReport {
    let Some(testsuite_root) = env::testsuite_dir() else {
        return SuiteReport::unavailable(suite_name, env::missing(env::TESTSUITE_DIR_VAR));
    };
    let root = testsuite_root
        .join("rdf-tests")
        .join("sparql")
        .join(directory);
    if !root.is_dir() {
        return SuiteReport::unavailable(
            suite_name,
            env::not_found(
                env::TESTSUITE_DIR_VAR,
                &format!("rdf-tests/sparql/{directory}"),
            ),
        );
    }

    let mut report = SuiteReport::new(suite_name);
    for manifest_path in env::find_files_named(&root, "manifest.ttl") {
        run_manifest(&root, &manifest_path, &mut report);
    }
    report
}

/// Parses one `manifest.ttl` and runs every syntax-test entry it declares.
fn run_manifest(root: &Path, manifest_path: &Path, report: &mut SuiteReport) {
    let Ok(bytes) = fs::read(manifest_path) else {
        report.record_skip("manifest unreadable");
        return;
    };
    let relative_dir = manifest_path
        .parent()
        .and_then(|parent| parent.strip_prefix(root).ok())
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();
    let base_iri = if relative_dir.is_empty() {
        DUMMY_BASE.to_owned()
    } else {
        format!("{DUMMY_BASE}{relative_dir}/")
    };
    let Ok(index) = ManifestIndex::parse(&bytes, &base_iri) else {
        report.record_skip("manifest did not parse as Turtle");
        return;
    };

    for subject in index.subjects() {
        let Some(kind) = index.types(subject).find_map(Kind::from_type_iri) else {
            continue;
        };
        let Some(action_iri) = index.first_iri(subject, &format!("{MF}action")) else {
            continue;
        };
        let Some(relative) = strip_base(action_iri, &base_iri) else {
            report.record_skip("mf:action outside the manifest's own directory");
            continue;
        };
        let entry = Entry {
            id: subject.trim_start_matches(DUMMY_BASE).to_owned(),
            kind,
            path: manifest_path.parent().unwrap_or(root).join(relative),
        };
        run_entry(&entry, report);
    }
    // A manifest indexed nothing this driver recognizes — no `rdf:type` matching a known syntax
    // test — is not itself an error; `mf:Manifest` container entries (the ones that only
    // `mf:include` other files) are exactly this shape, and `find_files_named` already visits
    // every included file directly, so nothing is lost by not following `mf:include` here.
}

fn run_entry(entry: &Entry, report: &mut SuiteReport) {
    let Ok(text) = fs::read_to_string(&entry.path) else {
        report.record_skip(format!("{}: action file unreadable", entry.id));
        return;
    };
    let parser = SparqlParser::new()
        .with_base_iri(QUERY_BASE)
        .unwrap_or_default();

    if entry.kind.is_update() {
        match parser.parse_update(&text) {
            Ok(update) if entry.kind.is_positive() => {
                match parser.parse_update(&update.to_string()) {
                    Ok(again) if again == update => report.record_pass(),
                    Ok(again) => report.record_fail(
                        &entry.id,
                        format!("round-trip mismatch\n  first:  {update}\n  second: {again}"),
                    ),
                    Err(error) => {
                        report.record_fail(
                            &entry.id,
                            format!("printed form did not re-parse: {error}"),
                        );
                    }
                }
            }
            Ok(_) => report.record_fail(&entry.id, "expected a syntax error, but it parsed"),
            Err(error) if entry.kind.is_positive() => {
                report.record_fail(&entry.id, format!("unexpected parse error: {error}"));
            }
            Err(_) => report.record_pass(),
        }
    } else {
        match parser.parse_query(&text) {
            Ok(query) if entry.kind.is_positive() => match parser.parse_query(&query.to_string()) {
                Ok(again) if again == query => report.record_pass(),
                Ok(again) => report.record_fail(
                    &entry.id,
                    format!("round-trip mismatch\n  first:  {query}\n  second: {again}"),
                ),
                Err(error) => {
                    report
                        .record_fail(&entry.id, format!("printed form did not re-parse: {error}"));
                }
            },
            Ok(_) => report.record_fail(&entry.id, "expected a syntax error, but it parsed"),
            Err(error) if entry.kind.is_positive() => {
                report.record_fail(&entry.id, format!("unexpected parse error: {error}"));
            }
            Err(_) => report.record_pass(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_recognizes_every_manifest_type() {
        assert_eq!(
            Kind::from_type_iri(&format!("{MF}PositiveSyntaxTest")),
            Some(Kind::PositiveQuery)
        );
        assert_eq!(
            Kind::from_type_iri(&format!("{MF}PositiveSyntaxTest11")),
            Some(Kind::PositiveQuery)
        );
        assert_eq!(
            Kind::from_type_iri(&format!("{MF}NegativeSyntaxTest11")),
            Some(Kind::NegativeQuery)
        );
        assert_eq!(
            Kind::from_type_iri(&format!("{MF}PositiveUpdateSyntaxTest")),
            Some(Kind::PositiveUpdate)
        );
        assert_eq!(
            Kind::from_type_iri(&format!("{MF}NegativeUpdateSyntaxTest11")),
            Some(Kind::NegativeUpdate)
        );
        assert_eq!(Kind::from_type_iri("http://example.com/Other"), None);
    }

    #[test]
    fn a_small_embedded_manifest_runs_end_to_end() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-sparql-fixture-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        fs::write(dir.join("good.rq"), b"SELECT * WHERE { ?s ?p ?o }").expect("scratch file");
        fs::write(dir.join("bad.rq"), b"SELECT *").expect("scratch file");
        let manifest = dir.join("manifest.ttl");
        fs::write(
            &manifest,
            br#"
                @prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
                <#good> a mf:PositiveSyntaxTest ; mf:name "good" ; mf:action <good.rq> .
                <#bad> a mf:NegativeSyntaxTest ; mf:name "bad" ; mf:action <bad.rq> .
            "#,
        )
        .expect("scratch file");

        let mut report = SuiteReport::new("sparql-fixture");
        run_manifest(&dir, &manifest, &mut report);
        assert_eq!(report.passed, 2, "failures: {:?}", report.failures);
        assert_eq!(report.failed, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrongly_accepted_negative_case_is_reported_as_a_failure() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-sparql-fixture-neg-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        // A perfectly valid query, mislabelled as a negative test — the harness must catch this.
        fs::write(dir.join("actually-fine.rq"), b"SELECT * WHERE { ?s ?p ?o }")
            .expect("scratch file");
        let manifest = dir.join("manifest.ttl");
        fs::write(
            &manifest,
            br#"
                @prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
                <#mislabelled> a mf:NegativeSyntaxTest ; mf:name "x" ; mf:action <actually-fine.rq> .
            "#,
        )
        .expect("scratch file");

        let mut report = SuiteReport::new("sparql-fixture");
        run_manifest(&dir, &manifest, &mut report);
        assert_eq!(report.failed, 1);
        assert_eq!(report.passed, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::TESTSUITE_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        // sparql10 + sparql11 always; sparql12 too, but only when this crate is built with
        // `rdf-12` (see `run_sparql12`) — without it, sparql12 has no row at all, available or
        // not, the same way a suite whose environment variable is unset has no row for a suite
        // this build cannot attempt in the first place.
        let expected_len = if cfg!(feature = "rdf-12") { 3 } else { 2 };
        assert_eq!(reports.len(), expected_len);
        assert!(reports.iter().all(|report| !report.is_available()));
    }

    #[test]
    fn run_sparql12_is_present_only_under_the_rdf_12_feature() {
        let sparql12 = run_sparql12();
        assert_eq!(sparql12.is_some(), cfg!(feature = "rdf-12"));
        if let Some(report) = &sparql12 {
            assert_eq!(report.suite, "sparql/sparql12");
        }
    }
}
