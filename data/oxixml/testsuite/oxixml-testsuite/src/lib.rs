//! Workspace-wide W3C/oxigraph conformance harness for OxiXML.
//!
//! Every crate under this workspace already carries its own focused test suite; this crate does
//! not replace those. It exists for the questions a per-crate suite cannot answer on its own:
//! *across the whole workspace*, which W3C/oxigraph conformance suites can we actually run right
//! now, and what is the pass rate of each? It is a dev-only harness (see `publish = false` in
//! `Cargo.toml`) — nothing here is meant to be depended on by another crate.
//!
//! # Layout
//!
//! * [`report`] — the suite-agnostic [`SuiteReport`](report::SuiteReport) /
//!   [`ConformanceReport`] types every runner below returns.
//! * [`mod@env`] — environment-variable-driven suite discovery (never a hardcoded path).
//! * [`manifest`] — a small `mf:`-vocabulary Turtle manifest index shared by the RDF-family
//!   runners.
//! * [`assertions`] and [`xdm_util`] — the QT3/XSLT `<result>` assertion checker and the XDM tree
//!   navigation helpers it and the catalogue drivers share.
//! * [`isomorphism`] — blank-node-bijection graph/dataset comparison for eval-style RDF tests.
//! * One module per conformance suite: [`xmlconf`], [`xsts`], [`qt3`], [`xslt30`], [`relaxng`],
//!   [`schematron`], [`c14n`], [`xmldsig`], [`rdf_tests`], [`rdf_canon`], [`json_ld`], [`sparql`],
//!   [`rdfa`]. Each exposes a `run() -> Vec<SuiteReport>` that discovers its suite under one of
//!   the environment variables in [`mod@env`] and reports [`SuiteReport::unavailable`] — never a
//!   panic, never a hard error — when that variable is unset or the expected layout is not found
//!   under it.
//!
//! [`SuiteReport::unavailable`]: report::SuiteReport::unavailable
//!
//! # Suite discovery
//!
//! No suite path is ever hardcoded. Three environment variables, read only in [`mod@env`], name the
//! roots a checkout provisions:
//!
//! * `OXIXML_XML_SUITES_DIR` — the W3C XML suites root ([`xmlconf`], [`xsts`], [`qt3`],
//!   [`xslt30`], [`relaxng`], [`schematron`], [`c14n`], [`xmldsig`]).
//! * `OXIXML_TESTSUITE_DIR` — the oxigraph testsuite root ([`rdf_tests`], [`rdf_canon`],
//!   [`json_ld`], [`sparql`]).
//! * `OXIXML_RDFA_SUITE_DIR` — an RDFa suite root ([`rdfa`]), if one is provisioned.
//!
//! A checkout with none of these set still builds and tests cleanly: every suite-driven `#[test]`
//! skips (by asserting only that the reported suite is unavailable), and [`all_suites`] returns a
//! report full of `unavailable` entries rather than failing.
//!
//! # Example
//!
//! The shared report type needs no suite checkout to demonstrate:
//!
//! ```
//! use oxixml_testsuite::report::{ConformanceReport, SuiteReport};
//!
//! let mut report = SuiteReport::new("demo");
//! report.record_pass();
//! report.record_pass();
//! report.record_fail("case-1", "mismatch");
//!
//! assert_eq!(report.total, 3);
//! assert!((report.pass_rate().expect("cases were attempted") - 66.666_666_666).abs() < 1e-6);
//!
//! // `ConformanceReport` is what `all_suites()` returns and the `report` binary prints: an
//! // aligned table with one row per suite and a TOTAL row at the bottom.
//! let table: ConformanceReport = [report].into_iter().collect();
//! assert!(table.to_string().contains("demo"));
//! ```
//!
//! Driving a real suite needs the environment above; see [`all_suites`] for the entry point the
//! `oxixml-conformance-report` binary and the crate's own integration tests use.

#![deny(missing_docs)]

pub mod assertions;
pub mod c14n;
pub mod env;
pub mod isomorphism;
pub mod json_ld;
pub mod ledger;
pub mod manifest;
pub mod qt3;
pub mod rdf_canon;
pub mod rdf_tests;
pub mod rdfa;
pub mod relaxng;
pub mod report;
pub mod schematron;
pub mod sparql;
pub mod xdm_util;
pub mod xmlconf;
pub mod xmldsig;
pub mod xslt30;
pub mod xsts;

use report::ConformanceReport;

/// Runs every conformance suite this crate knows how to drive and returns one combined report.
///
/// A suite whose environment variable is unset (or whose expected layout is missing under it)
/// still contributes a row — [`SuiteReport::unavailable`](report::SuiteReport::unavailable) — so
/// the printed table always lists the crate's full suite coverage, not just what happened to be
/// provisioned on the machine that ran it. This is what the `oxixml-conformance-report` binary
/// and the crate's `#[ignore]`d integration tests both call.
///
/// # Example
///
/// ```no_run
/// // `no_run`: exercising this for real means walking every configured suite, which can take
/// // several minutes — exactly the "heavy" work `tests/conformance.rs` keeps behind `#[ignore]`.
/// // This only needs to type-check here; see that file (or the crate's own unit test of this
/// // same function) for a run that actually executes it.
/// let report = oxixml_testsuite::all_suites();
/// // At least one row per suite family is always present, configured or not.
/// assert!(!report.0.is_empty());
/// ```
#[must_use]
pub fn all_suites() -> ConformanceReport {
    let mut report = ConformanceReport::new();
    report.extend(xmlconf::run());
    report.extend(xsts::run());
    report.extend(qt3::run());
    report.extend(xslt30::run());
    report.extend(relaxng::run());
    report.extend(schematron::run());
    report.extend(c14n::run());
    report.extend(xmldsig::run());
    report.extend(rdf_tests::run());
    report.extend(rdf_canon::run());
    report.extend(json_ld::run());
    report.extend(sparql::run());
    report.extend(rdfa::run());
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_suites_lists_every_suite_family_when_nothing_is_configured() {
        // Mirrors every runner's own `unavailable_when_env_is_unset` test: with any of the three
        // suite-root variables set, `all_suites()` performs the crate's full, heavy run (every
        // W3C/oxigraph suite this crate knows how to drive) — exactly what `tests/conformance.rs`
        // keeps behind `#[ignore]` so that `cargo test -p oxixml-testsuite` stays fast. This test
        // therefore only asserts anything when none of them are set; run with `--ignored` (see
        // `tests/conformance.rs`) to exercise `all_suites()` against real suite checkouts.
        if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some()
            || std::env::var_os(env::TESTSUITE_DIR_VAR).is_some()
            || std::env::var_os(env::RDFA_SUITE_DIR_VAR).is_some()
            // A narrowed QT3 run deliberately renames its rows to say so (`qt3::ONLY_VAR`), which
            // is exactly the shape this test asserts is absent.
            || std::env::var_os(qt3::ONLY_VAR).is_some()
        {
            return;
        }
        let report = all_suites();
        let names: Vec<&str> = report.0.iter().map(|suite| suite.suite.as_str()).collect();
        for expected in [
            "xmlconf/wellformed",
            "xsts/1.0",
            "qt3-xpath",
            "xslt30",
            "relaxng/spectest",
            "schematron",
            "c14n",
            "xmldsig",
            "rdf-tests/turtle",
            "rdf-canon",
            "json-ld/expand",
            "sparql/sparql10",
            "rdfa/html5",
        ] {
            assert!(
                names.contains(&expected),
                "missing suite row {expected:?} in {names:?}"
            );
        }
        assert!(
            report.available().next().is_none(),
            "nothing should be available without any suite root configured: {report}"
        );
    }
}
