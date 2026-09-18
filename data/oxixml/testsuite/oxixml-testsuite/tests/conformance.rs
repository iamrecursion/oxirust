//! Suite-driven integration tests: real ("heavy") runs of every conformance suite
//! `oxixml-testsuite` knows how to drive.
//!
//! Every test below is marked `#[ignore]` and skips instantly (printing why, to stderr) when its
//! suite's environment variable is unset, so a plain `cargo test -p oxixml-testsuite` never
//! attempts to walk a multi-thousand-case W3C suite and stays fast. Run
//!
//! ```text
//! export OXIXML_XML_SUITES_DIR=/path/to/w3c-xml-suites
//! export OXIXML_TESTSUITE_DIR=/path/to/oxigraph/testsuite
//! cargo test -p oxixml-testsuite -- --ignored --nocapture
//! ```
//!
//! to drive every suite for real and see each one's pass rate.
//!
//! These are deliberately not zero-failure gates: several of the suites here (QT3/FOTS, the XSD
//! test suite, `xslt30-test`, …) are enormous W3C conformance corpora that even mature, long-lived
//! processors do not clear at 100%. Measuring and reporting that number honestly is
//! `oxixml-testsuite`'s whole job — see the `oxixml-conformance-report` binary and the
//! `README`'s conformance section — not failing CI every time a documented gap remains. What each
//! test *does* assert is that the harness itself works end to end: the suite was discovered
//! ([`SuiteReport::is_available`]) and at least one case actually ran (`total > 0`). A regression
//! in either would mean the discovery or manifest-parsing logic broke, silently turning a suite
//! run into a no-op that a pass-rate-only report could not tell apart from "nothing to do".
//!
//! # Where a suite *is* a gate
//!
//! "Not a zero-failure gate" is a statement about the pass *rate*, and it stops being an excuse
//! the moment a suite's failures are enumerated. Three of them are enumerated, so three of them
//! are gates on their exact residue rather than on a percentage:
//!
//! * `xslt30` — every failing case is named in [`xslt30::ledger`] with a class and its evidence,
//!   and the run asserts the failing set is **exactly** that table, in both directions. A case
//!   that starts failing is a regression; a case that stops failing and is left in the table is a
//!   stale claim, and both fail the same way.
//! * `xsts` — the XML Schema suite's residue is pinned per slot in [`xsts::ledger`],
//!   and `oxixml-schema`'s own runner pins the same residue at slot granularity.
//! * `qt3` — pinned at **zero wrong answers**. Not at zero failures: a QT3 case whose cost is a
//!   large fraction of the per-case budget measures the machine as much as the processor, so a
//!   loaded machine produces timeouts that say nothing about conformance. The number quoted for
//!   release is taken at a budget that removes that variable (`OXIXML_QT3_CASE_BUDGET_SECS=180`),
//!   and what is *asserted* here is the part that is the processor's: no case answers wrongly.

use oxixml_testsuite::report::SuiteReport;
use oxixml_testsuite::{
    c14n, env, json_ld, qt3, rdf_canon, rdf_tests, rdfa, relaxng, schematron, sparql, xmlconf,
    xmldsig, xslt30, xsts,
};

/// Prints every report (so `--nocapture` shows the pass rate) and asserts the two harness-health
/// invariants common to every suite-driven test: the suite was found, and running it was not a
/// silent no-op.
fn assert_ran(reports: &[SuiteReport]) {
    assert!(!reports.is_empty());
    for report in reports {
        println!("{report}");
        assert!(
            report.is_available(),
            "{report} should have been discovered"
        );
        assert!(
            report.total > 0,
            "{}: discovered the suite but ran zero cases",
            report.suite
        );
        for failure in report.failures.iter().take(10) {
            println!("  ! {failure}");
        }
    }
}

/// `true` and prints a reason to stderr when `var` is unset — the shared "skip this heavy test"
/// check every test below opens with.
fn skip_unless_set(var: &str) -> bool {
    if std::env::var_os(var).is_none() {
        eprintln!("skipping: {var} is not set");
        return true;
    }
    false
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn xmlconf_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    assert_ran(&xmlconf::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn xsts_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    let reports = xsts::run();
    assert_ran(&reports);
    // Both passes are reported before either is asserted on: a run that took two hours must say
    // everything it found, not stop at the first thing.
    let mut failed = false;
    for report in &reports {
        let verdict = xsts::ledger::audit(report);
        println!("ledger[{}]: {verdict}", report.suite);
        failed |= verdict.is_a_gate_failure();
    }
    assert!(!failed, "the failing slots are not the expected residue");
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn qt3_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    let reports = qt3::run();
    assert_ran(&reports);
    for report in &reports {
        let wrong = qt3::wrong_answers(report);
        for failure in &wrong {
            println!("  wrong answer: {failure}");
        }
        assert!(
            wrong.is_empty(),
            "{}: {} case(s) answered wrongly — a QT3 failure that is not a timeout is a \
             conformance regression",
            report.suite,
            wrong.len()
        );
    }
}

/// The whole `xslt30-test` catalogue, gated on its failing set being *exactly* the expected-failure
/// ledger.
///
/// The two directions are both checked and both fail the run, which is the whole reason the ledger
/// is worth keeping: see [`xslt30::ledger`] for why a stale entry is treated as seriously as a
/// regression, and for the two borderline cases (`catalog-003`, `catalog-007`) that are
/// deliberately absent from it because they measure the machine rather than the processor.
#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn xslt30_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    let reports = xslt30::run();
    assert_ran(&reports);
    for report in &reports {
        let verdict = xslt30::ledger::audit(report);
        println!("ledger: {verdict}");
        assert!(!verdict.is_a_gate_failure(), "{verdict}");
    }
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn relaxng_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    assert_ran(&relaxng::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn schematron_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    assert_ran(&schematron::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn c14n_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    assert_ran(&c14n::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR"]
fn xmldsig_runs_against_the_real_suite() {
    if skip_unless_set(env::XML_SUITES_DIR_VAR) {
        return;
    }
    assert_ran(&xmldsig::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_TESTSUITE_DIR"]
fn rdf_tests_runs_against_the_real_suite() {
    if skip_unless_set(env::TESTSUITE_DIR_VAR) {
        return;
    }
    assert_ran(&rdf_tests::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_TESTSUITE_DIR"]
fn rdf_canon_runs_against_the_real_suite() {
    if skip_unless_set(env::TESTSUITE_DIR_VAR) {
        return;
    }
    assert_ran(&rdf_canon::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_TESTSUITE_DIR"]
fn json_ld_runs_against_the_real_suite() {
    if skip_unless_set(env::TESTSUITE_DIR_VAR) {
        return;
    }
    assert_ran(&json_ld::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_TESTSUITE_DIR"]
fn sparql_runs_against_the_real_suite() {
    if skip_unless_set(env::TESTSUITE_DIR_VAR) {
        return;
    }
    assert_ran(&sparql::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_RDFA_SUITE_DIR"]
fn rdfa_runs_against_the_real_suite() {
    if skip_unless_set(env::RDFA_SUITE_DIR_VAR) {
        return;
    }
    assert_ran(&rdfa::run());
}

#[test]
#[ignore = "heavy: needs OXIXML_XML_SUITES_DIR and/or OXIXML_TESTSUITE_DIR"]
fn all_suites_prints_the_full_conformance_table() {
    if std::env::var_os(env::XML_SUITES_DIR_VAR).is_none()
        && std::env::var_os(env::TESTSUITE_DIR_VAR).is_none()
    {
        eprintln!(
            "skipping: neither {} nor {} is set",
            env::XML_SUITES_DIR_VAR,
            env::TESTSUITE_DIR_VAR
        );
        return;
    }
    let report = oxixml_testsuite::all_suites();
    println!("{report}");
    assert!(!report.0.is_empty());
}
