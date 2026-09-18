//! `oxixml-conformance-report` — runs every conformance suite [`oxixml_testsuite`] knows how to
//! drive and prints one aligned table of the results.
//!
//! This is the same [`oxixml_testsuite::all_suites`] the crate's own `#[ignore]`d integration
//! tests call; the binary exists so CI (and anyone at a terminal) can get the whole workspace's
//! conformance picture — including which suites are not even provisioned — without running
//! `cargo test -- --ignored` and picking through the output.
//!
//! # Usage
//!
//! ```text
//! export OXIXML_XML_SUITES_DIR=/path/to/w3c-xml-suites
//! export OXIXML_TESTSUITE_DIR=/path/to/oxigraph/testsuite
//! export OXIXML_RDFA_SUITE_DIR=/path/to/rdfa-suite   # optional
//! cargo run -p oxixml-testsuite --bin oxixml-conformance-report --release
//! ```
//!
//! Every variable is optional: suites whose variable is unset (or whose expected layout is not
//! found under it) print as `unavailable` rather than being skipped from the table, so the report
//! always names the crate's full intended suite coverage.
//!
//! The process exit code is always `0` — this binary reports the workspace's current conformance
//! level, which is expected to include partial suites while the workspace is under active
//! development; it is not a pass/fail gate. A CI job that wants a hard gate on a specific suite
//! should instead run that suite's own `#[ignore]`d `#[test]` (see the crate's `tests/`
//! directory), which fails the build on a regression.

fn main() {
    let started = std::time::Instant::now();
    println!("OxiXML workspace conformance report");
    println!("====================================\n");

    let report = oxixml_testsuite::all_suites();
    print!("{report}");

    // A skip is not a pass, and the table's single `Skipped` column cannot say whether a case was
    // skipped because the suite declares a dependency this workspace does not implement or
    // because the runner could not adjudicate it. The breakdown says which, so that the second
    // kind — a verdict nobody knows — stays visible instead of being averaged away.
    let skips = report.skip_breakdown();
    if !skips.is_empty() {
        println!("\nWhy cases were skipped");
        println!("----------------------");
        print!("{skips}");
    }

    println!(
        "\n(discovered and ran in {:.2}s)",
        started.elapsed().as_secs_f64()
    );
}
