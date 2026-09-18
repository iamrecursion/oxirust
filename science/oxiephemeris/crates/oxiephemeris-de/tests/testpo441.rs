//! Golden test against the public-domain JPL `testpo.441` file (DE441).
//!
//! DE441 (R. S. Park et al., "The JPL Planetary and Lunar Ephemerides
//! DE440 and DE441", AJ 161, 105 (2021)) is the long-timespan companion to
//! DE440: the Linux classic-binary export covers roughly −13200 to +17191
//! (JPL `linux_m13000p17000.441`), versus DE440's 1550-2650. Two things
//! this stresses that DE440/DE440t do not:
//!
//! * `NCON > 400` header constants (same tail-parsing path already
//!   exercised by `DE440t` in `tests/testpo.rs`, checked again here as a
//!   second, independent data point), and
//! * a much larger, and far more negative, record-index range: `SS(1)` is
//!   millions of days before the JD epoch, and the file holds on the order
//!   of 3 x10^5 data records. `crates/oxiephemeris-de/src/eval.rs`'s
//!   record-lookup arithmetic (`normalize_epoch`/`series_state`) is plain
//!   `f64`/`usize`, with no `i32` step, so it has no 32-bit overflow
//!   surface to begin with; this test is the empirical check that the
//!   epoch-to-record mapping is still sign-correct at this file's extreme
//!   negative span (see the header-shape assertions below and the golden
//!   run itself, which fails loudly via `DeError::Corrupt`/`EpochOutOfRange`
//!   if record lookup ever lands on the wrong record).
//!
//! The DE441 binary and `testpo.441` live under `data/de441/` (gitignored,
//! fetched by `scripts/fetch_de441.sh`); this test skips gracefully when
//! either is absent, or when the binary's download is still in progress
//! (an `.aria2` control file next to it).

mod testpo_common;

use oxiephemeris_de::DeFile;
use testpo_common::{load_binary, load_testpo, run_golden, TestResult};

/// The `testpo.441` text parser handles the full file (mirrors
/// `testpo_440_parses_over_13000_cases` in `tests/testpo.rs`): DE441's
/// testpo file spans roughly 30000 years and should hold at least as many
/// cases as DE440's ~13000-case, 1100-year file.
#[test]
fn testpo_441_parses_over_13000_cases() -> TestResult {
    let Some(cases) = load_testpo("de441", "testpo.441")? else {
        println!("skipping: data/de441/testpo.441 not present");
        return Ok(());
    };
    assert!(
        cases.len() > 13_000,
        "expected > 13000 testpo cases, parsed {}",
        cases.len()
    );
    Ok(())
}

/// Every `testpo.441` case whose epoch falls inside the DE441 binary's
/// actual coverage (`run_golden` clips to `de.span()`) must reproduce the
/// published value to <= 1e-13 (same gate as `testpo.rs`'s DE440/DE440t
/// runs).
#[test]
fn golden_de441() -> TestResult {
    let Some(bytes) = load_binary("de441", "linux_m13000p17000.441")? else {
        println!("skipping: data/de441/linux_m13000p17000.441 not present/complete");
        return Ok(());
    };
    let Some(cases) = load_testpo("de441", "testpo.441")? else {
        println!("skipping: data/de441/testpo.441 not present");
        return Ok(());
    };
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE441: {e}"))?;
    assert_eq!(de.numde(), 441, "NUMDE must be 441");
    let (start, stop) = de.span();
    println!(
        "DE441: span = [{start}, {stop}] JD, {} data records, NCON = {}, NCOEFF = {}",
        de.record_count(),
        de.constant_count(),
        de.ncoeff()
    );
    // DE441 (like DE440t) carries more than the classic 400 constant names;
    // this is the same NCON>400 tail-parsing path (`OFF_TAIL` onward in
    // `parse.rs`), exercised here over a second, independently downloaded
    // file as a cross-check on that DE440t-derived assertion.
    assert!(
        de.constant_count() > 400,
        "expected DE441 to carry NCON > 400 header constants, got {}",
        de.constant_count()
    );
    // Sanity on the record-index arithmetic at this file's extreme negative
    // span: the file's own start/stop JDs must bracket every in-span
    // testpo case, and there must be more than one data record (a
    // regression to "everything lands in record 0" would otherwise pass a
    // vacuous single-record golden run).
    assert!(
        de.record_count() > 1000,
        "expected DE441 to hold thousands of data records, got {}",
        de.record_count()
    );
    run_golden("DE441", &de, &cases)
}
