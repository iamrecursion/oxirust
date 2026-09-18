//! Golden tests against the public-domain JPL `testpo` files (DE440,
//! `DE440t`). DE441 has its own test file, `tests/testpo441.rs`, sharing the
//! parsing/golden-run machinery in `tests/testpo_common/mod.rs`.

mod testpo_common;

use oxiephemeris_de::{DeFile, Series};
use testpo_common::{load_binary, load_testpo, run_golden, TestResult};

/// CLAUDE.md phase-1 step 1: the testpo text parser handles the full file.
#[test]
fn testpo_440_parses_over_13000_cases() -> TestResult {
    let Some(cases) = load_testpo("de440", "testpo.440")? else {
        println!("skipping: data/de440/testpo.440 not present");
        return Ok(());
    };
    assert!(
        cases.len() > 13_000,
        "expected > 13000 testpo cases, parsed {}",
        cases.len()
    );
    let first = cases.first().ok_or("no cases")?;
    assert!((first.jed - 2_287_195.5).abs() < 1e-9);
    assert_eq!(first.target, 14);
    assert_eq!(first.center, 0);
    assert_eq!(first.coord, 1);
    assert!((first.value - 0.000_015_727_411_719_797_6).abs() < 1e-24);
    Ok(())
}

/// CLAUDE.md phase-1 step 5 exit criterion: DE440 fully green at <= 1e-13 AU.
#[test]
fn golden_de440() -> TestResult {
    let Some(bytes) = load_binary("de440", "linux_p1550p2650.440")? else {
        println!("skipping: data/de440/linux_p1550p2650.440 not present/complete");
        return Ok(());
    };
    let Some(cases) = load_testpo("de440", "testpo.440")? else {
        println!("skipping: data/de440/testpo.440 not present");
        return Ok(());
    };
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE440: {e}"))?;
    // Never hardcoded in the library: DE440's record shape, asserted here.
    assert_eq!(de.ncoeff(), 1018, "DE440 must have NCOEFF = KSIZE/2 = 1018");
    assert_eq!(de.numde(), 440);
    assert!(
        de.pointer_triplet(Series::TtTdb).is_none(),
        "plain DE440 carries no TT-TDB series"
    );
    run_golden("DE440", &de, &cases)
}

/// Same golden run over the `DE440t` variant, which exercises `NCON > 400`
/// extra constant names plus the RPT/TPT pointers and the TT-TDB series.
#[test]
fn golden_de440t() -> TestResult {
    let Some(bytes) = load_binary("de440t", "linux_p1550p2650.440t")? else {
        println!("skipping: data/de440t/linux_p1550p2650.440t not present/complete");
        return Ok(());
    };
    let Some(cases) = load_testpo("de440t", "testpo.440t")? else {
        println!("skipping: data/de440t/testpo.440t not present");
        return Ok(());
    };
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE440t: {e}"))?;
    assert_eq!(de.numde(), 440);
    assert!(
        de.pointer_triplet(Series::TtTdb).is_some(),
        "DE440t must carry the TT-TDB series"
    );
    assert!(
        cases.iter().any(|c| c.target == 17),
        "testpo.440t must exercise TT-TDB (target 17)"
    );
    run_golden("DE440t", &de, &cases)
}
