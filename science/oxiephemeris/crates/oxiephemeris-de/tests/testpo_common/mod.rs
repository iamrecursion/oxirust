//! Shared `testpo.XXX` parsing and golden-test machinery, used by both
//! `tests/testpo.rs` (DE440/DE440t) and `tests/testpo441.rs` (DE441).
//!
//! Lives under `tests/testpo_common/` (not `tests/testpo_common.rs`) so
//! Cargo's test-target autodiscovery — which only turns direct
//! `tests/*.rs` files into their own test binaries — does not also try to
//! build this file as a standalone (empty) test crate; each of the two
//! `mod testpo_common;` call sites instead compiles its own copy into its
//! own integration-test binary, the standard Rust idiom for shared test
//! helpers.
//!
//! `testpo.XXX` format (from the header comments of JPL `testeph.f`,
//! 2013-03-25): after an `EOT` marker line, each line holds
//! `de# date jed target# center# coord# value` where targets/centers are
//! 1..=9 Mercury..Pluto, 10 Moon, 11 Sun, 12 SSB, 13 EMB, 14 nutations,
//! 15 librations (and 17 TT-TDB in the `*.440t`/`*.441` variants); coordinates
//! are 1..=6 for x..zdot in AU and AU/day (radians for angles, seconds for
//! TT-TDB).
//!
//! Pass criterion (as in `testeph.f`): `|computed - expected| <= 1e-13`,
//! with the lunar-libration psi angle (target 15, coordinate 3) error
//! scaled down by `1 + 100*|jed - JDEPOC|/365.25` because that angle winds
//! by thousands of radians over the file span.
//!
//! The DE binaries live under `data/` (gitignored); every caller skips
//! gracefully when its data file is absent.
#![allow(dead_code)] // Not every constant/function is used by every caller.

use std::error::Error;
use std::path::{Path, PathBuf};

use oxiephemeris_de::DeFile;

pub type TestResult = Result<(), Box<dyn Error>>;

/// One parsed `testpo` test case.
#[derive(Debug, Clone, Copy)]
pub struct TestCase {
    pub jed: f64,
    pub target: u32,
    pub center: u32,
    pub coord: u32,
    pub value: f64,
}

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

// Column layout from the public-domain JPL `testeph.f` (2013-03-25) data
// read format:
//
// ```text
// READ(*,'(15X,D10.1,3I3,F30.20)',END=9)  ET,NTARG,NCTR,NCOORD,XI
// ```
//
// This is a **fixed-width** Fortran format, not whitespace-delimited:
// `testpo.440`/`testpo.440t` happen to carry a space between the `de#`
// prefix and the date, and between the date and the jed, but
// `testpo.441`'s wider signed 5-digit year (e.g. `441-13200.09.01-3099998.5`)
// consumes exactly that separator, leaving no whitespace there at all.
// Splitting on whitespace therefore silently breaks for DE441; slicing by
// the format's own fixed columns works for every `testpo.XXX` variant.
/// `15X`: the `de#` field plus the `date` field, skipped entirely.
const COL_SKIP_END: usize = 15;
/// `D10.1`: the `jed` field.
const COL_JED_END: usize = COL_SKIP_END + 10;
/// First `I3`: the `target#` field.
const COL_TARGET_END: usize = COL_JED_END + 3;
/// Second `I3`: the `center#` field.
const COL_CENTER_END: usize = COL_TARGET_END + 3;
/// Third `I3`: the `coord#` field. `F30.20` (the value) is the remainder
/// of the line, taken to end-of-line rather than a fixed 30 so that
/// shorter or longer trailing representations are still captured whole.
const COL_COORD_END: usize = COL_CENTER_END + 3;

/// Slices `line[start..end]`, erroring with the line number and field name
/// if the line is too short (ASCII-only content, so byte offsets are char
/// boundaries).
fn column<'x>(
    line: &'x str,
    start: usize,
    end: usize,
    field: &str,
    line_no: usize,
) -> Result<&'x str, String> {
    line.get(start..end).ok_or_else(|| {
        format!("testpo line {line_no}: too short for {field} column ({start}..{end})")
    })
}

/// Parses the whole text of a `testpo.XXX` file.
pub fn parse_testpo(text: &str) -> Result<Vec<TestCase>, String> {
    let mut lines = text.lines().enumerate();
    let mut saw_eot = false;
    for (_, line) in lines.by_ref() {
        if line.trim() == "EOT" {
            saw_eot = true;
            break;
        }
    }
    if !saw_eot {
        return Err("testpo file has no EOT marker".to_string());
    }
    let mut cases = Vec::new();
    for (line_no, line) in lines {
        if line.trim().is_empty() {
            continue;
        }
        let jed = column(line, COL_SKIP_END, COL_JED_END, "jed", line_no)?
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("testpo line {line_no}: bad jed float: {e}"))?;
        let target = column(line, COL_JED_END, COL_TARGET_END, "target", line_no)?
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("testpo line {line_no}: bad target int: {e}"))?;
        let center = column(line, COL_TARGET_END, COL_CENTER_END, "center", line_no)?
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("testpo line {line_no}: bad center int: {e}"))?;
        let coord = column(line, COL_CENTER_END, COL_COORD_END, "coord", line_no)?
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("testpo line {line_no}: bad coord int: {e}"))?;
        let value = line
            .get(COL_COORD_END..)
            .ok_or_else(|| format!("testpo line {line_no}: missing value column"))?
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("testpo line {line_no}: bad value float: {e}"))?;
        cases.push(TestCase {
            jed,
            target,
            center,
            coord,
            value,
        });
    }
    Ok(cases)
}

/// Loads a testpo file if present.
pub fn load_testpo(subdir: &str, name: &str) -> Result<Option<Vec<TestCase>>, Box<dyn Error>> {
    let path = data_dir().join(subdir).join(name);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(Some(parse_testpo(&text)?))
}

/// Loads a DE binary if it is present and its download is complete
/// (an aria2 control file next to it means "still downloading").
pub fn load_binary(subdir: &str, name: &str) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let dir = data_dir().join(subdir);
    let path = dir.join(name);
    let control = dir.join(format!("{name}.aria2"));
    if !path.exists() || control.exists() {
        return Ok(None);
    }
    Ok(Some(std::fs::read(&path)?))
}

/// Worst offenders, for diagnostics on failure.
#[derive(Debug, Clone, Copy)]
pub struct Deviation {
    pub del: f64,
    pub case: TestCase,
    pub computed: f64,
}

/// Evaluates every in-span testpo case and enforces the 1e-13 criterion.
pub fn run_golden(label: &str, de: &DeFile<'_>, cases: &[TestCase]) -> TestResult {
    // JDEPOC is carried in the header constants; testeph.f falls back to
    // the DE405-era epoch when absent.
    let jdepoc = de.constant("JDEPOC").unwrap_or(2_440_400.5);
    let (start, stop) = de.span();
    let mut deviations: Vec<Deviation> = Vec::with_capacity(cases.len());
    for case in cases {
        if case.jed < start || case.jed > stop {
            continue;
        }
        let computed = de
            .testpo_value(case.target, case.center, case.coord, (case.jed, 0.0))
            .map_err(|e| {
                format!(
                    "{label}: t={} c={} x={} jed={}: {e}",
                    case.target, case.center, case.coord, case.jed
                )
            })?;
        let mut del = (computed - case.value).abs();
        if case.target == 15 && case.coord == 3 {
            // testeph.f: DEL = DEL/(1 + 100*|ET-JDEPOC|/365.25) for the
            // accumulating libration psi angle.
            del /= 1.0 + 100.0 * (case.jed - jdepoc).abs() / 365.25;
        }
        deviations.push(Deviation {
            del,
            case: *case,
            computed,
        });
    }
    assert!(
        !deviations.is_empty(),
        "{label}: no testpo case fell inside the file span"
    );
    deviations.sort_by(|a, b| b.del.total_cmp(&a.del));
    let max_del = deviations.first().map_or(0.0, |d| d.del);
    if max_del > 1e-13 {
        eprintln!("{label}: worst 20 deviations:");
        for d in deviations.iter().take(20) {
            eprintln!(
                "  t={:>2} c={:>2} x={} jed={:.1} expected={:+.20e} computed={:+.20e} del={:.3e}",
                d.case.target,
                d.case.center,
                d.case.coord,
                d.case.jed,
                d.case.value,
                d.computed,
                d.del
            );
        }
        return Err(format!(
            "{label}: max deviation {max_del:.3e} > 1e-13 over {} cases",
            deviations.len()
        )
        .into());
    }
    println!(
        "{label}: {} testpo cases evaluated, max del = {max_del:.3e} (criterion 1e-13)",
        deviations.len()
    );
    Ok(())
}
