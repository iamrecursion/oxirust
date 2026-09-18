//! Oracle test: `oxiephemeris-core`'s Fairhead–Bretagnon–Lestrade (1988)
//! TDB−TT analytical series versus the numerically integrated TT−TDB
//! series carried by the JPL `DE440t` export (slot 15/`TPT`, `Series::TtTdb`
//! in `oxiephemeris-de`), which is the strongest published value set
//! available for this comparison (CLAUDE.md phase-1 step 3 acceptance:
//! "target < 1 µs vs published values").
//!
//! # Sign convention (verified empirically, not merely asserted)
//!
//! `oxiephemeris_core::time::tdb_minus_tt_seconds` returns **TDB − TT**
//! (see its doc comment). The `DE440t` `TPT` series (`Series::TtTdb`,
//! `testpo` target 17) is documented in `oxiephemeris-de` as "TT-TDB at the
//! geocenter" and the public-domain `testpo.440t` golden file target-17
//! lines are reproduced by `DeFile::testpo_value(17, _, 1, jd)` — the same
//! function this test calls. To pin the sign empirically (rather than by
//! reading SE/SOFA source, which is off limits) this test's author checked
//! by hand, against one `testpo.440t` target-17 line
//! (`jed = 2288350.5`, expected value `-0.00160194383019564464` s):
//!
//! ```text
//! core::tdb_minus_tt_seconds(jed)               = +0.00160225330744 s
//! testpo target-17 value at the same jed        = -0.00160194383019564464 s
//! sum (core + testpo value)                     = +3.09e-7 s
//! ```
//!
//! The sum is small (consistent with the ~100 ns truncation level the
//! Fairhead–Bretagnon–Lestrade paper claims for the full Table I near
//! J2000), confirming: **no sign flip is needed** —
//! `delta = core.tdb_minus_tt_seconds(jd) + de440t.testpo_value(17, 0, 1, jd)`
//! is the correct combination to compare against zero, because
//! `TDB - TT + TT - TDB == 0` for the same instant.
//!
//! # TT-vs-TDB argument note
//!
//! `tdb_minus_tt_seconds` is documented as being evaluated with `t` taken
//! from the TT epoch directly (the paper's argument is formally TDB, but
//! using TT changes the result by ≲ 10⁻¹⁵ s — see that function's module
//! docs). This test likewise evaluates both series at the *same* nominal
//! JD (built once, shared bit for bit between the two calls) rather than
//! attempting a TT/TDB epoch conversion for the `DE440t` lookup; the
//! resulting argument mismatch (at most the ~1.7 ms peak-to-peak TDB−TT
//! amplitude times the ~6e-13 slope difference between using TT vs TDB as
//! argument) is documented here as negligible and is not separately
//! corrected.
//!
//! The DE binary lives under `data/de440t/` (gitignored); this test skips
//! gracefully when it is absent.

use std::error::Error;
use std::path::{Path, PathBuf};

use oxiephemeris_core::time::{tdb_minus_tt_seconds, JulianDate};
use oxiephemeris_de::DeFile;

type TestResult = Result<(), Box<dyn Error>>;

/// Number of uniformly spaced sample epochs over 1900-2100 (CLAUDE.md
/// step 3 acceptance range); comfortably above the 20000 floor.
const N_SAMPLES: usize = 20_001;

/// Number of informational samples over the file's full 1550-2650 span.
const N_FULL_SPAN_SAMPLES: usize = 5_001;

/// One-microsecond acceptance bound (CLAUDE.md step 3: "target < 1 µs vs
/// published values").
const MAX_DELTA_BOUND_S: f64 = 1e-6;

/// Approximate JD of 1900-01-01 00:00 (a few hours' epoch ambiguity is
/// irrelevant here: this is a 200-year sampling window, not a calendar
/// round-trip test — see `core::time::julday`/`revjul` for exact calendar
/// conversions, which are out of this test's scope).
const JD_1900: f64 = 2_415_020.5;

/// 200 Julian years (365.25-day) later: approximate JD of 2100-01-01.
const JD_2100: f64 = JD_1900 + 200.0 * 365.25;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

/// Loads a DE binary if present and its download is complete (an aria2
/// control file next to it means "still downloading").
fn load_binary(subdir: &str, name: &str) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let dir = data_dir().join(subdir);
    let path = dir.join(name);
    let control = dir.join(format!("{name}.aria2"));
    if !path.exists() || control.exists() {
        return Ok(None);
    }
    Ok(Some(std::fs::read(&path)?))
}

/// Splits a single-`f64` JD into a `(whole_day, fraction)` two-part pair,
/// shared bit-for-bit between the `core` and `de` evaluations below so
/// neither side introduces its own rounding of the sample epoch.
fn two_part(jd: f64) -> (f64, f64) {
    let whole = jd.trunc();
    (whole, jd - whole)
}

/// Running statistics over a sequence of `|delta|` samples.
#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    max_abs: f64,
    max_abs_jed: f64,
    sum: f64,
    sum_abs: f64,
    sum_sq: f64,
    count: usize,
}

impl Stats {
    fn observe(&mut self, jed: f64, delta: f64) {
        let abs_delta = delta.abs();
        if abs_delta > self.max_abs {
            self.max_abs = abs_delta;
            self.max_abs_jed = jed;
        }
        self.sum += delta;
        self.sum_abs += abs_delta;
        self.sum_sq += delta * delta;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / count_to_f64(self.count)
        }
    }

    fn rms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_sq / count_to_f64(self.count)).sqrt()
        }
    }
}

/// Exact `usize -> f64` conversion for sample counts/indices used in this
/// test (at most tens of thousands, far under `f64`'s 2^53 exact-integer
/// range), spelled out once to keep `clippy::cast_precision_loss` honest
/// elsewhere in this file.
#[allow(clippy::cast_precision_loss)]
fn count_to_f64(n: usize) -> f64 {
    n as f64
}

/// Evaluates `delta = core.tdb_minus_tt_seconds(jd) + de.testpo_value(17, 0,
/// 1, jd)` (seconds) at `n` epochs uniformly spaced over `[start, stop]`
/// (inclusive), skipping any epoch the DE file cannot serve (only expected
/// at the extreme edges of the file's own span, if ever).
fn sample_delta(de: &DeFile<'_>, start: f64, stop: f64, n_samples: usize) -> Result<Stats, String> {
    assert!(
        n_samples >= 2,
        "need at least two samples to span [start, stop]"
    );
    let mut stats = Stats::default();
    let increment = (stop - start) / count_to_f64(n_samples - 1);
    for i in 0..n_samples {
        let jed = start + count_to_f64(i) * increment;
        let jd_pair = two_part(jed);
        let core_tdb_minus_tt = tdb_minus_tt_seconds(JulianDate::new(jd_pair.0, jd_pair.1));
        let de_tt_minus_tdb = match de.testpo_value(17, 0, 1, jd_pair) {
            Ok(v) => v,
            Err(e) => {
                // Only tolerated at the very edge of the file's own span;
                // report which epoch and error for diagnosis rather than
                // silently skipping a systematic failure.
                return Err(format!(
                    "testpo_value(17, 0, 1, jed={jed}) failed: {e} \
                     (de span = {:?})",
                    de.span()
                ));
            }
        };
        stats.observe(jed, core_tdb_minus_tt + de_tt_minus_tdb);
    }
    Ok(stats)
}

/// CLAUDE.md phase-1 step 3 acceptance: the truncated Fairhead–Bretagnon–
/// Lestrade TDB−TT series in `oxiephemeris-core` agrees with the `DE440t`
/// numerically integrated TT−TDB series to better than 1 microsecond over
/// 1900-2100, sampled at >= 20000 epochs.
#[test]
fn core_tdb_minus_tt_matches_de440t_within_one_microsecond() -> TestResult {
    let Some(bytes) = load_binary("de440t", "linux_p1550p2650.440t")? else {
        println!("skipping: data/de440t/linux_p1550p2650.440t not present/complete");
        return Ok(());
    };
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE440t: {e}"))?;
    assert!(
        de.pointer_triplet(oxiephemeris_de::Series::TtTdb).is_some(),
        "DE440t must carry the TT-TDB series"
    );

    let (span_start, span_stop) = de.span();
    // Clip the 1900-2100 window to the file's own span defensively (DE440t
    // covers 1550-2650, so this should be a no-op in practice).
    let start = JD_1900.max(span_start + 1.0);
    let stop = JD_2100.min(span_stop - 1.0);
    assert!(
        start < stop,
        "1900-2100 window does not fall inside the DE440t span {:?}",
        de.span()
    );

    let stats =
        sample_delta(&de, start, stop, N_SAMPLES).map_err(|e| format!("1900-2100 window: {e}"))?;
    println!(
        "1900-2100 window: {} samples, max|delta| = {:.3e} s at jed={:.1}, mean = {:.3e} s, rms = {:.3e} s (bound {:.0e} s)",
        stats.count, stats.max_abs, stats.max_abs_jed, stats.mean(), stats.rms(), MAX_DELTA_BOUND_S
    );

    // Informational-only: the same comparison over the file's full
    // 1550-2650 span, further from the series' expansion point (J2000)
    // where the truncated Fairhead-Bretagnon-Lestrade series is expected to
    // degrade; NOT asserted against the 1 microsecond bound.
    let full_start = span_start + 1.0;
    let full_stop = span_stop - 1.0;
    match sample_delta(&de, full_start, full_stop, N_FULL_SPAN_SAMPLES) {
        Ok(full_stats) => println!(
            "1550-2650 full-span window (informational, not asserted): {} samples, \
             max|delta| = {:.3e} s at jed={:.1}, mean = {:.3e} s, rms = {:.3e} s",
            full_stats.count,
            full_stats.max_abs,
            full_stats.max_abs_jed,
            full_stats.mean(),
            full_stats.rms()
        ),
        Err(e) => println!("1550-2650 full-span window (informational): evaluation error: {e}"),
    }

    assert!(
        stats.max_abs < MAX_DELTA_BOUND_S,
        "1900-2100: max|delta| = {:.3e} s at jed={:.1} exceeds the {:.0e} s bound \
         (mean = {:.3e} s, rms = {:.3e} s over {} samples)",
        stats.max_abs,
        stats.max_abs_jed,
        MAX_DELTA_BOUND_S,
        stats.mean(),
        stats.rms(),
        stats.count
    );
    Ok(())
}
