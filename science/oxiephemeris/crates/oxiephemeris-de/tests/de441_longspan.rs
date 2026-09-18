//! Long-span property tests for DE441 at BCE epochs (deeply negative JDs).
//!
//! `tests/testpo441.rs` already checks DE441 against JPL's own published
//! values (an *oracle*). This file checks internal-consistency *laws* that
//! must hold at any epoch, exercised specifically over the BCE half of the
//! span (JD ≈ −3.1e6 … +1.72e6), where sign errors in the epoch-to-record
//! arithmetic, two-part JD handling, or calendar conversion would surface:
//!
//! 1. **Split invariance** — the same epoch expressed as different
//!    `(hi, lo)` two-part JDs must produce the same state.
//! 2. **Record-boundary continuity** — states extrapolated onto a record
//!    boundary from both sides must agree (a wrong-record lookup is a
//!    ~10⁵ km error; the Chebyshev fit's own boundary mismatch is ≪ 1 m).
//! 3. **Rate = d(position)/dt** — the evaluated Chebyshev rate must match
//!    a central difference of positions.
//! 4. **Physical sanity** — Mercury's heliocentric and the Moon's
//!    geocentric distances stay inside their orbital bounds across the
//!    whole BCE span (over ±13 kyr, secular drift of both is negligible
//!    next to the bands used here).
//! 5. **Calendar round-trip** — `revjul` ∘ `julday` is the identity over
//!    the full file span, for both proleptic calendars, and the span
//!    endpoints land in the documented years (−13200 / +17191).
//! 6. **Span edges** — evaluation succeeds exactly at `SS(1)`/`SS(2)` and
//!    fails with `EpochOutOfRange` just outside.
//!
//! Everything runs inside a single `#[test]` so the ~2.6 GB DE441 binary
//! is read into memory once; `cargo nextest` runs each test in its own
//! process, and several concurrent copies of the file would be a real
//! memory hazard. The test skips gracefully when the binary is absent.

mod testpo_common;

use oxiephemeris_core::time::{julday, revjul, Calendar, JulianDate};
use oxiephemeris_de::{Body, DeError, DeFile, Series};
use testpo_common::{load_binary, TestResult};

/// Bodies whose SSB states drive the split/boundary/rate properties:
/// the fastest planet, the EMB (the Earth-based pipelines' workhorse),
/// and the barycentric Moon (which exercises the EMB−Moon combination).
const PROBE_BODIES: [Body; 3] = [Body::Mercury, Body::Emb, Body::Moon];

/// Julian date of 1 CE Jan 1.5 (proleptic Julian calendar) — everything
/// below this is "BCE" for sampling purposes.
const BCE_END_JD: f64 = 1_721_423.5;

fn norm3(v: &[f64]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn de441_longspan_bce_properties() -> TestResult {
    let Some(bytes) = load_binary("de441", "linux_m13000p17000.441")? else {
        println!("skipping: data/de441/linux_m13000p17000.441 not present/complete");
        return Ok(());
    };
    let de = DeFile::parse(&bytes).map_err(|e| format!("parse DE441: {e}"))?;
    let (start, stop) = de.span();
    // The `linux_m13000p17000.441` export: SS = [-3027215.5, 7930192.5],
    // i.e. years ~-13000 to ~+17000 (testpo.441 reaches a little beyond
    // this; the golden test clips to the binary's span).
    assert!(
        start < -3.0e6 && stop > 7.9e6,
        "DE441 span [{start}, {stop}] is not the expected ±15 kyr file"
    );
    split_invariance(&de, start)?;
    record_boundary_continuity(&de, start, stop)?;
    rate_matches_central_difference(&de, start)?;
    physical_sanity(&de, start)?;
    calendar_round_trip(start, stop)?;
    span_edges(&de, start, stop);
    Ok(())
}

/// Property 1: `series_state` sees only the *real number* `hi + lo`, not
/// the split. All parts below are exactly representable and sum exactly,
/// so `normalize_epoch` must produce identical `(whole, frac)` pairs and
/// the states must agree to the last bit; the gate leaves a little slack
/// so a future (still correct) renormalization does not fail it.
fn split_invariance(de: &DeFile<'_>, start: f64) -> TestResult {
    // Quarter-day-aligned BCE epochs: exact under ±0.25/±1000-day resplits.
    for k in 0..20 {
        let base = start + 8.25 + 150_000.25 * f64::from(k);
        let splits = [
            (base, 0.0),
            (base - 0.25, 0.25),
            (base - 0.75, 0.75),
            (base - 1000.0, 1000.0),
            (base + 512.5, -512.5),
        ];
        for body in PROBE_BODIES {
            let reference = de
                .state_km(body, splits[0])
                .map_err(|e| format!("state at jd {base}: {e}"))?;
            for split in &splits[1..] {
                let state = de
                    .state_km(body, *split)
                    .map_err(|e| format!("state at split {split:?}: {e}"))?;
                for (c, (a, b)) in reference.iter().zip(&state).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "{body:?} jd {base} split {split:?} coord {c}: {a} vs {b}"
                    );
                }
            }
        }
    }
    println!("split invariance: 20 BCE epochs x 5 splits x 3 bodies OK");
    Ok(())
}

/// Property 2: extrapolating the state onto a record boundary from ε
/// before and ε after must agree. With ε = 10⁻⁷ day the neglected
/// acceleration term is ≲ 10⁻¹¹ km even for Mercury, so any disagreement
/// is the (≪ 1 m) fit mismatch of adjacent records — or a record-lookup
/// bug, which is instead a ~10⁵ km disagreement.
fn record_boundary_continuity(de: &DeFile<'_>, start: f64, stop: f64) -> TestResult {
    let record_count = u32::try_from(de.record_count()).map_err(|e| format!("nrec: {e}"))?;
    let record_days = (stop - start) / f64::from(record_count);
    assert!(
        (record_days - record_days.round()).abs() < 1e-9,
        "record length {record_days} days is not integral"
    );
    let mut boundaries = 0u32;
    let mut worst_km = 0.0f64;
    let eps_days = 1e-7;
    // Every 2999th interior boundary (co-prime with nothing relevant, just
    // a stride that spreads ~50 samples over the ~150k BCE records).
    let mut k = 2_999u32;
    loop {
        let jd_b = start + record_days * f64::from(k);
        if jd_b >= BCE_END_JD || k >= record_count {
            break;
        }
        for body in PROBE_BODIES {
            let before = de
                .state_km(body, (jd_b, -eps_days))
                .map_err(|e| format!("state before boundary {jd_b}: {e}"))?;
            let after = de
                .state_km(body, (jd_b, eps_days))
                .map_err(|e| format!("state after boundary {jd_b}: {e}"))?;
            let mut gap = [0.0f64; 3];
            for c in 0..3 {
                let from_below = before[c] + before[c + 3] * eps_days;
                let from_above = after[c] - after[c + 3] * eps_days;
                gap[c] = from_below - from_above;
            }
            let gap_km = norm3(&gap);
            worst_km = worst_km.max(gap_km);
            assert!(
                gap_km < 0.01,
                "{body:?} boundary jd {jd_b}: sides disagree by {gap_km} km"
            );
        }
        boundaries += 1;
        k += 2_999;
    }
    assert!(boundaries > 40, "only {boundaries} BCE boundaries sampled");
    println!("boundary continuity: {boundaries} BCE boundaries, worst gap {worst_km:.3e} km");
    Ok(())
}

/// Property 3: the evaluated rate is the derivative of the evaluated
/// position. Central difference with h = 0.01 day has an h²/6·f''' error
/// of ≲ 0.4 km/day even for Mercury (ω³r ≈ 2×10⁴ km/day³), i.e. ≲ 10⁻⁷
/// of the velocity itself; the gate is 100× looser.
fn rate_matches_central_difference(de: &DeFile<'_>, start: f64) -> TestResult {
    let h = 0.01;
    for k in 0..25 {
        let jd = start + 100.5 + 190_000.25 * f64::from(k);
        if jd >= BCE_END_JD {
            break;
        }
        for body in PROBE_BODIES {
            let state = de
                .state_km(body, (jd, 0.0))
                .map_err(|e| format!("state at jd {jd}: {e}"))?;
            let plus = de
                .state_km(body, (jd, h))
                .map_err(|e| format!("state at jd {jd}+h: {e}"))?;
            let minus = de
                .state_km(body, (jd, -h))
                .map_err(|e| format!("state at jd {jd}-h: {e}"))?;
            let speed = norm3(&state[3..6]);
            for c in 0..3 {
                let numeric = (plus[c] - minus[c]) / (2.0 * h);
                let dv = (numeric - state[c + 3]).abs();
                assert!(
                    dv < 1e-5 * speed + 0.5,
                    "{body:?} jd {jd} coord {c}: rate {} vs central diff {numeric}",
                    state[c + 3]
                );
            }
        }
    }
    println!("rate vs central difference: 25 BCE epochs x 3 bodies OK");
    Ok(())
}

/// Property 4: orbital-scale sanity across the whole BCE span. Mercury's
/// heliocentric distance stays within its (perihelion, aphelion) band and
/// the geocentric Moon inside its (perigee, apogee) band — a wrong-record
/// or wrong-granule lookup that survived the bracket check would blow
/// straight through these.
fn physical_sanity(de: &DeFile<'_>, start: f64) -> TestResult {
    let au_km = de.constant("AU").ok_or("DE441 header lacks AU")?;
    let stride = (BCE_END_JD - start - 40.0) / 200.0;
    for k in 0..200 {
        let jd = (start + 20.5 + stride * f64::from(k), 0.0);
        let mercury = de
            .state_km(Body::Mercury, jd)
            .map_err(|e| format!("Mercury at {jd:?}: {e}"))?;
        let sun = de
            .state_km(Body::Sun, jd)
            .map_err(|e| format!("Sun at {jd:?}: {e}"))?;
        let helio = [
            mercury[0] - sun[0],
            mercury[1] - sun[1],
            mercury[2] - sun[2],
        ];
        let r_au = norm3(&helio) / au_km;
        assert!(
            (0.29..0.48).contains(&r_au),
            "Mercury heliocentric {r_au} AU at jd {jd:?}"
        );
        let moon = de
            .series_state(Series::Moon, jd)
            .map_err(|e| format!("geocentric Moon at {jd:?}: {e}"))?;
        let r_km = norm3(&moon.value);
        assert!(
            (350_000.0..410_000.0).contains(&r_km),
            "geocentric Moon {r_km} km at jd {jd:?}"
        );
    }
    println!("physical sanity: 200 BCE epochs OK (Mercury + geocentric Moon)");
    Ok(())
}

/// Property 5: `revjul` ∘ `julday` is the identity over the full DE441
/// span for both proleptic calendars, and the endpoints land in the years
/// the export's file name promises (~−13000 and ~+17000, astronomical
/// numbering).
fn calendar_round_trip(start: f64, stop: f64) -> TestResult {
    let mut samples = 0u32;
    let mut jd = start;
    while jd <= stop {
        for calendar in [Calendar::Gregorian, Calendar::Julian] {
            let (date, hours) = revjul(JulianDate::from_f64(jd), calendar)
                .map_err(|e| format!("revjul {jd} {calendar:?}: {e}"))?;
            let back = julday(calendar, date.year, date.month, date.day, hours)
                .map_err(|e| format!("julday {date:?} {calendar:?}: {e}"))?;
            let residual = (back.value() - jd).abs();
            assert!(
                residual < 1e-8,
                "{calendar:?} round trip at jd {jd}: {date:?} + {hours} h off by {residual} days"
            );
        }
        samples += 1;
        jd += 1013.25;
    }
    assert!(samples > 10_000, "only {samples} round-trip samples");
    for (jd, expected_year) in [(start, -13_000), (stop, 17_000)] {
        let (date, _) = revjul(JulianDate::from_f64(jd), Calendar::Gregorian)
            .map_err(|e| format!("revjul {jd}: {e}"))?;
        assert!(
            (date.year - expected_year).abs() <= 2,
            "span endpoint jd {jd} maps to year {}, expected ~{expected_year}",
            date.year
        );
    }
    println!("calendar round trip: {samples} epochs x 2 calendars over the full span OK");
    Ok(())
}

/// Property 6: the span edges themselves evaluate (`SS(2)` via the
/// last-record clamp), and one day outside either edge is
/// `EpochOutOfRange` — not a wrong record, not a panic.
fn span_edges(de: &DeFile<'_>, start: f64, stop: f64) {
    for jd in [start, stop] {
        assert!(
            de.state_km(Body::Emb, (jd, 0.0)).is_ok(),
            "span edge jd {jd} must evaluate"
        );
    }
    for jd in [start - 1.0, stop + 1.0] {
        assert!(
            matches!(
                de.state_km(Body::Emb, (jd, 0.0)),
                Err(DeError::EpochOutOfRange)
            ),
            "jd {jd} outside the span must be EpochOutOfRange"
        );
    }
    println!("span edges: in-range at SS(1)/SS(2), EpochOutOfRange outside OK");
}
