//! TDB − TT series tests: bounds, annual zero crossings, J2000 value, and
//! an independent cross-check against USNO Circular 179 eq. 2.6.

use oxiephemeris_core::time::{julday, tdb_minus_tt_seconds, Calendar, JulianDate, J2000_JD};
use oxiephemeris_core::CoreError;

/// Kaplan (2005), USNO Circular 179, eq. 2.6 — an independent transcription
/// of the seven dominant terms (seconds; T in Julian centuries of TT from
/// J2000.0, rates in rad/century). Used only as a cross-check oracle.
fn usno_circular_179_eq_2_6(jd_tt: JulianDate) -> f64 {
    let t = (jd_tt.value() - J2000_JD) / 36_525.0;
    0.001_657 * (628.3076 * t + 6.2401).sin()
        + 0.000_022 * (575.3385 * t + 4.2970).sin()
        + 0.000_014 * (1256.6152 * t + 6.1969).sin()
        + 0.000_005 * (606.9777 * t + 4.0212).sin()
        + 0.000_005 * (52.9691 * t + 0.4444).sin()
        + 0.000_002 * (21.3299 * t + 5.5431).sin()
        + 0.000_010 * t * (628.3076 * t + 4.2490).sin()
}

#[test]
fn bounded_by_two_ms_and_crosses_zero_every_year_1600_2400() -> Result<(), CoreError> {
    for year in 1600..=2400 {
        let start = julday(Calendar::Gregorian, year, 1, 1, 0.0)?;
        let mut previous: Option<f64> = None;
        let mut sign_changes = 0u32;
        for step in 0..48 {
            // 48 samples x 7.6 d covers the year with dense margin.
            let jd = start.add_days(f64::from(step) * 7.6);
            let x = tdb_minus_tt_seconds(jd);
            assert!(x.abs() <= 2e-3, "|TDB-TT| = {x} > 2 ms in year {year}");
            if let Some(p) = previous {
                if (p < 0.0) != (x < 0.0) {
                    sign_changes += 1;
                }
            }
            previous = Some(x);
        }
        assert!(sign_changes >= 1, "no zero crossing found in year {year}");
    }
    Ok(())
}

#[test]
fn j2000_value_in_published_band() {
    // The leading terms give TDB-TT(J2000.0) ~ -96 us (dominant annual term
    // near a descending node at the epoch).
    let x = tdb_minus_tt_seconds(JulianDate::from_f64(J2000_JD));
    assert!(
        (-120e-6..=-20e-6).contains(&x),
        "TDB-TT at J2000.0 = {x} s outside [-120, -20] us"
    );
}

#[test]
fn agrees_with_usno_circular_179_truncation() -> Result<(), CoreError> {
    // Everything the 7-term eq. 2.6 omits sums to a few tens of us at most
    // over 1600-2400, so the full 123-term series must stay within 50 us of
    // it. This catches transcription blunders (wrong units, rates, signs)
    // in the dominant terms.
    for year in (1600..=2400).step_by(7) {
        for month in [1u8, 4, 7, 10] {
            let jd = julday(Calendar::Gregorian, year, month, 15, 7.5)?;
            let full = tdb_minus_tt_seconds(jd);
            let truncated = usno_circular_179_eq_2_6(jd);
            assert!(
                (full - truncated).abs() < 5e-5,
                "{year}-{month:02}: full {full} vs eq. 2.6 {truncated}"
            );
        }
    }
    Ok(())
}

#[test]
fn two_part_input_is_honored() {
    // Splitting the epoch across hi/lo must not change the result
    // significantly (two-part-aware argument reduction).
    let a = tdb_minus_tt_seconds(JulianDate::from_f64(J2000_JD + 123.456));
    let b = tdb_minus_tt_seconds(JulianDate::new(J2000_JD, 123.456));
    assert!((a - b).abs() < 1e-12);
}
