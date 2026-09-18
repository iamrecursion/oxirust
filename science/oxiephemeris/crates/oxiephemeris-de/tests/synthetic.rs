//! Unit tests on synthetic classic-binary DE files built in memory:
//! known low-order Chebyshev polynomials must interpolate exactly (position
//! and velocity), and a byte-swapped file must parse identically via
//! endianness detection.
//!
//! The synthetic layout follows the record structure written by the
//! public-domain JPL `asc2eph.f` (see `src/parse.rs` for the byte map).

use std::error::Error;

use oxiephemeris_de::{Body, DeError, DeFile, Endianness, Series, SeriesState};

type TestResult = Result<(), Box<dyn Error>>;

/// File span: two 32-day records.
const SS: [f64; 3] = [2_451_536.5, 2_451_600.5, 32.0];
const NREC: usize = 2;
/// Doubles per record: 2 (span) + Mercury 3*15*8 + (EMB, Moon, Sun) 3*5*2.
const NCOEFF: usize = 452;
const RECLEN: usize = NCOEFF * 8;
const AU_KM: f64 = 149_597_870.7;
const EMRAT: f64 = 81.300_568_2;

/// Synthetic series: `(slot, start_1based, ncf, na)`.
const SPECS: [(u32, usize, usize, usize); 4] = [
    (0, 3, 15, 8),   // Mercury
    (2, 363, 5, 2),  // Earth-Moon barycenter
    (9, 393, 5, 2),  // Moon (geocentric)
    (10, 423, 5, 2), // Sun
];

/// Deterministic quadratic Chebyshev coefficients `[a0, a1, a2]` for a
/// given (series slot, record, granule, component).
fn poly(slot: u32, rec: u32, gran: u32, comp: u32) -> [f64; 3] {
    let s = f64::from(slot);
    let r = f64::from(rec);
    let g = f64::from(gran);
    let c = f64::from(comp);
    [
        1.5 + 3.0 * s + 10.0 * c + r,
        -2.25 + 0.5 * g + c - 0.125 * s,
        0.375 + 0.25 * r + 0.125 * c + 0.062_5 * g,
    ]
}

/// Closed-form position of the quadratic: `a0*T0 + a1*T1(tc) + a2*T2(tc)`.
fn closed_value(slot: u32, rec: u32, gran: u32, comp: u32, tc: f64) -> f64 {
    let a = poly(slot, rec, gran, comp);
    a[0] + a[1] * tc + a[2] * (2.0 * tc * tc - 1.0)
}

/// Closed-form velocity per day: `d/dtc * (2*na/span_days)`.
fn closed_rate(slot: u32, rec: u32, gran: u32, comp: u32, tc: f64, na: usize) -> f64 {
    let a = poly(slot, rec, gran, comp);
    let na_f = f64::from(u32::try_from(na).unwrap_or(0));
    (a[1] + 4.0 * a[2] * tc) * (2.0 * na_f / SS[2])
}

fn put_bytes(buf: &mut [u8], off: usize, bytes: &[u8]) -> Result<(), String> {
    buf.get_mut(off..off + bytes.len())
        .ok_or_else(|| format!("synthetic builder: write at {off} out of bounds"))?
        .copy_from_slice(bytes);
    Ok(())
}

fn put_f64(buf: &mut [u8], off: usize, v: f64, big: bool) -> Result<(), String> {
    let b = if big {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    };
    put_bytes(buf, off, &b)
}

fn put_i32(buf: &mut [u8], off: usize, v: i32, big: bool) -> Result<(), String> {
    let b = if big {
        v.to_be_bytes()
    } else {
        v.to_le_bytes()
    };
    put_bytes(buf, off, &b)
}

fn put_i32_usize(buf: &mut [u8], off: usize, v: usize, big: bool) -> Result<(), String> {
    put_i32(buf, off, i32::try_from(v).map_err(|e| e.to_string())?, big)
}

/// Builds a complete synthetic classic-binary DE file.
fn build_synthetic(big: bool) -> Result<Vec<u8>, String> {
    let mut buf = vec![0_u8; (NREC + 2) * RECLEN];
    // TTL and CNAM regions are blank-padded ASCII.
    for byte in buf
        .get_mut(0..2652)
        .ok_or_else(|| "header region out of bounds".to_string())?
    {
        *byte = b' ';
    }
    put_bytes(&mut buf, 252, b"AU    ")?;
    put_bytes(&mut buf, 258, b"EMRAT ")?;
    put_bytes(&mut buf, 264, b"DENUM ")?;
    put_f64(&mut buf, 2652, SS[0], big)?;
    put_f64(&mut buf, 2660, SS[1], big)?;
    put_f64(&mut buf, 2668, SS[2], big)?;
    put_i32(&mut buf, 2676, 3, big)?; // NCON
    put_f64(&mut buf, 2680, AU_KM, big)?;
    put_f64(&mut buf, 2688, EMRAT, big)?;
    for (slot, start, ncf, na) in SPECS {
        let off = 2696 + 12 * usize::try_from(slot).map_err(|e| e.to_string())?;
        put_i32_usize(&mut buf, off, start, big)?;
        put_i32_usize(&mut buf, off + 4, ncf, big)?;
        put_i32_usize(&mut buf, off + 8, na, big)?;
    }
    put_i32(&mut buf, 2840, 440, big)?; // NUMDE
                                        // LPT (2844), RPT (2856), TPT (2868) stay zero: absent series.

    // Record 2: constant values.
    put_f64(&mut buf, RECLEN, AU_KM, big)?;
    put_f64(&mut buf, RECLEN + 8, EMRAT, big)?;
    put_f64(&mut buf, RECLEN + 16, 440.0, big)?;

    // Data records.
    for rec in 0..NREC {
        let rec_off = (2 + rec) * RECLEN;
        let t0 = SS[2].mul_add(
            f64::from(u32::try_from(rec).map_err(|e| e.to_string())?),
            SS[0],
        );
        put_f64(&mut buf, rec_off, t0, big)?;
        put_f64(&mut buf, rec_off + 8, t0 + SS[2], big)?;
        for (slot, start, ncf, na) in SPECS {
            for gran in 0..na {
                for comp in 0..3_usize {
                    let coeffs = poly(
                        slot,
                        u32::try_from(rec).map_err(|e| e.to_string())?,
                        u32::try_from(gran).map_err(|e| e.to_string())?,
                        u32::try_from(comp).map_err(|e| e.to_string())?,
                    );
                    let base = (start - 1) + gran * ncf * 3 + comp * ncf;
                    for (k, &a) in coeffs.iter().enumerate() {
                        put_f64(&mut buf, rec_off + 8 * (base + k), a, big)?;
                    }
                }
            }
        }
    }
    Ok(buf)
}

/// JD of the point at normalized argument `tc` inside (record, granule) of
/// a series with `na` granules per record.
fn jd_at(rec: u32, gran: u32, tc: f64, na: usize) -> f64 {
    let gran_len = SS[2] / f64::from(u32::try_from(na).unwrap_or(1));
    SS[0] + SS[2] * f64::from(rec) + gran_len * f64::from(gran) + gran_len * (tc + 1.0) / 2.0
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{what}: {a} vs {b} (diff {:.3e} > {tol:.1e})",
        (a - b).abs()
    );
}

#[test]
fn header_fields_round_trip() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    assert_eq!(de.endianness(), Endianness::Little);
    let (start, stop) = de.span();
    assert_close(start, SS[0], 0.0, "span start");
    assert_close(stop, SS[1], 0.0, "span stop");
    assert_close(de.step_days(), SS[2], 0.0, "step");
    assert_close(de.au_km(), AU_KM, 0.0, "AU");
    assert_close(de.emrat(), EMRAT, 0.0, "EMRAT");
    assert_eq!(de.numde(), 440);
    assert_eq!(de.ncoeff(), NCOEFF);
    assert_eq!(de.record_count(), NREC);
    assert_eq!(de.constant_count(), 3);
    assert_eq!(de.constant_name(0), Some("AU"));
    assert_close(
        de.constant("AU").ok_or("AU missing")?,
        AU_KM,
        0.0,
        "AU by name",
    );
    assert_close(
        de.constant("DENUM").ok_or("DENUM missing")?,
        440.0,
        0.0,
        "DENUM",
    );
    assert_eq!(de.constant("NOPE"), None);
    assert_eq!(de.pointer_triplet(Series::Mercury), Some((3, 15, 8)));
    assert_eq!(de.pointer_triplet(Series::Sun), Some((423, 5, 2)));
    assert_eq!(de.pointer_triplet(Series::Nutation), None);
    assert_eq!(de.pointer_triplet(Series::TtTdb), None);
    assert_eq!(de.constants().count(), 3);
    Ok(())
}

/// A known quadratic must interpolate to the closed form, position AND
/// velocity, across records, granules, and both interior/boundary `tc`.
#[test]
fn known_polynomial_interpolates_exactly() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    let probes: [(u32, u32, f64); 6] = [
        (0, 0, -1.0),
        (0, 3, 0.25),
        (0, 7, 0.75),
        (1, 0, 0.0),
        (1, 6, -0.5),
        (1, 7, 0.5),
    ];
    for (rec, gran, tc) in probes {
        let jd = jd_at(rec, gran, tc, 8);
        let s: SeriesState = de.series_state(Series::Mercury, (jd, 0.0))?;
        assert_eq!(s.ncomp, 3);
        for comp in 0..3_u32 {
            let idx = usize::try_from(comp)?;
            assert_close(
                s.value[idx],
                closed_value(0, rec, gran, comp, tc),
                1e-12,
                &format!("Mercury pos r={rec} g={gran} tc={tc} comp={comp}"),
            );
            assert_close(
                s.rate[idx],
                closed_rate(0, rec, gran, comp, tc, 8),
                1e-12,
                &format!("Mercury vel r={rec} g={gran} tc={tc} comp={comp}"),
            );
        }
    }
    Ok(())
}

/// Two-part JD splitting must reach the same granule/argument.
#[test]
fn two_part_epoch_matches_single_part() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    let jd = jd_at(1, 3, 0.25, 8);
    let whole = de.series_state(Series::Mercury, (jd, 0.0))?;
    let split_a = de.series_state(Series::Mercury, (SS[0], jd - SS[0]))?;
    let split_b = de.series_state(Series::Mercury, (jd - 0.25, 0.25))?;
    for comp in 0..3 {
        assert_close(whole.value[comp], split_a.value[comp], 1e-12, "split A pos");
        assert_close(whole.value[comp], split_b.value[comp], 1e-12, "split B pos");
        assert_close(whole.rate[comp], split_a.rate[comp], 1e-12, "split A vel");
        assert_close(whole.rate[comp], split_b.rate[comp], 1e-12, "split B vel");
    }
    Ok(())
}

/// jd == stop JD must clamp into the last record / last granule (tc = 1).
#[test]
fn stop_epoch_clamps_to_last_granule() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    let s = de.series_state(Series::Mercury, (SS[1], 0.0))?;
    for comp in 0..3_u32 {
        let idx = usize::try_from(comp)?;
        assert_close(
            s.value[idx],
            closed_value(0, 1, 7, comp, 1.0),
            1e-12,
            "pos at stop JD",
        );
        assert_close(
            s.rate[idx],
            closed_rate(0, 1, 7, comp, 1.0, 8),
            1e-12,
            "vel at stop JD",
        );
    }
    Ok(())
}

#[test]
fn epoch_out_of_range_is_rejected() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    assert_eq!(
        de.series_state(Series::Mercury, (SS[0] - 1.0, 0.0)),
        Err(DeError::EpochOutOfRange)
    );
    assert_eq!(
        de.series_state(Series::Mercury, (SS[1] + 1.0, 0.0)),
        Err(DeError::EpochOutOfRange)
    );
    // Two-part epochs are combined before the range check.
    assert_eq!(
        de.series_state(Series::Mercury, (SS[1], 1.5)),
        Err(DeError::EpochOutOfRange)
    );
    Ok(())
}

#[test]
fn absent_series_is_rejected() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    assert_eq!(
        de.series_state(Series::Nutation, (SS[0] + 1.0, 0.0)),
        Err(DeError::SeriesUnavailable)
    );
    assert_eq!(
        de.testpo_value(17, 0, 1, (SS[0] + 1.0, 0.0)),
        Err(DeError::SeriesUnavailable)
    );
    Ok(())
}

#[test]
fn bad_arguments_are_rejected() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    let jd = (SS[0] + 1.0, 0.0);
    assert_eq!(de.testpo_value(18, 0, 1, jd), Err(DeError::BadArgument));
    assert_eq!(de.testpo_value(1, 2, 0, jd), Err(DeError::BadArgument));
    assert_eq!(de.testpo_value(1, 2, 7, jd), Err(DeError::BadArgument));
    assert_eq!(de.testpo_value(0, 2, 1, jd), Err(DeError::BadArgument));
    Ok(())
}

/// A byte-swapped file must be detected as big-endian and evaluate to the
/// bit-identical states.
#[test]
fn big_endian_detection_matches_little() -> TestResult {
    let le_bytes = build_synthetic(false)?;
    let be_bytes = build_synthetic(true)?;
    assert_ne!(le_bytes, be_bytes);
    let le = DeFile::parse(&le_bytes)?;
    let be = DeFile::parse(&be_bytes)?;
    assert_eq!(le.endianness(), Endianness::Little);
    assert_eq!(be.endianness(), Endianness::Big);
    assert_eq!(le.au_km().to_bits(), be.au_km().to_bits());
    assert_eq!(le.emrat().to_bits(), be.emrat().to_bits());
    assert_eq!(le.constant("AU"), be.constant("AU"));
    let jd = (jd_at(1, 5, -0.375, 8), 0.0);
    let state_little = le.series_state(Series::Mercury, jd)?;
    let state_big = be.series_state(Series::Mercury, jd)?;
    for comp in 0..3 {
        assert_eq!(
            state_little.value[comp].to_bits(),
            state_big.value[comp].to_bits()
        );
        assert_eq!(
            state_little.rate[comp].to_bits(),
            state_big.rate[comp].to_bits()
        );
    }
    Ok(())
}

/// Barycentric combinations: Earth/Moon derived from EMB + geocentric Moon.
#[test]
fn body_states_and_testpo_combinations() -> TestResult {
    let bytes = build_synthetic(false)?;
    let de = DeFile::parse(&bytes)?;
    // (rec=1, gran=1, tc=0.25) for the na=2 series: offset 16 + 10 days.
    let jd = jd_at(1, 1, 0.25, 2);
    let jd2 = (jd, 0.0);

    let emb = de.state_km(Body::Emb, jd2)?;
    let moon_geo = de.series_state(Series::Moon, jd2)?;
    let earth = de.state_km(Body::Earth, jd2)?;
    let moon = de.state_km(Body::Moon, jd2)?;
    let sun = de.state_km(Body::Sun, jd2)?;
    let ssb = de.state_km(Body::Ssb, jd2)?;

    for comp in 0..3_u32 {
        let idx = usize::try_from(comp)?;
        // Raw series against closed forms (slot 2 = EMB, 9 = Moon, 10 = Sun).
        assert_close(
            emb[idx],
            closed_value(2, 1, 1, comp, 0.25),
            1e-12,
            "EMB pos",
        );
        assert_close(
            moon_geo.value[idx],
            closed_value(9, 1, 1, comp, 0.25),
            1e-12,
            "Moon geo pos",
        );
        assert_close(
            sun[idx],
            closed_value(10, 1, 1, comp, 0.25),
            1e-12,
            "Sun pos",
        );
        assert_close(
            sun[idx + 3],
            closed_rate(10, 1, 1, comp, 0.25, 2),
            1e-12,
            "Sun vel",
        );
        assert_close(ssb[idx], 0.0, 0.0, "SSB");
    }
    for comp in 0..6 {
        // Earth = EMB - Moon_geo/(1+EMRAT); Moon = Earth + Moon_geo.
        let moon6 = if comp < 3 {
            moon_geo.value[comp]
        } else {
            moon_geo.rate[comp - 3]
        };
        assert_close(
            earth[comp],
            emb[comp] - moon6 / (1.0 + EMRAT),
            1e-12,
            "Earth from EMB",
        );
        assert_close(moon[comp], earth[comp] + moon6, 1e-12, "Moon SSB");
    }

    // testpo semantics: AU units, target minus center.
    for coord in 1..=6_u32 {
        let idx = usize::try_from(coord)? - 1;
        let moon6 = if idx < 3 {
            moon_geo.value[idx]
        } else {
            moon_geo.rate[idx - 3]
        };
        assert_close(
            de.testpo_value(13, 12, coord, jd2)?,
            emb[idx] / AU_KM,
            1e-15,
            "EMB - SSB",
        );
        assert_close(
            de.testpo_value(10, 3, coord, jd2)?,
            moon6 / AU_KM,
            1e-15,
            "Moon - Earth (special pair)",
        );
        assert_close(
            de.testpo_value(3, 10, coord, jd2)?,
            -moon6 / AU_KM,
            1e-15,
            "Earth - Moon (special pair)",
        );
        assert_close(
            de.testpo_value(11, 12, coord, jd2)?,
            sun[idx] / AU_KM,
            1e-15,
            "Sun - SSB",
        );
        assert_close(
            de.testpo_value(3, 11, coord, jd2)?,
            earth[idx] / AU_KM - sun[idx] / AU_KM,
            1e-15,
            "Earth - Sun",
        );
        assert_close(de.testpo_value(11, 11, coord, jd2)?, 0.0, 0.0, "self-self");
    }
    Ok(())
}

/// Truncated inputs must fail cleanly, not panic.
#[test]
fn truncated_input_is_rejected() -> TestResult {
    let bytes = build_synthetic(false)?;
    let header_only = bytes.get(..3000).ok_or("slice")?;
    assert!(matches!(
        DeFile::parse(header_only),
        Err(DeError::Truncated)
    ));
    let missing_last_record = bytes.get(..bytes.len() - 8).ok_or("slice")?;
    assert!(matches!(
        DeFile::parse(missing_last_record),
        Err(DeError::Truncated)
    ));
    assert!(matches!(DeFile::parse(&[]), Err(DeError::Truncated)));
    assert!(matches!(
        DeFile::parse(&[0_u8; 4096]),
        Err(DeError::InvalidHeader)
    ));
    Ok(())
}
