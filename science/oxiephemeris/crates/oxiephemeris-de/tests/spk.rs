//! SPK (`.bsp`) reader verification.
//!
//! * **Cross-format oracle** (types 2/3): NAIF's `de440.bsp` stores the
//!   same JPL DE440 integration as the classic binary
//!   `linux_p1550p2650.440` this crate already reads and proves against
//!   `testpo.440` — so both readers must agree to interpolation-noise
//!   level on every body, across the shared span. A disagreement of any
//!   size flags a reader bug; the two files' Chebyshev fits are not
//!   byte-identical (different granule packaging), so the gates below
//!   are set at the measured fit-repackaging level with margin, and the
//!   measured maxima are always printed.
//! * **Type 13 (Hermite)**: `codes_300ast_20100725.bsp` (NAIF generic
//!   kernels, 300 asteroids, heliocentric, window 8 / "degree 15").
//!   Hermite interpolation passes through its nodes by construction, so
//!   evaluating at a stored epoch must reproduce the stored state to
//!   float noise — a rigorous self-oracle needing no external data —
//!   plus a central-difference velocity consistency check and physical
//!   sanity on Ceres.
//! * **Synthetic type 13** (no data files): a hand-built one-segment DAF
//!   whose states sample an exact degree-15 polynomial. Only the
//!   spec-correct window decode (trailer stores `WINDOW − 1`, so window
//!   8 → a degree-15 interpolant) reproduces the polynomial at
//!   *off-node* epochs; a one-too-small window (degree 13) misses by
//!   ~10⁻³ km — node-reproduction tests are blind to the window size,
//!   this one is not. Plus the `N = multiple of 100` directory-count
//!   edge (`⌊(N−1)/100⌋` entries per `spk.req`) and a NaN-epoch
//!   corruption check.
//!
//! The NAIF kernels live under `data/spk/` (gitignored); those tests
//! skip cleanly when they are absent. The synthetic tests always run.

use std::path::PathBuf;

use oxiephemeris_de::{Body, DeFile, Series, SpkFile};

/// TDB seconds per day (SPK `ET` is TDB seconds past J2000.0).
const SECONDS_PER_DAY: f64 = 86_400.0;

fn data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data")
        .join(rel)
}

fn load_optional(rel: &str) -> Option<Vec<u8>> {
    let path = data_path(rel);
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => panic!("failed to read {}: {e}", path.display()),
    }
}

fn parse_spk(bytes: &[u8]) -> SpkFile<'_> {
    match SpkFile::parse(bytes) {
        Ok(f) => f,
        Err(e) => panic!("SPK parse failed: {e}"),
    }
}

/// 200 deterministic sample epochs over 1650–2350 CE (well inside both
/// files' spans), with irrational-stride fractional parts so granule
/// boundaries are not sampled systematically.
fn sample_epochs() -> Vec<(f64, f64)> {
    (0..200)
        .map(|k| {
            let kf = f64::from(k);
            let day = -128_000.0 + kf * 1_280.0;
            let frac = (kf * 0.618_033_988_749_895).fract();
            (2_451_545.0 + day, frac)
        })
        .collect()
}

#[test]
fn de440_bsp_agrees_with_classic_de440() {
    let Some(bsp_bytes) = load_optional("spk/de440.bsp") else {
        return;
    };
    let Some(cls_bytes) = load_optional("de440/linux_p1550p2650.440") else {
        return;
    };
    let spk = parse_spk(&bsp_bytes);
    let de = match DeFile::parse(&cls_bytes) {
        Ok(d) => d,
        Err(e) => panic!("classic DE440 parse failed: {e}"),
    };

    // NAIF id -> classic barycentric body. 1–9 are the planetary-system
    // barycenters w.r.t. the SSB (0), 10 the Sun, 3 the EMB.
    let pairs: [(i32, i32, Body); 11] = [
        (1, 0, Body::Mercury),
        (2, 0, Body::Venus),
        (3, 0, Body::Emb),
        (4, 0, Body::Mars),
        (5, 0, Body::Jupiter),
        (6, 0, Body::Saturn),
        (7, 0, Body::Uranus),
        (8, 0, Body::Neptune),
        (9, 0, Body::Pluto),
        (10, 0, Body::Sun),
        (399, 3, Body::Earth), // Earth w.r.t. EMB, handled below
    ];

    let mut max_pos_km = 0.0_f64;
    let mut max_vel_km_day = 0.0_f64;
    for jd in sample_epochs() {
        for &(target, center, body) in &pairs {
            let s = match spk.state(target, center, jd) {
                Ok(s) => s,
                Err(e) => panic!("bsp state({target},{center}) at {jd:?}: {e}"),
            };
            assert_eq!(s.frame_id, 1, "planetary kernel must be J2000/ICRF");
            let classic = if target == 399 {
                // Earth w.r.t. EMB = Earth_bary - EMB_bary.
                let e = match de.state_km(Body::Earth, jd) {
                    Ok(x) => x,
                    Err(err) => panic!("classic Earth: {err}"),
                };
                let emb = match de.state_km(Body::Emb, jd) {
                    Ok(x) => x,
                    Err(err) => panic!("classic EMB: {err}"),
                };
                core::array::from_fn::<f64, 6, _>(|i| e[i] - emb[i])
            } else {
                match de.state_km(body, jd) {
                    Ok(x) => x,
                    Err(err) => panic!("classic {body:?}: {err}"),
                }
            };
            for c in 0..3 {
                max_pos_km = max_pos_km.max((s.pos_km[c] - classic[c]).abs());
                max_vel_km_day =
                    max_vel_km_day.max((s.vel_km_s[c] * SECONDS_PER_DAY - classic[c + 3]).abs());
            }
        }
    }
    println!(
        "de440.bsp vs classic DE440: max |dpos| = {max_pos_km:.3e} km, \
         max |dvel| = {max_vel_km_day:.3e} km/day over 200 epochs x 11 bodies"
    );
    // Same integration, so any reader bug shows up as km-scale error;
    // the two packagings themselves agree at the millimeter level
    // (measured max 9.5e-7 km position / 2.3e-9 km/day velocity over
    // these samples); the gates carry a 10x margin over that.
    assert!(max_pos_km < 1e-5, "position mismatch: {max_pos_km} km");
    assert!(
        max_vel_km_day < 1e-7,
        "velocity mismatch: {max_vel_km_day} km/day"
    );
}

#[test]
fn de440_bsp_moon_geocentric_matches_classic_moon_series() {
    let Some(bsp_bytes) = load_optional("spk/de440.bsp") else {
        return;
    };
    let Some(cls_bytes) = load_optional("de440/linux_p1550p2650.440") else {
        return;
    };
    let spk = parse_spk(&bsp_bytes);
    let de = match DeFile::parse(&cls_bytes) {
        Ok(d) => d,
        Err(e) => panic!("classic DE440 parse failed: {e}"),
    };

    let mut max_pos_km = 0.0_f64;
    for jd in sample_epochs() {
        // Geocentric Moon = (Moon w.r.t. EMB) - (Earth w.r.t. EMB).
        let m = match spk.state(301, 3, jd) {
            Ok(s) => s,
            Err(e) => panic!("bsp Moon: {e}"),
        };
        let e = match spk.state(399, 3, jd) {
            Ok(s) => s,
            Err(e) => panic!("bsp Earth: {e}"),
        };
        let classic = match de.series_state(Series::Moon, jd) {
            Ok(s) => s,
            Err(err) => panic!("classic Moon series: {err}"),
        };
        for c in 0..3 {
            max_pos_km = max_pos_km.max((m.pos_km[c] - e.pos_km[c] - classic.value[c]).abs());
        }
    }
    println!("de440.bsp geocentric Moon vs classic series: max |dpos| = {max_pos_km:.3e} km");
    assert!(
        max_pos_km < 1e-6,
        "geocentric Moon mismatch: {max_pos_km} km"
    );
}

#[test]
fn codes_300ast_type13_reproduces_its_own_nodes() {
    let Some(bytes) = load_optional("spk/codes_300ast_20100725.bsp") else {
        return;
    };
    let spk = parse_spk(&bytes);

    // Collect the segment table once.
    let mut segments = Vec::new();
    if let Err(e) = spk.for_each_segment(|s| segments.push(*s)) {
        panic!("segment walk failed: {e}");
    }
    assert_eq!(segments.len(), 300, "the kernel packages 300 asteroids");
    assert!(segments.iter().all(|s| s.data_type == 13 && s.center == 10));

    // Node-reproduction oracle on a spread of segments: read the raw
    // epochs/states through the addressed-word accessor (independent of
    // the evaluation path) and demand the interpolant return the stored
    // state at stored epochs.
    let mut max_rel_pos = 0.0_f64;
    let mut max_rel_vel = 0.0_f64;
    for seg in segments.iter().step_by(37) {
        let (start, end) = seg.word_range();
        let mut trailer = [0.0_f64; 2];
        if let Err(e) = spk.read_words(end - 1, &mut trailer) {
            panic!("trailer read: {e}");
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = trailer[1] as usize;
        assert!(n > 10, "implausibly small record count {n}");
        for k in [1_usize, n / 3, n / 2, (4 * n) / 5, n - 2] {
            let mut epoch = [0.0_f64; 1];
            let mut stored = [0.0_f64; 6];
            if let Err(e) = spk.read_words(start + 6 * n + k, &mut epoch) {
                panic!("epoch read: {e}");
            }
            if let Err(e) = spk.read_words(start + 6 * k, &mut stored) {
                panic!("state read: {e}");
            }
            let s = match spk.state_at_et(seg.target, seg.center, epoch[0]) {
                Ok(s) => s,
                Err(e) => panic!("eval at node: {e}"),
            };
            for c in 0..3 {
                let pos_scale = stored[c].abs().max(1e6);
                let vel_scale = stored[c + 3].abs().max(1e-3);
                max_rel_pos = max_rel_pos.max((s.pos_km[c] - stored[c]).abs() / pos_scale);
                max_rel_vel = max_rel_vel.max((s.vel_km_s[c] - stored[c + 3]).abs() / vel_scale);
            }
        }
    }
    println!(
        "type 13 node reproduction: max rel pos err = {max_rel_pos:.3e}, \
         max rel vel err = {max_rel_vel:.3e}"
    );
    assert!(
        max_rel_pos < 1e-12,
        "node position not reproduced: {max_rel_pos}"
    );
    assert!(
        max_rel_vel < 1e-9,
        "node velocity not reproduced: {max_rel_vel}"
    );
}

#[test]
fn codes_300ast_velocity_consistent_with_position_derivative() {
    let Some(bytes) = load_optional("spk/codes_300ast_20100725.bsp") else {
        return;
    };
    let spk = parse_spk(&bytes);
    // Ceres, mid-span epoch (2000-01-01 is inside 1800-2200).
    let target = 2_000_001;
    let et = 0.0_f64; // J2000.0
    let h = 30.0_f64; // seconds
    let s0 = match spk.state_at_et(target, 10, et) {
        Ok(s) => s,
        Err(e) => panic!("Ceres state: {e}"),
    };
    let sp = match spk.state_at_et(target, 10, et + h) {
        Ok(s) => s,
        Err(e) => panic!("Ceres state (+h): {e}"),
    };
    let sm = match spk.state_at_et(target, 10, et - h) {
        Ok(s) => s,
        Err(e) => panic!("Ceres state (-h): {e}"),
    };
    for c in 0..3 {
        let fd = (sp.pos_km[c] - sm.pos_km[c]) / (2.0 * h);
        assert!(
            (fd - s0.vel_km_s[c]).abs() < 1e-6,
            "component {c}: finite-diff {fd} vs stored-vel {}",
            s0.vel_km_s[c]
        );
    }
}

#[test]
fn codes_300ast_ceres_is_physically_sane() {
    let Some(bytes) = load_optional("spk/codes_300ast_20100725.bsp") else {
        return;
    };
    let spk = parse_spk(&bytes);
    let au_km = 149_597_870.7_f64;
    // Several epochs across the kernel's span.
    for year_offset in [-150.0_f64, -60.0, 0.0, 80.0, 170.0] {
        let et = year_offset * 365.25 * SECONDS_PER_DAY;
        let s = match spk.state_at_et(2_000_001, 10, et) {
            Ok(s) => s,
            Err(e) => panic!("Ceres at {year_offset} yr: {e}"),
        };
        let r_au = (s.pos_km[0].powi(2) + s.pos_km[1].powi(2) + s.pos_km[2].powi(2)).sqrt() / au_km;
        let v = (s.vel_km_s[0].powi(2) + s.vel_km_s[1].powi(2) + s.vel_km_s[2].powi(2)).sqrt();
        // Ceres: a = 2.77 AU, e ~ 0.08 -> r in [2.54, 3.00] AU,
        // v in ~[16.5, 19.5] km/s.
        assert!((2.5..3.05).contains(&r_au), "heliocentric r = {r_au} AU");
        assert!((15.0..21.0).contains(&v), "|v| = {v} km/s");
    }
}

/// Builds a minimal single-segment little-endian DAF/SPK file holding
/// one type-13 segment: `states` (km, km/s) at `epochs` (TDB seconds
/// past J2000.0, strictly increasing), with the given Hermite window.
/// Layout per `daf.req`/`spk.req`: file record, one summary record, one
/// (zeroed) name record, then the segment data — `N` states, `N`
/// epochs, `⌊(N−1)/100⌋` directory epochs, and the `(WINDOW − 1, N)`
/// trailer.
fn build_type13_daf(states: &[[f64; 6]], epochs: &[f64], window: usize) -> Vec<u8> {
    assert_eq!(states.len(), epochs.len());
    let n = states.len();
    let ndir = (n - 1) / 100;
    let nwords = 7 * n + ndir + 2;
    let start_word = 3 * 128 + 1; // data begins at record 4
    let end_word = start_word + nwords - 1;
    let total_records = 3 + nwords.div_ceil(128);
    let mut bytes = vec![0_u8; total_records * 1024];

    // File record: id word, ND/NI, FWARD, binary format word.
    bytes[0..8].copy_from_slice(b"DAF/SPK ");
    bytes[8..12].copy_from_slice(&2_i32.to_le_bytes());
    bytes[12..16].copy_from_slice(&6_i32.to_le_bytes());
    bytes[76..80].copy_from_slice(&2_i32.to_le_bytes());
    bytes[88..96].copy_from_slice(b"LTL-IEEE");

    // Summary record (record 2): NEXT = 0, PREV = 0, NSUM = 1, then one
    // summary of 2 doubles + 6 packed i32s: target 2000001, center 10,
    // frame 1 (J2000), type 13, start/end word addresses.
    let sr = 1024;
    bytes[sr + 16..sr + 24].copy_from_slice(&1.0_f64.to_le_bytes());
    let s = sr + 24;
    bytes[s..s + 8].copy_from_slice(&epochs[0].to_le_bytes());
    bytes[s + 8..s + 16].copy_from_slice(&epochs[n - 1].to_le_bytes());
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    // word addresses of a few-KB test file
    for (k, v) in [2_000_001, 10, 1, 13, start_word as i32, end_word as i32]
        .into_iter()
        .enumerate()
    {
        bytes[s + 16 + 4 * k..s + 20 + 4 * k].copy_from_slice(&v.to_le_bytes());
    }

    // Segment data from record 4 (word address `start_word`).
    let mut off = (start_word - 1) * 8;
    let put = |bytes: &mut [u8], off: &mut usize, x: f64| {
        bytes[*off..*off + 8].copy_from_slice(&x.to_le_bytes());
        *off += 8;
    };
    for st in states {
        for &c in st {
            put(&mut bytes, &mut off, c);
        }
    }
    for &e in epochs {
        put(&mut bytes, &mut off, e);
    }
    for k in 0..ndir {
        // Every 100th epoch (a search accelerator this reader ignores).
        put(&mut bytes, &mut off, epochs[(k + 1) * 100 - 1]);
    }
    #[allow(clippy::cast_precision_loss)] // window/n are tiny integers here
    {
        put(&mut bytes, &mut off, (window - 1) as f64);
        put(&mut bytes, &mut off, n as f64);
    }
    bytes
}

/// The synthetic degree-15 trajectory: per-component polynomial in
/// `u = (t − 12)/4` with all 16 coefficients populated (value, then
/// exact derivative in km/s).
fn poly15(c: usize, t_s: f64) -> (f64, f64) {
    #[allow(clippy::cast_precision_loss)] // c is 0..3
    let a = 500.0 + 100.0 * (c as f64 + 1.0);
    let u = (t_s - 12.0) / 4.0;
    let mut value = 0.0;
    let mut rate_u = 0.0;
    for k in (0..=15_u32).rev() {
        value = value * u + a;
        if k > 0 {
            rate_u = rate_u * u + a * f64::from(k);
        }
    }
    // Horner above accumulates value = Σ a·u^k and rate_u = Σ a·k·u^(k−1).
    (value, rate_u / 4.0)
}

/// Off-node evaluation of the degree-15 polynomial: exact for the
/// spec-correct window 8 (degree-15 Hermite), off by ~10⁻³ km for a
/// window one too small — the discriminating regression test for the
/// `WINDOW − 1` trailer decode.
#[test]
fn synthetic_type13_degree15_polynomial_off_node_reproduction() {
    let n = 24;
    // Uneven, strictly increasing node epochs (seconds past J2000.0).
    let epochs: Vec<f64> = (0..n)
        .map(|j| f64::from(j) + 0.2 * f64::from(j % 2))
        .collect();
    let states: Vec<[f64; 6]> = epochs
        .iter()
        .map(|&t| {
            let (x, vx) = poly15(0, t);
            let (y, vy) = poly15(1, t);
            let (z, vz) = poly15(2, t);
            [x, y, z, vx, vy, vz]
        })
        .collect();
    let bytes = build_type13_daf(&states, &epochs, 8);
    let spk = parse_spk(&bytes);

    for &t in &[11.6_f64, 12.5, 13.6] {
        let jd = (2_451_545.0, t / SECONDS_PER_DAY);
        let st = spk
            .state(2_000_001, 10, jd)
            .unwrap_or_else(|e| panic!("synthetic type-13 eval at t={t}: {e}"));
        for c in 0..3 {
            let (p, v) = poly15(c, t);
            assert!(
                (st.pos_km[c] - p).abs() < 1e-6,
                "t={t} component {c}: pos {} vs exact {p} (window decode?)",
                st.pos_km[c]
            );
            assert!(
                (st.vel_km_s[c] - v).abs() < 1e-5,
                "t={t} component {c}: vel {} vs exact {v}",
                st.vel_km_s[c]
            );
        }
    }
}

/// `N` an exact multiple of 100 stores `N/100 − 1` directory entries
/// (`spk.req`: "if N is Q * 100 then only Q - 1 directory entries are
/// stored") — such a segment must parse and evaluate, not be rejected.
#[test]
fn synthetic_type13_round_state_count_parses() {
    let n = 100;
    let epochs: Vec<f64> = (0..n).map(f64::from).collect();
    // Uniform linear motion: Hermite of any window reproduces it.
    let states: Vec<[f64; 6]> = epochs
        .iter()
        .map(|&t| {
            [
                10.0 + 2.0 * t,
                -5.0 + 3.0 * t,
                1.0 - 4.0 * t,
                2.0,
                3.0,
                -4.0,
            ]
        })
        .collect();
    let bytes = build_type13_daf(&states, &epochs, 8);
    let spk = parse_spk(&bytes);
    let t = 49.5;
    let st = spk
        .state(2_000_001, 10, (2_451_545.0, t / SECONDS_PER_DAY))
        .unwrap_or_else(|e| panic!("N=100 type-13 segment rejected: {e}"));
    assert!((st.pos_km[0] - (10.0 + 2.0 * t)).abs() < 1e-9);
    assert!((st.vel_km_s[1] - 3.0).abs() < 1e-12);
}

/// A NaN node epoch inside the evaluation window must be a typed
/// `Corrupt` error, never silent NaN output.
#[test]
fn synthetic_type13_nan_epoch_is_corrupt() {
    let n = 24;
    let mut epochs: Vec<f64> = (0..n).map(f64::from).collect();
    let states: Vec<[f64; 6]> = epochs.iter().map(|&t| [t, t, t, 1.0, 1.0, 1.0]).collect();
    epochs[12] = f64::NAN;
    let bytes = build_type13_daf(&states, &epochs, 8);
    let spk = parse_spk(&bytes);
    assert!(matches!(
        spk.state(2_000_001, 10, (2_451_545.0, 12.5 / SECONDS_PER_DAY)),
        Err(oxiephemeris_de::SpkError::Corrupt)
    ));
}

#[test]
fn spk_error_paths_are_typed() {
    let Some(bytes) = load_optional("spk/de440.bsp") else {
        return;
    };
    let spk = parse_spk(&bytes);
    // Unknown pair.
    assert!(matches!(
        spk.state(12_345, 0, (2_451_545.0, 0.0)),
        Err(oxiephemeris_de::SpkError::NoSegment)
    ));
    // Epoch far outside the span.
    assert!(matches!(
        spk.state(1, 0, (9_999_999.5, 0.0)),
        Err(oxiephemeris_de::SpkError::NoSegment)
    ));
    // Not a DAF at all.
    assert!(matches!(
        SpkFile::parse(&[0_u8; 2048]),
        Err(oxiephemeris_de::SpkError::InvalidDaf)
    ));
}
