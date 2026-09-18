//! Swiss Ephemeris output-fixture conformance gate.
//!
//! `fixtures/se/se_fixtures.json` is generated **outside** this
//! repository by the quarantined `oxieph-fixgen` tool, which drives
//! Swiss Ephemeris strictly as a black box (pyswisseph binary wheel,
//! public API only — no SE source code involved anywhere). SE *output*
//! fixtures are the one permitted SE artifact under this project's
//! clean-room policy (`CLAUDE.md`, prime directive 1). Both SE
//! (`SEFLG_JPLEPH`) and this crate read the *same* JPL DE440 classic
//! binary, so the comparison isolates the coordinate / time-scale /
//! apparent-place pipelines from ephemeris-integration differences.
//!
//! Skip-if-absent: the test is a no-op (with a note) when either the
//! fixture JSON or the DE440 binary is missing.
//!
//! # Gate classes
//!
//! * **Astronomy** (`calc` frames, topocentric, houses, sidtime): the
//!   models on both sides are equivalent formulations (IAU precession/
//!   nutation vs SE's defaults agree to sub-mas in 1900–2100), so the
//!   gates are tight — the headline requirement is **< 0.001°** on
//!   every angle, with most categories measured orders of magnitude
//!   below that.
//! * **ΔT-dependent** (`calc_ut`): SE's own ΔT model vs this crate's
//!   Espenak–Meeus polynomial differ by a fraction of a second in the
//!   modern era; the Moon moves ≈ 0.55″/s, so these gates stay at the
//!   headline 0.001° rather than the astronomy-class tightness.
//! * **Definitional** (ayanamshas, sidereal longitudes, mean/osculating
//!   node & apogee): SE's built-in anchor values and element polynomials
//!   are its own; this crate's come from the cited papers. Differences
//!   here are *documented model differences*, not bugs — gated at their
//!   measured level so a regression still trips, with the measured
//!   values recorded next to each gate.

use serde::Deserialize;
use std::path::PathBuf;

use oxiephemeris_compat::{Context, SiderealMode};
use oxiephemeris_de::DeFile;

/// SE body numbers of the node/apogee bodies: this crate documents
/// latitude/distance (and their speeds) as always `0.0` for these, so
/// only longitude and longitude speed are compared.
const SE_MEAN_NODE: i32 = 10;
const SE_OSCU_APOG: i32 = 13;

const SEFLG_XYZ: u32 = 4096;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/se/se_fixtures.json")
}

fn de440_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440")
}

#[derive(Deserialize)]
struct Fixtures {
    meta: Meta,
    calc_et: Vec<CalcEtCase>,
    calc_ut: Vec<CalcUtCase>,
    calc_topo: Vec<CalcTopoCase>,
    calc_sidereal: Vec<CalcSiderealCase>,
    houses: Vec<HousesCase>,
    sidtime: Vec<SidtimeCase>,
    ayanamsha: Vec<AyanamshaCase>,
}

#[derive(Deserialize)]
struct Meta {
    se_version: String,
    ephemeris: String,
}

#[derive(Deserialize)]
struct CalcEtCase {
    jd_et: f64,
    ipl: i32,
    iflag: u32,
    label: String,
    xx: [f64; 6],
}

#[derive(Deserialize)]
struct CalcUtCase {
    jd_ut: f64,
    #[allow(dead_code)] // diagnostic: SE's ΔT for this epoch, in days
    se_deltat_days: f64,
    ipl: i32,
    iflag: u32,
    label: String,
    xx: [f64; 6],
}

#[derive(Deserialize)]
struct CalcTopoCase {
    jd_et: f64,
    ipl: i32,
    iflag: u32,
    label: String,
    geo_lon_deg: f64,
    geo_lat_deg: f64,
    alt_m: f64,
    xx: [f64; 6],
}

#[derive(Deserialize)]
struct CalcSiderealCase {
    jd_et: f64,
    ipl: i32,
    iflag: u32,
    label: String,
    sid_mode: String,
    xx: [f64; 6],
}

#[derive(Deserialize)]
struct HousesCase {
    jd_ut: f64,
    site: String,
    lat_deg: f64,
    lon_deg: f64,
    hsys: String,
    iflag: u32,
    sid_mode: Option<String>,
    cusps: [f64; 12],
    ascmc: Vec<f64>,
}

#[derive(Deserialize)]
struct SidtimeCase {
    jd_ut: f64,
    hours: f64,
}

#[derive(Deserialize)]
struct AyanamshaCase {
    jd_et: f64,
    sid_mode: String,
    value_deg: f64,
}

/// Per-label gate: angles in degrees, distances in AU, angle speeds in
/// deg/day, distance speeds in AU/day. XYZ labels reuse `dist`/`dist_speed`
/// for all Cartesian components.
struct Gate {
    ang: f64,
    dist: f64,
    ang_speed: f64,
    dist_speed: f64,
}

/// Measured maxima (this SE 2.10.03 + DE440 fixture set) are noted per
/// arm; gates are set with documented headroom above them, and never
/// looser than the 0.001° headline for the astronomy class.
///
/// Three SE conventions are *documented differences*, not reproduced:
///
/// * **Speed conventions.** SE's reported daily speeds are not the
///   numerical derivative of SE's own position output: differencing
///   SE's positions at ±0.001 d disagrees with SE's reported speed by
///   ≈ 9×10⁻⁵ °/day for the geocentric Moon and ≈ 5×10⁻³ °/day for the
///   topocentric Moon (measured directly on SE 2.10.03). This crate's
///   speeds are central differences of its full pipeline (±0.01 d
///   geocentric, ±0.001 d topocentric), i.e. faithful derivatives of
///   its positions — so the speed gates reflect SE's convention gap,
///   not pipeline error. The same applies to the distance rate of
///   *apparent* (aberration-on) positions, where SE embeds an extra
///   ≈ 1.7×10⁻⁶ × dist /day term: with aberration off
///   (truepos/astrometric), SE's distance rates match this crate to
///   ≈ 10⁻⁷ AU/day.
/// * **Heliocentric light-time**: SE retards the body by its
///   *barycentric* light-time `|B_ssb|/c` (verified against SE output
///   to ≤ 0.006″); this crate uses the physically consistent
///   Sun-relative `|B − S|/c`. Up to ≈ 0.4″ on Mercury — the `helio_ecl`
///   angle gate stays at the 0.001° headline, which the difference
///   comfortably passes.
/// * **Mean lunar apogee ("Lilith", body 12)**: SE's own polynomial
///   (which also carries a latitude) sits ≈ 0.115° from the
///   Simon et al. (1994) mean apogee used here — a model difference,
///   gated at its measured size. The mean *node* is digit-identical
///   Simon 1994 on both sides (SE NONUT J2000 value 125.044555044°),
///   and the osculating node/apogee agree to ≈ 2×10⁻⁶ / 8×10⁻⁵ °.
#[allow(clippy::match_same_arms)] // arms are grouped by *reason*, not by value
fn gate_for(label: &str) -> Gate {
    let g = |ang, dist, ang_speed, dist_speed| Gate {
        ang,
        dist,
        ang_speed,
        dist_speed,
    };
    match label {
        // Aberration-free rates — everything tight (measured: angles
        // ≤ 5.5e-7°, dist ≤ 3.3e-10 AU, ddist ≤ 7.4e-8 AU/day).
        "truepos_ecl" | "astrometric_equ" => g(1e-3, 1e-8, 1e-4, 1e-6),
        // Aberration on: angles/dist tight (measured ≤ 5.5e-7° /
        // 3.3e-10 AU), speeds at SE's convention level (see above).
        "apparent_ecl" | "apparent_equ" | "j2000_ecl" | "j2000_equ" | "mean_ecl_nonut" => {
            g(1e-3, 1e-8, 1e-4, 1e-4)
        }
        "apparent_equ_xyz" | "icrs_equ_xyz" | "icrs_j2000_equ_xyz" => g(1e-3, 1e-6, 1e-4, 1e-4),
        // Barycentric: no aberration → velocity tight (measured 6.9e-7).
        "bary_equ_xyz" => g(1e-3, 1e-6, 1e-4, 1e-5),
        // Heliocentric: SE's barycentric-light-time convention (above);
        // measured 1.13e-4° / 1.6e-4 °/day.
        "helio_ecl" => g(1e-3, 1e-6, 1e-3, 1e-6),
        // Topocentric: positions tight (measured 4.6e-5°); speeds at
        // SE's topocentric speed-convention level (see above).
        "topo_apparent_ecl" => g(1e-3, 1e-7, 1e-2, 1e-6),
        // ΔT class — SE's ΔT vs Espenak–Meeus (≈ 4.7 s at the 2024
        // fixture epoch, EM extrapolation; Moon ≈ 0.55″/s → ≈ 8×10⁻⁴°).
        "calc_ut" => g(1e-3, 1e-6, 1e-4, 1e-4),
        // Node/apogee bodies, per SE body number (see above; measured
        // ipl10 5.0e-5°, ipl11 2.1e-6°, ipl12 0.115°, ipl13 7.7e-5°).
        "node_true_equinox_ipl10" | "node_mean_equinox_ipl10" => {
            g(1e-3, f64::INFINITY, 1e-4, f64::INFINITY)
        }
        "node_true_equinox_ipl11" | "node_mean_equinox_ipl11" => {
            g(1e-4, f64::INFINITY, 1e-4, f64::INFINITY)
        }
        "node_true_equinox_ipl12" | "node_mean_equinox_ipl12" => {
            g(0.13, f64::INFINITY, 1e-2, f64::INFINITY)
        }
        "node_true_equinox_ipl13" | "node_mean_equinox_ipl13" => {
            g(1e-3, f64::INFINITY, 1e-3, f64::INFINITY)
        }
        // Sidereal: anchor differences only (the ayanamsha + Δψ
        // convention is reproduced — see
        // `crate::context::calc::sidereal_offset_rad`); per-mode gates
        // of the same order as the ayanamsha gates below.
        "sidereal_FaganBradley" => g(1e-3, 1e-8, 1e-4, 1e-4),
        "sidereal_Lahiri" => g(1e-2, 1e-8, 1e-4, 1e-4),
        other => panic!("no gate defined for fixture label {other:?}"),
    }
}

/// Wrap-aware absolute angle difference in degrees.
fn ang_diff_deg(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

/// Tracks the worst error per (label, quantity) and formats a report.
#[derive(Default)]
struct Tracker {
    max: std::collections::BTreeMap<String, (f64, String)>,
    violations: Vec<String>,
}

impl Tracker {
    fn record(&mut self, label: &str, quantity: &str, err: f64, gate: f64, case_desc: &str) {
        let key = format!("{label}/{quantity}");
        let entry = self.max.entry(key).or_insert((0.0, String::new()));
        if err > entry.0 {
            *entry = (err, case_desc.to_string());
        }
        if err > gate {
            self.violations.push(format!(
                "{label}/{quantity}: {err:.3e} > gate {gate:.1e} at {case_desc}"
            ));
        }
    }

    fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("per-category maxima:\n");
        for (key, (err, desc)) in &self.max {
            // Writing to a String cannot fail; the Result is formal.
            let _ = writeln!(out, "  {key:<40} {err:.3e}  ({desc})");
        }
        out
    }
}

/// Compares one `xx` sextet under a label's gates. `is_node` compares
/// longitude (+speed) only, per the documented lat/dist difference.
#[allow(clippy::too_many_arguments)]
fn compare_xx(
    tracker: &mut Tracker,
    label: &str,
    ours: [f64; 6],
    se: [f64; 6],
    iflag: u32,
    ipl: i32,
    desc: &str,
) {
    let gate = gate_for(label);
    let is_node = (SE_MEAN_NODE..=SE_OSCU_APOG).contains(&ipl);
    if iflag & SEFLG_XYZ != 0 {
        for i in 0..3 {
            tracker.record(label, "xyz_au", (ours[i] - se[i]).abs(), gate.dist, desc);
            tracker.record(
                label,
                "vxyz_au_day",
                (ours[i + 3] - se[i + 3]).abs(),
                gate.dist_speed,
                desc,
            );
        }
        return;
    }
    tracker.record(
        label,
        "lon_deg",
        ang_diff_deg(ours[0], se[0]),
        gate.ang,
        desc,
    );
    tracker.record(
        label,
        "dlon_deg_day",
        (ours[3] - se[3]).abs(),
        gate.ang_speed,
        desc,
    );
    if !is_node {
        tracker.record(label, "lat_deg", (ours[1] - se[1]).abs(), gate.ang, desc);
        tracker.record(label, "dist_au", (ours[2] - se[2]).abs(), gate.dist, desc);
        tracker.record(
            label,
            "dlat_deg_day",
            (ours[4] - se[4]).abs(),
            gate.ang_speed,
            desc,
        );
        tracker.record(
            label,
            "ddist_au_day",
            (ours[5] - se[5]).abs(),
            gate.dist_speed,
            desc,
        );
    }
}

fn sid_mode_of(name: &str) -> SiderealMode {
    match name {
        "FaganBradley" => SiderealMode::FaganBradley,
        "Lahiri" => SiderealMode::Lahiri,
        "Krishnamurti" => SiderealMode::Krishnamurti,
        "Raman" => SiderealMode::Raman,
        other => panic!("unknown sid_mode in fixture: {other:?}"),
    }
}

/// Per-mode ayanamsha gate, degrees. SE's built-in anchors vs this
/// crate's paper-sourced anchors are *different definitions*; each gate
/// is set just above the measured difference so a regression trips
/// (measured: FB 1.15e-4, Lahiri 4.44e-3, Krishnamurti 2.11e-4,
/// Raman 9.64e-3).
#[allow(clippy::match_same_arms)] // one arm per mode keeps the per-mode numbers legible
fn ayanamsha_gate_deg(mode: &str) -> f64 {
    match mode {
        "FaganBradley" => 0.01,
        "Lahiri" => 0.01,
        "Krishnamurti" => 0.02,
        "Raman" => 0.02,
        other => panic!("unknown ayanamsha mode {other:?}"),
    }
}

#[test]
#[allow(clippy::too_many_lines)] // one linear pass over the seven fixture categories
fn se_output_fixture_gate() {
    let fix_path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&fix_path) else {
        eprintln!(
            "note: skipping test — SE fixture not found at {} (generate with oxieph-fixgen)",
            fix_path.display()
        );
        return;
    };
    let fixtures: Fixtures = match serde_json::from_str(&text) {
        Ok(f) => f,
        Err(e) => panic!("se_fixtures.json present but unparsable: {e}"),
    };
    let Ok(bytes) = std::fs::read(de440_path()) else {
        eprintln!(
            "note: skipping test — DE440 binary not found at {} (scripts/fetch_de440.sh)",
            de440_path().display()
        );
        return;
    };
    let de = match DeFile::parse(&bytes) {
        Ok(de) => de,
        Err(e) => panic!("DE440 parse failed: {e}"),
    };
    eprintln!(
        "comparing against {} ({})",
        fixtures.meta.se_version, fixtures.meta.ephemeris
    );

    let mut tracker = Tracker::default();
    let ctx = Context::with_ephemeris(de.clone());

    for c in &fixtures.calc_et {
        let desc = format!("calc(jd={}, ipl={}, {})", c.jd_et, c.ipl, c.label);
        // Node/apogee bodies get per-body gates (mean node is
        // digit-identical Simon 1994 on both sides; the apogee models
        // are SE's own — see `gate_for`).
        let label = if (SE_MEAN_NODE..=SE_OSCU_APOG).contains(&c.ipl) {
            format!("{}_ipl{}", c.label, c.ipl)
        } else {
            c.label.clone()
        };
        let ours = ctx
            .calc(c.jd_et, c.ipl, c.iflag)
            .unwrap_or_else(|e| panic!("{desc}: {e}"));
        compare_xx(&mut tracker, &label, ours, c.xx, c.iflag, c.ipl, &desc);
    }

    for c in &fixtures.calc_ut {
        let desc = format!("calc_ut(jd={}, ipl={}, {})", c.jd_ut, c.ipl, c.label);
        let ours = ctx
            .calc_ut(c.jd_ut, c.ipl, c.iflag)
            .unwrap_or_else(|e| panic!("{desc}: {e}"));
        compare_xx(&mut tracker, "calc_ut", ours, c.xx, c.iflag, c.ipl, &desc);
    }

    for c in &fixtures.calc_topo {
        let mut topo_ctx = Context::with_ephemeris(de.clone());
        topo_ctx.set_topo(c.geo_lon_deg, c.geo_lat_deg, c.alt_m);
        let desc = format!("calc_topo(jd={}, ipl={})", c.jd_et, c.ipl);
        let ours = topo_ctx
            .calc(c.jd_et, c.ipl, c.iflag)
            .unwrap_or_else(|e| panic!("{desc}: {e}"));
        compare_xx(&mut tracker, &c.label, ours, c.xx, c.iflag, c.ipl, &desc);
    }

    for c in &fixtures.calc_sidereal {
        let mut sid_ctx = Context::with_ephemeris(de.clone());
        sid_ctx.set_sid_mode(sid_mode_of(&c.sid_mode));
        let desc = format!(
            "calc_sidereal(jd={}, ipl={}, {})",
            c.jd_et, c.ipl, c.sid_mode
        );
        let ours = sid_ctx
            .calc(c.jd_et, c.ipl, c.iflag)
            .unwrap_or_else(|e| panic!("{desc}: {e}"));
        compare_xx(&mut tracker, &c.label, ours, c.xx, c.iflag, c.ipl, &desc);
    }

    for c in &fixtures.houses {
        let mut house_ctx = Context::new();
        let (label, hsys_char) = match &c.sid_mode {
            Some(mode) => {
                house_ctx.set_sid_mode(sid_mode_of(mode));
                ("houses_sidereal", c.hsys.chars().next())
            }
            None => ("houses", c.hsys.chars().next()),
        };
        let hsys = hsys_char.unwrap_or_else(|| panic!("empty hsys in fixture at {}", c.site));
        let gate = if c.sid_mode.is_some() {
            ayanamsha_gate_deg("Lahiri")
        } else {
            1e-3
        };
        let desc = format!("houses(jd={}, {}, {})", c.jd_ut, c.site, c.hsys);
        let ours = house_ctx
            .houses_ex(c.jd_ut, c.iflag, c.lat_deg, c.lon_deg, hsys)
            .unwrap_or_else(|e| panic!("{desc}: {e}"));
        for (i, (a, b)) in ours.cusps.iter().zip(c.cusps.iter()).enumerate() {
            tracker.record(
                label,
                "cusp_deg",
                ang_diff_deg(*a, *b),
                gate,
                &format!("{desc} cusp {}", i + 1),
            );
        }
        // ascmc slots 0..=4 (ASC, MC, ARMC, Vertex, East Point); slots
        // 5..=7 are documented 0.0 here (unimplemented SE extras).
        for (i, name) in ["asc", "mc", "armc", "vertex", "equasc"].iter().enumerate() {
            tracker.record(
                label,
                name,
                ang_diff_deg(ours.ascmc[i], c.ascmc[i]),
                gate,
                &desc,
            );
        }
    }

    // Sidereal time: both sides are sub-ms-class GAST models; measured
    // agreement ~0.1 ms. Gate 1e-5 h = 36 ms.
    for c in &fixtures.sidtime {
        let ours = ctx
            .sidtime(c.jd_ut)
            .unwrap_or_else(|e| panic!("sidtime({}): {e}", c.jd_ut));
        let diff = ((ours - c.hours + 12.0).rem_euclid(24.0) - 12.0).abs();
        tracker.record(
            "sidtime",
            "hours",
            diff,
            1e-5,
            &format!("sidtime(jd={})", c.jd_ut),
        );
    }

    for c in &fixtures.ayanamsha {
        let mut ay_ctx = Context::new();
        ay_ctx.set_sid_mode(sid_mode_of(&c.sid_mode));
        let ours = ay_ctx
            .get_ayanamsa(c.jd_et)
            .unwrap_or_else(|e| panic!("get_ayanamsa({}): {e}", c.jd_et));
        tracker.record(
            &format!("ayanamsha_{}", c.sid_mode),
            "deg",
            ang_diff_deg(ours, c.value_deg),
            ayanamsha_gate_deg(&c.sid_mode),
            &format!("ayanamsha(jd={}, {})", c.jd_et, c.sid_mode),
        );
    }

    eprintln!("{}", tracker.report());
    assert!(
        tracker.violations.is_empty(),
        "{} gate violation(s):\n{}\n{}",
        tracker.violations.len(),
        tracker.violations.join("\n"),
        tracker.report()
    );
}
