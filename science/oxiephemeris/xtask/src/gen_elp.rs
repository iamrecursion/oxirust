//! `gen-elp`: regenerate the committed truncated ELP2000-82B tables in
//! `crates/oxiephemeris-analytic/src/elp2000_tables/` from the CDS VI/79
//! series files (`data/elp2000/ELP1`..`ELP36`, fetched by the data
//! scripts, not committed).
//!
//! # Source format
//!
//! Chapront-Touzé & Chapront 1983, A&A 124, 50 and 1988, A&A 190, 342;
//! record layouts and evaluation constants from the BDL reference
//! subroutine `ELP82B` (CDS VI/79 `elp82b.f`, format statements
//! `1001`..`1003`). Longitude/latitude amplitudes are arcseconds,
//! distance amplitudes kilometers.
//!
//! # Baked-in fit corrections
//!
//! The main-problem files (ELP1–3) carry six partial derivatives per
//! term; `ELP82B` combines them with the DE200/LE200 fit corrections
//! (δν, δΓ, δE, δn′, δε′) at run time. Those corrections are constants,
//! so this generator applies them once and commits only the corrected
//! amplitude — exactly `ELP82B`'s
//! `A + (B1 + dtasm·B5)(δn′ − m·δν) + B2·δΓ + B3·δE + B4·δε′`, with the
//! extra `A −= 2A·δν/3` first for the distance.
//!
//! # Truncation
//!
//! Main problem, Earth-figure/tidal/Moon-figure/relativity/solar-
//! eccentricity groups, and planetary Table 2 are committed in full.
//! Planetary Table 1 (26,190 of the 30,000 terms) is cut at
//! `|A| ≥ 0.0005` (arcsec/km) for its periodic files and
//! `|A|·5 ≥ 0.0005` for its Poisson (×t, t in Julian centuries) files —
//! the ±500-year worst case. The dropped linear sums are written into
//! the generated doc comments; being incoherent phases, their RMS
//! effect is ~0.01″, far below the theory-vs-DE440 residual.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// One ζ-group term: `(iz, Delaunay multipliers, phase_rad, A)`.
type ZetaTerm = (i8, [i8; 4], f64, f64);

/// Truncation threshold for planetary Table 1 (arcsec or km).
const TABLE1_CUTOFF: f64 = 5e-4;
/// Poisson weight: |t| ≤ 5 Julian centuries (±500 years).
const T1_WEIGHT: f64 = 5.0;

/// π to f64 precision (the std constant).
const PI: f64 = core::f64::consts::PI;

pub fn run() -> Result<(), String> {
    let root = workspace_root()?;
    let in_dir = root.join("data/elp2000");
    let out_dir = root.join("crates/oxiephemeris-analytic/src/elp2000_tables");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    emit_main_problem(&in_dir, &out_dir)?;
    emit_zeta_group(&in_dir, &out_dir)?;
    emit_planetary(&in_dir, &out_dir)?;
    Ok(())
}

/// Reads `ELP<n>`, returning its data lines (header skipped).
fn read_elp(in_dir: &Path, n: usize) -> Result<Vec<String>, String> {
    let path = in_dir.join(format!("ELP{n}"));
    let text = fs::read_to_string(&path).map_err(|e| {
        format!(
            "cannot read {} (run the data fetch scripts): {e}",
            path.display()
        )
    })?;
    Ok(text.lines().skip(1).map(str::to_owned).collect())
}

/// Fixed-column i8 at `line[start..start+3]` (Fortran `i3`).
fn int3(line: &str, start: usize, ctx: &str) -> Result<i8, String> {
    let raw = line
        .get(start..start + 3)
        .ok_or_else(|| format!("{ctx}: record shorter than column {}", start + 3))?;
    raw.trim()
        .parse()
        .map_err(|e| format!("{ctx}: bad integer {raw:?}: {e}"))
}

/// Fixed-column f64 at `line[start..end]`.
fn float(line: &str, start: usize, end: usize, ctx: &str) -> Result<f64, String> {
    let raw = line
        .get(start..end)
        .ok_or_else(|| format!("{ctx}: record shorter than column {end}"))?;
    raw.trim()
        .parse()
        .map_err(|e| format!("{ctx}: bad float {raw:?}: {e}"))
}

/// Main problem (ELP1–3): bake the DE200 fit corrections, keep every
/// term, emit `MAIN_LON` / `MAIN_LAT` / `MAIN_DIST`.
#[allow(clippy::similar_names)] // delnu/delnp etc.: verbatim ELP82B names
fn emit_main_problem(in_dir: &Path, out_dir: &Path) -> Result<(), String> {
    // Constants of `ELP82B` (see the module docs).
    let rad = 648_000.0 / PI; // arcsec per radian
    let am = 0.074_801_329_518;
    let alfa = 0.002_571_881_335;
    let dtasm = 2.0 * alfa / (3.0 * am);
    let w12 = 1_732_559_343.736_04 / rad; // W1 rate, rad/cy
    let delnu = 0.55604 / rad / w12;
    let dele = 0.01789 / rad;
    let delg = -0.08066 / rad;
    let delnp = -0.06424 / rad / w12;
    let delep = -0.12879 / rad;

    let names = ["MAIN_LON", "MAIN_LAT", "MAIN_DIST"];
    let mut out = String::new();
    writeln!(
        out,
        "//! ELP2000-82B main problem (ELP1–3), committed in full with the\n\
         //! DE200/LE200 fit corrections baked into the amplitudes (see\n\
         //! `xtask/src/gen_elp.rs`). Longitude/latitude in arcsec (sine\n\
         //! series), distance in km (cosine series).\n\
         //!\n\
         //! GENERATED FILE — DO NOT EDIT. Regenerate with\n\
         //! `cargo run -p xtask -- gen-elp` (reads `data/elp2000/`)."
    )
    .map_fmt()?;
    for (which, name) in names.iter().enumerate() {
        let lines = read_elp(in_dir, which + 1)?;
        let mut terms: Vec<([i8; 4], f64)> = Vec::new();
        for (n, line) in lines.iter().enumerate() {
            let ctx = format!("ELP{} line {}", which + 1, n + 2);
            let mut ilu = [0i8; 4];
            for (i, slot) in ilu.iter_mut().enumerate() {
                *slot = int3(line, i * 3, &ctx)?;
            }
            // Format 1001: A in cols 15-27, then B1..B6 every 12 cols.
            let a = float(line, 14, 27, &ctx)?;
            let mut b = [0.0f64; 6];
            for (i, slot) in b.iter_mut().enumerate() {
                *slot = float(line, 29 + 12 * i, 39 + 12 * i, &ctx)?;
            }
            let tgv = b[0] + dtasm * b[4];
            let a = if which == 2 {
                a - 2.0 * a * delnu / 3.0
            } else {
                a
            };
            let corrected =
                a + tgv * (delnp - am * delnu) + b[1] * delg + b[2] * dele + b[3] * delep;
            terms.push((ilu, corrected));
        }
        writeln!(
            out,
            "\n/// {} terms `(D, l', l, F, A)`.\n\
             #[rustfmt::skip]\n\
             #[allow(clippy::unreadable_literal, clippy::approx_constant)] // corrected theory coefficients\n\
             pub(crate) const {name}: &[(i8, i8, i8, i8, f64)] = &[",
            terms.len()
        )
        .map_fmt()?;
        pack_lines(&mut out, terms.iter(), |line, (ilu, a)| {
            write!(
                line,
                "({}, {}, {}, {}, {a:?}), ",
                ilu[0], ilu[1], ilu[2], ilu[3]
            )
            .map_fmt()
        })?;
        writeln!(out, "];").map_fmt()?;
        println!("gen-elp: {name}: {} terms (full)", terms.len());
    }
    let output = out_dir.join("main_problem.rs");
    fs::write(&output, out).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    Ok(())
}

/// Earth-figure, tidal, Moon-figure, relativity and solar-eccentricity
/// groups (ELP4–9, ELP22–36): the `iz·ζ + Σ ilu·Delaunay` argument
/// family, committed in full, grouped per coordinate and power of t.
fn emit_zeta_group(in_dir: &Path, out_dir: &Path) -> Result<(), String> {
    // (file, coordinate 0..3, power of t 0..3)
    let files: [(usize, usize, usize); 21] = [
        (4, 0, 0),
        (5, 1, 0),
        (6, 2, 0), // Earth figure
        (7, 0, 1),
        (8, 1, 1),
        (9, 2, 1), // Earth figure × t
        (22, 0, 0),
        (23, 1, 0),
        (24, 2, 0), // tides
        (25, 0, 1),
        (26, 1, 1),
        (27, 2, 1), // tides × t
        (28, 0, 0),
        (29, 1, 0),
        (30, 2, 0), // Moon figure
        (31, 0, 0),
        (32, 1, 0),
        (33, 2, 0), // relativity
        (34, 0, 2),
        (35, 1, 2),
        (36, 2, 2), // solar eccentricity × t²
    ];
    let mut groups: [[Vec<ZetaTerm>; 3]; 3] = Default::default();
    for (file, coord, tpow) in files {
        for (n, line) in read_elp(in_dir, file)?.iter().enumerate() {
            let ctx = format!("ELP{file} line {}", n + 2);
            let iz = int3(line, 0, &ctx)?;
            let mut ilu = [0i8; 4];
            for (i, slot) in ilu.iter_mut().enumerate() {
                *slot = int3(line, 3 + i * 3, &ctx)?;
            }
            // Format 1002: phase (deg) cols 17-25, amplitude cols 27-35.
            let phase_deg = float(line, 16, 25, &ctx)?;
            let amp = float(line, 26, 35, &ctx)?;
            groups[coord][tpow].push((iz, ilu, phase_deg.to_radians(), amp));
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "//! ELP2000-82B Earth-figure, tidal, Moon-figure, relativistic and\n\
         //! solar-eccentricity perturbations (ELP4–9, ELP22–36), committed in\n\
         //! full: sine terms `(iz, D, l', l, F, phase_rad, A)` on the argument\n\
         //! `phase + iz·ζ + Σ ilu·Delaunay`, amplitude × t^k per table.\n\
         //! Longitude/latitude in arcsec, distance in km.\n\
         //!\n\
         //! GENERATED FILE — DO NOT EDIT. Regenerate with\n\
         //! `cargo run -p xtask -- gen-elp` (reads `data/elp2000/`)."
    )
    .map_fmt()?;
    let coord_names = ["LON", "LAT", "DIST"];
    for (coord, name) in coord_names.iter().enumerate() {
        for (tpow, terms) in groups[coord].iter().enumerate() {
            writeln!(
                out,
                "\n/// {} terms, amplitude × t^{tpow}.\n\
                 #[rustfmt::skip]\n\
                 #[allow(clippy::unreadable_literal, clippy::approx_constant)] // verbatim theory coefficients\n\
                 pub(crate) const ZETA_{name}_T{tpow}: &[(i8, i8, i8, i8, i8, f64, f64)] = &[",
                terms.len()
            )
            .map_fmt()?;
            pack_lines(&mut out, terms.iter(), |line, (iz, ilu, ph, a)| {
                write!(
                    line,
                    "({iz}, {}, {}, {}, {}, {ph:?}, {a:?}), ",
                    ilu[0], ilu[1], ilu[2], ilu[3]
                )
                .map_fmt()
            })?;
            writeln!(out, "];").map_fmt()?;
        }
    }
    let output = out_dir.join("zeta.rs");
    fs::write(&output, out).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    println!("gen-elp: zeta group written (full)");
    Ok(())
}

/// Planetary perturbations, Tables 1 (ELP10–15, truncated, into
/// `planetary1.rs`) and 2 (ELP16–21, full, into `planetary2.rs`):
/// `([i8; 11], phase_rad, A)` sine terms.
fn emit_planetary(in_dir: &Path, out_dir: &Path) -> Result<(), String> {
    emit_planetary_half(
        in_dir,
        out_dir,
        "planetary1.rs",
        "Table 1 arguments are the 8 planetary mean\n\
         //! longitudes Mercury..Neptune then (D, l, F). Truncated at\n\
         //! |A| >= 0.0005 (x-t files at |A|*5 >= 0.0005, the ±500-year worst\n\
         //! case); the per-table dropped linear sums are recorded below.",
        &[
            (10, "PLAN1_LON_T0", false),
            (11, "PLAN1_LAT_T0", false),
            (12, "PLAN1_DIST_T0", false),
            (13, "PLAN1_LON_T1", true),
            (14, "PLAN1_LAT_T1", true),
            (15, "PLAN1_DIST_T1", true),
        ],
    )?;
    emit_planetary_half(
        in_dir,
        out_dir,
        "planetary2.rs",
        "Table 2 arguments are the 7 planetary mean\n\
         //! longitudes Mercury..Uranus then (D, l', l, F). Committed in full.",
        &[
            (16, "PLAN2_LON_T0", false),
            (17, "PLAN2_LAT_T0", false),
            (18, "PLAN2_DIST_T0", false),
            (19, "PLAN2_LON_T1", true),
            (20, "PLAN2_LAT_T1", true),
            (21, "PLAN2_DIST_T1", true),
        ],
    )
}

/// Emits one of the two planetary-perturbation files.
fn emit_planetary_half(
    in_dir: &Path,
    out_dir: &Path,
    filename: &str,
    describe: &str,
    tables: &[(usize, &str, bool); 6],
) -> Result<(), String> {
    let mut out = String::new();
    writeln!(
        out,
        "//! ELP2000-82B planetary perturbations: sine terms\n\
         //! `([multipliers; 11], phase_rad, A)`, longitude/latitude in\n\
         //! arcsec, distance in km; `_T1` tables scale their amplitude by\n\
         //! t (Julian centuries). {describe}\n\
         //!\n\
         //! GENERATED FILE — DO NOT EDIT. Regenerate with\n\
         //! `cargo run -p xtask -- gen-elp` (reads `data/elp2000/`)."
    )
    .map_fmt()?;
    for &(file, name, poisson) in tables {
        let truncate = (10..=15).contains(&file);
        let weight = if poisson { T1_WEIGHT } else { 1.0 };
        let mut kept: Vec<([i8; 11], f64, f64)> = Vec::new();
        let mut dropped = 0usize;
        let mut dropped_sum = 0.0f64;
        for (n, line) in read_elp(in_dir, file)?.iter().enumerate() {
            let ctx = format!("ELP{file} line {}", n + 2);
            let mut ipla = [0i8; 11];
            for (i, slot) in ipla.iter_mut().enumerate() {
                *slot = int3(line, i * 3, &ctx)?;
            }
            // Format 1003: phase (deg) cols 35-43, amplitude cols 45-53.
            let phase_deg = float(line, 34, 43, &ctx)?;
            let amp = float(line, 44, 53, &ctx)?;
            if truncate && amp.abs() * weight < TABLE1_CUTOFF {
                dropped += 1;
                dropped_sum += amp.abs() * weight;
                continue;
            }
            kept.push((ipla, phase_deg.to_radians(), amp));
        }
        writeln!(
            out,
            "\n/// {} terms kept, {dropped} dropped (linear sum {dropped_sum:.4},\n\
             /// ±500-yr worst case).\n\
             #[rustfmt::skip]\n\
             #[allow(clippy::unreadable_literal, clippy::approx_constant)] // verbatim theory coefficients\n\
             pub(crate) const {name}: &[([i8; 11], f64, f64)] = &[",
            kept.len()
        )
        .map_fmt()?;
        pack_lines(&mut out, kept.iter(), |line, (ipla, ph, a)| {
            write!(line, "({ipla:?}, {ph:?}, {a:?}), ").map_fmt()
        })?;
        writeln!(out, "];").map_fmt()?;
        println!(
            "gen-elp: {name}: kept {} dropped {dropped} (Σ|A| dropped {dropped_sum:.4})",
            kept.len()
        );
    }
    let output = out_dir.join(filename);
    fs::write(&output, out).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    Ok(())
}

/// Writes `items` as packed source lines (~130 chars per line).
fn pack_lines<T>(
    out: &mut String,
    items: impl ExactSizeIterator<Item = T>,
    mut write_one: impl FnMut(&mut String, T) -> Result<(), String>,
) -> Result<(), String> {
    let total = items.len();
    let mut line = String::new();
    for (i, item) in items.enumerate() {
        write_one(&mut line, item)?;
        if line.len() > 130 || i + 1 == total {
            writeln!(out, "    {}", line.trim_end()).map_fmt()?;
            line.clear();
        }
    }
    Ok(())
}

/// `Result`-friendly `write!` adapter.
trait FmtOk {
    fn map_fmt(self) -> Result<(), String>;
}
impl FmtOk for core::fmt::Result {
    fn map_fmt(self) -> Result<(), String> {
        self.map_err(|e| format!("formatting failed: {e}"))
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "xtask manifest dir has no parent".to_owned())
}
