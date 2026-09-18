//! `gen-vsop87`: regenerate the committed truncated VSOP87E tables in
//! `crates/oxiephemeris-analytic/src/vsop87_tables/` from the full CDS
//! VI/81 series files (`data/vsop87/VSOP87E.*`, fetched by the data
//! scripts, not committed).
//!
//! # Source format
//!
//! Bretagnon & Francou 1988, A&A 202, 309; file layout from the CDS
//! `vsop87.txt` notice. Each term record carries the compact form
//! `A·cos(B + C·T)` in fixed Fortran columns
//! (`1x,4i1,i5,12i3,f15.11,2f18.11,f14.11,f20.11`): A in
//! `au/(tjy^alpha)`, B in rad, C in rad/tjy, with T in thousands of
//! Julian years from J2000 and alpha the Poisson power of T (0..=5).
//!
//! # Truncation
//!
//! A term is kept when `|A| * TMAX_TJY^alpha >= cutoff(body)`, i.e. by
//! its worst-case contribution anywhere in the ±2000-year window the
//! crate documents, so slow-growing Poisson terms are weighted fairly
//! against periodic ones. The per-body cutoffs sit two to three orders
//! of magnitude below the theory's own accuracy (`p0·a0` of the
//! notice), so truncation never dominates the error budget; the sum of
//! every dropped `|A| * TMAX^alpha` is written into each generated
//! file's doc comment as a hard upper bound on the truncation error.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

/// Half-width of the documented validity window, in thousands of
/// Julian years from J2000 (±2000 years).
const TMAX_TJY: f64 = 2.0;

/// `(file suffix, module name, cutoff in au)`. Cutoffs: 1e-9 au for the
/// bodies whose theory error `p0·a0` is ~1e-8..1e-7 au (see the notice's
/// PRECISION table; the Sun's barycentric series is a sum of planetary
/// reflexes, so it gets the tightest cutoff), 1e-8 au for the giant
/// planets whose theory error is ~1e-6..1e-5 au.
const BODIES: [(&str, &str, f64); 9] = [
    ("sun", "sun", 1e-9),
    ("mer", "mercury", 1e-9),
    ("ven", "venus", 1e-9),
    ("ear", "earth", 1e-9),
    ("mar", "mars", 3e-9),
    ("jup", "jupiter", 1e-8),
    ("sat", "saturn", 3e-8),
    ("ura", "uranus", 1e-8),
    ("nep", "neptune", 1e-8),
];

/// One parsed `A·cos(B + C·T)` term.
#[derive(Clone, Copy)]
struct Term {
    a: f64,
    b: f64,
    c: f64,
}

/// The six Poisson powers of one coordinate.
type Coordinate = [Vec<Term>; 6];

pub fn run() -> Result<(), String> {
    let root = workspace_root()?;
    let out_dir = root.join("crates/oxiephemeris-analytic/src/vsop87_tables");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    for (suffix, module, cutoff) in BODIES {
        let input = root.join(format!("data/vsop87/VSOP87E.{suffix}"));
        let text = fs::read_to_string(&input).map_err(|e| {
            format!(
                "cannot read {} (run the data fetch scripts): {e}",
                input.display()
            )
        })?;
        let coords = parse_body(&text)?;
        let output = out_dir.join(format!("{module}.rs"));
        let (kept, total) = emit_body(&coords, module, cutoff, &output)?;
        println!("gen-vsop87: {module}: kept {kept}/{total} terms (cutoff {cutoff:e} au)");
    }
    Ok(())
}

/// Parses one `VSOP87E.*` file into its three coordinates.
#[allow(clippy::similar_names)] // ic/it/iv: the notice's own field names
fn parse_body(text: &str) -> Result<[Coordinate; 3], String> {
    let mut coords: [Coordinate; 3] = Default::default();
    // (coordinate index 0..=2, alpha 0..=5) of the series being read.
    let mut current: Option<(usize, usize)> = None;
    for (n, line) in text.lines().enumerate() {
        let lineno = n + 1;
        if line.contains("VSOP87 VERSION") {
            if !line.contains("BARYCENTRIC") {
                return Err(format!(
                    "line {lineno}: expected a VSOP87E (barycentric) file"
                ));
            }
            let ic = field_after(line, "VARIABLE ", lineno)?;
            let alpha = field_after(line, "*T**", lineno)?;
            if !(1..=3).contains(&ic) || alpha > 5 {
                return Err(format!("line {lineno}: variable {ic} / alpha {alpha}"));
            }
            current = Some((ic - 1, alpha));
            continue;
        }
        let Some((ic, alpha)) = current else {
            return Err(format!("line {lineno}: term record before any header"));
        };
        // Term record, fixed columns of the documented Fortran format:
        // A f18.11 in cols 80-97, B f14.11 in 98-111, C f20.11 in
        // 112-131 (1-based, inclusive).
        let a = fixed_f64(line, 79, 97, lineno)?;
        let b = fixed_f64(line, 97, 111, lineno)?;
        let c = fixed_f64(line, 111, 131, lineno)?;
        // Cross-check the record's own version/coordinate/alpha digits
        // (cols 2-5) against the current header.
        let tag = line
            .get(1..5)
            .ok_or_else(|| format!("line {lineno}: short record"))?;
        let expect_ic = char::from(b'0' + u8::try_from(ic + 1).unwrap_or(0));
        let expect_it = char::from(b'0' + u8::try_from(alpha).unwrap_or(9));
        let mut tag_chars = tag.chars();
        let (iv, _ib, ic_c, it_c) = (
            tag_chars.next(),
            tag_chars.next(),
            tag_chars.next(),
            tag_chars.next(),
        );
        if iv != Some('5') || ic_c != Some(expect_ic) || it_c != Some(expect_it) {
            return Err(format!(
                "line {lineno}: record tag {tag:?} mismatches header"
            ));
        }
        coords[ic][alpha].push(Term { a, b, c });
    }
    Ok(coords)
}

/// Parses the integer immediately following `marker` in `line`.
fn field_after(line: &str, marker: &str, lineno: usize) -> Result<usize, String> {
    let at = line
        .find(marker)
        .ok_or_else(|| format!("line {lineno}: missing {marker:?}"))?;
    let rest = &line[at + marker.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits
        .parse()
        .map_err(|e| format!("line {lineno}: bad integer after {marker:?}: {e}"))
}

/// Parses the fixed-column float `line[start..end]` (0-based, exclusive).
fn fixed_f64(line: &str, start: usize, end: usize, lineno: usize) -> Result<f64, String> {
    let raw = line
        .get(start..end)
        .ok_or_else(|| format!("line {lineno}: record shorter than column {end}"))?;
    raw.trim()
        .parse()
        .map_err(|e| format!("line {lineno}: bad float in cols {start}..{end}: {e}"))
}

/// Writes one body's truncated tables; returns `(kept, total)`.
fn emit_body(
    coords: &[Coordinate; 3],
    module: &str,
    cutoff: f64,
    output: &PathBuf,
) -> Result<(usize, usize), String> {
    let names = ["X", "Y", "Z"];
    let mut kept_total = 0usize;
    let mut term_total = 0usize;
    let mut dropped_bound = [0.0f64; 3];
    let mut body = String::new();
    for (ci, coord) in coords.iter().enumerate() {
        for (alpha, terms) in coord.iter().enumerate() {
            let weight = TMAX_TJY.powi(i32::try_from(alpha).map_err(|e| e.to_string())?);
            let kept: Vec<&Term> = terms
                .iter()
                .filter(|t| t.a.abs() * weight >= cutoff)
                .collect();
            dropped_bound[ci] += terms
                .iter()
                .filter(|t| t.a.abs() * weight < cutoff)
                .map(|t| t.a.abs() * weight)
                .sum::<f64>();
            kept_total += kept.len();
            term_total += terms.len();
            writeln!(
                body,
                "\n/// {} · T^{alpha}: {} of {} terms.\n\
                 #[rustfmt::skip]\n\
                 #[allow(clippy::unreadable_literal, clippy::approx_constant)] // verbatim theory coefficients\n\
                 const {}{alpha}: &[(f64, f64, f64)] = &[",
                names[ci],
                kept.len(),
                terms.len(),
                names[ci]
            )
            .map_fmt()?;
            let mut line = String::new();
            for (i, t) in kept.iter().enumerate() {
                write!(line, "({:?}, {:?}, {:?}), ", t.a, t.b, t.c).map_fmt()?;
                if line.len() > 130 || i + 1 == kept.len() {
                    writeln!(body, "    {}", line.trim_end()).map_fmt()?;
                    line.clear();
                }
            }
            writeln!(body, "];").map_fmt()?;
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "//! VSOP87E {module}: truncated barycentric rectangular series,\n\
         //! dynamical ecliptic and equinox J2000 (Bretagnon & Francou 1988,\n\
         //! A&A 202, 309; CDS VI/81 `VSOP87E.*`).\n\
         //!\n\
         //! GENERATED FILE — DO NOT EDIT. Regenerate with\n\
         //! `cargo run -p xtask -- gen-vsop87` (reads `data/vsop87/`).\n\
         //!\n\
         //! Truncation: terms with `|A|·{TMAX_TJY}^alpha < {cutoff:e}` au dropped\n\
         //! ({kept_total} of {term_total} terms kept). Upper bound of the dropped\n\
         //! amplitude anywhere in |T| <= {TMAX_TJY} (±2000 yr):\n\
         //! X {:.3e} au, Y {:.3e} au, Z {:.3e} au.",
        dropped_bound[0], dropped_bound[1], dropped_bound[2]
    )
    .map_fmt()?;
    out.push_str(&body);
    writeln!(
        out,
        "\n/// `[coordinate][alpha]` slices: X/Y/Z, each with the Poisson\n\
         /// powers T^0..=T^5 of `A·cos(B + C·T)` terms.\n\
         #[allow(clippy::type_complexity)]\n\
         pub(crate) const TABLES: [[&[(f64, f64, f64)]; 6]; 3] = [\n\
         \x20   [X0, X1, X2, X3, X4, X5],\n\
         \x20   [Y0, Y1, Y2, Y3, Y4, Y5],\n\
         \x20   [Z0, Z1, Z2, Z3, Z4, Z5],\n\
         ];"
    )
    .map_fmt()?;
    fs::write(output, out).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    Ok((kept_total, term_total))
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
