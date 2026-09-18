//! `gen-stars`: regenerate the committed bright-star table
//! `crates/oxiephemeris-astro/src/stars_table.rs` from the Hipparcos
//! main catalog (`data/hipparcos/hip_main.dat`, ESA 1997, CDS I/239 —
//! fetched by the data scripts, not committed).
//!
//! Selection: `Vmag <= 3.0` with complete astrometry (RA/Dec ICRS at
//! epoch J1991.25, parallax, both proper-motion components) — 177
//! entries as of the CDS file this was written against. Output is one
//! compact tuple per star so the generated file stays far below the
//! 2000-line policy.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

/// Magnitude cut for the committed subset.
const VMAG_LIMIT: f64 = 3.0;

pub fn run() -> Result<(), String> {
    let root = workspace_root()?;
    let input = root.join("data/hipparcos/hip_main.dat");
    let output = root.join("crates/oxiephemeris-astro/src/stars_table.rs");

    let text = fs::read_to_string(&input).map_err(|e| {
        format!(
            "cannot read {} (run the data fetch scripts): {e}",
            input.display()
        )
    })?;

    let mut rows: Vec<(u32, f64, f64, f64, f64, f64, f64)> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f.len() < 14 {
            continue;
        }
        let (Ok(hip), Ok(vmag)) = (f[1].trim().parse::<u32>(), f[5].trim().parse::<f64>()) else {
            continue;
        };
        if vmag > VMAG_LIMIT {
            continue;
        }
        let (Ok(ra), Ok(dec), Ok(plx), Ok(pma), Ok(pmd)) = (
            f[8].trim().parse::<f64>(),
            f[9].trim().parse::<f64>(),
            f[11].trim().parse::<f64>(),
            f[12].trim().parse::<f64>(),
            f[13].trim().parse::<f64>(),
        ) else {
            continue; // incomplete astrometry
        };
        rows.push((hip, vmag, ra, dec, plx, pma, pmd));
    }
    if rows.len() < 100 {
        return Err(format!(
            "only {} rows selected; the catalog file looks wrong",
            rows.len()
        ));
    }
    rows.sort_by_key(|r| r.0);

    let mut out = String::new();
    writeln!(
        out,
        "//! Bright-star subset of the Hipparcos main catalog (ESA 1997,\n\
         //! CDS I/239 `hip_main.dat`): every star with `Vmag <= {VMAG_LIMIT}` and\n\
         //! complete astrometry — {} entries.\n\
         //!\n\
         //! GENERATED FILE — DO NOT EDIT. Regenerate with\n\
         //! `cargo run -p xtask -- gen-stars` (reads\n\
         //! `data/hipparcos/hip_main.dat`).\n\
         \n\
         /// `(HIP, Vmag, RA_deg, Dec_deg, plx_mas, pmRA*_mas_yr,\n\
         /// pmDec_mas_yr)` — ICRS, epoch J1991.25 (see\n\
         /// [`crate::stars`] for the typed accessors).\n\
         #[rustfmt::skip]\n\
         #[allow(clippy::unreadable_literal)] // verbatim catalog values\n\
         pub(crate) const HIPPARCOS_BRIGHT: &[(u32, f64, f64, f64, f64, f64, f64)] = &[",
        rows.len()
    )
    .map_fmt()?;
    for (hip, vmag, ra, dec, plx, pma, pmd) in &rows {
        writeln!(
            out,
            "    ({hip}, {vmag:?}, {ra:?}, {dec:?}, {plx:?}, {pma:?}, {pmd:?}),"
        )
        .map_fmt()?;
    }
    writeln!(out, "];").map_fmt()?;

    fs::write(&output, out).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    println!(
        "gen-stars: wrote {} entries to {}",
        rows.len(),
        output.display()
    );
    Ok(())
}

/// `Result`-friendly `write!` adapter (std `fmt::Write` returns
/// `fmt::Error` which carries no message).
trait FmtOk {
    fn map_fmt(self) -> Result<(), String>;
}
impl FmtOk for core::fmt::Result {
    fn map_fmt(self) -> Result<(), String> {
        self.map_err(|e| format!("formatting failed: {e}"))
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    // xtask/ lives directly under the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "xtask manifest dir has no parent".to_owned())
}
