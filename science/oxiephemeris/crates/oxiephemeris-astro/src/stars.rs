//! Fixed stars: a committed bright-star subset of the Hipparcos main
//! catalog with typed accessors, feeding
//! [`oxiephemeris_bodies::star::apparent_star`].
//!
//! # Data
//!
//! [`all`]/[`by_hip`]/[`by_name`] expose the generated table in
//! `stars_table.rs` (ESA 1997, *The Hipparcos and Tycho Catalogues*,
//! CDS I/239 `hip_main.dat`; `Vmag <= 3.0`, complete astrometry, 177
//! stars; regenerate with `cargo run -p xtask -- gen-stars`). Positions
//! are ICRS at the Hipparcos epoch J1991.25; radial velocities are not
//! part of Hipparcos and are set to `0.0` (the perspective-acceleration
//! term this drops is ≲ 1 mas/century² for every star in the table).
//!
//! # Names
//!
//! [`by_name`] resolves a small hand-maintained subset of IAU-approved
//! proper names (IAU Working Group on Star Names bulletin list,
//! <https://www.iau.org/public/themes/naming_stars/>) to their HIP
//! numbers — the famous navigation-grade stars only, not the full WGSN
//! list. Matching is ASCII-case-insensitive.

use oxiephemeris_bodies::star::{CatalogStar, HIPPARCOS_EPOCH_JD_TT};

use crate::stars_table::HIPPARCOS_BRIGHT;

/// One catalog star of the committed bright subset.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    /// Hipparcos catalog number.
    pub hip: u32,
    /// IAU proper name, when the star is in this crate's named subset.
    pub name: Option<&'static str>,
    /// Johnson V magnitude (Hipparcos `Vmag` field).
    pub vmag: f64,
    /// The astrometric parameters, ready for
    /// [`oxiephemeris_bodies::star::apparent_star`].
    pub catalog: CatalogStar,
}

/// IAU proper name ↔ HIP for the famous stars of the subset (IAU WGSN
/// bulletin list; every entry is asserted to exist in the generated
/// table by `tests/stars.rs`).
const NAMED: &[(&str, u32)] = &[
    ("Sirius", 32_349),
    ("Canopus", 30_438),
    ("Arcturus", 69_673),
    ("Vega", 91_262),
    ("Capella", 24_608),
    ("Rigel", 24_436),
    ("Procyon", 37_279),
    ("Achernar", 7_588),
    ("Betelgeuse", 27_989),
    ("Altair", 97_649),
    ("Aldebaran", 21_421),
    ("Antares", 80_763),
    ("Spica", 65_474),
    ("Pollux", 37_826),
    ("Fomalhaut", 113_368),
    ("Deneb", 102_098),
    ("Regulus", 49_669),
    ("Polaris", 11_767),
];

/// Builds the typed [`Star`] from a raw table row.
fn star_of(row: &(u32, f64, f64, f64, f64, f64, f64)) -> Star {
    let (hip, vmag, ra_deg, dec_deg, plx_mas, pm_ra, pm_dec) = *row;
    Star {
        hip,
        name: NAMED.iter().find(|(_, h)| *h == hip).map(|(n, _)| *n),
        vmag,
        catalog: CatalogStar::new(
            ra_deg,
            dec_deg,
            pm_ra,
            pm_dec,
            plx_mas,
            0.0,
            HIPPARCOS_EPOCH_JD_TT,
        ),
    }
}

/// All stars of the subset, ascending HIP number.
pub fn all() -> impl Iterator<Item = Star> {
    HIPPARCOS_BRIGHT.iter().map(star_of)
}

/// The star with Hipparcos number `hip`, if it is in the subset.
#[must_use]
pub fn by_hip(hip: u32) -> Option<Star> {
    HIPPARCOS_BRIGHT
        .binary_search_by_key(&hip, |row| row.0)
        .ok()
        .map(|i| star_of(&HIPPARCOS_BRIGHT[i]))
}

/// The star with the IAU proper name `name`
/// (ASCII-case-insensitive), if it is in the named subset.
#[must_use]
pub fn by_name(name: &str) -> Option<Star> {
    NAMED
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .and_then(|&(_, hip)| by_hip(hip))
}
