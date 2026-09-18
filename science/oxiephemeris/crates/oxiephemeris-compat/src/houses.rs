//! `houses` / `houses_ex`: the `swe_houses` / `swe_houses_ex` analogues,
//! plus the SE-documented `ascmc` output block and house-system
//! character codes.
//!
//! # House-system characters
//!
//! The SE documentation selects the house system by an `int` holding an
//! ASCII character (swephprg.htm §14.1, consulted 2026-07-06). The seven
//! systems this workspace implements map as:
//!
//! | char | system |
//! |------|--------|
//! | `'P'` | Placidus |
//! | `'K'` | Koch |
//! | `'O'` | Porphyry ("Porphyrius") |
//! | `'R'` | Regiomontanus |
//! | `'C'` | Campanus |
//! | `'A'` or `'E'` | Equal (cusp 1 = Ascendant) |
//! | `'W'` | Whole Sign |
//!
//! Every other character — including systems SE documents but this
//! workspace does not implement (Alcabitius `'B'`, equal/MC `'D'`,
//! Gauquelin `'G'`, azimuthal `'H'`, Sunshine `'I'`/`'i'`, Morinus
//! `'M'`, Polich/Page `'T'`, Krusinski `'U'`, vehlow `'V'`, axial `'X'`,
//! APC `'Y'`, …) — is [`CompatError::UnsupportedHouseSystem`].
//!
//! # The `ascmc` block
//!
//! `swe_houses` returns ten additional points in an `ascmc[10]` array;
//! the SE documentation assigns index constants `SE_ASC` = 0 … through
//! `SE_NASCMC` = 8. [`Houses::ascmc`] keeps the same slot layout for the
//! first eight (see the index constants below). Slots
//! [`ASCMC_COASC1`] (co-ascendant, W. Koch), [`ASCMC_COASC2`]
//! (co-ascendant, M. Munkasey) and [`ASCMC_POLASC`] (polar ascendant,
//! M. Munkasey) are **always `0.0`** — `oxiephemeris-astro` does not
//! implement those points (documented crate difference; see the crate
//! docs).
//!
//! # Polar latitudes — documented difference
//!
//! Where the SE documentation describes falling back to Porphyry cusps
//! when Placidus/Koch are undefined inside the polar circles, this crate
//! **returns a typed error instead of silently switching systems**
//! ([`oxiephemeris_astro::houses::HousesError::Undefined`] via
//! [`CompatError::Houses`]) — the caller decides what fallback, if any,
//! is acceptable.

use oxiephemeris_astro::angles::{ascendant, east_point, local_sidereal_time, mc, vertex};
use oxiephemeris_astro::ayanamsha::ayanamsha_rad;
use oxiephemeris_astro::houses::{cusps as astro_cusps, HouseSystem};
use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, nutation_iau2000a};
use oxiephemeris_bodies::sidereal::gast_iau2006;
use oxiephemeris_core::angle::{normalize_0_two_pi, DEG2RAD, RAD2DEG};
use oxiephemeris_core::time::{JulianDate, J2000_JD};

use crate::context::{jd_tt_from_ut, Context};
use crate::flags::{SEFLG_RADIANS, SEFLG_SIDEREAL};
use crate::CompatError;

/// Days per Julian century (workspace convention).
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// [`Houses::ascmc`] slot of the Ascendant (SE: `SE_ASC`).
pub const ASCMC_ASC: usize = 0;
/// [`Houses::ascmc`] slot of the Midheaven (SE: `SE_MC`).
pub const ASCMC_MC: usize = 1;
/// [`Houses::ascmc`] slot of the ARMC / sidereal time as an angle
/// (SE: `SE_ARMC`).
pub const ASCMC_ARMC: usize = 2;
/// [`Houses::ascmc`] slot of the Vertex (SE: `SE_VERTEX`).
pub const ASCMC_VERTEX: usize = 3;
/// [`Houses::ascmc`] slot of the "equatorial ascendant" / East Point
/// (SE: `SE_EQUASC`).
pub const ASCMC_EQUASC: usize = 4;
/// [`Houses::ascmc`] slot of the co-ascendant (W. Koch) — **always 0.0
/// here** (SE: `SE_COASC1`; not implemented, see the module docs).
pub const ASCMC_COASC1: usize = 5;
/// [`Houses::ascmc`] slot of the co-ascendant (M. Munkasey) — **always
/// 0.0 here** (SE: `SE_COASC2`; not implemented).
pub const ASCMC_COASC2: usize = 6;
/// [`Houses::ascmc`] slot of the polar ascendant (M. Munkasey) —
/// **always 0.0 here** (SE: `SE_POLASC`; not implemented).
pub const ASCMC_POLASC: usize = 7;

/// A validated house-system selector: the SE-documented system
/// character plus its resolved [`HouseSystem`]. See the module docs for
/// the character table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HouseSystemChar {
    ch: char,
    system: HouseSystem,
}

impl HouseSystemChar {
    /// Validates an SE house-system character (uppercase, per the SE
    /// documentation; `'A'` and `'E'` both mean Equal).
    ///
    /// # Errors
    ///
    /// [`CompatError::UnsupportedHouseSystem`] for any character outside
    /// the module-docs table.
    pub fn from_char(ch: char) -> Result<Self, CompatError> {
        let system = match ch {
            'P' => HouseSystem::Placidus,
            'K' => HouseSystem::Koch,
            'O' => HouseSystem::Porphyry,
            'R' => HouseSystem::Regiomontanus,
            'C' => HouseSystem::Campanus,
            'A' | 'E' => HouseSystem::Equal,
            'W' => HouseSystem::WholeSign,
            other => return Err(CompatError::UnsupportedHouseSystem(other)),
        };
        Ok(Self { ch, system })
    }

    /// The validated character.
    #[must_use]
    pub const fn as_char(self) -> char {
        self.ch
    }

    /// The resolved internal house system.
    #[must_use]
    pub const fn system(self) -> HouseSystem {
        self.system
    }
}

/// The `swe_houses` output block: twelve cusps and the `ascmc` points.
///
/// **Cusp indexing** (documented crate difference): `cusps[0]` is cusp 1
/// — plain 0-based Rust indexing, *not* SE's C convention of
/// `cusps[1..=12]` with `cusps[0]` unused. `ascmc` keeps SE's slot
/// layout (see the `ASCMC_*` constants).
///
/// Units: degrees in `[0, 360)` by default; radians in `[0, 2π)` when
/// [`SEFLG_RADIANS`] was passed to [`Context::houses_ex`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Houses {
    /// House cusps 1–12 (`cusps[0]` = cusp 1).
    pub cusps: [f64; 12],
    /// The additional points, SE slot layout (`ASCMC_*` constants);
    /// slots 5–7 are always `0.0` (see the module docs).
    pub ascmc: [f64; 8],
}

impl Houses {
    /// The Ascendant ([`ASCMC_ASC`]).
    #[must_use]
    pub const fn asc(&self) -> f64 {
        self.ascmc[ASCMC_ASC]
    }

    /// The Midheaven ([`ASCMC_MC`]).
    #[must_use]
    pub const fn mc(&self) -> f64 {
        self.ascmc[ASCMC_MC]
    }

    /// The ARMC ([`ASCMC_ARMC`]).
    #[must_use]
    pub const fn armc(&self) -> f64 {
        self.ascmc[ASCMC_ARMC]
    }

    /// The Vertex ([`ASCMC_VERTEX`]).
    #[must_use]
    pub const fn vertex(&self) -> f64 {
        self.ascmc[ASCMC_VERTEX]
    }

    /// The equatorial ascendant / East Point ([`ASCMC_EQUASC`]).
    #[must_use]
    pub const fn equatorial_ascendant(&self) -> f64 {
        self.ascmc[ASCMC_EQUASC]
    }
}

/// The `houses_ex` flag bits this crate implements: sidereal cusps and
/// radians output. Any other bit is an explicit
/// [`CompatError::UnsupportedFlags`] (SE documents `SEFLG_SIDEREAL` and
/// — for `swe_houses_ex2` speed output, unimplemented here — the speed
/// bits; nothing else applies to houses).
const HOUSES_FLAG_MASK: u32 = SEFLG_SIDEREAL | SEFLG_RADIANS;

impl Context<'_> {
    /// House cusps and chart angles, mirroring
    /// `swe_houses(tjd_ut, geolat, geolon, hsys, cusps, ascmc)`:
    /// identical to [`Context::houses_ex`] with `iflag = 0`.
    ///
    /// # Errors
    ///
    /// See [`Context::houses_ex`].
    pub fn houses(
        &self,
        jd_ut: f64,
        geolat_deg: f64,
        geolon_deg: f64,
        hsys: char,
    ) -> Result<Houses, CompatError> {
        self.houses_ex(jd_ut, 0, geolat_deg, geolon_deg, hsys)
    }

    /// House cusps and chart angles with flags, mirroring
    /// `swe_houses_ex(tjd_ut, iflag, geolat, geolon, hsys, cusps,
    /// ascmc)`. `geolon_deg` is positive **east** (SE convention);
    /// `iflag` accepts [`SEFLG_SIDEREAL`] (sidereal cusps under the
    /// [`Context::set_sid_mode`] ayanamsha, Fagan/Bradley default) and
    /// [`SEFLG_RADIANS`].
    ///
    /// The time chain matches [`Context::calc_ut`] and
    /// [`Context::sidtime`]: `jd_ut` is UT1 (`dUT1 = 0` unless
    /// [`Context::set_topo_with_eop`] supplied one), TT comes from the
    /// Espenak–Meeus ΔT model, the obliquity is the **true** obliquity
    /// of date (IAU 2006 mean + full IAU 2000A nutation), and the ARMC
    /// is the local apparent sidereal time. Needs no DE ephemeris.
    ///
    /// # Errors
    ///
    /// [`CompatError::UnsupportedFlags`] for any `iflag` bit other than
    /// the two documented ones; [`CompatError::UnsupportedHouseSystem`]
    /// for an unknown `hsys`; [`CompatError::Core`] for a non-finite /
    /// unconvertible `jd_ut`; [`CompatError::Houses`] /
    /// [`CompatError::Angles`] when the system or an angle is undefined
    /// for these inputs (polar Placidus/Koch — **no silent Porphyry
    /// fallback**, see the module docs; ecliptic-horizon coincidence).
    #[allow(clippy::similar_names)] // jd_ut / jd_tt are the domain-standard names
    pub fn houses_ex(
        &self,
        jd_ut: f64,
        iflag: u32,
        geolat_deg: f64,
        geolon_deg: f64,
        hsys: char,
    ) -> Result<Houses, CompatError> {
        let unknown = iflag & !HOUSES_FLAG_MASK;
        if unknown != 0 {
            return Err(CompatError::UnsupportedFlags(unknown));
        }
        let sidereal = iflag & SEFLG_SIDEREAL != 0;
        let radians = iflag & SEFLG_RADIANS != 0;
        let system = HouseSystemChar::from_char(hsys)?;

        // Time chain (see the method docs).
        let dut1_s = self.topo.as_ref().map_or(0.0, |t| t.eop.dut1_s);
        let jd_ut1 = JulianDate::from_f64(jd_ut).add_seconds(dut1_s);
        let jd_tt = jd_tt_from_ut(jd_ut)?;
        let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;

        let gast = gast_iau2006(jd_ut1, t);
        let theta = local_sidereal_time(gast, geolon_deg * DEG2RAD);
        let nut = nutation_iau2000a(t);
        let eps = mean_obliquity_iau2006(t) + nut.deps_rad;
        let phi = geolat_deg * DEG2RAD;

        let cusp_arr = astro_cusps(system.system(), theta, phi, eps)?;
        let asc = ascendant(theta, phi, eps)?;
        let mc_val = mc(theta, eps);
        let vtx = vertex(theta, phi, eps)?;
        let equasc = east_point(theta, eps);

        // Sidereal shift: every *ecliptic longitude* moves by the
        // ayanamsha referred to the true equinox of date (the mean
        // ayanamsha + Δψ — the cusps are true-equinox longitudes, and a
        // star-anchored zodiac must not carry the nutation wobble; this
        // matches SE's output convention, see
        // `crate::context::calc::sidereal_offset_rad`). The ARMC is a
        // right-ascension-like angle and stays.
        let shift = if sidereal {
            ayanamsha_rad(self.sidereal_mode().to_ayanamsha(), t) + nut.dpsi_rad
        } else {
            0.0
        };
        let scale = if radians { 1.0 } else { RAD2DEG };
        let out = |ecl_lon_rad: f64| normalize_0_two_pi(ecl_lon_rad - shift) * scale;

        let mut cusps_out = [0.0_f64; 12];
        for (slot, cusp) in cusps_out.iter_mut().zip(cusp_arr) {
            *slot = out(cusp);
        }
        let mut ascmc = [0.0_f64; 8];
        ascmc[ASCMC_ASC] = out(asc);
        ascmc[ASCMC_MC] = out(mc_val);
        ascmc[ASCMC_ARMC] = theta * scale;
        ascmc[ASCMC_VERTEX] = out(vtx);
        ascmc[ASCMC_EQUASC] = out(equasc);

        Ok(Houses {
            cusps: cusps_out,
            ascmc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HouseSystemChar, ASCMC_ARMC, ASCMC_ASC, ASCMC_COASC1, ASCMC_MC};
    use crate::flags::{SEFLG_RADIANS, SEFLG_SIDEREAL};
    use crate::{CompatError, Context};
    use oxiephemeris_astro::houses::HouseSystem;
    use oxiephemeris_bodies::frames::nutation_iau2000a;
    use oxiephemeris_core::angle::RAD2DEG;

    #[test]
    fn house_system_char_table() {
        for (ch, sys) in [
            ('P', HouseSystem::Placidus),
            ('K', HouseSystem::Koch),
            ('O', HouseSystem::Porphyry),
            ('R', HouseSystem::Regiomontanus),
            ('C', HouseSystem::Campanus),
            ('A', HouseSystem::Equal),
            ('E', HouseSystem::Equal),
            ('W', HouseSystem::WholeSign),
        ] {
            let parsed = HouseSystemChar::from_char(ch)
                .unwrap_or_else(|e| panic!("char {ch:?} should parse: {e}"));
            assert_eq!(parsed.system(), sys);
            assert_eq!(parsed.as_char(), ch);
        }
        for bad in ['B', 'G', 'H', 'M', 'T', 'U', 'V', 'X', 'Y', 'p', 'Z', '?'] {
            assert!(matches!(
                HouseSystemChar::from_char(bad),
                Err(CompatError::UnsupportedHouseSystem(c)) if c == bad
            ));
        }
    }

    #[test]
    fn houses_needs_no_ephemeris_and_asc_mc_match_cusps() {
        let ctx = Context::new();
        // 1999-03-31 05:15 UT, Sapporo-ish coordinates.
        let jd_ut = Context::julday(1999, 3, 30, 20.25, crate::flags::SE_GREG_CAL)
            .unwrap_or_else(|e| panic!("{e}"));
        let h = ctx
            .houses(jd_ut, 43.06, 141.35, 'P')
            .unwrap_or_else(|e| panic!("houses failed: {e}"));
        // Quadrant system: cusp 1 = ASC, cusp 10 = MC, opposition holds.
        assert!((h.cusps[0] - h.ascmc[ASCMC_ASC]).abs() < 1e-12);
        assert!((h.cusps[9] - h.ascmc[ASCMC_MC]).abs() < 1e-12);
        for i in 0..6 {
            let diff = (h.cusps[i + 6] - h.cusps[i]).rem_euclid(360.0);
            assert!(
                (diff - 180.0).abs() < 1e-9,
                "cusp {} vs {}: {diff}",
                i + 7,
                i + 1
            );
        }
        for v in h.cusps.iter().chain(h.ascmc.iter()) {
            assert!((0.0..360.0).contains(v), "out of range: {v}");
        }
        assert!((h.ascmc[ASCMC_COASC1]).abs() < 1e-15, "slot 5 must be 0");
    }

    #[test]
    #[allow(clippy::similar_names)] // jd_ut / jd_tt are the domain-standard names
    fn sidereal_flag_shifts_cusps_by_the_ayanamsha() {
        let ctx = Context::new();
        let jd_ut = 2_451_545.0;
        let trop = ctx
            .houses_ex(jd_ut, 0, 35.0, 139.0, 'E')
            .unwrap_or_else(|e| panic!("{e}"));
        let sid = ctx
            .houses_ex(jd_ut, SEFLG_SIDEREAL, 35.0, 139.0, 'E')
            .unwrap_or_else(|e| panic!("{e}"));
        // The shift is the true-equinox ayanamsha: mean ayanamsha + Δψ
        // (see the sidereal-shift comment in `houses_ex`).
        let jd_tt = jd_ut + 64.0 / 86_400.0; // ~ TT of this jd_ut
        let ay = ctx.get_ayanamsa(jd_tt).unwrap_or_else(|e| panic!("{e}"));
        let t = (jd_tt - 2_451_545.0) / 36_525.0;
        let expected = ay + nutation_iau2000a(t).dpsi_rad * RAD2DEG;
        let diff = (trop.cusps[0] - sid.cusps[0]).rem_euclid(360.0);
        assert!(
            (diff - expected).abs() < 2e-4,
            "cusp shift {diff} vs true-equinox ayanamsha {expected}"
        );
        // ARMC is RA-like and must NOT shift.
        assert!((trop.ascmc[ASCMC_ARMC] - sid.ascmc[ASCMC_ARMC]).abs() < 1e-12);
    }

    #[test]
    fn radians_flag_scales_output() {
        let ctx = Context::new();
        let deg = ctx
            .houses_ex(2_451_545.0, 0, 51.5, 0.0, 'R')
            .unwrap_or_else(|e| panic!("{e}"));
        let rad = ctx
            .houses_ex(2_451_545.0, SEFLG_RADIANS, 51.5, 0.0, 'R')
            .unwrap_or_else(|e| panic!("{e}"));
        for (d, r) in deg.cusps.iter().zip(rad.cusps.iter()) {
            assert!((d.to_radians() - r).abs() < 1e-12);
        }
    }

    #[test]
    fn polar_placidus_is_a_typed_error_not_a_fallback() {
        let ctx = Context::new();
        match ctx.houses(2_451_545.0, 78.0, 15.0, 'P') {
            Err(CompatError::Houses(_)) => {}
            other => panic!("expected CompatError::Houses, got {other:?}"),
        }
        // ... while Campanus stays defined at the same latitude.
        assert!(ctx.houses(2_451_545.0, 78.0, 15.0, 'C').is_ok());
    }

    #[test]
    fn unknown_houses_flag_is_rejected() {
        let ctx = Context::new();
        assert!(matches!(
            ctx.houses_ex(2_451_545.0, crate::flags::SEFLG_SPEED, 35.0, 139.0, 'P'),
            Err(CompatError::UnsupportedFlags(_))
        ));
    }
}
