//! SE-documented calendar and calculation-flag bit constants, plus the
//! internal decoder that routes them onto `oxiephemeris-bodies`'
//! `Center`/`Frame`/`Options` machinery.
//!
//! # Provenance
//!
//! Every numeric value below is copied from the **published Swiss
//! Ephemeris programmer's documentation** (Astrodienst, "Programming
//! interface to the Swiss Ephemeris", <https://www.astro.com/swisseph/swephprg.htm>,
//! sections 3.2 "Planets and stars" and 3.3.1 "Calculation flags",
//! consulted 2026-07-06) — never from the SE source distribution itself
//! (`swephexp.h`), which this project's clean-room policy forbids
//! opening. Per `docs/design.md` section 1's license-boundary table, SE
//! *documentation* (API shapes, flag/constant meanings) is fair game for
//! compatibility work; only SE *source code* is off limits. Flag bit
//! values are plain interface constants, not algorithm descriptions.
//!
//! Not every documented bit is given independent semantics here — see
//! the internal decoder and the per-bit doc comments for which ones this crate
//! actually implements versus accepts-and-ignores versus rejects.

use oxiephemeris_bodies::apparent::Center;

use crate::CompatError;

// ---------------------------------------------------------------------
// Calendar flag (`swe_julday`/`swe_revjul` `gregflag`), swephprg.htm §9.1.
// ---------------------------------------------------------------------

/// Julian calendar, for the `gregflag`/`calflag` argument of
/// [`crate::Context::julday`] / [`crate::Context::revjul`].
pub const SE_JUL_CAL: i32 = 0;

/// Proleptic Gregorian calendar (the common default), for the same
/// argument.
pub const SE_GREG_CAL: i32 = 1;

// ---------------------------------------------------------------------
// Calculation flags (`swe_calc`/`swe_calc_ut` `iflag`), swephprg.htm §3.3.1.
// ---------------------------------------------------------------------

/// Use the JPL ephemeris. **Accepted and ignored**: this crate has
/// exactly one ephemeris backend (a JPL DE classic-binary file via
/// [`oxiephemeris_de`]), so there is no alternative source to select.
/// Real-world SE calling code very commonly sets one of
/// [`SEFLG_JPLEPH`]/[`SEFLG_SWIEPH`]/[`SEFLG_MOSEPH`] out of habit (SE's
/// own default is [`SEFLG_SWIEPH`]); rejecting the single most common SE
/// flag value would make this crate unusable for migration, so these
/// three ephemeris-selector bits are a deliberate, documented exception
/// to the "never silent" rule that governs every other flag below.
pub const SEFLG_JPLEPH: u32 = 1;

/// Use the Swiss Ephemeris file-based ephemeris (SE's own documented
/// default). **Accepted and ignored** — see [`SEFLG_JPLEPH`].
pub const SEFLG_SWIEPH: u32 = 2;

/// Use the Moshier analytical-series ephemeris. **Accepted and
/// ignored** — see [`SEFLG_JPLEPH`]. (`oxiephemeris` has no Moshier-style
/// fallback yet.)
pub const SEFLG_MOSEPH: u32 = 4;

/// Heliocentric position ([`Center::Heliocentric`]).
pub const SEFLG_HELCTR: u32 = 8;

/// "True"/geometric position rather than the apparent position: the
/// body is evaluated at the observation epoch itself, with **no
/// light-time retardation** and no annual aberration or solar
/// gravitational deflection — SE's documented distinction from the
/// *astrometric* position ([`SEFLG_ASTROMETRIC`]), which keeps the
/// light-time correction and drops only aberration/deflection. Maps to
/// [`oxiephemeris_bodies::apparent::Options::light_time`] `= false`
/// plus the [`SEFLG_NOABERR`] `|` [`SEFLG_NOGDEFL`] effect.
pub const SEFLG_TRUEPOS: u32 = 16;

/// No precession: give the J2000.0 mean-equator-and-equinox frame
/// instead of the frame of date. See [`crate::Context::calc`] for how this crate
/// derives that frame (frame-bias-only rotation evaluated at `t = 0`,
/// since [`oxiephemeris_bodies::apparent::Frame`] has no "frozen epoch"
/// variant of its own).
pub const SEFLG_J2000: u32 = 32;

/// No nutation: mean equinox of date instead of true equinox of date.
pub const SEFLG_NONUT: u32 = 64;

/// Speed from 3 positions — SE's own documentation calls this a legacy
/// path ("do not use it, `SEFLG_SPEED` is faster and more precise").
/// This crate does not implement two distinct speed algorithms; setting
/// this bit (with or without [`SEFLG_SPEED`]) is treated identically to
/// [`SEFLG_SPEED`] and documented as such — never silently a *different*
/// number, just the same central-difference speed this crate always
/// uses when any speed bit is set.
pub const SEFLG_SPEED3: u32 = 128;

/// Compute and return daily speeds (see [`crate::Context::calc`] for the
/// central-difference method and step size used).
pub const SEFLG_SPEED: u32 = 256;

/// Turn off gravitational light deflection.
pub const SEFLG_NOGDEFL: u32 = 512;

/// Turn off annual aberration of light.
pub const SEFLG_NOABERR: u32 = 1024;

/// Astrometric position: light-time corrected, but without aberration or
/// light deflection. Documented composite of [`SEFLG_NOABERR`] `|`
/// [`SEFLG_NOGDEFL`] (not an independently SE-documented numeric literal
/// in the source consulted; the composite is a directly-derived `OR` of
/// two independently-confirmed bits, so no separate citation risk
/// applies).
pub const SEFLG_ASTROMETRIC: u32 = SEFLG_NOABERR | SEFLG_NOGDEFL;

/// Equatorial coordinates (right ascension / declination) instead of
/// ecliptic (longitude / latitude).
pub const SEFLG_EQUATORIAL: u32 = 2048;

/// Cartesian `(x, y, z)` (and, with [`SEFLG_SPEED`]/[`SEFLG_SPEED3`],
/// `(vx, vy, vz)`) instead of spherical coordinates, in AU (AU/day).
pub const SEFLG_XYZ: u32 = 4096;

/// Angular output in radians instead of degrees. A documented no-op when
/// combined with [`SEFLG_XYZ`] (Cartesian output has no "degrees" to
/// begin with).
pub const SEFLG_RADIANS: u32 = 8192;

/// Barycentric position ([`Center::Barycentric`]).
pub const SEFLG_BARYCTR: u32 = 16384;

/// Topocentric position; requires [`crate::Context::set_topo`] to have
/// been called first ([`CompatError::TopoNotSet`] otherwise).
pub const SEFLG_TOPOCTR: u32 = 32768;

/// Sidereal position: subtract the ayanamsha of
/// [`crate::Context::set_sid_mode`] (or the documented Fagan/Bradley
/// default if never called) from ecliptic longitudes.
pub const SEFLG_SIDEREAL: u32 = 65536;

/// ICRS **modifier**: leave the ICRS → J2000 frame-bias rotation out of
/// the output frame's rotation chain. SE's observed behavior (verified
/// against SE 2.10.03 output on the same DE440 file — output comparison
/// only, per the clean-room policy):
///
/// * together with [`SEFLG_J2000`]: the raw ICRS frame itself
///   ([`oxiephemeris_bodies::apparent::Frame::Icrs`], no rotation at
///   all) — J2000 output with the ≈ 23 mas frame bias omitted;
/// * **alone**: still the frame *of date*, but with the frame-bias
///   factor removed from the precession(-nutation) chain — the output
///   differs from the plain of-date frame only at the frame-bias level
///   (≈ 23 mas), not by the precession span.
///
/// This crate reproduces both cases (see [`crate::context::calc`]).
pub const SEFLG_ICRS: u32 = 131_072;

/// JPL-Horizons 1962+ frame-tie reproduction mode. **Not implemented**:
/// setting this bit is a [`CompatError::UnsupportedFlags`]. (This is a
/// distinct, narrower thing from the documented −53 mas Horizons
/// equinox tie that `oxiephemeris-bodies`' own test suite measures
/// against — see `crates/oxiephemeris-bodies/tests/horizons_spotcheck.rs`
/// — this flag would additionally have to reproduce Horizons' specific
/// IAU76/80-based frame chain for 1962–2003, which is out of scope.)
pub const SEFLG_DPSIDEPS_1980: u32 = 262_144;

/// Alias of [`SEFLG_DPSIDEPS_1980`] (SE documents the same numeric value
/// under both names). **Not implemented** — see
/// [`SEFLG_DPSIDEPS_1980`].
pub const SEFLG_JPLHOR: u32 = SEFLG_DPSIDEPS_1980;

/// Approximate JPL-Horizons frame reproduction. **Not implemented** —
/// see [`SEFLG_DPSIDEPS_1980`].
pub const SEFLG_JPLHOR_APPROX: u32 = 524_288;

/// Center of body (rather than planetary-system barycenter) for the
/// outer planets. **Not implemented**: `oxiephemeris-bodies`'
/// [`oxiephemeris_bodies::apparent::Target`] documents that Mars..Pluto
/// are always the DE file's system-barycenter series (see its
/// "DE barycenter caveat"); there is no center-of-body series to switch
/// to yet.
pub const SEFLG_CENTER_BODY: u32 = 1_048_576;

/// Every calculation-flag bit this module assigns a documented meaning
/// to (implemented, accepted-and-ignored, or explicitly rejected). A bit
/// outside this mask is unrecognized and always a
/// [`CompatError::UnsupportedFlags`] — never silently dropped.
const KNOWN_MASK: u32 = SEFLG_JPLEPH
    | SEFLG_SWIEPH
    | SEFLG_MOSEPH
    | SEFLG_HELCTR
    | SEFLG_TRUEPOS
    | SEFLG_J2000
    | SEFLG_NONUT
    | SEFLG_SPEED3
    | SEFLG_SPEED
    | SEFLG_NOGDEFL
    | SEFLG_NOABERR
    | SEFLG_EQUATORIAL
    | SEFLG_XYZ
    | SEFLG_RADIANS
    | SEFLG_BARYCTR
    | SEFLG_TOPOCTR
    | SEFLG_SIDEREAL
    | SEFLG_ICRS
    | SEFLG_DPSIDEPS_1980
    | SEFLG_JPLHOR_APPROX
    | SEFLG_CENTER_BODY;

/// Which fixed reference frame a decoded [`Options`](
/// oxiephemeris_bodies::apparent::Options) request resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameChoice {
    /// The frame of date: true or mean equinox, per `SEFLG_NONUT`.
    OfDate { true_equinox: bool },
    /// The J2000.0 mean-equator-and-equinox frame, reproduced by this
    /// crate as a fixed (`t = 0`) frame-bias rotation applied to the raw
    /// ICRS position — see [`crate::context::calc`] for the rotation
    /// matrices used.
    J2000,
    /// The raw ICRS/GCRS frame
    /// ([`oxiephemeris_bodies::apparent::Frame::Icrs`]): selected by
    /// `SEFLG_J2000 | SEFLG_ICRS` (J2000 output with the frame bias
    /// omitted — see [`SEFLG_ICRS`]).
    Icrs,
}

/// A fully decoded `iflag`, ready to drive
/// [`oxiephemeris_bodies::apparent::Options`] plus this crate's own
/// output-shaping (units, sidereal shift, Cartesian/speed handling).
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // a decoded bitmask: the bools *are* the domain
pub(crate) struct DecodedFlags {
    pub center: Center,
    pub topocentric: bool,
    pub frame: FrameChoice,
    /// `SEFLG_ICRS` without `SEFLG_J2000`: of-date output with the
    /// frame-bias factor removed from the rotation chain (see
    /// [`SEFLG_ICRS`]). Only meaningful with [`FrameChoice::OfDate`].
    pub icrs_no_bias: bool,
    pub equatorial: bool,
    pub aberration: bool,
    pub deflection: bool,
    /// `SEFLG_TRUEPOS`: geometric position, no light-time retardation
    /// (maps to `Options::light_time = false`).
    pub truepos: bool,
    pub with_speed: bool,
    pub xyz: bool,
    pub radians: bool,
    pub sidereal: bool,
}

/// Decodes a raw `iflag` bitmask, validating every bit and every
/// documented-conflicting combination.
///
/// # Errors
///
/// [`CompatError::UnsupportedFlags`] for any bit outside [`KNOWN_MASK`]
/// or for a bit this crate explicitly does not implement
/// ([`SEFLG_DPSIDEPS_1980`]/[`SEFLG_JPLHOR`]/[`SEFLG_JPLHOR_APPROX`]/
/// [`SEFLG_CENTER_BODY`]); [`CompatError::ConflictingFlags`] for more
/// than one of `HELCTR`/`BARYCTR`/`TOPOCTR`, or for `SIDEREAL` with
/// `EQUATORIAL` or `XYZ` (the sidereal shift is only defined for a
/// spherical ecliptic longitude); [`CompatError::TopoNotSet`] if
/// `TOPOCTR` is set but [`crate::Context::set_topo`] was never called.
pub(crate) fn decode(iflag: u32, topo_is_set: bool) -> Result<DecodedFlags, CompatError> {
    let unknown = iflag & !KNOWN_MASK;
    if unknown != 0 {
        return Err(CompatError::UnsupportedFlags(unknown));
    }
    for (bit, name) in [
        (SEFLG_DPSIDEPS_1980, "SEFLG_DPSIDEPS_1980/SEFLG_JPLHOR"),
        (SEFLG_JPLHOR_APPROX, "SEFLG_JPLHOR_APPROX"),
        (SEFLG_CENTER_BODY, "SEFLG_CENTER_BODY"),
    ] {
        if iflag & bit != 0 {
            return Err(CompatError::UnsupportedFlagName(name));
        }
    }

    let helctr = iflag & SEFLG_HELCTR != 0;
    let baryctr = iflag & SEFLG_BARYCTR != 0;
    let topoctr = iflag & SEFLG_TOPOCTR != 0;
    if u32::from(helctr) + u32::from(baryctr) + u32::from(topoctr) > 1 {
        return Err(CompatError::ConflictingFlags(
            "at most one of SEFLG_HELCTR / SEFLG_BARYCTR / SEFLG_TOPOCTR may be set",
        ));
    }
    if topoctr && !topo_is_set {
        return Err(CompatError::TopoNotSet);
    }
    let center = if helctr {
        Center::Heliocentric
    } else if baryctr {
        Center::Barycentric
    } else {
        Center::Geocentric
    };

    let j2000 = iflag & SEFLG_J2000 != 0;
    let icrs = iflag & SEFLG_ICRS != 0;
    let equatorial = iflag & SEFLG_EQUATORIAL != 0;
    // SEFLG_ICRS is a bias-omission *modifier* (see its docs): with
    // SEFLG_J2000 it selects the raw ICRS frame; alone it keeps the
    // of-date frame and only strips the frame-bias factor.
    let frame = if j2000 && icrs {
        FrameChoice::Icrs
    } else if j2000 {
        FrameChoice::J2000
    } else {
        FrameChoice::OfDate {
            true_equinox: iflag & SEFLG_NONUT == 0,
        }
    };
    let icrs_no_bias = icrs && !j2000;

    let truepos = iflag & SEFLG_TRUEPOS != 0;
    let no_aberr = iflag & SEFLG_NOABERR != 0 || truepos;
    let no_gdefl = iflag & SEFLG_NOGDEFL != 0 || truepos;

    let sidereal = iflag & SEFLG_SIDEREAL != 0;
    let xyz = iflag & SEFLG_XYZ != 0;
    if sidereal && equatorial {
        return Err(CompatError::ConflictingFlags(
            "SEFLG_SIDEREAL only shifts an ecliptic longitude; combine it with the default \
             ecliptic output, not SEFLG_EQUATORIAL",
        ));
    }
    if sidereal && xyz {
        return Err(CompatError::ConflictingFlags(
            "SEFLG_SIDEREAL is not defined for SEFLG_XYZ Cartesian output",
        ));
    }
    if sidereal && (j2000 || icrs) {
        return Err(CompatError::ConflictingFlags(
            "SEFLG_SIDEREAL subtracts an of-date ayanamsha: subtracting it from a fixed \
             SEFLG_J2000 / SEFLG_ICRS longitude would be neither tropical nor sidereal",
        ));
    }

    Ok(DecodedFlags {
        center,
        topocentric: topoctr,
        frame,
        icrs_no_bias,
        equatorial,
        aberration: !no_aberr,
        deflection: !no_gdefl,
        truepos,
        with_speed: iflag & (SEFLG_SPEED | SEFLG_SPEED3) != 0,
        xyz,
        radians: iflag & SEFLG_RADIANS != 0,
        sidereal,
    })
}

#[cfg(test)]
mod tests {
    use super::{decode, FrameChoice, SEFLG_EQUATORIAL, SEFLG_HELCTR, SEFLG_SPEED, SEFLG_TOPOCTR};
    use crate::CompatError;
    use oxiephemeris_bodies::apparent::Center;

    #[test]
    fn documented_numeric_values() {
        // swephprg.htm sections 3.2 / 3.3.1 (see module docs for the
        // exact citation).
        assert_eq!(super::SEFLG_JPLEPH, 1);
        assert_eq!(super::SEFLG_SWIEPH, 2);
        assert_eq!(super::SEFLG_MOSEPH, 4);
        assert_eq!(super::SEFLG_HELCTR, 8);
        assert_eq!(super::SEFLG_TRUEPOS, 16);
        assert_eq!(super::SEFLG_J2000, 32);
        assert_eq!(super::SEFLG_NONUT, 64);
        assert_eq!(super::SEFLG_SPEED3, 128);
        assert_eq!(super::SEFLG_SPEED, 256);
        assert_eq!(super::SEFLG_NOGDEFL, 512);
        assert_eq!(super::SEFLG_NOABERR, 1024);
        assert_eq!(super::SEFLG_EQUATORIAL, 2048);
        assert_eq!(super::SEFLG_XYZ, 4096);
        assert_eq!(super::SEFLG_RADIANS, 8192);
        assert_eq!(super::SEFLG_BARYCTR, 16384);
        assert_eq!(super::SEFLG_TOPOCTR, 32768);
        assert_eq!(super::SEFLG_SIDEREAL, 65536);
        assert_eq!(super::SEFLG_ICRS, 131_072);
        assert_eq!(super::SEFLG_DPSIDEPS_1980, 262_144);
        assert_eq!(super::SEFLG_JPLHOR, super::SEFLG_DPSIDEPS_1980);
        assert_eq!(super::SEFLG_JPLHOR_APPROX, 524_288);
        assert_eq!(super::SEFLG_CENTER_BODY, 1_048_576);
        assert_eq!(super::SE_JUL_CAL, 0);
        assert_eq!(super::SE_GREG_CAL, 1);
    }

    #[test]
    fn unknown_bit_is_an_explicit_error() {
        let huge_unknown_bit = 1_u32 << 30;
        match decode(huge_unknown_bit, false) {
            Err(CompatError::UnsupportedFlags(bits)) => assert_eq!(bits, huge_unknown_bit),
            other => panic!("expected UnsupportedFlags, got {other:?}"),
        }
    }

    #[test]
    fn ephemeris_selector_bits_are_silently_accepted() {
        let decoded = decode(super::SEFLG_SWIEPH | super::SEFLG_JPLEPH, false)
            .unwrap_or_else(|e| panic!("unexpected error: {e}"));
        assert_eq!(decoded.center, Center::Geocentric);
    }

    #[test]
    fn topoctr_without_set_topo_is_an_error() {
        assert!(matches!(
            decode(SEFLG_TOPOCTR, false),
            Err(CompatError::TopoNotSet)
        ));
        assert!(decode(SEFLG_TOPOCTR, true).is_ok());
    }

    #[test]
    fn conflicting_centers_are_rejected() {
        assert!(matches!(
            decode(SEFLG_HELCTR | SEFLG_TOPOCTR, true),
            Err(CompatError::ConflictingFlags(_))
        ));
    }

    #[test]
    fn default_frame_is_of_date_true_equinox() {
        let decoded = decode(0, false).unwrap_or_else(|e| panic!("unexpected error: {e}"));
        assert_eq!(decoded.frame, FrameChoice::OfDate { true_equinox: true });
        assert!(decoded.aberration && decoded.deflection);
        assert!(!decoded.with_speed && !decoded.xyz && !decoded.radians && !decoded.sidereal);
    }

    #[test]
    fn nonut_selects_mean_equinox() {
        let decoded =
            decode(super::SEFLG_NONUT, false).unwrap_or_else(|e| panic!("unexpected error: {e}"));
        assert_eq!(
            decoded.frame,
            FrameChoice::OfDate {
                true_equinox: false
            }
        );
    }

    #[test]
    fn truepos_disables_aberration_and_deflection() {
        let decoded = decode(super::SEFLG_TRUEPOS, false).unwrap_or_else(|e| panic!("{e}"));
        assert!(!decoded.aberration && !decoded.deflection);
    }

    #[test]
    fn speed3_behaves_like_speed() {
        let decoded = decode(super::SEFLG_SPEED3, false).unwrap_or_else(|e| panic!("{e}"));
        assert!(decoded.with_speed);
        let decoded2 = decode(SEFLG_SPEED, false).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(decoded.with_speed, decoded2.with_speed);
    }

    #[test]
    fn icrs_is_a_bias_omission_modifier() {
        // Alone: still of-date, but flagged for bias omission.
        let alone = decode(super::SEFLG_ICRS, false).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(alone.frame, FrameChoice::OfDate { .. }));
        assert!(alone.icrs_no_bias);
        // With J2000: the raw ICRS frame, no post-hoc bias stripping.
        let with_j2000 =
            decode(super::SEFLG_J2000 | super::SEFLG_ICRS, false).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(with_j2000.frame, FrameChoice::Icrs));
        assert!(!with_j2000.icrs_no_bias);
        // Plain J2000 keeps the bias in.
        let plain = decode(super::SEFLG_J2000, false).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(plain.frame, FrameChoice::J2000));
        assert!(!plain.icrs_no_bias);
    }

    #[test]
    fn sidereal_conflicts_with_equatorial_and_xyz() {
        assert!(matches!(
            decode(super::SEFLG_SIDEREAL | SEFLG_EQUATORIAL, false),
            Err(CompatError::ConflictingFlags(_))
        ));
        assert!(matches!(
            decode(super::SEFLG_SIDEREAL | super::SEFLG_XYZ, false),
            Err(CompatError::ConflictingFlags(_))
        ));
        assert!(decode(super::SEFLG_SIDEREAL, false).is_ok());
    }

    #[test]
    fn sidereal_conflicts_with_fixed_epoch_frames() {
        // Subtracting an of-date ayanamsha from a fixed J2000/ICRS
        // longitude would produce a mixed-epoch value.
        assert!(matches!(
            decode(super::SEFLG_SIDEREAL | super::SEFLG_J2000, false),
            Err(CompatError::ConflictingFlags(_))
        ));
        assert!(matches!(
            decode(super::SEFLG_SIDEREAL | super::SEFLG_ICRS, false),
            Err(CompatError::ConflictingFlags(_))
        ));
    }

    #[test]
    fn unimplemented_bits_are_named_errors() {
        for bit in [
            super::SEFLG_DPSIDEPS_1980,
            super::SEFLG_JPLHOR_APPROX,
            super::SEFLG_CENTER_BODY,
        ] {
            assert!(matches!(
                decode(bit, false),
                Err(CompatError::UnsupportedFlagName(_))
            ));
        }
    }
}
