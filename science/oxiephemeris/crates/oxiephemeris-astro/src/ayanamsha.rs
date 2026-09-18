//! Sidereal-zodiac offsets ("ayanamshas") from documented, cited
//! definitions.
//!
//! # Model
//!
//! Every [`Ayanamsha`] variant is modeled as a single documented anchor
//! pair `(t0, value_at_t0)` — a specific epoch and the ayanamsha value
//! published for it — propagated to any other epoch `t` by the
//! **IAU 2006 general precession in ecliptic longitude**, `p_A(t)`:
//!
//! ```text
//! ayanamsha(t) = value_at_t0 + [p_A(t) - p_A(t0)]   (mod 2*pi)
//! ```
//!
//! `p_A` is the P03 precession-theory quantity from Capitaine, Wallace &
//! Chapront (2003), A&A 412, 567–586, "Expressions for IAU 2000
//! precession quantities", eq. (39) (arcseconds, `t` in Julian centuries
//! TDB/TT — see [`oxiephemeris_bodies::frames`]'s "Time argument"
//! section for the TT-for-TDB substitution used throughout this
//! workspace, negligible at these accuracies):
//!
//! ```text
//! p_A = 5028.796195" t + 1.1054348" t^2 + 0.00007964" t^3
//!       - 0.000023857" t^4 - 0.0000000383" t^5
//! ```
//!
//! This is the same P03 solution [`oxiephemeris_bodies::frames`]
//! implements ([`oxiephemeris_bodies::frames::mean_obliquity_iau2006`]
//! carries the `eps_A` polynomial from this *same* eq. (39) group, and
//! [`oxiephemeris_bodies::frames::fw_angles_iau2006`] the
//! Fukushima–Williams `psi_bar`/`phi_bar`/`gamma_bar` angles) — but
//! `p_A` itself (the classical "general precession in longitude",
//! accumulated along the *fixed* J2000 ecliptic) is **not** one of the
//! quantities `bodies` exposes; it is reimplemented here directly from
//! the cited equation rather than derived from the Fukushima–Williams
//! angles, which parameterize a different rotation (referred to the
//! GCRS pole, with frame bias folded in) and would not reproduce
//! `p_A` — using them instead would silently change the ~50.29
//! arcsec/year ayanamsha drift rate to accord with `psi_bar`'s own
//! (`5038.481507"` per century) rate.
//!
//! # Honesty about accuracy
//!
//! This single-anchor, single-precession-model design is a deliberate
//! simplification (documented, as required by this crate's mandate):
//! real-world implementations of these ayanamshas (including Swiss
//! Ephemeris, and the original defining committees/authors) do not all
//! use the modern IAU 2006 `p_A` polynomial to extrapolate — some use
//! older precession constants (e.g. Newcomb-era values close to but not
//! identical to `50.2879"`/year), some fold in nutation for a "true"
//! rather than "mean" value, and some re-anchor periodically. The
//! result is that this crate's values, extrapolated decades away from
//! their anchor epoch, can differ from other software's by more than
//! the sub-arcsecond level this workspace targets elsewhere — up to a
//! few arcminutes for [`Ayanamsha::Lahiri`] by 2000 (its anchor is
//! 1956), sub-arcminute for [`Ayanamsha::FaganBradley`] (its anchor is
//! 1950, so the multi-decade extrapolation is shorter). For
//! [`Ayanamsha::Krishnamurti`] and [`Ayanamsha::Raman`] the anchors
//! are older (1900/1912) but the divergence is dominated by the rate
//! convention, and stays *small*: their definers' constant table
//! rates (50.2388"/yr and ~50 1/3"/yr respectively) differ from
//! `p_A`'s ~50.29"/yr by only ~4-5" per century, so this crate tracks
//! their published tables to roughly ten arcseconds across
//! 1900-2050 — the few-arcminute spreads seen between popular KP/Raman
//! calculators come from *their* differing re-anchorings, not from
//! this model. `astro/tests/ayanamsha.rs` prints the actual residual
//! against independently published spot values so this is measured,
//! not asserted away.
//!
//! # Anchor citations
//!
//! - [`Ayanamsha::Lahiri`] ("Chitrapaksha"): the Government of India's
//!   Calendar Reform Committee (1955, chaired by Meghnad Saha, with
//!   N.C. Lahiri as a member) fixed the ayanamsha at **23 deg 15' 00"**
//!   for **1956 March 21, 0h**, subsequently adopted by the Indian
//!   Astronomical Ephemeris (Positional Astronomy Centre) as the
//!   national standard; this is the widely-republished historical
//!   decree value and predates any ephemeris software. (A 1985
//!   refinement to a nutation-inclusive "true" value,
//!   23 deg 15' 00.658", is not implemented here — this crate models
//!   only the original mean-value decree, consistent with the
//!   mean-precession-only model above.)
//! - [`Ayanamsha::FaganBradley`]: defined by the mean sidereal longitude
//!   of the "Synetic Vernal Point" (SVP) at the Besselian epoch
//!   **B1950.0** (JD 2 433 282.4235) as **335 deg 57' 28.64"** (tropical
//!   longitude), equivalently an ayanamsha of **24 deg 02' 31.36"** —
//!   the published definition of the Fagan/Bradley system (Cyril Fagan
//!   and Donald Bradley's Western siderealist tradition; per, e.g., the
//!   "SVP" entry of Kepler software's published astrological dictionary,
//!   independent of this workspace's clean-room boundary since it
//!   documents a numeric convention, not an algorithm).
//! - [`Ayanamsha::Krishnamurti`]: the K.S. Krishnamurti ("KP") system's
//!   published anchor, **22 deg 21' 50"** at **1900 January 1, 0h**,
//!   from the ayanamsha table Krishnamurti published with his
//!   *Krishnamurti Padhdhati* readers/ephemeris (stated growth rate
//!   50.2388"/year — not used here, see the model note in "Honesty
//!   about accuracy" below). The anchor pair is republished verbatim
//!   across the KP literature (e.g. V. Subramanian, *KP Ayanamsa — An
//!   Analysis*) and in mainstream astrology-software documentation
//!   (e.g. Solar Fire documents exactly "22-21-50, epoch Jan 1,
//!   1900"), so it is treated here as the KP system's defining pair.
//! - [`Ayanamsha::Raman`]: the B.V. Raman system's anchor, **21 deg
//!   11' 29"** for the year **1912** (pinned here to 1912 January 1,
//!   0h, JD 2419402.5), from Raman's own *Hindu Predictive Astrology*,
//!   ch. X, whose worked example subtracts "Ayanamsa for 1912 …
//!   21 11 29" and states the rate as "50 1/2 seconds of arc per
//!   year", explicitly at year granularity ("precession for odd days
//!   may conveniently be omitted") — so pinning the anchor to Jan 1
//!   rather than elsewhere within 1912 is an arbitrary sub-arcminute
//!   choice, below the fidelity of the definition itself. Consistency
//!   checks: backdating this anchor at the 50 1/3"/year rate the
//!   secondary literature attributes to Raman reaches zero near
//!   **397 CE**, the published Raman zero year; and the resulting
//!   ayanamsha sits ~1 deg 26' below [`Ayanamsha::Lahiri`], matching
//!   the published "about 1.5 deg less than Lahiri" relationship.
//!   (An earlier draft of this file mistakenly carried a Lahiri-style
//!   1900.0 value, 22 deg 27' 37", as the Raman anchor; the pair
//!   difference tests in `astro/tests/ayanamsha.rs` now guard against
//!   that class of transposition.)
//! - [`Ayanamsha::J2000Zero`]: **our own convention**, not a published
//!   ayanamsha at all — the ayanamsha is defined to be exactly zero at
//!   J2000.0 and to grow from there by `p_A` alone. Useful as a
//!   precession-only baseline / test harness (e.g. to isolate `p_A`
//!   itself via [`ayanamsha_rad`]) and for software that wants a
//!   sidereal frame tied to the J2000 tropical zero point with no
//!   historical offset.
//! - [`Ayanamsha::Custom`]: caller-supplied anchor, `t0_jd_tt` (Julian
//!   Date, TT) and `value_at_t0_rad` (radians).

use libm::fmod;

/// Arcseconds to radians: `pi / (180 * 3600)`.
const ARCSEC_TO_RAD: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// Degrees to radians: `pi / 180`.
const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// One full turn, radians.
const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

/// Julian Date (TT) of the J2000.0 epoch.
const JD_J2000_TT: f64 = 2_451_545.0;

/// Days per Julian century.
const DAYS_PER_JULIAN_CENTURY: f64 = 36_525.0;

/// Converts a Julian Date (TT) to Julian centuries TT since J2000.0.
const fn centuries_since_j2000(jd_tt: f64) -> f64 {
    (jd_tt - JD_J2000_TT) / DAYS_PER_JULIAN_CENTURY
}

/// Julian Date (0h UT, treated as TT — see the module docs on
/// mean-value-only modeling) of the Lahiri/Chitrapaksha decree epoch,
/// 1956 March 21.
const JD_LAHIRI_ANCHOR: f64 = 2_435_553.5;

/// Julian Date of the Besselian epoch B1950.0 (the Fagan/Bradley SVP
/// reference epoch): `1950.0 - (1950.0 - 1900.0) draconic/tropic year
/// correction`, standard value per the definition of the Besselian
/// epoch.
const JD_B1950_0: f64 = 2_433_282.423_5;

/// Julian Date (0h UT) of 1900 January 1, the Krishnamurti anchor
/// epoch.
const JD_1900_01_01: f64 = 2_415_020.5;

/// Julian Date (0h UT) of 1912 January 1, the epoch this crate pins
/// B.V. Raman's year-granular "ayanamsha for 1912" table value to
/// (see the module docs' anchor citations).
const JD_1912_01_01: f64 = 2_419_402.5;

/// Lahiri anchor value: 23 deg 15' 00".
const LAHIRI_ANCHOR_DEG: f64 = 23.0 + 15.0 / 60.0;

/// Fagan/Bradley anchor value: 24 deg 02' 31.36".
const FAGAN_BRADLEY_ANCHOR_DEG: f64 = 24.0 + 2.0 / 60.0 + 31.36 / 3600.0;

/// Krishnamurti anchor value: 22 deg 21' 50" (at 1900 January 1).
const KRISHNAMURTI_ANCHOR_DEG: f64 = 22.0 + 21.0 / 60.0 + 50.0 / 3600.0;

/// Raman anchor value: 21 deg 11' 29" (Raman's published value for
/// the year 1912, *Hindu Predictive Astrology* ch. X).
const RAMAN_ANCHOR_DEG: f64 = 21.0 + 11.0 / 60.0 + 29.0 / 3600.0;

/// Sidereal-zodiac convention. See the module docs for the anchor
/// citations and the general model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ayanamsha {
    /// Cyril Fagan / Donald Bradley Western siderealist system.
    FaganBradley,
    /// N.C. Lahiri / Chitrapaksha, the Indian government standard.
    Lahiri,
    /// K.S. Krishnamurti ("KP") system.
    Krishnamurti,
    /// B.V. Raman system.
    Raman,
    /// This crate's own convention: zero at J2000.0, `p_A`-only growth.
    J2000Zero,
    /// Caller-supplied anchor.
    Custom {
        /// Anchor epoch, Julian Date TT.
        t0_jd_tt: f64,
        /// Ayanamsha value at `t0_jd_tt`, radians.
        value_at_t0_rad: f64,
    },
}

impl Ayanamsha {
    /// Every *named* ayanamsha, in a stable order.
    ///
    /// [`Ayanamsha::Custom`] is deliberately absent: it carries
    /// caller-supplied anchor data, so it is a family of ayanamshas
    /// rather than one nameable member and cannot be enumerated.
    pub const ALL_NAMED: [Self; 5] = [
        Self::FaganBradley,
        Self::Lahiri,
        Self::Krishnamurti,
        Self::Raman,
        Self::J2000Zero,
    ];

    /// A short, stable kebab-case name — the same string the CLI accepts
    /// for `--sidereal` and emits in its JSON `"sidereal"` field.
    ///
    /// Returns `None` for [`Ayanamsha::Custom`], which has no fixed name.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::FaganBradley => "fagan-bradley",
            Self::Lahiri => "lahiri",
            Self::Krishnamurti => "krishnamurti",
            Self::Raman => "raman",
            Self::J2000Zero => "j2000-zero",
            Self::Custom { .. } => return None,
        })
    }
}

/// An anchor pair, internal representation: `t0` in Julian centuries TT
/// since J2000.0, and the ayanamsha value there, radians.
struct Anchor {
    t0_centuries_tt: f64,
    value_at_t0_rad: f64,
}

fn anchor(kind: Ayanamsha) -> Anchor {
    match kind {
        Ayanamsha::FaganBradley => Anchor {
            t0_centuries_tt: centuries_since_j2000(JD_B1950_0),
            value_at_t0_rad: FAGAN_BRADLEY_ANCHOR_DEG * DEG_TO_RAD,
        },
        Ayanamsha::Lahiri => Anchor {
            t0_centuries_tt: centuries_since_j2000(JD_LAHIRI_ANCHOR),
            value_at_t0_rad: LAHIRI_ANCHOR_DEG * DEG_TO_RAD,
        },
        Ayanamsha::Krishnamurti => Anchor {
            t0_centuries_tt: centuries_since_j2000(JD_1900_01_01),
            value_at_t0_rad: KRISHNAMURTI_ANCHOR_DEG * DEG_TO_RAD,
        },
        Ayanamsha::Raman => Anchor {
            t0_centuries_tt: centuries_since_j2000(JD_1912_01_01),
            value_at_t0_rad: RAMAN_ANCHOR_DEG * DEG_TO_RAD,
        },
        Ayanamsha::J2000Zero => Anchor {
            t0_centuries_tt: 0.0,
            value_at_t0_rad: 0.0,
        },
        Ayanamsha::Custom {
            t0_jd_tt,
            value_at_t0_rad,
        } => Anchor {
            t0_centuries_tt: centuries_since_j2000(t0_jd_tt),
            value_at_t0_rad,
        },
    }
}

/// Wraps `x` (radians) into `[0, 2*pi)`.
fn wrap_0_2pi(x: f64) -> f64 {
    let r = fmod(x, TWO_PI);
    if r < 0.0 {
        r + TWO_PI
    } else {
        r
    }
}

/// The IAU 2006 (P03) general precession in ecliptic longitude, `p_A`,
/// in **arcseconds**, for `t` Julian centuries TT since J2000.0.
///
/// Capitaine, Wallace & Chapront (2003), A&A 412, 567, eq. (39):
///
/// ```text
/// p_A = 5028.796195" t + 1.1054348" t^2 + 0.00007964" t^3
///       - 0.000023857" t^4 - 0.0000000383" t^5
/// ```
///
/// (Same equation group as the `eps_A` polynomial in
/// [`oxiephemeris_bodies::frames::mean_obliquity_iau2006`] — verified
/// against that function's cited coefficients as a consistency check
/// on the source citation.)
#[must_use]
fn general_precession_longitude_arcsec(t_centuries_tt: f64) -> f64 {
    let t = t_centuries_tt;
    t * (5_028.796_195
        + t * (1.105_434_8 + t * (0.000_079_64 + t * (-0.000_023_857 + t * (-3.83e-8)))))
}

/// The IAU 2006 (P03) general precession in ecliptic longitude, `p_A`,
/// in **radians**, for `t` Julian centuries TT since J2000.0. See this
/// module's private `general_precession_longitude_arcsec` (just above)
/// for the polynomial and its citation.
#[must_use]
pub fn general_precession_iau2006_rad(t_centuries_tt: f64) -> f64 {
    general_precession_longitude_arcsec(t_centuries_tt) * ARCSEC_TO_RAD
}

/// The ayanamsha (sidereal-zodiac offset), radians in `[0, 2*pi)`, for
/// `kind` at `t_tt_centuries` Julian centuries TT since J2000.0.
///
/// See the module docs for the anchor-plus-`p_A` model and its
/// citations.
#[must_use]
pub fn ayanamsha_rad(kind: Ayanamsha, t_tt_centuries: f64) -> f64 {
    let a = anchor(kind);
    let delta_rad = general_precession_iau2006_rad(t_tt_centuries)
        - general_precession_iau2006_rad(a.t0_centuries_tt);
    wrap_0_2pi(a.value_at_t0_rad + delta_rad)
}

/// Converts a tropical ecliptic longitude (radians) to the
/// corresponding sidereal longitude under `kind`, at `t_tt_centuries`
/// Julian centuries TT since J2000.0: `sidereal = tropical - ayanamsha`
/// (mod `2*pi`).
#[must_use]
pub fn sidereal_from_tropical(tropical_lon_rad: f64, kind: Ayanamsha, t_tt_centuries: f64) -> f64 {
    wrap_0_2pi(tropical_lon_rad - ayanamsha_rad(kind, t_tt_centuries))
}
