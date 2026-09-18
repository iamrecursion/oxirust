//! `OxiEphemeris` `compat` crate: an SE-shaped API surface — `Context`,
//! body/flag constants, house systems, sidereal modes — implemented as a
//! thin, documented mapping onto `oxiephemeris-core`/`-de`/`-bodies`/
//! `-astro`.
//!
//! # Clean-room provenance
//!
//! No Swiss Ephemeris **source code** was ever opened while writing this
//! crate. The Swiss Ephemeris **programmer's documentation**
//! (Astrodienst, <https://www.astro.com/swisseph/swephprg.htm>) was
//! consulted for API shape and the numeric values of the documented
//! `SEFLG_*`/`SE_*` constants — explicitly permitted by `docs/design.md`
//! section 1's license-boundary table ("SE documentation … OK to consult
//! docs for API shape; do not transcribe algorithm descriptions
//! verbatim"). Every algorithm behind these entry points — time scales,
//! precession/nutation, the apparent-place pipeline, house systems,
//! aspects, nodes, ayanamshas — is implemented in the crates this one
//! wraps, from published papers and standards only (see their own
//! crate-level docs).
//!
//! # Design: no global state
//!
//! Swiss Ephemeris is a C API with process-global state (the loaded
//! ephemeris files, the topocentric observer, the sidereal mode are all
//! set once via `swe_set_*` calls and read back implicitly by every
//! `swe_calc`-family call afterward). [`Context`] makes that state
//! explicit and instance-local instead: every quantity a calculation
//! needs (which DE file, which ground station, which ayanamsha) lives on
//! the `Context` value the caller constructs and threads through, so
//! multiple independent contexts (e.g. one per thread, or one per
//! ephemeris file) can coexist safely with no `unsafe`, no
//! `static mut`, and no serialization requirement between calls.
//!
//! # Differences from Swiss Ephemeris (read this before porting code)
//!
//! - **No asteroids or Chiron yet.** Only [`bodies::SE_SUN`]`..=`
//!   [`bodies::SE_EARTH`] (0–14) are implemented; see [`bodies`] for the
//!   exact supported set and [`CompatError::UnsupportedBody`] for
//!   everything else. No fixed stars (`swe_fixstar*`) at all.
//! - **No SE file formats.** This crate reads JPL DE classic-binary
//!   files directly through [`oxiephemeris_de`]; there is no `sepl*.se1`
//!   / `semo*.se1` / `seas*.se1` support and no automatic ephemeris-file
//!   search path.
//! - **`Result`, not `retflag`/`serr`.** Every fallible entry point
//!   returns a typed [`CompatError`] instead of SE's `(return code,
//!   error-string-out-parameter)` convention; there is no analogue of
//!   SE's non-fatal *warning* strings (e.g. "using Moshier ephemeris
//!   outside file range") — every failure here is a hard `Err`.
//! - **Different time-scale models.** `calc_ut`/`houses`/`sidtime` take
//!   `jd_ut` and convert to TT with `oxiephemeris_core`'s
//!   Espenak–Meeus ΔT polynomial (see
//!   [`oxiephemeris_core::time::delta_t_seconds`]), not SE's own ΔT
//!   model; the two agree to a fraction of a second in the modern era
//!   but can differ by up to a few seconds for ancient/very distant
//!   epochs, and neither reproduces true measured `UT1−UTC` (this crate
//!   has no leap-second-table-based path here at all, unlike
//!   `oxiephemeris_core::time::utc_to_tt` — this is a deliberate choice
//!   so pre-1972 dates, routine in natal astrology, still work). `jd_ut`
//!   is treated as UT1 directly (`ΔUT1 = 0`, or the topocentric
//!   [`oxiephemeris_bodies::topocentric::Eop::dut1_s`] if
//!   [`Context::set_topo`] was called), matching SE's own documented
//!   `jd_ut` semantics.
//! - **Frames.** Precession is IAU 2006 (P03) and nutation is the full
//!   IAU `2000A_R06` series (see [`oxiephemeris_bodies::frames`]), not
//!   SE's own default frame model; both are modern, more accurate
//!   successors, but numerically distinct at the sub-arcsecond level
//!   from an SE build using an older default.
//! - **Sidereal-longitude convention.** `SEFLG_SIDEREAL` subtracts the
//!   ayanamsha referred to the **true equinox of date** (the mean
//!   ayanamsha plus the nutation in longitude `Δψ`), so sidereal
//!   longitudes are free of the equinox's nutation wobble — this
//!   matches SE's measured output convention, while
//!   [`Context::get_ayanamsa`] returns the mean-equinox value, also
//!   matching `swe_get_ayanamsa`. (See
//!   `context::calc::sidereal_offset_rad` for the verification note.
//!   `oxiephemeris-astro` itself exposes only the mean-equinox
//!   ayanamsha value; the `Δψ` handling lives in this crate's calc /
//!   houses entry points.)
//! - **Ayanamsha catalog.** Only [`oxiephemeris_astro::ayanamsha::Ayanamsha`]'s
//!   five variants (Fagan/Bradley, Lahiri, Krishnamurti, Raman,
//!   J2000-zero, plus a caller-supplied custom anchor) are available —
//!   see [`SiderealMode`] — not SE's full `SE_SIDM_*` catalog (dozens of
//!   modes). This crate deliberately does **not** reproduce numeric
//!   `SE_SIDM_*` mode IDs: unlike the `SEFLG_*`/`SE_*` body constants,
//!   those IDs could not be confirmed against the published *programmer
//!   documentation* page alone (independent secondary sources
//!   disagreed), and confirming them against the SE header would mean
//!   opening SE source — forbidden by this project's clean-room policy.
//!   [`SiderealMode`] is this crate's own enum instead.
//! - **No radius/latitude convention for nodes and apogee.** SE's
//!   documented output for `SE_MEAN_NODE`/`SE_TRUE_NODE`/
//!   `SE_MEAN_APOG`/`SE_OSCU_APOG` includes a `dist` (and `dist` speed)
//!   component whose exact defining convention is not derivable from
//!   the published *programmer* documentation alone. This crate's node
//!   and apogee longitudes come from published mean-element polynomials
//!   and two-body osculating-element definitions (see
//!   [`oxiephemeris_astro::nodes`]) that have no natural "distance"; see
//!   [`Context::calc`] for exactly what is returned instead (`0.0`,
//!   documented, never silently something else).
//! - **`houses`/`houses_ex` cusp indexing.** SE's C API returns
//!   `cusps[1..=12]` with `cusps[0]` reserved/unused (1-based, matching
//!   Fortran/legacy convention). [`Context::houses`] /
//!   [`Context::houses_ex`] return a plain `[f64; 12]` with `array[0]` =
//!   cusp 1 — the same 0-based convention
//!   [`oxiephemeris_astro::houses::cusps`] already uses — documented
//!   here as a deliberate Rust-ergonomic difference, not an oversight.
//! - **`ascmc` slots 5–7 are `0.0`.** The co-ascendant (Koch), co-ascendant
//!   (Munkasey), and polar-ascendant (Munkasey) points are not
//!   implemented (`oxiephemeris_astro::angles` does not expose them);
//!   see [`Houses`].

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod bodies;
pub mod context;
pub mod flags;
pub mod houses;

pub use context::Context;
pub use flags::{
    SEFLG_ASTROMETRIC, SEFLG_BARYCTR, SEFLG_CENTER_BODY, SEFLG_DPSIDEPS_1980, SEFLG_EQUATORIAL,
    SEFLG_HELCTR, SEFLG_ICRS, SEFLG_JPLEPH, SEFLG_JPLHOR, SEFLG_JPLHOR_APPROX, SEFLG_MOSEPH,
    SEFLG_NOABERR, SEFLG_NOGDEFL, SEFLG_NONUT, SEFLG_RADIANS, SEFLG_SIDEREAL, SEFLG_SPEED,
    SEFLG_SPEED3, SEFLG_SWIEPH, SEFLG_TOPOCTR, SEFLG_TRUEPOS, SEFLG_XYZ, SE_GREG_CAL, SE_JUL_CAL,
};
pub use houses::{HouseSystemChar, Houses};

pub use bodies::{
    SE_EARTH, SE_JUPITER, SE_MARS, SE_MEAN_APOG, SE_MEAN_NODE, SE_MERCURY, SE_MOON, SE_NEPTUNE,
    SE_OSCU_APOG, SE_PLUTO, SE_SATURN, SE_SUN, SE_TRUE_NODE, SE_URANUS, SE_VENUS,
};

/// Sidereal-zodiac mode for [`Context::set_sid_mode`] /
/// [`Context::get_ayanamsa`], mirroring `swe_set_sid_mode`'s built-in
/// modes plus its `SE_SIDM_USER` custom-anchor case. See the crate docs'
/// "Differences from Swiss Ephemeris" section for why this is this
/// crate's own enum rather than a reproduction of `SE_SIDM_*` numeric
/// IDs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SiderealMode {
    /// Cyril Fagan / Donald Bradley Western siderealist system. SE's own
    /// documented default when `swe_set_sid_mode` is never called.
    FaganBradley,
    /// N.C. Lahiri / Chitrapaksha, the Indian government standard.
    Lahiri,
    /// K.S. Krishnamurti ("KP") system.
    Krishnamurti,
    /// B.V. Raman system.
    Raman,
    /// This crate's own convention: zero at J2000.0, precession-only
    /// growth (see
    /// [`oxiephemeris_astro::ayanamsha::Ayanamsha::J2000Zero`]).
    J2000Zero,
    /// A caller-supplied anchor: `swe_set_sid_mode(SE_SIDM_USER, t0,
    /// ayan_t0)`'s two extra parameters, folded into the variant since
    /// they are meaningful only here (SE's own documentation states they
    /// are ignored for the built-in modes).
    User {
        /// Reference epoch, Julian Date TT.
        t0_jd_tt: f64,
        /// Ayanamsha value at `t0_jd_tt`, in **degrees**.
        ayan_t0_deg: f64,
    },
}

impl SiderealMode {
    /// Maps to the underlying [`oxiephemeris_astro::ayanamsha::Ayanamsha`].
    #[must_use]
    pub(crate) fn to_ayanamsha(self) -> oxiephemeris_astro::ayanamsha::Ayanamsha {
        use oxiephemeris_astro::ayanamsha::Ayanamsha;
        match self {
            Self::FaganBradley => Ayanamsha::FaganBradley,
            Self::Lahiri => Ayanamsha::Lahiri,
            Self::Krishnamurti => Ayanamsha::Krishnamurti,
            Self::Raman => Ayanamsha::Raman,
            Self::J2000Zero => Ayanamsha::J2000Zero,
            Self::User {
                t0_jd_tt,
                ayan_t0_deg,
            } => Ayanamsha::Custom {
                t0_jd_tt,
                value_at_t0_rad: ayan_t0_deg * oxiephemeris_core::angle::DEG2RAD,
            },
        }
    }
}

/// Crate-level error type: every fallible entry point of [`Context`]
/// returns `Result<_, CompatError>` instead of SE's `(retflag, serr)`
/// out-parameter convention (see the crate docs).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatError {
    /// A time-scale conversion in `oxiephemeris-core` failed (e.g. a
    /// non-finite or out-of-range Julian Date, or a calendar date with an
    /// invalid month/day).
    Core(oxiephemeris_core::CoreError),
    /// The underlying DE ephemeris evaluation failed (epoch out of the
    /// file's span, series unavailable, truncated file, …).
    De(oxiephemeris_de::DeError),
    /// The apparent-place pipeline failed (target coincides with the
    /// observation center, topocentric observer misuse, a pre-1972
    /// leap-second-table time conversion for the *station* rotation —
    /// see [`oxiephemeris_bodies::apparent::BodiesError`]).
    Bodies(oxiephemeris_bodies::apparent::BodiesError),
    /// House-cusp computation failed (non-finite input, polar latitude,
    /// or an undefined/degenerate configuration).
    Houses(oxiephemeris_astro::houses::HousesError),
    /// A chart-angle computation (Ascendant/MC/Vertex/East Point) failed.
    Angles(oxiephemeris_astro::angles::AnglesError),
    /// A lunar node/apogee computation failed (see
    /// [`oxiephemeris_astro::nodes::NodeError`]; this crate's
    /// `SE_TRUE_NODE`/`SE_OSCU_APOG` route through it).
    Node(oxiephemeris_astro::nodes::NodeError),
    /// No DE ephemeris is loaded in this [`Context`]
    /// ([`Context::with_ephemeris`] was never called), but the requested
    /// body needs one (every body except the pure-calendar/pure-house
    /// functions do).
    NoEphemeris,
    /// The requested SE body number is outside the documented supported
    /// set; see [`bodies`] for exactly which numbers are implemented.
    UnsupportedBody(i32),
    /// The requested house-system character is not one of the seven
    /// documented systems; see [`HouseSystemChar`].
    UnsupportedHouseSystem(char),
    /// An `iflag`/`cuspflag` bit outside the documented `SEFLG_*` mask
    /// this crate recognizes (see [`flags`]); the payload is exactly the
    /// unrecognized bits, so the caller can inspect them.
    UnsupportedFlags(u32),
    /// A recognized `SEFLG_*` bit that this crate deliberately does not
    /// implement (see the per-constant docs in [`flags`] for which ones
    /// and why); the payload names the bit(s).
    UnsupportedFlagName(&'static str),
    /// Two or more requested flags/parameters contradict each other
    /// (e.g. `SEFLG_HELCTR` together with `SEFLG_TOPOCTR`); the payload
    /// explains which.
    ConflictingFlags(&'static str),
    /// `SEFLG_TOPOCTR` was requested but [`Context::set_topo`] was never
    /// called.
    TopoNotSet,
}

impl core::fmt::Display for CompatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Core(e) => write!(f, "time-scale conversion failed: {e}"),
            Self::De(e) => write!(f, "ephemeris evaluation failed: {e}"),
            Self::Bodies(e) => write!(f, "apparent-place pipeline failed: {e}"),
            Self::Houses(e) => write!(f, "house-cusp computation failed: {e}"),
            Self::Angles(e) => write!(f, "chart-angle computation failed: {e}"),
            Self::Node(e) => write!(f, "node/apogee computation failed: {e}"),
            Self::NoEphemeris => {
                f.write_str("no DE ephemeris loaded in this Context (see Context::with_ephemeris)")
            }
            Self::UnsupportedBody(n) => {
                write!(f, "SE body number {n} is not implemented by this crate")
            }
            Self::UnsupportedHouseSystem(c) => {
                write!(
                    f,
                    "house-system character {c:?} is not implemented by this crate"
                )
            }
            Self::UnsupportedFlags(bits) => {
                write!(f, "unrecognized SEFLG_* bit(s): {bits:#x}")
            }
            Self::UnsupportedFlagName(name) => {
                write!(f, "{name} is not implemented by this crate")
            }
            Self::ConflictingFlags(msg) => write!(f, "conflicting flags: {msg}"),
            Self::TopoNotSet => {
                f.write_str("SEFLG_TOPOCTR requires Context::set_topo to be called first")
            }
        }
    }
}

impl core::error::Error for CompatError {}

impl From<oxiephemeris_core::CoreError> for CompatError {
    fn from(e: oxiephemeris_core::CoreError) -> Self {
        Self::Core(e)
    }
}

impl From<oxiephemeris_de::DeError> for CompatError {
    fn from(e: oxiephemeris_de::DeError) -> Self {
        Self::De(e)
    }
}

impl From<oxiephemeris_bodies::apparent::BodiesError> for CompatError {
    fn from(e: oxiephemeris_bodies::apparent::BodiesError) -> Self {
        Self::Bodies(e)
    }
}

impl From<oxiephemeris_astro::houses::HousesError> for CompatError {
    fn from(e: oxiephemeris_astro::houses::HousesError) -> Self {
        Self::Houses(e)
    }
}

impl From<oxiephemeris_astro::angles::AnglesError> for CompatError {
    fn from(e: oxiephemeris_astro::angles::AnglesError) -> Self {
        Self::Angles(e)
    }
}

impl From<oxiephemeris_astro::nodes::NodeError> for CompatError {
    fn from(e: oxiephemeris_astro::nodes::NodeError) -> Self {
        Self::Node(e)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompatError, SiderealMode};
    use oxiephemeris_astro::ayanamsha::Ayanamsha;

    #[test]
    fn sidereal_mode_maps_to_ayanamsha() {
        assert_eq!(
            SiderealMode::FaganBradley.to_ayanamsha(),
            Ayanamsha::FaganBradley
        );
        assert_eq!(SiderealMode::Lahiri.to_ayanamsha(), Ayanamsha::Lahiri);
        assert_eq!(
            SiderealMode::Krishnamurti.to_ayanamsha(),
            Ayanamsha::Krishnamurti
        );
        assert_eq!(SiderealMode::Raman.to_ayanamsha(), Ayanamsha::Raman);
        assert_eq!(SiderealMode::J2000Zero.to_ayanamsha(), Ayanamsha::J2000Zero);
        match (SiderealMode::User {
            t0_jd_tt: 2_451_545.0,
            ayan_t0_deg: 10.0,
        })
        .to_ayanamsha()
        {
            Ayanamsha::Custom {
                t0_jd_tt,
                value_at_t0_rad,
            } => {
                assert!((t0_jd_tt - 2_451_545.0).abs() < 1e-9);
                assert!((value_at_t0_rad - 10.0_f64.to_radians()).abs() < 1e-15);
            }
            other => panic!("expected Ayanamsha::Custom, got {other:?}"),
        }
    }

    #[test]
    fn error_display_is_informative() {
        extern crate alloc;
        use alloc::string::ToString;
        assert!(CompatError::NoEphemeris.to_string().contains("ephemeris"));
        assert!(CompatError::UnsupportedBody(42).to_string().contains("42"));
        assert!(CompatError::TopoNotSet.to_string().contains("set_topo"));
        assert!(CompatError::UnsupportedHouseSystem('Z')
            .to_string()
            .contains('Z'));
    }
}
