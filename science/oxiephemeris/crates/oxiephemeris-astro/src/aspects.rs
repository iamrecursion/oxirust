//! Aspects: angular relationships between two ecliptic longitudes,
//! configurable per-aspect orbs, and applying/separating classification.
//!
//! # Angle wrapping
//!
//! The two longitudes are compared through the **minimal angular
//! separation** `sep = |wrap(lon1 - lon2)| in [0, pi]`, where `wrap` maps
//! any real difference to `(-pi, pi]` via `atan2(sin(x), cos(x))` (exact,
//! robust for any input magnitude — no pre-reduction of `lon1`/`lon2`
//! is required). All aspects are matched against this single `sep`
//! value: e.g. a separation of 359 degrees and one of 1 degree both
//! give `sep = 1 degree` and are both conjunctions.
//!
//! # Offset sign convention
//!
//! Let `raw = wrap(lon1 - lon2)` (signed, before taking the absolute
//! value) and `s = sign(raw)` (`+1` if `raw >= 0`, else `-1`). For an
//! aspect with exact angle `e` (always in `[0, pi]`), the reported
//! [`AspectHit::offset_rad`] is
//!
//! ```text
//! offset = s * (sep - e)
//! ```
//!
//! which satisfies `|offset| = |sep - e|` (so "offset always `<= orb`"
//! holds by construction — see [`find_aspect`]) while additionally
//! carrying the *orientation* of the raw signed difference. This is the
//! convention that makes `offset` flip sign, and `applying` stay
//! invariant, when the two bodies are swapped (`(lon1, speed1) <->
//! (lon2, speed2)`): `raw` (and hence `s`) negates under swap while `sep`
//! and `e` do not, so `offset` negates too. This crate's proptest suite
//! (`astro/tests/aspects.rs`) checks exactly this property.
//!
//! # Applying / separating
//!
//! "Applying" means the **orb is shrinking**: the aspect is approaching
//! exactness. The orb magnitude is `|offset|` (identically
//! `|sep - e|`), so applying is `d|offset|/dt < 0`.
//!
//! A short derivation: write `offset = s * sep - s * e`. Away from the
//! measure-zero instants where `raw` crosses `0` or `+-pi` (where `s`
//! itself is discontinuous), `s` is locally constant, so
//!
//! ```text
//! d(offset)/dt = s * d(sep)/dt = s * s * d(raw)/dt = d(raw)/dt
//!              = speed1 - speed2
//! ```
//!
//! (the two factors of `s` cancel because `s^2 = 1`). So the *signed*
//! offset's rate of change is simply the relative angular speed
//! `speed1 - speed2`, independent of which branch (`s = +-1`) we are on.
//! Then
//!
//! ```text
//! d|offset|/dt = sign(offset) * d(offset)/dt
//!              = sign(offset) * (speed1 - speed2)
//! ```
//!
//! and `find_aspect` classifies **applying** as
//!
//! ```text
//! offset * (speed1 - speed2) < 0.0
//! ```
//!
//! (product of two same-signed quantities that would make
//! `d|offset|/dt` negative). At exact exactness (`offset == 0.0`) the
//! instantaneous rate of `|offset|` has a corner (it was decreasing just
//! before and increases just after, for nonzero relative speed); we
//! define this boundary instant as **not applying** (`applying = false`)
//! so the classification stays a total, deterministic function — this
//! is the one place the rule is a convention rather than a derivative.
//!
//! **Retrograde motion** requires no special-casing at all: the rule
//! above is linear in the single quantity `speed1 - speed2`, so it is
//! correct verbatim whether `speed1`, `speed2` are direct (positive),
//! retrograde (negative), or both retrograde — only the *relative*
//! speed matters. `astro/tests/aspects.rs` includes a hand-computed
//! truth table covering direct/direct, direct/retrograde, and
//! retrograde/retrograde combinations.
//!
//! # Provenance
//!
//! The classical aspect angles (conjunction/opposition/trine/square/
//! sextile) and the common "minor" set (semisextile, semisquare,
//! sesquiquadrate, quincunx, quintile, biquintile) are standard,
//! centuries-old named divisions of the circle (`360 / n` for small
//! integer `n`, or `360 / n` combined with a half-step) — plain
//! spherical/angular geometry, not sourced from any ephemeris
//! implementation. The default orb widths in [`OrbPolicy::default`]
//! are **this crate's own convention** (documented on that impl), not a
//! reproduction of any particular published or software table; callers
//! wanting different behavior should build their own `OrbPolicy`.

use libm::{atan2, cos, fabs, sin};

/// Degrees to radians: `pi / 180`.
const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// The classical Ptolemaic aspects plus the common "minor" aspect set.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AspectKind {
    /// 0 degrees.
    Conjunction,
    /// 180 degrees.
    Opposition,
    /// 120 degrees.
    Trine,
    /// 90 degrees.
    Square,
    /// 60 degrees.
    Sextile,
    /// 30 degrees.
    Semisextile,
    /// 45 degrees.
    Semisquare,
    /// 135 degrees.
    Sesquiquadrate,
    /// 150 degrees.
    Quincunx,
    /// 72 degrees (a fifth of the circle).
    Quintile,
    /// 144 degrees (two fifths of the circle).
    Biquintile,
}

impl AspectKind {
    /// A short, stable name (lower-case, no spaces).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Conjunction => "conjunction",
            Self::Opposition => "opposition",
            Self::Trine => "trine",
            Self::Square => "square",
            Self::Sextile => "sextile",
            Self::Semisextile => "semisextile",
            Self::Semisquare => "semisquare",
            Self::Sesquiquadrate => "sesquiquadrate",
            Self::Quincunx => "quincunx",
            Self::Quintile => "quintile",
            Self::Biquintile => "biquintile",
        }
    }
}

/// An aspect: its kind and exact angular separation, in radians,
/// always in `[0, pi]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aspect {
    /// Which classical aspect this is.
    pub kind: AspectKind,
    /// Exact angular separation, radians, in `[0, pi]`.
    pub exact_angle_rad: f64,
}

impl Aspect {
    /// Conjunction, 0 degrees.
    pub const CONJUNCTION: Self = Self {
        kind: AspectKind::Conjunction,
        exact_angle_rad: 0.0,
    };
    /// Semisextile, 30 degrees (`pi / 6`).
    pub const SEMISEXTILE: Self = Self {
        kind: AspectKind::Semisextile,
        exact_angle_rad: core::f64::consts::FRAC_PI_6,
    };
    /// Semisquare, 45 degrees (`pi / 4`).
    pub const SEMISQUARE: Self = Self {
        kind: AspectKind::Semisquare,
        exact_angle_rad: core::f64::consts::FRAC_PI_4,
    };
    /// Sextile, 60 degrees (`pi / 3`).
    pub const SEXTILE: Self = Self {
        kind: AspectKind::Sextile,
        exact_angle_rad: core::f64::consts::FRAC_PI_3,
    };
    /// Quintile, 72 degrees (`2 pi / 5`).
    pub const QUINTILE: Self = Self {
        kind: AspectKind::Quintile,
        exact_angle_rad: 2.0 * core::f64::consts::PI / 5.0,
    };
    /// Square, 90 degrees (`pi / 2`).
    pub const SQUARE: Self = Self {
        kind: AspectKind::Square,
        exact_angle_rad: core::f64::consts::FRAC_PI_2,
    };
    /// Trine, 120 degrees (`2 pi / 3`).
    pub const TRINE: Self = Self {
        kind: AspectKind::Trine,
        exact_angle_rad: 2.0 * core::f64::consts::PI / 3.0,
    };
    /// Sesquiquadrate, 135 degrees (`3 pi / 4`).
    pub const SESQUIQUADRATE: Self = Self {
        kind: AspectKind::Sesquiquadrate,
        exact_angle_rad: 3.0 * core::f64::consts::FRAC_PI_4,
    };
    /// Biquintile, 144 degrees (`4 pi / 5`).
    pub const BIQUINTILE: Self = Self {
        kind: AspectKind::Biquintile,
        exact_angle_rad: 4.0 * core::f64::consts::PI / 5.0,
    };
    /// Quincunx (inconjunct), 150 degrees (`5 pi / 6`).
    pub const QUINCUNX: Self = Self {
        kind: AspectKind::Quincunx,
        exact_angle_rad: 5.0 * core::f64::consts::PI / 6.0,
    };
    /// Opposition, 180 degrees (`pi`).
    pub const OPPOSITION: Self = Self {
        kind: AspectKind::Opposition,
        exact_angle_rad: core::f64::consts::PI,
    };

    /// All eleven aspects, in increasing order of exact angle.
    pub const ALL: &'static [Self] = &[
        Self::CONJUNCTION,
        Self::SEMISEXTILE,
        Self::SEMISQUARE,
        Self::SEXTILE,
        Self::QUINTILE,
        Self::SQUARE,
        Self::TRINE,
        Self::SESQUIQUADRATE,
        Self::BIQUINTILE,
        Self::QUINCUNX,
        Self::OPPOSITION,
    ];

    /// Looks up the [`Aspect`] value for a given [`AspectKind`].
    #[must_use]
    pub const fn for_kind(kind: AspectKind) -> Self {
        match kind {
            AspectKind::Conjunction => Self::CONJUNCTION,
            AspectKind::Semisextile => Self::SEMISEXTILE,
            AspectKind::Semisquare => Self::SEMISQUARE,
            AspectKind::Sextile => Self::SEXTILE,
            AspectKind::Quintile => Self::QUINTILE,
            AspectKind::Square => Self::SQUARE,
            AspectKind::Trine => Self::TRINE,
            AspectKind::Sesquiquadrate => Self::SESQUIQUADRATE,
            AspectKind::Biquintile => Self::BIQUINTILE,
            AspectKind::Quincunx => Self::QUINCUNX,
            AspectKind::Opposition => Self::OPPOSITION,
        }
    }
}

/// Per-aspect orb (maximum allowed `|offset|`), in radians.
///
/// # Default (this crate's own convention)
///
/// There is no single universally-published orb table — every
/// astrological tradition and every piece of software picks its own.
/// [`OrbPolicy::default`] documents *our* choice, loosely following
/// widely-used classical practice (tighter orbs for the "minor"
/// aspects, wider for the Ptolemaic majors):
///
/// | aspect          | orb (degrees) |
/// |-----------------|---------------|
/// | conjunction      | 8             |
/// | opposition       | 8             |
/// | trine            | 8             |
/// | square           | 7             |
/// | sextile          | 6             |
/// | semisextile      | 2             |
/// | semisquare       | 2             |
/// | sesquiquadrate   | 2             |
/// | quincunx         | 3             |
/// | quintile         | 2             |
/// | biquintile       | 2             |
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbPolicy {
    /// Conjunction orb, radians.
    pub conjunction_rad: f64,
    /// Opposition orb, radians.
    pub opposition_rad: f64,
    /// Trine orb, radians.
    pub trine_rad: f64,
    /// Square orb, radians.
    pub square_rad: f64,
    /// Sextile orb, radians.
    pub sextile_rad: f64,
    /// Semisextile orb, radians.
    pub semisextile_rad: f64,
    /// Semisquare orb, radians.
    pub semisquare_rad: f64,
    /// Sesquiquadrate orb, radians.
    pub sesquiquadrate_rad: f64,
    /// Quincunx orb, radians.
    pub quincunx_rad: f64,
    /// Quintile orb, radians.
    pub quintile_rad: f64,
    /// Biquintile orb, radians.
    pub biquintile_rad: f64,
}

impl OrbPolicy {
    /// The configured orb, radians, for a given [`AspectKind`].
    #[must_use]
    pub const fn orb_rad(&self, kind: AspectKind) -> f64 {
        match kind {
            AspectKind::Conjunction => self.conjunction_rad,
            AspectKind::Opposition => self.opposition_rad,
            AspectKind::Trine => self.trine_rad,
            AspectKind::Square => self.square_rad,
            AspectKind::Sextile => self.sextile_rad,
            AspectKind::Semisextile => self.semisextile_rad,
            AspectKind::Semisquare => self.semisquare_rad,
            AspectKind::Sesquiquadrate => self.sesquiquadrate_rad,
            AspectKind::Quincunx => self.quincunx_rad,
            AspectKind::Quintile => self.quintile_rad,
            AspectKind::Biquintile => self.biquintile_rad,
        }
    }
}

impl Default for OrbPolicy {
    /// See the "Default" section of the [`OrbPolicy`] docs for the table
    /// of values and the honesty disclaimer that this is our own
    /// convention.
    fn default() -> Self {
        Self {
            conjunction_rad: 8.0 * DEG_TO_RAD,
            opposition_rad: 8.0 * DEG_TO_RAD,
            trine_rad: 8.0 * DEG_TO_RAD,
            square_rad: 7.0 * DEG_TO_RAD,
            sextile_rad: 6.0 * DEG_TO_RAD,
            semisextile_rad: 2.0 * DEG_TO_RAD,
            semisquare_rad: 2.0 * DEG_TO_RAD,
            sesquiquadrate_rad: 2.0 * DEG_TO_RAD,
            quincunx_rad: 3.0 * DEG_TO_RAD,
            quintile_rad: 2.0 * DEG_TO_RAD,
            biquintile_rad: 2.0 * DEG_TO_RAD,
        }
    }
}

/// The result of a successful [`find_aspect`] match.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AspectHit {
    /// The matched aspect (kind and exact angle).
    pub aspect: Aspect,
    /// Signed offset from exactness, radians. See the module docs
    /// ("Offset sign convention") for the exact definition; always
    /// `|offset_rad| <= policy.orb_rad(aspect.kind)`.
    pub offset_rad: f64,
    /// `true` if the orb (`|offset_rad|`) is shrinking, i.e. the two
    /// bodies are moving toward exactness; see the module docs
    /// ("Applying / separating").
    pub applying: bool,
}

/// Finds the best-matching aspect (smallest `|offset|` among all
/// aspects within their configured orb) between two ecliptic
/// longitudes, or `None` if no aspect in `policy` matches.
///
/// `lon1_rad`, `lon2_rad` are ecliptic longitudes in radians (any
/// range — internally reduced exactly via `atan2(sin, cos)`, see the
/// module docs). `speed1_rad_per_day`, `speed2_rad_per_day` are the
/// corresponding rates of change (e.g. daily motion in longitude);
/// negative values represent retrograde motion and require no special
/// handling (see "Applying / separating" in the module docs).
///
/// Pure `f64` math (`libm` only), `no_std`-safe, no allocation.
#[must_use]
pub fn find_aspect(
    lon1_rad: f64,
    speed1_rad_per_day: f64,
    lon2_rad: f64,
    speed2_rad_per_day: f64,
    policy: &OrbPolicy,
) -> Option<AspectHit> {
    let diff = lon1_rad - lon2_rad;
    // Exact wrap to (-pi, pi]: atan2(sin(x), cos(x)) is well-defined for
    // any finite x, with no explicit modulo arithmetic needed.
    let raw = atan2(sin(diff), cos(diff));
    let sep = fabs(raw);
    let sign_raw = if raw < 0.0 { -1.0 } else { 1.0 };

    let mut best: Option<AspectHit> = None;
    for &aspect in Aspect::ALL {
        let offset = sign_raw * (sep - aspect.exact_angle_rad);
        let orb = policy.orb_rad(aspect.kind);
        if fabs(offset) > orb {
            continue;
        }
        let is_better = match &best {
            None => true,
            Some(current) => fabs(offset) < fabs(current.offset_rad),
        };
        if is_better {
            let rel_speed = speed1_rad_per_day - speed2_rad_per_day;
            let applying = offset * rel_speed < 0.0;
            best = Some(AspectHit {
                aspect,
                offset_rad: offset,
                applying,
            });
        }
    }
    best
}
