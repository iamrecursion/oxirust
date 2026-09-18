//! House-cusp computation: Placidus, Koch, Whole Sign, Equal, Porphyry,
//! Regiomontanus, Campanus.
//!
//! # Inputs and conventions
//!
//! Same conventions as [`crate::angles`]: `theta` is the local apparent
//! sidereal time (RAMC) in radians, `phi` the geodetic latitude, `eps`
//! the true obliquity of date (`0 ≤ eps < π/2`); all returned cusps are
//! ecliptic longitudes of date in radians, normalized to `[0, 2π)`.
//!
//! **Indexing**: [`cusps`] returns `[f64; 12]` with `array[0]` = cusp 1,
//! `array[i]` = cusp `i + 1`, …, `array[11]` = cusp 12.
//!
//! # Invariants (enforced by construction)
//!
//! For every quadrant system (Placidus, Koch, Porphyry, Regiomontanus,
//! Campanus): `array[0]` is bit-identical to [`crate::angles::ascendant`]
//! and `array[9]` to [`crate::angles::mc`]; opposite cusps satisfy
//! `cusp_{i+6} = cusp_i + π (mod 2π)` because they are *assembled* that
//! way — only cusps 11, 12, 2, 3 are solved for. Equal and Whole Sign
//! place twelve exact 30° steps from their cusp 1.
//!
//! # Domain and high latitudes
//!
//! - All systems reject non-finite inputs and `|φ| ≥ π/2` (at the exact
//!   poles the horizon-based construction degenerates).
//! - **Placidus and Koch** additionally require
//!   `|tan φ · tan ε| < 1`, i.e. `|φ|` strictly below the polar circle
//!   `π/2 − ε`: inside the polar caps, ecliptic declination circles with
//!   `|tan φ · tan δ| ≥ 1` never rise or set, so the defining
//!   semi-diurnal arcs do not exist for part of every diurnal cycle.
//!   [`HousesError::Undefined`] is returned — never a silent fallback.
//! - Regiomontanus, Campanus, Porphyry, Equal and Whole Sign remain
//!   defined at every `|φ| < π/2`, except on the measure-zero
//!   configurations where the ecliptic coincides with the horizon
//!   (`|φ| = π/2 − ε`, `θ = ±π/2`) or with a house circle, which also
//!   report [`HousesError::Undefined`].
//!
//! # Clean-room sources
//!
//! - Wikipedia, "House (astrology)" (accessed 2026-07-06): geometric
//!   definitions — Placidus trisects each degree's time from rising to
//!   culmination; Regiomontanus divides the **celestial equator** into
//!   twelve, projected on the ecliptic along great circles through the
//!   north and south points of the horizon; Campanus divides the **prime
//!   vertical** the same way; Porphyry trisects the ecliptic arcs of the
//!   ASC–MC quadrants; Equal and Whole Sign are 30° steps.
//! - morinus-astrology.com, "How to calculate house cusps in the
//!   Placidus system?" (accessed 2026-07-06): the classical fixed-point
//!   iteration `RA_{n+1} = RAMC + arccos(−sin RA_n · tan φ · tan ε)/F`
//!   and the polar-circle caveat.
//! - Urania Trust, "The Astronomy of Houses" and Astro\*Dictionary,
//!   "Koch House System" (both accessed 2026-07-06): Koch cusps are the
//!   *ascendants* at the sidereal times obtained by adding 1, 2, 4 and 5
//!   thirds of the MC degree's semi-diurnal arc to the sidereal time of
//!   the MC degree's rising (equivalently: birth time ∓ k/3 of
//!   `SDA(δ_MC)`).
//! - R. W. Holden, *The Elements of House Division* (Fowler, 1977):
//!   standard mathematical treatment (background reference; the closed
//!   forms below are re-derived from the plane-intersection geometry of
//!   [`crate::angles`]).
//!
//! No Swiss Ephemeris or SOFA/ERFA source code was consulted.

use crate::angles::{ascendant, mc};
use core::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_6, PI};
use libm::{acos, atan2, cos, fabs, floor, sin, sqrt, tan};
use oxiephemeris_core::angle::normalize_0_two_pi;

/// The supported house systems.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HouseSystem {
    /// Placidus: cusps 11, 12, 2, 3 are the ecliptic points whose hour
    /// angle is the matching fraction of their **own** semi-diurnal
    /// (above horizon) or semi-nocturnal (below) arc; solved iteratively.
    Placidus,
    /// Koch ("birthplace houses"): cusps 11, 12, 2, 3 are the ascendants
    /// at sidereal times `θ ∓ k/3 · SDA(δ_MC)`, `k = 1, 2`.
    Koch,
    /// Whole Sign: cusp 1 is 0° of the zodiac sign containing the
    /// ascendant; 30° steps.
    WholeSign,
    /// Equal: cusp 1 is the ascendant itself; 30° steps.
    Equal,
    /// Porphyry: the ecliptic arcs of the four ASC–MC quadrants are
    /// trisected.
    Porphyry,
    /// Regiomontanus: house circles through the north/south points of
    /// the horizon divide the **celestial equator** into twelve 30° arcs
    /// from the RAMC.
    Regiomontanus,
    /// Campanus: house circles through the north/south points of the
    /// horizon divide the **prime vertical** into twelve 30° arcs from
    /// the zenith.
    Campanus,
}

impl HouseSystem {
    /// All seven supported house systems, in a stable order.
    pub const ALL: [Self; 7] = [
        Self::Placidus,
        Self::Koch,
        Self::WholeSign,
        Self::Equal,
        Self::Porphyry,
        Self::Regiomontanus,
        Self::Campanus,
    ];

    /// A short, stable kebab-case name — the same string the CLI accepts
    /// for `--system` and emits in its JSON `"system"` field.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Placidus => "placidus",
            Self::Koch => "koch",
            Self::WholeSign => "whole-sign",
            Self::Equal => "equal",
            Self::Porphyry => "porphyry",
            Self::Regiomontanus => "regiomontanus",
            Self::Campanus => "campanus",
        }
    }
}

/// Errors of the house-cusp computation. No function here panics.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HousesError {
    /// An input (`theta`, `phi` or `eps`) was NaN or infinite.
    NonFiniteInput,
    /// `|φ| ≥ π/2`: at (or beyond) the exact poles the horizon-based
    /// house constructions degenerate for every system.
    PolarLatitude,
    /// The requested system is undefined for these inputs: Placidus/Koch
    /// with `|φ|` at or above the polar circle (`|tan φ · tan ε| ≥ 1`,
    /// where some ecliptic declination circles never rise or set), or —
    /// for any quadrant system — an exactly degenerate configuration
    /// where the ecliptic coincides with the horizon or a house circle.
    Undefined,
}

impl core::fmt::Display for HousesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("non-finite input angle"),
            Self::PolarLatitude => f.write_str("latitude at or beyond the poles (|phi| >= pi/2)"),
            Self::Undefined => f.write_str(
                "house system undefined for these inputs (polar circle or \
                 degenerate configuration)",
            ),
        }
    }
}

impl core::error::Error for HousesError {}

/// Exact multiples of 30° used for the Equal / Whole Sign step ladder.
const HOUSE_STEP: [f64; 12] = [
    0.0,
    FRAC_PI_6,
    FRAC_PI_3,
    FRAC_PI_2,
    2.0 * FRAC_PI_3,
    5.0 * FRAC_PI_6,
    PI,
    7.0 * FRAC_PI_6,
    4.0 * FRAC_PI_3,
    3.0 * FRAC_PI_2,
    5.0 * FRAC_PI_3,
    11.0 * FRAC_PI_6,
];

/// Fixed-point stopping tolerance on the Placidus right ascension,
/// radians. The true error at acceptance is below `2 × 10⁻¹³` rad (see
/// [`placidus_alpha`]), inside the documented `10⁻¹²` budget.
const PLACIDUS_TOL: f64 = 1e-13;

/// Iteration cap for the Placidus fixed point. The contraction rate is
/// `< 2/3` everywhere below the polar circle (proof at
/// [`placidus_alpha`]), so 96 iterations bound the error by
/// `2π · (2/3)⁹⁶ ≈ 8 × 10⁻¹⁷` — the tolerance is always reached first.
const PLACIDUS_MAX_ITER: usize = 96;

/// Bisection steps of the defensive Placidus fallback: the bracket is at
/// most `2π/3` long, and `2π/3 / 2⁶⁴ ≈ 10⁻¹⁹` rad.
const PLACIDUS_BISECT_STEPS: usize = 64;

/// Degeneracy threshold for the closed-form house-circle intersections
/// (same meaning as in [`crate::angles`]: the sine of the angle between
/// the ecliptic and the house circle).
const COINCIDENCE_TOL: f64 = 1e-12;

/// House cusps 1–12 for `system` at local apparent sidereal time
/// `theta`, geodetic latitude `phi`, true obliquity `eps` (radians).
///
/// `array[0]` is cusp 1 (the ascendant for every quadrant system),
/// `array[9]` cusp 10 (the MC for every quadrant system); see the
/// module docs for the full indexing, invariants and domain.
///
/// # Errors
///
/// - [`HousesError::NonFiniteInput`] — an input was NaN or infinite.
/// - [`HousesError::PolarLatitude`] — `|φ| ≥ π/2`.
/// - [`HousesError::Undefined`] — Placidus/Koch at or above the polar
///   circle (`|tan φ · tan ε| ≥ 1`), or an exactly degenerate
///   ecliptic/house-circle coincidence for the other quadrant systems.
///
/// # Examples
///
/// ```
/// use oxiephemeris_astro::houses::{cusps, HouseSystem};
/// use oxiephemeris_astro::angles::{ascendant, mc};
///
/// let theta = 1.234_f64;
/// let phi = 51.5_f64.to_radians();
/// let eps = 23.4367_f64.to_radians();
/// let c = cusps(HouseSystem::Placidus, theta, phi, eps).unwrap_or([0.0; 12]);
/// // Cusp 1 is the ascendant and cusp 10 the MC, by construction.
/// assert_eq!(c[0].to_bits(), ascendant(theta, phi, eps).unwrap_or(f64::NAN).to_bits());
/// assert_eq!(c[9].to_bits(), mc(theta, eps).to_bits());
/// ```
pub fn cusps(
    system: HouseSystem,
    theta: f64,
    phi: f64,
    eps: f64,
) -> Result<[f64; 12], HousesError> {
    if !(theta.is_finite() && phi.is_finite() && eps.is_finite()) {
        return Err(HousesError::NonFiniteInput);
    }
    if fabs(phi) >= FRAC_PI_2 {
        return Err(HousesError::PolarLatitude);
    }
    let asc = ascendant(theta, phi, eps).map_err(|_| HousesError::Undefined)?;
    let mc_val = mc(theta, eps);
    match system {
        HouseSystem::WholeSign => Ok(steps_from(FRAC_PI_6 * floor(asc / FRAC_PI_6))),
        HouseSystem::Equal => Ok(steps_from(asc)),
        HouseSystem::Porphyry => Ok(porphyry(asc, mc_val)),
        HouseSystem::Regiomontanus => regiomontanus(theta, phi, eps, asc, mc_val),
        HouseSystem::Campanus => campanus(theta, phi, eps, asc, mc_val),
        HouseSystem::Placidus => placidus(theta, phi, eps, asc, mc_val),
        HouseSystem::Koch => koch(theta, phi, eps, asc, mc_val),
    }
}

/// Twelve exact 30° steps starting at `start` (Equal / Whole Sign).
fn steps_from(start: f64) -> [f64; 12] {
    let mut out = [0.0_f64; 12];
    for (slot, step) in out.iter_mut().zip(HOUSE_STEP) {
        *slot = normalize_0_two_pi(start + step);
    }
    out
}

/// Assembles the full wheel from the four computed intermediate cusps:
/// cusps 4–9 are the antipodes of 10, 11, 12, 1, 2, 3, and cusps 1/10
/// are the ascendant/MC verbatim — this *is* the enforcement of the
/// module-level invariants.
fn assemble(asc: f64, mc_val: f64, c11: f64, c12: f64, c2: f64, c3: f64) -> [f64; 12] {
    [
        asc,
        c2,
        c3,
        normalize_0_two_pi(mc_val + PI),
        normalize_0_two_pi(c11 + PI),
        normalize_0_two_pi(c12 + PI),
        normalize_0_two_pi(asc + PI),
        normalize_0_two_pi(c2 + PI),
        normalize_0_two_pi(c3 + PI),
        mc_val,
        c11,
        c12,
    ]
}

/// Porphyry: the zodiacal arc from the MC forward to the ascendant is
/// trisected for cusps 11 and 12, and the arc from the ascendant forward
/// to the IC for cusps 2 and 3 (Wikipedia, "House (astrology)").
fn porphyry(asc: f64, mc_val: f64) -> [f64; 12] {
    let quadrant_mc_asc = normalize_0_two_pi(asc - mc_val);
    let c11 = normalize_0_two_pi(mc_val + quadrant_mc_asc / 3.0);
    let c12 = normalize_0_two_pi(mc_val + 2.0 * quadrant_mc_asc / 3.0);
    let ic = normalize_0_two_pi(mc_val + PI);
    let quadrant_asc_ic = normalize_0_two_pi(ic - asc);
    let c2 = normalize_0_two_pi(asc + quadrant_asc_ic / 3.0);
    let c3 = normalize_0_two_pi(asc + 2.0 * quadrant_asc_ic / 3.0);
    assemble(asc, mc_val, c11, c12, c2, c3)
}

/// Regiomontanus. House circle `k` (measured 30°·k eastward along the
/// equator from the meridian) passes through the north/south points of
/// the horizon `±N_h` and the equator point
/// `Q_k = (cos ρ_k, sin ρ_k, 0)`, `ρ_k = θ + k·π/6`. Its plane normal is
///
/// ```text
/// N_h × Q_k = (−cos φ sin ρ_k, cos φ cos ρ_k, −sin φ sin(k·π/6)),
/// ```
///
/// and `p(λ) · (N_h × Q_k) = 0` solves to the closed form
///
/// ```text
/// λ_k = atan2(cos φ sin ρ_k, cos φ cos ρ_k cos ε − sin φ sin(k·π/6) sin ε).
/// ```
///
/// The `atan2` branch is the correct one: at `φ = 0` it reduces to the
/// ecliptic point of right ascension `ρ_k` (continuous, MC-to-IC
/// eastward sweep), at `k = 0` to the MC and at `k = 3` to the
/// ascendant, and it is continuous in `φ` up to the degeneracies handled
/// in [`branch_longitude`]. Cusps 11, 12, 2, 3 use `k = 1, 2, 4, 5`.
fn regiomontanus(
    theta: f64,
    phi: f64,
    eps: f64,
    asc: f64,
    mc_val: f64,
) -> Result<[f64; 12], HousesError> {
    let cusp = |k_angle: f64| -> Result<f64, HousesError> {
        let rho = theta + k_angle;
        let y = cos(phi) * sin(rho);
        let x = cos(phi) * cos(rho) * cos(eps) - sin(phi) * sin(k_angle) * sin(eps);
        branch_longitude(y, x)
    };
    let c11 = cusp(FRAC_PI_6)?;
    let c12 = cusp(FRAC_PI_3)?;
    let c2 = cusp(2.0 * FRAC_PI_3)?;
    let c3 = cusp(5.0 * FRAC_PI_6)?;
    Ok(assemble(asc, mc_val, c11, c12, c2, c3))
}

/// Campanus. House circle `k` passes through the north/south points of
/// the horizon and the prime-vertical point `a = k·π/6` from the zenith
/// toward the east point, `P_a = cos a · Z + sin a · E`. Its plane
/// normal is
///
/// ```text
/// N_h × P_a = (−cos a sin θ − sin a cos φ cos θ,
///               cos a cos θ − sin a cos φ sin θ,  −sin φ sin a),
/// ```
///
/// and `p(λ) · (N_h × P_a) = 0` solves to
///
/// ```text
/// λ_k = atan2(cos a sin θ + sin a cos φ cos θ,
///             cos ε (cos a cos θ − sin a cos φ sin θ) − sin ε sin φ sin a).
/// ```
///
/// At `a = 0` this is the MC, at `a = π/2` the ascendant, at `a = π` the
/// IC, and at `φ = 0` the ecliptic point of right ascension `θ + a` —
/// fixing the `atan2` branch by continuity. Cusps 11, 12, 2, 3 use
/// `k = 1, 2, 4, 5`.
fn campanus(
    theta: f64,
    phi: f64,
    eps: f64,
    asc: f64,
    mc_val: f64,
) -> Result<[f64; 12], HousesError> {
    let cusp = |a: f64| -> Result<f64, HousesError> {
        let (sin_a, cos_a) = (sin(a), cos(a));
        let y = cos_a * sin(theta) + sin_a * cos(phi) * cos(theta);
        let x = cos(eps) * (cos_a * cos(theta) - sin_a * cos(phi) * sin(theta))
            - sin(eps) * sin(phi) * sin_a;
        branch_longitude(y, x)
    };
    let c11 = cusp(FRAC_PI_6)?;
    let c12 = cusp(FRAC_PI_3)?;
    let c2 = cusp(2.0 * FRAC_PI_3)?;
    let c3 = cusp(5.0 * FRAC_PI_6)?;
    Ok(assemble(asc, mc_val, c11, c12, c2, c3))
}

/// Placidus. Above the horizon (cusps 11, 12) the defining condition is
/// `H(λ) = −f · SDA(δ(λ))` with `f = 1/3, 2/3`: the point trisects, in
/// its own diurnal time, the path from rising (`H = −SDA`) to
/// culmination (`H = 0`). Below the horizon (cusps 2, 3) it is
/// `H(λ) = −(SDA + f · SNA)` with `f = 1/3, 2/3` and the semi-nocturnal
/// arc `SNA = π − SDA`: trisection of the path from rising back to the
/// anti-culmination. With `α = θ − H` and `SNA = π − SDA` both cases
/// collapse to
///
/// ```text
/// α = θ + c + g · SDA(δ(α)),
///     cusp 11: (c, g) = (0,    1/3)     cusp 12: (0,    2/3)
///     cusp  2: (c, g) = (π/3,  2/3)     cusp  3: (2π/3, 1/3)
/// ```
///
/// which is the classical Placidus iteration
/// `RA_{n+1} = RAMC + arccos(−sin RA_n tan φ tan ε)/F`
/// (morinus-astrology.com; `F = 3, 3/2` for cusps 11, 12, and the
/// equivalent nocturnal forms for 2, 3), using
/// `tan δ(α) = tan ε · sin α` for a point on the ecliptic.
fn placidus(
    theta: f64,
    phi: f64,
    eps: f64,
    asc: f64,
    mc_val: f64,
) -> Result<[f64; 12], HousesError> {
    let tan_product = tan(phi) * tan(eps);
    if fabs(tan_product) >= 1.0 {
        return Err(HousesError::Undefined);
    }
    let one_third = 1.0 / 3.0;
    let two_thirds = 2.0 / 3.0;
    let c11 = ecliptic_of_ra(placidus_alpha(theta, 0.0, one_third, tan_product), eps);
    let c12 = ecliptic_of_ra(placidus_alpha(theta, 0.0, two_thirds, tan_product), eps);
    let c2 = ecliptic_of_ra(
        placidus_alpha(theta, FRAC_PI_3, two_thirds, tan_product),
        eps,
    );
    let c3 = ecliptic_of_ra(
        placidus_alpha(theta, 2.0 * FRAC_PI_3, one_third, tan_product),
        eps,
    );
    Ok(assemble(asc, mc_val, c11, c12, c2, c3))
}

/// Solves the Placidus fixed point `α = T(α) = θ + c + g·acos(−t sin α)`
/// for `t = tan φ · tan ε`, `|t| < 1`.
///
/// # Convergence proof and iteration bound
///
/// `T` maps all of ℝ into the interval `I = [θ+c, θ+c+gπ]` (since
/// `acos ∈ [0, π]`), and on ℝ
///
/// ```text
/// |T′(α)| = g · |t cos α| / √(1 − t² sin²α)
///          = g · |t| √(1 − sin²α) / √(1 − t² sin²α) ≤ g·|t| < 2/3,
/// ```
///
/// because `(1 − s²)/(1 − t²s²)` is decreasing in `s² ∈ [0, 1]` for
/// `|t| < 1` (its `s²`-derivative has sign `t² − 1 < 0`), so the maximum
/// is at `s = 0`. `T` is therefore a global contraction with rate
/// `k = g·|t| < 2/3` for **every** latitude below the polar circle; by
/// the Banach fixed-point theorem the solution exists, is unique, and
/// the iteration error after `n` steps is at most
/// `kⁿ/(1−k) · |α₁ − α₀| ≤ 3 · (2/3)ⁿ · gπ`. The stopping rule
/// `|α_{n+1} − α_n| < 10⁻¹³` bounds the true error by
/// `k/(1−k) · 10⁻¹³ < 2 × 10⁻¹³`, and [`PLACIDUS_MAX_ITER`] = 96 pushes
/// the a-priori bound to `≈ 8 × 10⁻¹⁷` — so the loop always terminates
/// via the tolerance. A bracketed bisection on
/// `F(α) = α − θ − c − g·acos(−t sin α)` (sign change across `I` since
/// `F(θ+c) < 0 < F(θ+c+gπ)`) is kept as a defensive fallback so that
/// termination is unconditional even under floating-point pathologies.
fn placidus_alpha(theta: f64, c: f64, g: f64, t: f64) -> f64 {
    let base = theta + c;
    let sda = |alpha: f64| acos((-t * sin(alpha)).clamp(-1.0, 1.0));
    let mut alpha = base + g * FRAC_PI_2;
    for _ in 0..PLACIDUS_MAX_ITER {
        let next = base + g * sda(alpha);
        let step = fabs(next - alpha);
        alpha = next;
        if step < PLACIDUS_TOL {
            return alpha;
        }
    }
    // Defensive fallback (unreachable in exact arithmetic; see proof).
    let (mut lo, mut hi) = (base, base + g * PI);
    for _ in 0..PLACIDUS_BISECT_STEPS {
        let mid = 0.5 * (lo + hi);
        if mid - base - g * sda(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Ecliptic longitude of the ecliptic point with right ascension
/// `alpha`: `λ = atan2(sin α, cos α cos ε)` (the inverse of the MC
/// relation `tan α = tan λ cos ε`, quadrant-correct).
fn ecliptic_of_ra(alpha: f64, eps: f64) -> f64 {
    normalize_0_two_pi(atan2(sin(alpha), cos(alpha) * cos(eps)))
}

/// Koch ("birthplace" system). The MC degree — right ascension `θ`,
/// declination `δ_MC` with `tan δ_MC = tan ε sin θ` — rose when the
/// sidereal time was `θ − H₀`, where `H₀ = SDA(δ_MC) =
/// acos(−tan φ tan δ_MC)` is its semi-diurnal arc. Koch divides the
/// rotation `H₀` into thirds and takes the **ascendants** when the
/// meridian has turned by each third (Astro\*Dictionary, "Koch House
/// System"; Urania Trust): the ascendant at `θ − H₀` is the MC degree
/// itself (cusp 10 "rising"), and
///
/// ```text
/// cusp 11 = Asc(θ − 2H₀/3)      cusp 12 = Asc(θ − H₀/3)
/// cusp  1 = Asc(θ)              cusp  2 = Asc(θ + H₀/3)
/// cusp  3 = Asc(θ + 2H₀/3)      (cusp 4 = Asc(θ + H₀) = IC degree)
/// ```
///
/// — the antipodal IC degree rises exactly when the MC degree sets, so
/// the ladder closes on the IC and the assembled wheel is consistent.
fn koch(theta: f64, phi: f64, eps: f64, asc: f64, mc_val: f64) -> Result<[f64; 12], HousesError> {
    let tan_product = tan(phi) * tan(eps);
    if fabs(tan_product) >= 1.0 {
        return Err(HousesError::Undefined);
    }
    // cos H0 = −tan φ tan δ_MC = −tan φ tan ε sin θ.
    let h0 = acos((-tan_product * sin(theta)).clamp(-1.0, 1.0));
    let third = h0 / 3.0;
    let shifted_asc = |shift: f64| -> Result<f64, HousesError> {
        ascendant(theta + shift, phi, eps).map_err(|_| HousesError::Undefined)
    };
    let c11 = shifted_asc(-2.0 * third)?;
    let c12 = shifted_asc(-third)?;
    let c2 = shifted_asc(third)?;
    let c3 = shifted_asc(2.0 * third)?;
    Ok(assemble(asc, mc_val, c11, c12, c2, c3))
}

/// Shared tail of the Regiomontanus/Campanus closed forms; `(y, x)` are
/// the ecliptic-frame components of the house-circle pole crossed with
/// the ecliptic pole (see [`crate::angles`]). A vanishing norm means the
/// house circle coincides with the ecliptic (possible only at or above
/// the polar circle) and the cusp is undefined.
fn branch_longitude(y: f64, x: f64) -> Result<f64, HousesError> {
    if sqrt(x * x + y * y) < COINCIDENCE_TOL {
        return Err(HousesError::Undefined);
    }
    Ok(normalize_0_two_pi(atan2(y, x)))
}
