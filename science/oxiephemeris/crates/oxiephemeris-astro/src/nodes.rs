//! Mean and true (osculating) lunar node and apogee.
//!
//! Two independent layers:
//!
//! 1. **Mean elements** (data-free): the mean longitude of the Moon's
//!    ascending node and of its perigee from the published polynomials of
//!    Simon et al. (1994) — [`mean_node`], [`mean_perigee`],
//!    [`mean_apogee`].
//! 2. **Osculating elements** (state-driven): the classical two-body
//!    elements of an instantaneous geocentric state vector —
//!    [`osculating_nodes`] as a pure function of `(r, v, mu)`, and
//!    [`true_node_of_date`] as the DE-ephemeris convenience that yields the
//!    Swiss-Ephemeris-style *true node* and *osculating apogee* of date.
//!
//! # Clean-room provenance
//!
//! * J. L. Simon, P. Bretagnon, J. Chapront, M. Chapront-Touzé, G. Francou
//!   & J. Laskar (1994), "Numerical expressions for precession formulae
//!   and mean elements for the Moon and the planets", A&A 282, 663 —
//!   Sect. 3.4 (b.3): lunar mean elements referred to the **mean ecliptic
//!   and equinox of date**, computed for the "1992 values" of the
//!   constants (their Table 2) with the precession constant of Williams
//!   et al. (1991), `P_1 = 5028.8200"/cy^2`. This is the same solution the
//!   IERS Conventions (2010), TN36 eq. (5.43), adopt for the Delaunay
//!   arguments (already used by `oxiephemeris-bodies` for nutation), so
//!   the mean node here is *identical* to the Delaunay argument `Om`, and
//!   the mean perigee satisfies `pi = F + Om - l` exactly.
//! * Osculating elements: textbook two-body definitions (e.g. R. R. Bate,
//!   D. D. Mueller & J. E. White, *Fundamentals of Astrodynamics*, Dover
//!   1971, §2.4, "Determining the orbital elements from r and v"):
//!   `h = r × v`, node vector `n = ẑ × h`, eccentricity vector
//!   `e = (v × h)/mu − r/|r|`.
//!
//! No Swiss Ephemeris or SOFA/ERFA source was consulted; behavioral
//! compatibility with SE is limited to the documented output conventions.
//!
//! # Conventions
//!
//! All angles are radians normalized to `[0, 2*pi)`. The time argument `t`
//! of the mean elements is Julian centuries **TDB** from J2000.0
//! (JD 2451545.0 TDB); using TT instead shifts `t` by < 2 ms, i.e. the
//! node by < 4 nano-arcseconds — negligible against the accuracy of the
//! mean theory itself. Longitudes of [`true_node_of_date`] are referred to
//! the **mean ecliptic and equinox of date** (the frame of the Simon et
//! al. mean elements, so mean and true node are directly comparable). A
//! true-equinox (apparent, SE default output frame) longitude is obtained
//! by adding the nutation in longitude `dpsi`
//! ([`oxiephemeris_bodies::frames::Nutation`]), which shifts the equinox
//! along the ecliptic.

use core::f64::consts::PI;

use libm::{atan2, sqrt};
use oxiephemeris_bodies::frames::{mean_of_date_to_ecliptic, precession_bias_matrix};
use oxiephemeris_bodies::math::{cross, dot, norm, scale, sub, Vec3};
use oxiephemeris_core::angle::{normalize_0_two_pi, AS2R};
use oxiephemeris_core::time::{tdb_minus_tt_seconds, JulianDate, J2000_JD};
use oxiephemeris_de::{DeError, DeFile, Series};

/// Arcseconds per full turn (`360 * 3600`).
const TURN_ARCSEC: f64 = 1_296_000.0;

/// Days per Julian century (IERS TN36 eq. 5.2).
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// Relative floor for the specific angular momentum: `|h|` must exceed
/// `MIN_H_REL * |r| * |v|`, otherwise the state is (numerically)
/// rectilinear and no orbit plane exists. `|h| = |r||v| sin(zenith angle)`,
/// so this is a `~2e-10` degree floor on the flight-path geometry — pure
/// guard against corrupt input, never binding for the Moon
/// (`sin ~ 0.99`).
const MIN_H_REL: f64 = 1e-12;

/// Relative floor for the node vector: `|n| = |h| sin i` must exceed
/// `MIN_NODE_SINE * |h|`, i.e. the osculating inclination to the reference
/// plane must exceed `~2e-7` degrees, otherwise the ascending node is
/// undefined. The Moon's inclination to the ecliptic stays within
/// `5.15 deg +- 0.15 deg`, four million times above the floor.
const MIN_NODE_SINE: f64 = 1e-9;

/// Absolute floor for the eccentricity (and for the in-plane projection of
/// the eccentricity vector onto the reference plane): below it the apsis
/// direction is undefined. The Moon's osculating eccentricity stays within
/// `0.026..0.077` (Simon et al. 1994, Table 4: `0.0555 +- 0.0142 ...`),
/// seven orders above the floor.
const MIN_ECCENTRICITY: f64 = 1e-9;

/// Sanity window for the DE header constant `GMB` in AU^3/day^2
/// (`asc2eph.f` GROUP 1041 stores the integration GMs in AU^3/day^2; the
/// DE440/DE441 value is `8.997011392947347e-10`, Park et al. 2021,
/// AJ 161, 105, Table 3). One decade of slack on each side.
const GMB_SANITY_RANGE: (f64, f64) = (1e-10, 1e-8);

/// Errors of the lunar node/apogee computations. No function in this
/// module panics; every failure is reported through this type.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeError {
    /// The underlying DE evaluation failed (epoch out of range, truncated
    /// file, ...).
    De(DeError),
    /// `|r|`, `|v|` or `|r x v|` vanishes: the state is rectilinear (or
    /// zero) and defines no orbit plane.
    NoOrbitPlane,
    /// The orbit plane coincides with the reference plane
    /// (`|z_hat x h| ~ 0`): the ascending node is undefined.
    NodeUndefined,
    /// The eccentricity vector vanishes (circular orbit) or is normal to
    /// the reference plane: the apsis longitude is undefined.
    ApsisUndefined,
    /// The DE header lacks a sane `GMB` constant, so `mu = GM_Earth +
    /// GM_Moon` cannot be derived from the file. There is deliberately no
    /// literature fallback: the osculating eccentricity comparison level
    /// (1e-5, see the oracle tests) requires the exact constant the
    /// ephemeris was integrated with.
    MuUnavailable,
}

impl core::fmt::Display for NodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::De(e) => write!(f, "ephemeris evaluation failed: {e}"),
            Self::NoOrbitPlane => f.write_str("state is rectilinear: no orbit plane"),
            Self::NodeUndefined => f.write_str("orbit lies in the reference plane: node undefined"),
            Self::ApsisUndefined => {
                f.write_str("orbit is circular or apsis is normal to the reference plane")
            }
            Self::MuUnavailable => {
                f.write_str("DE header lacks a sane GMB constant for mu = GM_E + GM_M")
            }
        }
    }
}

impl core::error::Error for NodeError {}

impl From<DeError> for NodeError {
    fn from(e: DeError) -> Self {
        Self::De(e)
    }
}

/// Mean longitude of the Moon's **ascending node**, radians in
/// `[0, 2*pi)`, referred to the mean ecliptic and equinox of date, for `t`
/// Julian centuries TDB (~TT, see the module docs) from J2000.0.
///
/// Simon et al. (1994), Sect. 3.4 (b.3) ("1992 values" of the constants,
/// precession constant `P_1 = 5028.8200"/cy^2` from Williams et al. 1991;
/// constant term published in degrees, `t^n` coefficients in `"/cy^n`):
///
/// ```text
/// <Omega> = 125.04455501 deg - 6962890.5431" t + 7.4722" t^2
///           + 0.007702" t^3 - 0.00005939" t^4
/// ```
///
/// with `125.04455501 deg = 450160.398036"`. This polynomial is digit-for-
/// digit the Delaunay argument `Om` of IERS TN36 eq. (5.43) (same source),
/// i.e. exactly the node polynomial `oxiephemeris-bodies` feeds the
/// IAU 2000A nutation series. Validity: Simon et al. Sect. 3.6 discuss
/// [4000 BC, AD 8000]; the arcsecond-level accuracy degrades outside a few
/// tens of centuries around J2000.
#[must_use]
pub fn mean_node(t: f64) -> f64 {
    let omega_arcsec = 450_160.398_036
        + t * (-6_962_890.543_1 + t * (7.4722 + t * (0.007_702 + t * (-0.000_059_39))));
    normalize_0_two_pi((omega_arcsec % TURN_ARCSEC) * AS2R)
}

/// Mean longitude of the Moon's **perigee**, radians in `[0, 2*pi)`,
/// referred to the mean ecliptic and equinox of date, for `t` Julian
/// centuries TDB (~TT) from J2000.0.
///
/// Simon et al. (1994), Sect. 3.4 (b.3) (same solution as [`mean_node`]):
///
/// ```text
/// <pi> = 83.35324312 deg + 14648449.0869" t - 37.1582" t^2
///        - 0.044970" t^3 + 0.00018948" t^4
/// ```
///
/// with `83.35324312 deg = 300071.675232"`. Consistency: `<pi>` equals
/// `F + Om - l` of the Delaunay arguments of IERS TN36 eq. (5.43)
/// coefficient-for-coefficient (both are the Simon et al. "1992 values"
/// solution).
#[must_use]
pub fn mean_perigee(t: f64) -> f64 {
    let pi_arcsec = 300_071.675_232
        + t * (14_648_449.086_9 + t * (-37.1582 + t * (-0.044_970 + t * 0.000_189_48)));
    normalize_0_two_pi((pi_arcsec % TURN_ARCSEC) * AS2R)
}

/// Mean longitude of the Moon's **apogee**, radians in `[0, 2*pi)`,
/// referred to the mean ecliptic and equinox of date:
/// `mean_apogee(t) = mean_perigee(t) + 180 deg (mod 360 deg)`.
///
/// This is the point Swiss-Ephemeris-style APIs expose as the "mean
/// apogee" (a.k.a. Lilith / the mean Black Moon); the SE programmer
/// documentation defines it as the mean perigee + 180 deg, which is the
/// documented convention adopted here (API-surface compatibility only; no
/// SE code was consulted).
#[must_use]
pub fn mean_apogee(t: f64) -> f64 {
    normalize_0_two_pi(mean_perigee(t) + PI)
}

/// Osculating (instantaneous two-body) elements of a geocentric state,
/// referred to the reference plane and origin of longitude of the frame
/// the state was supplied in (see [`osculating_nodes`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OsculatingElements {
    /// Longitude of the ascending node `Omega = atan2(n_y, n_x)`, radians
    /// in `[0, 2*pi)`.
    pub node_lon_rad: f64,
    /// Ecliptic longitude of the **apogee direction**: the projection of
    /// `-e` onto the reference plane, `atan2(-e_y, -e_x)`, radians in
    /// `[0, 2*pi)`. This is the SE-style "osculating apogee" (true
    /// Lilith) longitude convention. Note it is *not* the dog-leg angle
    /// `Omega + omega + pi`: for the Moon's `i ~ 5.15 deg` the in-plane
    /// and projected apsis longitudes differ by up to
    /// `~ tan^2(i/2) sin(2(omega + pi)) ~ 0.12 deg`.
    pub apogee_lon_rad: f64,
    /// Argument of perigee `omega`: in-plane angle from the ascending node
    /// to the eccentricity vector, measured about `+h`, radians in
    /// `[0, 2*pi)`.
    pub arg_perigee_rad: f64,
    /// Inclination to the reference plane `i = atan2(|n|, h_z)`, radians
    /// in `[0, pi]`.
    pub inclination_rad: f64,
    /// Eccentricity `e = |e_vec|` (dimensionless).
    pub eccentricity: f64,
    /// Semi-major axis `a = 1 / (2/|r| - v^2/mu)`, in the length unit of
    /// `r` (AU for [`true_node_of_date`]). Negative for a hyperbolic
    /// state, `+-inf` for an exactly parabolic one (never the case for
    /// the Moon).
    pub semi_major_axis: f64,
}

/// Osculating node/apogee elements as a **pure function** of a geocentric
/// state vector: position `r` (AU), velocity `v` (AU/day) and
/// `mu = GM_Earth + GM_Moon` (AU^3/day^2). Any consistent length/time
/// units work; the outputs are angles, `eccentricity`, and
/// `semi_major_axis` in the caller's length unit.
///
/// # Frame
///
/// The caller chooses (and must document) the frame of `r`/`v`; the
/// elements are referred to its x-y plane (reference plane) and +x axis
/// (origin of longitudes). Typical choices, both reachable with
/// `oxiephemeris-bodies` rotations applied to the ICRS/GCRS vectors of a
/// DE file:
///
/// * **J2000/ICRF ecliptic** — e.g. `r1(eps)` on ICRF axes (what JPL
///   Horizons calls "Ecliptic of J2000.0"), or the frame-bias-corrected
///   J2000 mean ecliptic;
/// * **mean ecliptic and equinox of date** —
///   [`mean_of_date_to_ecliptic`]`(t) *` [`precession_bias_matrix`]`(t)`,
///   which is what [`true_node_of_date`] uses.
///
/// # Definitions (Bate, Mueller & White 1971, §2.4)
///
/// ```text
/// h = r x v                    (specific angular momentum)
/// n = z_hat x h                (node vector, in the reference plane)
/// e = (v x h)/mu - r/|r|       (eccentricity vector, points to perigee)
/// Omega = atan2(n_y, n_x)      i = atan2(|n|, h_z)
/// omega = angle from n to e about +h
/// lambda_apogee = atan2(-e_y, -e_x)
/// a = 1 / (2/|r| - v^2/mu)     (vis-viva)
/// ```
///
/// # Errors
///
/// * [`NodeError::NoOrbitPlane`] — `|r|`, `|v|` or `|h|` vanishes
///   (rectilinear/zero state).
/// * [`NodeError::NodeUndefined`] — orbit lies in the reference plane
///   (`sin i` below `MIN_NODE_SINE`).
/// * [`NodeError::ApsisUndefined`] — circular orbit, or apsis direction
///   normal to the reference plane (below `MIN_ECCENTRICITY`).
pub fn osculating_nodes(
    r_au: Vec3,
    v_au_day: Vec3,
    mu_au3_day2: f64,
) -> Result<OsculatingElements, NodeError> {
    let r_norm = norm(r_au);
    let v_norm = norm(v_au_day);
    if r_norm <= 0.0 || v_norm <= 0.0 || !(r_norm.is_finite() && v_norm.is_finite()) {
        return Err(NodeError::NoOrbitPlane);
    }
    let h_vec = cross(r_au, v_au_day);
    let h_norm = norm(h_vec);
    if h_norm <= MIN_H_REL * r_norm * v_norm {
        return Err(NodeError::NoOrbitPlane);
    }

    // Node vector n = z_hat x h = (-h_y, h_x, 0); |n| = |h| sin i.
    let n_vec: Vec3 = [-h_vec[1], h_vec[0], 0.0];
    let n_norm = sqrt(n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1]);
    if n_norm <= MIN_NODE_SINE * h_norm {
        return Err(NodeError::NodeUndefined);
    }
    let node_lon_rad = normalize_0_two_pi(atan2(n_vec[1], n_vec[0]));
    let inclination_rad = atan2(n_norm, h_vec[2]);

    // Eccentricity vector e = (v x h)/mu - r_hat.
    let e_vec = sub(
        scale(cross(v_au_day, h_vec), 1.0 / mu_au3_day2),
        scale(r_au, 1.0 / r_norm),
    );
    let eccentricity = norm(e_vec);
    let e_proj = sqrt(e_vec[0] * e_vec[0] + e_vec[1] * e_vec[1]);
    if eccentricity <= MIN_ECCENTRICITY || e_proj <= MIN_ECCENTRICITY {
        return Err(NodeError::ApsisUndefined);
    }
    let apogee_lon_rad = normalize_0_two_pi(atan2(-e_vec[1], -e_vec[0]));

    // Argument of perigee: in-plane angle from n to e about +h. Both the
    // sine (via the triple product) and the cosine carry the common factor
    // |n||e|, which cancels inside atan2.
    let arg_perigee_rad = normalize_0_two_pi(atan2(
        dot(h_vec, cross(n_vec, e_vec)) / h_norm,
        dot(n_vec, e_vec),
    ));

    // Vis-viva. An exactly parabolic state yields +-inf, documented.
    let semi_major_axis = 1.0 / (2.0 / r_norm - v_norm * v_norm / mu_au3_day2);

    Ok(OsculatingElements {
        node_lon_rad,
        apogee_lon_rad,
        arg_perigee_rad,
        inclination_rad,
        eccentricity,
        semi_major_axis,
    })
}

/// `mu = GM_Earth + GM_Moon` in AU^3/day^2, from the DE header constant
/// `GMB`.
///
/// The classic-binary DE header (record 2, `asc2eph.f` GROUP 1040/1041)
/// carries the gravitational parameter of the Earth-Moon **barycenter**,
/// `GMB`, in AU^3/day^2 — by construction the sum `GM_Earth + GM_Moon`
/// that drives the barycenter decomposition (the individual values follow
/// as `GM_Earth = GMB * EMRAT / (1 + EMRAT)`, `GM_Moon = GMB / (1 +
/// EMRAT)`), which is exactly the two-body `mu` of the geocentric lunar
/// orbit. DE440/DE441 value: `8.997011392947347e-10` — identical to the
/// "Keplerian GM" JPL Horizons quotes for geocentric lunar osculating
/// elements (see the `horizons_elements` fixtures).
///
/// # Errors
///
/// [`NodeError::MuUnavailable`] if the header lacks `GMB` or its value is
/// outside `GMB_SANITY_RANGE` (deliberately no literature fallback; see
/// the error's docs).
pub fn gm_earth_moon_au3_day2(de: &DeFile<'_>) -> Result<f64, NodeError> {
    match de.constant("GMB") {
        Some(gmb) if gmb > GMB_SANITY_RANGE.0 && gmb < GMB_SANITY_RANGE.1 => Ok(gmb),
        _ => Err(NodeError::MuUnavailable),
    }
}

/// True (osculating) lunar node and apogee **of date** from a DE
/// ephemeris: geocentric Moon state at `jd_tt`, rotated GCRS → mean
/// ecliptic and equinox of date, elements via [`osculating_nodes`] with
/// `mu` from the DE header ([`gm_earth_moon_au3_day2`]).
///
/// Pipeline (mirrors `oxiephemeris_bodies::apparent`):
///
/// 1. TT → TDB once (Fairhead–Bretagnon,
///    [`oxiephemeris_core::time::tdb_minus_tt_seconds`]); the DE argument
///    is TDB (Park et al. 2021, AJ 161, 105, §2).
/// 2. Geocentric Moon position and velocity from the file's `Moon` series
///    (km, km/day, ICRS/GCRS axes), converted to AU and AU/day with the
///    header `AU`.
/// 3. Rotation into the mean ecliptic and equinox of date:
///    [`mean_of_date_to_ecliptic`]`(t) *` [`precession_bias_matrix`]`(t)`
///    with `t` Julian centuries TT from J2000.0 — the same matrix as the
///    `EclipticMeanOfDate` output frame of the apparent pipeline. The
///    velocity is rotated by the same fixed matrix: the neglected frame
///    rotation rate (general precession, `~5029"/cy = 7.8e-9 rad/day`)
///    perturbs `v` by `< 2e-8` relative, i.e. the elements at the 1e-8
///    level — far below the mean-vs-true comparison scales.
/// 4. [`osculating_nodes`] in that frame.
///
/// # Output convention (SE-compatible)
///
/// `node_lon_rad` / `apogee_lon_rad` are of-date ecliptic longitudes from
/// the **mean equinox of date** — directly comparable to [`mean_node`] /
/// [`mean_apogee`]. For an apparent (true-equinox) longitude add the
/// nutation in longitude `dpsi` of
/// [`oxiephemeris_bodies::frames::nutation_iau2000a`].
///
/// # Errors
///
/// [`NodeError::De`] when the DE evaluation fails (epoch outside the file
/// span, ...), [`NodeError::MuUnavailable`] for a header without a sane
/// `GMB`, and the degenerate-geometry variants of [`osculating_nodes`]
/// (unreachable for physical lunar states).
pub fn true_node_of_date(
    de: &DeFile<'_>,
    jd_tt: JulianDate,
) -> Result<OsculatingElements, NodeError> {
    let mu = gm_earth_moon_au3_day2(de)?;
    let jd_tdb = jd_tt.add_seconds(tdb_minus_tt_seconds(jd_tt));
    let state = de.series_state(Series::Moon, (jd_tdb.hi, jd_tdb.lo))?;
    let au_km = de.au_km();
    let r_gcrs: Vec3 = [
        state.value[0] / au_km,
        state.value[1] / au_km,
        state.value[2] / au_km,
    ];
    let v_gcrs: Vec3 = [
        state.rate[0] / au_km,
        state.rate[1] / au_km,
        state.rate[2] / au_km,
    ];
    let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;
    let rot = mean_of_date_to_ecliptic(t).mul(&precession_bias_matrix(t));
    osculating_nodes(rot.apply(r_gcrs), rot.apply(v_gcrs), mu)
}

#[cfg(test)]
mod tests {
    use super::{mean_apogee, mean_node, mean_perigee, osculating_nodes, NodeError, PI};
    use libm::{atan2, cos, fabs, sin, sqrt};
    use oxiephemeris_bodies::math::{r1, r3, Vec3};
    use oxiephemeris_core::angle::{normalize_pm_pi, DEG2RAD};

    /// DE440-like `mu` for synthetic states, AU^3/day^2.
    const MU: f64 = 8.997e-10;

    #[test]
    fn mean_elements_match_published_constants_at_j2000() {
        // Simon et al. (1994), Sect. 3.4 (b.3), constant terms in degrees.
        assert!(fabs(mean_node(0.0) - 125.044_555_01 * DEG2RAD) < 1e-12);
        assert!(fabs(mean_perigee(0.0) - 83.353_243_12 * DEG2RAD) < 1e-12);
        assert!(fabs(mean_apogee(0.0) - 263.353_243_12 * DEG2RAD) < 1e-12);
    }

    #[test]
    fn mean_apogee_is_perigee_plus_half_turn() {
        for k in 0_u32..40 {
            let t = -2.0 + 0.1 * f64::from(k);
            let d = normalize_pm_pi(mean_apogee(t) - mean_perigee(t) - PI);
            assert!(fabs(d) < 1e-12, "t = {t}: apogee - perigee - pi = {d}");
        }
    }

    /// Builds the ecliptic-frame state of a Keplerian orbit at true
    /// anomaly `nu` (perifocal state rotated by `R3(-Omega) R1(-i)
    /// R3(-omega)`), then checks that `osculating_nodes` recovers every
    /// element. Semi-major axis and eccentricity are Moon-like.
    #[test]
    fn recovers_elements_of_a_synthetic_keplerian_state() {
        let a = 0.002_57; // AU
        let ecc = 0.054_9;
        let incl = 5.145 * DEG2RAD;
        let node = 40.0 * DEG2RAD;
        let argp = 70.0 * DEG2RAD;
        let nu = 123.0 * DEG2RAD;

        let p = a * (1.0 - ecc * ecc); // semi-latus rectum
        let r_mag = p / (1.0 + ecc * cos(nu));
        let r_pf: Vec3 = [r_mag * cos(nu), r_mag * sin(nu), 0.0];
        let vs = sqrt(MU / p);
        let v_pf: Vec3 = [-vs * sin(nu), vs * (ecc + cos(nu)), 0.0];
        // Perifocal -> ecliptic: the inverse of the frame rotation
        // R3(omega) R1(i) R3(Omega).
        let m = r3(-node).mul(&r1(-incl)).mul(&r3(-argp));
        let r_ecl = m.apply(r_pf);
        let v_ecl = m.apply(v_pf);

        let osc = match osculating_nodes(r_ecl, v_ecl, MU) {
            Ok(o) => o,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(fabs(normalize_pm_pi(osc.node_lon_rad - node)) < 1e-12);
        assert!(fabs(normalize_pm_pi(osc.arg_perigee_rad - argp)) < 1e-12);
        assert!(fabs(osc.inclination_rad - incl) < 1e-12);
        assert!(fabs(osc.eccentricity - ecc) < 1e-12);
        assert!(fabs(osc.semi_major_axis - a) / a < 1e-12);

        // Independent spherical-trig identity for the projected apogee
        // longitude: with u the in-plane angle of the apogee from the
        // node (u = omega + pi), tan(lambda_apo - Omega) = cos i tan u.
        let u = argp + PI;
        let expected_apo = node + atan2(cos(incl) * sin(u), cos(u));
        assert!(fabs(normalize_pm_pi(osc.apogee_lon_rad - expected_apo)) < 1e-12);
    }

    #[test]
    fn equatorial_orbit_has_no_node() {
        // Circularish orbit exactly in the reference plane.
        let r: Vec3 = [0.00257, 0.0, 0.0];
        let v: Vec3 = [0.0, sqrt(MU / 0.00257), 0.0];
        assert!(matches!(
            osculating_nodes(r, v, MU),
            Err(NodeError::NodeUndefined)
        ));
    }

    #[test]
    fn circular_orbit_has_no_apsis() {
        // Exactly circular, inclined orbit: node defined, apsis not.
        let r_mag = 0.00257;
        let v_mag = sqrt(MU / r_mag);
        let incl = 5.0 * DEG2RAD;
        let r: Vec3 = [r_mag, 0.0, 0.0];
        let v: Vec3 = [0.0, v_mag * cos(incl), v_mag * sin(incl)];
        assert!(matches!(
            osculating_nodes(r, v, MU),
            Err(NodeError::ApsisUndefined)
        ));
    }

    #[test]
    fn rectilinear_state_has_no_orbit_plane() {
        let r: Vec3 = [0.00257, 0.0, 0.0];
        let v: Vec3 = [1e-4, 0.0, 0.0]; // radial fall: r x v = 0
        assert!(matches!(
            osculating_nodes(r, v, MU),
            Err(NodeError::NoOrbitPlane)
        ));
        assert!(matches!(
            osculating_nodes([0.0, 0.0, 0.0], v, MU),
            Err(NodeError::NoOrbitPlane)
        ));
    }

    #[test]
    fn error_display_is_informative() {
        extern crate alloc;
        use alloc::string::ToString;
        assert!(NodeError::NoOrbitPlane.to_string().contains("orbit plane"));
        assert!(NodeError::NodeUndefined.to_string().contains("node"));
        assert!(NodeError::ApsisUndefined.to_string().contains("circular"));
        assert!(NodeError::MuUnavailable.to_string().contains("GMB"));
        assert!(NodeError::from(oxiephemeris_de::DeError::EpochOutOfRange)
            .to_string()
            .contains("ephemeris"));
    }
}
