//! IAU 2006 precession, Fukushima–Williams parameterization, frame bias
//! included.
//!
//! References:
//! - IERS Conventions (2010), TN36 eq. (5.40) for the polynomial
//!   developments of the angles `gamma_bar`, `phi_bar`, `psi_bar` (the
//!   last three series are from Table 1 of Hilton et al. 2006, Celest.
//!   Mech. Dyn. Astron. 94, 351).
//! - TN36 §5.4.5 for the rotation composition (method "proposed by
//!   Fukushima (2003) as an extension to the GCRS of the method
//!   originally proposed by Williams (1994)").

use super::obliquity::mean_obliquity_iau2006;
use super::ARCSEC_TO_RAD;
use crate::math::{r1, r3, Mat3};

/// The four Fukushima–Williams precession angles, in radians.
///
/// `gamma_bar` is the GCRS right ascension of the intersection of the
/// ecliptic of date with the GCRS equator, `phi_bar` the obliquity of the
/// ecliptic of date on the GCRS equator, `psi_bar` the precession angle
/// plus bias in longitude along the ecliptic of date, and `eps_a` the mean
/// obliquity of date (TN36 §5.6.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwAngles {
    /// `gamma_bar`, radians.
    pub gamma_bar_rad: f64,
    /// `phi_bar`, radians.
    pub phi_bar_rad: f64,
    /// `psi_bar`, radians.
    pub psi_bar_rad: f64,
    /// Mean obliquity of date `eps_A`, radians.
    pub eps_a_rad: f64,
}

/// Fukushima–Williams bias-precession angles for `t` Julian centuries TT
/// since J2000.0.
///
/// Polynomials in arcseconds (IERS TN36 eq. 5.40; Hilton et al. 2006,
/// Table 1):
///
/// ```text
/// gamma_bar = -0.052928" + 10.556378" t + 0.4932044" t^2 - 0.00031238" t^3
///             - 0.000002788" t^4 + 0.0000000260" t^5
/// phi_bar   = 84381.412819" - 46.811016" t + 0.0511268" t^2
///             + 0.00053289" t^3 - 0.000000440" t^4 - 0.0000000176" t^5
/// psi_bar   = -0.041775" + 5038.481484" t + 1.5584175" t^2
///             - 0.00018522" t^3 - 0.000026452" t^4 - 0.0000000148" t^5
/// ```
///
/// `eps_a` is [`mean_obliquity_iau2006`].
#[must_use]
// The bindings deliberately carry the names of the literature
// (`phi_bar` vs `psi_bar` are distinct angles in TN36 eq. 5.40).
#[allow(clippy::similar_names)]
pub fn fw_angles_iau2006(t: f64) -> FwAngles {
    let gamma_bar = -0.052_928
        + t * (10.556_378
            + t * (0.493_204_4 + t * (-0.000_312_38 + t * (-0.000_002_788 + t * 2.60e-8))));
    let phi_bar = 84_381.412_819
        + t * (-46.811_016
            + t * (0.051_126_8 + t * (0.000_532_89 + t * (-4.40e-7 + t * (-1.76e-8)))));
    let psi_bar = -0.041_775
        + t * (5_038.481_484
            + t * (1.558_417_5 + t * (-0.000_185_22 + t * (-0.000_026_452 + t * (-1.48e-8)))));
    FwAngles {
        gamma_bar_rad: gamma_bar * ARCSEC_TO_RAD,
        phi_bar_rad: phi_bar * ARCSEC_TO_RAD,
        psi_bar_rad: psi_bar * ARCSEC_TO_RAD,
        eps_a_rad: mean_obliquity_iau2006(t),
    }
}

/// Combined frame-bias + precession matrix `PB(t)`: rotates GCRS vectors
/// to the mean equator and equinox of date, **frame bias included**.
///
/// Composition (TN36 §5.4.5, with nutation set to zero):
///
/// ```text
/// PB(t) = R1(-eps_A) . R3(-psi_bar) . R1(phi_bar) . R3(gamma_bar)
/// ```
///
/// At `t = 0` this is exactly the GCRS frame-bias matrix `B` (the FW
/// angles absorb the celestial-pole offsets `xi_0`, `eta_0` and the
/// equinox offset `d_alpha_0` of TN36 eqs. 5.21/5.33): its off-diagonal
/// elements are of order tens of milliarcseconds, not zero.
#[must_use]
pub fn precession_bias_matrix(t: f64) -> Mat3 {
    let fw = fw_angles_iau2006(t);
    fw_rotation(&fw, 0.0, 0.0)
}

/// Assembles `R1(-(eps_a + deps)) . R3(-(psi_bar + dpsi)) . R1(phi_bar)
/// . R3(gamma_bar)` — the general Fukushima–Williams 4-rotation of TN36
/// §5.4.5. With `dpsi = deps = 0` it is the bias-precession matrix `PB`;
/// with the nutation components added it is the full `N . PB` (the two
/// forms are algebraically identical because
/// `R1(a) . R3(b) . R1(-a) . R1(a) = R1(a) . R3(b)` collapses the inner
/// rotations).
pub(crate) fn fw_rotation(fw: &FwAngles, dpsi_rad: f64, deps_rad: f64) -> Mat3 {
    r1(-(fw.eps_a_rad + deps_rad))
        .mul(&r3(-(fw.psi_bar_rad + dpsi_rad)))
        .mul(&r1(fw.phi_bar_rad))
        .mul(&r3(fw.gamma_bar_rad))
}
