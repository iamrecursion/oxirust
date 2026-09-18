//! Mean obliquity of the ecliptic, IAU 2006 (P03).

use super::ARCSEC_TO_RAD;

/// Mean obliquity of the ecliptic `eps_A`, in radians, for `t` Julian
/// centuries TT since J2000.0.
///
/// IAU 2006 (P03) development, from IERS Conventions (2010), TN36
/// eq. (5.40) — identical to Capitaine, Wallace & Chapront (2003),
/// A&A 412, 567, eq. (39) — with `eps_0 = 84381.406` arcsec:
///
/// ```text
/// eps_A = 84381.406" - 46.836769" t - 0.0001831" t^2 + 0.00200340" t^3
///         - 0.000000576" t^4 - 0.0000000434" t^5
/// ```
#[must_use]
pub fn mean_obliquity_iau2006(t: f64) -> f64 {
    let arcsec = 84_381.406
        + t * (-46.836_769
            + t * (-0.000_183_1 + t * (0.002_003_40 + t * (-5.76e-7 + t * (-4.34e-8)))));
    arcsec * ARCSEC_TO_RAD
}
