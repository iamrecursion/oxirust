//! Angle unit constants and normalization helpers.
//!
//! All constants are derived from [`core::f64::consts::PI`]; no decimal
//! approximations are hard-coded.

use core::f64::consts::PI;

/// 2π, the full circle in radians.
pub const TWO_PI: f64 = 2.0 * PI;

/// Degrees to radians: π / 180.
pub const DEG2RAD: f64 = PI / 180.0;

/// Radians to degrees: 180 / π.
pub const RAD2DEG: f64 = 180.0 / PI;

/// Arcseconds to radians: π / (180 × 3600).
pub const AS2R: f64 = DEG2RAD / 3600.0;

/// Milliarcseconds to radians.
pub const MAS2R: f64 = AS2R / 1e3;

/// Microarcseconds to radians.
pub const UAS2R: f64 = AS2R / 1e6;

/// Normalize an angle in radians to the half-open interval `[0, 2π)`.
///
/// Uses Euclidean-remainder logic (`x − 2π · ⌊x / 2π⌋`) so that negative
/// inputs land in `[0, 2π)` as well. A result that rounds up to exactly 2π
/// (possible for tiny negative inputs) is mapped to `0.0`.
///
/// Non-finite inputs propagate (NaN stays NaN).
#[must_use]
pub fn normalize_0_two_pi(angle: f64) -> f64 {
    let r = angle - TWO_PI * libm::floor(angle / TWO_PI);
    // Rounding can produce r == 2π when `angle` is a tiny negative number.
    if r >= TWO_PI {
        0.0
    } else {
        r
    }
}

/// Normalize an angle in radians to the half-open interval `[−π, π)`.
///
/// Uses Euclidean-remainder logic on the shifted angle, with a final guard
/// against boundary rounding so the contract `−π ≤ result < π` holds for
/// every finite input.
///
/// Non-finite inputs propagate (NaN stays NaN).
#[must_use]
pub fn normalize_pm_pi(angle: f64) -> f64 {
    let r = angle - TWO_PI * libm::floor((angle + PI) / TWO_PI);
    if r < -PI {
        r + TWO_PI
    } else if r >= PI {
        r - TWO_PI
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_0_two_pi, normalize_pm_pi, AS2R, DEG2RAD, MAS2R, TWO_PI};
    use core::f64::consts::PI;

    #[test]
    fn constants_are_consistent() {
        assert!((DEG2RAD * 180.0 - PI).abs() < 1e-15);
        assert!((AS2R * 3600.0 - DEG2RAD).abs() < 1e-18);
        assert!((MAS2R * 1000.0 - AS2R).abs() < 1e-21);
    }

    #[test]
    fn normalize_zero_two_pi_ranges() {
        let samples = [
            -1e9, -12345.678, -TWO_PI, -PI, -1e-18, 0.0, 1.0, PI, TWO_PI, 12345.678, 1e9,
        ];
        for &x in &samples {
            let r = normalize_0_two_pi(x);
            assert!((0.0..TWO_PI).contains(&r), "out of range for {x}: {r}");
            // The result differs from the input by a whole number of turns.
            let turns = (x - r) / TWO_PI;
            assert!(
                (turns - libm::round(turns)).abs() < 1e-6,
                "not a whole turn for {x}"
            );
        }
    }

    #[test]
    fn normalize_pm_pi_ranges() {
        let samples = [
            -1e9, -12345.678, -TWO_PI, -PI, -1e-18, 0.0, 1.0, PI, TWO_PI, 12345.678, 1e9,
        ];
        for &x in &samples {
            let r = normalize_pm_pi(x);
            assert!((-PI..PI).contains(&r), "out of range for {x}: {r}");
            let turns = (x - r) / TWO_PI;
            assert!(
                (turns - libm::round(turns)).abs() < 1e-6,
                "not a whole turn for {x}"
            );
        }
    }

    #[test]
    fn pi_maps_to_minus_pi() {
        // π is normalized into [−π, π) as −π.
        let r = normalize_pm_pi(PI);
        assert!((r + PI).abs() < 1e-15);
    }
}
