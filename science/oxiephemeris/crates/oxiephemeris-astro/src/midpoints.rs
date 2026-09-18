//! Ecliptic midpoints: the near (short-arc) and far midpoints of two
//! longitudes, used for midpoint trees and composite (midpoint) charts.
//!
//! # Model
//!
//! Two ecliptic longitudes divide the circle into two arcs; each arc has
//! a midpoint, `180°` apart. The **near midpoint** bisects the shorter
//! arc — the conventional astrological "midpoint" — and the **far
//! midpoint** bisects the longer arc:
//!
//! ```text
//! near = a + wrap_pm_pi(b - a) / 2
//! far  = near + pi
//! ```
//!
//! `wrap_pm_pi` maps the signed difference into `(-pi, pi]`, so the half
//! step never crosses more than a quarter circle and the result is
//! independent of the argument order (`midpoint(a, b) == midpoint(b, a)`).
//!
//! A **composite chart** is built by taking the near midpoint of each
//! like pair of bodies (and angles) between two charts; [`midpoints`]
//! maps that operation over parallel longitude slices.
//!
//! # Clean-room provenance
//!
//! Midpoints are elementary circular means; no ephemeris source was
//! consulted.

use oxiephemeris_core::angle::{normalize_0_two_pi, normalize_pm_pi};

/// The near (short-arc) midpoint of two ecliptic longitudes (radians),
/// wrapped to `[0, 2*pi)`.
///
/// Symmetric in its arguments. When the two points are exactly opposite
/// (`pi` apart) the shorter arc is ambiguous; `normalize_pm_pi` resolves
/// the tie toward `+pi`, giving a deterministic result.
#[must_use]
pub fn midpoint(a_lon: f64, b_lon: f64) -> f64 {
    normalize_0_two_pi(a_lon + normalize_pm_pi(b_lon - a_lon) / 2.0)
}

/// The far (long-arc) midpoint: the near midpoint plus `pi`.
#[must_use]
pub fn far_midpoint(a_lon: f64, b_lon: f64) -> f64 {
    normalize_0_two_pi(midpoint(a_lon, b_lon) + core::f64::consts::PI)
}

/// The near midpoint of each aligned pair from two equal-length slices —
/// the longitudes of a composite (midpoint) chart.
///
/// Returns `None` if the slices differ in length. Writes
/// `min(a.len(), b.len())` results into `out`; `out` must be at least
/// that long. This `no_std`-friendly, allocation-free shape lets callers
/// own the storage.
#[must_use]
pub fn midpoints<'a>(a: &[f64], b: &[f64], out: &'a mut [f64]) -> Option<&'a [f64]> {
    if a.len() != b.len() || out.len() < a.len() {
        return None;
    }
    for (slot, (av, bv)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *slot = midpoint(*av, *bv);
    }
    Some(&out[..a.len()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    fn approx(a: f64, b: f64) -> bool {
        libm::fabs(normalize_pm_pi(a - b)) < 1e-12
    }

    #[test]
    fn simple_midpoint() {
        // 10 and 50 -> 30.
        assert!(approx(midpoint(deg(10.0), deg(50.0)), deg(30.0)));
    }

    #[test]
    fn midpoint_takes_the_short_arc_across_zero() {
        // 350 and 10 -> 0, not 180.
        assert!(approx(midpoint(deg(350.0), deg(10.0)), deg(0.0)));
        // 20 and 340 -> 0.
        assert!(approx(midpoint(deg(20.0), deg(340.0)), deg(0.0)));
    }

    #[test]
    fn midpoint_is_symmetric() {
        let mut x = 0.0;
        while x < 360.0 {
            let mut y = 0.0;
            while y < 360.0 {
                assert!(approx(midpoint(deg(x), deg(y)), midpoint(deg(y), deg(x))));
                y += 37.0;
            }
            x += 41.0;
        }
    }

    #[test]
    fn far_is_near_plus_half_turn() {
        assert!(approx(far_midpoint(deg(10.0), deg(50.0)), deg(30.0) + PI));
    }

    #[test]
    fn midpoints_maps_over_slices() {
        let a = [deg(10.0), deg(350.0)];
        let b = [deg(50.0), deg(10.0)];
        let mut out = [0.0; 2];
        let Some(result) = midpoints(&a, &b, &mut out) else {
            panic!("equal-length slices should map");
        };
        assert!(approx(result[0], deg(30.0)));
        assert!(approx(result[1], deg(0.0)));
    }

    #[test]
    fn midpoints_rejects_mismatched_lengths() {
        let a = [deg(10.0)];
        let b = [deg(50.0), deg(10.0)];
        let mut out = [0.0; 2];
        assert!(midpoints(&a, &b, &mut out).is_none());
    }
}
