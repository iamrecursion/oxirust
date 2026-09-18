//! Rate-distortion (RD) curve analysis and optimal QP/CRF finder.
//!
//! Provides types and algorithms for working with rate-distortion data:
//! building piecewise-linear interpolating curves from measured data points,
//! querying bitrate at a given quality level (and vice versa), and locating
//! the QP/CRF value that best matches a target bitrate.

use serde::{Deserialize, Serialize};

/// A single point on an RD curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdPoint {
    /// Quantisation parameter or constant rate factor (lower = better quality).
    pub qp_or_crf: f32,
    /// Measured bitrate in kbps.
    pub bitrate_kbps: f64,
    /// Measured PSNR in dB.
    pub psnr_db: f64,
}

impl RdPoint {
    /// Create a new RD point.
    #[must_use]
    pub fn new(qp_or_crf: f32, bitrate_kbps: f64, psnr_db: f64) -> Self {
        Self {
            qp_or_crf,
            bitrate_kbps,
            psnr_db,
        }
    }
}

/// An RD curve consisting of multiple [`RdPoint`] measurements.
///
/// Internally the points are kept sorted by bitrate (ascending) for efficient
/// interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdCurve {
    /// Sorted (ascending bitrate) RD points.
    pub points: Vec<RdPoint>,
}

impl RdCurve {
    /// Create a new empty RD curve.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Build an RD curve from a collection of points, sorting by bitrate.
    #[must_use]
    pub fn from_points(mut points: Vec<RdPoint>) -> Self {
        points.sort_by(|a, b| {
            a.bitrate_kbps
                .partial_cmp(&b.bitrate_kbps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { points }
    }

    /// Add a point and maintain sorted order.
    pub fn add_point(&mut self, pt: RdPoint) {
        let pos = self
            .points
            .partition_point(|p| p.bitrate_kbps < pt.bitrate_kbps);
        self.points.insert(pos, pt);
    }

    /// Number of points on the curve.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the curve has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    // ------------------------------------------------------------------
    // Interpolation — bitrate at target PSNR
    // ------------------------------------------------------------------

    /// Interpolate the bitrate (kbps) required to achieve the given PSNR level.
    ///
    /// The curve must have at least two points.  Returns `None` if the target
    /// PSNR is outside the range of the curve or fewer than 2 points exist.
    ///
    /// The interpolation is performed in the log-bitrate domain to better
    /// represent the exponential nature of codec RD curves.
    #[must_use]
    pub fn interpolate_bitrate_at_psnr(&self, psnr: f64) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        // Sort a temporary copy by PSNR (ascending) for this query.
        let mut by_psnr = self.points.clone();
        by_psnr.sort_by(|a, b| {
            a.psnr_db
                .partial_cmp(&b.psnr_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let first_psnr = by_psnr.first()?.psnr_db;
        let last_psnr = by_psnr.last()?.psnr_db;

        if psnr < first_psnr || psnr > last_psnr {
            return None;
        }

        // Find bracket.
        let idx = by_psnr.partition_point(|p| p.psnr_db < psnr);
        let idx = idx.clamp(1, by_psnr.len() - 1);

        let p0 = &by_psnr[idx - 1];
        let p1 = &by_psnr[idx];

        Some(log_interp(
            p0.psnr_db,
            p0.bitrate_kbps,
            p1.psnr_db,
            p1.bitrate_kbps,
            psnr,
        ))
    }

    // ------------------------------------------------------------------
    // Interpolation — PSNR at target bitrate
    // ------------------------------------------------------------------

    /// Interpolate the PSNR (dB) achievable at the given bitrate.
    ///
    /// Returns `None` if the target bitrate is outside the range or fewer
    /// than 2 points exist.
    #[must_use]
    pub fn interpolate_psnr_at_bitrate(&self, bitrate: f64) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }

        let first_rate = self.points.first()?.bitrate_kbps;
        let last_rate = self.points.last()?.bitrate_kbps;

        if bitrate < first_rate || bitrate > last_rate {
            return None;
        }

        let idx = self.points.partition_point(|p| p.bitrate_kbps < bitrate);
        let idx = idx.clamp(1, self.points.len() - 1);

        let p0 = &self.points[idx - 1];
        let p1 = &self.points[idx];

        Some(linear_interp(
            p0.bitrate_kbps,
            p0.psnr_db,
            p1.bitrate_kbps,
            p1.psnr_db,
            bitrate,
        ))
    }

    // ------------------------------------------------------------------
    // Curve statistics
    // ------------------------------------------------------------------

    /// Return the bitrate range `(min_kbps, max_kbps)` covered by the curve.
    #[must_use]
    pub fn bitrate_range(&self) -> Option<(f64, f64)> {
        let first = self.points.first()?.bitrate_kbps;
        let last = self.points.last()?.bitrate_kbps;
        Some((first, last))
    }

    /// Return the PSNR range `(min_db, max_db)` covered by the curve.
    #[must_use]
    pub fn psnr_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &self.points {
            if p.psnr_db < lo {
                lo = p.psnr_db;
            }
            if p.psnr_db > hi {
                hi = p.psnr_db;
            }
        }
        Some((lo, hi))
    }
}

impl Default for RdCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------------
// RD optimiser
// ------------------------------------------------------------------

/// Finds optimal QP/CRF values from an [`RdCurve`].
pub struct RdOptimizer;

impl RdOptimizer {
    /// Find the QP/CRF value that minimises the bitrate while meeting the
    /// `target_bitrate` constraint (i.e., the highest QP/CRF whose bitrate is
    /// still ≤ `target_bitrate`).
    ///
    /// Returns `None` when the curve is empty, all points exceed the target
    /// bitrate, or the curve has no point at or below the target.
    ///
    /// QP/CRF values are linearly interpolated between the two bracketing
    /// points to give a more precise estimate.
    #[must_use]
    pub fn find_optimal_qp(curve: &RdCurve, target_bitrate: f64) -> Option<f32> {
        if curve.points.is_empty() {
            return None;
        }

        // Find the highest bitrate point ≤ target.
        let idx_lo = curve
            .points
            .iter()
            .rposition(|p| p.bitrate_kbps <= target_bitrate)?;

        if idx_lo + 1 >= curve.points.len() {
            // Target is at or beyond the highest bitrate point — return its QP.
            return Some(curve.points[idx_lo].qp_or_crf);
        }

        let p_lo = &curve.points[idx_lo];
        let p_hi = &curve.points[idx_lo + 1];

        // Linearly interpolate QP between the two bracketing bitrates.
        let t = if (p_hi.bitrate_kbps - p_lo.bitrate_kbps).abs() < f64::EPSILON {
            0.0_f64
        } else {
            (target_bitrate - p_lo.bitrate_kbps) / (p_hi.bitrate_kbps - p_lo.bitrate_kbps)
        };

        let qp = p_lo.qp_or_crf as f64 + t * (p_hi.qp_or_crf as f64 - p_lo.qp_or_crf as f64);
        Some(qp as f32)
    }

    /// Find the QP/CRF that achieves a target PSNR level.
    ///
    /// Uses [`RdCurve::interpolate_bitrate_at_psnr`] to determine the
    /// required bitrate, then delegates to [`Self::find_optimal_qp`].
    ///
    /// Returns `None` when the target PSNR is out of range.
    #[must_use]
    pub fn find_qp_for_psnr(curve: &RdCurve, target_psnr: f64) -> Option<f32> {
        let bitrate = curve.interpolate_bitrate_at_psnr(target_psnr)?;
        Self::find_optimal_qp(curve, bitrate)
    }

    /// Compute the "quality efficiency" of the curve — PSNR gain per doubling
    /// of bitrate, averaged across all adjacent point pairs.
    ///
    /// Returns `None` for curves with fewer than 2 points.
    #[must_use]
    pub fn average_psnr_per_bitrate_doubling(curve: &RdCurve) -> Option<f64> {
        if curve.points.len() < 2 {
            return None;
        }
        let mut total = 0.0_f64;
        let mut count = 0_usize;
        let pts = &curve.points;
        for i in 1..pts.len() {
            if pts[i - 1].bitrate_kbps <= 0.0 || pts[i].bitrate_kbps <= 0.0 {
                continue;
            }
            let log_ratio = (pts[i].bitrate_kbps / pts[i - 1].bitrate_kbps).log2();
            if log_ratio.abs() < f64::EPSILON {
                continue;
            }
            total += (pts[i].psnr_db - pts[i - 1].psnr_db) / log_ratio;
            count += 1;
        }
        if count == 0 {
            return None;
        }
        Some(total / count as f64)
    }
}

// ------------------------------------------------------------------
// Interpolation helpers
// ------------------------------------------------------------------

/// Linear interpolation of `y` at `x`, given two knots (x0,y0) and (x1,y1).
fn linear_interp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if (x1 - x0).abs() < f64::EPSILON {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Log-domain interpolation: interpolates in `log10(x)` space for `x > 0`.
fn log_interp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if y0 <= 0.0 || y1 <= 0.0 {
        return linear_interp(x0, y0, x1, y1, x);
    }
    let lx0 = x0;
    let lx1 = x1;
    let ly0 = y0.log10();
    let ly1 = y1.log10();
    10_f64.powf(linear_interp(lx0, ly0, lx1, ly1, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple RD curve covering CRF 18–51 for a synthetic codec.
    fn make_av1_like_curve() -> RdCurve {
        // Approximate AV1 behaviour: bitrate ≈ 8000 * exp(−CRF * 0.12)
        let crfs = [18.0_f32, 23.0, 28.0, 33.0, 38.0, 43.0, 51.0];
        let pts = crfs
            .iter()
            .map(|&c| {
                let bitrate = 8000.0 * (-c as f64 * 0.12).exp();
                let psnr = 50.0 - c as f64 * 0.4;
                RdPoint::new(c, bitrate, psnr)
            })
            .collect();
        RdCurve::from_points(pts)
    }

    #[test]
    fn test_rd_point_new() {
        let pt = RdPoint::new(28.0, 1500.0, 38.5);
        assert!((pt.qp_or_crf - 28.0).abs() < 1e-6);
        assert!((pt.bitrate_kbps - 1500.0).abs() < 1e-9);
        assert!((pt.psnr_db - 38.5).abs() < 1e-9);
    }

    #[test]
    fn test_rd_curve_sorted_on_construction() {
        let pts = vec![
            RdPoint::new(51.0, 200.0, 30.0),
            RdPoint::new(18.0, 4000.0, 46.0),
            RdPoint::new(33.0, 900.0, 36.8),
        ];
        let curve = RdCurve::from_points(pts);
        // Should be sorted ascending by bitrate.
        for i in 1..curve.len() {
            assert!(
                curve.points[i].bitrate_kbps >= curve.points[i - 1].bitrate_kbps,
                "curve not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_add_point_maintains_order() {
        let mut curve = RdCurve::new();
        curve.add_point(RdPoint::new(28.0, 1500.0, 38.5));
        curve.add_point(RdPoint::new(18.0, 4000.0, 46.0));
        curve.add_point(RdPoint::new(33.0, 900.0, 36.0));
        assert_eq!(curve.len(), 3);
        assert!(curve.points[0].bitrate_kbps <= curve.points[1].bitrate_kbps);
        assert!(curve.points[1].bitrate_kbps <= curve.points[2].bitrate_kbps);
    }

    #[test]
    fn test_interpolate_psnr_at_known_bitrate() {
        let curve = make_av1_like_curve();
        // At the exact bitrate of the CRF-28 point, PSNR should be exact.
        let pt = curve
            .points
            .iter()
            .find(|p| (p.qp_or_crf - 28.0).abs() < 0.1)
            .expect("pt");
        let psnr = curve
            .interpolate_psnr_at_bitrate(pt.bitrate_kbps)
            .expect("should interpolate");
        assert!(
            (psnr - pt.psnr_db).abs() < 1e-6,
            "PSNR at knot should be exact, got {psnr:.6} expected {:.6}",
            pt.psnr_db
        );
    }

    #[test]
    fn test_interpolate_bitrate_at_known_psnr() {
        let curve = make_av1_like_curve();
        let pt = curve
            .points
            .iter()
            .find(|p| (p.qp_or_crf - 23.0).abs() < 0.1)
            .expect("pt");
        let bitrate = curve
            .interpolate_bitrate_at_psnr(pt.psnr_db)
            .expect("should interpolate");
        assert!(
            (bitrate - pt.bitrate_kbps).abs() / pt.bitrate_kbps < 0.01,
            "bitrate at knot should be near-exact, got {bitrate:.2} expected {:.2}",
            pt.bitrate_kbps
        );
    }

    #[test]
    fn test_interpolate_psnr_out_of_range_returns_none() {
        let curve = make_av1_like_curve();
        assert!(curve.interpolate_psnr_at_bitrate(1.0).is_none());
        assert!(curve.interpolate_psnr_at_bitrate(100_000.0).is_none());
    }

    #[test]
    fn test_interpolate_bitrate_out_of_range_returns_none() {
        let curve = make_av1_like_curve();
        assert!(curve.interpolate_bitrate_at_psnr(10.0).is_none());
        assert!(curve.interpolate_bitrate_at_psnr(60.0).is_none());
    }

    #[test]
    fn test_single_point_curve_returns_none() {
        let mut curve = RdCurve::new();
        curve.add_point(RdPoint::new(30.0, 1000.0, 37.0));
        assert!(curve.interpolate_psnr_at_bitrate(1000.0).is_none());
        assert!(curve.interpolate_bitrate_at_psnr(37.0).is_none());
    }

    #[test]
    fn test_find_optimal_qp() {
        let curve = make_av1_like_curve();
        // Ask for a bitrate that's between two measured points.
        // Curve is sorted ascending by bitrate; higher CRF = lower bitrate, so
        // points[2].qp_or_crf > points[3].qp_or_crf (e.g. 43 > 38).
        let lo_idx = 2;
        let hi_idx = 3;
        let mid_bitrate =
            (curve.points[lo_idx].bitrate_kbps + curve.points[hi_idx].bitrate_kbps) / 2.0;
        let qp = RdOptimizer::find_optimal_qp(&curve, mid_bitrate).expect("should find qp");
        let qp_lo = curve.points[lo_idx]
            .qp_or_crf
            .min(curve.points[hi_idx].qp_or_crf);
        let qp_hi = curve.points[lo_idx]
            .qp_or_crf
            .max(curve.points[hi_idx].qp_or_crf);
        assert!(
            qp >= qp_lo && qp <= qp_hi,
            "QP {qp} should be between {qp_lo} and {qp_hi}"
        );
    }

    #[test]
    fn test_find_optimal_qp_empty_curve_none() {
        let curve = RdCurve::new();
        assert!(RdOptimizer::find_optimal_qp(&curve, 1000.0).is_none());
    }

    #[test]
    fn test_find_optimal_qp_too_low_bitrate_returns_none() {
        let curve = make_av1_like_curve();
        // Request a bitrate lower than the minimum on the curve (CRF 51 ≈ lowest bitrate).
        let min_bitrate = curve.points.first().expect("first").bitrate_kbps;
        // A bitrate below the minimum → None.
        assert!(RdOptimizer::find_optimal_qp(&curve, min_bitrate - 1.0).is_none());
    }

    #[test]
    fn test_bitrate_range() {
        let curve = make_av1_like_curve();
        let (lo, hi) = curve.bitrate_range().expect("range");
        assert!(hi > lo);
    }

    #[test]
    fn test_psnr_range() {
        let curve = make_av1_like_curve();
        let (lo, hi) = curve.psnr_range().expect("range");
        assert!(hi > lo);
    }

    #[test]
    fn test_average_psnr_per_bitrate_doubling() {
        let curve = make_av1_like_curve();
        let eff = RdOptimizer::average_psnr_per_bitrate_doubling(&curve).expect("eff");
        // Typical video codecs: ~3–6 dB per bitrate doubling.
        assert!(eff > 0.0, "efficiency should be positive, got {eff:.4}");
        assert!(
            eff < 20.0,
            "efficiency should be < 20 dB per doubling, got {eff:.4}"
        );
    }
}
