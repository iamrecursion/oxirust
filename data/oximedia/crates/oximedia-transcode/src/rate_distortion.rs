#![allow(dead_code)]
//! Rate-distortion analysis for optimal encoding parameter selection.
//!
//! Models the trade-off between bitrate and quality to help select the best
//! CRF, QP, or bitrate for a given quality target. Provides RD-curve fitting,
//! operating point selection, and Bjontegaard-delta (BD-rate) comparison.

use std::fmt;

/// A single rate-distortion measurement point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdPoint {
    /// Bitrate in kilobits per second.
    pub bitrate_kbps: f64,
    /// Quality metric value (e.g. PSNR in dB, SSIM, VMAF score).
    pub quality: f64,
}

impl RdPoint {
    /// Create a new RD point.
    #[must_use]
    pub fn new(bitrate_kbps: f64, quality: f64) -> Self {
        Self {
            bitrate_kbps,
            quality,
        }
    }
}

impl fmt::Display for RdPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.1} kbps, {:.2})", self.bitrate_kbps, self.quality)
    }
}

/// Quality metric type used in the analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMetric {
    /// Peak Signal-to-Noise Ratio (dB).
    Psnr,
    /// Structural Similarity Index.
    Ssim,
    /// Video Multimethod Assessment Fusion.
    Vmaf,
}

impl fmt::Display for QualityMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Psnr => write!(f, "PSNR"),
            Self::Ssim => write!(f, "SSIM"),
            Self::Vmaf => write!(f, "VMAF"),
        }
    }
}

/// An RD curve consisting of multiple measurement points.
#[derive(Debug, Clone)]
pub struct RdCurve {
    /// Label for this curve (e.g. codec name, preset).
    pub label: String,
    /// Quality metric used.
    pub metric: QualityMetric,
    /// Measurement points sorted by bitrate ascending.
    points: Vec<RdPoint>,
}

impl RdCurve {
    /// Create a new empty RD curve.
    pub fn new(label: impl Into<String>, metric: QualityMetric) -> Self {
        Self {
            label: label.into(),
            metric,
            points: Vec::new(),
        }
    }

    /// Add a measurement point and maintain sorted order.
    pub fn add_point(&mut self, point: RdPoint) {
        self.points.push(point);
        self.points.sort_by(|a, b| {
            a.bitrate_kbps
                .partial_cmp(&b.bitrate_kbps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return the number of points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Return all points as a slice.
    #[must_use]
    pub fn points(&self) -> &[RdPoint] {
        &self.points
    }

    /// Find the point with the highest quality.
    #[must_use]
    pub fn best_quality(&self) -> Option<&RdPoint> {
        self.points.iter().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the point with the lowest bitrate.
    #[must_use]
    pub fn lowest_bitrate(&self) -> Option<&RdPoint> {
        self.points.first()
    }

    /// Find the operating point closest to a target quality.
    #[must_use]
    pub fn find_nearest_quality(&self, target: f64) -> Option<&RdPoint> {
        self.points.iter().min_by(|a, b| {
            let da = (a.quality - target).abs();
            let db = (b.quality - target).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the operating point closest to a target bitrate.
    #[must_use]
    pub fn find_nearest_bitrate(&self, target_kbps: f64) -> Option<&RdPoint> {
        self.points.iter().min_by(|a, b| {
            let da = (a.bitrate_kbps - target_kbps).abs();
            let db = (b.bitrate_kbps - target_kbps).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Linearly interpolate quality at a given bitrate.
    /// Returns `None` if outside the curve range or fewer than 2 points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn interpolate_quality(&self, bitrate_kbps: f64) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }
        let first = self.points.first()?;
        let last = self.points.last()?;
        if bitrate_kbps < first.bitrate_kbps || bitrate_kbps > last.bitrate_kbps {
            return None;
        }
        // Find the two bounding points
        for window in self.points.windows(2) {
            let lo = &window[0];
            let hi = &window[1];
            if bitrate_kbps >= lo.bitrate_kbps && bitrate_kbps <= hi.bitrate_kbps {
                let range = hi.bitrate_kbps - lo.bitrate_kbps;
                if range.abs() < f64::EPSILON {
                    return Some(lo.quality);
                }
                let t = (bitrate_kbps - lo.bitrate_kbps) / range;
                return Some(lo.quality + t * (hi.quality - lo.quality));
            }
        }
        None
    }
}

/// Compute the average quality difference between two RD curves
/// over their overlapping bitrate range (simplified BD-rate style comparison).
/// Positive means `curve_b` has higher quality at the same bitrate.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn average_quality_delta(curve_a: &RdCurve, curve_b: &RdCurve, samples: usize) -> Option<f64> {
    if curve_a.point_count() < 2 || curve_b.point_count() < 2 || samples == 0 {
        return None;
    }

    let a_min = curve_a.points().first()?.bitrate_kbps;
    let a_max = curve_a.points().last()?.bitrate_kbps;
    let b_min = curve_b.points().first()?.bitrate_kbps;
    let b_max = curve_b.points().last()?.bitrate_kbps;

    let lo = a_min.max(b_min);
    let hi = a_max.min(b_max);
    if lo >= hi {
        return None;
    }

    let step = (hi - lo) / samples as f64;
    let mut sum = 0.0;
    let mut count = 0u64;

    let mut br = lo;
    while br <= hi {
        if let (Some(qa), Some(qb)) = (
            curve_a.interpolate_quality(br),
            curve_b.interpolate_quality(br),
        ) {
            sum += qb - qa;
            count += 1;
        }
        br += step;
    }

    if count == 0 {
        return None;
    }
    Some(sum / count as f64)
}

/// Compute the efficiency of a point as quality per kbps.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn efficiency(point: &RdPoint) -> f64 {
    if point.bitrate_kbps.abs() < f64::EPSILON {
        return 0.0;
    }
    point.quality / point.bitrate_kbps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_curve(label: &str) -> RdCurve {
        let mut c = RdCurve::new(label, QualityMetric::Psnr);
        c.add_point(RdPoint::new(500.0, 30.0));
        c.add_point(RdPoint::new(1000.0, 35.0));
        c.add_point(RdPoint::new(2000.0, 38.0));
        c.add_point(RdPoint::new(4000.0, 40.0));
        c
    }

    #[test]
    fn test_rd_point_display() {
        let p = RdPoint::new(1000.0, 35.5);
        assert_eq!(p.to_string(), "(1000.0 kbps, 35.50)");
    }

    #[test]
    fn test_quality_metric_display() {
        assert_eq!(QualityMetric::Psnr.to_string(), "PSNR");
        assert_eq!(QualityMetric::Ssim.to_string(), "SSIM");
        assert_eq!(QualityMetric::Vmaf.to_string(), "VMAF");
    }

    #[test]
    fn test_curve_sorted() {
        let mut c = RdCurve::new("test", QualityMetric::Vmaf);
        c.add_point(RdPoint::new(2000.0, 90.0));
        c.add_point(RdPoint::new(500.0, 70.0));
        c.add_point(RdPoint::new(1000.0, 80.0));
        assert_eq!(c.points()[0].bitrate_kbps as u64, 500);
        assert_eq!(c.points()[1].bitrate_kbps as u64, 1000);
        assert_eq!(c.points()[2].bitrate_kbps as u64, 2000);
    }

    #[test]
    fn test_best_quality() {
        let c = sample_curve("x");
        let best = c.best_quality().expect("should succeed in test");
        assert!((best.quality - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lowest_bitrate() {
        let c = sample_curve("x");
        let low = c.lowest_bitrate().expect("should succeed in test");
        assert!((low.bitrate_kbps - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_nearest_quality() {
        let c = sample_curve("x");
        let p = c
            .find_nearest_quality(36.0)
            .expect("should succeed in test");
        assert!((p.quality - 35.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_nearest_bitrate() {
        let c = sample_curve("x");
        let p = c
            .find_nearest_bitrate(1200.0)
            .expect("should succeed in test");
        assert!((p.bitrate_kbps - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_quality_midpoint() {
        let c = sample_curve("x");
        let q = c
            .interpolate_quality(750.0)
            .expect("should succeed in test");
        // Midpoint between (500, 30) and (1000, 35) => 32.5
        assert!((q - 32.5).abs() < 0.01);
    }

    #[test]
    fn test_interpolate_quality_out_of_range() {
        let c = sample_curve("x");
        assert!(c.interpolate_quality(100.0).is_none());
        assert!(c.interpolate_quality(5000.0).is_none());
    }

    #[test]
    fn test_interpolate_quality_insufficient_points() {
        let mut c = RdCurve::new("x", QualityMetric::Psnr);
        c.add_point(RdPoint::new(1000.0, 35.0));
        assert!(c.interpolate_quality(1000.0).is_none());
    }

    #[test]
    fn test_average_quality_delta_same_curve() {
        let c = sample_curve("x");
        let delta = average_quality_delta(&c, &c, 10).expect("should succeed in test");
        assert!(delta.abs() < 0.01);
    }

    #[test]
    fn test_average_quality_delta_better_curve() {
        let a = sample_curve("a");
        let mut b = RdCurve::new("b", QualityMetric::Psnr);
        b.add_point(RdPoint::new(500.0, 32.0));
        b.add_point(RdPoint::new(1000.0, 37.0));
        b.add_point(RdPoint::new(2000.0, 40.0));
        b.add_point(RdPoint::new(4000.0, 42.0));
        let delta = average_quality_delta(&a, &b, 20).expect("should succeed in test");
        assert!(delta > 0.0, "curve b should be better");
    }

    #[test]
    fn test_average_quality_delta_no_overlap() {
        let mut a = RdCurve::new("a", QualityMetric::Psnr);
        a.add_point(RdPoint::new(100.0, 20.0));
        a.add_point(RdPoint::new(200.0, 25.0));
        let mut b = RdCurve::new("b", QualityMetric::Psnr);
        b.add_point(RdPoint::new(500.0, 30.0));
        b.add_point(RdPoint::new(1000.0, 35.0));
        assert!(average_quality_delta(&a, &b, 10).is_none());
    }

    #[test]
    fn test_efficiency() {
        let p = RdPoint::new(1000.0, 35.0);
        assert!((efficiency(&p) - 0.035).abs() < 0.001);
    }

    #[test]
    fn test_efficiency_zero_bitrate() {
        let p = RdPoint::new(0.0, 35.0);
        assert!((efficiency(&p) - 0.0).abs() < f64::EPSILON);
    }
}
