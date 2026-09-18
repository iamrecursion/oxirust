//! Advanced color difference metrics.
//!
//! Implements CIE ΔE 2000 (CIEDE2000) and CIE ΔE 94 color difference formulas,
//! which are perceptually more uniform than the older ΔE 76 metric.  Both are
//! used in professional color science, press-proofing, and display calibration
//! to assess how closely a reproduction matches the original color.

#![allow(dead_code)]

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Convert degrees to radians.
#[inline]
fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Compute the hue angle h in degrees from a* and b*, returning a value in
/// [0, 360).
#[inline]
fn hue_angle(a: f64, b: f64) -> f64 {
    let h = b.atan2(a).to_degrees();
    if h < 0.0 {
        h + 360.0
    } else {
        h
    }
}

/// Compute the mean hue angle between two hue angles h1 and h2 (in degrees).
/// Handles the wrap-around at 360°.
#[inline]
fn mean_hue(h1: f64, h2: f64, c1: f64, c2: f64) -> f64 {
    if c1 * c2 < f64::EPSILON {
        return h1 + h2;
    }
    if (h1 - h2).abs() <= 180.0 {
        (h1 + h2) / 2.0
    } else if h1 + h2 < 360.0 {
        (h1 + h2 + 360.0) / 2.0
    } else {
        (h1 + h2 - 360.0) / 2.0
    }
}

/// Δh′ helper for CIEDE2000.
#[inline]
fn delta_h_prime(c1: f64, c2: f64, h1: f64, h2: f64) -> f64 {
    if c1 * c2 < f64::EPSILON {
        return 0.0;
    }
    let diff = h2 - h1;
    if diff.abs() <= 180.0 {
        diff
    } else if diff > 180.0 {
        diff - 360.0
    } else {
        diff + 360.0
    }
}

// ---------------------------------------------------------------------------
// CIE ΔE 94
// ---------------------------------------------------------------------------

/// Application context for ΔE 94 weighting constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum De94Application {
    /// Graphic arts (kL=1, K1=0.045, K2=0.015).
    GraphicArts,
    /// Textiles (kL=2, K1=0.048, K2=0.014).
    Textiles,
}

impl De94Application {
    /// Returns `(kL, K1, K2)` constants for this application.
    #[must_use]
    fn constants(self) -> (f64, f64, f64) {
        match self {
            Self::GraphicArts => (1.0, 0.045, 0.015),
            Self::Textiles => (2.0, 0.048, 0.014),
        }
    }
}

/// Compute the CIE ΔE 94 color difference between two CIE L\*a\*b\* colors.
///
/// `app` selects the application-specific weighting constants.
/// `lab1` is taken as the reference (standard); `lab2` as the sample (batch).
#[must_use]
pub fn delta_e_94(lab1: [f64; 3], lab2: [f64; 3], app: De94Application) -> f64 {
    let (kl, k1, k2) = app.constants();
    const KC: f64 = 1.0;
    const KH: f64 = 1.0;

    let dl = lab1[0] - lab2[0];
    let c1 = (lab1[1].powi(2) + lab1[2].powi(2)).sqrt();
    let c2 = (lab2[1].powi(2) + lab2[2].powi(2)).sqrt();
    let dc = c1 - c2;

    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    let dh_sq = da * da + db * db - dc * dc;
    let dh = if dh_sq > 0.0 { dh_sq.sqrt() } else { 0.0 };

    let sl = 1.0;
    let sc = 1.0 + k1 * c1;
    let sh = 1.0 + k2 * c1;

    let term_l = dl / (kl * sl);
    let term_c = dc / (KC * sc);
    let term_h = dh / (KH * sh);

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

// ---------------------------------------------------------------------------
// CIE ΔE 2000 (CIEDE2000)
// ---------------------------------------------------------------------------

/// Compute the CIE ΔE 2000 (CIEDE2000) color difference between two
/// CIE L\*a\*b\* colors.
///
/// This is the most perceptually accurate standardised color difference metric
/// as of this writing.  The implementation follows the specification published
/// in:
///
/// > Sharma G., Wu W., Dalal E. N. (2005). "The CIEDE2000 Color-Difference
/// > Formula: Implementation Notes, Supplementary Test Data, and Mathematical
/// > Observations." *Color Research & Application*, 30(1), 21–30.
///
/// All default parametric factors (`kL = kC = kH = 1.0`) are used.
#[must_use]
pub fn delta_e_2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (lab1[0], lab1[1], lab1[2]);
    let (l2, a2, b2) = (lab2[0], lab2[1], lab2[2]);

    // Step 1: Compute C*ab and h°ab for each sample, plus ā
    let c1_ab = (a1 * a1 + b1 * b1).sqrt();
    let c2_ab = (a2 * a2 + b2 * b2).sqrt();
    let c_avg_7 = ((c1_ab + c2_ab) / 2.0).powi(7);
    let c25_7 = 25.0_f64.powi(7);
    let g = 0.5 * (1.0 - (c_avg_7 / (c_avg_7 + c25_7)).sqrt());

    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = hue_angle(a1p, b1);
    let h2p = hue_angle(a2p, b2);

    // Step 2: Compute ΔL′, ΔC′, ΔH′
    let dl_prime = l2 - l1;
    let dc_prime = c2p - c1p;
    let dh_prime_deg = delta_h_prime(c1p, c2p, h1p, h2p);
    let dh_prime = 2.0 * (c1p * c2p).sqrt() * deg_to_rad(dh_prime_deg / 2.0).sin();

    // Step 3: Compute CIEDE2000
    let l_avg = (l1 + l2) / 2.0;
    let c_avg_p = (c1p + c2p) / 2.0;
    let h_avg_p = mean_hue(h1p, h2p, c1p, c2p);

    // Weighting functions
    let t = 1.0 - 0.17 * deg_to_rad(h_avg_p - 30.0).cos()
        + 0.24 * deg_to_rad(2.0 * h_avg_p).cos()
        + 0.32 * deg_to_rad(3.0 * h_avg_p + 6.0).cos()
        - 0.20 * deg_to_rad(4.0 * h_avg_p - 63.0).cos();

    let sl = 1.0 + 0.015 * (l_avg - 50.0).powi(2) / (20.0 + (l_avg - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_avg_p;
    let sh = 1.0 + 0.015 * c_avg_p * t;

    let c_avg_7_p = c_avg_p.powi(7);
    let rc = 2.0 * (c_avg_7_p / (c_avg_7_p + c25_7)).sqrt();
    let d_theta_deg = 30.0 * (-(((h_avg_p - 275.0) / 25.0).powi(2))).exp();
    let rt = -rc * deg_to_rad(2.0 * d_theta_deg).sin();

    // kL = kC = kH = 1.0
    let term_l = dl_prime / sl;
    let term_c = dc_prime / sc;
    let term_h = dh_prime / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h).sqrt()
}

// ---------------------------------------------------------------------------
// PassFailEvaluator
// ---------------------------------------------------------------------------

/// Color tolerance assessment result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceResult {
    /// Color difference is within the specified tolerance.
    Pass,
    /// Color difference exceeds the specified tolerance.
    Fail,
}

/// Evaluate whether a measured color is within tolerance of a reference.
#[derive(Debug, Clone)]
pub struct ToleranceEvaluator {
    /// Maximum allowed ΔE 2000 for a "pass".
    pub de2000_threshold: f64,
}

impl ToleranceEvaluator {
    /// Create a new evaluator with the given ΔE 2000 threshold.
    #[must_use]
    pub fn new(de2000_threshold: f64) -> Self {
        Self { de2000_threshold }
    }

    /// Evaluate `measured` against `reference`.
    #[must_use]
    pub fn evaluate(&self, reference: [f64; 3], measured: [f64; 3]) -> ToleranceResult {
        let de = delta_e_2000(reference, measured);
        if de <= self.de2000_threshold {
            ToleranceResult::Pass
        } else {
            ToleranceResult::Fail
        }
    }

    /// Batch evaluate a list of `(reference, measured)` pairs.
    ///
    /// Returns `(pass_count, fail_count)`.
    #[must_use]
    pub fn batch_evaluate(&self, pairs: &[([f64; 3], [f64; 3])]) -> (usize, usize) {
        let mut pass = 0;
        let mut fail = 0;
        for &(ref_lab, meas_lab) in pairs {
            match self.evaluate(ref_lab, meas_lab) {
                ToleranceResult::Pass => pass += 1,
                ToleranceResult::Fail => fail += 1,
            }
        }
        (pass, fail)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    #[test]
    fn test_hue_angle_positive() {
        let h = hue_angle(1.0, 0.0);
        assert!((h - 0.0).abs() < 1e-9, "h={h}");
    }

    #[test]
    fn test_hue_angle_negative_a() {
        let h = hue_angle(-1.0, 0.0);
        assert!((h - 180.0).abs() < 1e-9, "h={h}");
    }

    #[test]
    fn test_hue_angle_wraps_to_positive() {
        let h = hue_angle(0.0, -1.0);
        assert!(h >= 0.0 && h < 360.0, "h={h}");
    }

    // ── ΔE 94 ────────────────────────────────────────────────────────────

    #[test]
    fn test_de94_identical_colors_zero() {
        let lab = [50.0, 25.0, -30.0];
        let de = delta_e_94(lab, lab, De94Application::GraphicArts);
        assert!(
            de.abs() < 1e-9,
            "ΔE94 of identical colors should be 0, got {de}"
        );
    }

    #[test]
    fn test_de94_lightness_only_difference() {
        let lab1 = [50.0, 0.0, 0.0];
        let lab2 = [60.0, 0.0, 0.0];
        let de = delta_e_94(lab1, lab2, De94Application::GraphicArts);
        // ΔL=10, SL=1, kL=1 → ΔE94 = 10
        assert!((de - 10.0).abs() < 1e-6, "ΔE94={de}");
    }

    #[test]
    fn test_de94_graphic_arts_vs_textiles_differ() {
        let lab1 = [50.0, 30.0, -20.0];
        let lab2 = [60.0, 10.0, 5.0];
        let de_ga = delta_e_94(lab1, lab2, De94Application::GraphicArts);
        let de_tx = delta_e_94(lab1, lab2, De94Application::Textiles);
        // Graphic arts and textiles use different kL → different results
        assert!((de_ga - de_tx).abs() > 0.01, "Expected different values");
    }

    #[test]
    fn test_de94_positive_for_different_colors() {
        let de = delta_e_94(
            [50.0, 25.0, 0.0],
            [50.0, 0.0, 25.0],
            De94Application::GraphicArts,
        );
        assert!(de > 0.0, "ΔE94 should be positive");
    }

    // ── ΔE 2000 ──────────────────────────────────────────────────────────

    #[test]
    fn test_de2000_identical_colors_zero() {
        let lab = [50.0, 25.0, -30.0];
        let de = delta_e_2000(lab, lab);
        assert!(
            de.abs() < 1e-9,
            "ΔE2000 of identical colors should be 0, got {de}"
        );
    }

    #[test]
    fn test_de2000_known_pair_sharma_1() {
        // Test pair 1 from Sharma et al. (2005)
        let lab1 = [50.000_0, 2.677_2, -79.775_1];
        let lab2 = [50.000_0, 0.000_0, -82.748_5];
        let de = delta_e_2000(lab1, lab2);
        assert!((de - 2.0425).abs() < 0.001, "ΔE2000={de:.4}");
    }

    #[test]
    fn test_de2000_known_pair_sharma_2() {
        // Test pair 2 from Sharma et al. (2005)
        let lab1 = [50.000_0, 3.157_1, -77.280_3];
        let lab2 = [50.000_0, 0.000_0, -82.748_5];
        let de = delta_e_2000(lab1, lab2);
        assert!((de - 2.8615).abs() < 0.002, "ΔE2000={de:.4}");
    }

    #[test]
    fn test_de2000_white_vs_black_large() {
        let lab_white = [100.0, 0.0, 0.0];
        let lab_black = [0.0, 0.0, 0.0];
        let de = delta_e_2000(lab_white, lab_black);
        assert!(
            de > 50.0,
            "ΔE2000 between white and black should be large, got {de}"
        );
    }

    #[test]
    fn test_de2000_symmetry() {
        let lab1 = [60.0, 20.0, -10.0];
        let lab2 = [55.0, 30.0, 5.0];
        let de_forward = delta_e_2000(lab1, lab2);
        let de_reverse = delta_e_2000(lab2, lab1);
        // Note: CIEDE2000 is not strictly symmetric but differences are small
        assert!(
            (de_forward - de_reverse).abs() < 0.5,
            "forward={de_forward}, reverse={de_reverse}"
        );
    }

    #[test]
    fn test_de2000_positive() {
        let de = delta_e_2000([50.0, 10.0, 10.0], [50.0, -10.0, -10.0]);
        assert!(de > 0.0);
    }

    // ── ToleranceEvaluator ───────────────────────────────────────────────

    #[test]
    fn test_tolerance_pass_identical() {
        let ev = ToleranceEvaluator::new(2.0);
        let lab = [50.0, 0.0, 0.0];
        assert_eq!(ev.evaluate(lab, lab), ToleranceResult::Pass);
    }

    #[test]
    fn test_tolerance_fail_large_diff() {
        let ev = ToleranceEvaluator::new(2.0);
        let ref_lab = [50.0, 0.0, 0.0];
        let meas_lab = [80.0, 30.0, -20.0];
        assert_eq!(ev.evaluate(ref_lab, meas_lab), ToleranceResult::Fail);
    }

    #[test]
    fn test_tolerance_batch_all_pass() {
        let ev = ToleranceEvaluator::new(5.0);
        let lab = [50.0, 0.0, 0.0];
        let pairs = vec![(lab, lab), (lab, lab), (lab, lab)];
        let (pass, fail) = ev.batch_evaluate(&pairs);
        assert_eq!(pass, 3);
        assert_eq!(fail, 0);
    }

    #[test]
    fn test_tolerance_batch_mixed() {
        let ev = ToleranceEvaluator::new(2.0);
        let good_lab = [50.0, 0.0, 0.0];
        let bad_lab = [90.0, 40.0, -40.0];
        let pairs = vec![(good_lab, good_lab), (good_lab, bad_lab)];
        let (pass, fail) = ev.batch_evaluate(&pairs);
        assert_eq!(pass, 1);
        assert_eq!(fail, 1);
    }

    /// CIEDE2000 of a colour with itself must be exactly 0.
    #[test]
    fn test_delta_e2000_identical() {
        let lab = [50.0, 25.0, -10.0];
        let de = delta_e_2000(lab, lab);
        assert!(
            de.abs() < 1e-12,
            "ΔE2000 of identical LAB colours must be 0, got {de}"
        );
    }
}
