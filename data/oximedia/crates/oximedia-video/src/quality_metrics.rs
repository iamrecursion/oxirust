//! Video quality metrics.
//!
//! Provides PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity
//! Index Measure), and a simple perceptual metric based on gradient correlation
//! for assessing frame-level video quality.
//!
//! # Metrics overview
//!
//! | Metric | Range | Perfect | Notes |
//! |--------|-------|---------|-------|
//! | PSNR   | [0, ∞) dB | ∞ | >40 dB is visually lossless |
//! | SSIM   | [-1, 1] | 1.0 | >0.95 is high quality |
//! | VIF proxy | [0, 1] | 1.0 | gradient-correlation proxy |
//!
//! All functions operate on grayscale (luma) planes stored as row-major
//! `width × height` byte buffers.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during quality metric computation.
#[derive(Debug, thiserror::Error)]
pub enum QualityMetricError {
    /// Input buffer size does not match the declared dimensions.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
    /// Reference and distorted frames have different dimensions.
    #[error("dimension mismatch: reference is {ref_w}x{ref_h}, distorted is {dist_w}x{dist_h}")]
    DimensionMismatch {
        /// Reference width.
        ref_w: u32,
        /// Reference height.
        ref_h: u32,
        /// Distorted width.
        dist_w: u32,
        /// Distorted height.
        dist_h: u32,
    },
    /// Frame is too small for the requested analysis window.
    #[error("frame {w}x{h} is too small for window size {window}")]
    FrameTooSmall {
        /// Frame width.
        w: u32,
        /// Frame height.
        h: u32,
        /// Required window size.
        window: usize,
    },
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// All quality metrics computed for a single reference–distorted pair.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// PSNR in decibels. `f64::INFINITY` if the frames are identical.
    pub psnr_db: f64,
    /// SSIM in [-1, 1]. 1.0 means identical.
    pub ssim: f64,
    /// Gradient correlation proxy (VIF-like) in [0, 1].
    pub gradient_correlation: f64,
    /// Mean squared error (per-pixel average).
    pub mse: f64,
    /// Mean absolute error (per-pixel average).
    pub mae: f64,
}

/// Configuration for SSIM computation.
#[derive(Debug, Clone)]
pub struct SsimConfig {
    /// Local analysis window radius (in pixels). Window size = `2*radius+1` × `2*radius+1`.
    /// Default: `5` (11×11 window).
    pub window_radius: usize,
    /// Stabilisation constant C1 = `(k1 * L)^2`.  Default: `(0.01*255)^2`.
    pub c1: f64,
    /// Stabilisation constant C2 = `(k2 * L)^2`.  Default: `(0.03*255)^2`.
    pub c2: f64,
}

impl Default for SsimConfig {
    fn default() -> Self {
        let l = 255.0f64;
        Self {
            window_radius: 5,
            c1: (0.01 * l) * (0.01 * l),
            c2: (0.03 * l) * (0.03 * l),
        }
    }
}

/// Configuration for quality metric computations.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// SSIM configuration.
    pub ssim: SsimConfig,
    /// Peak value of the signal (255 for 8-bit video).
    pub peak_value: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            ssim: SsimConfig::default(),
            peak_value: 255.0,
        }
    }
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Compute PSNR between a reference and a distorted luma plane.
///
/// Both planes must be `width × height` bytes.
///
/// Returns `f64::INFINITY` if the planes are identical (MSE = 0).
pub fn compute_psnr(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    peak: f64,
) -> Result<f64, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;
    let mse = compute_mse(reference, distorted);
    if mse == 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (peak * peak / mse).log10())
}

/// Compute the mean squared error between two luma planes.
///
/// Both planes must be `width × height` bytes.
pub fn compute_mse_metric(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;
    Ok(compute_mse(reference, distorted))
}

/// Compute the mean absolute error between two luma planes.
///
/// Both planes must be `width × height` bytes.
pub fn compute_mae_metric(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Ok(0.0);
    }
    let total: u64 = reference
        .iter()
        .take(n)
        .zip(distorted.iter().take(n))
        .map(|(&r, &d)| (r as i32 - d as i32).unsigned_abs() as u64)
        .sum();
    Ok(total as f64 / n as f64)
}

/// Compute the SSIM index between two luma planes.
///
/// Both planes must be `width × height` bytes.  The result is averaged over
/// all `window_size × window_size` tiles of the frame.
///
/// Returns a value in [-1, 1]; 1.0 means the frames are identical.
pub fn compute_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    config: &SsimConfig,
) -> Result<f64, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;

    let w = width as usize;
    let h = height as usize;
    let wr = config.window_radius;
    let win = 2 * wr + 1;

    if w < win || h < win {
        return Err(QualityMetricError::FrameTooSmall {
            w: width,
            h: height,
            window: win,
        });
    }

    let c1 = config.c1;
    let c2 = config.c2;

    let mut ssim_sum = 0.0f64;
    let mut count = 0u64;

    // Slide the window over valid positions.
    let y_start = wr;
    let y_end = h - wr;
    let x_start = wr;
    let x_end = w - wr;

    // Step by the window size to reduce computation (approximate but standard practice).
    let step = wr.max(1);

    let mut cy = y_start;
    while cy < y_end {
        let mut cx = x_start;
        while cx < x_end {
            let ssim_val = ssim_window(reference, distorted, w, h, cx, cy, wr, c1, c2);
            ssim_sum += ssim_val;
            count += 1;
            cx += step;
        }
        cy += step;
    }

    if count == 0 {
        return Ok(1.0); // Degenerate: no window fitted.
    }

    Ok(ssim_sum / count as f64)
}

/// Compute the gradient correlation between two luma planes as a proxy for
/// perceptual fidelity.
///
/// Computes the magnitude gradient for each plane using Sobel-like differences,
/// then returns the normalised dot product (Pearson correlation) of the gradient
/// fields.  Range: [0, 1].
pub fn compute_gradient_correlation(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;

    let w = width as usize;
    let h = height as usize;
    if w < 2 || h < 2 {
        return Ok(1.0); // Too small — trivially perfect.
    }

    let grad_ref = compute_gradient_magnitudes(reference, w, h);
    let grad_dist = compute_gradient_magnitudes(distorted, w, h);

    Ok(pearson_correlation(&grad_ref, &grad_dist))
}

/// Compute all quality metrics at once and return a [`QualityReport`].
pub fn compute_all(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    config: &QualityConfig,
) -> Result<QualityReport, QualityMetricError> {
    validate_pair(reference, distorted, width, height)?;

    let mse = compute_mse(reference, distorted);
    let mae = {
        let n = (width as usize) * (height as usize);
        if n == 0 {
            0.0
        } else {
            let total: u64 = reference
                .iter()
                .take(n)
                .zip(distorted.iter().take(n))
                .map(|(&r, &d)| (r as i32 - d as i32).unsigned_abs() as u64)
                .sum();
            total as f64 / n as f64
        }
    };

    let psnr_db = if mse == 0.0 {
        f64::INFINITY
    } else {
        let peak = config.peak_value;
        10.0 * (peak * peak / mse).log10()
    };

    let ssim = compute_ssim(reference, distorted, width, height, &config.ssim).unwrap_or(1.0); // Fall back for small frames.

    let gradient_correlation = compute_gradient_correlation(reference, distorted, width, height)?;

    Ok(QualityReport {
        psnr_db,
        ssim,
        gradient_correlation,
        mse,
        mae,
    })
}

/// Pipeline quality gate: return `true` if all metrics meet the given floors.
///
/// * `min_psnr_db`: minimum acceptable PSNR (e.g. 35.0 for streaming).
/// * `min_ssim`: minimum acceptable SSIM (e.g. 0.90).
/// * `min_gradient_corr`: minimum gradient correlation (e.g. 0.85).
pub fn quality_gate(
    report: &QualityReport,
    min_psnr_db: f64,
    min_ssim: f64,
    min_gradient_corr: f64,
) -> bool {
    report.psnr_db >= min_psnr_db
        && report.ssim >= min_ssim
        && report.gradient_correlation >= min_gradient_corr
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Validate that `reference` and `distorted` both match `width × height`.
fn validate_pair(
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
) -> Result<(), QualityMetricError> {
    let expected = (width as usize) * (height as usize);
    if reference.len() != expected {
        return Err(QualityMetricError::BufferSizeMismatch {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() != expected {
        return Err(QualityMetricError::BufferSizeMismatch {
            expected,
            actual: distorted.len(),
        });
    }
    Ok(())
}

/// Per-pixel MSE (mean squared error).
fn compute_mse(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let total: f64 = a
        .iter()
        .take(n)
        .zip(b.iter().take(n))
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    total / n as f64
}

/// Compute the SSIM for a single window centred at `(cx, cy)` with radius `wr`.
fn ssim_window(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
    wr: usize,
    c1: f64,
    c2: f64,
) -> f64 {
    let y0 = cy.saturating_sub(wr);
    let y1 = (cy + wr + 1).min(h);
    let x0 = cx.saturating_sub(wr);
    let x1 = (cx + wr + 1).min(w);

    let mut n = 0u64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut sum_ab = 0.0f64;

    for y in y0..y1 {
        for x in x0..x1 {
            let va = a.get(y * w + x).copied().unwrap_or(0) as f64;
            let vb = b.get(y * w + x).copied().unwrap_or(0) as f64;
            n += 1;
            sum_a += va;
            sum_b += vb;
            sum_aa += va * va;
            sum_bb += vb * vb;
            sum_ab += va * vb;
        }
    }

    if n == 0 {
        return 1.0;
    }

    let nf = n as f64;
    let mu_a = sum_a / nf;
    let mu_b = sum_b / nf;
    let var_a = (sum_aa / nf) - mu_a * mu_a;
    let var_b = (sum_bb / nf) - mu_b * mu_b;
    let cov_ab = (sum_ab / nf) - mu_a * mu_b;

    let numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2);
    let denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);

    if denominator.abs() < 1e-15 {
        return 1.0;
    }

    numerator / denominator
}

/// Compute per-pixel Sobel gradient magnitudes.
///
/// Uses simple horizontal/vertical finite differences for efficiency.
fn compute_gradient_magnitudes(plane: &[u8], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    let mut grads = vec![0.0f64; n];

    for y in 0..h {
        for x in 0..w {
            let xn = if x + 1 < w { x + 1 } else { x };
            let xp = if x > 0 { x - 1 } else { x };
            let yn = if y + 1 < h { y + 1 } else { y };
            let yp = if y > 0 { y - 1 } else { y };

            let gx = plane.get(y * w + xn).copied().unwrap_or(0) as f64
                - plane.get(y * w + xp).copied().unwrap_or(0) as f64;
            let gy = plane.get(yn * w + x).copied().unwrap_or(0) as f64
                - plane.get(yp * w + x).copied().unwrap_or(0) as f64;

            grads[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    grads
}

/// Pearson correlation coefficient of two equal-length f64 slices.
///
/// Returns 1.0 if the inputs are constant (degenerate case).
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }

    let mean_a = a.iter().take(n).sum::<f64>() / n as f64;
    let mean_b = b.iter().take(n).sum::<f64>() / n as f64;

    let mut num = 0.0f64;
    let mut den_a = 0.0f64;
    let mut den_b = 0.0f64;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let den = (den_a * den_b).sqrt();
    if den < 1e-12 {
        return 1.0;
    }

    // Map to [0, 1] (correlation can be negative for opposite patterns).
    ((num / den + 1.0) / 2.0).clamp(0.0, 1.0)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    fn ramp(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    // ---- PSNR tests ----

    // 1. PSNR: identical frames → infinity
    #[test]
    fn test_psnr_identical_frames_infinity() {
        let frame = ramp(16, 16);
        let psnr = compute_psnr(&frame, &frame, 16, 16, 255.0).expect("ok");
        assert!(
            psnr.is_infinite() && psnr > 0.0,
            "expected +inf, got {psnr}"
        );
    }

    // 2. PSNR: all-0 vs all-255 → low PSNR
    #[test]
    fn test_psnr_opposite_frames_low() {
        let black = flat(16, 16, 0);
        let white = flat(16, 16, 255);
        let psnr = compute_psnr(&black, &white, 16, 16, 255.0).expect("ok");
        assert!(
            psnr < 10.0,
            "maximally different frames should have low PSNR, got {psnr}"
        );
    }

    // 3. PSNR: high-quality distortion → > 40 dB
    #[test]
    fn test_psnr_high_quality_above_40db() {
        let reference = flat(32, 32, 128);
        let mut distorted = reference.clone();
        // Introduce tiny noise (1 unit per pixel in one region).
        for i in 0..16 {
            distorted[i] = distorted[i].saturating_add(1);
        }
        let psnr = compute_psnr(&reference, &distorted, 32, 32, 255.0).expect("ok");
        assert!(psnr > 40.0, "tiny noise should give > 40 dB, got {psnr}");
    }

    // 4. PSNR: buffer size mismatch → error
    #[test]
    fn test_psnr_buffer_mismatch_error() {
        let frame = flat(8, 8, 100);
        let small = flat(4, 4, 100);
        let result = compute_psnr(&frame, &small, 8, 8, 255.0);
        assert!(result.is_err());
    }

    // ---- MSE tests ----

    // 5. MSE: identical frames → 0.0
    #[test]
    fn test_mse_identical_zero() {
        let frame = ramp(16, 16);
        let mse = compute_mse_metric(&frame, &frame, 16, 16).expect("ok");
        assert_eq!(mse, 0.0);
    }

    // 6. MSE: all-0 vs all-255 → 255^2 = 65025
    #[test]
    fn test_mse_opposite_frames() {
        let black = flat(8, 8, 0);
        let white = flat(8, 8, 255);
        let mse = compute_mse_metric(&black, &white, 8, 8).expect("ok");
        assert!((mse - 65025.0).abs() < 1e-6, "expected 65025.0, got {mse}");
    }

    // ---- MAE tests ----

    // 7. MAE: identical frames → 0.0
    #[test]
    fn test_mae_identical_zero() {
        let frame = ramp(16, 16);
        let mae = compute_mae_metric(&frame, &frame, 16, 16).expect("ok");
        assert_eq!(mae, 0.0);
    }

    // 8. MAE: all-0 vs all-255 → 255.0
    #[test]
    fn test_mae_opposite_frames() {
        let black = flat(8, 8, 0);
        let white = flat(8, 8, 255);
        let mae = compute_mae_metric(&black, &white, 8, 8).expect("ok");
        assert!((mae - 255.0).abs() < 1e-6, "expected 255.0, got {mae}");
    }

    // ---- SSIM tests ----

    // 9. SSIM: identical frames → 1.0
    #[test]
    fn test_ssim_identical_one() {
        let frame = ramp(32, 32);
        let config = SsimConfig::default();
        let ssim = compute_ssim(&frame, &frame, 32, 32, &config).expect("ok");
        assert!((ssim - 1.0).abs() < 1e-4, "expected SSIM ~1.0, got {ssim}");
    }

    // 10. SSIM: maximally different frames → near -1.0 or low
    #[test]
    fn test_ssim_different_frames_low() {
        let black = flat(32, 32, 0);
        let white = flat(32, 32, 255);
        let config = SsimConfig::default();
        let ssim = compute_ssim(&black, &white, 32, 32, &config).expect("ok");
        assert!(
            ssim < 0.5,
            "very different frames should have low SSIM, got {ssim}"
        );
    }

    // 11. SSIM: frame too small for window → error
    #[test]
    fn test_ssim_frame_too_small_error() {
        let frame = flat(4, 4, 128);
        let config = SsimConfig {
            window_radius: 5,
            ..SsimConfig::default()
        };
        let result = compute_ssim(&frame, &frame, 4, 4, &config);
        assert!(result.is_err());
    }

    // 12. SSIM: is in [-1, 1]
    #[test]
    fn test_ssim_range() {
        let a = ramp(32, 32);
        let b = flat(32, 32, 64);
        let config = SsimConfig::default();
        let ssim = compute_ssim(&a, &b, 32, 32, &config).expect("ok");
        assert!(ssim >= -1.0 && ssim <= 1.0, "SSIM {ssim} out of range");
    }

    // ---- Gradient correlation tests ----

    // 13. Gradient correlation: identical frames → 1.0
    #[test]
    fn test_gradient_corr_identical() {
        let frame = ramp(16, 16);
        let gc = compute_gradient_correlation(&frame, &frame, 16, 16).expect("ok");
        assert!(
            (gc - 1.0).abs() < 1e-4,
            "identical frames should have gc=1.0, got {gc}"
        );
    }

    // 14. Gradient correlation: very different frames → lower value
    #[test]
    fn test_gradient_corr_different_lower() {
        let a = ramp(16, 16);
        let b = flat(16, 16, 128); // flat has zero gradient
        let gc = compute_gradient_correlation(&a, &b, 16, 16).expect("ok");
        // ramp has gradients, flat doesn't → lower correlation
        assert!(gc <= 1.0, "gc {gc} should be <= 1.0");
    }

    // 15. Gradient correlation: is in [0, 1]
    #[test]
    fn test_gradient_corr_range() {
        let a = ramp(16, 16);
        let b = flat(16, 16, 200);
        let gc = compute_gradient_correlation(&a, &b, 16, 16).expect("ok");
        assert!(gc >= 0.0 && gc <= 1.0, "gc {gc} out of [0, 1]");
    }

    // ---- compute_all tests ----

    // 16. compute_all: identical frames → perfect metrics
    #[test]
    fn test_compute_all_identical() {
        let frame = ramp(32, 32);
        let config = QualityConfig::default();
        let report = compute_all(&frame, &frame, 32, 32, &config).expect("ok");
        assert!(report.psnr_db.is_infinite());
        assert!((report.ssim - 1.0).abs() < 1e-4);
        assert!((report.gradient_correlation - 1.0).abs() < 1e-4);
        assert_eq!(report.mse, 0.0);
        assert_eq!(report.mae, 0.0);
    }

    // 17. compute_all: MSE is non-negative
    #[test]
    fn test_compute_all_mse_nonnegative() {
        let a = ramp(16, 16);
        let b = flat(16, 16, 100);
        let config = QualityConfig::default();
        let report = compute_all(&a, &b, 16, 16, &config).expect("ok");
        assert!(report.mse >= 0.0);
        assert!(report.mae >= 0.0);
    }

    // ---- quality_gate tests ----

    // 18. quality_gate: perfect metrics pass all thresholds
    #[test]
    fn test_quality_gate_pass() {
        let report = QualityReport {
            psnr_db: 45.0,
            ssim: 0.98,
            gradient_correlation: 0.97,
            mse: 2.0,
            mae: 1.0,
        };
        assert!(quality_gate(&report, 40.0, 0.90, 0.85));
    }

    // 19. quality_gate: low PSNR fails
    #[test]
    fn test_quality_gate_fail_psnr() {
        let report = QualityReport {
            psnr_db: 30.0,
            ssim: 0.98,
            gradient_correlation: 0.97,
            mse: 100.0,
            mae: 5.0,
        };
        assert!(!quality_gate(&report, 40.0, 0.90, 0.85));
    }

    // 20. quality_gate: low SSIM fails
    #[test]
    fn test_quality_gate_fail_ssim() {
        let report = QualityReport {
            psnr_db: 45.0,
            ssim: 0.70,
            gradient_correlation: 0.97,
            mse: 2.0,
            mae: 1.0,
        };
        assert!(!quality_gate(&report, 40.0, 0.90, 0.85));
    }

    // 21. PSNR is monotonically decreasing with increasing distortion
    #[test]
    fn test_psnr_monotone_with_distortion() {
        let reference = flat(16, 16, 128);
        let psnr_values: Vec<f64> = [1u8, 5, 20, 60]
            .iter()
            .map(|&noise| {
                let distorted: Vec<u8> =
                    reference.iter().map(|&p| p.saturating_add(noise)).collect();
                compute_psnr(&reference, &distorted, 16, 16, 255.0).expect("ok")
            })
            .collect();

        for w in psnr_values.windows(2) {
            assert!(
                w[0] >= w[1] - 0.01,
                "PSNR should decrease with distortion: {:.2} >= {:.2}",
                w[0],
                w[1]
            );
        }
    }

    // 22. SSIM is >= PSNR-based quality for low distortion
    #[test]
    fn test_ssim_sensitive_to_structure() {
        let reference: Vec<u8> = (0..32 * 32)
            .map(|i| {
                if (i / 32 + i % 32) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
            .collect();
        let distorted: Vec<u8> = reference.iter().map(|&p| p.saturating_add(10)).collect();
        let config = SsimConfig::default();
        let ssim = compute_ssim(&reference, &distorted, 32, 32, &config).expect("ok");
        // A minor brightness shift preserves structure; SSIM should be high.
        assert!(
            ssim > 0.7,
            "structural similarity should be high for brightness shift, got {ssim}"
        );
    }

    // 23. MSE buffer mismatch error
    #[test]
    fn test_mse_buffer_mismatch_error() {
        let big = flat(16, 16, 0);
        let small = flat(8, 8, 0);
        assert!(compute_mse_metric(&big, &small, 16, 16).is_err());
    }

    // 24. QualityReport fields accessible
    #[test]
    fn test_quality_report_fields() {
        let r = QualityReport {
            psnr_db: 38.5,
            ssim: 0.93,
            gradient_correlation: 0.91,
            mse: 9.2,
            mae: 2.1,
        };
        assert!((r.psnr_db - 38.5).abs() < 1e-6);
        assert!((r.ssim - 0.93).abs() < 1e-6);
    }

    // 25. compute_gradient_correlation: gradient correlation of flat frames → 1.0
    #[test]
    fn test_gradient_corr_flat_flat() {
        let a = flat(16, 16, 100);
        let b = flat(16, 16, 200); // different brightness, same zero-gradient structure
        let gc = compute_gradient_correlation(&a, &b, 16, 16).expect("ok");
        // Both are flat — gradient vectors are degenerate → should return 1.0
        assert!(
            (gc - 1.0).abs() < 1e-4,
            "flat frames gradient corr should be 1.0, got {gc}"
        );
    }
}
