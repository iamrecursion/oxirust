//! Feature Similarity Index (FSIM) calculation.
//!
//! FSIM is based on the fact that human visual system (HVS) understands
//! an image mainly according to its low-level features. It uses phase
//! congruency (PC) and gradient magnitude (GM) as features.
//!
//! The implementation follows the original paper with a multi-scale
//! log-Gabor filter bank for phase congruency estimation, and Scharr
//! operators for gradient magnitude.
//!
//! # Algorithm
//!
//! ```text
//! FSIM = Σ(FSIM_c(x) · PC_m(x)) / Σ PC_m(x)
//!
//! where:
//!   PC_m(x) = max(PC₁(x), PC₂(x))            — master weight map
//!   FSIM_c(x) = GM_c(x)^α · PC_c(x)^β        — α = β = 1
//!   GM_c(x)   = (2·gm₁·gm₂ + T₂) / (gm₁² + gm₂² + T₂)
//!   PC_c(x)   = (2·pc₁·pc₂ + T₁) / (pc₁² + pc₂² + T₁)
//! ```
//!
//! # Reference
//!
//! L. Zhang, L. Zhang, X. Mou, D. Zhang, "FSIM: A Feature Similarity
//! Index for Image Quality Assessment," IEEE TIP, 2011.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

// ── Log-Gabor filter parameters ──────────────────────────────────────────────

/// Number of scales in the log-Gabor filter bank.
const LOG_GABOR_SCALES: usize = 4;
/// Number of orientations in the log-Gabor filter bank.
const LOG_GABOR_ORIENTATIONS: usize = 4;
/// Centre frequency of the smallest scale filter.
const LOG_GABOR_MIN_WAVELENGTH: f64 = 6.0;
/// Scaling factor between successive filters.
const LOG_GABOR_MULT: f64 = 2.0;
/// Ratio of the standard deviation of the Gaussian describing the log-Gabor
/// filter's transfer function in the frequency domain to the filter centre
/// frequency.
const LOG_GABOR_SIGMA_ON_F: f64 = 0.55;
/// Standard deviation of the angular Gaussian spread function (radians).
const LOG_GABOR_ANGULAR_SIGMA: f64 = 0.5;
/// Noise compensation floor for phase congruency energy.
const LOG_GABOR_NOISE_FLOOR: f64 = 0.001;

// ─────────────────────────────────────────────────────────────────────────────

/// Compute FSIM between two frames using phase congruency and gradient magnitude.
///
/// Returns a value in \[0, 1\] where 1 means identical frames.
///
/// # Errors
///
/// Returns an error if the frame dimensions differ.
pub fn compute_fsim(reference: &Frame, distorted: &Frame) -> OxiResult<f32> {
    let calc = FsimCalculator::new();
    let score = calc.calculate(reference, distorted)?;
    Ok(score.score as f32)
}

// ─────────────────────────────────────────────────────────────────────────────

/// FSIM calculator for video quality assessment.
pub struct FsimCalculator {
    /// Stability constant for phase congruency similarity (T₁ in the paper).
    t1: f64,
    /// Stability constant for gradient magnitude similarity (T₂ in the paper).
    t2: f64,
    /// Exponent α applied to gradient magnitude component.
    alpha: f64,
    /// Exponent β applied to phase congruency component.
    beta: f64,
}

impl FsimCalculator {
    /// Creates a new FSIM calculator with default parameters from the paper.
    ///
    /// T₁ = 0.85, T₂ = 160, α = β = 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            t1: 0.85,
            t2: 160.0,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Creates a new FSIM calculator with custom parameters.
    #[must_use]
    pub fn with_params(t1: f64, t2: f64, alpha: f64, beta: f64) -> Self {
        Self {
            t1,
            t2,
            alpha,
            beta,
        }
    }

    /// Calculates FSIM between reference and distorted frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::Fsim, 0.0);

        // Calculate FSIM for Y plane
        let y_fsim = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
        )?;

        score.add_component("Y", y_fsim);
        score.score = y_fsim;

        Ok(score)
    }

    /// Calculates FSIM for a single plane.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<f64> {
        // Convert to f64 in range [0, 255]
        let ref_f64 = plane_to_f64(ref_plane);
        let dist_f64 = plane_to_f64(dist_plane);

        // Compute gradient magnitudes (Scharr operator)
        let (ref_gx, ref_gy) = scharr_gradient_f64(&ref_f64, width, height);
        let (dist_gx, dist_gy) = scharr_gradient_f64(&dist_f64, width, height);
        let gm_ref = gradient_magnitude_f64(&ref_gx, &ref_gy);
        let gm_dist = gradient_magnitude_f64(&dist_gx, &dist_gy);

        // Compute phase congruency maps via log-Gabor filter bank approximation
        let pc_ref = phase_congruency_approx_f64(&ref_f64, width, height);
        let pc_dist = phase_congruency_approx_f64(&dist_f64, width, height);

        // Compute per-pixel similarity maps and weighted sum
        let n = ref_f64.len();
        let mut sum_fsim = 0.0_f64;
        let mut sum_weight = 0.0_f64;

        for i in 0..n {
            // Gradient magnitude similarity
            let gm1 = gm_ref[i];
            let gm2 = gm_dist[i];
            let gm_c = (2.0 * gm1 * gm2 + self.t2) / (gm1 * gm1 + gm2 * gm2 + self.t2);

            // Phase congruency similarity
            let pc1 = pc_ref[i];
            let pc2 = pc_dist[i];
            let pc_c = (2.0 * pc1 * pc2 + self.t1) / (pc1 * pc1 + pc2 * pc2 + self.t1);

            // Feature similarity with exponents
            let fsim_c = gm_c.powf(self.alpha) * pc_c.powf(self.beta);

            // Master weight = max phase congruency
            let weight = pc1.max(pc2);

            sum_fsim += fsim_c * weight;
            sum_weight += weight;
        }

        let fsim = if sum_weight > 1e-10 {
            sum_fsim / sum_weight
        } else {
            // Uniform image: gradients and PC are both near zero, treat as identical
            1.0
        };

        Ok(fsim.clamp(0.0, 1.0))
    }

    /// Computes gradient magnitude map (Scharr operator).
    ///
    /// Exposed for testing and standalone use.
    pub fn compute_gradient_magnitude(
        &self,
        plane: &[f64],
        width: usize,
        height: usize,
    ) -> Vec<f64> {
        let (gx, gy) = scharr_gradient_f64(plane, width, height);
        gradient_magnitude_f64(&gx, &gy)
    }

    /// Computes phase congruency map using log-Gabor approximation.
    ///
    /// Exposed for testing and standalone use.
    pub fn compute_phase_congruency(&self, plane: &[f64], width: usize, height: usize) -> Vec<f64> {
        phase_congruency_approx_f64(plane, width, height)
    }
}

impl Default for FsimCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Public free-function helpers ──────────────────────────────────────────────

/// Compute the Scharr gradient components (Gx, Gy) of a luma plane.
///
/// The Scharr operator has better rotational symmetry than Sobel and is the
/// gradient kernel recommended by the FSIM paper.
///
/// Input values are expected in the range \[0, 255\].
/// Returns `(gx, gy)` each of length `width × height`.
#[must_use]
pub fn scharr_gradient(frame: &Frame) -> (Vec<f32>, Vec<f32>) {
    let plane = frame.luma();
    let width = frame.width;
    let height = frame.height;
    let f64_plane: Vec<f64> = plane.iter().map(|&v| f64::from(v)).collect();
    let (gx64, gy64) = scharr_gradient_f64(&f64_plane, width, height);
    let gx = gx64.into_iter().map(|v| v as f32).collect();
    let gy = gy64.into_iter().map(|v| v as f32).collect();
    (gx, gy)
}

/// Compute gradient magnitude from Gx and Gy component vectors.
///
/// Returns `sqrt(gx² + gy²)` element-wise.
#[must_use]
pub fn gradient_magnitude(gx: &[f32], gy: &[f32]) -> Vec<f32> {
    gx.iter()
        .zip(gy.iter())
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect()
}

/// Compute an approximation of phase congruency for a frame using a
/// multi-scale, multi-orientation log-Gabor filter bank.
///
/// The log-Gabor energy is accumulated across all scales and orientations,
/// then normalised to \[0, 1\].
#[must_use]
pub fn phase_congruency_approx(frame: &Frame) -> Vec<f32> {
    let plane = frame.luma();
    let width = frame.width;
    let height = frame.height;
    let f64_plane: Vec<f64> = plane.iter().map(|&v| f64::from(v)).collect();
    let pc64 = phase_congruency_approx_f64(&f64_plane, width, height);
    pc64.into_iter().map(|v| v as f32).collect()
}

// ── Internal f64 helpers ──────────────────────────────────────────────────────

fn plane_to_f64(plane: &[u8]) -> Vec<f64> {
    plane.iter().map(|&v| f64::from(v)).collect()
}

/// Scharr gradient on an f64 plane.  Returns (gx, gy).
fn scharr_gradient_f64(plane: &[f64], width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    // Scharr kernels
    //   Gx = [ -3  0  3 ]    Gy = [ -3 -10 -3 ]
    //        [-10  0 10 ]         [  0   0  0 ]
    //        [ -3  0  3 ]         [  3  10  3 ]
    let scharr_x: [f64; 9] = [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0];
    let scharr_y: [f64; 9] = [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0];

    let n = plane.len();
    let mut gx = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];

    if width < 3 || height < 3 {
        return (gx, gy);
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut cx = 0.0_f64;
            let mut cy = 0.0_f64;
            for dy in 0..3usize {
                for dx in 0..3usize {
                    let idx = (y + dy - 1) * width + (x + dx - 1);
                    let k = dy * 3 + dx;
                    cx += plane[idx] * scharr_x[k];
                    cy += plane[idx] * scharr_y[k];
                }
            }
            gx[y * width + x] = cx;
            gy[y * width + x] = cy;
        }
    }

    (gx, gy)
}

/// Gradient magnitude from f64 component vectors.
fn gradient_magnitude_f64(gx: &[f64], gy: &[f64]) -> Vec<f64> {
    gx.iter()
        .zip(gy.iter())
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect()
}

/// Phase congruency approximation using a spatial log-Gabor filter bank.
///
/// # Implementation notes
///
/// A true frequency-domain log-Gabor implementation requires an FFT.  Since
/// this crate is intentionally FFT-free for the default feature set, we use a
/// spatial approximation: for each scale *s* and orientation *o* we construct
/// a pair of quadrature steerable filters (even/odd symmetric) whose shape
/// approximates the log-Gabor envelope at that scale.  The energy at each
/// pixel is then the sum of squared even and odd responses, and phase
/// congruency is the normalised sum of energies across scales, weighted by an
/// orientation selectivity term.
///
/// This matches the behaviour described in Section III-A of the FSIM paper
/// closely enough to reproduce the key property: smooth regions get low PC,
/// edges and textures get high PC.
fn phase_congruency_approx_f64(plane: &[f64], width: usize, height: usize) -> Vec<f64> {
    let n = plane.len();
    let mut total_energy = vec![0.0_f64; n];
    let mut total_weight = vec![0.0_f64; n];

    for scale in 0..LOG_GABOR_SCALES {
        // Wavelength at this scale
        let wavelength = LOG_GABOR_MIN_WAVELENGTH * LOG_GABOR_MULT.powi(scale as i32);
        let sigma_f = LOG_GABOR_SIGMA_ON_F * wavelength;
        // Spatial kernel half-width: 3σ of the Gaussian envelope
        let half = (3.0 * sigma_f).ceil() as usize;
        let half = half.max(1);

        for orient in 0..LOG_GABOR_ORIENTATIONS {
            let angle = std::f64::consts::PI * orient as f64 / LOG_GABOR_ORIENTATIONS as f64;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Build separable 1-D spatial log-Gabor even/odd kernels
            // We approximate the log-Gabor in the spatial domain by a
            // Gabor wavelet: envelope = Gaussian(r/sigma_f), carrier = cos/sin(2πr/λ)
            let kernel_len = 2 * half + 1;
            let mut even_kernel = vec![0.0_f64; kernel_len * kernel_len];
            let mut odd_kernel = vec![0.0_f64; kernel_len * kernel_len];

            let mut sum_sq = 0.0_f64;
            for ky in 0..kernel_len {
                for kx in 0..kernel_len {
                    let dy = ky as f64 - half as f64;
                    let dx = kx as f64 - half as f64;
                    // Project onto the orientation axis
                    let proj = dx * cos_a + dy * sin_a;
                    let r = (dx * dx + dy * dy).sqrt();
                    // Gaussian envelope
                    let env = if r == 0.0 {
                        1.0
                    } else {
                        // log-Gabor: exp( -(log(r/σ_f))² / (2*log(k)²) ) where k ~ sigma_on_f
                        let log_ratio = if r / sigma_f > 1e-10 {
                            (r / sigma_f).ln()
                        } else {
                            -10.0
                        };
                        (-log_ratio * log_ratio
                            / (2.0 * LOG_GABOR_SIGMA_ON_F * LOG_GABOR_SIGMA_ON_F))
                            .exp()
                    };
                    // Angular selectivity: Gaussian in orientation
                    let ang_diff = (dx * (-sin_a) + dy * cos_a).atan2(proj.abs() + 1e-10);
                    let ang_weight = (-ang_diff * ang_diff
                        / (2.0 * LOG_GABOR_ANGULAR_SIGMA * LOG_GABOR_ANGULAR_SIGMA))
                        .exp();

                    let phase = 2.0 * std::f64::consts::PI * proj / wavelength;
                    let k_idx = ky * kernel_len + kx;
                    even_kernel[k_idx] = env * ang_weight * phase.cos();
                    odd_kernel[k_idx] = env * ang_weight * phase.sin();
                    sum_sq += (env * ang_weight).powi(2);
                }
            }

            // Normalise kernels by RMS to make response scale-independent
            let rms = sum_sq.sqrt().max(1e-12);
            for v in &mut even_kernel {
                *v /= rms;
            }
            for v in &mut odd_kernel {
                *v /= rms;
            }

            // Convolve (full 2-D, clamped border)
            if width >= kernel_len && height >= kernel_len {
                for y in 0..height {
                    for x in 0..width {
                        let mut even_resp = 0.0_f64;
                        let mut odd_resp = 0.0_f64;
                        for ky in 0..kernel_len {
                            let sy = y as isize + ky as isize - half as isize;
                            let sy = sy.clamp(0, height as isize - 1) as usize;
                            for kx in 0..kernel_len {
                                let sx = x as isize + kx as isize - half as isize;
                                let sx = sx.clamp(0, width as isize - 1) as usize;
                                let px = plane[sy * width + sx];
                                let k_idx = ky * kernel_len + kx;
                                even_resp += px * even_kernel[k_idx];
                                odd_resp += px * odd_kernel[k_idx];
                            }
                        }
                        let energy = even_resp * even_resp + odd_resp * odd_resp;
                        total_energy[y * width + x] += energy.sqrt();
                        total_weight[y * width + x] += 1.0;
                    }
                }
            }
        }
    }

    // Find the maximum energy across all pixels.
    let max_e = total_energy.iter().copied().fold(0.0_f64, f64::max);

    if max_e < 1e-12 {
        // Entirely flat image — no edges anywhere, all PC = 0
        return vec![0.0; n];
    }

    // Compute the noise floor as the energy of a flat (DC) image.
    // For log-Gabor filters the even and odd kernels are constructed to be
    // bandpass: the sum of each kernel is approximately zero for even and
    // exactly zero for odd (by antisymmetry).  The residual DC energy from
    // floating-point accumulation over the kernel grid is proportional to
    // the square of the kernel rounding error.  We estimate this noise floor
    // as the *median* energy across all pixels: for images with distinct
    // edges the median lies well within the flat background.
    let mut sorted_e = total_energy.clone();
    sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = sorted_e[n / 2]; // median energy

    // Threshold: subtract the noise floor and re-normalise
    let norm = (max_e - noise_floor).max(1e-12);

    (0..n)
        .map(|i| {
            let e = (total_energy[i] - noise_floor).max(0.0);
            (e / norm).clamp(0.0, 1.0)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn create_uniform_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    fn create_gradient_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for y in 0..height {
            for x in 0..width {
                frame.planes[0][y * width + x] = (x * 255 / width) as u8;
            }
        }
        frame
    }

    fn create_checkerboard_frame(width: usize, height: usize, block: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for y in 0..height {
            for x in 0..width {
                let val = if ((x / block) + (y / block)) % 2 == 0 {
                    255u8
                } else {
                    0u8
                };
                frame.planes[0][y * width + x] = val;
            }
        }
        frame
    }

    fn create_noise_frame(width: usize, height: usize, seed: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for (i, v) in frame.planes[0].iter_mut().enumerate() {
            // Deterministic pseudo-noise
            // lcg constants: a=1664525, c=1013904223 (Numerical Recipes)
            *v = ((i as u32)
                .wrapping_mul(1664525)
                .wrapping_add(seed as u32)
                .wrapping_add(1013904223)
                >> 24) as u8;
        }
        frame
    }

    // ── FsimCalculator tests ─────────────────────────────────────────────────

    #[test]
    fn test_fsim_identical_frames_near_one() {
        let calc = FsimCalculator::new();
        let frame = create_gradient_frame(32, 32);
        let result = calc.calculate(&frame, &frame).expect("should succeed");
        // Identical frames must yield ~1.0
        assert!(
            result.score >= 0.99,
            "identical frames FSIM={:.4}, expected >= 0.99",
            result.score
        );
    }

    #[test]
    fn test_fsim_uniform_identical_near_one() {
        let calc = FsimCalculator::new();
        let frame = create_uniform_frame(32, 32, 128);
        let result = calc.calculate(&frame, &frame).expect("should succeed");
        // Uniform → PC ≈ 0 → fallback 1.0
        assert!(
            result.score >= 0.99,
            "uniform identical FSIM={:.4}",
            result.score
        );
    }

    #[test]
    fn test_fsim_range_valid() {
        let calc = FsimCalculator::new();
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_uniform_frame(32, 32, 128);
        let result = calc.calculate(&frame1, &frame2).expect("should succeed");
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "FSIM out of range: {}",
            result.score
        );
    }

    #[test]
    fn test_fsim_different_content_less_than_identical() {
        let calc = FsimCalculator::new();
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_checkerboard_frame(32, 32, 4);
        let result_diff = calc.calculate(&frame1, &frame2).expect("should succeed");
        let result_same = calc.calculate(&frame1, &frame1).expect("should succeed");
        assert!(
            result_diff.score <= result_same.score,
            "different content FSIM ({:.4}) should be <= identical ({:.4})",
            result_diff.score,
            result_same.score
        );
    }

    #[test]
    fn test_fsim_dimension_mismatch_error() {
        let calc = FsimCalculator::new();
        let frame1 = create_uniform_frame(32, 32, 100);
        let frame2 = create_uniform_frame(64, 64, 100);
        assert!(calc.calculate(&frame1, &frame2).is_err());
    }

    #[test]
    fn test_fsim_components_contain_y() {
        let calc = FsimCalculator::new();
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_uniform_frame(32, 32, 128);
        let result = calc.calculate(&frame1, &frame2).expect("should succeed");
        assert!(result.components.contains_key("Y"), "missing Y component");
    }

    #[test]
    fn test_fsim_custom_params() {
        // Using tighter constants should still return valid range
        let calc = FsimCalculator::with_params(0.5, 100.0, 1.0, 1.0);
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_uniform_frame(32, 32, 128);
        let result = calc.calculate(&frame1, &frame2).expect("should succeed");
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    // ── compute_fsim free-function ────────────────────────────────────────────

    #[test]
    fn test_compute_fsim_free_fn_identical() {
        let frame = create_gradient_frame(32, 32);
        let score = compute_fsim(&frame, &frame).expect("should succeed");
        assert!(score >= 0.99f32, "compute_fsim identical={score:.4}");
    }

    #[test]
    fn test_compute_fsim_free_fn_range() {
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_noise_frame(32, 32, 42);
        let score = compute_fsim(&frame1, &frame2).expect("should succeed");
        assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
    }

    // ── scharr_gradient ───────────────────────────────────────────────────────

    #[test]
    fn test_scharr_gradient_uniform_is_zero() {
        let frame = create_uniform_frame(16, 16, 128);
        let (gx, gy) = scharr_gradient(&frame);
        // Interior pixels of a uniform image must have zero gradient
        for y in 1..15usize {
            for x in 1..15usize {
                let i = y * 16 + x;
                assert!(gx[i].abs() < 1e-6, "gx non-zero at ({x},{y}): {}", gx[i]);
                assert!(gy[i].abs() < 1e-6, "gy non-zero at ({x},{y}): {}", gy[i]);
            }
        }
    }

    #[test]
    fn test_scharr_gradient_edge_response() {
        let frame = create_gradient_frame(16, 16);
        let (gx, _gy) = scharr_gradient(&frame);
        // Horizontal gradient image → strong Gx response in interior
        let max_gx = gx.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_gx > 0.0, "expected non-zero Gx on gradient frame");
    }

    #[test]
    fn test_scharr_gradient_length() {
        let frame = create_uniform_frame(20, 15, 100);
        let (gx, gy) = scharr_gradient(&frame);
        assert_eq!(gx.len(), 20 * 15);
        assert_eq!(gy.len(), 20 * 15);
    }

    // ── gradient_magnitude ───────────────────────────────────────────────────

    #[test]
    fn test_gradient_magnitude_pythagorean() {
        let gx = vec![3.0f32, 0.0, 4.0];
        let gy = vec![4.0f32, 5.0, 3.0];
        let gm = gradient_magnitude(&gx, &gy);
        assert!((gm[0] - 5.0).abs() < 1e-5, "expected 5.0, got {}", gm[0]);
        assert!((gm[1] - 5.0).abs() < 1e-5, "expected 5.0, got {}", gm[1]);
        assert!((gm[2] - 5.0).abs() < 1e-5, "expected 5.0, got {}", gm[2]);
    }

    #[test]
    fn test_gradient_magnitude_zero_inputs() {
        let gx = vec![0.0f32; 10];
        let gy = vec![0.0f32; 10];
        let gm = gradient_magnitude(&gx, &gy);
        assert!(gm.iter().all(|&v| v == 0.0));
    }

    // ── phase_congruency_approx ──────────────────────────────────────────────

    #[test]
    fn test_phase_congruency_uniform_low() {
        let frame = create_uniform_frame(32, 32, 128);
        let pc = phase_congruency_approx(&frame);
        assert_eq!(pc.len(), 32 * 32);
        // Uniform image → very low PC everywhere
        let max_pc = pc.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_pc < 0.5,
            "uniform frame PC too high: {max_pc:.4} (expected < 0.5)"
        );
    }

    #[test]
    fn test_phase_congruency_edge_higher_than_uniform() {
        let uniform = create_uniform_frame(32, 32, 128);
        let edges = create_checkerboard_frame(32, 32, 4);
        let pc_uniform = phase_congruency_approx(&uniform);
        let pc_edges = phase_congruency_approx(&edges);
        let mean_uniform: f32 = pc_uniform.iter().sum::<f32>() / pc_uniform.len() as f32;
        let mean_edges: f32 = pc_edges.iter().sum::<f32>() / pc_edges.len() as f32;
        assert!(
            mean_edges > mean_uniform,
            "edge frame PC ({mean_edges:.4}) should exceed uniform ({mean_uniform:.4})"
        );
    }

    #[test]
    fn test_phase_congruency_output_length() {
        let frame = create_gradient_frame(24, 18);
        let pc = phase_congruency_approx(&frame);
        assert_eq!(pc.len(), 24 * 18);
    }

    // ── compute_gradient_magnitude / compute_phase_congruency via calc ────────

    #[test]
    fn test_calc_gradient_magnitude() {
        let calc = FsimCalculator::new();
        let plane = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0,
            200.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let gm = calc.compute_gradient_magnitude(&plane, 5, 4);
        assert_eq!(gm.len(), 20);
        assert!(gm.iter().any(|&v| v > 0.0), "expected non-zero gradient");
    }

    #[test]
    fn test_calc_phase_congruency_length() {
        let calc = FsimCalculator::new();
        let plane = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0,
            200.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let pc = calc.compute_phase_congruency(&plane, 5, 4);
        assert_eq!(pc.len(), 20);
    }
}
