//! Deep image prior-style iterative denoising (network-free).
//!
//! This module implements a self-supervised iterative denoising approach
//! inspired by the Deep Image Prior (Ulyanov et al. 2018) philosophy:
//! the *structure* of the update rule — not learned weights — encodes a
//! spatial-smoothness prior that suppresses noise while preserving edges.
//!
//! ## Algorithm (per iteration)
//! 1. Extract high-frequency residual: `residual = u - gaussian_blur(u, σ=1)`
//! 2. Estimate noise variance from the residual (Donoho–Johnstone MAD).
//! 3. Apply guided filter (edge-preserving, O(n) via SAT box filter):
//!    `smoothed = guided_filter(u, u, radius=guided_radius, eps=noise_est)`
//! 4. Blend to avoid over-smoothing:
//!    `u = blend_factor * smoothed + (1 - blend_factor) * noisy_frame`

use std::fmt;

/// Configuration for the deep-prior iterative denoiser.
#[derive(Clone, Debug)]
pub struct DeepPriorConfig {
    /// Maximum number of refinement iterations (more → smoother, slower).
    pub max_iters: u32,
    /// Fraction of the guided-filtered estimate to blend per iteration.
    /// 0.9 = strong smoothing convergence; lower preserves more detail.
    pub blend_factor: f32,
    /// Guided-filter neighbourhood radius in pixels (≥ 1).
    pub guided_radius: u32,
}

impl Default for DeepPriorConfig {
    fn default() -> Self {
        Self {
            max_iters: 5,
            blend_factor: 0.9,
            guided_radius: 3,
        }
    }
}

impl fmt::Display for DeepPriorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeepPriorConfig {{ iters={}, blend={:.2}, radius={} }}",
            self.max_iters, self.blend_factor, self.guided_radius
        )
    }
}

/// Error type for deep-prior denoising.
#[derive(Debug, Clone, PartialEq)]
pub enum DeepPriorError {
    /// Width × height doesn't match slice length.
    SizeMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual slice length.
        got: usize,
    },
    /// Zero-size image is not allowed.
    ZeroSize,
}

impl fmt::Display for DeepPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected} pixels, got {got}")
            }
            Self::ZeroSize => write!(f, "image dimensions must be > 0"),
        }
    }
}

impl std::error::Error for DeepPriorError {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply deep-image-prior-style iterative denoising to a grayscale f32 image.
///
/// # Arguments
/// * `frame`  – noisy input pixel values (row-major, any finite range).
/// * `w`      – image width in pixels.
/// * `h`      – image height in pixels.
/// * `cfg`    – algorithm configuration.
///
/// # Returns
/// A new `Vec<f32>` of the same length containing the denoised estimate.
///
/// # Errors
/// Returns [`DeepPriorError`] if dimensions are inconsistent.
pub fn deep_prior_denoise(
    frame: &[f32],
    w: u32,
    h: u32,
    cfg: &DeepPriorConfig,
) -> Result<Vec<f32>, DeepPriorError> {
    let n = w as usize * h as usize;
    if w == 0 || h == 0 {
        return Err(DeepPriorError::ZeroSize);
    }
    if frame.len() != n {
        return Err(DeepPriorError::SizeMismatch {
            expected: n,
            got: frame.len(),
        });
    }
    if cfg.max_iters == 0 {
        return Ok(frame.to_vec());
    }

    let blend = cfg.blend_factor.clamp(0.0, 1.0);
    let inv_blend = 1.0 - blend;
    let radius = cfg.guided_radius.max(1);

    let mut u = frame.to_vec();

    for _ in 0..cfg.max_iters {
        // Step 1: Gaussian blur (σ=1) via separable 5-tap kernel
        let blurred = gaussian_blur_5tap(&u, w, h);

        // Step 2: Residual = high-frequency content
        let residual: Vec<f32> = u.iter().zip(blurred.iter()).map(|(&a, &b)| a - b).collect();

        // Step 3: Estimate noise variance from residual (Donoho–Johnstone)
        let noise_var = estimate_noise_variance(&residual);

        // Step 4: Guided filter (self-guided)
        let eps = noise_var.max(1e-6);
        let smoothed = guided_filter_gray(&u, &u, w, h, radius, eps);

        // Step 5: Blend
        for (ui, si) in u.iter_mut().zip(smoothed.iter()) {
            *ui = blend * si + inv_blend * (*ui);
        }
    }

    Ok(u)
}

/// Guided filter for grayscale images using box-filter (SAT) approximation — O(n).
///
/// Implements He et al. (2013) "Guided Image Filtering" using integral images
/// to compute per-pixel mean and variance in O(1) per pixel.
///
/// For each pixel `k` in a window `ω_k` of radius `r`:
/// - `mean_I` = mean of guide in window
/// - `mean_p` = mean of input in window
/// - `var_I`  = variance of guide in window
/// - `cov_Ip` = covariance of guide and input in window = `E[I*P] - E[I]*E[P]`
/// - `a_k = cov_Ip / (var_I + eps)`
/// - `b_k = mean_p - a_k * mean_I`
/// - Output: `q[i] = mean_a[i] * I[i] + mean_b[i]`
///
/// # Arguments
/// * `input`  – pixel values to filter (row-major, `w × h`).
/// * `guide`  – guidance image (same size); for self-guided filtering, pass
///              the same slice as `input`.
/// * `w`, `h` – image dimensions.
/// * `radius` – neighbourhood radius (`(2r+1)²` box).
/// * `eps`    – regularisation term (prevents division by near-zero variance).
///
/// # Panics
/// Panics if `input.len() != w * h` or `guide.len() != w * h`.
pub fn guided_filter_gray(
    input: &[f32],
    guide: &[f32],
    w: u32,
    h: u32,
    radius: u32,
    eps: f32,
) -> Vec<f32> {
    let ww = w as usize;
    let hh = h as usize;
    let n = ww * hh;
    assert_eq!(input.len(), n, "input size mismatch");
    assert_eq!(guide.len(), n, "guide size mismatch");

    // Build four SATs:
    //   sat_i  = SAT of guide (I)
    //   sat_i2 = SAT of guide² (I²)
    //   sat_p  = SAT of input (P)
    //   sat_ip = SAT of guide * input (I×P)
    let sat_i = build_sat(guide, ww, hh);
    let sat_i2 = build_sat_sq(guide, ww, hh);
    let sat_p = build_sat(input, ww, hh);
    let sat_ip = build_sat_product(guide, input, ww, hh);

    let r = radius as usize;
    let mut a_buf = vec![0.0f32; n];
    let mut b_buf = vec![0.0f32; n];

    for y in 0..hh {
        for x in 0..ww {
            let (y1, y2, x1, x2) = window_coords(x, y, ww, hh, r);
            let cnt = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f32;

            let sum_i = sat_query(&sat_i, x1, y1, x2, y2, ww);
            let sum_i2 = sat_query(&sat_i2, x1, y1, x2, y2, ww);
            let sum_p = sat_query(&sat_p, x1, y1, x2, y2, ww);
            let sum_ip = sat_query(&sat_ip, x1, y1, x2, y2, ww);

            let mean_i = sum_i / cnt;
            let mean_p = sum_p / cnt;
            let mean_i2 = sum_i2 / cnt;
            let mean_ip = sum_ip / cnt;

            // var_I = E[I²] - (E[I])²
            let var_i = (mean_i2 - mean_i * mean_i).max(0.0);
            // cov(I,P) = E[I*P] - E[I]*E[P]
            let cov_ip = mean_ip - mean_i * mean_p;

            let a = cov_ip / (var_i + eps);
            let b = mean_p - a * mean_i;
            a_buf[y * ww + x] = a;
            b_buf[y * ww + x] = b;
        }
    }

    // Output: q[i] = mean_a(i) * I[i] + mean_b(i)
    let sat_a = build_sat(&a_buf, ww, hh);
    let sat_b = build_sat(&b_buf, ww, hh);

    let mut out = vec![0.0f32; n];
    for y in 0..hh {
        for x in 0..ww {
            let (y1, y2, x1, x2) = window_coords(x, y, ww, hh, r);
            let cnt = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f32;
            let mean_a = sat_query(&sat_a, x1, y1, x2, y2, ww) / cnt;
            let mean_b = sat_query(&sat_b, x1, y1, x2, y2, ww) / cnt;
            out[y * ww + x] = mean_a * guide[y * ww + x] + mean_b;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Estimate noise variance from a residual via Donoho–Johnstone MAD estimator.
///
/// σ̂ = median(|residual| ) / 0.6745
fn estimate_noise_variance(residual: &[f32]) -> f32 {
    if residual.is_empty() {
        return 0.0;
    }
    let mut abs_vals: Vec<f32> = residual.iter().map(|v| v.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = abs_vals[abs_vals.len() / 2];
    let sigma = median / 0.6745;
    sigma * sigma // return variance
}

/// 5-tap approximation of Gaussian blur (σ ≈ 1.0), separable.
///
/// Kernel: [1, 4, 6, 4, 1] / 16
fn gaussian_blur_5tap(src: &[f32], w: u32, h: u32) -> Vec<f32> {
    let ww = w as usize;
    let hh = h as usize;
    let n = ww * hh;
    let kernel = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let ksum: f32 = kernel.iter().sum();

    // Horizontal pass into tmp
    let mut tmp = vec![0.0f32; n];
    for y in 0..hh {
        for x in 0..ww {
            let mut val = 0.0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let nx = (x as i64 + ki as i64 - 2).clamp(0, ww as i64 - 1) as usize;
                val += k * src[y * ww + nx];
            }
            tmp[y * ww + x] = val / ksum;
        }
    }

    // Vertical pass into dst
    let mut dst = vec![0.0f32; n];
    for y in 0..hh {
        for x in 0..ww {
            let mut val = 0.0f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let ny = (y as i64 + ki as i64 - 2).clamp(0, hh as i64 - 1) as usize;
                val += k * tmp[ny * ww + x];
            }
            dst[y * ww + x] = val / ksum;
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// SAT (Summed Area Table) utilities
// ---------------------------------------------------------------------------

/// Build a SAT for a f32 image (row-major, width × height).
/// SAT is (height+1) × (width+1) with an extra row/col of zeros as border.
fn build_sat(data: &[f32], w: usize, h: usize) -> Vec<f64> {
    let sw = w + 1;
    let sh = h + 1;
    let mut sat = vec![0.0f64; sw * sh];
    for y in 1..=h {
        for x in 1..=w {
            let val = f64::from(data[(y - 1) * w + (x - 1)]);
            sat[y * sw + x] =
                val + sat[(y - 1) * sw + x] + sat[y * sw + (x - 1)] - sat[(y - 1) * sw + (x - 1)];
        }
    }
    sat
}

/// Build SAT for squared values.
fn build_sat_sq(data: &[f32], w: usize, h: usize) -> Vec<f64> {
    let sw = w + 1;
    let sh = h + 1;
    let mut sat = vec![0.0f64; sw * sh];
    for y in 1..=h {
        for x in 1..=w {
            let v = f64::from(data[(y - 1) * w + (x - 1)]);
            sat[y * sw + x] =
                v * v + sat[(y - 1) * sw + x] + sat[y * sw + (x - 1)] - sat[(y - 1) * sw + (x - 1)];
        }
    }
    sat
}

/// Build SAT for product of two images (a[i] * b[i]).
fn build_sat_product(a: &[f32], b: &[f32], w: usize, h: usize) -> Vec<f64> {
    let sw = w + 1;
    let sh = h + 1;
    let mut sat = vec![0.0f64; sw * sh];
    for y in 1..=h {
        for x in 1..=w {
            let idx = (y - 1) * w + (x - 1);
            let v = f64::from(a[idx]) * f64::from(b[idx]);
            sat[y * sw + x] =
                v + sat[(y - 1) * sw + x] + sat[y * sw + (x - 1)] - sat[(y - 1) * sw + (x - 1)];
        }
    }
    sat
}

/// Query a SAT for the sum over rectangle (x1,y1)..(x2,y2) inclusive.
fn sat_query(sat: &[f64], x1: usize, y1: usize, x2: usize, y2: usize, w: usize) -> f32 {
    let sw = w + 1;
    let sum = sat[(y2 + 1) * sw + (x2 + 1)] - sat[y1 * sw + (x2 + 1)] - sat[(y2 + 1) * sw + x1]
        + sat[y1 * sw + x1];
    sum as f32
}

/// Compute window bounds, clamped to image.
fn window_coords(x: usize, y: usize, w: usize, h: usize, r: usize) -> (usize, usize, usize, usize) {
    let y1 = y.saturating_sub(r);
    let y2 = (y + r).min(h - 1);
    let x1 = x.saturating_sub(r);
    let x2 = (x + r).min(w - 1);
    (y1, y2, x1, x2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn psnr(a: &[f32], b: &[f32]) -> f32 {
        let mse = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            / a.len() as f32;
        if mse < 1e-10 {
            return f32::INFINITY;
        }
        10.0 * (255.0f32 * 255.0 / mse).log10()
    }

    /// Pseudo-random noise for reproducible tests.
    fn rng_noise(seed: u64, n: usize, sigma: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // map to normal-ish via Box-Muller with two uniform values
                let u1 = (state & 0xFFFF) as f32 / 65535.0;
                let u2 = ((state >> 16) & 0xFFFF) as f32 / 65535.0;
                let normal =
                    ((-2.0 * (u1 + 1e-9).ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).cos();
                normal * sigma
            })
            .collect()
    }

    #[test]
    fn test_deep_prior_psnr_improvement() {
        // Synthetic: solid 128 + Gaussian noise σ=20
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let clean: Vec<f32> = vec![128.0f32; n];
        let noise = rng_noise(42, n, 20.0);
        let noisy: Vec<f32> = clean
            .iter()
            .zip(noise.iter())
            .map(|(&c, &n)| (c + n).clamp(0.0, 255.0))
            .collect();

        let psnr_noisy = psnr(&clean, &noisy);

        let cfg = DeepPriorConfig::default();
        let denoised =
            deep_prior_denoise(&noisy, w, h, &cfg).expect("deep_prior_denoise should succeed");

        let psnr_denoised = psnr(&clean, &denoised);

        assert!(
            psnr_denoised >= psnr_noisy + 2.0,
            "Expected PSNR improvement ≥ 2 dB: noisy={psnr_noisy:.2}, denoised={psnr_denoised:.2}"
        );
    }

    #[test]
    fn test_deep_prior_identity_on_clean() {
        // Zero noise: output should be very close to input
        let w = 32u32;
        let h = 32u32;
        let n = (w * h) as usize;
        // Smooth gradient (no noise to remove)
        let clean: Vec<f32> = (0..n).map(|i| 50.0 + (i % 200) as f32 * 0.5).collect();

        let cfg = DeepPriorConfig {
            max_iters: 3,
            ..Default::default()
        };
        let out = deep_prior_denoise(&clean, w, h, &cfg).expect("should succeed");

        // Max pixel difference should be small (< 20 LSB for smooth content)
        let max_diff = clean
            .iter()
            .zip(out.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 25.0,
            "Clean image changed too much: max_diff={max_diff:.2}"
        );
    }

    #[test]
    fn test_deep_prior_zero_iters() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let cfg = DeepPriorConfig {
            max_iters: 0,
            ..Default::default()
        };
        let out = deep_prior_denoise(&data, 2, 2, &cfg).expect("should succeed");
        assert_eq!(out, data);
    }

    #[test]
    fn test_deep_prior_size_mismatch() {
        let data = vec![1.0f32; 10];
        let cfg = DeepPriorConfig::default();
        let result = deep_prior_denoise(&data, 4, 4, &cfg);
        assert!(matches!(result, Err(DeepPriorError::SizeMismatch { .. })));
    }

    #[test]
    fn test_deep_prior_zero_size() {
        let cfg = DeepPriorConfig::default();
        let result = deep_prior_denoise(&[], 0, 0, &cfg);
        assert!(matches!(result, Err(DeepPriorError::ZeroSize)));
    }

    #[test]
    fn test_guided_filter_uniform() {
        // Uniform image → output should remain uniform
        let n = 16 * 16;
        let data = vec![100.0f32; n];
        let out = guided_filter_gray(&data, &data, 16, 16, 3, 0.01);
        for &v in &out {
            assert!((v - 100.0).abs() < 1.0, "Uniform image drifted: {v}");
        }
    }

    #[test]
    fn test_gaussian_blur_5tap_uniform() {
        let data = vec![128.0f32; 32 * 32];
        let blurred = gaussian_blur_5tap(&data, 32, 32);
        for &v in &blurred {
            assert!(
                (v - 128.0).abs() < 0.01,
                "Blur of uniform should stay uniform: {v}"
            );
        }
    }

    #[test]
    fn test_estimate_noise_variance_zero() {
        let zeros = vec![0.0f32; 64];
        let var = estimate_noise_variance(&zeros);
        assert!((var - 0.0).abs() < 1e-6, "Zero residual → zero variance");
    }

    #[test]
    fn test_estimate_noise_variance_nonzero() {
        let residual: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0) - 0.5).collect();
        let var = estimate_noise_variance(&residual);
        assert!(var > 0.0, "Non-zero residual should give positive variance");
    }

    #[test]
    fn test_deep_prior_config_display() {
        let cfg = DeepPriorConfig::default();
        let s = format!("{cfg}");
        assert!(s.contains("iters="));
        assert!(s.contains("blend="));
        assert!(s.contains("radius="));
    }
}
