//! Non-local means (NLM) denoising.
//!
//! NLM is a patch-based algorithm: each output pixel is a weighted average of
//! all pixels whose surrounding patch is similar to the patch around the target
//! pixel.  The similarity weight decays exponentially with the patch distance.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Non-Local Means denoiser.
#[derive(Debug, Clone)]
pub struct NlmConfig {
    /// Half-size of the comparison patch (full patch = 2*patch_radius+1 square).
    pub patch_radius: usize,
    /// Half-size of the search window (full window = 2*search_radius+1 square).
    pub search_radius: usize,
    /// Filtering parameter: controls the decay of the similarity weights.
    /// Higher values → more smoothing; lower values → sharper but noisier.
    pub h: f32,
}

impl NlmConfig {
    /// Default balanced configuration.
    pub fn default() -> Self {
        Self {
            patch_radius: 3,
            search_radius: 10,
            h: 10.0,
        }
    }

    /// Strong denoising (larger patch, wider search).
    pub fn strong() -> Self {
        Self {
            patch_radius: 5,
            search_radius: 15,
            h: 20.0,
        }
    }

    /// Mild denoising (small patch, narrow search).
    pub fn mild() -> Self {
        Self {
            patch_radius: 2,
            search_radius: 7,
            h: 5.0,
        }
    }

    /// Patch size in one dimension.
    pub fn patch_size(&self) -> usize {
        2 * self.patch_radius + 1
    }

    /// Search window size in one dimension.
    pub fn search_size(&self) -> usize {
        2 * self.search_radius + 1
    }
}

// ---------------------------------------------------------------------------
// Patch distance and weight
// ---------------------------------------------------------------------------

/// Compute the **sum of squared differences** (SSD) between two equal-length
/// patches.
///
/// The patch slices must have the same length; if they differ the function
/// returns 0.
pub fn patch_ssd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Compute the NLM similarity weight from a patch SSD.
///
/// w = exp(−SSD / h²)
pub fn nlm_weight(ssd: f32, h: f32) -> f32 {
    let h2 = h * h;
    if h2 < f32::EPSILON {
        return 0.0;
    }
    (-ssd / h2).exp()
}

// ---------------------------------------------------------------------------
// Single-pixel NLM
// ---------------------------------------------------------------------------

/// Extract a (2r+1)×(2r+1) patch centred at (cx, cy) into `buf`.
///
/// Out-of-bounds pixels are clamped to the nearest in-bounds value.
fn extract_patch(
    src: &[f32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
    buf: &mut Vec<f32>,
) {
    buf.clear();
    let cx = cx as isize;
    let cy = cy as isize;
    let r = radius as isize;
    let w = width as isize;
    let h = height as isize;

    for dy in -r..=r {
        for dx in -r..=r {
            let px = (cx + dx).clamp(0, w - 1) as usize;
            let py = (cy + dy).clamp(0, h - 1) as usize;
            buf.push(src[py * width + px]);
        }
    }
}

/// Compute the NLM-denoised value for a single pixel at (px, py).
pub fn nlm_pixel(
    src: &[f32],
    width: usize,
    height: usize,
    px: usize,
    py: usize,
    cfg: &NlmConfig,
) -> f32 {
    let mut ref_patch = Vec::with_capacity(cfg.patch_size() * cfg.patch_size());
    let mut cmp_patch = Vec::with_capacity(cfg.patch_size() * cfg.patch_size());

    extract_patch(src, width, height, px, py, cfg.patch_radius, &mut ref_patch);

    let sr = cfg.search_radius as isize;
    let px_i = px as isize;
    let py_i = py as isize;
    let w_i = width as isize;
    let h_i = height as isize;

    let mut weight_sum = 0.0_f32;
    let mut value_sum = 0.0_f32;

    for qy in (py_i - sr).max(0)..=(py_i + sr).min(h_i - 1) {
        for qx in (px_i - sr).max(0)..=(px_i + sr).min(w_i - 1) {
            extract_patch(
                src,
                width,
                height,
                qx as usize,
                qy as usize,
                cfg.patch_radius,
                &mut cmp_patch,
            );
            let ssd = patch_ssd(&ref_patch, &cmp_patch);
            let w = nlm_weight(ssd, cfg.h);
            weight_sum += w;
            value_sum += w * src[qy as usize * width + qx as usize];
        }
    }

    if weight_sum < f32::EPSILON {
        src[py * width + px]
    } else {
        value_sum / weight_sum
    }
}

// ---------------------------------------------------------------------------
// Full-frame NLM denoiser
// ---------------------------------------------------------------------------

/// Non-Local Means denoiser.
pub struct NlmDenoiser {
    /// Denoising configuration.
    pub config: NlmConfig,
}

impl NlmDenoiser {
    /// Create a new NLM denoiser with the given configuration.
    pub fn new(config: NlmConfig) -> Self {
        Self { config }
    }

    /// Denoise an image.
    ///
    /// `src` contains floating-point pixel values (any range).
    /// Returns a denoised image of the same dimensions.
    pub fn denoise(&self, src: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] = nlm_pixel(src, width, height, x, y, &self.config);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- NlmConfig ----------

    #[test]
    fn test_config_default_patch_size() {
        let c = NlmConfig::default();
        assert_eq!(c.patch_size(), 7); // 2*3+1
    }

    #[test]
    fn test_config_strong_patch_size() {
        let c = NlmConfig::strong();
        assert_eq!(c.patch_size(), 11); // 2*5+1
    }

    #[test]
    fn test_config_mild_patch_size() {
        let c = NlmConfig::mild();
        assert_eq!(c.patch_size(), 5); // 2*2+1
    }

    #[test]
    fn test_config_search_size() {
        let c = NlmConfig::default();
        assert_eq!(c.search_size(), 21); // 2*10+1
    }

    // ---------- patch_ssd ----------

    #[test]
    fn test_patch_ssd_identical() {
        let patch = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert!((patch_ssd(&patch, &patch)).abs() < 1e-6);
    }

    #[test]
    fn test_patch_ssd_known() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        // SSD = 1 + 1 = 2
        assert!((patch_ssd(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_patch_ssd_mismatched_lengths_returns_zero() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32];
        assert!((patch_ssd(&a, &b)).abs() < 1e-6);
    }

    // ---------- nlm_weight ----------

    #[test]
    fn test_nlm_weight_zero_ssd() {
        // exp(0) = 1
        assert!((nlm_weight(0.0, 10.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nlm_weight_large_ssd_small_weight() {
        let w = nlm_weight(10_000.0, 10.0);
        assert!(w < 1e-6);
    }

    #[test]
    fn test_nlm_weight_zero_h_returns_zero() {
        assert!((nlm_weight(1.0, 0.0)).abs() < 1e-6);
    }

    // ---------- nlm_pixel ----------

    #[test]
    fn test_nlm_pixel_uniform_image() {
        // Uniform image: all pixels identical → output equals input.
        let src = vec![128.0_f32; 16 * 16];
        let cfg = NlmConfig::mild();
        let out = nlm_pixel(&src, 16, 16, 8, 8, &cfg);
        assert!((out - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_nlm_pixel_returns_value_in_range() {
        // Values in [0,255]: output should stay in that range.
        let src: Vec<f32> = (0..100).map(|i| (i % 256) as f32).collect();
        let cfg = NlmConfig::mild();
        let out = nlm_pixel(&src, 10, 10, 5, 5, &cfg);
        assert!(out >= 0.0 && out <= 255.0);
    }

    // ---------- NlmDenoiser ----------

    #[test]
    fn test_denoiser_output_length() {
        let src = vec![0.0_f32; 8 * 8];
        let d = NlmDenoiser::new(NlmConfig::mild());
        let out = d.denoise(&src, 8, 8);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_denoiser_uniform_image_unchanged() {
        let src = vec![100.0_f32; 8 * 8];
        let d = NlmDenoiser::new(NlmConfig::mild());
        let out = d.denoise(&src, 8, 8);
        for v in &out {
            assert!((v - 100.0).abs() < 1.0);
        }
    }

    #[test]
    fn test_denoiser_reduces_noise() {
        // Create a signal with additive noise.
        let signal = 128.0_f32;
        let noise_amplitude = 30.0_f32;
        let src: Vec<f32> = (0..16 * 16)
            .map(|i| {
                signal
                    + if i % 2 == 0 {
                        noise_amplitude
                    } else {
                        -noise_amplitude
                    }
            })
            .collect();
        let d = NlmDenoiser::new(NlmConfig::default());
        let out = d.denoise(&src, 16, 16);
        // Mean of output should be close to signal
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - signal).abs() < 20.0);
    }
}
