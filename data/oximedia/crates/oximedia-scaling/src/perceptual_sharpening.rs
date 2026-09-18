//! Content-adaptive perceptual sharpening algorithms.
//!
//! Provides multiple sharpening strategies ranging from classic unsharp
//! masking through AMD's Contrast Adaptive Sharpening (CAS) to a fully
//! adaptive technique that adjusts intensity based on local frequency content.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── Sharpness mode ────────────────────────────────────────────────────────────

/// Selects the perceptual sharpening algorithm to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum SharpnessMode {
    /// Classic unsharp mask: `output = input + amount × (input − blur)`.
    ///
    /// Pixels where `|input − blur| ≤ threshold / 255` are not sharpened
    /// (noise suppression).
    Unsharp {
        /// Gaussian blur standard deviation in pixels.
        sigma: f32,
        /// Sharpening amplification factor (1.0 = moderate, 2.0 = strong).
        amount: f32,
        /// Minimum absolute difference (in 0–255 scale) to trigger sharpening.
        threshold: u8,
    },

    /// High-frequency boost: subtract a low-pass signal and add the residual
    /// back at a given strength.
    HighFreqBoost {
        /// Amplification of the high-frequency residual (0.0–2.0 typical).
        strength: f32,
    },

    /// Contrast Adaptive Sharpening (FidelityFX CAS).
    ///
    /// Uses a local neighbourhood min/max to derive a spatially varying
    /// sharpening weight, which naturally reduces sharpening on already-sharp
    /// edges and noise while boosting perceived sharpness in smooth areas.
    CAS {
        /// Sharpness in [0.0, 1.0] — 0 = gentler, 1 = strongest.
        sharpness: f32,
    },

    /// Fully adaptive: detects local frequency content via Laplacian magnitude
    /// and applies stronger sharpening to smooth regions, gentler to edges.
    Adaptive,
}

// ── Gaussian utilities ────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of the given half-radius.
///
/// Returns a kernel whose weights sum to 1.0.
fn build_gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    // Radius: 3σ rounded up, minimum 1.
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    if sum > 1e-10 {
        for w in &mut kernel {
            *w /= sum;
        }
    }
    kernel
}

/// Apply a 1-D Gaussian blur to a flat 2-D image (separable X pass).
///
/// `src` has dimensions `width × height` in row-major order.
/// Returns a new buffer of the same size.
fn gaussian_blur_x(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let sx = x as i64 + k as i64 - radius as i64;
                let clamped = sx.clamp(0, width as i64 - 1) as usize;
                acc += src[y * width + clamped] * w;
                w_sum += w;
            }
            dst[y * width + x] = if w_sum > 1e-10 { acc / w_sum } else { 0.0 };
        }
    }
    dst
}

/// Apply a 1-D Gaussian blur to a flat 2-D image (separable Y pass).
fn gaussian_blur_y(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            let mut w_sum = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let sy = y as i64 + k as i64 - radius as i64;
                let clamped = sy.clamp(0, height as i64 - 1) as usize;
                acc += src[clamped * width + x] * w;
                w_sum += w;
            }
            dst[y * width + x] = if w_sum > 1e-10 { acc / w_sum } else { 0.0 };
        }
    }
    dst
}

/// Separable 2-D Gaussian blur.
///
/// Returns a new buffer of the same size as `src`.
#[must_use]
pub fn gaussian_blur_1d(src: &[f32], size: usize, sigma: f32) -> Vec<f32> {
    if size == 0 || src.is_empty() {
        return src.to_vec();
    }
    let height = src.len() / size;
    if height == 0 {
        return src.to_vec();
    }
    let kernel = build_gaussian_kernel_1d(sigma);
    let h_pass = gaussian_blur_x(src, size, height, &kernel);
    gaussian_blur_y(&h_pass, size, height, &kernel)
}

// ── 5-point Laplacian stencil ─────────────────────────────────────────────────

/// Compute the discrete Laplacian at pixel `(x, y)` using a 5-point stencil.
///
/// `stride` is the row stride (typically the image width).
/// Border pixels clamp to the image boundary.
#[must_use]
pub fn local_laplacian(data: &[f32], x: usize, y: usize, stride: usize) -> f32 {
    if stride == 0 || data.is_empty() {
        return 0.0;
    }
    let height = data.len() / stride;
    let get = |px: i64, py: i64| -> f32 {
        let cx = px.clamp(0, stride as i64 - 1) as usize;
        let cy = py.clamp(0, height as i64 - 1) as usize;
        data[cy * stride + cx]
    };
    let xi = x as i64;
    let yi = y as i64;
    let centre = get(xi, yi);
    let left = get(xi - 1, yi);
    let right = get(xi + 1, yi);
    let up = get(xi, yi - 1);
    let down = get(xi, yi + 1);
    left + right + up + down - 4.0 * centre
}

// ── UnsharpMask ───────────────────────────────────────────────────────────────

/// Unsharp mask sharpener.
///
/// Applies a Gaussian blur to the source and then adds back a scaled
/// difference signal wherever the local contrast exceeds `threshold`.
#[derive(Debug, Clone)]
pub struct UnsharpMask {
    /// Gaussian sigma for the blur.
    pub sigma: f32,
    /// Amplification factor.
    pub amount: f32,
    /// Noise suppression threshold (0–255 scale).
    pub threshold: u8,
}

impl UnsharpMask {
    /// Create an unsharp mask operator.
    #[must_use]
    pub fn new(sigma: f32, amount: f32, threshold: u8) -> Self {
        Self {
            sigma,
            amount,
            threshold,
        }
    }

    /// Sharpen a single-channel image stored as `f32` values in `[0, 1]`.
    ///
    /// `width` is the row stride / image width.  Returns a new buffer of the
    /// same size.
    #[must_use]
    pub fn apply(&self, src: &[f32], width: usize) -> Vec<f32> {
        if src.is_empty() || width == 0 {
            return src.to_vec();
        }
        let blurred = gaussian_blur_1d(src, width, self.sigma);
        let threshold_f = self.threshold as f32 / 255.0;
        src.iter()
            .zip(blurred.iter())
            .map(|(&s, &b)| {
                let diff = s - b;
                if diff.abs() > threshold_f {
                    s + self.amount * diff
                } else {
                    s
                }
            })
            .collect()
    }
}

// ── CasSharpener ──────────────────────────────────────────────────────────────

/// FidelityFX Contrast Adaptive Sharpening (CAS).
///
/// For each pixel `p` with 8-neighbour min/max:
/// ```text
/// min_g          = min(8-neighbours ∪ {p})
/// max_g          = max(8-neighbours ∪ {p})
/// sharpness_f    = sqrt(min_g / (1 − max_g + ε))
/// weight         = sharpness_f · (−0.125·sharpness − 0.125)
/// output         = (p + 4·weight·p) / (1 + 4·weight)
/// ```
///
/// Sharpness = 0.0 is the gentlest setting; 1.0 is the strongest.
#[derive(Debug, Clone)]
pub struct CasSharpener {
    /// Sharpness parameter in [0.0, 1.0].
    pub sharpness: f32,
}

impl CasSharpener {
    /// Create a CAS sharpener.
    #[must_use]
    pub fn new(sharpness: f32) -> Self {
        Self {
            sharpness: sharpness.clamp(0.0, 1.0),
        }
    }

    /// Apply CAS to a single-channel float image.
    ///
    /// Pixels outside the interior border are copied unchanged.
    #[must_use]
    pub fn apply(&self, src: &[f32], width: usize) -> Vec<f32> {
        if src.is_empty() || width == 0 {
            return src.to_vec();
        }
        let height = src.len() / width;
        if height < 3 {
            return src.to_vec();
        }

        let mut dst = src.to_vec();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let p = src[y * width + x];
                let neighbours = [
                    src[(y - 1) * width + (x - 1)],
                    src[(y - 1) * width + x],
                    src[(y - 1) * width + (x + 1)],
                    src[y * width + (x - 1)],
                    src[y * width + (x + 1)],
                    src[(y + 1) * width + (x - 1)],
                    src[(y + 1) * width + x],
                    src[(y + 1) * width + (x + 1)],
                ];

                let min_g = neighbours.iter().cloned().fold(p, f32::min);
                let max_g = neighbours.iter().cloned().fold(p, f32::max);

                // Avoid division by zero or sqrt of negative.
                let denom = (1.0 - max_g).max(1e-6);
                let sharpness_factor = (min_g / denom).max(0.0).sqrt();

                let weight = sharpness_factor * (-0.125 * self.sharpness - 0.125);
                let denom_cas = 1.0 + 4.0 * weight;
                let output = if denom_cas.abs() > 1e-8 {
                    (p + 4.0 * weight * p) / denom_cas
                } else {
                    p
                };

                dst[y * width + x] = output;
            }
        }
        dst
    }
}

// ── AdaptiveSharpener ─────────────────────────────────────────────────────────

/// Adaptive sharpener using local frequency content.
///
/// Detects local frequency content by computing the Laplacian magnitude at
/// each pixel.  Smooth regions (low Laplacian) receive stronger sharpening
/// while already-sharp edges (high Laplacian) receive less.
#[derive(Debug, Clone)]
pub struct AdaptiveSharpener {
    /// Base sharpening strength (0.0 = no sharpening).
    pub base_strength: f32,
    /// Gaussian sigma for the blur used in the sharpening pass.
    pub blur_sigma: f32,
}

impl AdaptiveSharpener {
    /// Create an adaptive sharpener.
    #[must_use]
    pub fn new(base_strength: f32, blur_sigma: f32) -> Self {
        Self {
            base_strength,
            blur_sigma,
        }
    }

    /// Apply adaptive sharpening to a single-channel float image.
    ///
    /// `width` is the row stride.  Returns a new buffer of the same size.
    #[must_use]
    pub fn apply(&self, src: &[f32], width: usize) -> Vec<f32> {
        if src.is_empty() || width == 0 {
            return src.to_vec();
        }
        let height = src.len() / width;
        if height == 0 {
            return src.to_vec();
        }

        // Compute Laplacian magnitudes for the whole image.
        let laplacians: Vec<f32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| local_laplacian(src, x, y, width).abs()))
            .collect();

        // Find the maximum Laplacian for normalization.
        let max_lap = laplacians.iter().cloned().fold(0.0_f32, f32::max).max(1e-8);

        // Compute the Gaussian-blurred image once.
        let blurred = gaussian_blur_1d(src, width, self.blur_sigma);

        src.iter()
            .enumerate()
            .map(|(idx, &s)| {
                let lap_norm = laplacians[idx] / max_lap; // 0 = smooth, 1 = sharp edge
                                                          // Smooth areas get full base_strength; sharp areas get almost none.
                let local_strength = self.base_strength * (1.0 - lap_norm).max(0.0);
                let diff = s - blurred[idx];
                s + local_strength * diff
            })
            .collect()
    }
}

// ── Top-level dispatch ────────────────────────────────────────────────────────

/// Apply the selected sharpness mode to a single-channel float image.
///
/// `width` is the row stride.  Returns a sharpened buffer of the same size.
/// Values are not clamped — callers should clamp to `[0.0, 1.0]` as needed.
#[must_use]
pub fn sharpen(src: &[f32], width: usize, mode: &SharpnessMode) -> Vec<f32> {
    match mode {
        SharpnessMode::Unsharp {
            sigma,
            amount,
            threshold,
        } => UnsharpMask::new(*sigma, *amount, *threshold).apply(src, width),

        SharpnessMode::HighFreqBoost { strength } => {
            if src.is_empty() || width == 0 {
                return src.to_vec();
            }
            let blurred = gaussian_blur_1d(src, width, 1.0);
            src.iter()
                .zip(blurred.iter())
                .map(|(&s, &b)| s + strength * (s - b))
                .collect()
        }

        SharpnessMode::CAS { sharpness } => CasSharpener::new(*sharpness).apply(src, width),

        SharpnessMode::Adaptive => AdaptiveSharpener::new(1.5, 1.0).apply(src, width),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── gaussian_blur_1d ──────────────────────────────────────────────────────

    #[test]
    fn gaussian_blur_1d_uniform_image_unchanged() {
        let src = vec![0.5f32; 64];
        let blurred = gaussian_blur_1d(&src, 8, 1.0);
        for &v in &blurred {
            assert!((v - 0.5).abs() < 0.001, "uniform image blurred to {v}");
        }
    }

    #[test]
    fn gaussian_blur_1d_reduces_peak() {
        let mut src = vec![0.0f32; 81];
        src[40] = 1.0; // centre spike in 9×9
        let blurred = gaussian_blur_1d(&src, 9, 1.5);
        assert!(blurred[40] < 1.0, "blur should reduce peak");
        assert!(blurred[40] > 0.0, "peak should not vanish entirely");
    }

    #[test]
    fn gaussian_blur_1d_empty() {
        let result = gaussian_blur_1d(&[], 8, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn gaussian_blur_1d_zero_width() {
        let src = vec![0.5f32; 8];
        let result = gaussian_blur_1d(&src, 0, 1.0);
        assert_eq!(result.len(), src.len()); // returns src as-is
    }

    // ── local_laplacian ───────────────────────────────────────────────────────

    #[test]
    fn local_laplacian_uniform_image_zero() {
        let src = vec![0.5f32; 25]; // 5×5
        for y in 1..4 {
            for x in 1..4 {
                let lap = local_laplacian(&src, x, y, 5);
                assert!(lap.abs() < 1e-5, "uniform Laplacian at ({x},{y}) = {lap}");
            }
        }
    }

    #[test]
    fn local_laplacian_spike_positive() {
        let mut src = vec![0.0f32; 25]; // 5×5
        src[12] = 1.0; // centre pixel spike
        let lap = local_laplacian(&src, 2, 2, 5);
        // Laplacian of isolated spike = 0 + 0 + 0 + 0 - 4×1 = -4
        assert!((lap - (-4.0)).abs() < 1e-4, "spike Laplacian = {lap}");
    }

    #[test]
    fn local_laplacian_border_clamping() {
        let src = vec![0.0f32; 25];
        // Should not panic or produce NaN at borders.
        let v = local_laplacian(&src, 0, 0, 5);
        assert!(v.is_finite());
        let v2 = local_laplacian(&src, 4, 4, 5);
        assert!(v2.is_finite());
    }

    // ── UnsharpMask ───────────────────────────────────────────────────────────

    #[test]
    fn unsharp_mask_uniform_unchanged() {
        let src = vec![0.5f32; 64];
        let um = UnsharpMask::new(1.0, 1.0, 0);
        let out = um.apply(&src, 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn unsharp_mask_sharpens_edge() {
        // Construct a hard edge in an 8×1 image.
        let src: Vec<f32> = (0..8).map(|i| if i < 4 { 0.0 } else { 1.0 }).collect();
        let um = UnsharpMask::new(1.0, 2.0, 0);
        let out = um.apply(&src, 8);
        // Pixels near the edge should be pushed further apart.
        assert!(
            out[3] <= src[3],
            "edge pixel should be pushed down or unchanged"
        );
        assert!(
            out[4] >= src[4],
            "edge pixel should be pushed up or unchanged"
        );
    }

    #[test]
    fn unsharp_mask_threshold_suppresses_noise() {
        // Tiny noise below threshold should not be amplified.
        let src: Vec<f32> = (0..64)
            .map(|i| 0.5 + if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let um = UnsharpMask::new(1.0, 5.0, 10); // threshold = 10/255 ≈ 0.039
        let out = um.apply(&src, 8);
        for (&s, &o) in src.iter().zip(out.iter()) {
            assert!(
                (s - o).abs() < 0.01,
                "noise above threshold after suppression"
            );
        }
    }

    #[test]
    fn unsharp_mask_empty_returns_empty() {
        let um = UnsharpMask::new(1.0, 1.0, 0);
        let out = um.apply(&[], 8);
        assert!(out.is_empty());
    }

    // ── CasSharpener ──────────────────────────────────────────────────────────

    #[test]
    fn cas_uniform_image_nearly_unchanged() {
        let src = vec![0.5f32; 64];
        let cas = CasSharpener::new(0.5);
        let out = cas.apply(&src, 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.05, "CAS changed uniform pixel to {v}");
        }
    }

    #[test]
    fn cas_sharpness_clamped_to_unit() {
        let cas = CasSharpener::new(5.0);
        assert!(cas.sharpness <= 1.0);
        let cas2 = CasSharpener::new(-1.0);
        assert!(cas2.sharpness >= 0.0);
    }

    #[test]
    fn cas_returns_same_size() {
        let src = vec![0.5f32; 64];
        let cas = CasSharpener::new(0.5);
        let out = cas.apply(&src, 8);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn cas_empty_returns_empty() {
        let cas = CasSharpener::new(0.5);
        let out = cas.apply(&[], 8);
        assert!(out.is_empty());
    }

    #[test]
    fn cas_output_finite() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 / 63.0).min(1.0)).collect();
        let cas = CasSharpener::new(0.8);
        let out = cas.apply(&src, 8);
        for &v in &out {
            assert!(v.is_finite(), "CAS produced non-finite value");
        }
    }

    // ── AdaptiveSharpener ─────────────────────────────────────────────────────

    #[test]
    fn adaptive_uniform_unchanged() {
        let src = vec![0.5f32; 64];
        let ad = AdaptiveSharpener::new(1.0, 1.0);
        let out = ad.apply(&src, 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.001, "adaptive changed uniform to {v}");
        }
    }

    #[test]
    fn adaptive_sharpens_smooth_region() {
        // A smooth gradient should be sharpened in the middle.
        let src: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let ad = AdaptiveSharpener::new(2.0, 0.5);
        let out = ad.apply(&src, 8);
        // Output values should differ from input somewhere.
        let total_diff: f32 = src
            .iter()
            .zip(out.iter())
            .map(|(&s, &o)| (s - o).abs())
            .sum();
        assert!(total_diff > 0.0, "adaptive produced no change on gradient");
    }

    #[test]
    fn adaptive_output_finite() {
        let src: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let ad = AdaptiveSharpener::new(1.5, 1.0);
        let out = ad.apply(&src, 8);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── sharpen dispatch ──────────────────────────────────────────────────────

    #[test]
    fn sharpen_dispatch_unsharp() {
        let src = vec![0.5f32; 64];
        let mode = SharpnessMode::Unsharp {
            sigma: 1.0,
            amount: 1.0,
            threshold: 0,
        };
        let out = sharpen(&src, 8, &mode);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn sharpen_dispatch_high_freq_boost() {
        let src = vec![0.5f32; 64];
        let mode = SharpnessMode::HighFreqBoost { strength: 1.0 };
        let out = sharpen(&src, 8, &mode);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn sharpen_dispatch_cas() {
        let src = vec![0.5f32; 64];
        let mode = SharpnessMode::CAS { sharpness: 0.5 };
        let out = sharpen(&src, 8, &mode);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn sharpen_dispatch_adaptive() {
        let src = vec![0.5f32; 64];
        let out = sharpen(&src, 8, &SharpnessMode::Adaptive);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn sharpen_high_freq_boost_increases_contrast() {
        // Build a clear high-contrast region.
        let mut src = vec![0.5f32; 64];
        for i in 0..8 {
            src[i] = if i < 4 { 0.2 } else { 0.8 };
        }
        let mode = SharpnessMode::HighFreqBoost { strength: 1.5 };
        let out = sharpen(&src, 8, &mode);
        // At least one pixel should have moved away from 0.5.
        let max_diff = src
            .iter()
            .zip(out.iter())
            .map(|(&s, &o)| (s - o).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 0.0, "high-freq boost produced no change");
    }
}
