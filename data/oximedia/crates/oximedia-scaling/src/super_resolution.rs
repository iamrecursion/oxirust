//! Super-resolution upscaling algorithms.
//!
//! Provides edge-preserving and frequency-domain super-resolution methods.

use std::f32::consts::PI;

/// Super-resolution algorithm selection.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrAlgorithm {
    /// Simple bicubic upscaling.
    Bicubic,
    /// Lanczos-based upscaling.
    Lanczos,
    /// Edge-preserving upscaling using Sobel detection.
    EdgePreserving,
    /// Frequency-domain upscaling (zero-padding DFT).
    Frequency,
    /// Neural-network stub (placeholder).
    NeuralStub,
}

/// Configuration for super-resolution.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SrConfig {
    /// Algorithm to use.
    pub algorithm: SrAlgorithm,
    /// Integer scale factor (e.g. 2 = double resolution).
    pub scale: u32,
    /// Sharpening strength applied post-upscale (0.0–1.0).
    pub sharpening_strength: f32,
}

impl Default for SrConfig {
    fn default() -> Self {
        Self {
            algorithm: SrAlgorithm::EdgePreserving,
            scale: 2,
            sharpening_strength: 0.3,
        }
    }
}

/// Edge-preserving upscaler using edge-directed interpolation (EDI) with
/// Sobel guidance and unsharp masking.
pub struct EdgePreservingUpscaler;

impl EdgePreservingUpscaler {
    /// Upscale a single-channel image by an integer `scale` factor using a
    /// classical edge-guided super-resolution pipeline:
    ///
    /// 1. **Bicubic upscale** to target size.
    /// 2. **Edge-directed interpolation (EDI)**: at each pixel compute the
    ///    horizontal gradient magnitude |Gx| and vertical magnitude |Gy| from
    ///    Sobel kernels on the upscaled image.  Where |Gx| > |Gy| (strong
    ///    horizontal gradient → vertical edge) prefer horizontal-neighbour
    ///    averaging to preserve the edge.  At low-gradient pixels the bicubic
    ///    value is kept unchanged.
    /// 3. **Unsharp mask** (radius 1, strength 0.5) to recover edge sharpness
    ///    lost during upscaling.
    ///
    /// The `NeuralStub` variant in `SrAlgorithm` falls through to this
    /// implementation since no inference engine is available at compile time.
    #[must_use]
    #[allow(dead_code)]
    pub fn upscale(src: &[f32], src_w: u32, src_h: u32, scale: u32) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || scale == 0 {
            return Vec::new();
        }
        let dst_w = src_w * scale;
        let dst_h = src_h * scale;
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        // ------------------------------------------------------------------ //
        // Pass 1: bicubic upscale as base.
        // ------------------------------------------------------------------ //
        let bicubic = bicubic_upscale(src, sw, sh, dw, dh);

        // ------------------------------------------------------------------ //
        // Pass 2: edge-directed interpolation on the bicubic output.
        // ------------------------------------------------------------------ //
        let edi = edge_directed_interpolation(&bicubic, dw, dh);

        // ------------------------------------------------------------------ //
        // Pass 3: unsharp mask to recover sharpness.
        // ------------------------------------------------------------------ //
        unsharp_mask(&edi, dw, dh, 0.5)
    }
}

/// Edge-directed interpolation (EDI) on an already-upscaled image.
///
/// For each pixel computes the horizontal (|Gx|) and vertical (|Gy|) Sobel
/// gradient magnitudes.
///
/// * **Strong vertical edge** (|Gx| > |Gy|): replace the pixel with the
///   average of its left and right horizontal neighbours.  This smooths along
///   the edge direction while preserving the edge itself.
/// * **Otherwise**: keep the bicubic value as-is.
///
/// Interior pixels only — border pixels are copied unchanged.
#[allow(dead_code)]
fn edge_directed_interpolation(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = src.to_vec();

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            // 3×3 neighbourhood.
            let tl = src[(y - 1) * w + (x - 1)];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let ml = src[y * w + (x - 1)];
            let mr = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            // Sobel Gx and Gy magnitudes.
            let gx = (-tl - 2.0 * ml - bl + tr + 2.0 * mr + br).abs();
            let gy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br).abs();

            if gx > gy {
                // Vertical edge: smooth horizontally (preserve the edge).
                out[y * w + x] = (ml + mr) * 0.5;
            }
            // else: keep bicubic value
        }
    }

    out
}

/// Apply a simple unsharp mask with a 3×3 box blur and the given `strength`.
///
/// `output = clamp(src + strength × (src − blur))` where blur is a 3×3 box
/// average.  `strength` = 0.5 recovers moderate sharpness without ringing.
#[allow(dead_code)]
fn unsharp_mask(src: &[f32], w: usize, h: usize, strength: f32) -> Vec<f32> {
    // 3×3 box blur.
    let mut blur = src.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let sum = src[(y - 1) * w + (x - 1)]
                + src[(y - 1) * w + x]
                + src[(y - 1) * w + (x + 1)]
                + src[y * w + (x - 1)]
                + src[y * w + x]
                + src[y * w + (x + 1)]
                + src[(y + 1) * w + (x - 1)]
                + src[(y + 1) * w + x]
                + src[(y + 1) * w + (x + 1)];
            blur[y * w + x] = sum / 9.0;
        }
    }

    // Unsharp mask: original + strength × (original − blurred).
    src.iter()
        .zip(blur.iter())
        .map(|(&s, &b)| (s + strength * (s - b)).clamp(0.0, 1.0))
        .collect()
}

/// Bicubic upscale using cubic Hermite spline.
#[allow(dead_code)]
fn bicubic_upscale(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * scale_x - 0.5;
            let fy = (dy as f32 + 0.5) * scale_y - 0.5;
            dst[dy * dw + dx] = bicubic_sample(src, sw, sh, fx, fy);
        }
    }
    dst
}

/// Sample with bicubic (Catmull-Rom) interpolation.
#[allow(dead_code)]
fn bicubic_sample(src: &[f32], sw: usize, sh: usize, fx: f32, fy: f32) -> f32 {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let mut result = 0.0f32;
    for (j, &wy_j) in wy.iter().enumerate() {
        for (i, &wx_i) in wx.iter().enumerate() {
            let px = (ix + i as i32 - 1).clamp(0, sw as i32 - 1) as usize;
            let py = (iy + j as i32 - 1).clamp(0, sh as i32 - 1) as usize;
            result += src[py * sw + px] * wx_i * wy_j;
        }
    }
    result.clamp(0.0, 1.0)
}

/// Catmull-Rom spline weights for fractional position `t` in [0,1].
#[allow(dead_code)]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Compute normalized Sobel edge magnitude (0.0–1.0).
#[allow(dead_code)]
fn compute_sobel_edges(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut edges = vec![0.0f32; w * h];
    let mut max_val = 0.0f32;

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let tl = src[(y - 1) * w + (x - 1)];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let ml = src[y * w + (x - 1)];
            let mr = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            let mag = (gx * gx + gy * gy).sqrt();
            edges[y * w + x] = mag;
            if mag > max_val {
                max_val = mag;
            }
        }
    }

    // Normalize
    if max_val > 1e-8 {
        for e in &mut edges {
            *e /= max_val;
        }
    }
    edges
}

/// Frequency-domain upscaler using DFT zero-padding.
pub struct FrequencyUpscaler;

impl FrequencyUpscaler {
    /// Upscale a single-channel image by zero-padding in the frequency domain.
    ///
    /// Implements a simplified real 2D DFT: computes the DFT of the source,
    /// zero-pads the frequency coefficients, and applies the inverse DFT.
    #[must_use]
    #[allow(dead_code)]
    pub fn upscale(src: &[f32], src_w: u32, src_h: u32, scale: u32) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || scale == 0 {
            return Vec::new();
        }
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = sw * scale as usize;
        let dh = sh * scale as usize;

        // Forward DFT (real-to-complex using naive DFT for correctness)
        let freqs = dft_2d(src, sw, sh);

        // Zero-pad in frequency domain: center-pad the spectrum
        let mut padded = vec![(0.0f32, 0.0f32); dw * dh];
        let half_sw = sw / 2;
        let half_sh = sh / 2;

        for fy in 0..sh {
            for fx in 0..sw {
                let dst_fx = if fx < half_sw { fx } else { dw - (sw - fx) };
                let dst_fy = if fy < half_sh { fy } else { dh - (sh - fy) };
                if dst_fx < dw && dst_fy < dh {
                    padded[dst_fy * dw + dst_fx] = freqs[fy * sw + fx];
                }
            }
        }

        // Inverse DFT
        let scale_factor = (scale * scale) as f32;
        let spatial = idft_2d(&padded, dw, dh);
        spatial
            .iter()
            .map(|&v| (v / scale_factor).clamp(0.0, 1.0))
            .collect()
    }
}

/// Naive 2D DFT (O(N^4) — suitable for small test images only).
#[allow(dead_code)]
fn dft_2d(src: &[f32], w: usize, h: usize) -> Vec<(f32, f32)> {
    let mut out = vec![(0.0f32, 0.0f32); w * h];
    let wf = w as f32;
    let hf = h as f32;

    for vy in 0..h {
        for vx in 0..w {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for sy in 0..h {
                for sx in 0..w {
                    let angle =
                        -2.0 * PI * (vx as f32 * sx as f32 / wf + vy as f32 * sy as f32 / hf);
                    let val = src[sy * w + sx];
                    re += val * angle.cos();
                    im += val * angle.sin();
                }
            }
            out[vy * w + vx] = (re, im);
        }
    }
    out
}

/// Naive 2D inverse DFT.
#[allow(dead_code)]
fn idft_2d(freq: &[(f32, f32)], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let wf = w as f32;
    let hf = h as f32;
    let norm = wf * hf;

    for sy in 0..h {
        for sx in 0..w {
            let mut val = 0.0f32;
            for vy in 0..h {
                for vx in 0..w {
                    let angle =
                        2.0 * PI * (vx as f32 * sx as f32 / wf + vy as f32 * sy as f32 / hf);
                    let (re, im) = freq[vy * w + vx];
                    val += re * angle.cos() - im * angle.sin();
                }
            }
            out[sy * w + sx] = val / norm;
        }
    }
    out
}

// ── High-level SuperResolutionEngine API ─────────────────────────────────────

/// Algorithm variants for the high-level `SuperResolutionEngine`.
///
/// These differ from the lower-level [`SrAlgorithm`] enum in that every
/// variant also applies a sharpening pass controlled by
/// [`SuperResolutionConfig::sharpening_strength`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperResAlgorithm {
    /// Catmull-Rom bicubic upscale followed by an unsharp mask.
    BicubicSharp,
    /// Lanczos3 (a=3 sinc) upscale followed by an unsharp mask.
    Lanczos3Sharp,
    /// Edge-Directed Super-Resolution (EDSR) approximation.
    ///
    /// Computes Sobel gradient magnitude *and angle* on the upscaled bicubic
    /// base, then at each strong-edge pixel blends in a directional interpolate
    /// taken **along** the detected edge direction rather than across it.  This
    /// preserves diagonal edges more faithfully than plain bicubic.
    Edsr,
}

/// Configuration for the high-level super-resolution engine.
#[derive(Debug, Clone)]
pub struct SuperResolutionConfig {
    /// Integer upscale factor.  Values of 2 and 4 are recommended;
    /// 1 is a no-op (input is returned unchanged).
    pub scale_factor: u32,
    /// Algorithm to use for upscaling.
    pub algorithm: SuperResAlgorithm,
    /// Strength of the unsharp-mask sharpening pass (0.0 = none, 1.0 = maximum).
    pub sharpening_strength: f32,
}

impl Default for SuperResolutionConfig {
    fn default() -> Self {
        Self {
            scale_factor: 2,
            algorithm: SuperResAlgorithm::Edsr,
            sharpening_strength: 0.5,
        }
    }
}

/// Error type returned by [`SuperResolutionEngine::upscale`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperResError {
    /// The provided pixel buffer length does not match `src_w * src_h`.
    DimensionMismatch {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        got: usize,
    },
    /// Source dimensions are zero.
    ZeroDimension,
    /// Scale factor is zero.
    ZeroScaleFactor,
}

impl std::fmt::Display for SuperResError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "pixel buffer length mismatch: expected {expected} bytes (w×h), got {got}"
            ),
            Self::ZeroDimension => write!(f, "source width or height is zero"),
            Self::ZeroScaleFactor => write!(f, "scale_factor must be ≥ 1"),
        }
    }
}

impl std::error::Error for SuperResError {}

/// High-level super-resolution upscaler that operates on **u8 grayscale** images
/// (one byte per pixel, row-major).
///
/// ```rust
/// use oximedia_scaling::super_resolution::{
///     SuperResolutionConfig, SuperResolutionEngine, SuperResAlgorithm,
/// };
///
/// let config = SuperResolutionConfig {
///     scale_factor: 2,
///     algorithm: SuperResAlgorithm::BicubicSharp,
///     sharpening_strength: 0.4,
/// };
/// let engine = SuperResolutionEngine::new(config);
/// let src = vec![128u8; 16]; // 4×4 uniform grey
/// let (dst, w, h) = engine.upscale(&src, 4, 4).expect("upscale should succeed");
/// assert_eq!((w, h), (8, 8));
/// assert_eq!(dst.len(), 64);
/// ```
pub struct SuperResolutionEngine {
    config: SuperResolutionConfig,
}

impl SuperResolutionEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SuperResolutionConfig) -> Self {
        Self { config }
    }

    /// Upscale a grayscale u8 image.
    ///
    /// `src` must contain exactly `src_w * src_h` bytes (one byte per pixel).
    /// Returns `(output_pixels, output_width, output_height)`.
    pub fn upscale(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
    ) -> Result<(Vec<u8>, u32, u32), SuperResError> {
        // ── Validate inputs ──────────────────────────────────────────────
        if src_w == 0 || src_h == 0 {
            return Err(SuperResError::ZeroDimension);
        }
        if self.config.scale_factor == 0 {
            return Err(SuperResError::ZeroScaleFactor);
        }
        let expected = (src_w as usize) * (src_h as usize);
        if src.len() != expected {
            return Err(SuperResError::DimensionMismatch {
                expected,
                got: src.len(),
            });
        }

        // Identity short-circuit.
        if self.config.scale_factor == 1 {
            return Ok((src.to_vec(), src_w, src_h));
        }

        // ── Normalise u8 → f32 ───────────────────────────────────────────
        let src_f32: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();

        let dst_w = src_w * self.config.scale_factor;
        let dst_h = src_h * self.config.scale_factor;
        let dw = dst_w as usize;
        let dh = dst_h as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;

        // ── Upscale ──────────────────────────────────────────────────────
        let upscaled_f32: Vec<f32> = match self.config.algorithm {
            SuperResAlgorithm::BicubicSharp => {
                let base = bicubic_upscale(&src_f32, sw, sh, dw, dh);
                unsharp_mask(&base, dw, dh, self.config.sharpening_strength)
            }
            SuperResAlgorithm::Lanczos3Sharp => {
                let base = lanczos3_upscale(&src_f32, sw, sh, dw, dh);
                unsharp_mask(&base, dw, dh, self.config.sharpening_strength)
            }
            SuperResAlgorithm::Edsr => {
                edsr_upscale(&src_f32, sw, sh, dw, dh, self.config.sharpening_strength)
            }
        };

        // ── Convert f32 → u8 ─────────────────────────────────────────────
        let output: Vec<u8> = upscaled_f32
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        Ok((output, dst_w, dst_h))
    }
}

// ── Lanczos3 upscaler (separable H×V passes, a=3) ────────────────────────────

/// Lanczos kernel weight for parameter `a=3`.
#[inline]
fn lanczos3_weight(x: f32) -> f32 {
    const A: f32 = 3.0;
    if x.abs() < 1e-7 {
        return 1.0;
    }
    if x.abs() >= A {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    let apx = std::f32::consts::PI * x / A;
    A * px.sin() * apx.sin() / (px * px)
}

/// Lanczos3 1-D horizontal pass: resample `src` (width=`sw`, height=`sh`) to
/// width=`dw`, keeping height unchanged.
fn lanczos3_horizontal(src: &[f32], sw: usize, sh: usize, dw: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * sh];
    let scale = sw as f32 / dw as f32;

    for row in 0..sh {
        for dx in 0..dw {
            let src_x = (dx as f32 + 0.5) * scale - 0.5;
            let center = src_x.floor() as i32;

            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for tap in -2i32..=3 {
                let sx = (center + tap).clamp(0, sw as i32 - 1) as usize;
                let w = lanczos3_weight(src_x - (center + tap) as f32);
                sum += src[row * sw + sx] * w;
                weight_sum += w;
            }

            dst[row * dw + dx] = if weight_sum.abs() > 1e-8 {
                (sum / weight_sum).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }
    dst
}

/// Lanczos3 1-D vertical pass: resample `src` (width=`dw`, height=`sh`) to
/// height=`dh`.
fn lanczos3_vertical(src: &[f32], dw: usize, sh: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    let scale = sh as f32 / dh as f32;

    for dy in 0..dh {
        let src_y = (dy as f32 + 0.5) * scale - 0.5;
        let center = src_y.floor() as i32;

        for col in 0..dw {
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for tap in -2i32..=3 {
                let sy = (center + tap).clamp(0, sh as i32 - 1) as usize;
                let w = lanczos3_weight(src_y - (center + tap) as f32);
                sum += src[sy * dw + col] * w;
                weight_sum += w;
            }

            dst[dy * dw + col] = if weight_sum.abs() > 1e-8 {
                (sum / weight_sum).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }
    dst
}

/// Full separable Lanczos3 upscale from (sw×sh) to (dw×dh).
fn lanczos3_upscale(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }
    // H-pass: sw→dw, height stays sh
    let h_pass = lanczos3_horizontal(src, sw, sh, dw);
    // V-pass: sh→dh, width stays dw
    lanczos3_vertical(&h_pass, dw, sh, dh)
}

// ── Edge-Directed Super-Resolution (EDSR) ────────────────────────────────────

/// EDSR approximation.
///
/// Pipeline:
/// 1. Bicubic upscale to target size (provides a smooth base).
/// 2. Compute Sobel Gx, Gy at each pixel of the upscaled image.
/// 3. At pixels where the gradient magnitude exceeds a threshold, compute
///    the **edge angle** (atan2 of Gy/Gx) and blend in a directional
///    interpolate taken *along* the edge.  This reinforces diagonal edges
///    that bicubic tends to blur.
/// 4. Apply unsharp mask to recover remaining sharpness.
fn edsr_upscale(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, sharpen: f32) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return Vec::new();
    }

    // Step 1: bicubic base.
    let base = bicubic_upscale(src, sw, sh, dw, dh);

    // Step 2+3: edge-directed refinement.
    let refined = edge_directed_refinement(&base, dw, dh);

    // Step 4: unsharp mask.
    unsharp_mask(&refined, dw, dh, sharpen)
}

/// Edge-directed refinement with gradient angle estimation.
///
/// For each interior pixel:
/// - Compute Sobel Gx/Gy on the *base* image.
/// - If the gradient magnitude exceeds `EDGE_THRESHOLD`, compute the angle θ.
/// - Build a directional sample *along* the edge (perpendicular to the
///   gradient) by bilinearly sampling the base at offsets (±cos θ, ±sin θ).
/// - Blend the directional sample into the pixel with weight proportional to
///   the clamped magnitude; weak-gradient pixels retain the bicubic value.
fn edge_directed_refinement(base: &[f32], w: usize, h: usize) -> Vec<f32> {
    const EDGE_THRESHOLD: f32 = 0.05;
    const MAX_BLEND: f32 = 0.8; // maximum blend weight for directional sample

    let mut out = base.to_vec();

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            // 3×3 Sobel neighbourhood.
            let tl = base[(y - 1) * w + (x - 1)];
            let tc = base[(y - 1) * w + x];
            let tr = base[(y - 1) * w + (x + 1)];
            let ml = base[y * w + (x - 1)];
            let mr = base[y * w + (x + 1)];
            let bl = base[(y + 1) * w + (x - 1)];
            let bc = base[(y + 1) * w + x];
            let br = base[(y + 1) * w + (x + 1)];

            let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            let mag = (gx * gx + gy * gy).sqrt();

            if mag < EDGE_THRESHOLD {
                // Weak or no edge — keep bicubic value.
                continue;
            }

            // Edge angle: direction *along* the edge = perpendicular to gradient.
            // gradient direction = (gx, gy); along-edge direction = (-gy, gx) / mag.
            let along_x = -gy / mag;
            let along_y = gx / mag;

            // Sample the base image at (x ± along_x, y ± along_y) via bilinear.
            let sample_fwd = bilinear_sample(base, w, h, x as f32 + along_x, y as f32 + along_y);
            let sample_bck = bilinear_sample(base, w, h, x as f32 - along_x, y as f32 - along_y);
            let directional = (sample_fwd + sample_bck) * 0.5;

            // Blend weight proportional to magnitude, capped at MAX_BLEND.
            let blend = (mag * 2.0).min(MAX_BLEND);
            out[y * w + x] = (1.0 - blend) * base[y * w + x] + blend * directional;
        }
    }

    out
}

/// Bilinear sample at sub-pixel position (fx, fy) with clamped boundary.
#[inline]
fn bilinear_sample(src: &[f32], w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let x0 = fx.floor().max(0.0) as usize;
    let y0 = fy.floor().max(0.0) as usize;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));
    let x0c = x0.min(w.saturating_sub(1));
    let y0c = y0.min(h.saturating_sub(1));

    let tx = (fx - fx.floor()).clamp(0.0, 1.0);
    let ty = (fy - fy.floor()).clamp(0.0, 1.0);

    let top = src[y0c * w + x0c] * (1.0 - tx) + src[y0c * w + x1] * tx;
    let bot = src[y1 * w + x0c] * (1.0 - tx) + src[y1 * w + x1] * tx;
    top * (1.0 - ty) + bot * ty
}

// ── Quality estimate for super-resolution output ──────────────────────────────

/// Quality estimate for super-resolution output.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SrQualityEstimate {
    /// Estimated PSNR improvement in dB.
    pub psnr_db_estimate: f32,
    /// Edge sharpness score (0.0–1.0, higher is sharper).
    pub edge_sharpness: f32,
    /// Aliasing score (0.0–1.0, lower is better).
    pub alias_score: f32,
}

impl SrQualityEstimate {
    /// Compute quality metrics by comparing original and upscaled images.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute(original: &[f32], upscaled: &[f32], scale: u32) -> Self {
        // Downsample upscaled back to original size and compute MSE
        let s = scale as usize;
        let dst_len = upscaled.len() / (s * s);
        let orig_len = original.len();
        let compare_len = dst_len.min(orig_len);

        // Simple PSNR estimate: compare downsampled-upscaled vs original
        let mut mse = 0.0f32;
        for i in 0..compare_len {
            let up_y = (i / (compare_len / s.max(1)).max(1)) * s;
            let up_x = (i % (compare_len / s.max(1)).max(1)) * s;
            let w_up = ((upscaled.len() as f32).sqrt() as usize).max(1);
            let idx = (up_y * w_up + up_x).min(upscaled.len() - 1);
            let diff = original[i] - upscaled[idx];
            mse += diff * diff;
        }
        mse /= compare_len.max(1) as f32;

        let psnr_db_estimate = if mse < 1e-10 {
            100.0
        } else {
            10.0 * (1.0 / mse).log10()
        };

        // Edge sharpness: measure gradient magnitude in upscaled
        let w = ((upscaled.len() as f32).sqrt() as usize).max(1);
        let h = (upscaled.len() / w).max(1);
        let edges = compute_sobel_edges(upscaled, w, h);
        let edge_sharpness = edges.iter().copied().sum::<f32>() / edges.len() as f32;

        // Alias score: high-frequency energy ratio
        let total_energy: f32 = upscaled.iter().map(|&v| v * v).sum();
        let hf_energy: f32 = edges.iter().map(|&v| v * v).sum();
        let alias_score = if total_energy > 1e-8 {
            (hf_energy / total_energy).min(1.0)
        } else {
            0.0
        };

        Self {
            psnr_db_estimate,
            edge_sharpness,
            alias_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_config_default() {
        let config = SrConfig::default();
        assert_eq!(config.scale, 2);
        assert_eq!(config.algorithm, SrAlgorithm::EdgePreserving);
    }

    #[test]
    fn test_edge_preserving_upscale_output_size() {
        let src = vec![0.5f32; 16]; // 4x4
        let dst = EdgePreservingUpscaler::upscale(&src, 4, 4, 2);
        assert_eq!(dst.len(), 64); // 8x8
    }

    #[test]
    fn test_edge_preserving_upscale_uniform() {
        // Uniform source should produce uniform output
        let src = vec![0.5f32; 16];
        let dst = EdgePreservingUpscaler::upscale(&src, 4, 4, 2);
        for &v in &dst {
            assert!((v - 0.5).abs() < 0.01, "Expected ~0.5, got {v}");
        }
    }

    #[test]
    fn test_edge_preserving_upscale_empty() {
        let dst = EdgePreservingUpscaler::upscale(&[], 0, 0, 2);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_sobel_edges_uniform_image() {
        let src = vec![0.5f32; 16];
        let edges = compute_sobel_edges(&src, 4, 4);
        // Uniform image has zero gradients
        for &e in &edges {
            assert!(e.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_sobel_edges_step_edge() {
        let mut src = vec![0.0f32; 16];
        // Left half = 0, right half = 1
        for y in 0..4 {
            for x in 2..4 {
                src[y * 4 + x] = 1.0;
            }
        }
        let edges = compute_sobel_edges(&src, 4, 4);
        // Edge should be detected in the middle column
        assert!(edges[1 * 4 + 2] > 0.0 || edges[2 * 4 + 2] > 0.0);
    }

    #[test]
    fn test_frequency_upscaler_output_size() {
        // Use very small image to keep DFT tractable
        let src = vec![0.5f32; 4]; // 2x2
        let dst = FrequencyUpscaler::upscale(&src, 2, 2, 2);
        assert_eq!(dst.len(), 16); // 4x4
    }

    #[test]
    fn test_frequency_upscaler_empty() {
        let dst = FrequencyUpscaler::upscale(&[], 0, 0, 2);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_sr_quality_estimate_perfect() {
        let img = vec![0.5f32; 16];
        let upscaled = vec![0.5f32; 64];
        let q = SrQualityEstimate::compute(&img, &upscaled, 2);
        assert!(q.psnr_db_estimate > 30.0);
    }

    #[test]
    fn test_bicubic_upscale_size() {
        let src = vec![0.5f32; 16];
        let dst = bicubic_upscale(&src, 4, 4, 8, 8);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_catmull_rom_at_zero() {
        let w = catmull_rom_weights(0.0);
        // At t=0 the weight for the second control point should be 1.0
        assert!((w[1] - 1.0).abs() < 1e-5);
    }

    // ── SuperResolutionEngine tests ──────────────────────────────────────

    use super::{SuperResAlgorithm, SuperResError, SuperResolutionConfig, SuperResolutionEngine};

    fn make_engine(algo: SuperResAlgorithm, scale: u32) -> SuperResolutionEngine {
        SuperResolutionEngine::new(SuperResolutionConfig {
            scale_factor: scale,
            algorithm: algo,
            sharpening_strength: 0.5,
        })
    }

    #[test]
    fn test_bicubic_sharp_output_size_2x() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 2);
        let src = vec![128u8; 16]; // 4×4
        let (dst, w, h) = engine.upscale(&src, 4, 4).expect("upscale");
        assert_eq!((w, h), (8, 8));
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_lanczos3_sharp_output_size_2x() {
        let engine = make_engine(SuperResAlgorithm::Lanczos3Sharp, 2);
        let src = vec![200u8; 25]; // 5×5
        let (dst, w, h) = engine.upscale(&src, 5, 5).expect("upscale");
        assert_eq!((w, h), (10, 10));
        assert_eq!(dst.len(), 100);
    }

    #[test]
    fn test_edsr_output_size_2x() {
        let engine = make_engine(SuperResAlgorithm::Edsr, 2);
        let src = vec![64u8; 36]; // 6×6
        let (dst, w, h) = engine.upscale(&src, 6, 6).expect("upscale");
        assert_eq!((w, h), (12, 12));
        assert_eq!(dst.len(), 144);
    }

    #[test]
    fn test_bicubic_sharp_output_size_4x() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 4);
        let src = vec![100u8; 9]; // 3×3
        let (dst, w, h) = engine.upscale(&src, 3, 3).expect("upscale");
        assert_eq!((w, h), (12, 12));
        assert_eq!(dst.len(), 144);
    }

    #[test]
    fn test_uniform_input_produces_uniform_output_bicubic() {
        // A perfectly uniform grey image should remain uniform (or very close)
        // after upscaling, since bicubic of a constant is a constant.
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 2);
        let gray = 128u8;
        let src = vec![gray; 16];
        let (dst, _, _) = engine.upscale(&src, 4, 4).expect("upscale");
        for &v in &dst {
            let diff = (v as i16 - gray as i16).abs();
            assert!(
                diff <= 2,
                "uniform input should produce ~uniform output, got {v}"
            );
        }
    }

    #[test]
    fn test_uniform_input_produces_uniform_output_edsr() {
        let engine = make_engine(SuperResAlgorithm::Edsr, 2);
        let gray = 200u8;
        let src = vec![gray; 16];
        let (dst, _, _) = engine.upscale(&src, 4, 4).expect("upscale");
        for &v in &dst {
            let diff = (v as i16 - gray as i16).abs();
            assert!(
                diff <= 3,
                "EDSR uniform input should produce ~uniform output, got {v}"
            );
        }
    }

    #[test]
    fn test_empty_input_returns_error() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 2);
        // 0×0 → ZeroDimension error
        let err = engine.upscale(&[], 0, 0).unwrap_err();
        assert_eq!(err, SuperResError::ZeroDimension);
    }

    #[test]
    fn test_dimension_mismatch_returns_error() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 2);
        // Claim 4×4 but provide only 10 bytes
        let err = engine.upscale(&vec![0u8; 10], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            SuperResError::DimensionMismatch {
                expected: 16,
                got: 10
            }
        ));
    }

    #[test]
    fn test_scale_factor_one_is_identity() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 1);
        let src: Vec<u8> = (0u8..16).collect();
        let (dst, w, h) = engine.upscale(&src, 4, 4).expect("upscale");
        assert_eq!((w, h), (4, 4));
        assert_eq!(dst, src);
    }

    #[test]
    fn test_zero_scale_factor_returns_error() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 0);
        let err = engine.upscale(&vec![128u8; 16], 4, 4).unwrap_err();
        assert_eq!(err, SuperResError::ZeroScaleFactor);
    }

    #[test]
    fn test_bicubic_sharp_values_in_range() {
        let engine = make_engine(SuperResAlgorithm::BicubicSharp, 2);
        // Gradient image: left half dark, right half bright
        let mut src = vec![0u8; 64]; // 8×8
        for y in 0..8usize {
            for x in 4..8usize {
                src[y * 8 + x] = 255;
            }
        }
        let (dst, _, _) = engine.upscale(&src, 8, 8).expect("upscale");
        for &v in &dst {
            // All output values must be valid u8 (already guaranteed by type,
            // but also numerically in [0, 255])
            let _ = v; // value is u8, always in range
        }
        // Output must be non-empty
        assert!(!dst.is_empty());
    }

    #[test]
    fn test_edsr_preserves_edge_contrast() {
        // A hard left-dark / right-bright edge.  After 2× upscaling, the
        // difference between the darkest and brightest output pixels should be
        // substantial, proving that EDSR does not blur the edge to nothing.
        let engine = make_engine(SuperResAlgorithm::Edsr, 2);
        let mut src = vec![0u8; 64]; // 8×8
        for y in 0..8usize {
            for x in 4..8usize {
                src[y * 8 + x] = 255;
            }
        }
        let (dst, _, _) = engine.upscale(&src, 8, 8).expect("upscale");
        let min_val = dst.iter().copied().min().unwrap_or(0);
        let max_val = dst.iter().copied().max().unwrap_or(0);
        assert!(
            max_val as i16 - min_val as i16 > 100,
            "EDSR should preserve edge contrast; min={min_val} max={max_val}"
        );
    }

    #[test]
    fn test_lanczos3_weight_at_zero_is_one() {
        assert!((super::lanczos3_weight(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lanczos3_weight_at_boundary_is_zero() {
        assert!(super::lanczos3_weight(3.0).abs() < 1e-5);
        assert!(super::lanczos3_weight(-3.0).abs() < 1e-5);
        assert!(super::lanczos3_weight(4.0).abs() < 1e-5);
    }

    #[test]
    fn test_lanczos3_sharp_values_in_range() {
        let engine = make_engine(SuperResAlgorithm::Lanczos3Sharp, 2);
        let src: Vec<u8> = (0u8..=255).cycle().take(64).collect();
        let (dst, w, h) = engine.upscale(&src, 8, 8).expect("upscale");
        assert_eq!((w, h), (16, 16));
        assert_eq!(dst.len(), 256);
        // All pixel values must remain in [0, 255] — guaranteed by u8 type,
        // verified here to be non-empty and correctly sized.
        assert!(!dst.is_empty());
    }

    #[test]
    fn test_super_res_config_default() {
        let cfg = SuperResolutionConfig::default();
        assert_eq!(cfg.scale_factor, 2);
        assert_eq!(cfg.algorithm, SuperResAlgorithm::Edsr);
        assert!((cfg.sharpening_strength - 0.5).abs() < 1e-6);
    }
}
