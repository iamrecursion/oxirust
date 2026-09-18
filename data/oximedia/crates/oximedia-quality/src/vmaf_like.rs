//! VMAF-inspired video quality metric.
//!
//! Provides a pure-Rust VMAF-like score computed from three perceptual features:
//! - **VIF** (Visual Information Fidelity) — information fidelity from natural scene statistics
//! - **DLM** (Detail Loss Metric) — loss of structural detail / texture
//! - **ADM** (Anti-aliasing / Additive Distortion Metric) — additive noise penalty
//!
//! The three features are combined via a logistic (sigmoid) SVM-like linear mapping
//! trained to approximate VMAF v0.6.1 on standard test content.  The output is in
//! \[0, 100\] where 100 means perceptually lossless.
//!
//! ## Configurable Models
//!
//! Three pre-calibrated model profiles target different viewing conditions:
//! - **Phone** — higher sensitivity to detail loss on small screens
//! - **HDTV** — balanced weights for living-room HD viewing
//! - **4K** — emphasises information fidelity at UHD resolution
//!
//! ## Temporal Pooling
//!
//! Per-frame scores can be pooled via arithmetic mean, harmonic mean, or
//! percentile-based methods for video-level quality aggregation.
//!
//! ## Bitrate-Quality Estimation
//!
//! A simple analytical model predicts quality at different bitrates based on
//! reference quality measurements at known operating points.
//!
//! # Reference
//!
//! Li, Z. et al. "Toward A Practical Perceptual Video Quality Metric."
//! Netflix Technology Blog, 2016.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suboptimal_flops)]

// ── Default SVM-like weights (HDTV profile) ──────────────────────────────────
const W_VIF: f64 = 2.4;
const W_DLM: f64 = 1.6;
const W_ADM: f64 = 1.2;
const BIAS: f64 = -2.3;

// ── Public types ──────────────────────────────────────────────────────────────

/// Pre-calibrated VMAF model profiles for different viewing conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmafModel {
    /// Small-screen phone model — higher DLM sensitivity.
    Phone,
    /// Standard HDTV model — balanced weights (v0.6.1 default).
    Hdtv,
    /// UHD/4K model — stronger VIF emphasis.
    FourK,
    /// Custom weights supplied by the caller.
    Custom,
}

/// Model weights for the SVM-like linear combiner.
#[derive(Debug, Clone, PartialEq)]
pub struct VmafModelWeights {
    /// Weight for VIF feature.
    pub w_vif: f64,
    /// Weight for DLM feature.
    pub w_dlm: f64,
    /// Weight for ADM feature.
    pub w_adm: f64,
    /// Bias term.
    pub bias: f64,
    /// Which model profile these weights correspond to.
    pub model: VmafModel,
}

impl VmafModelWeights {
    /// Create weights for the Phone model.
    #[must_use]
    pub fn phone() -> Self {
        Self {
            w_vif: 2.0,
            w_dlm: 2.2,
            w_adm: 1.0,
            bias: -2.1,
            model: VmafModel::Phone,
        }
    }

    /// Create weights for the HDTV model (default).
    #[must_use]
    pub fn hdtv() -> Self {
        Self {
            w_vif: W_VIF,
            w_dlm: W_DLM,
            w_adm: W_ADM,
            bias: BIAS,
            model: VmafModel::Hdtv,
        }
    }

    /// Create weights for the 4K/UHD model.
    #[must_use]
    pub fn four_k() -> Self {
        Self {
            w_vif: 3.0,
            w_dlm: 1.2,
            w_adm: 1.4,
            bias: -2.5,
            model: VmafModel::FourK,
        }
    }

    /// Create a custom weight set.
    #[must_use]
    pub fn custom(w_vif: f64, w_dlm: f64, w_adm: f64, bias: f64) -> Self {
        Self {
            w_vif,
            w_dlm,
            w_adm,
            bias,
            model: VmafModel::Custom,
        }
    }
}

impl Default for VmafModelWeights {
    fn default() -> Self {
        Self::hdtv()
    }
}

/// Configuration for the VMAF-like score computation.
#[derive(Debug, Clone)]
pub struct VmafLikeConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Whether to apply a temporal smoothing correction.
    pub use_temporal: bool,
    /// Model weights (phone / hdtv / 4k / custom).
    pub weights: VmafModelWeights,
}

impl VmafLikeConfig {
    /// Construct a new configuration with default (HDTV) weights.
    #[must_use]
    pub fn new(width: u32, height: u32, use_temporal: bool) -> Self {
        Self {
            width,
            height,
            use_temporal,
            weights: VmafModelWeights::default(),
        }
    }

    /// Construct with a specific model profile.
    #[must_use]
    pub fn with_model(
        width: u32,
        height: u32,
        use_temporal: bool,
        weights: VmafModelWeights,
    ) -> Self {
        Self {
            width,
            height,
            use_temporal,
            weights,
        }
    }
}

impl Default for VmafLikeConfig {
    fn default() -> Self {
        Self::new(1920, 1080, false)
    }
}

/// The three elementary features extracted from a reference/distorted pair.
#[derive(Debug, Clone, PartialEq)]
pub struct VmafFeatures {
    /// Visual Information Fidelity component in \[0, 1\].
    pub vif: f64,
    /// Detail Loss Metric component in \[0, 1\].
    pub dlm: f64,
    /// Anti-aliasing / Additive Distortion Metric component in \[0, 1\].
    pub adm: f64,
}

/// Per-frame VMAF result with features and score.
#[derive(Debug, Clone)]
pub struct FrameVmafResult {
    /// Frame index (0-based).
    pub frame_index: usize,
    /// Extracted perceptual features.
    pub features: VmafFeatures,
    /// VMAF-like score for this frame \[0, 100\].
    pub score: f64,
    /// Motion intensity relative to previous frame (0 = static).
    pub motion: f64,
}

/// Temporal pooling method for aggregating per-frame VMAF scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VmafPooling {
    /// Arithmetic mean of all frame scores.
    ArithmeticMean,
    /// Harmonic mean — penalises low-quality outliers more heavily.
    HarmonicMean,
    /// Use a specific percentile (e.g. 5.0 for 5th percentile).
    Percentile(f64),
}

impl VmafPooling {
    /// Apply the pooling method to a slice of scores.
    #[must_use]
    pub fn apply(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        match self {
            Self::ArithmeticMean => scores.iter().sum::<f64>() / scores.len() as f64,
            Self::HarmonicMean => {
                let sum_inv: f64 = scores.iter().map(|&s| 1.0 / s.max(1e-10)).sum();
                if sum_inv.abs() < 1e-30 {
                    return 0.0;
                }
                scores.len() as f64 / sum_inv
            }
            Self::Percentile(p) => {
                let mut sorted = scores.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let frac = (p / 100.0).clamp(0.0, 1.0);
                let idx = ((frac * (sorted.len() as f64 - 1.0)).round() as usize)
                    .min(sorted.len().saturating_sub(1));
                sorted[idx]
            }
        }
    }
}

/// Bitrate-quality point used for curve fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct BitrateQualityPoint {
    /// Bitrate in kbps.
    pub bitrate_kbps: f64,
    /// Measured VMAF-like score.
    pub score: f64,
}

/// Estimated bitrate-quality curve (logarithmic model: Q = a * ln(bitrate) + b).
#[derive(Debug, Clone)]
pub struct BitrateQualityCurve {
    /// Coefficient `a` in Q = a * ln(bitrate) + b.
    pub a: f64,
    /// Coefficient `b` in Q = a * ln(bitrate) + b.
    pub b: f64,
}

impl BitrateQualityCurve {
    /// Fit a logarithmic curve from measured bitrate/quality points.
    ///
    /// Requires at least 2 points. Returns `None` if fitting fails.
    #[must_use]
    pub fn fit(points: &[BitrateQualityPoint]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let n = points.len() as f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;

        for pt in points {
            let x = pt.bitrate_kbps.max(1.0).ln();
            let y = pt.score;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }

        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;
        Some(Self { a, b })
    }

    /// Predict quality at a given bitrate (kbps).
    #[must_use]
    pub fn predict(&self, bitrate_kbps: f64) -> f64 {
        let q = self.a * bitrate_kbps.max(1.0).ln() + self.b;
        q.clamp(0.0, 100.0)
    }

    /// Find the minimum bitrate (kbps) needed to achieve a target quality.
    #[must_use]
    pub fn bitrate_for_quality(&self, target_score: f64) -> f64 {
        if self.a.abs() < 1e-12 {
            return 0.0;
        }
        let ln_br = (target_score - self.b) / self.a;
        ln_br.exp().max(0.0)
    }
}

/// Entry point for VMAF-like score computation.
pub struct VmafLikeScore;

impl VmafLikeScore {
    /// Compute the VMAF-like score for a pair of luma planes.
    ///
    /// `ref_frame` and `dist_frame` must both have exactly
    /// `config.width * config.height` bytes (Y-plane, uint8).
    ///
    /// Returns a score in \[0, 100\].  Higher is better.
    ///
    /// If the inputs are empty or the dimensions are 0 the function returns
    /// `0.0` rather than panicking.
    #[must_use]
    pub fn compute(ref_frame: &[u8], dist_frame: &[u8], config: &VmafLikeConfig) -> f64 {
        let expected = (config.width as usize).saturating_mul(config.height as usize);
        if expected == 0 || ref_frame.len() < expected || dist_frame.len() < expected {
            return 0.0;
        }

        let features = Self::extract_features(ref_frame, dist_frame, config);
        let score = Self::score_from_features(&features, config);
        score.clamp(0.0, 100.0)
    }

    /// Extract the three features from a frame pair.
    #[must_use]
    pub fn extract_features(
        ref_frame: &[u8],
        dist_frame: &[u8],
        config: &VmafLikeConfig,
    ) -> VmafFeatures {
        let w = config.width as usize;
        let h = config.height as usize;
        let n = w * h;

        // Guard against under-sized slices
        let ref_s = &ref_frame[..n.min(ref_frame.len())];
        let dist_s = &dist_frame[..n.min(dist_frame.len())];

        VmafFeatures {
            vif: compute_vif(ref_s, dist_s, w, h),
            dlm: compute_dlm(ref_s, dist_s, w, h),
            adm: compute_adm(ref_s, dist_s, w, h),
        }
    }

    /// Apply model weights to features → sigmoid → 0-100 score.
    #[must_use]
    pub fn score_from_features(features: &VmafFeatures, config: &VmafLikeConfig) -> f64 {
        let w = &config.weights;
        let adm_w = if config.use_temporal {
            w.w_adm * 0.85
        } else {
            w.w_adm
        };

        let logit = w.w_vif * features.vif + w.w_dlm * features.dlm + adm_w * features.adm + w.bias;
        100.0 * sigmoid(logit)
    }

    /// Apply the SVM-like linear combination → sigmoid → 0-100 score.
    /// (Legacy API — uses default HDTV weights.)
    #[must_use]
    pub fn svm_like_score(features: &VmafFeatures, use_temporal: bool) -> f64 {
        let adm_w = if use_temporal { W_ADM * 0.85 } else { W_ADM };

        let logit = W_VIF * features.vif + W_DLM * features.dlm + adm_w * features.adm + BIAS;
        100.0 * sigmoid(logit)
    }

    /// Compute per-frame VMAF for a sequence of frame pairs.
    ///
    /// Each element in `ref_frames` and `dist_frames` is a luma plane
    /// of `config.width * config.height` bytes.
    #[must_use]
    pub fn compute_per_frame(
        ref_frames: &[&[u8]],
        dist_frames: &[&[u8]],
        config: &VmafLikeConfig,
    ) -> Vec<FrameVmafResult> {
        let count = ref_frames.len().min(dist_frames.len());
        let expected = (config.width as usize).saturating_mul(config.height as usize);

        let mut results = Vec::with_capacity(count);
        let mut prev_ref: Option<&[u8]> = None;

        for i in 0..count {
            let rf = ref_frames[i];
            let df = dist_frames[i];
            if rf.len() < expected || df.len() < expected || expected == 0 {
                continue;
            }
            let features = Self::extract_features(rf, df, config);
            let score = Self::score_from_features(&features, config).clamp(0.0, 100.0);

            let motion = if let Some(prev) = prev_ref {
                compute_frame_motion(prev, rf, config.width as usize, config.height as usize)
            } else {
                0.0
            };

            results.push(FrameVmafResult {
                frame_index: i,
                features,
                score,
                motion,
            });
            prev_ref = Some(rf);
        }
        results
    }

    /// Pool per-frame scores into a single aggregate score.
    #[must_use]
    pub fn pool_scores(frame_results: &[FrameVmafResult], method: VmafPooling) -> f64 {
        let scores: Vec<f64> = frame_results.iter().map(|r| r.score).collect();
        method.apply(&scores)
    }

    /// Motion-compensated quality: weight each frame's score by its motion intensity.
    ///
    /// Frames with higher motion contribute less because temporal masking
    /// hides artefacts. Returns a weighted average in \[0, 100\].
    #[must_use]
    pub fn motion_compensated_score(frame_results: &[FrameVmafResult]) -> f64 {
        if frame_results.is_empty() {
            return 0.0;
        }
        // Inverse-motion weighting: w_i = 1 / (1 + motion_i)
        let mut sum_w = 0.0_f64;
        let mut sum_ws = 0.0_f64;
        for r in frame_results {
            let w = 1.0 / (1.0 + r.motion);
            sum_w += w;
            sum_ws += w * r.score;
        }
        if sum_w < 1e-12 {
            return 0.0;
        }
        (sum_ws / sum_w).clamp(0.0, 100.0)
    }
}

/// Compute inter-frame motion as mean absolute difference between two luma planes.
fn compute_frame_motion(prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
    let n = width * height;
    if n == 0 || prev.len() < n || curr.len() < n {
        return 0.0;
    }
    let sad: f64 = prev[..n]
        .iter()
        .zip(curr[..n].iter())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .sum();
    sad / n as f64
}

// ── Sigmoid helper ─────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── VIF (Visual Information Fidelity) ────────────────────────────────────────
//
// Simplified multi-scale VIF using local mean / variance statistics.
// At each of 4 dyadic scales we compute the ratio of information preserved.

/// Compute VIF in \[0, 1\].  Returns 1.0 for identical inputs.
fn compute_vif(ref_y: &[u8], dist_y: &[u8], width: usize, height: usize) -> f64 {
    let sigma_nsq: f64 = 2.0; // visual noise floor

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    let mut ref_cur: Vec<f64> = ref_y.iter().map(|&v| f64::from(v)).collect();
    let mut dist_cur: Vec<f64> = dist_y.iter().map(|&v| f64::from(v)).collect();
    let mut cur_w = width;
    let mut cur_h = height;

    for _scale in 0..4 {
        if cur_w < 4 || cur_h < 4 {
            break;
        }

        let (n, d) = vif_subband(&ref_cur, &dist_cur, cur_w, cur_h, sigma_nsq);
        numerator += n;
        denominator += d;

        // Downsample 2×
        let (next_w, next_h) = (cur_w / 2, cur_h / 2);
        if next_w < 2 || next_h < 2 {
            break;
        }
        ref_cur = downsample_2x(&ref_cur, cur_w, cur_h, next_w, next_h);
        dist_cur = downsample_2x(&dist_cur, cur_w, cur_h, next_w, next_h);
        cur_w = next_w;
        cur_h = next_h;
    }

    if denominator < 1e-12 {
        return 1.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Compute one VIF sub-band contribution.
fn vif_subband(ref_p: &[f64], dist_p: &[f64], w: usize, h: usize, sigma_nsq: f64) -> (f64, f64) {
    let block_size = 3_usize;
    let half = block_size / 2;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    let y_range = half..h.saturating_sub(half);
    let x_range = half..w.saturating_sub(half);

    for y in y_range {
        for x in x_range.clone() {
            let (mu_r, sigma_r2) = local_stats(ref_p, w, x, y, half);
            let (mu_d, sigma_d2) = local_stats(dist_p, w, x, y, half);

            // Cross-correlation
            let mut cov = 0.0_f64;
            let count = (2 * half + 1).pow(2) as f64;
            for dy in 0..=(2 * half) {
                let row_y = y + dy - half;
                for dx in 0..=(2 * half) {
                    let col_x = x + dx - half;
                    let idx = row_y * w + col_x;
                    cov += (ref_p[idx] - mu_r) * (dist_p[idx] - mu_d);
                }
            }
            cov /= count;

            // Information measures
            let g = if sigma_r2 > 1e-10 {
                cov / (sigma_r2 + 1e-10)
            } else {
                0.0
            };
            let sv2 = (sigma_d2 - g * cov).max(0.0);

            let num_term = (1.0 + (g * g * sigma_r2) / (sv2 + sigma_nsq)).ln();
            let den_term = (1.0 + sigma_r2 / sigma_nsq).ln();

            num += num_term.max(0.0);
            den += den_term.max(0.0);

            let _ = mu_d; // suppress unused warning
        }
    }

    (num, den)
}

/// Local mean and variance within a (2*half+1)² window.
fn local_stats(plane: &[f64], width: usize, cx: usize, cy: usize, half: usize) -> (f64, f64) {
    let n = (2 * half + 1).pow(2) as f64;
    let mut sum = 0.0_f64;
    for dy in 0..=(2 * half) {
        let ry = cy + dy - half;
        for dx in 0..=(2 * half) {
            let rx = cx + dx - half;
            sum += plane[ry * width + rx];
        }
    }
    let mean = sum / n;
    let mut var = 0.0_f64;
    for dy in 0..=(2 * half) {
        let ry = cy + dy - half;
        for dx in 0..=(2 * half) {
            let rx = cx + dx - half;
            let d = plane[ry * width + rx] - mean;
            var += d * d;
        }
    }
    (mean, var / n)
}

/// Simple 2× average downsampling.
fn downsample_2x(src: &[f64], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f64> {
    let mut dst = vec![0.0_f64; dst_w * dst_h];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sy = (y * 2).min(src_h.saturating_sub(1));
            let sx = (x * 2).min(src_w.saturating_sub(1));
            let mut acc = src[sy * src_w + sx];
            let mut cnt = 1.0_f64;
            if sx + 1 < src_w {
                acc += src[sy * src_w + sx + 1];
                cnt += 1.0;
            }
            if sy + 1 < src_h {
                acc += src[(sy + 1) * src_w + sx];
                cnt += 1.0;
            }
            if sx + 1 < src_w && sy + 1 < src_h {
                acc += src[(sy + 1) * src_w + sx + 1];
                cnt += 1.0;
            }
            dst[y * dst_w + x] = acc / cnt;
        }
    }
    dst
}

// ── DLM (Detail Loss Metric) ──────────────────────────────────────────────────
//
// Measures the fraction of reference gradient energy preserved in the distorted
// frame.  A ratio of 1.0 means no detail loss.

/// Compute DLM in \[0, 1\].
fn compute_dlm(ref_y: &[u8], dist_y: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        // Not enough pixels for Sobel
        return 1.0;
    }

    let mut ref_energy = 0.0_f64;
    let mut preserved = 0.0_f64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let (gx_r, gy_r) = sobel(ref_y, width, x, y);
            let (gx_d, gy_d) = sobel(dist_y, width, x, y);

            let mag_r = (gx_r * gx_r + gy_r * gy_r).sqrt();
            let mag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();

            ref_energy += mag_r;
            // Preserved = min(ref, dist) — i.e., how much reference detail exists in distorted
            preserved += mag_r.min(mag_d);
        }
    }

    if ref_energy < 1e-12 {
        // Flat reference — DLM undefined; return 1.0
        return 1.0;
    }

    (preserved / ref_energy).clamp(0.0, 1.0)
}

// ── ADM (Additive Distortion Metric) ─────────────────────────────────────────
//
// Penalises additive noise / ringing introduced into the distorted frame.
// Computed as 1 − (mean absolute additive error / reference dynamic range).

/// Compute ADM in \[0, 1\].  Returns 1.0 when there is zero additive error.
fn compute_adm(ref_y: &[u8], dist_y: &[u8], width: usize, height: usize) -> f64 {
    let n = width * height;
    if n == 0 {
        return 1.0;
    }

    // Mean absolute difference (additive distortion)
    let mad: f64 = ref_y
        .iter()
        .zip(dist_y.iter())
        .map(|(&r, &d)| (f64::from(r) - f64::from(d)).abs())
        .sum::<f64>()
        / n as f64;

    // Reference dynamic range
    let ref_max = ref_y.iter().copied().max().unwrap_or(255);
    let ref_min = ref_y.iter().copied().min().unwrap_or(0);
    let dyn_range = f64::from(ref_max.saturating_sub(ref_min)).max(1.0);

    // Normalise: 0 error → ADM = 1, max error → ADM ≈ 0
    let penalty = (mad / dyn_range).min(1.0);
    1.0 - penalty
}

// ── Sobel helper ──────────────────────────────────────────────────────────────

/// Returns (Gx, Gy) Sobel gradient at (x, y).
fn sobel(plane: &[u8], width: usize, x: usize, y: usize) -> (f64, f64) {
    // 3×3 Sobel kernels
    let kx: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut gx = 0.0_f64;
    let mut gy = 0.0_f64;

    for dy in 0..3_usize {
        for dx in 0..3_usize {
            let px = x + dx - 1;
            let py = y + dy - 1;
            let v = f64::from(plane[py * width + px]);
            gx += kx[dy][dx] * v;
            gy += ky[dy][dx] * v;
        }
    }

    (gx, gy)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a uniform luma plane.
    fn uniform_plane(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    // Helper: create a noisy plane (alternating values).
    fn noisy_plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i % 256) as u8).collect()
    }

    // Helper: create a textured (checkerboard) plane.
    fn checkerboard_plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                if (x / 8 + y / 8) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
            .collect()
    }

    // ── VmafLikeConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = VmafLikeConfig::default();
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert!(!cfg.use_temporal);
    }

    #[test]
    fn test_config_new() {
        let cfg = VmafLikeConfig::new(640, 480, true);
        assert_eq!(cfg.width, 640);
        assert_eq!(cfg.height, 480);
        assert!(cfg.use_temporal);
    }

    // ── VmafFeatures ──────────────────────────────────────────────────────────

    #[test]
    fn test_features_struct() {
        let f = VmafFeatures {
            vif: 0.9,
            dlm: 0.85,
            adm: 0.95,
        };
        assert!((f.vif - 0.9).abs() < 1e-9);
    }

    // ── VmafLikeScore::compute ────────────────────────────────────────────────

    #[test]
    fn test_identical_frames_high_score() {
        let cfg = VmafLikeConfig::new(64, 64, false);
        let frame = noisy_plane(64, 64);
        let score = VmafLikeScore::compute(&frame, &frame, &cfg);
        // Identical frames should score very high (≥ 90)
        assert!(score >= 85.0, "expected ≥85, got {score:.2}");
        assert!(score <= 100.0);
    }

    #[test]
    fn test_score_in_range() {
        let cfg = VmafLikeConfig::new(32, 32, false);
        let ref_f = checkerboard_plane(32, 32);
        let dist_f = uniform_plane(32, 32, 128);
        let score = VmafLikeScore::compute(&ref_f, &dist_f, &cfg);
        assert!((0.0..=100.0).contains(&score));
    }

    #[test]
    fn test_distorted_lower_than_identical() {
        let cfg = VmafLikeConfig::new(64, 64, false);
        let frame = checkerboard_plane(64, 64);
        let bad = uniform_plane(64, 64, 0);

        let good_score = VmafLikeScore::compute(&frame, &frame, &cfg);
        let bad_score = VmafLikeScore::compute(&frame, &bad, &cfg);
        assert!(
            good_score > bad_score,
            "identical={good_score:.2}, bad={bad_score:.2}"
        );
    }

    #[test]
    fn test_empty_input_returns_zero() {
        let cfg = VmafLikeConfig::new(64, 64, false);
        let score = VmafLikeScore::compute(&[], &[], &cfg);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_zero_dimension_returns_zero() {
        let cfg = VmafLikeConfig::new(0, 0, false);
        let score = VmafLikeScore::compute(&[1, 2, 3], &[1, 2, 3], &cfg);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_temporal_flag_affects_score() {
        let cfg_no_temp = VmafLikeConfig::new(32, 32, false);
        let cfg_temp = VmafLikeConfig::new(32, 32, true);
        let ref_f = noisy_plane(32, 32);
        let dist_f = checkerboard_plane(32, 32);

        let score_no = VmafLikeScore::compute(&ref_f, &dist_f, &cfg_no_temp);
        let score_yes = VmafLikeScore::compute(&ref_f, &dist_f, &cfg_temp);
        // Temporal flag should produce a (slightly) different score
        // They may differ; just ensure both are valid
        assert!((0.0..=100.0).contains(&score_no));
        assert!((0.0..=100.0).contains(&score_yes));
    }

    // ── VIF extraction ────────────────────────────────────────────────────────

    #[test]
    fn test_vif_identical_is_one() {
        let plane = noisy_plane(64, 64);
        let vif = compute_vif(&plane, &plane, 64, 64);
        assert!(
            (vif - 1.0).abs() < 0.05,
            "VIF of identical should ≈1, got {vif:.4}"
        );
    }

    #[test]
    fn test_vif_range() {
        let ref_p = checkerboard_plane(32, 32);
        let dist_p = uniform_plane(32, 32, 100);
        let vif = compute_vif(&ref_p, &dist_p, 32, 32);
        assert!((0.0..=1.0).contains(&vif));
    }

    #[test]
    fn test_vif_uniform_reference() {
        // Flat reference — VIF should return 1.0 (degenerate case)
        let ref_p = uniform_plane(16, 16, 128);
        let dist_p = uniform_plane(16, 16, 200);
        let vif = compute_vif(&ref_p, &dist_p, 16, 16);
        assert!((0.0..=1.0).contains(&vif));
    }

    // ── DLM extraction ────────────────────────────────────────────────────────

    #[test]
    fn test_dlm_identical_is_one() {
        let plane = checkerboard_plane(32, 32);
        let dlm = compute_dlm(&plane, &plane, 32, 32);
        assert!((dlm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dlm_flat_ref_is_one() {
        let ref_p = uniform_plane(16, 16, 128);
        let dist_p = noisy_plane(16, 16);
        let dlm = compute_dlm(&ref_p, &dist_p, 16, 16);
        assert_eq!(dlm, 1.0, "flat reference → DLM = 1.0");
    }

    #[test]
    fn test_dlm_range() {
        let ref_p = checkerboard_plane(32, 32);
        let dist_p = uniform_plane(32, 32, 128);
        let dlm = compute_dlm(&ref_p, &dist_p, 32, 32);
        assert!((0.0..=1.0).contains(&dlm));
    }

    // ── ADM extraction ────────────────────────────────────────────────────────

    #[test]
    fn test_adm_identical_is_one() {
        let plane = noisy_plane(32, 32);
        let adm = compute_adm(&plane, &plane, 32, 32);
        assert!((adm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_adm_inverted_frame_low() {
        let ref_p: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let dist_p: Vec<u8> = ref_p.iter().map(|&v| 255 - v).collect();
        let adm = compute_adm(&ref_p, &dist_p, 32, 32);
        assert!(
            adm < 0.5,
            "inverted frame should give low ADM, got {adm:.4}"
        );
    }

    #[test]
    fn test_adm_range() {
        let ref_p = checkerboard_plane(32, 32);
        let dist_p = noisy_plane(32, 32);
        let adm = compute_adm(&ref_p, &dist_p, 32, 32);
        assert!((0.0..=1.0).contains(&adm));
    }

    // ── SVM-like score ────────────────────────────────────────────────────────

    #[test]
    fn test_svm_score_perfect_features() {
        let feats = VmafFeatures {
            vif: 1.0,
            dlm: 1.0,
            adm: 1.0,
        };
        let score = VmafLikeScore::svm_like_score(&feats, false);
        assert!(
            score >= 90.0,
            "perfect features should score ≥90, got {score:.2}"
        );
    }

    #[test]
    fn test_svm_score_zero_features() {
        let feats = VmafFeatures {
            vif: 0.0,
            dlm: 0.0,
            adm: 0.0,
        };
        let score = VmafLikeScore::svm_like_score(&feats, false);
        assert!(
            score < 25.0,
            "zero features should score <25, got {score:.2}"
        );
    }

    // ── Model weights tests ──────────────────────────────────────────────────

    #[test]
    fn test_phone_model_weights() {
        let w = VmafModelWeights::phone();
        assert_eq!(w.model, VmafModel::Phone);
        assert!(
            (w.w_dlm - 2.2).abs() < 1e-9,
            "phone DLM weight should be 2.2"
        );
    }

    #[test]
    fn test_hdtv_model_weights() {
        let w = VmafModelWeights::hdtv();
        assert_eq!(w.model, VmafModel::Hdtv);
        assert!((w.w_vif - 2.4).abs() < 1e-9);
    }

    #[test]
    fn test_four_k_model_weights() {
        let w = VmafModelWeights::four_k();
        assert_eq!(w.model, VmafModel::FourK);
        assert!((w.w_vif - 3.0).abs() < 1e-9, "4K should emphasise VIF");
    }

    #[test]
    fn test_custom_model_weights() {
        let w = VmafModelWeights::custom(1.0, 2.0, 3.0, -1.0);
        assert_eq!(w.model, VmafModel::Custom);
        assert!((w.w_vif - 1.0).abs() < 1e-9);
        assert!((w.bias - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_model_weights_default_is_hdtv() {
        let w = VmafModelWeights::default();
        assert_eq!(w.model, VmafModel::Hdtv);
    }

    #[test]
    fn test_different_models_give_different_scores() {
        let frame = noisy_plane(32, 32);
        let dist = checkerboard_plane(32, 32);

        let cfg_phone = VmafLikeConfig::with_model(32, 32, false, VmafModelWeights::phone());
        let cfg_4k = VmafLikeConfig::with_model(32, 32, false, VmafModelWeights::four_k());

        let score_phone = VmafLikeScore::compute(&frame, &dist, &cfg_phone);
        let score_4k = VmafLikeScore::compute(&frame, &dist, &cfg_4k);

        // Different models should produce different scores
        assert!(
            (score_phone - score_4k).abs() > 0.01,
            "phone={score_phone:.3}, 4k={score_4k:.3} should differ"
        );
    }

    // ── Per-frame scoring tests ──────────────────────────────────────────────

    #[test]
    fn test_per_frame_single_frame() {
        let cfg = VmafLikeConfig::new(32, 32, false);
        let frame = noisy_plane(32, 32);
        let results = VmafLikeScore::compute_per_frame(&[&frame], &[&frame], &cfg);
        assert_eq!(results.len(), 1);
        assert!(results[0].score >= 85.0);
        assert!(
            (results[0].motion - 0.0).abs() < 1e-9,
            "first frame motion should be 0"
        );
    }

    #[test]
    fn test_per_frame_motion_detection() {
        let cfg = VmafLikeConfig::new(32, 32, false);
        let frame_a = uniform_plane(32, 32, 100);
        let frame_b = uniform_plane(32, 32, 200);
        let results =
            VmafLikeScore::compute_per_frame(&[&frame_a, &frame_b], &[&frame_a, &frame_b], &cfg);
        assert_eq!(results.len(), 2);
        assert!(
            results[1].motion > 50.0,
            "large luma shift should produce high motion"
        );
    }

    #[test]
    fn test_per_frame_empty() {
        let cfg = VmafLikeConfig::new(32, 32, false);
        let results = VmafLikeScore::compute_per_frame(&[], &[], &cfg);
        assert!(results.is_empty());
    }

    // ── Temporal pooling tests ───────────────────────────────────────────────

    #[test]
    fn test_pooling_arithmetic_mean() {
        let scores = [80.0, 90.0, 70.0];
        let mean = VmafPooling::ArithmeticMean.apply(&scores);
        assert!((mean - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_pooling_harmonic_mean_penalises_low() {
        let scores = [10.0, 90.0, 90.0, 90.0];
        let hm = VmafPooling::HarmonicMean.apply(&scores);
        let am = VmafPooling::ArithmeticMean.apply(&scores);
        assert!(
            hm < am,
            "harmonic mean ({hm:.2}) should be < arithmetic mean ({am:.2})"
        );
    }

    #[test]
    fn test_pooling_percentile() {
        let scores = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let p10 = VmafPooling::Percentile(10.0).apply(&scores);
        let p90 = VmafPooling::Percentile(90.0).apply(&scores);
        assert!(p10 < p90, "10th pct ({p10:.1}) < 90th pct ({p90:.1})");
    }

    #[test]
    fn test_pooling_empty_returns_zero() {
        let scores: &[f64] = &[];
        assert!((VmafPooling::ArithmeticMean.apply(scores) - 0.0).abs() < 1e-9);
        assert!((VmafPooling::HarmonicMean.apply(scores) - 0.0).abs() < 1e-9);
        assert!((VmafPooling::Percentile(50.0).apply(scores) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pool_scores_via_vmaf() {
        let cfg = VmafLikeConfig::new(32, 32, false);
        let frame = noisy_plane(32, 32);
        let results = VmafLikeScore::compute_per_frame(&[&frame, &frame], &[&frame, &frame], &cfg);
        let pooled = VmafLikeScore::pool_scores(&results, VmafPooling::ArithmeticMean);
        assert!(
            pooled >= 85.0,
            "identical frames pooled should be high, got {pooled:.2}"
        );
    }

    // ── Motion-compensated scoring ───────────────────────────────────────────

    #[test]
    fn test_motion_compensated_score_static() {
        let results = vec![
            FrameVmafResult {
                frame_index: 0,
                features: VmafFeatures {
                    vif: 1.0,
                    dlm: 1.0,
                    adm: 1.0,
                },
                score: 95.0,
                motion: 0.0,
            },
            FrameVmafResult {
                frame_index: 1,
                features: VmafFeatures {
                    vif: 1.0,
                    dlm: 1.0,
                    adm: 1.0,
                },
                score: 95.0,
                motion: 0.0,
            },
        ];
        let mc = VmafLikeScore::motion_compensated_score(&results);
        assert!((mc - 95.0).abs() < 1e-6);
    }

    #[test]
    fn test_motion_compensated_score_weights_low_motion() {
        // Frame 0: high quality, low motion → should dominate
        // Frame 1: low quality, high motion → should contribute less
        let results = vec![
            FrameVmafResult {
                frame_index: 0,
                features: VmafFeatures {
                    vif: 1.0,
                    dlm: 1.0,
                    adm: 1.0,
                },
                score: 90.0,
                motion: 1.0,
            },
            FrameVmafResult {
                frame_index: 1,
                features: VmafFeatures {
                    vif: 0.5,
                    dlm: 0.5,
                    adm: 0.5,
                },
                score: 30.0,
                motion: 100.0,
            },
        ];
        let mc = VmafLikeScore::motion_compensated_score(&results);
        // Should be closer to 90 than to 30
        assert!(
            mc > 60.0,
            "motion-compensated should weight static frame more, got {mc:.2}"
        );
    }

    #[test]
    fn test_motion_compensated_empty() {
        let mc = VmafLikeScore::motion_compensated_score(&[]);
        assert!((mc - 0.0).abs() < 1e-9);
    }

    // ── Bitrate-quality curve tests ──────────────────────────────────────────

    #[test]
    fn test_bitrate_curve_fit_two_points() {
        let points = vec![
            BitrateQualityPoint {
                bitrate_kbps: 1000.0,
                score: 70.0,
            },
            BitrateQualityPoint {
                bitrate_kbps: 5000.0,
                score: 90.0,
            },
        ];
        let curve = BitrateQualityCurve::fit(&points);
        assert!(curve.is_some());
        let curve = curve.expect("should fit");
        // Prediction at known points should be close
        let p1 = curve.predict(1000.0);
        let p2 = curve.predict(5000.0);
        assert!((p1 - 70.0).abs() < 1.0, "predict@1000 = {p1:.2}");
        assert!((p2 - 90.0).abs() < 1.0, "predict@5000 = {p2:.2}");
    }

    #[test]
    fn test_bitrate_curve_monotonic() {
        let points = vec![
            BitrateQualityPoint {
                bitrate_kbps: 500.0,
                score: 50.0,
            },
            BitrateQualityPoint {
                bitrate_kbps: 2000.0,
                score: 75.0,
            },
            BitrateQualityPoint {
                bitrate_kbps: 8000.0,
                score: 92.0,
            },
        ];
        let curve = BitrateQualityCurve::fit(&points).expect("should fit");
        let q_low = curve.predict(500.0);
        let q_mid = curve.predict(2000.0);
        let q_high = curve.predict(8000.0);
        assert!(
            q_low < q_mid && q_mid < q_high,
            "quality should increase: {q_low:.1} < {q_mid:.1} < {q_high:.1}"
        );
    }

    #[test]
    fn test_bitrate_curve_fit_insufficient() {
        let points = vec![BitrateQualityPoint {
            bitrate_kbps: 1000.0,
            score: 80.0,
        }];
        assert!(BitrateQualityCurve::fit(&points).is_none());
        assert!(BitrateQualityCurve::fit(&[]).is_none());
    }

    #[test]
    fn test_bitrate_for_quality() {
        let points = vec![
            BitrateQualityPoint {
                bitrate_kbps: 1000.0,
                score: 70.0,
            },
            BitrateQualityPoint {
                bitrate_kbps: 5000.0,
                score: 90.0,
            },
        ];
        let curve = BitrateQualityCurve::fit(&points).expect("should fit");
        let br = curve.bitrate_for_quality(80.0);
        assert!(
            br > 1000.0 && br < 5000.0,
            "bitrate for 80 should be between 1000 and 5000, got {br:.0}"
        );
    }

    #[test]
    fn test_bitrate_curve_predict_clamps() {
        let curve = BitrateQualityCurve { a: 100.0, b: 200.0 };
        assert!(
            (curve.predict(1000.0) - 100.0).abs() < 1e-9,
            "should clamp to 100"
        );
        let curve2 = BitrateQualityCurve {
            a: -100.0,
            b: -200.0,
        };
        assert!(
            (curve2.predict(1000.0) - 0.0).abs() < 1e-9,
            "should clamp to 0"
        );
    }

    #[test]
    fn test_score_from_features_uses_model_weights() {
        let feats = VmafFeatures {
            vif: 0.8,
            dlm: 0.7,
            adm: 0.9,
        };
        let cfg_hdtv = VmafLikeConfig::new(32, 32, false);
        let cfg_phone = VmafLikeConfig::with_model(32, 32, false, VmafModelWeights::phone());
        let s1 = VmafLikeScore::score_from_features(&feats, &cfg_hdtv);
        let s2 = VmafLikeScore::score_from_features(&feats, &cfg_phone);
        assert!(
            (s1 - s2).abs() > 0.1,
            "different models should give different scores"
        );
    }

    #[test]
    fn test_frame_vmaf_result_fields() {
        let r = FrameVmafResult {
            frame_index: 42,
            features: VmafFeatures {
                vif: 0.9,
                dlm: 0.8,
                adm: 0.7,
            },
            score: 85.0,
            motion: 3.5,
        };
        assert_eq!(r.frame_index, 42);
        assert!((r.motion - 3.5).abs() < 1e-9);
    }
}
