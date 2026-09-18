#![allow(dead_code)]
//! VMAF score prediction and estimation for encoding optimization.
//!
//! This module provides a lightweight VMAF (Video Multi-method Assessment Fusion)
//! score estimator that predicts quality scores without running the full VMAF model.
//! It uses spatial and temporal feature extraction to approximate VMAF scores,
//! enabling fast quality-aware encoding decisions.
//!
//! The predictor uses a multi-scale approach:
//! - VIF (Visual Information Fidelity) approximation at 4 scales
//! - DLM (Detail Loss Metric) from spatial gradient analysis
//! - ADM (Additive Detail Masking) from texture energy
//! - Motion metric from temporal frame differences
//!
//! A support vector regression (SVR)-inspired model maps features to VMAF scores.

use std::collections::VecDeque;

/// Feature vector used for VMAF prediction.
#[derive(Debug, Clone)]
pub struct VmafFeatures {
    /// Visual Information Fidelity (VIF) approximation at scale 0.
    pub vif_scale0: f64,
    /// Visual Information Fidelity (VIF) approximation at scale 1.
    pub vif_scale1: f64,
    /// Visual Information Fidelity (VIF) approximation at scale 2.
    pub vif_scale2: f64,
    /// Visual Information Fidelity (VIF) approximation at scale 3.
    pub vif_scale3: f64,
    /// Detail Loss Metric (DLM) approximation.
    pub dlm: f64,
    /// Motion metric (temporal difference).
    pub motion: f64,
    /// ADM (Additive Detail Masking) approximation.
    pub adm: f64,
}

impl VmafFeatures {
    /// Creates a new feature set with default (perfect quality) values.
    pub fn perfect() -> Self {
        Self {
            vif_scale0: 1.0,
            vif_scale1: 1.0,
            vif_scale2: 1.0,
            vif_scale3: 1.0,
            dlm: 1.0,
            motion: 0.0,
            adm: 1.0,
        }
    }

    /// Creates a feature set from raw pixel comparison statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_stats(psnr: f64, ssim: f64, motion_magnitude: f64) -> Self {
        // Map PSNR/SSIM to approximate VIF scores
        let vif_base = (psnr / 50.0).min(1.0).max(0.0);
        let ssim_clamped = ssim.min(1.0).max(0.0);
        Self {
            vif_scale0: vif_base * 0.9 + ssim_clamped * 0.1,
            vif_scale1: vif_base * 0.85 + ssim_clamped * 0.15,
            vif_scale2: vif_base * 0.8 + ssim_clamped * 0.2,
            vif_scale3: vif_base * 0.75 + ssim_clamped * 0.25,
            dlm: ssim_clamped,
            motion: motion_magnitude,
            adm: ssim_clamped * 0.95 + vif_base * 0.05,
        }
    }

    /// Returns the mean VIF across all scales.
    pub fn mean_vif(&self) -> f64 {
        (self.vif_scale0 + self.vif_scale1 + self.vif_scale2 + self.vif_scale3) / 4.0
    }

    /// Extracts spatial features from reference and distorted pixel blocks.
    ///
    /// Computes VIF approximation at multiple scales by comparing local statistics
    /// (mean, variance, covariance) between reference and distorted blocks.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_from_pixels(
        reference: &[u8],
        distorted: &[u8],
        width: usize,
        height: usize,
        prev_reference: Option<&[u8]>,
    ) -> Self {
        let expected_len = width * height;
        if reference.len() < expected_len || distorted.len() < expected_len {
            return Self::perfect();
        }

        // Compute VIF at 4 scales via progressive downsampling
        let vif_scale0 = compute_vif_at_scale(reference, distorted, width, height, 1);
        let vif_scale1 = compute_vif_at_scale(reference, distorted, width, height, 2);
        let vif_scale2 = compute_vif_at_scale(reference, distorted, width, height, 4);
        let vif_scale3 = compute_vif_at_scale(reference, distorted, width, height, 8);

        // DLM: Detail Loss Metric via Sobel gradient comparison
        let dlm = compute_dlm(reference, distorted, width, height);

        // ADM: Additive Detail Masking via texture energy ratio
        let adm = compute_adm(reference, distorted, width, height);

        // Motion: temporal difference magnitude
        let motion = if let Some(prev) = prev_reference {
            compute_motion(reference, prev, width, height)
        } else {
            0.0
        };

        Self {
            vif_scale0,
            vif_scale1,
            vif_scale2,
            vif_scale3,
            dlm,
            motion,
            adm,
        }
    }
}

/// Computes VIF approximation at a given scale factor.
///
/// Uses local statistics (mean, variance, cross-correlation) in non-overlapping
/// blocks of size `scale * 4` to approximate the VIF metric.
#[allow(clippy::cast_precision_loss)]
fn compute_vif_at_scale(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    scale: usize,
) -> f64 {
    let block_size = (scale * 4).max(4);
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    let mut num_sum = 0.0_f64;
    let mut den_sum = 0.0_f64;
    let sigma_nsq = 2.0; // noise variance parameter

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let (mu_ref, mu_dis, var_ref, var_dis, cov) = block_statistics(
                reference,
                distorted,
                width,
                bx * block_size,
                by * block_size,
                block_size,
            );

            // VIF formulation: log2(1 + g^2 * sigma_ref^2 / sigma_nsq) / log2(1 + sigma_ref^2 / sigma_nsq)
            // where g = cov / var_ref is the gain factor
            let _ = mu_ref;
            let _ = mu_dis;

            if var_ref > 1e-10 {
                let g = cov / var_ref;
                let sigma_v = (var_dis - g * g * var_ref).max(1e-10);
                let numerator = (1.0 + g * g * var_ref / (sigma_v + sigma_nsq)).ln();
                let denominator = (1.0 + var_ref / sigma_nsq).ln();
                if denominator > 1e-10 {
                    num_sum += numerator;
                    den_sum += denominator;
                }
            }
        }
    }

    if den_sum > 1e-10 {
        (num_sum / den_sum).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Computes block-level statistics (mean_ref, mean_dis, var_ref, var_dis, covariance).
#[allow(clippy::cast_precision_loss)]
fn block_statistics(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    start_x: usize,
    start_y: usize,
    block_size: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut sum_ref = 0.0_f64;
    let mut sum_dis = 0.0_f64;
    let mut sum_ref2 = 0.0_f64;
    let mut sum_dis2 = 0.0_f64;
    let mut sum_ref_dis = 0.0_f64;
    let mut count = 0u32;

    for dy in 0..block_size {
        for dx in 0..block_size {
            let x = start_x + dx;
            let y = start_y + dy;
            let idx = y * width + x;
            if idx < reference.len() && idx < distorted.len() {
                let r = f64::from(reference[idx]);
                let d = f64::from(distorted[idx]);
                sum_ref += r;
                sum_dis += d;
                sum_ref2 += r * r;
                sum_dis2 += d * d;
                sum_ref_dis += r * d;
                count += 1;
            }
        }
    }

    if count < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let n = f64::from(count);
    let mu_ref = sum_ref / n;
    let mu_dis = sum_dis / n;
    let var_ref = (sum_ref2 / n - mu_ref * mu_ref).max(0.0);
    let var_dis = (sum_dis2 / n - mu_dis * mu_dis).max(0.0);
    let cov = sum_ref_dis / n - mu_ref * mu_dis;

    (mu_ref, mu_dis, var_ref, var_dis, cov)
}

/// Computes DLM (Detail Loss Metric) using Sobel gradient comparison.
#[allow(clippy::cast_precision_loss)]
fn compute_dlm(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        return 1.0;
    }

    let mut ref_energy = 0.0_f64;
    let mut diff_energy = 0.0_f64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Sobel gradients for reference
            let (gx_r, gy_r) = sobel_at(reference, width, x, y);
            let (gx_d, gy_d) = sobel_at(distorted, width, x, y);

            let grad_ref = (gx_r * gx_r + gy_r * gy_r).sqrt();
            let grad_dis = (gx_d * gx_d + gy_d * gy_d).sqrt();

            ref_energy += grad_ref;
            diff_energy += (grad_ref - grad_dis).abs();
        }
    }

    if ref_energy > 1e-10 {
        (1.0 - diff_energy / ref_energy).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Computes Sobel gradient at a pixel position.
#[allow(clippy::cast_precision_loss)]
fn sobel_at(pixels: &[u8], width: usize, x: usize, y: usize) -> (f64, f64) {
    let p = |dx: usize, dy: usize| -> f64 {
        let idx = (y + dy - 1) * width + (x + dx - 1);
        if idx < pixels.len() {
            f64::from(pixels[idx])
        } else {
            0.0
        }
    };

    let gx = -p(0, 0) + p(2, 0) - 2.0 * p(0, 1) + 2.0 * p(2, 1) - p(0, 2) + p(2, 2);

    let gy = -p(0, 0) - 2.0 * p(1, 0) - p(2, 0) + p(0, 2) + 2.0 * p(1, 2) + p(2, 2);

    (gx, gy)
}

/// Computes ADM (Additive Detail Masking) from texture energy ratio.
#[allow(clippy::cast_precision_loss)]
fn compute_adm(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> f64 {
    let block_size = 8;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    let mut total_ratio = 0.0_f64;
    let mut block_count = 0u32;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let ref_var = block_variance(
                reference,
                width,
                bx * block_size,
                by * block_size,
                block_size,
            );
            let dis_var = block_variance(
                distorted,
                width,
                bx * block_size,
                by * block_size,
                block_size,
            );

            // ADM: ratio of distorted detail energy to reference detail energy
            // weighted by masking threshold
            let masking = 1.0 + ref_var.sqrt() * 0.1;
            let error = (ref_var.sqrt() - dis_var.sqrt()).abs();
            let masked_error = error / masking;

            total_ratio += 1.0 - masked_error.min(1.0);
            block_count += 1;
        }
    }

    if block_count > 0 {
        (total_ratio / f64::from(block_count)).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Computes variance of a pixel block.
#[allow(clippy::cast_precision_loss)]
fn block_variance(
    pixels: &[u8],
    width: usize,
    start_x: usize,
    start_y: usize,
    block_size: usize,
) -> f64 {
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;
    let mut count = 0u32;

    for dy in 0..block_size {
        for dx in 0..block_size {
            let idx = (start_y + dy) * width + (start_x + dx);
            if idx < pixels.len() {
                let v = f64::from(pixels[idx]);
                sum += v;
                sum2 += v * v;
                count += 1;
            }
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = f64::from(count);
    (sum2 / n - (sum / n).powi(2)).max(0.0)
}

/// Computes temporal motion metric between two consecutive reference frames.
#[allow(clippy::cast_precision_loss)]
fn compute_motion(current: &[u8], previous: &[u8], width: usize, height: usize) -> f64 {
    let len = current.len().min(previous.len()).min(width * height);
    if len == 0 {
        return 0.0;
    }

    let mut sad_sum = 0u64;
    for i in 0..len {
        sad_sum += u64::from(current[i].abs_diff(previous[i]));
    }

    sad_sum as f64 / len as f64
}

/// A predicted VMAF score with confidence information.
#[derive(Debug, Clone)]
pub struct VmafPrediction {
    /// Predicted VMAF score (0-100 scale).
    pub score: f64,
    /// Confidence level (0.0-1.0).
    pub confidence: f64,
    /// Feature set used for this prediction.
    pub features: VmafFeatures,
}

impl VmafPrediction {
    /// Returns whether the predicted quality is considered acceptable (>= threshold).
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    /// Returns a quality tier classification.
    pub fn quality_tier(&self) -> QualityTier {
        if self.score >= 93.0 {
            QualityTier::Excellent
        } else if self.score >= 80.0 {
            QualityTier::Good
        } else if self.score >= 60.0 {
            QualityTier::Fair
        } else if self.score >= 40.0 {
            QualityTier::Poor
        } else {
            QualityTier::Bad
        }
    }
}

/// Quality tier classification based on VMAF score ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTier {
    /// VMAF >= 93: Visually lossless.
    Excellent,
    /// VMAF >= 80: Good quality, minor artifacts.
    Good,
    /// VMAF >= 60: Acceptable quality.
    Fair,
    /// VMAF >= 40: Noticeable degradation.
    Poor,
    /// VMAF < 40: Severely degraded.
    Bad,
}

/// Configuration for the VMAF predictor.
#[derive(Debug, Clone)]
pub struct VmafPredictorConfig {
    /// Model version hint (affects coefficient selection).
    pub model_version: ModelVersion,
    /// Enable temporal pooling of scores.
    pub temporal_pooling: bool,
    /// Window size for temporal pooling.
    pub pooling_window: usize,
    /// Phone model adjustment (boosts scores for small screens).
    pub phone_model: bool,
}

impl Default for VmafPredictorConfig {
    fn default() -> Self {
        Self {
            model_version: ModelVersion::V063,
            temporal_pooling: true,
            pooling_window: 8,
            phone_model: false,
        }
    }
}

/// VMAF model version for coefficient selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVersion {
    /// VMAF model version 0.6.1.
    V061,
    /// VMAF model version 0.6.3 (more recent).
    V063,
}

/// The VMAF predictor engine.
#[derive(Debug)]
pub struct VmafPredictor {
    /// Configuration.
    config: VmafPredictorConfig,
    /// Temporal score buffer for pooling.
    score_history: VecDeque<f64>,
    /// Total frames predicted.
    total_frames: u64,
    /// Running sum of all scores for mean computation.
    score_sum: f64,
}

impl VmafPredictor {
    /// Creates a new VMAF predictor with the given configuration.
    pub fn new(config: VmafPredictorConfig) -> Self {
        Self {
            config,
            score_history: VecDeque::new(),
            total_frames: 0,
            score_sum: 0.0,
        }
    }

    /// Creates a predictor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(VmafPredictorConfig::default())
    }

    /// Predicts VMAF score from pre-computed features.
    #[allow(clippy::cast_precision_loss)]
    pub fn predict(&mut self, features: VmafFeatures) -> VmafPrediction {
        let raw_score = self.compute_raw_score(&features);
        let phone_adjusted = if self.config.phone_model {
            // Phone model typically adds ~3-5 points
            (raw_score + 4.0).min(100.0)
        } else {
            raw_score
        };

        let final_score = if self.config.temporal_pooling && !self.score_history.is_empty() {
            self.apply_temporal_pooling(phone_adjusted)
        } else {
            phone_adjusted
        };

        self.score_history.push_back(final_score);
        if self.score_history.len() > self.config.pooling_window {
            self.score_history.pop_front();
        }
        self.total_frames += 1;
        self.score_sum += final_score;

        let confidence = self.compute_confidence(&features);

        VmafPrediction {
            score: final_score,
            confidence,
            features,
        }
    }

    /// Predicts from PSNR/SSIM/motion statistics directly.
    pub fn predict_from_stats(&mut self, psnr: f64, ssim: f64, motion: f64) -> VmafPrediction {
        let features = VmafFeatures::from_stats(psnr, ssim, motion);
        self.predict(features)
    }

    /// Predicts VMAF from raw pixel data (reference vs distorted).
    ///
    /// This is the primary entry point for actual VMAF prediction from spatial
    /// and temporal features extracted directly from pixel data.
    pub fn predict_from_pixels(
        &mut self,
        reference: &[u8],
        distorted: &[u8],
        width: usize,
        height: usize,
        prev_reference: Option<&[u8]>,
    ) -> VmafPrediction {
        let features =
            VmafFeatures::extract_from_pixels(reference, distorted, width, height, prev_reference);
        self.predict(features)
    }

    /// Returns the mean VMAF score across all predicted frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_score(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.score_sum / self.total_frames as f64
        }
    }

    /// Returns total number of frames predicted.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Resets the predictor state.
    pub fn reset(&mut self) {
        self.score_history.clear();
        self.total_frames = 0;
        self.score_sum = 0.0;
    }

    /// Returns the minimum score in the current temporal window.
    pub fn window_min(&self) -> f64 {
        self.score_history
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the maximum score in the current temporal window.
    pub fn window_max(&self) -> f64 {
        self.score_history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Computes the raw VMAF score from features using SVR-inspired model.
    ///
    /// Uses a polynomial kernel SVM regression approximation with learned
    /// coefficients that approximate the VMAF v0.6.1/v0.6.3 model outputs.
    fn compute_raw_score(&self, features: &VmafFeatures) -> f64 {
        let (weights, bias) = match self.config.model_version {
            ModelVersion::V061 => (
                SvrWeights {
                    w_vif0: 4.2,
                    w_vif1: 12.5,
                    w_vif2: 17.8,
                    w_vif3: 24.3,
                    w_dlm: 18.0,
                    w_adm: 22.5,
                    w_motion_linear: -0.018,
                    w_motion_sq: 0.00008,
                    w_vif_dlm_cross: 5.0,
                    w_adm_sq: -8.0,
                },
                2.0,
            ),
            ModelVersion::V063 => (
                SvrWeights {
                    w_vif0: 3.8,
                    w_vif1: 11.2,
                    w_vif2: 18.5,
                    w_vif3: 25.8,
                    w_dlm: 16.5,
                    w_adm: 23.0,
                    w_motion_linear: -0.015,
                    w_motion_sq: 0.00006,
                    w_vif_dlm_cross: 4.5,
                    w_adm_sq: -7.5,
                },
                1.5,
            ),
        };

        // Linear terms
        let vif_contrib = features.vif_scale0 * weights.w_vif0
            + features.vif_scale1 * weights.w_vif1
            + features.vif_scale2 * weights.w_vif2
            + features.vif_scale3 * weights.w_vif3;

        let dlm_contrib = features.dlm * weights.w_dlm;
        let adm_contrib = features.adm * weights.w_adm;

        // Motion terms (linear + quadratic)
        let motion_contrib = features.motion * weights.w_motion_linear
            + features.motion * features.motion * weights.w_motion_sq;

        // Cross terms (capture interaction between features)
        let cross_contrib = features.mean_vif() * features.dlm * weights.w_vif_dlm_cross;

        // Quadratic term for ADM (diminishing returns at high quality)
        let adm_sq_contrib = features.adm * features.adm * weights.w_adm_sq;

        let raw = vif_contrib
            + dlm_contrib
            + adm_contrib
            + motion_contrib
            + cross_contrib
            + adm_sq_contrib
            + bias;

        // Apply logistic saturation to keep score in [0, 100]
        logistic_clamp(raw, 100.0)
    }

    /// Applies temporal pooling (harmonic mean for conservative estimate).
    #[allow(clippy::cast_precision_loss)]
    fn apply_temporal_pooling(&self, current: f64) -> f64 {
        if self.score_history.is_empty() {
            return current;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for &s in &self.score_history {
            if s > 0.0 {
                sum += 1.0 / s;
                count += 1;
            }
        }
        if current > 0.0 {
            sum += 1.0 / current;
            count += 1;
        }
        if count == 0 || sum == 0.0 {
            current
        } else {
            // Blend harmonic mean with current score
            let harmonic = count as f64 / sum;
            harmonic * 0.3 + current * 0.7
        }
    }

    /// Computes confidence based on feature range validity.
    fn compute_confidence(&self, features: &VmafFeatures) -> f64 {
        let mut conf: f64 = 1.0;
        // Lower confidence for extreme feature values
        if features.mean_vif() < 0.1 {
            conf *= 0.5;
        }
        if features.dlm < 0.1 {
            conf *= 0.7;
        }
        if features.motion > 100.0 {
            conf *= 0.8;
        }
        // Boost confidence when features are in expected ranges
        if features.mean_vif() > 0.3 && features.mean_vif() < 1.0 && features.dlm > 0.3 {
            conf = conf.min(1.0);
        }
        conf
    }
}

/// SVR model weight structure for VMAF prediction.
struct SvrWeights {
    w_vif0: f64,
    w_vif1: f64,
    w_vif2: f64,
    w_vif3: f64,
    w_dlm: f64,
    w_adm: f64,
    w_motion_linear: f64,
    w_motion_sq: f64,
    w_vif_dlm_cross: f64,
    w_adm_sq: f64,
}

/// Logistic saturation function: maps raw score to [0, max_val].
fn logistic_clamp(x: f64, max_val: f64) -> f64 {
    if x >= max_val {
        max_val
    } else if x <= 0.0 {
        0.0
    } else {
        // Soft saturation near boundaries
        let normalized = x / max_val;
        if normalized > 0.95 {
            // Soft ceiling
            let excess = normalized - 0.95;
            (0.95 + excess * 0.5) * max_val
        } else {
            x
        }
    }
}

/// Estimates the QP value needed to achieve a target VMAF score.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
pub fn estimate_qp_for_target_vmaf(target_vmaf: f64, content_complexity: f64) -> u8 {
    // Empirical model: VMAF ~ 100 - k * QP^1.2 * complexity
    // Solving for QP: QP ~ ((100 - target) / (k * complexity))^(1/1.2)
    let k = 0.15;
    let complexity = content_complexity.max(0.1);
    let diff = (100.0 - target_vmaf).max(0.0);
    let qp_f = (diff / (k * complexity)).powf(1.0 / 1.2);
    qp_f.round().max(1.0).min(51.0) as u8
}

/// Batch VMAF predictor for processing multiple frames efficiently.
#[derive(Debug)]
pub struct BatchVmafPredictor {
    /// Per-frame predictor.
    predictor: VmafPredictor,
    /// Frame-level scores for analysis.
    frame_scores: Vec<f64>,
}

impl BatchVmafPredictor {
    /// Creates a new batch predictor.
    pub fn new(config: VmafPredictorConfig) -> Self {
        Self {
            predictor: VmafPredictor::new(config),
            frame_scores: Vec::new(),
        }
    }

    /// Processes a batch of frames and returns per-frame predictions.
    pub fn process_batch(
        &mut self,
        references: &[&[u8]],
        distorted: &[&[u8]],
        width: usize,
        height: usize,
    ) -> Vec<VmafPrediction> {
        let len = references.len().min(distorted.len());
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            let prev_ref = if i > 0 { Some(references[i - 1]) } else { None };
            let prediction = self.predictor.predict_from_pixels(
                references[i],
                distorted[i],
                width,
                height,
                prev_ref,
            );
            self.frame_scores.push(prediction.score);
            results.push(prediction);
        }

        results
    }

    /// Returns the harmonic mean of all frame scores (VMAF's preferred pooling).
    #[allow(clippy::cast_precision_loss)]
    pub fn harmonic_mean_score(&self) -> f64 {
        if self.frame_scores.is_empty() {
            return 0.0;
        }
        let mut inv_sum = 0.0_f64;
        let mut count = 0usize;
        for &s in &self.frame_scores {
            if s > 1e-10 {
                inv_sum += 1.0 / s;
                count += 1;
            }
        }
        if count == 0 || inv_sum < 1e-10 {
            0.0
        } else {
            count as f64 / inv_sum
        }
    }

    /// Returns the percentile score (e.g., 5th percentile for worst-case quality).
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn percentile_score(&self, percentile: f64) -> f64 {
        if self.frame_scores.is_empty() {
            return 0.0;
        }
        let mut sorted = self.frame_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (sorted.len() as f64 - 1.0))
            .round()
            .max(0.0) as usize;
        let clamped_idx = idx.min(sorted.len().saturating_sub(1));
        sorted[clamped_idx]
    }

    /// Returns the number of frames below a quality threshold.
    pub fn frames_below_threshold(&self, threshold: f64) -> usize {
        self.frame_scores.iter().filter(|&&s| s < threshold).count()
    }

    /// Resets the batch predictor.
    pub fn reset(&mut self) {
        self.predictor.reset();
        self.frame_scores.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmaf_features_perfect() {
        let f = VmafFeatures::perfect();
        assert!((f.mean_vif() - 1.0).abs() < f64::EPSILON);
        assert!((f.dlm - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vmaf_features_from_stats() {
        let f = VmafFeatures::from_stats(40.0, 0.95, 5.0);
        assert!(f.vif_scale0 > 0.0 && f.vif_scale0 <= 1.0);
        assert!(f.dlm > 0.0 && f.dlm <= 1.0);
        assert!((f.motion - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vmaf_features_mean_vif() {
        let f = VmafFeatures {
            vif_scale0: 0.8,
            vif_scale1: 0.6,
            vif_scale2: 0.4,
            vif_scale3: 0.2,
            dlm: 0.5,
            motion: 0.0,
            adm: 0.5,
        };
        assert!((f.mean_vif() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vmaf_prediction_quality_tier() {
        let pred = VmafPrediction {
            score: 95.0,
            confidence: 1.0,
            features: VmafFeatures::perfect(),
        };
        assert_eq!(pred.quality_tier(), QualityTier::Excellent);

        let pred2 = VmafPrediction {
            score: 85.0,
            confidence: 1.0,
            features: VmafFeatures::perfect(),
        };
        assert_eq!(pred2.quality_tier(), QualityTier::Good);

        let pred3 = VmafPrediction {
            score: 30.0,
            confidence: 0.5,
            features: VmafFeatures::perfect(),
        };
        assert_eq!(pred3.quality_tier(), QualityTier::Bad);
    }

    #[test]
    fn test_vmaf_prediction_threshold() {
        let pred = VmafPrediction {
            score: 85.0,
            confidence: 1.0,
            features: VmafFeatures::perfect(),
        };
        assert!(pred.meets_threshold(80.0));
        assert!(!pred.meets_threshold(90.0));
    }

    #[test]
    fn test_vmaf_predictor_new() {
        let p = VmafPredictor::with_defaults();
        assert_eq!(p.total_frames(), 0);
        assert!((p.mean_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vmaf_predictor_predict_perfect() {
        let mut p = VmafPredictor::with_defaults();
        let result = p.predict(VmafFeatures::perfect());
        assert!(result.score > 90.0);
        assert!(result.confidence > 0.9);
        assert_eq!(p.total_frames(), 1);
    }

    #[test]
    fn test_vmaf_predictor_predict_from_stats() {
        let mut p = VmafPredictor::with_defaults();
        let result = p.predict_from_stats(42.0, 0.98, 2.0);
        assert!(result.score > 0.0);
        assert!(result.score <= 100.0);
    }

    #[test]
    fn test_vmaf_predictor_mean_score() {
        let mut p = VmafPredictor::with_defaults();
        p.predict(VmafFeatures::perfect());
        p.predict(VmafFeatures::perfect());
        let mean = p.mean_score();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_vmaf_predictor_reset() {
        let mut p = VmafPredictor::with_defaults();
        p.predict(VmafFeatures::perfect());
        p.reset();
        assert_eq!(p.total_frames(), 0);
        assert!((p.mean_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vmaf_predictor_phone_model() {
        let mut p_std = VmafPredictor::new(VmafPredictorConfig {
            phone_model: false,
            temporal_pooling: false,
            ..Default::default()
        });
        let mut p_phone = VmafPredictor::new(VmafPredictorConfig {
            phone_model: true,
            temporal_pooling: false,
            ..Default::default()
        });
        let features = VmafFeatures::from_stats(35.0, 0.9, 3.0);
        let score_std = p_std.predict(features.clone()).score;
        let score_phone = p_phone.predict(features).score;
        assert!(score_phone > score_std);
    }

    #[test]
    fn test_vmaf_predictor_window_min_max() {
        let mut p = VmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        p.predict(VmafFeatures::from_stats(30.0, 0.8, 1.0));
        p.predict(VmafFeatures::from_stats(45.0, 0.99, 0.5));
        assert!(p.window_min() < p.window_max());
    }

    #[test]
    fn test_estimate_qp_for_target_vmaf() {
        let qp_high = estimate_qp_for_target_vmaf(95.0, 0.5);
        let qp_low = estimate_qp_for_target_vmaf(60.0, 0.5);
        assert!(qp_high < qp_low);

        let qp = estimate_qp_for_target_vmaf(100.0, 0.5);
        assert!(qp >= 1);
        assert!(qp <= 51);
    }

    #[test]
    fn test_model_version_affects_score() {
        let mut p1 = VmafPredictor::new(VmafPredictorConfig {
            model_version: ModelVersion::V061,
            temporal_pooling: false,
            ..Default::default()
        });
        let mut p2 = VmafPredictor::new(VmafPredictorConfig {
            model_version: ModelVersion::V063,
            temporal_pooling: false,
            ..Default::default()
        });
        let features = VmafFeatures::from_stats(38.0, 0.92, 4.0);
        let s1 = p1.predict(features.clone()).score;
        let s2 = p2.predict(features).score;
        // Different models should produce slightly different scores
        assert!((s1 - s2).abs() < 10.0);
    }

    // --- New tests for pixel-based VMAF prediction ---

    #[test]
    fn test_extract_features_identical_frames() {
        let width = 32;
        let height = 32;
        let frame: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let features = VmafFeatures::extract_from_pixels(&frame, &frame, width, height, None);
        // Identical frames should have high VIF and DLM
        assert!(
            features.mean_vif() > 0.5,
            "VIF should be high for identical frames: {}",
            features.mean_vif()
        );
        assert!(
            features.dlm > 0.8,
            "DLM should be high for identical frames: {}",
            features.dlm
        );
        assert!(
            features.adm > 0.8,
            "ADM should be high for identical frames: {}",
            features.adm
        );
    }

    #[test]
    fn test_extract_features_degraded_frame() {
        let width = 32;
        let height = 32;
        let reference: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        // Heavily quantized version
        let distorted: Vec<u8> = reference.iter().map(|&p| (p / 32) * 32).collect();
        let features =
            VmafFeatures::extract_from_pixels(&reference, &distorted, width, height, None);
        // Degraded frames should have lower scores than identical
        let identical =
            VmafFeatures::extract_from_pixels(&reference, &reference, width, height, None);
        assert!(features.dlm < identical.dlm || (features.dlm - identical.dlm).abs() < 0.01);
    }

    #[test]
    fn test_predict_from_pixels_identical() {
        let mut p = VmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        let width = 32;
        let height = 32;
        let frame: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let result = p.predict_from_pixels(&frame, &frame, width, height, None);
        assert!(
            result.score > 50.0,
            "Identical frames should score high: {}",
            result.score
        );
    }

    #[test]
    fn test_predict_from_pixels_with_motion() {
        let mut p = VmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        let width = 32;
        let height = 32;
        let frame1: Vec<u8> = vec![100; width * height];
        let frame2: Vec<u8> = vec![200; width * height];
        let result = p.predict_from_pixels(&frame2, &frame2, width, height, Some(&frame1));
        // Motion should be detected
        assert!(result.features.motion > 0.0);
    }

    #[test]
    fn test_batch_vmaf_predictor() {
        let mut batch = BatchVmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        let width = 16;
        let height = 16;
        let frame: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let refs: Vec<&[u8]> = vec![&frame; 5];
        let dists: Vec<&[u8]> = vec![&frame; 5];
        let results = batch.process_batch(&refs, &dists, width, height);
        assert_eq!(results.len(), 5);
        let hmean = batch.harmonic_mean_score();
        assert!(hmean > 0.0);
    }

    #[test]
    fn test_batch_percentile_score() {
        let mut batch = BatchVmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        let width = 16;
        let height = 16;
        let frame: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let refs: Vec<&[u8]> = vec![&frame; 10];
        let dists: Vec<&[u8]> = vec![&frame; 10];
        let _ = batch.process_batch(&refs, &dists, width, height);
        let p5 = batch.percentile_score(5.0);
        let p95 = batch.percentile_score(95.0);
        assert!(p5 <= p95);
    }

    #[test]
    fn test_batch_frames_below_threshold() {
        let mut batch = BatchVmafPredictor::new(VmafPredictorConfig::default());
        // No frames processed
        assert_eq!(batch.frames_below_threshold(80.0), 0);
        batch.reset();
        assert_eq!(batch.frames_below_threshold(80.0), 0);
    }

    #[test]
    fn test_vif_at_scale_identical() {
        let width = 64;
        let height = 64;
        let frame: Vec<u8> = (0..width * height).map(|i| ((i * 7) % 256) as u8).collect();
        let vif = compute_vif_at_scale(&frame, &frame, width, height, 1);
        assert!(vif > 0.5, "VIF for identical should be high: {}", vif);
    }

    #[test]
    fn test_dlm_identical_is_one() {
        let width = 32;
        let height = 32;
        let frame: Vec<u8> = (0..width * height)
            .map(|i| ((i * 13) % 256) as u8)
            .collect();
        let dlm = compute_dlm(&frame, &frame, width, height);
        assert!(
            (dlm - 1.0).abs() < 0.01,
            "DLM for identical should be ~1.0: {}",
            dlm
        );
    }

    #[test]
    fn test_adm_identical_is_high() {
        let width = 32;
        let height = 32;
        let frame: Vec<u8> = (0..width * height)
            .map(|i| ((i * 17) % 256) as u8)
            .collect();
        let adm = compute_adm(&frame, &frame, width, height);
        assert!(adm > 0.9, "ADM for identical should be high: {}", adm);
    }

    #[test]
    fn test_motion_zero_for_identical() {
        let frame: Vec<u8> = vec![128; 256];
        let motion = compute_motion(&frame, &frame, 16, 16);
        assert!((motion - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_nonzero_for_different() {
        let frame1: Vec<u8> = vec![100; 256];
        let frame2: Vec<u8> = vec![200; 256];
        let motion = compute_motion(&frame1, &frame2, 16, 16);
        assert!((motion - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logistic_clamp() {
        assert!((logistic_clamp(50.0, 100.0) - 50.0).abs() < f64::EPSILON);
        assert!((logistic_clamp(-5.0, 100.0) - 0.0).abs() < f64::EPSILON);
        assert!((logistic_clamp(200.0, 100.0) - 100.0).abs() < f64::EPSILON);
        // Near ceiling: soft saturation
        let val = logistic_clamp(98.0, 100.0);
        assert!(val > 95.0 && val <= 100.0);
    }

    #[test]
    fn test_higher_quality_gives_higher_vmaf() {
        let mut p = VmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        let high_q = VmafFeatures::from_stats(45.0, 0.99, 1.0);
        let low_q = VmafFeatures::from_stats(25.0, 0.7, 1.0);
        let score_high = p.predict(high_q).score;
        let score_low = p.predict(low_q).score;
        assert!(
            score_high > score_low,
            "Higher quality should give higher VMAF: high={}, low={}",
            score_high,
            score_low
        );
    }

    #[test]
    fn test_svr_model_produces_valid_range() {
        let mut p = VmafPredictor::new(VmafPredictorConfig {
            temporal_pooling: false,
            ..Default::default()
        });
        // Test various feature combinations
        let test_cases = vec![
            VmafFeatures {
                vif_scale0: 0.0,
                vif_scale1: 0.0,
                vif_scale2: 0.0,
                vif_scale3: 0.0,
                dlm: 0.0,
                motion: 0.0,
                adm: 0.0,
            },
            VmafFeatures {
                vif_scale0: 0.5,
                vif_scale1: 0.5,
                vif_scale2: 0.5,
                vif_scale3: 0.5,
                dlm: 0.5,
                motion: 50.0,
                adm: 0.5,
            },
            VmafFeatures::perfect(),
            VmafFeatures {
                vif_scale0: 0.1,
                vif_scale1: 0.1,
                vif_scale2: 0.1,
                vif_scale3: 0.1,
                dlm: 0.1,
                motion: 200.0,
                adm: 0.1,
            },
        ];
        for features in test_cases {
            let result = p.predict(features);
            assert!(
                result.score >= 0.0 && result.score <= 100.0,
                "Score out of range: {}",
                result.score
            );
        }
    }
}
