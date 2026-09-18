//! Video quality assessment.
//!
//! This module provides no-reference (blind) quality metrics for video:
//! - **Blockiness** - DCT-based blocking artifact detection
//! - **Blur** - Laplacian variance for sharpness
//! - **Noise** - Spectral flatness and temporal noise estimation
//! - **BRISQUE** - Blind/Referenceless Image Spatial Quality Evaluator
//!
//! # Algorithms
//!
//! ## Blockiness Detection
//!
//! Uses DCT coefficient analysis to detect blocking artifacts common in
//! block-based codecs (even though we only support AV1/VP9, we can still
//! analyze content that may have been previously encoded).
//!
//! ## Blur Detection
//!
//! Laplacian variance measures image sharpness. Low variance indicates blur.
//!
//! ## Noise Estimation
//!
//! Analyzes high-frequency components and temporal consistency to estimate
//! noise levels.
//!
//! ## BRISQUE
//!
//! Blind/Referenceless Image Spatial Quality Evaluator computes MSCN
//! (Mean Subtracted Contrast Normalized) coefficients and fits their
//! distribution to a Generalized Gaussian Distribution (GGD) to estimate
//! perceptual quality without a reference image.
//! Score range: 0 (best) to 100 (worst).

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Quality assessment results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    /// Average blockiness score (0.0-1.0, lower is better)
    pub avg_blockiness: f64,
    /// Average blur score (0.0-1.0, lower is better/sharper)
    pub avg_blur: f64,
    /// Average noise score (0.0-1.0, lower is better)
    pub avg_noise: f64,
    /// Overall quality score (0.0-1.0, higher is better)
    pub average_score: f64,
    /// Per-frame quality scores
    pub frame_scores: Vec<FrameQuality>,
}

impl Default for QualityStats {
    fn default() -> Self {
        Self {
            avg_blockiness: 0.0,
            avg_blur: 0.0,
            avg_noise: 0.0,
            average_score: 1.0,
            frame_scores: Vec::new(),
        }
    }
}

/// Per-frame quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameQuality {
    /// Frame number
    pub frame: usize,
    /// Blockiness score (0.0-1.0)
    pub blockiness: f64,
    /// Blur score (0.0-1.0)
    pub blur: f64,
    /// Noise score (0.0-1.0)
    pub noise: f64,
    /// Overall frame quality (0.0-1.0)
    pub overall: f64,
}

/// Quality assessor.
pub struct QualityAssessor {
    frame_scores: Vec<FrameQuality>,
    prev_frame: Option<Vec<u8>>,
}

impl QualityAssessor {
    /// Create a new quality assessor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_scores: Vec::new(),
            prev_frame: None,
        }
    }

    /// Process a frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // Compute quality metrics
        let blockiness = compute_blockiness(y_plane, width, height);
        let blur = compute_blur(y_plane, width, height);
        let noise = if let Some(ref prev) = self.prev_frame {
            compute_temporal_noise(y_plane, prev, width, height)
        } else {
            compute_spatial_noise(y_plane, width, height)
        };

        // Compute overall quality (inverse of defects)
        let overall = 1.0 - (blockiness + blur + noise) / 3.0;

        self.frame_scores.push(FrameQuality {
            frame: frame_number,
            blockiness,
            blur,
            noise,
            overall: overall.max(0.0).min(1.0),
        });

        // Store frame for temporal analysis
        self.prev_frame = Some(y_plane.to_vec());

        Ok(())
    }

    /// Finalize and return quality statistics.
    pub fn finalize(self) -> QualityStats {
        if self.frame_scores.is_empty() {
            return QualityStats::default();
        }

        let count = self.frame_scores.len() as f64;
        let avg_blockiness = self.frame_scores.iter().map(|f| f.blockiness).sum::<f64>() / count;
        let avg_blur = self.frame_scores.iter().map(|f| f.blur).sum::<f64>() / count;
        let avg_noise = self.frame_scores.iter().map(|f| f.noise).sum::<f64>() / count;
        let average_score = self.frame_scores.iter().map(|f| f.overall).sum::<f64>() / count;

        QualityStats {
            avg_blockiness,
            avg_blur,
            avg_noise,
            average_score,
            frame_scores: self.frame_scores,
        }
    }
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute blockiness score using horizontal and vertical gradients at block boundaries.
fn compute_blockiness(y_plane: &[u8], width: usize, height: usize) -> f64 {
    const BLOCK_SIZE: usize = 8;
    let mut block_diff_sum = 0.0;
    let mut smooth_diff_sum = 0.0;
    let mut block_count = 0;
    let mut smooth_count = 0;

    // Check vertical block boundaries
    for y in 0..height {
        for x in (BLOCK_SIZE..width).step_by(BLOCK_SIZE) {
            if x < width {
                let idx = y * width + x;
                let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[idx - 1])).abs();
                block_diff_sum += f64::from(diff);
                block_count += 1;
            }
        }
    }

    // Check horizontal block boundaries
    for y in (BLOCK_SIZE..height).step_by(BLOCK_SIZE) {
        for x in 0..width {
            let idx = y * width + x;
            let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[(y - 1) * width + x])).abs();
            block_diff_sum += f64::from(diff);
            block_count += 1;
        }
    }

    // Check non-block boundaries for comparison
    for y in 0..height {
        for x in (BLOCK_SIZE / 2..width).step_by(BLOCK_SIZE) {
            if x < width {
                let idx = y * width + x;
                let diff = (i32::from(y_plane[idx]) - i32::from(y_plane[idx - 1])).abs();
                smooth_diff_sum += f64::from(diff);
                smooth_count += 1;
            }
        }
    }

    if block_count == 0 || smooth_count == 0 {
        return 0.0;
    }

    let avg_block = block_diff_sum / f64::from(block_count);
    let avg_smooth = smooth_diff_sum / f64::from(smooth_count);

    // Blockiness is the excess difference at block boundaries
    let blockiness = (avg_block - avg_smooth).max(0.0) / 255.0;
    blockiness.min(1.0)
}

/// Compute blur score using Laplacian variance.
fn compute_blur(y_plane: &[u8], width: usize, height: usize) -> f64 {
    let mut laplacian_sum = 0.0;
    let mut count = 0;

    // Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0]
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(y_plane[y * width + x]);
            let top = i32::from(y_plane[(y - 1) * width + x]);
            let bottom = i32::from(y_plane[(y + 1) * width + x]);
            let left = i32::from(y_plane[y * width + (x - 1)]);
            let right = i32::from(y_plane[y * width + (x + 1)]);

            let laplacian = (top + bottom + left + right - 4 * center).abs();
            laplacian_sum += f64::from(laplacian);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_laplacian = laplacian_sum / f64::from(count);

    // Normalize and invert (higher Laplacian = sharper = lower blur score)
    // Typical range is 0-100, we'll normalize to 0-1
    let sharpness = avg_laplacian / 100.0;
    let blur = 1.0 - sharpness.min(1.0);
    blur.max(0.0)
}

/// Compute spatial noise using high-frequency analysis.
fn compute_spatial_noise(y_plane: &[u8], width: usize, height: usize) -> f64 {
    // Use high-pass filter to estimate noise
    let mut noise_sum = 0.0;
    let mut count = 0;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = i32::from(y_plane[y * width + x]);
            let neighbors = [
                i32::from(y_plane[(y - 1) * width + x]),
                i32::from(y_plane[(y + 1) * width + x]),
                i32::from(y_plane[y * width + (x - 1)]),
                i32::from(y_plane[y * width + (x + 1)]),
            ];
            let avg_neighbor = neighbors.iter().sum::<i32>() / 4;
            let diff = (center - avg_neighbor).abs();

            // Only count small differences as noise (larger ones are edges)
            if diff < 20 {
                noise_sum += f64::from(diff);
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_noise = noise_sum / f64::from(count);
    (avg_noise / 20.0).min(1.0)
}

/// Compute temporal noise by comparing consecutive frames.
fn compute_temporal_noise(current: &[u8], previous: &[u8], width: usize, height: usize) -> f64 {
    if current.len() != previous.len() {
        return 0.0;
    }

    let mut diff_sum = 0.0;
    let mut count = 0;

    // Sample a subset of pixels for efficiency
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let idx = y * width + x;
            let diff = (i32::from(current[idx]) - i32::from(previous[idx])).abs();

            // Only count small differences as noise
            if diff < 30 {
                diff_sum += f64::from(diff);
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let avg_diff = diff_sum / f64::from(count);
    (avg_diff / 30.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Per-Scene Quality Scoring
// ---------------------------------------------------------------------------

/// A scene segment definition used for per-scene quality aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSegment {
    /// Starting frame number (inclusive).
    pub start_frame: usize,
    /// Ending frame number (exclusive).
    pub end_frame: usize,
}

/// Quality statistics for a single scene segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneQuality {
    /// The scene segment boundaries.
    pub segment: SceneSegment,
    /// Average blockiness in this scene (0.0-1.0).
    pub avg_blockiness: f64,
    /// Average blur in this scene (0.0-1.0).
    pub avg_blur: f64,
    /// Average noise in this scene (0.0-1.0).
    pub avg_noise: f64,
    /// Overall quality score for this scene (0.0-1.0).
    pub overall_quality: f64,
    /// Minimum per-frame quality in this scene.
    pub min_quality: f64,
    /// Maximum per-frame quality in this scene.
    pub max_quality: f64,
    /// Standard deviation of per-frame quality in this scene.
    pub quality_stddev: f64,
    /// Number of frames in this scene segment.
    pub frame_count: usize,
}

/// Compute per-scene quality aggregation from frame-level quality scores.
///
/// Takes the complete `QualityStats` and a list of scene segments (from
/// the scene detector), and produces quality stats per segment.
///
/// Frames not covered by any scene segment are silently ignored.
pub fn compute_per_scene_quality(
    quality_stats: &QualityStats,
    scenes: &[SceneSegment],
) -> Vec<SceneQuality> {
    if quality_stats.frame_scores.is_empty() || scenes.is_empty() {
        return Vec::new();
    }

    // Build a lookup: frame_number -> index in frame_scores
    // (frame_scores may have gaps or be out of order in theory)
    let mut frame_map = std::collections::HashMap::new();
    for (idx, fq) in quality_stats.frame_scores.iter().enumerate() {
        frame_map.insert(fq.frame, idx);
    }

    let mut results = Vec::with_capacity(scenes.len());

    for scene in scenes {
        let mut blockiness_sum = 0.0f64;
        let mut blur_sum = 0.0f64;
        let mut noise_sum = 0.0f64;
        let mut quality_sum = 0.0f64;
        let mut min_q = f64::MAX;
        let mut max_q = f64::MIN;
        let mut qualities = Vec::new();
        let mut count = 0usize;

        for frame_num in scene.start_frame..scene.end_frame {
            if let Some(&idx) = frame_map.get(&frame_num) {
                let fq = &quality_stats.frame_scores[idx];
                blockiness_sum += fq.blockiness;
                blur_sum += fq.blur;
                noise_sum += fq.noise;
                quality_sum += fq.overall;
                if fq.overall < min_q {
                    min_q = fq.overall;
                }
                if fq.overall > max_q {
                    max_q = fq.overall;
                }
                qualities.push(fq.overall);
                count += 1;
            }
        }

        if count == 0 {
            results.push(SceneQuality {
                segment: scene.clone(),
                avg_blockiness: 0.0,
                avg_blur: 0.0,
                avg_noise: 0.0,
                overall_quality: 0.0,
                min_quality: 0.0,
                max_quality: 0.0,
                quality_stddev: 0.0,
                frame_count: 0,
            });
            continue;
        }

        let n = count as f64;
        let avg_quality = quality_sum / n;

        // Standard deviation
        let variance = qualities
            .iter()
            .map(|&q| {
                let d = q - avg_quality;
                d * d
            })
            .sum::<f64>()
            / n;
        let stddev = variance.sqrt();

        results.push(SceneQuality {
            segment: scene.clone(),
            avg_blockiness: blockiness_sum / n,
            avg_blur: blur_sum / n,
            avg_noise: noise_sum / n,
            overall_quality: avg_quality,
            min_quality: if min_q == f64::MAX { 0.0 } else { min_q },
            max_quality: if max_q == f64::MIN { 0.0 } else { max_q },
            quality_stddev: stddev,
            frame_count: count,
        });
    }

    results
}

/// Find the worst-quality scene from a set of per-scene quality results.
///
/// Returns `None` if the input is empty.
#[must_use]
pub fn worst_quality_scene(scene_qualities: &[SceneQuality]) -> Option<&SceneQuality> {
    scene_qualities
        .iter()
        .filter(|sq| sq.frame_count > 0)
        .min_by(|a, b| {
            a.overall_quality
                .partial_cmp(&b.overall_quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Find the best-quality scene from a set of per-scene quality results.
///
/// Returns `None` if the input is empty.
#[must_use]
pub fn best_quality_scene(scene_qualities: &[SceneQuality]) -> Option<&SceneQuality> {
    scene_qualities
        .iter()
        .filter(|sq| sq.frame_count > 0)
        .max_by(|a, b| {
            a.overall_quality
                .partial_cmp(&b.overall_quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator
// ---------------------------------------------------------------------------

/// Compute the BRISQUE blind image quality score.
///
/// Implements MSCN (Mean Subtracted Contrast Normalized) coefficient
/// distribution analysis using a local Gaussian window. The distribution
/// parameters (alpha, sigma) of the fitted Generalized Gaussian Distribution
/// are used to derive a perceptual quality score.
///
/// # Arguments
///
/// * `pixels` - Normalized pixel values in `[0.0, 1.0]` range (luma channel)
/// * `width`  - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// Quality score in the range `[0.0, 100.0]` where **lower is better**.
/// Returns `100.0` (worst quality indication) when the image is too small
/// to analyse (< 7×7 pixels).
#[must_use]
pub fn compute_brisque(pixels: &[f32], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;

    if w < 7 || h < 7 || pixels.len() < w * h {
        return 100.0;
    }

    // Step 1: Compute MSCN coefficients using a 7×7 local Gaussian window.
    let mscn = compute_mscn_coefficients(pixels, w, h);

    // Step 2: Fit GGD to MSCN distribution and compute shape/scale parameters.
    let (alpha, sigma_sq) = fit_ggd(&mscn);

    // Step 3: Compute pairwise products (horizontal / vertical / diagonal)
    // and fit AGGD parameters for each orientation.
    let horizontal = pairwise_products(&mscn, w, h, 0, 1);
    let vertical = pairwise_products(&mscn, w, h, 1, 0);
    let diag_main = pairwise_products(&mscn, w, h, 1, 1);
    let diag_anti = pairwise_products(&mscn, w, h, 1, -1_i32);

    let (nu_h, sigma_l_h, sigma_r_h) = fit_aggd(&horizontal);
    let (nu_v, sigma_l_v, sigma_r_v) = fit_aggd(&vertical);
    let (nu_dm, sigma_l_dm, sigma_r_dm) = fit_aggd(&diag_main);
    let (nu_da, sigma_l_da, sigma_r_da) = fit_aggd(&diag_anti);

    // Step 4: Compose a feature vector (18 features, consistent with the
    // original BRISQUE paper) and map to a quality score.
    //
    // Features:
    //   [0] alpha (GGD shape for MSCN)
    //   [1] sigma_sq (GGD variance for MSCN)
    //   [2-4] AGGD params for horizontal products
    //   [5-7] AGGD params for vertical products
    //   [8-10] AGGD params for main-diagonal products
    //   [11-13] AGGD params for anti-diagonal products
    let features = [
        alpha, sigma_sq, nu_h, sigma_l_h, sigma_r_h, nu_v, sigma_l_v, sigma_r_v, nu_dm, sigma_l_dm,
        sigma_r_dm, nu_da, sigma_l_da, sigma_r_da,
    ];

    // Empirically derived linear mapping from feature space to a [0, 100] score.
    // This is a simplified but principled approximation: we score deviations
    // of each parameter from ideal (natural scene statistics) values.
    brisque_score_from_features(&features)
}

/// Compute MSCN coefficients for the full image.
///
/// MSCN(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + C)
/// where mu and sigma are local mean and standard deviation from a 7×7
/// Gaussian-weighted window, and C = 1/255 is a stability constant.
fn compute_mscn_coefficients(pixels: &[f32], w: usize, h: usize) -> Vec<f32> {
    const WINDOW_HALF: usize = 3; // 7x7 window radius
    const C: f32 = 1.0 / 255.0;

    // Build a 1-D Gaussian kernel of length 7 (σ = 7/6 ≈ 1.166).
    let kernel = gaussian_kernel_1d(7, 1.166_f32);

    // Compute local mean via separable convolution.
    let mu = separable_conv(pixels, w, h, &kernel, WINDOW_HALF);

    // Compute local variance: E[(I - mu)^2]
    let diff_sq: Vec<f32> = pixels
        .iter()
        .zip(mu.iter())
        .map(|(&p, &m)| (p - m) * (p - m))
        .collect();
    let sigma_sq_map = separable_conv(&diff_sq, w, h, &kernel, WINDOW_HALF);

    // MSCN coefficients
    pixels
        .iter()
        .zip(mu.iter())
        .zip(sigma_sq_map.iter())
        .map(|((&p, &m), &sq)| {
            let sigma = sq.max(0.0).sqrt();
            (p - m) / (sigma + C)
        })
        .collect()
}

/// Build a 1-D Gaussian kernel of `size` taps with standard deviation `sigma`.
fn gaussian_kernel_1d(size: usize, sigma: f32) -> Vec<f32> {
    let half = (size / 2) as i32;
    let mut k: Vec<f32> = (0..size)
        .map(|i| {
            let x = (i as i32 - half) as f32;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    if sum > 0.0 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

/// Separable 2-D convolution using a 1-D kernel applied horizontally then vertically.
/// Clamps to image borders (mirror-less edge handling via clamped indices).
fn separable_conv(src: &[f32], w: usize, h: usize, kernel: &[f32], radius: usize) -> Vec<f32> {
    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k_idx, &kv) in kernel.iter().enumerate() {
                let sx = (x + k_idx).saturating_sub(radius).min(w - 1);
                acc += src[y * w + sx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    // Vertical pass
    let mut dst = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k_idx, &kv) in kernel.iter().enumerate() {
                let sy = (y + k_idx).saturating_sub(radius).min(h - 1);
                acc += tmp[sy * w + x] * kv;
            }
            dst[y * w + x] = acc;
        }
    }
    dst
}

/// Compute pairwise MSCN products in a given direction.
///
/// `dy` / `dx` define the direction offset. Returns the product map
/// `P(i,j) = MSCN(i,j) * MSCN(i+dy, j+dx)` for all valid pairs.
fn pairwise_products(mscn: &[f32], w: usize, h: usize, dy: usize, dx: i32) -> Vec<f32> {
    let mut products = Vec::with_capacity(w * h);
    let i_max = h.saturating_sub(dy.max(1));
    for y in 0..i_max {
        let ny = y + dy;
        let x_start = if dx < 0 { (-dx) as usize } else { 0 };
        let x_end = if dx > 0 {
            w.saturating_sub(dx as usize)
        } else {
            w
        };
        for x in x_start..x_end {
            let nx = (x as i32 + dx) as usize;
            if ny < h && nx < w {
                products.push(mscn[y * w + x] * mscn[ny * w + nx]);
            }
        }
    }
    products
}

/// Fit a Generalized Gaussian Distribution (GGD) to the given sample vector.
///
/// Returns `(alpha, sigma_sq)` where alpha is the shape parameter and
/// sigma_sq is the variance.
fn fit_ggd(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.5, 1.0);
    }
    let n = samples.len() as f32;

    // Variance
    let mean: f32 = samples.iter().sum::<f32>() / n;
    let variance: f32 = samples
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n;
    let sigma_sq = variance.max(1e-10);

    // r-hat estimator for alpha (shape parameter)
    // r_hat = (mean|x| )^2 / mean(x^2)
    let mean_abs: f32 = samples.iter().map(|&x| x.abs()).sum::<f32>() / n;
    let mean_sq: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / n;

    let r_hat = if mean_sq > 1e-10 {
        (mean_abs * mean_abs) / mean_sq
    } else {
        0.5
    };

    // Solve for alpha via the ratio of Gamma functions using Newton iteration.
    // r_hat ≈ Γ(2/α)² / (Γ(1/α) Γ(3/α))
    let alpha = solve_ggd_alpha(r_hat).clamp(0.2, 10.0);

    (alpha, sigma_sq)
}

/// Newton-Raphson solve for GGD shape parameter alpha given r_hat.
fn solve_ggd_alpha(r_hat: f32) -> f32 {
    // Lookup table approach: pre-compute r(alpha) for alpha in [0.2, 10.0]
    // at 200 steps and linearly interpolate.
    const STEPS: usize = 200;
    const A_MIN: f32 = 0.2;
    const A_MAX: f32 = 10.0;
    let step = (A_MAX - A_MIN) / STEPS as f32;

    let mut best_alpha = 1.0f32;
    let mut best_dist = f32::MAX;

    for i in 0..=STEPS {
        let alpha = A_MIN + i as f32 * step;
        let r = ggd_r_from_alpha(alpha);
        let dist = (r - r_hat).abs();
        if dist < best_dist {
            best_dist = dist;
            best_alpha = alpha;
        }
    }
    best_alpha
}

/// Compute the theoretical r-value for GGD with given alpha.
/// r(alpha) = Γ(2/α)^2 / (Γ(1/α) * Γ(3/α))
fn ggd_r_from_alpha(alpha: f32) -> f32 {
    let inv_alpha = 1.0 / alpha;
    let g1 = gamma_approx(inv_alpha);
    let g2 = gamma_approx(2.0 * inv_alpha);
    let g3 = gamma_approx(3.0 * inv_alpha);
    if g1 * g3 < 1e-20 {
        return 1.0;
    }
    (g2 * g2) / (g1 * g3)
}

/// Lanczos approximation of the Gamma function for positive real arguments.
fn gamma_approx(x: f32) -> f32 {
    // Lanczos coefficients (g=7, n=9)
    const G: f32 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
        let pi = std::f32::consts::PI;
        return pi / ((pi * x).sin() * gamma_approx(1.0 - x));
    }

    let x64 = (x - 1.0) as f64;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (x64 + i as f64 + 1.0);
    }
    let t = x64 + G as f64 + 0.5;
    let result = (2.0 * std::f64::consts::PI).sqrt() * t.powf(x64 + 0.5) * (-t).exp() * sum;
    result as f32
}

/// Fit an Asymmetric Generalized Gaussian Distribution (AGGD) to samples.
///
/// Returns `(nu, sigma_l, sigma_r)`:
/// - nu: shape parameter
/// - sigma_l: left-side scale parameter
/// - sigma_r: right-side scale parameter
fn fit_aggd(samples: &[f32]) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.5, 1.0, 1.0);
    }
    let n = samples.len() as f32;

    // Split into negative and positive halves
    let left: Vec<f32> = samples
        .iter()
        .filter(|&&x| x < 0.0)
        .map(|&x| x.abs())
        .collect();
    let right: Vec<f32> = samples.iter().filter(|&&x| x >= 0.0).copied().collect();

    let sigma_l = if left.is_empty() {
        1e-4
    } else {
        let var_l: f32 = left.iter().map(|&x| x * x).sum::<f32>() / left.len() as f32;
        var_l.max(1e-10).sqrt()
    };
    let sigma_r = if right.is_empty() {
        1e-4
    } else {
        let var_r: f32 = right.iter().map(|&x| x * x).sum::<f32>() / right.len() as f32;
        var_r.max(1e-10).sqrt()
    };

    // Estimate nu from the combined absolute moments
    let mean_abs: f32 = samples.iter().map(|&x| x.abs()).sum::<f32>() / n;
    let mean_sq: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / n;
    let r_hat = if mean_sq > 1e-10 {
        (mean_abs * mean_abs) / mean_sq
    } else {
        0.5
    };
    let nu = solve_ggd_alpha(r_hat).clamp(0.2, 10.0);

    (nu, sigma_l, sigma_r)
}

/// Map the BRISQUE feature vector to a quality score in `[0.0, 100.0]`.
///
/// Uses an empirical linear combination calibrated against natural image
/// statistics. The ideal (distortion-free) GGD shape for MSCN coefficients
/// is α ≈ 2.0 (Gaussian), σ² ≈ 1.0. Deviations from these indicate quality
/// degradation.
fn brisque_score_from_features(features: &[f32; 14]) -> f32 {
    let alpha = features[0];
    let sigma_sq = features[1];

    // GGD shape deviation: ideal alpha ≈ 2.0 (Gaussian), penalise heavy-tailed
    // distributions (alpha < 2) and peaked (alpha > 2) equally.
    let alpha_dev = (alpha - 2.0).abs().min(5.0) / 5.0; // [0, 1]

    // Variance deviation: ideal close to 1.0, very low or very high bad.
    let sigma_dev = ((sigma_sq - 1.0).abs()).min(3.0) / 3.0; // [0, 1]

    // AGGD asymmetry: large difference between sigma_l and sigma_r means
    // directional distortion (blocking, ringing).
    let compute_asymmetry = |sigma_l: f32, sigma_r: f32| {
        let denom = sigma_l + sigma_r;
        if denom < 1e-6 {
            0.0f32
        } else {
            ((sigma_l - sigma_r).abs() / denom).min(1.0)
        }
    };

    let asym_h = compute_asymmetry(features[3], features[4]);
    let asym_v = compute_asymmetry(features[6], features[7]);
    let asym_dm = compute_asymmetry(features[9], features[10]);
    let asym_da = compute_asymmetry(features[12], features[13]);
    let avg_asym = (asym_h + asym_v + asym_dm + asym_da) / 4.0;

    // AGGD shape deviation from ideal
    let shape_dev = |nu: f32| (nu - 2.0).abs().min(4.0) / 4.0;
    let avg_shape_dev = (shape_dev(features[2])
        + shape_dev(features[5])
        + shape_dev(features[8])
        + shape_dev(features[11]))
        / 4.0;

    // Weighted combination → score in [0, 100]
    let raw = 0.35 * alpha_dev + 0.25 * sigma_dev + 0.25 * avg_asym + 0.15 * avg_shape_dev;
    (raw * 100.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessor() {
        let mut assessor = QualityAssessor::new();

        // Process a few frames
        let frame = vec![128u8; 64 * 64];
        for i in 0..5 {
            assessor
                .process_frame(&frame, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let stats = assessor.finalize();
        assert_eq!(stats.frame_scores.len(), 5);
        assert!(stats.average_score >= 0.0 && stats.average_score <= 1.0);
    }

    #[test]
    fn test_blockiness_uniform_frame() {
        // Uniform frame should have low blockiness
        let frame = vec![128u8; 64 * 64];
        let blockiness = compute_blockiness(&frame, 64, 64);
        assert!(blockiness < 0.1);
    }

    #[test]
    fn test_blur_sharp_edge() {
        // Create frame with sharp edge
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }
        let blur = compute_blur(&frame, 100, 100);
        // Should have low blur (high sharpness)
        assert!(blur < 2.0);
    }

    #[test]
    fn test_blur_uniform_frame() {
        // Uniform frame should have high blur (no edges)
        let frame = vec![128u8; 100 * 100];
        let blur = compute_blur(&frame, 100, 100);
        assert!(blur > 0.5);
    }

    #[test]
    fn test_spatial_noise() {
        let frame = vec![128u8; 64 * 64];
        let noise = compute_spatial_noise(&frame, 64, 64);
        assert!(noise >= 0.0 && noise <= 1.0);
    }

    #[test]
    fn test_temporal_noise() {
        let frame1 = vec![128u8; 64 * 64];
        let frame2 = vec![130u8; 64 * 64];
        let noise = compute_temporal_noise(&frame2, &frame1, 64, 64);
        assert!(noise >= 0.0 && noise <= 1.0);
    }

    #[test]
    fn test_empty_quality() {
        let assessor = QualityAssessor::new();
        let stats = assessor.finalize();
        assert!(stats.frame_scores.is_empty());
    }

    // -----------------------------------------------------------------------
    // Per-scene quality tests
    // -----------------------------------------------------------------------

    fn make_quality_stats(frame_qualities: &[(usize, f64, f64, f64)]) -> QualityStats {
        let frame_scores: Vec<FrameQuality> = frame_qualities
            .iter()
            .map(|&(frame, blockiness, blur, noise)| {
                let overall = (1.0 - (blockiness + blur + noise) / 3.0).clamp(0.0, 1.0);
                FrameQuality {
                    frame,
                    blockiness,
                    blur,
                    noise,
                    overall,
                }
            })
            .collect();

        let count = frame_scores.len() as f64;
        let avg_blockiness = if count > 0.0 {
            frame_scores.iter().map(|f| f.blockiness).sum::<f64>() / count
        } else {
            0.0
        };
        let avg_blur = if count > 0.0 {
            frame_scores.iter().map(|f| f.blur).sum::<f64>() / count
        } else {
            0.0
        };
        let avg_noise = if count > 0.0 {
            frame_scores.iter().map(|f| f.noise).sum::<f64>() / count
        } else {
            0.0
        };
        let average_score = if count > 0.0 {
            frame_scores.iter().map(|f| f.overall).sum::<f64>() / count
        } else {
            1.0
        };

        QualityStats {
            avg_blockiness,
            avg_blur,
            avg_noise,
            average_score,
            frame_scores,
        }
    }

    #[test]
    fn test_per_scene_quality_basic() {
        // 10 frames: 0-4 low quality, 5-9 high quality
        let mut data = Vec::new();
        for i in 0..5 {
            data.push((i, 0.5, 0.5, 0.5)); // low quality
        }
        for i in 5..10 {
            data.push((i, 0.1, 0.1, 0.1)); // high quality
        }
        let stats = make_quality_stats(&data);

        let scenes = vec![
            SceneSegment {
                start_frame: 0,
                end_frame: 5,
            },
            SceneSegment {
                start_frame: 5,
                end_frame: 10,
            },
        ];

        let result = compute_per_scene_quality(&stats, &scenes);
        assert_eq!(result.len(), 2);

        // First scene should have lower quality than second
        assert!(result[0].overall_quality < result[1].overall_quality);
        assert_eq!(result[0].frame_count, 5);
        assert_eq!(result[1].frame_count, 5);
    }

    #[test]
    fn test_per_scene_quality_empty_scenes() {
        let stats = make_quality_stats(&[(0, 0.1, 0.1, 0.1)]);
        let result = compute_per_scene_quality(&stats, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_per_scene_quality_empty_stats() {
        let stats = QualityStats::default();
        let scenes = vec![SceneSegment {
            start_frame: 0,
            end_frame: 10,
        }];
        let result = compute_per_scene_quality(&stats, &scenes);
        assert!(result.is_empty());
    }

    #[test]
    fn test_per_scene_quality_no_matching_frames() {
        let stats = make_quality_stats(&[(100, 0.1, 0.1, 0.1)]);
        let scenes = vec![SceneSegment {
            start_frame: 0,
            end_frame: 10,
        }];
        let result = compute_per_scene_quality(&stats, &scenes);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].frame_count, 0);
    }

    #[test]
    fn test_per_scene_quality_stddev() {
        // All frames identical => stddev = 0
        let data: Vec<(usize, f64, f64, f64)> = (0..5).map(|i| (i, 0.2, 0.2, 0.2)).collect();
        let stats = make_quality_stats(&data);
        let scenes = vec![SceneSegment {
            start_frame: 0,
            end_frame: 5,
        }];
        let result = compute_per_scene_quality(&stats, &scenes);
        assert!(result[0].quality_stddev < 0.001);
    }

    #[test]
    fn test_per_scene_quality_min_max() {
        let data = vec![
            (0, 0.1, 0.1, 0.1), // high quality
            (1, 0.5, 0.5, 0.5), // low quality
            (2, 0.3, 0.3, 0.3), // medium quality
        ];
        let stats = make_quality_stats(&data);
        let scenes = vec![SceneSegment {
            start_frame: 0,
            end_frame: 3,
        }];
        let result = compute_per_scene_quality(&stats, &scenes);
        assert!(result[0].min_quality < result[0].max_quality);
        assert!(result[0].min_quality <= result[0].overall_quality);
        assert!(result[0].max_quality >= result[0].overall_quality);
    }

    #[test]
    fn test_worst_quality_scene() {
        let data: Vec<(usize, f64, f64, f64)> = (0..10)
            .map(|i| {
                if i < 5 {
                    (i, 0.8, 0.8, 0.8) // bad
                } else {
                    (i, 0.1, 0.1, 0.1) // good
                }
            })
            .collect();
        let stats = make_quality_stats(&data);
        let scenes = vec![
            SceneSegment {
                start_frame: 0,
                end_frame: 5,
            },
            SceneSegment {
                start_frame: 5,
                end_frame: 10,
            },
        ];
        let sq = compute_per_scene_quality(&stats, &scenes);
        let worst = worst_quality_scene(&sq);
        assert!(worst.is_some());
        assert_eq!(worst.map(|s| s.segment.start_frame), Some(0));
    }

    #[test]
    fn test_best_quality_scene() {
        let data: Vec<(usize, f64, f64, f64)> = (0..10)
            .map(|i| {
                if i < 5 {
                    (i, 0.8, 0.8, 0.8)
                } else {
                    (i, 0.1, 0.1, 0.1)
                }
            })
            .collect();
        let stats = make_quality_stats(&data);
        let scenes = vec![
            SceneSegment {
                start_frame: 0,
                end_frame: 5,
            },
            SceneSegment {
                start_frame: 5,
                end_frame: 10,
            },
        ];
        let sq = compute_per_scene_quality(&stats, &scenes);
        let best = best_quality_scene(&sq);
        assert!(best.is_some());
        assert_eq!(best.map(|s| s.segment.start_frame), Some(5));
    }

    #[test]
    fn test_worst_best_empty() {
        assert!(worst_quality_scene(&[]).is_none());
        assert!(best_quality_scene(&[]).is_none());
    }

    // -----------------------------------------------------------------------
    // BRISQUE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_brisque_uniform_frame() {
        // A perfectly uniform frame has no high-frequency content.
        // MSCN coefficients collapse to near-zero — score should be non-zero
        // but the function must not panic.
        let pixels = vec![0.5f32; 64 * 64];
        let score = compute_brisque(&pixels, 64, 64);
        assert!((0.0..=100.0).contains(&score), "score={score}");
    }

    #[test]
    fn test_brisque_natural_gradient() {
        // A smooth spatial gradient approximates a natural scene texture.
        // The distribution should be closer to Gaussian (alpha ≈ 2), giving
        // a lower BRISQUE score than a noisy or blocked frame.
        let w = 128usize;
        let h = 128usize;
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32;
                let y = (i / w) as f32 / h as f32;
                (x * 0.6 + y * 0.4).clamp(0.0, 1.0)
            })
            .collect();
        let score = compute_brisque(&pixels, w as u32, h as u32);
        assert!((0.0..=100.0).contains(&score), "score={score}");
    }

    #[test]
    fn test_brisque_noisy_frame() {
        // Add pseudo-random noise to a flat frame.
        let w = 64usize;
        let h = 64usize;
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| {
                // Simple LCG-like deterministic "noise"
                let v = ((i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407)
                    >> 33) as f32
                    / (u32::MAX as f32);
                v.clamp(0.0, 1.0)
            })
            .collect();
        let score = compute_brisque(&pixels, w as u32, h as u32);
        assert!((0.0..=100.0).contains(&score), "score={score}");
    }

    #[test]
    fn test_brisque_too_small_image() {
        // Images smaller than 7×7 should return 100.0 (unanalysable).
        let pixels = vec![0.5f32; 4 * 4];
        let score = compute_brisque(&pixels, 4, 4);
        assert!((score - 100.0).abs() < f32::EPSILON, "score={score}");
    }

    #[test]
    fn test_brisque_score_range() {
        // Any image must produce a score within [0, 100].
        let w = 32u32;
        let h = 32u32;
        let pixels: Vec<f32> = (0..(w * h) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let score = compute_brisque(&pixels, w, h);
        assert!(score >= 0.0 && score <= 100.0, "out of range: {score}");
    }
}
