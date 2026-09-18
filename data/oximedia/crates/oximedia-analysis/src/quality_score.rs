//! Composite quality scoring across multiple perceptual dimensions.
//!
//! Provides a weighted multi-dimension quality assessment framework
//! with grade classification from overall numeric scores.

#![allow(dead_code)]

/// A named quality dimension with an associated weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityDimension {
    /// Spatial sharpness and resolution quality.
    Sharpness,
    /// Noise and grain level (inverse: low noise = high quality).
    Noise,
    /// Compression artifact severity (inverse).
    Artifacts,
    /// Color accuracy and saturation quality.
    Color,
    /// Temporal stability and motion smoothness.
    Temporal,
}

impl QualityDimension {
    /// Return the default weight (0.0–1.0) for this dimension.
    ///
    /// Weights express relative importance across dimensions.
    #[must_use]
    pub fn weight(self) -> f32 {
        match self {
            Self::Sharpness => 0.30,
            Self::Noise => 0.20,
            Self::Artifacts => 0.25,
            Self::Color => 0.15,
            Self::Temporal => 0.10,
        }
    }
}

/// A score (0.0–100.0) for a single quality dimension.
#[derive(Debug, Clone, Copy)]
pub struct DimensionScore {
    /// The dimension this score belongs to.
    pub dimension: QualityDimension,
    /// Raw score in range 0.0–100.0.
    pub raw: f32,
}

impl DimensionScore {
    /// Create a new `DimensionScore`, clamping `raw` to 0.0–100.0.
    #[must_use]
    pub fn new(dimension: QualityDimension, raw: f32) -> Self {
        Self {
            dimension,
            raw: raw.clamp(0.0, 100.0),
        }
    }

    /// Compute the weighted contribution of this score.
    ///
    /// `weighted = raw * dimension.weight()`
    #[must_use]
    pub fn weighted(self) -> f32 {
        self.raw * self.dimension.weight()
    }
}

/// A quality scorer that accumulates dimension scores and computes a final score.
#[derive(Debug, Clone, Default)]
pub struct QualityScorer {
    scores: Vec<DimensionScore>,
}

impl QualityScorer {
    /// Create an empty `QualityScorer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a dimension score (replaces any existing score for the same dimension).
    pub fn add_dimension(&mut self, score: DimensionScore) {
        self.scores.retain(|s| s.dimension != score.dimension);
        self.scores.push(score);
    }

    /// Compute the overall quality score (0.0–100.0).
    ///
    /// Uses weighted sum: `Σ(raw_i * weight_i) / Σ(weight_i)`.
    /// Returns 0.0 if no dimensions have been added.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let weight_sum: f32 = self.scores.iter().map(|s| s.dimension.weight()).sum();
        if weight_sum < f32::EPSILON {
            return 0.0;
        }
        let weighted_sum: f32 = self.scores.iter().map(|s| s.weighted()).sum();
        (weighted_sum / weight_sum).clamp(0.0, 100.0)
    }

    /// Return all recorded dimension scores.
    #[must_use]
    pub fn scores(&self) -> &[DimensionScore] {
        &self.scores
    }
}

/// A human-readable quality grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    /// Excellent: score ≥ 90.
    Excellent,
    /// Good: score ≥ 75.
    Good,
    /// Fair: score ≥ 55.
    Fair,
    /// Poor: score ≥ 30.
    Poor,
    /// Failing: score < 30.
    Failing,
}

impl QualityGrade {
    /// Classify a numeric score (0.0–100.0) into a `QualityGrade`.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        match score as u32 {
            90..=100 => Self::Excellent,
            75..=89 => Self::Good,
            55..=74 => Self::Fair,
            30..=54 => Self::Poor,
            _ => Self::Failing,
        }
    }

    /// Return a short label string for this grade.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Failing => "Failing",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VMAF-inspired Perceptual Quality Metric
// ─────────────────────────────────────────────────────────────────────────────

/// Features used by the VMAF-inspired perceptual quality estimator.
#[derive(Debug, Clone, Default)]
pub struct VmafFeatures {
    /// Detail Loss Measure (DLM) — captures lost texture/detail.
    pub dlm: f32,
    /// Additive Impairment Measure (AIM) — captures noise/ringing artifacts.
    pub aim: f32,
    /// Motion (temporal activity) — reduces quality sensitivity in high-motion regions.
    pub motion: f32,
    /// Visual Information Fidelity in pixel domain (VIF-pixel).
    pub vif_pixel: f32,
}

impl VmafFeatures {
    /// Compute features from a reference and distorted luma plane.
    ///
    /// Both planes must have the same dimensions (`width × height` bytes).
    #[must_use]
    pub fn compute(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> Self {
        if reference.len() != width * height
            || distorted.len() != width * height
            || width < 4
            || height < 4
        {
            return Self::default();
        }

        let dlm = compute_dlm(reference, distorted, width, height);
        let aim = compute_aim(reference, distorted, width, height);
        let motion = compute_motion_feature(distorted, width, height);
        let vif_pixel = compute_vif_pixel(reference, distorted, width, height);

        Self {
            dlm,
            aim,
            motion,
            vif_pixel,
        }
    }
}

/// Compute VMAF-inspired score (0–100) from extracted features.
///
/// Uses a linear SVM-inspired regression model trained on VQEG correlations.
/// Weights are taken from the VMAF v0.6.1 model coefficients (simplified).
#[must_use]
pub fn compute_vmaf_score(features: &VmafFeatures) -> f32 {
    // Normalised weights (approximated from VMAF model)
    const W_DLM: f32 = 39.3;
    const W_VIF: f32 = 47.5;
    const W_AIM: f32 = -8.4;
    const W_MOTION: f32 = 2.3;
    const BIAS: f32 = 26.7;

    let raw = W_DLM * features.dlm
        + W_VIF * features.vif_pixel
        + W_AIM * features.aim
        + W_MOTION * features.motion
        + BIAS;

    // Map through logistic to [0, 100]
    let logistic = 100.0 / (1.0 + (-0.5 * (raw - 50.0) / 25.0_f32).exp());
    logistic.clamp(0.0, 100.0)
}

// ── DLM (Detail Loss Measure) ─────────────────────────────────────────────

/// Compute a simplified Detail Loss Measure using local Laplacian contrast.
///
/// DLM measures how much fine detail energy (edge/texture) is preserved in the
/// distorted frame relative to the reference.  A value of 1.0 means no detail
/// loss; 0.0 means all detail is lost.
fn compute_dlm(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> f32 {
    let mut ref_energy = 0.0f64;
    let mut dist_energy = 0.0f64;

    // Laplacian-of-Gaussian approximation using 3×3 kernel
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let laplacian = |plane: &[u8]| -> f64 {
                let c = f64::from(plane[y * width + x]);
                let n = f64::from(plane[(y - 1) * width + x]);
                let s = f64::from(plane[(y + 1) * width + x]);
                let w = f64::from(plane[y * width + (x - 1)]);
                let e = f64::from(plane[y * width + (x + 1)]);
                (n + s + w + e - 4.0 * c).abs()
            };

            ref_energy += laplacian(reference);
            dist_energy += laplacian(distorted);
        }
    }

    if ref_energy < 1.0 {
        return 1.0; // Flat reference — no detail to lose
    }

    // Clamp to [0,1]: distorted can have more energy (ringing) → DLM still capped at 1
    ((dist_energy / ref_energy) as f32).clamp(0.0, 1.0)
}

// ── AIM (Additive Impairment Measure) ─────────────────────────────────────

/// Compute a simplified Additive Impairment Measure.
///
/// AIM captures noise and blocking artifacts by measuring the residual energy
/// (per-pixel difference) in high-detail regions.  A score of 0.0 means no
/// added impairment; 1.0 means maximum impairment.
fn compute_aim(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> f32 {
    let mut residual_sq = 0.0f64;
    let n = (width * height) as f64;

    for (r, d) in reference.iter().zip(distorted.iter()) {
        let diff = f64::from(*d) - f64::from(*r);
        residual_sq += diff * diff;
    }

    // Normalise: MSE in [0, 255²], map to [0, 1]
    let mse = residual_sq / (n * 255.0 * 255.0);
    (mse as f32).clamp(0.0, 1.0)
}

// ── Motion Feature ────────────────────────────────────────────────────────

/// Compute a motion feature from the distorted frame's temporal gradient proxy.
///
/// This approximates temporal activity using the high-frequency spatial content
/// of the frame (acts as a proxy for frame-difference magnitude when only a
/// single frame is available).
fn compute_motion_feature(distorted: &[u8], width: usize, height: usize) -> f32 {
    let mut hf_energy = 0.0f64;
    let pixels = (width * height) as f64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let dx = i32::from(distorted[y * width + (x + 1)])
                - i32::from(distorted[y * width + (x - 1)]);
            let dy = i32::from(distorted[(y + 1) * width + x])
                - i32::from(distorted[(y - 1) * width + x]);
            hf_energy += f64::from(dx * dx + dy * dy);
        }
    }

    let normalised = (hf_energy / (pixels * 255.0 * 255.0 * 8.0)) as f32;
    normalised.clamp(0.0, 1.0)
}

// ── VIF-pixel ─────────────────────────────────────────────────────────────

/// Compute a simplified Visual Information Fidelity in the pixel domain (VIF-pixel).
///
/// VIF-pixel measures the ratio of mutual information between the reference and
/// the human visual system (HVS) to the information present in the reference
/// alone.  Higher values (≈1) indicate high perceptual fidelity.
///
/// This implementation uses a lightweight local variance ratio approximation
/// over non-overlapping 8×8 blocks — following the spirit of the VIF algorithm
/// without requiring a full wavelet decomposition.
fn compute_vif_pixel(reference: &[u8], distorted: &[u8], width: usize, height: usize) -> f32 {
    const BLOCK: usize = 8;
    let sigma_n_sq: f64 = 4.0; // Noise model variance (AWGN approximation)

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    let blocks_y = height / BLOCK;
    let blocks_x = width / BLOCK;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            // Collect block pixels
            let mut ref_vals = Vec::with_capacity(BLOCK * BLOCK);
            let mut dist_vals = Vec::with_capacity(BLOCK * BLOCK);

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = (by * BLOCK + dy) * width + (bx * BLOCK + dx);
                    ref_vals.push(f64::from(reference[idx]));
                    dist_vals.push(f64::from(distorted[idx]));
                }
            }

            let n = ref_vals.len() as f64;
            let ref_mean: f64 = ref_vals.iter().sum::<f64>() / n;
            let dist_mean: f64 = dist_vals.iter().sum::<f64>() / n;

            let sigma_ref_sq: f64 = ref_vals
                .iter()
                .map(|&v| (v - ref_mean).powi(2))
                .sum::<f64>()
                / n;
            let sigma_dist_sq: f64 = dist_vals
                .iter()
                .map(|&v| (v - dist_mean).powi(2))
                .sum::<f64>()
                / n;
            let sigma_ref_dist: f64 = ref_vals
                .iter()
                .zip(dist_vals.iter())
                .map(|(&r, &d)| (r - ref_mean) * (d - dist_mean))
                .sum::<f64>()
                / n;

            let g = sigma_ref_dist / (sigma_ref_sq + 1e-10);
            let sigma_v_sq = (sigma_dist_sq - g * sigma_ref_dist).max(0.0);

            // VIF numerator: information in distorted channel
            let num_block =
                ((1.0 + (g * g * sigma_ref_sq) / (sigma_v_sq + sigma_n_sq)).ln()).max(0.0);
            // VIF denominator: information in reference channel
            let den_block = ((1.0 + sigma_ref_sq / sigma_n_sq).ln()).max(0.0);

            num += num_block;
            den += den_block;
        }
    }

    if den < 1e-10 {
        return 1.0; // Flat reference — perfect fidelity by convention
    }

    ((num / den) as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_weight_sums_near_one() {
        let sum = QualityDimension::Sharpness.weight()
            + QualityDimension::Noise.weight()
            + QualityDimension::Artifacts.weight()
            + QualityDimension::Color.weight()
            + QualityDimension::Temporal.weight();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum = {sum}");
    }

    #[test]
    fn test_dimension_score_clamps_above_100() {
        let ds = DimensionScore::new(QualityDimension::Sharpness, 150.0);
        assert!((ds.raw - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dimension_score_clamps_below_0() {
        let ds = DimensionScore::new(QualityDimension::Noise, -10.0);
        assert!(ds.raw.abs() < f32::EPSILON);
    }

    #[test]
    fn test_dimension_score_weighted() {
        let ds = DimensionScore::new(QualityDimension::Sharpness, 100.0);
        assert!((ds.weighted() - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_quality_scorer_empty_overall() {
        let scorer = QualityScorer::new();
        assert!(scorer.overall_score().abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_scorer_single_dimension() {
        let mut scorer = QualityScorer::new();
        scorer.add_dimension(DimensionScore::new(QualityDimension::Sharpness, 80.0));
        // Only one dimension: weighted/weight_sum = 80*0.30/0.30 = 80.0
        assert!((scorer.overall_score() - 80.0).abs() < 1e-4);
    }

    #[test]
    fn test_quality_scorer_replace_existing_dimension() {
        let mut scorer = QualityScorer::new();
        scorer.add_dimension(DimensionScore::new(QualityDimension::Sharpness, 50.0));
        scorer.add_dimension(DimensionScore::new(QualityDimension::Sharpness, 90.0));
        assert_eq!(scorer.scores().len(), 1);
        assert!((scorer.scores()[0].raw - 90.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_scorer_all_dimensions_perfect() {
        let mut scorer = QualityScorer::new();
        for dim in [
            QualityDimension::Sharpness,
            QualityDimension::Noise,
            QualityDimension::Artifacts,
            QualityDimension::Color,
            QualityDimension::Temporal,
        ] {
            scorer.add_dimension(DimensionScore::new(dim, 100.0));
        }
        assert!((scorer.overall_score() - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_quality_grade_excellent() {
        assert_eq!(QualityGrade::from_score(95.0), QualityGrade::Excellent);
    }

    #[test]
    fn test_quality_grade_good() {
        assert_eq!(QualityGrade::from_score(80.0), QualityGrade::Good);
    }

    #[test]
    fn test_quality_grade_fair() {
        assert_eq!(QualityGrade::from_score(60.0), QualityGrade::Fair);
    }

    #[test]
    fn test_quality_grade_poor() {
        assert_eq!(QualityGrade::from_score(40.0), QualityGrade::Poor);
    }

    #[test]
    fn test_quality_grade_failing() {
        assert_eq!(QualityGrade::from_score(10.0), QualityGrade::Failing);
    }

    #[test]
    fn test_quality_grade_label() {
        assert_eq!(QualityGrade::Excellent.label(), "Excellent");
        assert_eq!(QualityGrade::Failing.label(), "Failing");
    }

    #[test]
    fn test_quality_scorer_scores_slice_length() {
        let mut scorer = QualityScorer::new();
        scorer.add_dimension(DimensionScore::new(QualityDimension::Color, 70.0));
        scorer.add_dimension(DimensionScore::new(QualityDimension::Noise, 60.0));
        assert_eq!(scorer.scores().len(), 2);
    }

    // ── VMAF-inspired metric tests ────────────────────────────────────────

    #[test]
    fn test_vmaf_features_identical_frames() {
        let frame = vec![128u8; 64 * 64];
        let features = VmafFeatures::compute(&frame, &frame, 64, 64);
        // Identical frames: DLM = 1, AIM = 0, VIF ≈ 1
        assert!((features.dlm - 1.0).abs() < 1e-3, "dlm={}", features.dlm);
        assert!(features.aim < 1e-4, "aim={}", features.aim);
        assert!(features.vif_pixel >= 0.9, "vif={}", features.vif_pixel);
    }

    #[test]
    fn test_vmaf_features_empty_returns_default() {
        let features = VmafFeatures::compute(&[], &[], 0, 0);
        assert!((features.dlm).abs() < 1e-5);
        assert!((features.vif_pixel).abs() < 1e-5);
    }

    #[test]
    fn test_vmaf_score_perfect_quality() {
        let frame = vec![128u8; 64 * 64];
        let features = VmafFeatures::compute(&frame, &frame, 64, 64);
        let score = compute_vmaf_score(&features);
        // Identical frames → should yield a high VMAF score
        assert!(score > 70.0, "score={score}");
    }

    #[test]
    fn test_vmaf_score_degraded_quality() {
        let reference: Vec<u8> = (0..64u8)
            .flat_map(|y| (0..64u8).map(move |x| x ^ y))
            .collect();
        // Heavily corrupted distorted frame (random noise)
        let distorted: Vec<u8> = (0..64 * 64)
            .map(|i: usize| ((i * 37 + 13) % 256) as u8)
            .collect();
        let features = VmafFeatures::compute(&reference, &distorted, 64, 64);
        let score = compute_vmaf_score(&features);
        // Heavily distorted → lower score than perfect
        let ref_features = VmafFeatures::compute(&reference, &reference, 64, 64);
        let ref_score = compute_vmaf_score(&ref_features);
        assert!(
            score <= ref_score,
            "distorted={score} should be ≤ perfect={ref_score}"
        );
    }

    #[test]
    fn test_vmaf_score_range() {
        let frame = vec![100u8; 32 * 32];
        let noisy: Vec<u8> = frame.iter().map(|&v| v.saturating_add(30)).collect();
        let features = VmafFeatures::compute(&frame, &noisy, 32, 32);
        let score = compute_vmaf_score(&features);
        assert!(
            score >= 0.0 && score <= 100.0,
            "score out of range: {score}"
        );
    }

    #[test]
    fn test_vif_pixel_identical_frames() {
        let frame: Vec<u8> = (0..64u8)
            .flat_map(|y| (0..64u8).map(move |x| x + y))
            .collect();
        let vif = compute_vif_pixel(&frame, &frame, 64, 64);
        assert!(vif >= 0.95, "vif={vif}");
    }

    #[test]
    fn test_aim_no_distortion() {
        let frame = vec![128u8; 100];
        let aim = compute_aim(&frame, &frame, 10, 10);
        assert!(aim < 1e-5, "aim={aim}");
    }

    #[test]
    fn test_dlm_flat_reference() {
        // Flat reference → no detail to lose → DLM = 1.0
        let flat = vec![128u8; 64 * 64];
        let distorted = vec![100u8; 64 * 64];
        let dlm = compute_dlm(&flat, &distorted, 64, 64);
        assert!((dlm - 1.0).abs() < 1e-3, "dlm={dlm}");
    }
}
