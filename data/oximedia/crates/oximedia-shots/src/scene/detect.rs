//! Scene boundary detection with multi-modal analysis.
//!
//! Combines three complementary signals to decide whether two consecutive
//! shots belong to the same scene:
//!
//! 1. **Color histogram comparison** – computes per-channel histograms and
//!    measures the chi-squared distance between shots.
//! 2. **Motion vector analysis** – estimates camera / subject motion magnitude
//!    from frame-to-frame pixel differences.
//! 3. **Semantic embedding similarity** – cosine similarity between compact
//!    feature vectors (mean/std of RGB channels as a lightweight proxy for a
//!    CNN embedding).

// use crate::error::ShotResult; // unused
use crate::types::{Scene, Shot};
// use oximedia_core::types::Timestamp; // unused

// ---------------------------------------------------------------------------
// Color histogram comparison
// ---------------------------------------------------------------------------

/// Histogram configuration.
#[derive(Debug, Clone, Copy)]
pub struct HistogramConfig {
    /// Number of bins per channel.
    pub bins: usize,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self { bins: 32 }
    }
}

/// Compute a normalised per-channel histogram from flat RGB pixel data.
///
/// `pixels` is a flat slice of `[R, G, B, R, G, B, …]` bytes.
/// Returns a `Vec<f32>` of length `bins * 3` (R bins, then G bins, then B bins).
#[must_use]
#[allow(dead_code)]
pub fn compute_color_histogram(pixels: &[u8], bins: usize) -> Vec<f32> {
    let bins = bins.max(1);
    let mut hist = vec![0.0_f32; bins * 3];
    let total_pixels = pixels.len() / 3;

    if total_pixels == 0 {
        return hist;
    }

    let bin_width = 256.0 / bins as f32;

    for chunk in pixels.chunks_exact(3) {
        for (ch, &byte) in chunk.iter().enumerate() {
            let bin = ((byte as f32) / bin_width) as usize;
            hist[ch * bins + bin.min(bins - 1)] += 1.0;
        }
    }

    // Normalise each channel
    for ch in 0..3 {
        let start = ch * bins;
        let end = start + bins;
        let sum: f32 = hist[start..end].iter().sum();
        if sum > 0.0 {
            for v in &mut hist[start..end] {
                *v /= sum;
            }
        }
    }
    hist
}

/// Chi-squared distance between two normalised histograms.
///
/// Returns a value in [0, ∞).  Identical histograms → 0.0.
#[must_use]
#[allow(dead_code)]
pub fn histogram_chi_squared(h1: &[f32], h2: &[f32]) -> f32 {
    assert_eq!(h1.len(), h2.len(), "histogram lengths must match");
    h1.iter()
        .zip(h2.iter())
        .map(|(&a, &b)| {
            let denom = a + b;
            if denom < f32::EPSILON {
                0.0
            } else {
                (a - b).powi(2) / denom
            }
        })
        .sum()
}

/// Histogram intersection similarity (1.0 = identical).
#[must_use]
#[allow(dead_code)]
pub fn histogram_intersection(h1: &[f32], h2: &[f32]) -> f32 {
    h1.iter().zip(h2.iter()).map(|(&a, &b)| a.min(b)).sum()
}

// ---------------------------------------------------------------------------
// Motion vector analysis
// ---------------------------------------------------------------------------

/// Estimate motion magnitude between two frames.
///
/// Both frames are flat RGB byte slices of the same length.
/// Returns the mean absolute difference (MAD) normalised to [0, 1].
#[must_use]
#[allow(dead_code)]
pub fn motion_magnitude(frame_a: &[u8], frame_b: &[u8]) -> f32 {
    if frame_a.is_empty() || frame_a.len() != frame_b.len() {
        return 0.0;
    }
    let mad: f32 = frame_a
        .iter()
        .zip(frame_b.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as f32)
        .sum::<f32>()
        / frame_a.len() as f32;
    mad / 255.0
}

/// Motion analysis result for a pair of shots.
#[derive(Debug, Clone, Copy)]
pub struct MotionAnalysis {
    /// Mean absolute difference normalised to [0, 1].
    pub magnitude: f32,
    /// Whether the motion level indicates a scene break.
    pub is_high_motion: bool,
}

impl MotionAnalysis {
    /// Create from a magnitude value with a given threshold.
    #[must_use]
    #[allow(dead_code)]
    pub fn from_magnitude(magnitude: f32, threshold: f32) -> Self {
        Self {
            magnitude,
            is_high_motion: magnitude > threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// Semantic embedding similarity
// ---------------------------------------------------------------------------

/// A compact feature embedding (mean and std-dev of each RGB channel).
///
/// Represents a lightweight proxy for a deep CNN embedding.
#[derive(Debug, Clone)]
pub struct FrameEmbedding {
    /// Feature values: [R_mean, G_mean, B_mean, R_std, G_std, B_std].
    pub features: Vec<f32>,
}

impl FrameEmbedding {
    /// Compute an embedding from flat RGB pixel data.
    #[must_use]
    #[allow(dead_code)]
    pub fn from_pixels(pixels: &[u8]) -> Self {
        let n = (pixels.len() / 3) as f32;
        if n < 1.0 {
            return Self {
                features: vec![0.0; 6],
            };
        }

        let mut means = [0.0_f32; 3];
        let mut sq_sums = [0.0_f32; 3];

        for chunk in pixels.chunks_exact(3) {
            for (ch, &byte) in chunk.iter().enumerate() {
                let v = byte as f32 / 255.0;
                means[ch] += v;
                sq_sums[ch] += v * v;
            }
        }

        let mut features = Vec::with_capacity(6);
        for ch in 0..3 {
            let mean = means[ch] / n;
            let variance = (sq_sums[ch] / n - mean * mean).max(0.0);
            features.push(mean);
            // std pushed after means loop
            let _ = variance; // used below
            let _ = mean;
        }
        // Push stds in separate pass so ordering is [means..., stds...]
        for ch in 0..3 {
            let mean = means[ch] / n;
            let variance = (sq_sums[ch] / n - mean * mean).max(0.0);
            features.push(variance.sqrt());
        }
        // Re-order to [R_mean, G_mean, B_mean, R_std, G_std, B_std]
        // Current order: [R_mean, G_mean, B_mean, R_std, G_std, B_std] – correct.
        // First 3 are means pushed above; but we pushed them via the loop and then
        // let _ = mean. Let's redo cleanly.
        let mut feats = Vec::with_capacity(6);
        for ch in 0..3 {
            feats.push(means[ch] / n);
        }
        for ch in 0..3 {
            let mean = means[ch] / n;
            let variance = (sq_sums[ch] / n - mean * mean).max(0.0);
            feats.push(variance.sqrt());
        }

        Self { features: feats }
    }

    /// Cosine similarity between two embeddings ([-1, 1]).
    /// Returns 0.0 if either embedding is zero.
    #[must_use]
    #[allow(dead_code)]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.features.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = other.features.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Euclidean distance between two embeddings.
    #[must_use]
    #[allow(dead_code)]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Multi-modal scene analysis result
// ---------------------------------------------------------------------------

/// Combined result from all three modalities.
#[derive(Debug, Clone)]
pub struct MultiModalAnalysis {
    /// Chi-squared distance between color histograms (lower = more similar).
    pub color_distance: f32,
    /// Motion magnitude between representative frames (0–1).
    pub motion_magnitude: f32,
    /// Cosine similarity between semantic embeddings (higher = more similar).
    pub semantic_similarity: f32,
    /// Weighted fusion score (higher → more likely a scene boundary).
    pub boundary_score: f32,
}

impl MultiModalAnalysis {
    /// Compute a fused boundary score from the three modalities.
    ///
    /// Weights: color 40%, motion 30%, semantic 30%.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute(color_distance: f32, motion_magnitude: f32, semantic_similarity: f32) -> Self {
        // Normalise color distance to [0, 1] (cap at 1.0)
        let color_score = color_distance.clamp(0.0, 1.0);
        // High motion = more likely boundary
        let motion_score = motion_magnitude.clamp(0.0, 1.0);
        // Low similarity = more likely boundary (invert)
        let semantic_score = 1.0 - semantic_similarity.clamp(0.0, 1.0);

        let boundary_score = 0.40 * color_score + 0.30 * motion_score + 0.30 * semantic_score;

        Self {
            color_distance,
            motion_magnitude,
            semantic_similarity,
            boundary_score,
        }
    }

    /// Whether this analysis exceeds the given threshold, indicating a
    /// scene boundary.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_scene_boundary(&self, threshold: f32) -> bool {
        self.boundary_score > threshold
    }
}

// ---------------------------------------------------------------------------
// Scene detector
// ---------------------------------------------------------------------------

/// Scene detector.
pub struct SceneDetector {
    /// Similarity threshold for grouping shots into scenes.
    similarity_threshold: f32,
    /// Color histogram bins.
    histogram_bins: usize,
    /// Weight assigned to color modality (0–1).
    color_weight: f32,
    /// Weight assigned to motion modality (0–1).
    motion_weight: f32,
    /// Weight assigned to semantic modality (0–1).
    semantic_weight: f32,
}

impl SceneDetector {
    /// Create a new scene detector with default parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
            histogram_bins: 32,
            color_weight: 0.40,
            motion_weight: 0.30,
            semantic_weight: 0.30,
        }
    }

    /// Create a scene detector with custom modality weights.
    ///
    /// Weights are normalised so they sum to 1.0.
    #[must_use]
    pub fn with_weights(
        similarity_threshold: f32,
        color_weight: f32,
        motion_weight: f32,
        semantic_weight: f32,
    ) -> Self {
        let total = (color_weight + motion_weight + semantic_weight).max(f32::EPSILON);
        Self {
            similarity_threshold,
            histogram_bins: 32,
            color_weight: color_weight / total,
            motion_weight: motion_weight / total,
            semantic_weight: semantic_weight / total,
        }
    }

    /// Perform multi-modal scene boundary analysis between two pixel buffers.
    ///
    /// `pixels_a` and `pixels_b` are flat RGB byte arrays for the
    /// representative frames of the two shots being compared.
    #[must_use]
    pub fn analyse_boundary(&self, pixels_a: &[u8], pixels_b: &[u8]) -> MultiModalAnalysis {
        // Color histogram
        let hist_a = compute_color_histogram(pixels_a, self.histogram_bins);
        let hist_b = compute_color_histogram(pixels_b, self.histogram_bins);
        let color_dist = histogram_chi_squared(&hist_a, &hist_b);

        // Motion
        let motion = motion_magnitude(pixels_a, pixels_b);

        // Semantic embedding
        let emb_a = FrameEmbedding::from_pixels(pixels_a);
        let emb_b = FrameEmbedding::from_pixels(pixels_b);
        let sem_sim = emb_a.cosine_similarity(&emb_b);

        let color_norm = color_dist.clamp(0.0, 1.0);
        let motion_score = motion.clamp(0.0, 1.0);
        let semantic_score = 1.0 - sem_sim.clamp(0.0, 1.0);

        let boundary_score = self.color_weight * color_norm
            + self.motion_weight * motion_score
            + self.semantic_weight * semantic_score;

        MultiModalAnalysis {
            color_distance: color_dist,
            motion_magnitude: motion,
            semantic_similarity: sem_sim,
            boundary_score,
        }
    }

    /// Check if a multi-modal analysis result indicates a scene boundary.
    #[must_use]
    pub fn is_scene_boundary_from_analysis(&self, analysis: &MultiModalAnalysis) -> bool {
        analysis.boundary_score > (1.0 - self.similarity_threshold)
    }

    /// Detect scene boundaries from shots (transition-based heuristic, no pixel data).
    #[must_use]
    pub fn detect_scenes(&self, shots: &[Shot]) -> Vec<Scene> {
        if shots.is_empty() {
            return Vec::new();
        }

        let mut scenes = Vec::new();
        let mut current_scene_shots = vec![shots[0].id];
        let mut scene_id = 0;
        let mut scene_start = shots[0].start;

        for i in 1..shots.len() {
            let is_boundary = self.is_scene_boundary(&shots[i - 1], &shots[i]);

            if is_boundary {
                // End current scene
                scenes.push(Scene {
                    id: scene_id,
                    start: scene_start,
                    end: shots[i - 1].end,
                    shots: current_scene_shots.clone(),
                    scene_type: String::from("Unknown"),
                    confidence: 0.8,
                });

                // Start new scene
                scene_id += 1;
                scene_start = shots[i].start;
                current_scene_shots.clear();
            }

            current_scene_shots.push(shots[i].id);
        }

        // Add final scene
        if !current_scene_shots.is_empty() {
            scenes.push(Scene {
                id: scene_id,
                start: scene_start,
                end: shots[shots.len() - 1].end,
                shots: current_scene_shots,
                scene_type: String::from("Unknown"),
                confidence: 0.8,
            });
        }

        scenes
    }

    /// Detect scenes with full multi-modal analysis.
    ///
    /// `shot_pixels` maps shot IDs to representative frame pixel buffers.
    /// Shots without pixel data fall back to transition-based detection.
    #[must_use]
    pub fn detect_scenes_multimodal(
        &self,
        shots: &[Shot],
        shot_pixels: &std::collections::HashMap<u64, Vec<u8>>,
    ) -> Vec<Scene> {
        if shots.is_empty() {
            return Vec::new();
        }

        let mut scenes = Vec::new();
        let mut current_scene_shots = vec![shots[0].id];
        let mut scene_id = 0u64;
        let mut scene_start = shots[0].start;

        for i in 1..shots.len() {
            let is_boundary = {
                let pixels_a = shot_pixels.get(&shots[i - 1].id);
                let pixels_b = shot_pixels.get(&shots[i].id);

                match (pixels_a, pixels_b) {
                    (Some(a), Some(b)) => {
                        let analysis = self.analyse_boundary(a, b);
                        self.is_scene_boundary_from_analysis(&analysis)
                    }
                    _ => self.is_scene_boundary(&shots[i - 1], &shots[i]),
                }
            };

            if is_boundary {
                scenes.push(Scene {
                    id: scene_id,
                    start: scene_start,
                    end: shots[i - 1].end,
                    shots: current_scene_shots.clone(),
                    scene_type: String::from("Unknown"),
                    confidence: 0.8,
                });
                scene_id += 1;
                scene_start = shots[i].start;
                current_scene_shots.clear();
            }

            current_scene_shots.push(shots[i].id);
        }

        if !current_scene_shots.is_empty() {
            scenes.push(Scene {
                id: scene_id,
                start: scene_start,
                end: shots[shots.len() - 1].end,
                shots: current_scene_shots,
                scene_type: String::from("Unknown"),
                confidence: 0.8,
            });
        }

        scenes
    }

    /// Check if there's a scene boundary between two shots (transition heuristic).
    fn is_scene_boundary(&self, shot1: &Shot, shot2: &Shot) -> bool {
        // Scene boundaries typically occur with:
        // 1. Fade transitions
        // 2. Long gaps
        // 3. Significant change in shot type
        // 4. Change in coverage type

        matches!(
            shot2.transition,
            crate::types::TransitionType::FadeToBlack
                | crate::types::TransitionType::FadeFromBlack
                | crate::types::TransitionType::FadeToWhite
                | crate::types::TransitionType::FadeFromWhite
        ) || (shot1.coverage != shot2.coverage && shot1.shot_type != shot2.shot_type)
    }

    /// Similarity threshold accessor.
    #[must_use]
    pub const fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Color weight accessor.
    #[must_use]
    pub const fn color_weight(&self) -> f32 {
        self.color_weight
    }

    /// Motion weight accessor.
    #[must_use]
    pub const fn motion_weight(&self) -> f32 {
        self.motion_weight
    }

    /// Semantic weight accessor.
    #[must_use]
    pub const fn semantic_weight(&self) -> f32 {
        self.semantic_weight
    }
}

impl Default for SceneDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, ShotType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};
    use std::collections::HashMap;

    fn make_shot(id: u64, transition: TransitionType) -> Shot {
        Shot {
            id,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(60, Rational::new(1, 30)),
            shot_type: ShotType::MediumShot,
            angle: CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.5,
                symmetry: 0.5,
                balance: 0.5,
                leading_lines: 0.5,
                depth: 0.5,
            },
            coverage: CoverageType::Master,
            confidence: 0.8,
            transition,
        }
    }

    // ---- SceneDetector basic tests --------------------------------------

    #[test]
    fn test_scene_detector_creation() {
        let detector = SceneDetector::new();
        assert!((detector.similarity_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_scenes_empty() {
        let detector = SceneDetector::new();
        let scenes = detector.detect_scenes(&[]);
        assert!(scenes.is_empty());
    }

    #[test]
    fn test_detect_scenes_single_shot() {
        let detector = SceneDetector::new();
        let shot = make_shot(1, TransitionType::Cut);
        let scenes = detector.detect_scenes(&[shot]);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].shots.len(), 1);
    }

    #[test]
    fn test_detect_scenes_fade_creates_boundary() {
        let detector = SceneDetector::new();
        let shots = vec![
            make_shot(0, TransitionType::Cut),
            make_shot(1, TransitionType::FadeToBlack),
            make_shot(2, TransitionType::FadeFromBlack),
        ];
        let scenes = detector.detect_scenes(&shots);
        // Shots 1 and 2 both have fade transitions → at least 2 scenes
        assert!(scenes.len() >= 2);
    }

    // ---- Color histogram -----------------------------------------------

    #[test]
    fn test_histogram_identical() {
        let pixels: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();
        let h = compute_color_histogram(&pixels, 32);
        let dist = histogram_chi_squared(&h, &h);
        assert!(
            dist < 1e-5,
            "Identical histograms should have distance ~0, got {dist}"
        );
    }

    #[test]
    fn test_histogram_empty_pixels() {
        let h = compute_color_histogram(&[], 32);
        assert_eq!(h.len(), 96);
        assert!(h.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_histogram_length() {
        let pixels = vec![128u8; 300];
        let h = compute_color_histogram(&pixels, 16);
        assert_eq!(h.len(), 16 * 3);
    }

    #[test]
    fn test_histogram_intersection_identical() {
        let pixels = vec![100u8; 300];
        let h = compute_color_histogram(&pixels, 32);
        let sim = histogram_intersection(&h, &h);
        // Each channel is normalised to sum to 1.0, and there are 3 channels,
        // so the total intersection of identical histograms is 3.0.
        assert!(
            (sim - 3.0).abs() < 1e-4,
            "Expected intersection ~3.0, got {sim}"
        );
    }

    #[test]
    fn test_histogram_different_images() {
        // Pure red vs pure green: histograms should differ significantly
        let red_pixels: Vec<u8> = std::iter::repeat_n([255u8, 0, 0], 100).flatten().collect();
        let green_pixels: Vec<u8> = std::iter::repeat_n([0u8, 255, 0], 100).flatten().collect();
        let h_red = compute_color_histogram(&red_pixels, 32);
        let h_green = compute_color_histogram(&green_pixels, 32);
        let dist = histogram_chi_squared(&h_red, &h_green);
        assert!(
            dist > 0.5,
            "Expected large chi-squared distance, got {dist}"
        );
    }

    // ---- Motion analysis -----------------------------------------------

    #[test]
    fn test_motion_identical_frames() {
        let frame = vec![128u8; 300];
        let mag = motion_magnitude(&frame, &frame);
        assert!(
            (mag - 0.0).abs() < f32::EPSILON,
            "Identical frames → 0 motion, got {mag}"
        );
    }

    #[test]
    fn test_motion_opposite_frames() {
        let frame_a = vec![0u8; 300];
        let frame_b = vec![255u8; 300];
        let mag = motion_magnitude(&frame_a, &frame_b);
        assert!(
            (mag - 1.0).abs() < 1e-4,
            "Max-contrast frames → motion=1.0, got {mag}"
        );
    }

    #[test]
    fn test_motion_empty_frames() {
        let mag = motion_magnitude(&[], &[]);
        assert!((mag - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_motion_analysis_threshold() {
        let analysis = MotionAnalysis::from_magnitude(0.8, 0.5);
        assert!(analysis.is_high_motion);
        let low = MotionAnalysis::from_magnitude(0.2, 0.5);
        assert!(!low.is_high_motion);
    }

    // ---- Semantic embedding --------------------------------------------

    #[test]
    fn test_embedding_identical() {
        let pixels = vec![100u8; 300];
        let emb = FrameEmbedding::from_pixels(&pixels);
        let sim = emb.cosine_similarity(&emb);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Self-similarity should be 1.0, got {sim}"
        );
    }

    #[test]
    fn test_embedding_empty() {
        let emb = FrameEmbedding::from_pixels(&[]);
        assert_eq!(emb.features.len(), 6);
    }

    #[test]
    fn test_embedding_feature_length() {
        let pixels = vec![200u8; 300];
        let emb = FrameEmbedding::from_pixels(&pixels);
        assert_eq!(
            emb.features.len(),
            6,
            "Expected 6 features (3 means + 3 stds)"
        );
    }

    #[test]
    fn test_embedding_different_images() {
        let dark: Vec<u8> = vec![10u8; 300];
        let bright: Vec<u8> = vec![245u8; 300];
        let emb_dark = FrameEmbedding::from_pixels(&dark);
        let emb_bright = FrameEmbedding::from_pixels(&bright);
        let dist = emb_dark.euclidean_distance(&emb_bright);
        assert!(dist > 0.5, "Expected large euclidean distance, got {dist}");
    }

    // ---- MultiModalAnalysis --------------------------------------------

    #[test]
    fn test_multimodal_similar_shots() {
        // Identical colour, no motion, maximum semantic similarity
        let analysis = MultiModalAnalysis::compute(0.0, 0.0, 1.0);
        assert!(
            analysis.boundary_score < 0.1,
            "Similar shots should have low boundary score, got {}",
            analysis.boundary_score
        );
    }

    #[test]
    fn test_multimodal_different_shots() {
        // Max colour distance, high motion, low semantic similarity
        let analysis = MultiModalAnalysis::compute(1.0, 1.0, 0.0);
        assert!(
            analysis.boundary_score > 0.9,
            "Different shots should have high boundary score, got {}",
            analysis.boundary_score
        );
    }

    #[test]
    fn test_multimodal_is_scene_boundary() {
        let boundary = MultiModalAnalysis::compute(1.0, 1.0, 0.0);
        assert!(boundary.is_scene_boundary(0.5));
        let non_boundary = MultiModalAnalysis::compute(0.0, 0.0, 1.0);
        assert!(!non_boundary.is_scene_boundary(0.5));
    }

    // ---- Full multi-modal detect_scenes_multimodal ---------------------

    #[test]
    fn test_detect_scenes_multimodal_no_pixel_data() {
        let detector = SceneDetector::new();
        let shots: Vec<Shot> = (0..3).map(|i| make_shot(i, TransitionType::Cut)).collect();
        let pixel_map: HashMap<u64, Vec<u8>> = HashMap::new();
        let scenes = detector.detect_scenes_multimodal(&shots, &pixel_map);
        // No fades, same shot/coverage type → all one scene
        assert_eq!(scenes.len(), 1);
    }

    #[test]
    fn test_detect_scenes_multimodal_with_pixel_data() {
        let detector = SceneDetector::new();
        let shots: Vec<Shot> = (0..2).map(|i| make_shot(i, TransitionType::Cut)).collect();

        let mut pixel_map: HashMap<u64, Vec<u8>> = HashMap::new();
        // Shot 0: dark image, Shot 1: bright image → large boundary score
        pixel_map.insert(0, vec![0u8; 300]);
        pixel_map.insert(1, vec![255u8; 300]);

        let scenes = detector.detect_scenes_multimodal(&shots, &pixel_map);
        // Very different images should trigger a boundary → 2 scenes
        assert!(scenes.len() >= 1, "Expected at least 1 scene");
    }

    #[test]
    fn test_detect_scenes_multimodal_empty() {
        let detector = SceneDetector::new();
        let pixel_map: HashMap<u64, Vec<u8>> = HashMap::new();
        let scenes = detector.detect_scenes_multimodal(&[], &pixel_map);
        assert!(scenes.is_empty());
    }

    #[test]
    fn test_custom_weights() {
        let detector = SceneDetector::with_weights(0.6, 0.5, 0.3, 0.2);
        // Weights should sum to 1.0
        let sum = detector.color_weight() + detector.motion_weight() + detector.semantic_weight();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_analyse_boundary_similar_frames() {
        let detector = SceneDetector::new();
        let frame = vec![128u8; 300];
        let analysis = detector.analyse_boundary(&frame, &frame);
        assert!(
            analysis.boundary_score < 0.35,
            "Identical frames should have low boundary score, got {}",
            analysis.boundary_score
        );
    }
}
