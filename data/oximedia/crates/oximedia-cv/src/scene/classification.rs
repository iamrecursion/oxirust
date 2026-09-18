//! Scene classification for video frames.
//!
//! This module provides a lightweight, pure-Rust scene classifier that
//! categorises a single video frame (or a short window of frames) into
//! one of several [`SceneType`] categories using hand-crafted features
//! derived from the luma plane:
//!
//! - Average brightness
//! - Edge density (Sobel magnitude)
//! - Motion energy (inter-frame absolute difference)
//! - Colour diversity (standard deviation of luma)
//! - Sky ratio (fraction of very-bright pixels in the top third)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::scene::classification::{
//!     extract_scene_features, classify_scene, brightness_from_luma, edge_density,
//! };
//!
//! let luma = vec![128u8; 320 * 240];
//! let features = extract_scene_features(&luma, 320, 240);
//! let cls = classify_scene(&features);
//! ```

#![allow(dead_code)]

/// High-level category of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneType {
    /// Indoor, enclosed environment.
    Indoor,
    /// Outdoor, open environment.
    Outdoor,
    /// Night-time scene (dark).
    NightScene,
    /// Daytime scene (bright).
    DayScene,
    /// High-motion action sequence.
    Action,
    /// Talking-head / dialogue.
    Talking,
    /// Interview or news format.
    Interview,
    /// Sports content with rapid movement.
    Sport,
    /// Natural landscape (nature, countryside).
    Nature,
    /// Urban / city environment.
    Urban,
}

impl SceneType {
    /// Human-readable label for the scene type.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::scene::classification::SceneType;
    ///
    /// assert_eq!(SceneType::Indoor.label(), "Indoor");
    /// assert_eq!(SceneType::NightScene.label(), "Night Scene");
    /// ```
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Indoor => "Indoor",
            Self::Outdoor => "Outdoor",
            Self::NightScene => "Night Scene",
            Self::DayScene => "Day Scene",
            Self::Action => "Action",
            Self::Talking => "Talking",
            Self::Interview => "Interview",
            Self::Sport => "Sport",
            Self::Nature => "Nature",
            Self::Urban => "Urban",
        }
    }
}

/// Result of classifying a single scene.
#[derive(Debug, Clone)]
pub struct SceneClassification {
    /// Primary detected scene type.
    pub scene_type: SceneType,
    /// Confidence in the primary label (0.0–1.0).
    pub confidence: f64,
    /// Secondary scene types ordered by likelihood (type, confidence).
    pub secondary: Vec<(SceneType, f64)>,
}

impl SceneClassification {
    /// Create a new classification result.
    #[must_use]
    pub fn new(scene_type: SceneType, confidence: f64, secondary: Vec<(SceneType, f64)>) -> Self {
        Self {
            scene_type,
            confidence,
            secondary,
        }
    }
}

/// Low-level scene features extracted from a luma plane.
#[derive(Debug, Clone)]
pub struct SceneFeatures {
    /// Mean luma value in `[0.0, 255.0]`.
    pub avg_brightness: f64,
    /// Fraction of pixels with a strong gradient (0.0–1.0).
    pub edge_density: f64,
    /// Mean inter-frame absolute-difference energy (0.0–255.0). `0.0` if no
    /// previous frame is provided.
    pub motion_energy: f64,
    /// Standard deviation of luma values (proxy for colour diversity).
    pub color_diversity: f64,
    /// Fraction of top-third pixels brighter than 200 (sky heuristic).
    pub sky_ratio: f64,
}

// ─── Public feature extraction ────────────────────────────────────────────────

/// Compute the average luma value of a luma plane.
///
/// # Arguments
///
/// * `luma` - Row-major 8-bit luma plane of length `width * height`.
///
/// # Returns
///
/// Mean value in `[0.0, 255.0]`, or `0.0` if the slice is empty.
///
/// # Examples
///
/// ```
/// use oximedia_cv::scene::classification::brightness_from_luma;
///
/// let luma = vec![100u8; 100];
/// assert!((brightness_from_luma(&luma) - 100.0).abs() < 0.01);
/// ```
#[must_use]
pub fn brightness_from_luma(luma: &[u8]) -> f64 {
    if luma.is_empty() {
        return 0.0;
    }
    let sum: u64 = luma.iter().map(|&p| p as u64).sum();
    sum as f64 / luma.len() as f64
}

/// Compute edge density using the Sobel operator on a luma plane.
///
/// The density is the fraction of pixels whose Sobel gradient magnitude
/// exceeds a fixed threshold of 30 (on a 0–255 scale).
///
/// # Arguments
///
/// * `luma`   - Row-major 8-bit luma plane.
/// * `width`  - Image width.
/// * `height` - Image height.
///
/// # Returns
///
/// Value in `[0.0, 1.0]`.
///
/// # Examples
///
/// ```
/// use oximedia_cv::scene::classification::edge_density;
///
/// let luma = vec![128u8; 16 * 16];
/// let density = edge_density(&luma, 16, 16);
/// // Uniform image → no edges
/// assert!(density < 0.05);
/// ```
#[must_use]
pub fn edge_density(luma: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 || luma.len() < width * height {
        return 0.0;
    }

    let threshold = 30.0_f64;
    let mut strong = 0u64;
    let total = ((width - 2) * (height - 2)) as u64;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = |dy: usize, dx: usize| luma[(y + dy - 1) * width + (x + dx - 1)] as f64;

            // Sobel Gx
            let gx =
                -idx(0, 0) + idx(0, 2) - 2.0 * idx(1, 0) + 2.0 * idx(1, 2) - idx(2, 0) + idx(2, 2);
            // Sobel Gy
            let gy =
                -idx(0, 0) - 2.0 * idx(0, 1) - idx(0, 2) + idx(2, 0) + 2.0 * idx(2, 1) + idx(2, 2);

            let mag = (gx * gx + gy * gy).sqrt();
            if mag > threshold {
                strong += 1;
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        strong as f64 / total as f64
    }
}

/// Extract all scene features from a luma plane.
///
/// `prev_luma` may be `None` when no prior frame is available; in that case
/// `motion_energy` is set to `0.0`.
///
/// # Arguments
///
/// * `luma`   - Current frame luma plane (`width * height` bytes).
/// * `width`  - Image width.
/// * `height` - Image height.
///
/// # Examples
///
/// ```
/// use oximedia_cv::scene::classification::extract_scene_features;
///
/// let luma = vec![128u8; 320 * 240];
/// let f = extract_scene_features(&luma, 320, 240);
/// assert!((f.avg_brightness - 128.0).abs() < 0.1);
/// ```
#[must_use]
pub fn extract_scene_features(luma: &[u8], width: usize, height: usize) -> SceneFeatures {
    let avg_brightness = brightness_from_luma(luma);
    let edge_dens = edge_density(luma, width, height);

    // Colour diversity: standard deviation of luma
    let color_diversity = {
        if luma.is_empty() {
            0.0
        } else {
            let mean = avg_brightness;
            let variance: f64 = luma
                .iter()
                .map(|&p| {
                    let diff = p as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / luma.len() as f64;
            variance.sqrt()
        }
    };

    // Sky ratio: fraction of top-third pixels above 200
    let sky_ratio = {
        let top_rows = (height / 3).max(1);
        let top_pixels = top_rows * width;
        if top_pixels == 0 || luma.len() < top_pixels {
            0.0
        } else {
            let bright: usize = luma[..top_pixels].iter().filter(|&&p| p > 200).count();
            bright as f64 / top_pixels as f64
        }
    };

    SceneFeatures {
        avg_brightness,
        edge_density: edge_dens,
        motion_energy: 0.0, // caller may override
        color_diversity,
        sky_ratio,
    }
}

/// Classify a scene from its feature vector.
///
/// Returns a [`SceneClassification`] with the most-likely scene type and
/// secondary alternatives.
///
/// The heuristic rules (in priority order):
///
/// 1. Very dark (avg < 40) → `NightScene`
/// 2. High sky ratio (> 0.4) AND bright (> 150) → `Outdoor`/`DayScene`
/// 3. High motion energy (> 50) AND high edges (> 0.3) → `Sport`/`Action`
/// 4. High edges AND medium brightness → `Urban`
/// 5. Low edges AND high sky + brightness → `Nature`
/// 6. Bright (> 160) AND very low edges (< 0.05) → `Interview`/`Talking`
/// 7. Otherwise → `Indoor`
///
/// # Examples
///
/// ```
/// use oximedia_cv::scene::classification::{SceneFeatures, classify_scene, SceneType};
///
/// let features = SceneFeatures {
///     avg_brightness: 20.0,
///     edge_density: 0.05,
///     motion_energy: 0.0,
///     color_diversity: 10.0,
///     sky_ratio: 0.0,
/// };
/// let cls = classify_scene(&features);
/// assert_eq!(cls.scene_type, SceneType::NightScene);
/// ```
#[must_use]
pub fn classify_scene(features: &SceneFeatures) -> SceneClassification {
    let b = features.avg_brightness;
    let e = features.edge_density;
    let m = features.motion_energy;
    let s = features.sky_ratio;

    // --- Rule 1: night ---
    if b < 40.0 {
        return SceneClassification::new(
            SceneType::NightScene,
            1.0 - b / 40.0,
            vec![(SceneType::Indoor, 0.3)],
        );
    }

    // --- Rule 2: outdoor / day ---
    if s > 0.4 && b > 150.0 {
        let conf = ((s - 0.4) / 0.6).min(1.0) * 0.7 + (b - 150.0) / 105.0 * 0.3;
        return SceneClassification::new(
            SceneType::Outdoor,
            conf.clamp(0.0, 1.0),
            vec![(SceneType::DayScene, 0.6), (SceneType::Nature, 0.4)],
        );
    }

    // --- Rule 3: sport / action ---
    if m > 50.0 && e > 0.3 {
        let conf = ((m - 50.0) / 200.0).min(1.0) * 0.5 + ((e - 0.3) / 0.7).min(1.0) * 0.5;
        return SceneClassification::new(
            SceneType::Sport,
            conf.clamp(0.0, 1.0),
            vec![(SceneType::Action, 0.7)],
        );
    }

    // --- Rule 4: high motion alone ---
    if m > 50.0 {
        return SceneClassification::new(
            SceneType::Action,
            ((m - 50.0) / 200.0).min(1.0),
            vec![(SceneType::Sport, 0.5)],
        );
    }

    // --- Rule 5: high edges → urban ---
    if e > 0.25 {
        let conf = ((e - 0.25) / 0.75).min(1.0);
        return SceneClassification::new(
            SceneType::Urban,
            conf,
            vec![(SceneType::Outdoor, 0.4), (SceneType::Indoor, 0.3)],
        );
    }

    // --- Rule 6: sky + bright but low edges → nature ---
    if s > 0.2 && b > 130.0 && e < 0.1 {
        return SceneClassification::new(
            SceneType::Nature,
            (s - 0.2) / 0.8,
            vec![(SceneType::Outdoor, 0.5), (SceneType::DayScene, 0.4)],
        );
    }

    // --- Rule 7: bright, low-edge indoor → talking / interview ---
    if b > 160.0 && e < 0.05 {
        return SceneClassification::new(
            SceneType::Talking,
            0.6,
            vec![(SceneType::Interview, 0.5), (SceneType::Indoor, 0.4)],
        );
    }

    // --- Default: indoor ---
    SceneClassification::new(
        SceneType::Indoor,
        0.5,
        vec![(SceneType::Talking, 0.3), (SceneType::Urban, 0.2)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(avg: f64, edge: f64, motion: f64, sky: f64, div: f64) -> SceneFeatures {
        SceneFeatures {
            avg_brightness: avg,
            edge_density: edge,
            motion_energy: motion,
            color_diversity: div,
            sky_ratio: sky,
        }
    }

    #[test]
    fn test_brightness_from_luma_uniform() {
        let luma = vec![100u8; 200];
        assert!((brightness_from_luma(&luma) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_brightness_from_luma_empty() {
        assert_eq!(brightness_from_luma(&[]), 0.0);
    }

    #[test]
    fn test_brightness_from_luma_mixed() {
        let luma = vec![0u8, 255];
        assert!((brightness_from_luma(&luma) - 127.5).abs() < 0.01);
    }

    #[test]
    fn test_edge_density_uniform_near_zero() {
        let luma = vec![128u8; 32 * 32];
        let d = edge_density(&luma, 32, 32);
        assert!(d < 0.01);
    }

    #[test]
    fn test_edge_density_checkerboard_nonzero() {
        // Use 4-pixel blocks so Sobel can detect the boundaries.
        // A 1-pixel checkerboard is above Nyquist for a 3×3 Sobel kernel
        // (gradient is zero at the centre of each cell).
        let w = 32usize;
        let h = 32usize;
        let block = 4usize; // each "square" in the checkerboard is 4×4 pixels
        let luma: Vec<u8> = (0..w * h)
            .map(|i| {
                let row = i / w;
                let col = i % w;
                if ((row / block) + (col / block)) % 2 == 0 {
                    0
                } else {
                    255
                }
            })
            .collect();
        let d = edge_density(&luma, w, h);
        assert!(
            d > 0.1,
            "block-checkerboard should have detectable edge density, got {d}"
        );
    }

    #[test]
    fn test_edge_density_too_small_image() {
        let luma = vec![128u8; 4];
        let d = edge_density(&luma, 2, 2);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_extract_scene_features_output() {
        let luma = vec![150u8; 64 * 64];
        let f = extract_scene_features(&luma, 64, 64);
        assert!((f.avg_brightness - 150.0).abs() < 0.01);
        assert!(f.edge_density < 0.01);
        assert_eq!(f.motion_energy, 0.0);
        assert!(f.color_diversity < 1.0);
    }

    #[test]
    fn test_scene_type_label() {
        assert_eq!(SceneType::Indoor.label(), "Indoor");
        assert_eq!(SceneType::Outdoor.label(), "Outdoor");
        assert_eq!(SceneType::NightScene.label(), "Night Scene");
        assert_eq!(SceneType::DayScene.label(), "Day Scene");
        assert_eq!(SceneType::Action.label(), "Action");
        assert_eq!(SceneType::Talking.label(), "Talking");
        assert_eq!(SceneType::Interview.label(), "Interview");
        assert_eq!(SceneType::Sport.label(), "Sport");
        assert_eq!(SceneType::Nature.label(), "Nature");
        assert_eq!(SceneType::Urban.label(), "Urban");
    }

    #[test]
    fn test_classify_night_scene() {
        let f = make_features(20.0, 0.02, 0.0, 0.0, 5.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::NightScene);
        assert!(cls.confidence > 0.0);
    }

    #[test]
    fn test_classify_outdoor_day() {
        let f = make_features(200.0, 0.10, 0.0, 0.6, 40.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Outdoor);
    }

    #[test]
    fn test_classify_sport() {
        let f = make_features(140.0, 0.4, 120.0, 0.1, 30.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Sport);
    }

    #[test]
    fn test_classify_action() {
        let f = make_features(120.0, 0.1, 80.0, 0.05, 20.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Action);
    }

    #[test]
    fn test_classify_urban() {
        let f = make_features(130.0, 0.4, 0.0, 0.1, 35.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Urban);
    }

    #[test]
    fn test_classify_nature() {
        // sky_ratio must be >0.2 but ≤0.4 (else Outdoor rule fires first),
        // brightness must be >130 but ≤150, and edge density <0.1.
        let f = make_features(140.0, 0.03, 0.0, 0.3, 25.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Nature);
    }

    #[test]
    fn test_classify_talking() {
        let f = make_features(200.0, 0.01, 0.0, 0.05, 15.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Talking);
    }

    #[test]
    fn test_classify_indoor_default() {
        let f = make_features(100.0, 0.05, 0.0, 0.05, 20.0);
        let cls = classify_scene(&f);
        assert_eq!(cls.scene_type, SceneType::Indoor);
    }

    #[test]
    fn test_classification_has_secondary() {
        let f = make_features(20.0, 0.02, 0.0, 0.0, 5.0);
        let cls = classify_scene(&f);
        assert!(!cls.secondary.is_empty());
    }

    #[test]
    fn test_confidence_range() {
        let cases = [
            make_features(20.0, 0.0, 0.0, 0.0, 0.0),
            make_features(200.0, 0.6, 150.0, 0.7, 40.0),
            make_features(150.0, 0.01, 0.0, 0.55, 20.0),
        ];
        for f in &cases {
            let cls = classify_scene(f);
            assert!(
                cls.confidence >= 0.0 && cls.confidence <= 1.0,
                "confidence out of range: {}",
                cls.confidence
            );
        }
    }
}
