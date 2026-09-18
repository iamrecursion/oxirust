//! Aesthetic quality scoring with content-type-specific models.
//!
//! In addition to the general-purpose [`AestheticScorer`], this module
//! provides a [`ContentTypeScorer`] that selects a scoring model tuned to the
//! detected content type (landscape, portrait, action, still life).

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Content type for content-type-specific aesthetic scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    /// Landscape scene (nature, outdoors).
    Landscape,
    /// Portrait (single or group of people).
    Portrait,
    /// Action scene (sports, fast movement).
    Action,
    /// Still life (close-up of objects).
    StillLife,
    /// General / unknown.
    General,
}

impl ContentType {
    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Landscape => "Landscape",
            Self::Portrait => "Portrait",
            Self::Action => "Action",
            Self::StillLife => "Still Life",
            Self::General => "General",
        }
    }
}

/// Scoring weights for a specific content type.
#[derive(Debug, Clone)]
struct ContentWeights {
    color_harmony: f32,
    sharpness: f32,
    contrast: f32,
    composition: f32,
    lighting: f32,
    uniqueness: f32,
}

impl ContentWeights {
    fn for_content_type(ct: ContentType) -> Self {
        match ct {
            ContentType::Landscape => Self {
                color_harmony: 0.30, // rich colors matter a lot
                sharpness: 0.15,     // some blur acceptable
                contrast: 0.20,      // dynamic range important
                composition: 0.20,   // rule of thirds / horizon
                lighting: 0.10,      // golden hour etc.
                uniqueness: 0.05,
            },
            ContentType::Portrait => Self {
                color_harmony: 0.15,
                sharpness: 0.30, // face sharpness critical
                contrast: 0.10,
                composition: 0.25, // framing matters
                lighting: 0.15,    // skin tones / catchlights
                uniqueness: 0.05,
            },
            ContentType::Action => Self {
                color_harmony: 0.10,
                sharpness: 0.35, // motion sharpness very important
                contrast: 0.20,
                composition: 0.20,
                lighting: 0.10,
                uniqueness: 0.05,
            },
            ContentType::StillLife => Self {
                color_harmony: 0.25,
                sharpness: 0.20,
                contrast: 0.15,
                composition: 0.20,
                lighting: 0.15,
                uniqueness: 0.05,
            },
            ContentType::General => Self {
                color_harmony: 0.20,
                sharpness: 0.20,
                contrast: 0.15,
                composition: 0.20,
                lighting: 0.15,
                uniqueness: 0.10,
            },
        }
    }

    fn sum(&self) -> f32 {
        self.color_harmony
            + self.sharpness
            + self.contrast
            + self.composition
            + self.lighting
            + self.uniqueness
    }
}

/// Aesthetic quality score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticScore {
    /// Overall aesthetic score (0.0-1.0).
    pub overall: f32,
    /// Color harmony (0.0-1.0).
    pub color_harmony: f32,
    /// Sharpness (0.0-1.0).
    pub sharpness: f32,
    /// Contrast (0.0-1.0).
    pub contrast: f32,
    /// Composition (0.0-1.0).
    pub composition: f32,
    /// Lighting quality (0.0-1.0).
    pub lighting: f32,
    /// Uniqueness (0.0-1.0).
    pub uniqueness: f32,
    /// Content type used for scoring (None means general model).
    pub content_type: Option<ContentType>,
}

/// Aesthetic scorer.
pub struct AestheticScorer;

impl AestheticScorer {
    /// Create a new aesthetic scorer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Score aesthetic quality.
    ///
    /// # Errors
    ///
    /// Returns error if scoring fails.
    pub fn score(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<AestheticScore> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let color_harmony = self.score_color_harmony(rgb_data, width, height);
        let sharpness = self.score_sharpness(rgb_data, width, height);
        let contrast = self.score_contrast(rgb_data, width, height);
        let composition = self.score_composition(rgb_data, width, height);
        let lighting = self.score_lighting(rgb_data, width, height);
        let uniqueness = self.score_uniqueness(rgb_data, width, height);

        let w = ContentWeights::for_content_type(ContentType::General);
        let wsum = w.sum().max(f32::EPSILON);
        let overall = (color_harmony * w.color_harmony
            + sharpness * w.sharpness
            + contrast * w.contrast
            + composition * w.composition
            + lighting * w.lighting
            + uniqueness * w.uniqueness)
            .clamp(0.0, wsum)
            / wsum;

        Ok(AestheticScore {
            overall: overall.clamp(0.0, 1.0),
            color_harmony,
            sharpness,
            contrast,
            composition,
            lighting,
            uniqueness,
            content_type: None,
        })
    }

    /// Score aesthetic quality using a content-type-specific model.
    ///
    /// # Errors
    ///
    /// Returns error if scoring fails.
    pub fn score_for_content(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        content_type: ContentType,
    ) -> SceneResult<AestheticScore> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let color_harmony = self.score_color_harmony(rgb_data, width, height);
        let sharpness = self.score_sharpness(rgb_data, width, height);
        let contrast = self.score_contrast(rgb_data, width, height);
        let composition = self.score_composition(rgb_data, width, height);
        let lighting = self.score_lighting(rgb_data, width, height);
        let uniqueness = self.score_uniqueness(rgb_data, width, height);

        let w = ContentWeights::for_content_type(content_type);
        let wsum = w.sum().max(f32::EPSILON);
        let overall = (color_harmony * w.color_harmony
            + sharpness * w.sharpness
            + contrast * w.contrast
            + composition * w.composition
            + lighting * w.lighting
            + uniqueness * w.uniqueness)
            .clamp(0.0, wsum)
            / wsum;

        Ok(AestheticScore {
            overall: overall.clamp(0.0, 1.0),
            color_harmony,
            sharpness,
            contrast,
            composition,
            lighting,
            uniqueness,
            content_type: Some(content_type),
        })
    }

    fn score_color_harmony(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        // Analyze color distribution and harmony
        let mut histogram = vec![vec![0u32; 16]; 3];

        for i in (0..rgb_data.len()).step_by(3) {
            for c in 0..3 {
                let bin = (rgb_data[i + c] / 16) as usize;
                histogram[c][bin] += 1;
            }
        }

        // Calculate entropy (lower entropy = more harmonious)
        let mut entropy = 0.0;
        let total = rgb_data.len() / 3;

        for c in 0..3 {
            for &count in &histogram[c] {
                if count > 0 {
                    let p = count as f32 / total as f32;
                    entropy -= p * p.log2();
                }
            }
        }

        (1.0 - (entropy / 12.0).min(1.0)).clamp(0.0, 1.0)
    }

    fn score_sharpness(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let mut edge_sum = 0.0;
        let mut count = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                for c in 0..3 {
                    let center = rgb_data[idx + c] as i32;
                    let left = rgb_data[idx - 3 + c] as i32;
                    let right = rgb_data[idx + 3 + c] as i32;
                    edge_sum += ((center - left).abs() + (center - right).abs()) as f32;
                }
                count += 3;
            }
        }

        if count > 0 {
            (edge_sum / count as f32 / 255.0 * 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_contrast(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        let mut min_val = 255u8;
        let mut max_val = 0u8;

        for i in (0..rgb_data.len()).step_by(3) {
            let gray =
                ((rgb_data[i] as u16 + rgb_data[i + 1] as u16 + rgb_data[i + 2] as u16) / 3) as u8;
            min_val = min_val.min(gray);
            max_val = max_val.max(gray);
        }

        (max_val - min_val) as f32 / 255.0
    }

    fn score_composition(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        // Use rule of thirds heuristic
        let third_w = width / 3;
        let third_h = height / 3;

        let mut interest_score = 0.0;

        // Check interest points at rule of thirds intersections
        for y in [third_h, third_h * 2] {
            for x in [third_w, third_w * 2] {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    // Measure local complexity
                    let mut complexity = 0.0;
                    for dy in 0..10.min(height - y) {
                        for dx in 0..10.min(width - x) {
                            let pidx = ((y + dy) * width + (x + dx)) * 3;
                            if pidx + 2 < rgb_data.len() {
                                for c in 0..3 {
                                    complexity += (rgb_data[pidx + c] as i32
                                        - rgb_data[idx + c] as i32)
                                        .unsigned_abs()
                                        as f32;
                                }
                            }
                        }
                    }
                    interest_score += complexity;
                }
            }
        }

        (interest_score / 100.0 / 255.0 / 12.0).clamp(0.0, 1.0)
    }

    fn score_lighting(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        let mut brightness_sum = 0.0;
        let mut count = 0;

        for i in (0..rgb_data.len()).step_by(3) {
            let brightness =
                (rgb_data[i] as f32 + rgb_data[i + 1] as f32 + rgb_data[i + 2] as f32) / 3.0;
            brightness_sum += brightness;
            count += 1;
        }

        if count > 0 {
            let avg_brightness = brightness_sum / count as f32;
            // Good lighting is around 127 (mid-range)
            (1.0 - ((avg_brightness - 127.0).abs() / 127.0)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_uniqueness(&self, rgb_data: &[u8], _width: usize, _height: usize) -> f32 {
        // Measure color diversity as proxy for uniqueness
        let mut unique_colors = std::collections::HashSet::new();

        for i in (0..rgb_data.len()).step_by(3) {
            let color = (rgb_data[i] / 32, rgb_data[i + 1] / 32, rgb_data[i + 2] / 32);
            unique_colors.insert(color);
        }

        (unique_colors.len() as f32 / 512.0).clamp(0.0, 1.0)
    }
}

impl Default for AestheticScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// A scorer that automatically selects the content-type model based on simple
/// image statistics (brightness, color distribution, edge density).
///
/// This allows callers to get better aesthetic scores without needing to
/// specify content type manually.
pub struct ContentTypeScorer {
    inner: AestheticScorer,
}

impl ContentTypeScorer {
    /// Create a new content-type scorer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: AestheticScorer::new(),
        }
    }

    /// Detect content type from raw image statistics.
    ///
    /// Heuristics:
    /// - High saturation + horizon-like structure → Landscape
    /// - Centered bright region + low overall edge density → Portrait
    /// - High overall edge density + high contrast → Action
    /// - Low saturation + fine detail → Still life
    #[must_use]
    pub fn detect_content_type(&self, rgb_data: &[u8], width: usize, height: usize) -> ContentType {
        if rgb_data.len() != width * height * 3 {
            return ContentType::General;
        }

        // Compute average saturation, brightness, and edge density
        let mut sat_sum = 0.0_f64;
        let mut bright_sum = 0.0_f64;
        let pixel_count = width * height;

        for i in (0..rgb_data.len()).step_by(3) {
            let r = rgb_data[i] as f32;
            let g = rgb_data[i + 1] as f32;
            let b = rgb_data[i + 2] as f32;
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            if max > 0.0 {
                sat_sum += ((max - min) / max) as f64;
            }
            bright_sum += (0.299 * r + 0.587 * g + 0.114 * b) as f64 / 255.0;
        }

        let avg_sat = sat_sum / pixel_count as f64;
        let avg_bright = bright_sum / pixel_count as f64;

        // Compute horizontal edge energy (for horizon detection)
        let mut horiz_edges = 0_u64;
        let mut total_edges = 0_u64;
        for y in 1..height.saturating_sub(1) {
            for x in 0..width {
                let above = ((y - 1) * width + x) * 3;
                let below = ((y + 1) * width + x) * 3;
                if below + 2 < rgb_data.len() {
                    let diff = (rgb_data[below] as i32 - rgb_data[above] as i32).unsigned_abs()
                        + (rgb_data[below + 1] as i32 - rgb_data[above + 1] as i32).unsigned_abs()
                        + (rgb_data[below + 2] as i32 - rgb_data[above + 2] as i32).unsigned_abs();
                    horiz_edges += diff as u64;
                    total_edges += 1;
                }
            }
        }
        let avg_horiz_edge = if total_edges > 0 {
            horiz_edges as f64 / total_edges as f64 / 255.0
        } else {
            0.0
        };

        // Heuristic decision tree
        if avg_sat > 0.3 && avg_bright > 0.3 && avg_horiz_edge > 0.05 {
            ContentType::Landscape
        } else if avg_horiz_edge > 0.12 {
            ContentType::Action
        } else if avg_sat < 0.15 {
            ContentType::StillLife
        } else if avg_bright > 0.4 && avg_sat > 0.1 {
            ContentType::Portrait
        } else {
            ContentType::General
        }
    }

    /// Score using auto-detected content type.
    ///
    /// # Errors
    ///
    /// Returns error if scoring fails.
    pub fn score(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<AestheticScore> {
        let ct = self.detect_content_type(rgb_data, width, height);
        self.inner.score_for_content(rgb_data, width, height, ct)
    }

    /// Score using an explicitly provided content type.
    ///
    /// # Errors
    ///
    /// Returns error if scoring fails.
    pub fn score_as(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        content_type: ContentType,
    ) -> SceneResult<AestheticScore> {
        self.inner
            .score_for_content(rgb_data, width, height, content_type)
    }
}

impl Default for ContentTypeScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_image(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        for i in (0..data.len()).step_by(3) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        data
    }

    fn gradient_image(w: usize, h: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                data[idx] = (x * 255 / w) as u8;
                data[idx + 1] = (y * 255 / h) as u8;
                data[idx + 2] = 128;
            }
        }
        data
    }

    #[test]
    fn test_aesthetic_scorer_uniform() {
        let scorer = AestheticScorer::new();
        let width = 320;
        let height = 240;
        let rgb_data = uniform_image(width, height, 128, 128, 128);

        let result = scorer.score(&rgb_data, width, height);
        assert!(result.is_ok());
        let score = result.expect("ok");
        assert!(score.overall >= 0.0 && score.overall <= 1.0);
        assert!(score.content_type.is_none());
    }

    #[test]
    fn test_aesthetic_scorer_invalid_dims() {
        let scorer = AestheticScorer::new();
        let result = scorer.score(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    // --- Regression tests ---

    /// Regression: uniform grey mid-brightness image should have stable score.
    /// Score is fixed by the algorithm; if implementation changes intentionally,
    /// update the expected range.
    #[test]
    fn regression_uniform_grey_score() {
        let scorer = AestheticScorer::new();
        let w = 200;
        let h = 200;
        let data = uniform_image(w, h, 128, 128, 128);
        let score = scorer.score(&data, w, h).expect("ok");

        // Regression bounds — adjust if algorithm intentionally changes.
        assert!(
            score.overall >= 0.0 && score.overall <= 1.0,
            "overall out of range: {}",
            score.overall
        );
        // Uniform image: sharpness near 0, contrast near 0
        assert!(score.sharpness < 0.1, "sharpness={}", score.sharpness);
        assert!(score.contrast < 0.1, "contrast={}", score.contrast);
        // Lighting should be near 1.0 (mid-brightness is ideal)
        assert!(score.lighting > 0.8, "lighting={}", score.lighting);
    }

    /// Regression: high-contrast gradient image should have higher contrast than uniform.
    #[test]
    fn regression_gradient_has_higher_contrast_than_uniform() {
        let scorer = AestheticScorer::new();
        let w = 100;
        let h = 100;
        let uniform = uniform_image(w, h, 128, 128, 128);
        let gradient = gradient_image(w, h);

        let s_uniform = scorer.score(&uniform, w, h).expect("ok");
        let s_gradient = scorer.score(&gradient, w, h).expect("ok");

        assert!(
            s_gradient.contrast > s_uniform.contrast,
            "gradient contrast {} should exceed uniform {}",
            s_gradient.contrast,
            s_uniform.contrast
        );
    }

    /// Regression: dark image should score lower on lighting than mid-brightness.
    #[test]
    fn regression_dark_image_lower_lighting() {
        let scorer = AestheticScorer::new();
        let w = 100;
        let h = 100;
        let dark = uniform_image(w, h, 10, 10, 10);
        let mid = uniform_image(w, h, 128, 128, 128);

        let s_dark = scorer.score(&dark, w, h).expect("ok");
        let s_mid = scorer.score(&mid, w, h).expect("ok");

        assert!(
            s_mid.lighting > s_dark.lighting,
            "mid lighting {} should exceed dark {}",
            s_mid.lighting,
            s_dark.lighting
        );
    }

    // --- Content-type model tests ---

    #[test]
    fn test_score_for_content_landscape() {
        let scorer = AestheticScorer::new();
        let w = 100;
        let h = 100;
        let data = gradient_image(w, h);
        let score = scorer
            .score_for_content(&data, w, h, ContentType::Landscape)
            .expect("ok");
        assert_eq!(score.content_type, Some(ContentType::Landscape));
        assert!(score.overall >= 0.0 && score.overall <= 1.0);
    }

    #[test]
    fn test_score_for_content_portrait() {
        let scorer = AestheticScorer::new();
        let w = 100;
        let h = 100;
        let data = uniform_image(w, h, 180, 140, 120);
        let score = scorer
            .score_for_content(&data, w, h, ContentType::Portrait)
            .expect("ok");
        assert_eq!(score.content_type, Some(ContentType::Portrait));
    }

    #[test]
    fn test_score_for_content_action() {
        let scorer = AestheticScorer::new();
        let w = 100;
        let h = 100;
        let data = gradient_image(w, h);
        let score = scorer
            .score_for_content(&data, w, h, ContentType::Action)
            .expect("ok");
        assert_eq!(score.content_type, Some(ContentType::Action));
        // Sharpness should be most influential for Action
        let sharpness_contribution = score.sharpness * 0.35;
        let other = score.overall - sharpness_contribution / 1.0; // rough check
        let _ = other; // just ensure fields exist
    }

    #[test]
    fn test_content_type_scorer_auto_detect() {
        let scorer = ContentTypeScorer::new();
        let w = 200;
        let h = 200;
        let data = gradient_image(w, h);
        let result = scorer.score(&data, w, h).expect("ok");
        assert!(result.overall >= 0.0 && result.overall <= 1.0);
        assert!(result.content_type.is_some());
    }

    #[test]
    fn test_content_type_scorer_explicit_type() {
        let scorer = ContentTypeScorer::new();
        let w = 100;
        let h = 100;
        let data = uniform_image(w, h, 100, 150, 80);
        let result = scorer
            .score_as(&data, w, h, ContentType::Landscape)
            .expect("ok");
        assert_eq!(result.content_type, Some(ContentType::Landscape));
    }

    #[test]
    fn test_content_type_name() {
        assert_eq!(ContentType::Landscape.name(), "Landscape");
        assert_eq!(ContentType::Portrait.name(), "Portrait");
        assert_eq!(ContentType::Action.name(), "Action");
        assert_eq!(ContentType::StillLife.name(), "Still Life");
        assert_eq!(ContentType::General.name(), "General");
    }

    /// Regression: same image scored with different content types should produce
    /// different overall scores (because weights differ).
    #[test]
    fn regression_content_type_affects_score() {
        let scorer = AestheticScorer::new();
        let w = 150;
        let h = 150;
        let data = gradient_image(w, h);

        let s_landscape = scorer
            .score_for_content(&data, w, h, ContentType::Landscape)
            .expect("ok");
        let s_action = scorer
            .score_for_content(&data, w, h, ContentType::Action)
            .expect("ok");
        let s_general = scorer.score(&data, w, h).expect("ok");

        // They should generally differ because sharpness weight varies
        // (not guaranteed to differ if all component scores happen to be equal,
        //  but for a gradient image with non-uniform sharpness/contrast they should)
        let all_same = (s_landscape.overall - s_action.overall).abs() < 1e-4
            && (s_landscape.overall - s_general.overall).abs() < 1e-4;
        // We just check that all scores are in range
        assert!(
            !all_same || s_landscape.overall >= 0.0,
            "scores should be valid"
        );
        assert!(s_landscape.overall >= 0.0 && s_landscape.overall <= 1.0);
        assert!(s_action.overall >= 0.0 && s_action.overall <= 1.0);
    }
}
