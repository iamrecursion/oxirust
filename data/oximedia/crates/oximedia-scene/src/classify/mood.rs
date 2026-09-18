//! Visual mood analysis for scenes.
//!
//! Analyzes the visual mood of video frames using photometric and color features.
//! Mood dimensions include brightness, warmth, contrast, and saturation.

use crate::common::Confidence;
use crate::error::SceneResult;
use serde::{Deserialize, Serialize};

/// Overall mood category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MoodCategory {
    /// Bright, cheerful, high-key.
    Bright,
    /// Dark, moody, low-key.
    Dark,
    /// Warm, golden, amber tones.
    Warm,
    /// Cool, blue, clinical tones.
    Cool,
    /// High contrast, dramatic.
    HighContrast,
    /// Low contrast, flat, foggy.
    LowContrast,
    /// Vibrant, saturated.
    Vibrant,
    /// Muted, desaturated, melancholic.
    Muted,
    /// Neutral, balanced.
    Neutral,
}

impl MoodCategory {
    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Bright => "Bright",
            Self::Dark => "Dark",
            Self::Warm => "Warm",
            Self::Cool => "Cool",
            Self::HighContrast => "High Contrast",
            Self::LowContrast => "Low Contrast",
            Self::Vibrant => "Vibrant",
            Self::Muted => "Muted",
            Self::Neutral => "Neutral",
        }
    }

    /// Get all mood categories.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Bright,
            Self::Dark,
            Self::Warm,
            Self::Cool,
            Self::HighContrast,
            Self::LowContrast,
            Self::Vibrant,
            Self::Muted,
            Self::Neutral,
        ]
    }
}

/// Photometric features extracted for mood analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodFeatures {
    /// Mean brightness (0.0 = black, 1.0 = white).
    pub brightness: f32,
    /// Brightness standard deviation (contrast indicator).
    pub brightness_std: f32,
    /// Mean saturation (0.0 = grey, 1.0 = fully saturated).
    pub saturation: f32,
    /// Color temperature bias (0.0 = cool, 1.0 = warm).
    pub warmth: f32,
    /// Shadow ratio: fraction of pixels below 20% brightness.
    pub shadow_ratio: f32,
    /// Highlight ratio: fraction of pixels above 80% brightness.
    pub highlight_ratio: f32,
    /// Hue variance (0.0 = monochromatic, 1.0 = rainbow).
    pub hue_variance: f32,
    /// Dominant hue angle in degrees (0–360).
    pub dominant_hue: f32,
}

impl Default for MoodFeatures {
    fn default() -> Self {
        Self {
            brightness: 0.5,
            brightness_std: 0.2,
            saturation: 0.5,
            warmth: 0.5,
            shadow_ratio: 0.1,
            highlight_ratio: 0.1,
            hue_variance: 0.5,
            dominant_hue: 0.0,
        }
    }
}

/// Result of mood analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodAnalysis {
    /// Primary mood category.
    pub primary_mood: MoodCategory,
    /// Confidence for primary mood.
    pub confidence: Confidence,
    /// Secondary mood (if relevant).
    pub secondary_mood: Option<MoodCategory>,
    /// Raw photometric features.
    pub features: MoodFeatures,
    /// Scores for each mood category.
    pub scores: Vec<(MoodCategory, f32)>,
}

/// Analyzes visual mood of video frames.
pub struct MoodAnalyzer {
    brightness_dark_threshold: f32,
    brightness_bright_threshold: f32,
    saturation_vibrant_threshold: f32,
    saturation_muted_threshold: f32,
    contrast_high_threshold: f32,
    contrast_low_threshold: f32,
}

impl MoodAnalyzer {
    /// Create a new mood analyzer with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            brightness_dark_threshold: 0.3,
            brightness_bright_threshold: 0.65,
            saturation_vibrant_threshold: 0.55,
            saturation_muted_threshold: 0.25,
            contrast_high_threshold: 0.3,
            contrast_low_threshold: 0.12,
        }
    }

    /// Analyze mood of a single RGB frame.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Raw RGB pixel data (3 bytes per pixel)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    ///
    /// # Errors
    ///
    /// Returns error if frame dimensions are inconsistent.
    pub fn analyze(&self, rgb: &[u8], width: usize, height: usize) -> SceneResult<MoodAnalysis> {
        crate::classify::validate_frame(rgb, width, height)?;

        let features = self.extract_features(rgb, width, height);
        let scores = self.compute_scores(&features);

        // Pick primary and secondary
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (primary_mood, primary_score) = sorted[0];
        let secondary_mood = if sorted.len() > 1 && sorted[1].1 > 0.4 {
            Some(sorted[1].0)
        } else {
            None
        };

        Ok(MoodAnalysis {
            primary_mood,
            confidence: Confidence::new(primary_score),
            secondary_mood,
            features,
            scores,
        })
    }

    fn extract_features(&self, rgb: &[u8], width: usize, height: usize) -> MoodFeatures {
        let pixel_count = width * height;
        let mut brightness_sum = 0.0f64;
        let mut brightness_sq_sum = 0.0f64;
        let mut saturation_sum = 0.0f64;
        let mut red_sum = 0.0f64;
        let mut blue_sum = 0.0f64;
        let mut hue_sin_sum = 0.0f64;
        let mut hue_cos_sum = 0.0f64;
        let mut shadow_count = 0u32;
        let mut highlight_count = 0u32;

        for chunk in rgb.chunks_exact(3) {
            let r = chunk[0] as f64 / 255.0;
            let g = chunk[1] as f64 / 255.0;
            let b = chunk[2] as f64 / 255.0;

            // Luma (perceptual brightness)
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            brightness_sum += luma;
            brightness_sq_sum += luma * luma;

            if luma < 0.2 {
                shadow_count += 1;
            }
            if luma > 0.8 {
                highlight_count += 1;
            }

            // HSV saturation
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            let delta = cmax - cmin;
            let sat = if cmax > 0.0 { delta / cmax } else { 0.0 };
            saturation_sum += sat;

            // Warmth: red/yellow bias vs blue
            red_sum += r;
            blue_sum += b;

            // Hue angle for variance calculation
            if delta > 0.01 {
                let hue = if (cmax - r).abs() < 1e-6 {
                    60.0 * (((g - b) / delta) % 6.0)
                } else if (cmax - g).abs() < 1e-6 {
                    60.0 * ((b - r) / delta + 2.0)
                } else {
                    60.0 * ((r - g) / delta + 4.0)
                };
                let hue_rad = hue.to_radians();
                hue_sin_sum += hue_rad.sin();
                hue_cos_sum += hue_rad.cos();
            }
        }

        let n = pixel_count as f64;
        let mean_brightness = (brightness_sum / n) as f32;
        let mean_sq = brightness_sq_sum / n;
        let brightness_std = ((mean_sq - (brightness_sum / n).powi(2)).max(0.0).sqrt()) as f32;

        let saturation = (saturation_sum / n) as f32;
        let warmth = ((red_sum - blue_sum) / n / 2.0 + 0.5).clamp(0.0, 1.0) as f32;

        let shadow_ratio = shadow_count as f32 / pixel_count as f32;
        let highlight_ratio = highlight_count as f32 / pixel_count as f32;

        // Hue variance via circular statistics
        let hue_r = ((hue_sin_sum / n).powi(2) + (hue_cos_sum / n).powi(2)).sqrt();
        let hue_variance = (1.0 - hue_r) as f32;

        let dominant_hue_rad = (hue_sin_sum / n).atan2(hue_cos_sum / n);
        let dominant_hue = (dominant_hue_rad.to_degrees() as f32 + 360.0) % 360.0;

        MoodFeatures {
            brightness: mean_brightness,
            brightness_std,
            saturation,
            warmth,
            shadow_ratio,
            highlight_ratio,
            hue_variance,
            dominant_hue,
        }
    }

    fn compute_scores(&self, f: &MoodFeatures) -> Vec<(MoodCategory, f32)> {
        // Compute raw scores
        let bright = self.score_bright(f);
        let dark = self.score_dark(f);
        let warm = self.score_warm(f);
        let cool = self.score_cool(f);
        let high_contrast = self.score_high_contrast(f);
        let low_contrast = self.score_low_contrast(f);
        let vibrant = self.score_vibrant(f);
        let muted = self.score_muted(f);
        let neutral = self.score_neutral(f);

        // Suppress contrast scores when primary photometric signals are strong.
        // Contrast describes *how* something looks; brightness/warmth/saturation
        // describe *what* the dominant visual impression is.
        let dominant_photometric = bright.max(dark).max(warm).max(cool).max(vibrant).max(muted);
        let contrast_suppression = if dominant_photometric > 0.45 {
            1.0 - (dominant_photometric - 0.45) / 0.55
        } else {
            1.0
        };
        let effective_low_contrast = low_contrast * contrast_suppression;
        let effective_high_contrast = high_contrast * contrast_suppression;

        vec![
            (MoodCategory::Bright, bright),
            (MoodCategory::Dark, dark),
            (MoodCategory::Warm, warm),
            (MoodCategory::Cool, cool),
            (MoodCategory::HighContrast, effective_high_contrast),
            (MoodCategory::LowContrast, effective_low_contrast),
            (MoodCategory::Vibrant, vibrant),
            (MoodCategory::Muted, muted),
            (MoodCategory::Neutral, neutral),
        ]
    }

    fn score_bright(&self, f: &MoodFeatures) -> f32 {
        if f.brightness >= self.brightness_bright_threshold {
            let excess = (f.brightness - self.brightness_bright_threshold)
                / (1.0 - self.brightness_bright_threshold);
            (0.5 + excess * 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_dark(&self, f: &MoodFeatures) -> f32 {
        if f.brightness <= self.brightness_dark_threshold {
            let deficit = 1.0 - f.brightness / self.brightness_dark_threshold;
            (0.5 + deficit * 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_warm(&self, f: &MoodFeatures) -> f32 {
        if f.warmth > 0.55 {
            ((f.warmth - 0.55) / 0.45).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_cool(&self, f: &MoodFeatures) -> f32 {
        if f.warmth < 0.45 {
            ((0.45 - f.warmth) / 0.45).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_high_contrast(&self, f: &MoodFeatures) -> f32 {
        if f.brightness_std >= self.contrast_high_threshold {
            ((f.brightness_std - self.contrast_high_threshold) / 0.4 + 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_low_contrast(&self, f: &MoodFeatures) -> f32 {
        if f.brightness_std <= self.contrast_low_threshold {
            (1.0 - f.brightness_std / self.contrast_low_threshold).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_vibrant(&self, f: &MoodFeatures) -> f32 {
        if f.saturation >= self.saturation_vibrant_threshold {
            ((f.saturation - self.saturation_vibrant_threshold) / 0.45 + 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_muted(&self, f: &MoodFeatures) -> f32 {
        if f.saturation <= self.saturation_muted_threshold {
            (1.0 - f.saturation / self.saturation_muted_threshold).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn score_neutral(&self, f: &MoodFeatures) -> f32 {
        let brightness_mid = 1.0 - (f.brightness - 0.5).abs() * 2.0;
        let saturation_mid = 1.0 - (f.saturation - 0.35).abs() * 2.0;
        let warmth_mid = 1.0 - (f.warmth - 0.5).abs() * 2.0;
        ((brightness_mid + saturation_mid + warmth_mid) / 3.0).clamp(0.0, 1.0)
    }
}

impl Default for MoodAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn test_mood_category_names() {
        assert_eq!(MoodCategory::Bright.name(), "Bright");
        assert_eq!(MoodCategory::Dark.name(), "Dark");
        assert_eq!(MoodCategory::Warm.name(), "Warm");
        assert_eq!(MoodCategory::Cool.name(), "Cool");
        assert_eq!(MoodCategory::HighContrast.name(), "High Contrast");
        assert_eq!(MoodCategory::LowContrast.name(), "Low Contrast");
        assert_eq!(MoodCategory::Vibrant.name(), "Vibrant");
        assert_eq!(MoodCategory::Muted.name(), "Muted");
        assert_eq!(MoodCategory::Neutral.name(), "Neutral");
    }

    #[test]
    fn test_all_categories() {
        assert_eq!(MoodCategory::all().len(), 9);
    }

    #[test]
    fn test_bright_frame() {
        // Pure white has brightness=1.0 and saturation=0.0 → Bright dominates
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(240, 240, 240, 64, 64);
        let result = analyzer
            .analyze(&frame, 64, 64)
            .expect("should succeed in test");
        // Bright frame: brightness ~0.94, saturation=0 → Bright scores highest
        assert!(
            result.primary_mood == MoodCategory::Bright
                || result.secondary_mood == Some(MoodCategory::Bright),
            "Expected Bright in top-2 moods, got {:?} / {:?}",
            result.primary_mood,
            result.secondary_mood
        );
    }

    #[test]
    fn test_dark_frame() {
        // Near-black frame: brightness ~0.06 → Dark
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(15, 15, 15, 64, 64);
        let result = analyzer
            .analyze(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(
            result.primary_mood == MoodCategory::Dark
                || result.secondary_mood == Some(MoodCategory::Dark),
            "Expected Dark, got {:?} / {:?}",
            result.primary_mood,
            result.secondary_mood
        );
    }

    #[test]
    fn test_warm_frame() {
        // Orange frame: warmth ~0.85, saturation ~0.82 → either Warm or Vibrant primary
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(220, 140, 40, 64, 64);
        let result = analyzer
            .analyze(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(
            result.features.warmth > 0.6,
            "Expected high warmth, got {}",
            result.features.warmth
        );
        // Warm must appear in scores significantly
        let warm_score = result
            .scores
            .iter()
            .find(|(m, _)| *m == MoodCategory::Warm)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            warm_score > 0.2,
            "Expected Warm score > 0.2, got {warm_score}"
        );
    }

    #[test]
    fn test_cool_frame() {
        // Blue frame: warmth ~0.17 → Cool
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(30, 80, 200, 64, 64);
        let result = analyzer
            .analyze(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(
            result.features.warmth < 0.4,
            "Expected low warmth for cool frame, got {}",
            result.features.warmth
        );
        let cool_score = result
            .scores
            .iter()
            .find(|(m, _)| *m == MoodCategory::Cool)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            cool_score > 0.2,
            "Expected Cool score > 0.2, got {cool_score}"
        );
    }

    #[test]
    fn test_muted_frame() {
        // Near-grey: saturation ~0.01 → Muted
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(120, 118, 119, 64, 64);
        let result = analyzer
            .analyze(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(
            result.features.saturation < 0.05,
            "Expected low saturation, got {}",
            result.features.saturation
        );
        let muted_score = result
            .scores
            .iter()
            .find(|(m, _)| *m == MoodCategory::Muted)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            muted_score > 0.3,
            "Expected Muted score > 0.3, got {muted_score}"
        );
    }

    #[test]
    fn test_features_brightness_range() {
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(128, 128, 128, 32, 32);
        let result = analyzer
            .analyze(&frame, 32, 32)
            .expect("should succeed in test");
        assert!(result.features.brightness > 0.0);
        assert!(result.features.brightness <= 1.0);
    }

    #[test]
    fn test_confidence_range() {
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(200, 100, 50, 32, 32);
        let result = analyzer
            .analyze(&frame, 32, 32)
            .expect("should succeed in test");
        assert!(result.confidence.value() >= 0.0);
        assert!(result.confidence.value() <= 1.0);
    }

    #[test]
    fn test_invalid_frame_size() {
        let analyzer = MoodAnalyzer::new();
        let frame = vec![0u8; 10]; // Too small
        let result = analyzer.analyze(&frame, 64, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_scores_all_categories_present() {
        let analyzer = MoodAnalyzer::new();
        let frame = solid_frame(150, 150, 150, 32, 32);
        let result = analyzer
            .analyze(&frame, 32, 32)
            .expect("should succeed in test");
        assert_eq!(result.scores.len(), 9);
    }

    #[test]
    fn test_default_analyzer() {
        let _analyzer = MoodAnalyzer::default();
    }
}
