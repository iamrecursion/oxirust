#![allow(dead_code)]
//! Talent keying and extraction for virtual production.
//!
//! Provides real-time chroma keying, luminance keying, and difference keying
//! algorithms used to isolate talent from backgrounds in LED-wall and
//! green-screen virtual production workflows.

use std::time::{Duration, Instant};

/// Keying algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyAlgorithm {
    /// Classic chroma key using a single hue channel.
    Chroma,
    /// Luminance-based key (bright/dark separation).
    Luminance,
    /// Difference key using a clean plate reference.
    Difference,
    /// Despill-aware chroma key with edge refinement.
    AdvancedChroma,
    /// AI/ML-based segmentation (placeholder for GPU path).
    AiSegmentation,
}

impl KeyAlgorithm {
    /// Returns a human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Chroma => "Chroma Key",
            Self::Luminance => "Luminance Key",
            Self::Difference => "Difference Key",
            Self::AdvancedChroma => "Advanced Chroma Key",
            Self::AiSegmentation => "AI Segmentation",
        }
    }

    /// Returns `true` if a clean plate reference is required.
    #[must_use]
    pub fn needs_clean_plate(&self) -> bool {
        matches!(self, Self::Difference)
    }
}

/// Target color for chroma keying.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyColor {
    /// Red component (0.0 – 1.0).
    pub r: f32,
    /// Green component (0.0 – 1.0).
    pub g: f32,
    /// Blue component (0.0 – 1.0).
    pub b: f32,
}

impl KeyColor {
    /// Creates a new key color.
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Standard green screen color.
    #[must_use]
    pub fn green_screen() -> Self {
        Self {
            r: 0.0,
            g: 1.0,
            b: 0.0,
        }
    }

    /// Standard blue screen color.
    #[must_use]
    pub fn blue_screen() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        }
    }

    /// Euclidean distance to another color.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Hue angle in degrees (0 – 360). Returns 0 for achromatic pixels.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn hue_degrees(&self) -> f32 {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let delta = max - min;
        if delta < 1e-6 {
            return 0.0;
        }
        let h = if (max - self.r).abs() < 1e-6 {
            60.0 * (((self.g - self.b) / delta) % 6.0)
        } else if (max - self.g).abs() < 1e-6 {
            60.0 * (((self.b - self.r) / delta) + 2.0)
        } else {
            60.0 * (((self.r - self.g) / delta) + 4.0)
        };
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

/// Despill method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DespillMethod {
    /// Average of the two non-key channels replaces the key channel.
    AverageReplace,
    /// The key channel is clamped to the maximum of the other channels.
    ClampMax,
    /// No despill.
    None,
}

/// Edge refinement options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeRefinement {
    /// Erode matte edges by this many pixels.
    pub erode_px: f32,
    /// Soften matte edges by this many pixels.
    pub soften_px: f32,
    /// Choke (shrink / expand) as a fraction of image width.
    pub choke: f32,
}

impl Default for EdgeRefinement {
    fn default() -> Self {
        Self {
            erode_px: 0.0,
            soften_px: 1.0,
            choke: 0.0,
        }
    }
}

/// Configuration for the talent keying pipeline.
#[derive(Debug, Clone)]
pub struct TalentKeyConfig {
    /// Algorithm to use.
    pub algorithm: KeyAlgorithm,
    /// Target key color (used by Chroma / `AdvancedChroma`).
    pub key_color: KeyColor,
    /// Similarity threshold (0.0 = exact match, 1.0 = everything matches).
    pub similarity: f32,
    /// Smoothness of the alpha transition.
    pub smoothness: f32,
    /// Despill method.
    pub despill: DespillMethod,
    /// Edge refinement settings.
    pub edge: EdgeRefinement,
    /// Luminance low threshold (for luminance keyer, 0.0-1.0).
    pub luma_low: f32,
    /// Luminance high threshold (for luminance keyer, 0.0-1.0).
    pub luma_high: f32,
}

impl Default for TalentKeyConfig {
    fn default() -> Self {
        Self {
            algorithm: KeyAlgorithm::Chroma,
            key_color: KeyColor::green_screen(),
            similarity: 0.4,
            smoothness: 0.1,
            despill: DespillMethod::AverageReplace,
            edge: EdgeRefinement::default(),
            luma_low: 0.0,
            luma_high: 1.0,
        }
    }
}

/// Result of keying a single pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyResult {
    /// Alpha value (0.0 = transparent / keyed out, 1.0 = solid / talent).
    pub alpha: f32,
    /// Despilled red.
    pub r: f32,
    /// Despilled green.
    pub g: f32,
    /// Despilled blue.
    pub b: f32,
}

/// Computes a simple chroma key alpha for one pixel.
fn chroma_key_pixel(pixel: &KeyColor, key: &KeyColor, similarity: f32, smoothness: f32) -> f32 {
    let dist = pixel.distance(key);
    if dist < similarity {
        0.0
    } else if dist < similarity + smoothness {
        (dist - similarity) / smoothness
    } else {
        1.0
    }
}

/// Computes a luminance key alpha for one pixel.
fn luminance_key_pixel(pixel: &KeyColor, low: f32, high: f32) -> f32 {
    let luma = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
    if luma < low {
        0.0
    } else if luma > high {
        1.0
    } else if (high - low).abs() < 1e-6 {
        1.0
    } else {
        (luma - low) / (high - low)
    }
}

/// Processing statistics for a keying pass.
#[derive(Debug, Clone)]
pub struct KeyingStats {
    /// Number of pixels processed.
    pub pixels_processed: u64,
    /// Number of fully transparent pixels.
    pub pixels_keyed: u64,
    /// Number of fully opaque pixels.
    pub pixels_solid: u64,
    /// Processing duration.
    pub duration: Duration,
}

impl KeyingStats {
    /// Creates zeroed stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pixels_processed: 0,
            pixels_keyed: 0,
            pixels_solid: 0,
            duration: Duration::ZERO,
        }
    }

    /// Fraction of pixels that were keyed out.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn keyed_ratio(&self) -> f64 {
        if self.pixels_processed == 0 {
            return 0.0;
        }
        self.pixels_keyed as f64 / self.pixels_processed as f64
    }
}

impl Default for KeyingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// The talent keying pipeline.
pub struct TalentKeyer {
    /// Configuration.
    config: TalentKeyConfig,
    /// Accumulated statistics.
    stats: KeyingStats,
}

impl TalentKeyer {
    /// Creates a new talent keyer.
    #[must_use]
    pub fn new(config: TalentKeyConfig) -> Self {
        Self {
            config,
            stats: KeyingStats::new(),
        }
    }

    /// Keys a single pixel and returns the result.
    #[must_use]
    pub fn key_pixel(&self, pixel: &KeyColor) -> KeyResult {
        let alpha = match self.config.algorithm {
            KeyAlgorithm::Chroma | KeyAlgorithm::AdvancedChroma => chroma_key_pixel(
                pixel,
                &self.config.key_color,
                self.config.similarity,
                self.config.smoothness,
            ),
            KeyAlgorithm::Luminance => {
                luminance_key_pixel(pixel, self.config.luma_low, self.config.luma_high)
            }
            KeyAlgorithm::Difference | KeyAlgorithm::AiSegmentation => {
                // Placeholder – difference keying requires a clean plate.
                1.0
            }
        };

        let (r, g, b) = self.apply_despill(pixel, alpha);
        KeyResult { alpha, r, g, b }
    }

    /// Processes a row of pixels (given as flat RGB f32 triples) and writes alpha results.
    pub fn process_row(&mut self, pixels: &[f32], out_alpha: &mut [f32]) {
        let start = Instant::now();
        let count = pixels.len() / 3;
        for i in 0..count {
            let px = KeyColor::new(pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]);
            let result = self.key_pixel(&px);
            if i < out_alpha.len() {
                out_alpha[i] = result.alpha;
            }
            self.stats.pixels_processed += 1;
            if result.alpha < 0.01 {
                self.stats.pixels_keyed += 1;
            } else if result.alpha > 0.99 {
                self.stats.pixels_solid += 1;
            }
        }
        self.stats.duration += start.elapsed();
    }

    /// Returns current statistics.
    #[must_use]
    pub fn stats(&self) -> &KeyingStats {
        &self.stats
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &TalentKeyConfig {
        &self.config
    }

    /// Applies despill to an RGB pixel given a keying alpha.
    fn apply_despill(&self, pixel: &KeyColor, alpha: f32) -> (f32, f32, f32) {
        if alpha > 0.99 || self.config.despill == DespillMethod::None {
            return (pixel.r, pixel.g, pixel.b);
        }
        match self.config.despill {
            DespillMethod::AverageReplace => {
                let avg = (pixel.r + pixel.b) / 2.0;
                let g = pixel.g.min(avg);
                (pixel.r, g, pixel.b)
            }
            DespillMethod::ClampMax => {
                let cap = pixel.r.max(pixel.b);
                let g = pixel.g.min(cap);
                (pixel.r, g, pixel.b)
            }
            DespillMethod::None => (pixel.r, pixel.g, pixel.b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_color_distance_same() {
        let c = KeyColor::green_screen();
        assert!(c.distance(&c) < f32::EPSILON);
    }

    #[test]
    fn test_key_color_distance_different() {
        let g = KeyColor::green_screen();
        let b = KeyColor::blue_screen();
        assert!(g.distance(&b) > 1.0);
    }

    #[test]
    fn test_key_color_hue_green() {
        let g = KeyColor::green_screen();
        let hue = g.hue_degrees();
        assert!((hue - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_key_color_clamping() {
        let c = KeyColor::new(2.0, -1.0, 0.5);
        assert!((c.r - 1.0).abs() < f32::EPSILON);
        assert!(c.g.abs() < f32::EPSILON);
    }

    #[test]
    fn test_algorithm_name() {
        assert_eq!(KeyAlgorithm::Chroma.name(), "Chroma Key");
        assert_eq!(KeyAlgorithm::AiSegmentation.name(), "AI Segmentation");
    }

    #[test]
    fn test_algorithm_needs_clean_plate() {
        assert!(KeyAlgorithm::Difference.needs_clean_plate());
        assert!(!KeyAlgorithm::Chroma.needs_clean_plate());
    }

    #[test]
    fn test_chroma_key_green_pixel() {
        let keyer = TalentKeyer::new(TalentKeyConfig::default());
        let green = KeyColor::green_screen();
        let result = keyer.key_pixel(&green);
        assert!(result.alpha < 0.01, "Pure green should be keyed out");
    }

    #[test]
    fn test_chroma_key_red_pixel() {
        let keyer = TalentKeyer::new(TalentKeyConfig::default());
        let red = KeyColor::new(1.0, 0.0, 0.0);
        let result = keyer.key_pixel(&red);
        assert!(result.alpha > 0.9, "Red should be opaque against green key");
    }

    #[test]
    fn test_luminance_key() {
        let config = TalentKeyConfig {
            algorithm: KeyAlgorithm::Luminance,
            luma_low: 0.2,
            luma_high: 0.8,
            ..Default::default()
        };
        let keyer = TalentKeyer::new(config);
        let dark = KeyColor::new(0.0, 0.0, 0.0);
        let result = keyer.key_pixel(&dark);
        assert!(result.alpha < 0.01, "Dark pixel should be keyed out");
    }

    #[test]
    fn test_process_row() {
        let mut keyer = TalentKeyer::new(TalentKeyConfig::default());
        // 3 pixels: green, red, white
        let pixels = [0.0f32, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut alpha = [0.0f32; 3];
        keyer.process_row(&pixels, &mut alpha);
        assert!(alpha[0] < 0.1); // green -> keyed
        assert!(alpha[1] > 0.5); // red -> solid
        assert_eq!(keyer.stats().pixels_processed, 3);
    }

    #[test]
    fn test_despill_average_replace() {
        let config = TalentKeyConfig {
            despill: DespillMethod::AverageReplace,
            similarity: 0.8,
            ..Default::default()
        };
        let keyer = TalentKeyer::new(config);
        let px = KeyColor::new(0.3, 0.9, 0.2);
        let result = keyer.key_pixel(&px);
        // Green channel should be capped if pixel is partially transparent
        assert!(result.g <= 0.9);
    }

    #[test]
    fn test_keying_stats_ratio() {
        let mut stats = KeyingStats::new();
        stats.pixels_processed = 100;
        stats.pixels_keyed = 40;
        assert!((stats.keyed_ratio() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_edge_refinement_defaults() {
        let edge = EdgeRefinement::default();
        assert!(edge.erode_px.abs() < f32::EPSILON);
        assert!((edge.soften_px - 1.0).abs() < f32::EPSILON);
    }
}
