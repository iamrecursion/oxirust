#![allow(dead_code)]
//! Visual theme analysis and consistency enforcement for automated edits.
//!
//! Analyses video frames to extract dominant colours, brightness profiles,
//! and stylistic fingerprints. The module can then score how well a set of
//! clips fits a target visual theme, enabling the auto-editor to prefer
//! visually cohesive content.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Represents an RGB colour with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgb {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB colour.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Euclidean distance to another colour.
    #[allow(clippy::cast_precision_loss)]
    pub fn distance(&self, other: &Self) -> f64 {
        let dr = self.r as f64 - other.r as f64;
        let dg = self.g as f64 - other.g as f64;
        let db = self.b as f64 - other.b as f64;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Perceived luminance (BT.709).
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        0.2126 * (self.r as f64 / 255.0)
            + 0.7152 * (self.g as f64 / 255.0)
            + 0.0722 * (self.b as f64 / 255.0)
    }

    /// Convert to quantised bucket (reduces colour space for histogram).
    pub fn quantise(&self, levels: u8) -> Self {
        let step = if levels == 0 {
            1
        } else {
            256u16 / levels as u16
        };
        let step = step.max(1) as u8;
        Self {
            r: (self.r / step) * step,
            g: (self.g / step) * step,
            b: (self.b / step) * step,
        }
    }
}

/// A colour palette extracted from a frame or clip.
#[derive(Debug, Clone)]
pub struct ColourPalette {
    /// Dominant colours sorted by frequency (most frequent first).
    pub colours: Vec<(Rgb, f64)>,
    /// Average luminance of the frame/clip.
    pub mean_luminance: f64,
}

impl ColourPalette {
    /// Create an empty palette.
    pub fn empty() -> Self {
        Self {
            colours: Vec::new(),
            mean_luminance: 0.0,
        }
    }

    /// Number of dominant colours.
    pub fn len(&self) -> usize {
        self.colours.len()
    }

    /// Whether the palette is empty.
    pub fn is_empty(&self) -> bool {
        self.colours.is_empty()
    }
}

/// Built-in visual theme presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemePreset {
    /// Warm / golden hour look.
    Warm,
    /// Cool / blue-toned look.
    Cool,
    /// High contrast / dramatic.
    Dramatic,
    /// Desaturated / muted.
    Muted,
    /// Vibrant / saturated.
    Vibrant,
}

/// Configuration for the visual-theme analyser.
#[derive(Debug, Clone)]
pub struct VisualThemeConfig {
    /// Number of quantisation levels per channel.
    pub quant_levels: u8,
    /// Maximum number of dominant colours to extract.
    pub max_colours: usize,
    /// Minimum proportion a colour must have to be considered dominant.
    pub min_proportion: f64,
    /// Theme preset (optional).
    pub preset: Option<ThemePreset>,
}

impl Default for VisualThemeConfig {
    fn default() -> Self {
        Self {
            quant_levels: 8,
            max_colours: 8,
            min_proportion: 0.01,
            preset: None,
        }
    }
}

impl VisualThemeConfig {
    /// New config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of quantisation levels.
    pub fn with_quant_levels(mut self, levels: u8) -> Self {
        self.quant_levels = levels;
        self
    }

    /// Set the maximum colours.
    pub fn with_max_colours(mut self, n: usize) -> Self {
        self.max_colours = n;
        self
    }

    /// Set a theme preset.
    pub fn with_preset(mut self, preset: ThemePreset) -> Self {
        self.preset = Some(preset);
        self
    }
}

/// Score describing how well a clip matches a visual theme.
#[derive(Debug, Clone)]
pub struct ThemeScore {
    /// Overall similarity (0..1, higher = better match).
    pub overall: f64,
    /// Luminance similarity.
    pub luminance_score: f64,
    /// Colour-palette similarity.
    pub palette_score: f64,
}

// ---------------------------------------------------------------------------
// Analyser
// ---------------------------------------------------------------------------

/// Visual-theme analyser.
#[derive(Debug, Clone)]
pub struct VisualThemeAnalyser {
    /// Configuration.
    config: VisualThemeConfig,
}

impl VisualThemeAnalyser {
    /// Create a new analyser.
    pub fn new(config: VisualThemeConfig) -> Self {
        Self { config }
    }

    /// Extract a colour palette from raw RGB pixel data.
    ///
    /// `pixels` must contain triplets `[R, G, B, R, G, B, ...]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_palette(&self, pixels: &[u8]) -> ColourPalette {
        if pixels.len() < 3 {
            return ColourPalette::empty();
        }
        let n_pixels = pixels.len() / 3;
        let mut hist: HashMap<Rgb, usize> = HashMap::new();
        let mut lum_sum = 0.0_f64;

        for chunk in pixels.chunks_exact(3) {
            let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
            let q = rgb.quantise(self.config.quant_levels);
            *hist.entry(q).or_insert(0) += 1;
            lum_sum += rgb.luminance();
        }

        let mean_luminance = lum_sum / n_pixels as f64;

        let mut entries: Vec<(Rgb, f64)> = hist
            .into_iter()
            .map(|(c, count)| (c, count as f64 / n_pixels as f64))
            .filter(|&(_, p)| p >= self.config.min_proportion)
            .collect();

        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(self.config.max_colours);

        ColourPalette {
            colours: entries,
            mean_luminance,
        }
    }

    /// Compute how similar two palettes are (0..1).
    #[allow(clippy::cast_precision_loss)]
    pub fn palette_similarity(&self, a: &ColourPalette, b: &ColourPalette) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        // Simple colour-histogram intersection
        let max_colours = a.len().max(b.len());
        let mut match_sum = 0.0_f64;
        for &(ca, pa) in &a.colours {
            for &(cb, pb) in &b.colours {
                let dist = ca.distance(&cb);
                if dist < 50.0 {
                    match_sum += pa.min(pb);
                }
            }
        }
        (match_sum / max_colours as f64).min(1.0)
    }

    /// Score a palette against the configured theme preset.
    pub fn score_against_theme(&self, palette: &ColourPalette) -> ThemeScore {
        let reference = self.reference_palette_for_preset();
        let palette_score = self.palette_similarity(palette, &reference);
        let lum_score = 1.0 - (palette.mean_luminance - reference.mean_luminance).abs();
        let lum_score = lum_score.clamp(0.0, 1.0);
        let overall = 0.6 * palette_score + 0.4 * lum_score;
        ThemeScore {
            overall,
            luminance_score: lum_score,
            palette_score,
        }
    }

    // -- helpers --

    fn reference_palette_for_preset(&self) -> ColourPalette {
        match self.config.preset {
            Some(ThemePreset::Warm) => ColourPalette {
                colours: vec![
                    (Rgb::new(192, 128, 64), 0.30),
                    (Rgb::new(224, 176, 96), 0.25),
                    (Rgb::new(160, 96, 48), 0.20),
                ],
                mean_luminance: 0.55,
            },
            Some(ThemePreset::Cool) => ColourPalette {
                colours: vec![
                    (Rgb::new(64, 128, 192), 0.30),
                    (Rgb::new(96, 160, 208), 0.25),
                    (Rgb::new(48, 96, 160), 0.20),
                ],
                mean_luminance: 0.45,
            },
            Some(ThemePreset::Dramatic) => ColourPalette {
                colours: vec![
                    (Rgb::new(32, 32, 32), 0.40),
                    (Rgb::new(224, 224, 224), 0.20),
                ],
                mean_luminance: 0.30,
            },
            Some(ThemePreset::Muted) => ColourPalette {
                colours: vec![
                    (Rgb::new(160, 160, 160), 0.30),
                    (Rgb::new(128, 128, 128), 0.25),
                ],
                mean_luminance: 0.50,
            },
            Some(ThemePreset::Vibrant) => ColourPalette {
                colours: vec![
                    (Rgb::new(224, 64, 64), 0.20),
                    (Rgb::new(64, 224, 64), 0.20),
                    (Rgb::new(64, 64, 224), 0.20),
                ],
                mean_luminance: 0.50,
            },
            None => ColourPalette::empty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_pixels(r: u8, g: u8, b: u8, count: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(count * 3);
        for _ in 0..count {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    #[test]
    fn test_rgb_distance_same() {
        let c = Rgb::new(100, 100, 100);
        assert!((c.distance(&c) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rgb_distance_different() {
        let a = Rgb::new(0, 0, 0);
        let b = Rgb::new(255, 255, 255);
        assert!(a.distance(&b) > 400.0);
    }

    #[test]
    fn test_rgb_luminance_black() {
        let c = Rgb::new(0, 0, 0);
        assert!((c.luminance() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rgb_luminance_white() {
        let c = Rgb::new(255, 255, 255);
        assert!((c.luminance() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quantise() {
        let c = Rgb::new(200, 100, 50);
        let q = c.quantise(4);
        assert!(q.r <= 200);
        assert!(q.g <= 100);
        assert!(q.b <= 50);
    }

    #[test]
    fn test_extract_palette_solid() {
        let analyser = VisualThemeAnalyser::new(VisualThemeConfig::default());
        let px = solid_pixels(128, 64, 32, 1000);
        let palette = analyser.extract_palette(&px);
        assert!(!palette.is_empty());
        assert!(palette.mean_luminance > 0.0);
    }

    #[test]
    fn test_extract_palette_empty() {
        let analyser = VisualThemeAnalyser::new(VisualThemeConfig::default());
        let palette = analyser.extract_palette(&[]);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_palette_similarity_identical() {
        let analyser = VisualThemeAnalyser::new(VisualThemeConfig::default());
        let px = solid_pixels(128, 64, 32, 500);
        let p = analyser.extract_palette(&px);
        let sim = analyser.palette_similarity(&p, &p);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_palette_similarity_empty() {
        let analyser = VisualThemeAnalyser::new(VisualThemeConfig::default());
        let empty = ColourPalette::empty();
        let sim = analyser.palette_similarity(&empty, &empty);
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_against_warm_theme() {
        let cfg = VisualThemeConfig::default().with_preset(ThemePreset::Warm);
        let analyser = VisualThemeAnalyser::new(cfg);
        let px = solid_pixels(192, 128, 64, 1000);
        let palette = analyser.extract_palette(&px);
        let score = analyser.score_against_theme(&palette);
        assert!(score.overall >= 0.0 && score.overall <= 1.0);
        assert!(score.luminance_score >= 0.0);
    }

    #[test]
    fn test_theme_presets_all() {
        for preset in &[
            ThemePreset::Warm,
            ThemePreset::Cool,
            ThemePreset::Dramatic,
            ThemePreset::Muted,
            ThemePreset::Vibrant,
        ] {
            let cfg = VisualThemeConfig::default().with_preset(*preset);
            let analyser = VisualThemeAnalyser::new(cfg);
            let px = solid_pixels(128, 128, 128, 100);
            let palette = analyser.extract_palette(&px);
            let score = analyser.score_against_theme(&palette);
            assert!(score.overall >= 0.0);
            assert!(score.overall <= 1.0);
        }
    }

    #[test]
    fn test_colour_palette_len() {
        let p = ColourPalette {
            colours: vec![(Rgb::new(0, 0, 0), 0.5), (Rgb::new(255, 255, 255), 0.5)],
            mean_luminance: 0.5,
        };
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let cfg = VisualThemeConfig::new()
            .with_quant_levels(16)
            .with_max_colours(12);
        assert_eq!(cfg.quant_levels, 16);
        assert_eq!(cfg.max_colours, 12);
    }
}
