//! Color analysis: dominant color extraction, color harmony detection, and palette generation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// An RGB color with f32 components in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RgbColor {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

impl RgbColor {
    /// Create a new RGB color.
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create from 8-bit integer components.
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
        }
    }

    /// Convert to 8-bit integer components.
    #[must_use]
    pub fn to_u8(&self) -> (u8, u8, u8) {
        (
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
        )
    }

    /// Euclidean distance to another color in RGB space.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    /// Convert to HSL (hue [0, 360), saturation [0, 1], lightness [0, 1]).
    #[must_use]
    pub fn to_hsl(&self) -> (f32, f32, f32) {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let delta = max - min;
        let l = (max + min) / 2.0;

        if delta < 1e-6 {
            return (0.0, 0.0, l);
        }

        let s = if l < 0.5 {
            delta / (max + min)
        } else {
            delta / (2.0 - max - min)
        };

        let h = if (max - self.r).abs() < 1e-6 {
            ((self.g - self.b) / delta) % 6.0
        } else if (max - self.g).abs() < 1e-6 {
            (self.b - self.r) / delta + 2.0
        } else {
            (self.r - self.g) / delta + 4.0
        };

        let h = (h * 60.0).rem_euclid(360.0);
        (h, s, l)
    }

    /// Luminance (perceptual, BT.709).
    #[must_use]
    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

/// A dominant color entry with its proportion in the image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantColor {
    /// The color value.
    pub color: RgbColor,
    /// Proportion of pixels (0.0 – 1.0).
    pub proportion: f32,
}

/// Color harmony type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorHarmony {
    /// Monochromatic: all colors close in hue.
    Monochromatic,
    /// Complementary: colors opposite on the color wheel (~180° apart).
    Complementary,
    /// Analogous: adjacent colors (~30° apart).
    Analogous,
    /// Triadic: three colors ~120° apart.
    Triadic,
    /// Split-complementary.
    SplitComplementary,
    /// No clear harmony detected.
    None,
}

/// Analyze the color harmony of a set of hues (in degrees 0–360).
#[must_use]
pub fn detect_harmony(hues: &[f32]) -> ColorHarmony {
    if hues.len() < 2 {
        return ColorHarmony::Monochromatic;
    }

    // Check complementary (largest pair ~180°).
    for i in 0..hues.len() {
        for j in (i + 1)..hues.len() {
            let diff = (hues[i] - hues[j]).abs().rem_euclid(360.0);
            let diff = if diff > 180.0 { 360.0 - diff } else { diff };
            if (diff - 180.0).abs() < 20.0 {
                return ColorHarmony::Complementary;
            }
        }
    }

    // Check analogous (all within 60°).
    if hues.len() >= 2 {
        let min_h = hues.iter().copied().fold(f32::INFINITY, f32::min);
        let max_h = hues.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_h - min_h < 60.0 {
            return ColorHarmony::Analogous;
        }
    }

    // Check triadic.
    if hues.len() >= 3 {
        for i in 0..hues.len() {
            for j in (i + 1)..hues.len() {
                for k in (j + 1)..hues.len() {
                    let d1 = (hues[j] - hues[i]).rem_euclid(360.0);
                    let d2 = (hues[k] - hues[j]).rem_euclid(360.0);
                    if (d1 - 120.0).abs() < 20.0 && (d2 - 120.0).abs() < 20.0 {
                        return ColorHarmony::Triadic;
                    }
                }
            }
        }
    }

    ColorHarmony::None
}

/// Simple k-means color quantizer for dominant color extraction.
#[derive(Debug)]
pub struct KMeansQuantizer {
    /// Number of clusters (dominant colors).
    pub k: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl KMeansQuantizer {
    /// Create a new quantizer.
    #[must_use]
    pub fn new(k: usize, max_iterations: usize) -> Self {
        Self { k, max_iterations }
    }

    /// Extract dominant colors from a list of RGB samples.
    ///
    /// Uses a simple deterministic seeding (evenly spread) to avoid randomness.
    #[must_use]
    pub fn quantize(&self, samples: &[RgbColor]) -> Vec<DominantColor> {
        if samples.is_empty() || self.k == 0 {
            return Vec::new();
        }

        let k = self.k.min(samples.len());

        // Seed centroids by evenly spacing through the sample array.
        let mut centroids: Vec<RgbColor> = (0..k).map(|i| samples[i * samples.len() / k]).collect();

        let mut assignments = vec![0usize; samples.len()];

        for _ in 0..self.max_iterations {
            // Assign each sample to nearest centroid.
            let mut changed = false;
            for (idx, sample) in samples.iter().enumerate() {
                let nearest = (0..k)
                    .min_by(|&a, &b| {
                        sample
                            .distance(&centroids[a])
                            .partial_cmp(&sample.distance(&centroids[b]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                if assignments[idx] != nearest {
                    assignments[idx] = nearest;
                    changed = true;
                }
            }

            // Recompute centroids.
            let mut sums = vec![(0.0f32, 0.0f32, 0.0f32); k];
            let mut counts = vec![0usize; k];
            for (idx, sample) in samples.iter().enumerate() {
                let c = assignments[idx];
                sums[c].0 += sample.r;
                sums[c].1 += sample.g;
                sums[c].2 += sample.b;
                counts[c] += 1;
            }
            for i in 0..k {
                if counts[i] > 0 {
                    centroids[i] = RgbColor::new(
                        sums[i].0 / counts[i] as f32,
                        sums[i].1 / counts[i] as f32,
                        sums[i].2 / counts[i] as f32,
                    );
                }
            }

            if !changed {
                break;
            }
        }

        // Compute proportions.
        let mut counts = vec![0usize; k];
        for &a in &assignments {
            counts[a] += 1;
        }
        let total = samples.len() as f32;

        let mut result: Vec<DominantColor> = (0..k)
            .map(|i| DominantColor {
                color: centroids[i],
                proportion: counts[i] as f32 / total,
            })
            .collect();

        result.sort_by(|a, b| {
            b.proportion
                .partial_cmp(&a.proportion)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

/// Color palette — a named collection of dominant colors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Extracted dominant colors, sorted by proportion descending.
    pub colors: Vec<DominantColor>,
    /// Detected color harmony.
    pub harmony: ColorHarmony,
    /// Mean luminance of the palette.
    pub mean_luminance: f32,
}

impl ColorPalette {
    /// Build a palette from dominant colors.
    #[must_use]
    pub fn from_dominant(colors: Vec<DominantColor>) -> Self {
        let hues: Vec<f32> = colors.iter().map(|d| d.color.to_hsl().0).collect();

        let harmony = detect_harmony(&hues);

        let mean_luminance = if colors.is_empty() {
            0.0
        } else {
            colors
                .iter()
                .map(|d| d.color.luminance() * d.proportion)
                .sum()
        };

        Self {
            colors,
            harmony,
            mean_luminance,
        }
    }
}

/// Color temperature estimation (warm / cool / neutral).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorTemperature {
    /// Warm (orange/red dominant).
    Warm,
    /// Cool (blue dominant).
    Cool,
    /// Neutral.
    Neutral,
}

/// Estimate color temperature from mean RGB.
#[must_use]
pub fn estimate_color_temperature(mean_r: f32, _mean_g: f32, mean_b: f32) -> ColorTemperature {
    if mean_r > mean_b + 0.1 {
        ColorTemperature::Warm
    } else if mean_b > mean_r + 0.1 {
        ColorTemperature::Cool
    } else {
        ColorTemperature::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_from_u8_roundtrip() {
        let color = RgbColor::from_u8(255, 128, 0);
        let (r, g, b) = color.to_u8();
        assert_eq!(r, 255);
        assert_eq!(g, 128);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_rgb_distance_same() {
        let c = RgbColor::new(0.5, 0.5, 0.5);
        assert!((c.distance(&c)).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_distance_red_blue() {
        let red = RgbColor::new(1.0, 0.0, 0.0);
        let blue = RgbColor::new(0.0, 0.0, 1.0);
        let d = red.distance(&blue);
        assert!((d - 2.0_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_rgb_to_hsl_white() {
        let white = RgbColor::new(1.0, 1.0, 1.0);
        let (_, s, l) = white.to_hsl();
        assert!((s).abs() < 1e-5);
        assert!((l - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsl_red() {
        let red = RgbColor::new(1.0, 0.0, 0.0);
        let (h, s, l) = red.to_hsl();
        assert!((h).abs() < 1.0); // hue ~ 0
        assert!(s > 0.9);
        assert!((l - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_luminance_white() {
        let white = RgbColor::new(1.0, 1.0, 1.0);
        assert!((white.luminance() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_luminance_black() {
        let black = RgbColor::new(0.0, 0.0, 0.0);
        assert!((black.luminance()).abs() < 1e-5);
    }

    #[test]
    fn test_detect_harmony_monochromatic() {
        let hues = vec![10.0_f32, 15.0, 20.0];
        let h = detect_harmony(&hues);
        assert_eq!(h, ColorHarmony::Analogous);
    }

    #[test]
    fn test_detect_harmony_complementary() {
        let hues = vec![0.0_f32, 180.0];
        let h = detect_harmony(&hues);
        assert_eq!(h, ColorHarmony::Complementary);
    }

    #[test]
    fn test_detect_harmony_triadic() {
        let hues = vec![0.0_f32, 120.0, 240.0];
        let h = detect_harmony(&hues);
        assert_eq!(h, ColorHarmony::Triadic);
    }

    #[test]
    fn test_kmeans_basic() {
        let samples: Vec<RgbColor> = (0..100)
            .map(|i| {
                if i < 50 {
                    RgbColor::new(1.0, 0.0, 0.0)
                } else {
                    RgbColor::new(0.0, 0.0, 1.0)
                }
            })
            .collect();
        let q = KMeansQuantizer::new(2, 20);
        let result = q.quantize(&samples);
        assert_eq!(result.len(), 2);
        // Each cluster should have ~50% proportion.
        assert!((result[0].proportion - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_kmeans_empty() {
        let q = KMeansQuantizer::new(3, 10);
        let result = q.quantize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_color_palette_from_dominant() {
        // Red (hue=0) and cyan (hue=180) are complementary in HSL
        let colors = vec![
            DominantColor {
                color: RgbColor::new(1.0, 0.0, 0.0),
                proportion: 0.5,
            },
            DominantColor {
                color: RgbColor::new(0.0, 1.0, 1.0),
                proportion: 0.5,
            },
        ];
        let palette = ColorPalette::from_dominant(colors);
        assert_eq!(palette.harmony, ColorHarmony::Complementary);
        assert!(palette.mean_luminance >= 0.0);
    }

    #[test]
    fn test_color_temperature_warm() {
        assert_eq!(
            estimate_color_temperature(0.9, 0.5, 0.2),
            ColorTemperature::Warm
        );
    }

    #[test]
    fn test_color_temperature_cool() {
        assert_eq!(
            estimate_color_temperature(0.2, 0.5, 0.9),
            ColorTemperature::Cool
        );
    }

    #[test]
    fn test_color_temperature_neutral() {
        assert_eq!(
            estimate_color_temperature(0.5, 0.5, 0.5),
            ColorTemperature::Neutral
        );
    }
}
