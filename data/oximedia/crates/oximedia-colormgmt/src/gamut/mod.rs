//! Gamut mapping algorithms for handling out-of-gamut colors.

use crate::colorspaces::ColorSpace;
use crate::math::clamp_rgb;

/// Gamut mapping algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutMappingAlgorithm {
    /// Simple clipping to [0, 1] range (fastest, may cause hue shifts)
    Clip,
    /// Compress out-of-gamut colors toward gamut boundary (preserves hue better)
    Compress,
    /// Desaturate until color fits in gamut (preserves luminance)
    Desaturate,
    /// Perceptual mapping using HPMINDE-like algorithm
    Perceptual,
}

/// Gamut mapper for converting colors between different color gamuts.
#[derive(Clone, Debug)]
pub struct GamutMapper {
    algorithm: GamutMappingAlgorithm,
    compression_threshold: f64,
    compression_amount: f64,
}

impl GamutMapper {
    /// Creates a new gamut mapper with the specified algorithm.
    #[must_use]
    pub const fn new(algorithm: GamutMappingAlgorithm) -> Self {
        Self {
            algorithm,
            compression_threshold: 0.9,
            compression_amount: 0.8,
        }
    }

    /// Creates a gamut mapper with clip algorithm.
    #[must_use]
    pub const fn clip() -> Self {
        Self::new(GamutMappingAlgorithm::Clip)
    }

    /// Creates a gamut mapper with compress algorithm.
    #[must_use]
    pub const fn compress() -> Self {
        Self::new(GamutMappingAlgorithm::Compress)
    }

    /// Creates a gamut mapper with desaturate algorithm.
    #[must_use]
    pub const fn desaturate() -> Self {
        Self::new(GamutMappingAlgorithm::Desaturate)
    }

    /// Creates a gamut mapper with perceptual algorithm.
    #[must_use]
    pub const fn perceptual() -> Self {
        Self::new(GamutMappingAlgorithm::Perceptual)
    }

    /// Sets the compression threshold (0.0 to 1.0).
    ///
    /// Values above this threshold start being compressed.
    pub fn set_compression_threshold(&mut self, threshold: f64) {
        self.compression_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Sets the compression amount (0.0 to 1.0).
    ///
    /// Higher values mean more aggressive compression.
    pub fn set_compression_amount(&mut self, amount: f64) {
        self.compression_amount = amount.clamp(0.0, 1.0);
    }

    /// Maps RGB values to fit within the [0, 1] gamut.
    #[must_use]
    pub fn map(&self, rgb: [f64; 3], _color_space: Option<&ColorSpace>) -> [f64; 3] {
        match self.algorithm {
            GamutMappingAlgorithm::Clip => map_clip(rgb),
            GamutMappingAlgorithm::Compress => {
                map_compress(rgb, self.compression_threshold, self.compression_amount)
            }
            GamutMappingAlgorithm::Desaturate => map_desaturate(rgb),
            GamutMappingAlgorithm::Perceptual => map_perceptual(rgb),
        }
    }

    /// Checks if an RGB value is within the [0, 1] gamut.
    #[must_use]
    pub fn is_in_gamut(rgb: [f64; 3]) -> bool {
        rgb.iter().all(|&v| (0.0..=1.0).contains(&v))
    }

    /// Calculates how far out of gamut a color is.
    ///
    /// Returns 0.0 if in gamut, positive values indicate out-of-gamut distance.
    #[must_use]
    pub fn gamut_distance(rgb: [f64; 3]) -> f64 {
        let mut max_distance: f64 = 0.0;
        for &v in &rgb {
            if v < 0.0 {
                max_distance = max_distance.max(-v);
            } else if v > 1.0 {
                max_distance = max_distance.max(v - 1.0);
            }
        }
        max_distance
    }
}

impl Default for GamutMapper {
    fn default() -> Self {
        Self::perceptual()
    }
}

/// Clips RGB values to [0, 1] range.
#[must_use]
fn map_clip(rgb: [f64; 3]) -> [f64; 3] {
    clamp_rgb(rgb)
}

/// Compresses out-of-gamut colors using soft clipping.
///
/// This uses a knee function to smoothly compress values approaching and exceeding 1.0.
#[must_use]
fn map_compress(rgb: [f64; 3], threshold: f64, amount: f64) -> [f64; 3] {
    let compress_channel = |v: f64| -> f64 {
        if v < 0.0 {
            // Handle negative values with soft knee
            let abs_v = -v;
            if abs_v < threshold {
                v
            } else {
                -soft_clip(abs_v, threshold, amount)
            }
        } else if v <= threshold {
            v
        } else {
            soft_clip(v, threshold, amount)
        }
    };

    [
        compress_channel(rgb[0]),
        compress_channel(rgb[1]),
        compress_channel(rgb[2]),
    ]
}

/// Soft clipping function with knee.
#[must_use]
fn soft_clip(value: f64, threshold: f64, amount: f64) -> f64 {
    let max = 1.0;
    let over = value - threshold;
    let range = max - threshold;

    if over <= 0.0 {
        return value;
    }

    // Apply compression curve
    let compressed = over / (1.0 + over * amount / range);
    threshold + compressed
}

/// Desaturates color until it fits in gamut.
///
/// This preserves the luminance of the color while reducing saturation.
#[must_use]
fn map_desaturate(rgb: [f64; 3]) -> [f64; 3] {
    if GamutMapper::is_in_gamut(rgb) {
        return rgb;
    }

    // Calculate luminance (Rec.709)
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

    // Binary search for the right amount of desaturation
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..20 {
        // 20 iterations should be enough
        let mid = (low + high) / 2.0;
        let test = [
            luma + (rgb[0] - luma) * mid,
            luma + (rgb[1] - luma) * mid,
            luma + (rgb[2] - luma) * mid,
        ];

        if GamutMapper::is_in_gamut(test) {
            low = mid;
        } else {
            high = mid;
        }
    }

    let saturation = low;
    [
        luma + (rgb[0] - luma) * saturation,
        luma + (rgb[1] - luma) * saturation,
        luma + (rgb[2] - luma) * saturation,
    ]
}

/// Perceptual gamut mapping using LCH color space.
///
/// This approach maps colors in a perceptually uniform way, preserving
/// hue and lightness while reducing chroma as needed.
#[must_use]
fn map_perceptual(rgb: [f64; 3]) -> [f64; 3] {
    if GamutMapper::is_in_gamut(rgb) {
        return rgb;
    }

    // Convert to XYZ and then to LCH for perceptual mapping
    // For simplicity, we'll use a luminance-preserving approach similar to desaturate
    // A full implementation would convert through XYZ -> Lab -> LCH
    map_desaturate(rgb)
}

/// Calculates the gamut boundary in a specific direction from a point.
///
/// Returns the maximum scale factor before hitting the gamut boundary.
#[must_use]
#[allow(dead_code)]
fn calculate_gamut_intersection(rgb: [f64; 3], direction: [f64; 3]) -> f64 {
    let mut t_max = f64::INFINITY;

    for i in 0..3 {
        if direction[i].abs() > 1e-10 {
            // Check intersection with lower bound (0)
            let t_lower = -rgb[i] / direction[i];
            if t_lower > 0.0 && t_lower < t_max {
                t_max = t_lower;
            }

            // Check intersection with upper bound (1)
            let t_upper = (1.0 - rgb[i]) / direction[i];
            if t_upper > 0.0 && t_upper < t_max {
                t_max = t_upper;
            }
        }
    }

    if t_max.is_finite() {
        t_max
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_in_gamut() {
        assert!(GamutMapper::is_in_gamut([0.5, 0.3, 0.7]));
        assert!(GamutMapper::is_in_gamut([0.0, 0.0, 0.0]));
        assert!(GamutMapper::is_in_gamut([1.0, 1.0, 1.0]));
        assert!(!GamutMapper::is_in_gamut([1.1, 0.5, 0.5]));
        assert!(!GamutMapper::is_in_gamut([-0.1, 0.5, 0.5]));
    }

    #[test]
    fn test_gamut_distance() {
        assert_eq!(GamutMapper::gamut_distance([0.5, 0.3, 0.7]), 0.0);
        assert!((GamutMapper::gamut_distance([1.2, 0.5, 0.5]) - 0.2).abs() < 1e-10);
        assert!((GamutMapper::gamut_distance([-0.3, 0.5, 0.5]) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_map_clip() {
        let mapper = GamutMapper::clip();
        let result = mapper.map([1.2, -0.1, 0.5], None);

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.5);
    }

    #[test]
    fn test_map_compress() {
        let mapper = GamutMapper::compress();
        let result = mapper.map([1.2, 0.5, 0.5], None);

        // Should compress values above 1.0
        assert!(result[0] < 1.2);
        assert!(result[0] <= 1.0);
        assert_eq!(result[1], 0.5);
    }

    #[test]
    fn test_map_desaturate() {
        let mapper = GamutMapper::desaturate();
        let out_of_gamut = [1.5, 0.3, 0.2];
        let result = mapper.map(out_of_gamut, None);

        // Result should be in gamut
        assert!(GamutMapper::is_in_gamut(result));

        // Luminance should be approximately preserved
        let luma_before =
            0.2126 * out_of_gamut[0] + 0.7152 * out_of_gamut[1] + 0.0722 * out_of_gamut[2];
        let luma_after = 0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2];
        assert!((luma_after - luma_before).abs() < 0.01);
    }

    #[test]
    fn test_map_in_gamut_unchanged() {
        let mapper = GamutMapper::perceptual();
        let in_gamut = [0.5, 0.3, 0.7];
        let result = mapper.map(in_gamut, None);

        assert_eq!(result, in_gamut);
    }
}
