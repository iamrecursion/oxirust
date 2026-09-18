//! Simplified Error Level Analysis (ELA) for JPEG authenticity detection.
//!
//! This module provides a lightweight ELA implementation that works directly
//! on raw pixel arrays, without requiring the `image` crate infrastructure.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Summary result of an ELA analysis pass.
#[derive(Debug, Clone, Copy)]
pub struct ElaResult {
    /// Maximum per-pixel ELA value in the analysed region.
    pub max_ela: f32,
    /// Mean per-pixel ELA value across the analysed region.
    pub mean_ela: f32,
    /// Whether the image is considered suspicious (mean_ela ≥ 15.0).
    pub suspicious: bool,
}

impl ElaResult {
    /// Return `true` when the mean ELA value is below `15.0`, suggesting the
    /// image has not been recently re-saved at a different quality level.
    #[must_use]
    pub fn is_authentic(&self) -> bool {
        self.mean_ela < 15.0
    }
}

/// Configuration for the ELA algorithm.
#[derive(Debug, Clone, Copy)]
pub struct ElaConfig {
    /// JPEG re-save quality in 0–100.
    pub quality: u8,
    /// Multiplicative scale applied to absolute pixel differences.
    pub scale: f32,
}

impl Default for ElaConfig {
    fn default() -> Self {
        Self {
            quality: 90,
            scale: 15.0,
        }
    }
}

/// Compute the ELA value for a single pixel pair.
///
/// The result is `|original − recompressed| * scale`.
#[must_use]
pub fn compute_ela_pixel(original: u8, recompressed: u8, scale: f32) -> f32 {
    let diff = (i16::from(original) - i16::from(recompressed)).unsigned_abs() as f32;
    diff * scale
}

/// Analyse a flat pixel slice using ELA and return summary statistics.
///
/// # Arguments
///
/// * `original_pixels`     – Raw pixel values from the original image.
/// * `recompressed_pixels` – The same pixels after JPEG re-save.
/// * `config`              – ELA configuration (quality, scale).
///
/// If either slice is empty the function returns an authentic result with all
/// zeros.
#[must_use]
pub fn analyze_ela(
    original_pixels: &[u8],
    recompressed_pixels: &[u8],
    config: &ElaConfig,
) -> ElaResult {
    let n = original_pixels.len().min(recompressed_pixels.len());
    if n == 0 {
        return ElaResult {
            max_ela: 0.0,
            mean_ela: 0.0,
            suspicious: false,
        };
    }

    let mut max_ela = 0.0_f32;
    let mut sum_ela = 0.0_f32;

    for i in 0..n {
        let v = compute_ela_pixel(original_pixels[i], recompressed_pixels[i], config.scale);
        if v > max_ela {
            max_ela = v;
        }
        sum_ela += v;
    }

    let mean_ela = sum_ela / n as f32;
    let suspicious = mean_ela >= 15.0;

    ElaResult {
        max_ela,
        mean_ela,
        suspicious,
    }
}

/// A 2D ELA map for spatially-resolved analysis.
#[derive(Debug, Clone)]
pub struct ElaMap {
    /// Flat row-major ELA values.
    pub map: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl ElaMap {
    /// Return the ELA value at pixel column `x`, row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `(y * width + x)` is out of bounds.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.map[y * self.width + x]
    }

    /// Maximum value in the map, or `0.0` if the map is empty.
    #[must_use]
    pub fn max(&self) -> f32 {
        self.map.iter().cloned().fold(0.0_f32, f32::max)
    }

    /// Arithmetic mean of all values in the map, or `0.0` if empty.
    #[must_use]
    pub fn mean(&self) -> f32 {
        if self.map.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.map.iter().sum();
        sum / self.map.len() as f32
    }

    /// Build an [`ElaMap`] from original and recompressed pixel slices.
    #[must_use]
    pub fn build(
        original: &[u8],
        recompressed: &[u8],
        width: usize,
        height: usize,
        config: &ElaConfig,
    ) -> Self {
        let n = original.len().min(recompressed.len()).min(width * height);
        let map: Vec<f32> = (0..n)
            .map(|i| compute_ela_pixel(original[i], recompressed[i], config.scale))
            .collect();
        Self { map, width, height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ElaResult ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ela_result_is_authentic_low_mean() {
        let r = ElaResult {
            max_ela: 10.0,
            mean_ela: 5.0,
            suspicious: false,
        };
        assert!(r.is_authentic());
    }

    #[test]
    fn test_ela_result_is_authentic_boundary() {
        // Exactly 15.0 is NOT authentic (condition is < 15.0)
        let r = ElaResult {
            max_ela: 20.0,
            mean_ela: 15.0,
            suspicious: true,
        };
        assert!(!r.is_authentic());
    }

    #[test]
    fn test_ela_result_not_authentic_high_mean() {
        let r = ElaResult {
            max_ela: 100.0,
            mean_ela: 40.0,
            suspicious: true,
        };
        assert!(!r.is_authentic());
    }

    // ── ElaConfig ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ela_config_default() {
        let cfg = ElaConfig::default();
        assert_eq!(cfg.quality, 90);
        assert!(cfg.scale > 0.0);
    }

    // ── compute_ela_pixel ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_ela_pixel_identical() {
        assert_eq!(compute_ela_pixel(128, 128, 15.0), 0.0);
    }

    #[test]
    fn test_compute_ela_pixel_diff_of_one() {
        let v = compute_ela_pixel(100, 101, 15.0);
        assert!((v - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_ela_pixel_diff_of_ten() {
        let v = compute_ela_pixel(0, 10, 1.0);
        assert!((v - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_ela_pixel_reversed_order() {
        // Should use absolute difference
        let v = compute_ela_pixel(200, 100, 1.0);
        assert!((v - 100.0).abs() < 1e-4);
    }

    // ── analyze_ela ────────────────────────────────────────────────────────────

    #[test]
    fn test_analyze_ela_empty_slices() {
        let cfg = ElaConfig::default();
        let r = analyze_ela(&[], &[], &cfg);
        assert_eq!(r.max_ela, 0.0);
        assert_eq!(r.mean_ela, 0.0);
        assert!(r.is_authentic());
    }

    #[test]
    fn test_analyze_ela_identical_images() {
        let pixels = vec![128_u8; 100];
        let cfg = ElaConfig::default();
        let r = analyze_ela(&pixels, &pixels, &cfg);
        assert_eq!(r.max_ela, 0.0);
        assert_eq!(r.mean_ela, 0.0);
        assert!(r.is_authentic());
    }

    #[test]
    fn test_analyze_ela_detects_suspicious() {
        // All pixels differ by 1 → ela = 15.0 → suspicious
        let original = vec![100_u8; 50];
        let recompressed = vec![101_u8; 50];
        let cfg = ElaConfig::default(); // scale = 15.0
        let r = analyze_ela(&original, &recompressed, &cfg);
        assert!(r.suspicious);
        assert!(!r.is_authentic());
    }

    // ── ElaMap ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_ela_map_at() {
        let map = ElaMap {
            map: vec![1.0, 2.0, 3.0, 4.0],
            width: 2,
            height: 2,
        };
        assert_eq!(map.at(0, 0), 1.0);
        assert_eq!(map.at(1, 0), 2.0);
        assert_eq!(map.at(0, 1), 3.0);
        assert_eq!(map.at(1, 1), 4.0);
    }

    #[test]
    fn test_ela_map_max() {
        let map = ElaMap {
            map: vec![5.0, 10.0, 3.0],
            width: 3,
            height: 1,
        };
        assert!((map.max() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_ela_map_mean() {
        let map = ElaMap {
            map: vec![2.0, 4.0, 6.0],
            width: 3,
            height: 1,
        };
        assert!((map.mean() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_ela_map_build() {
        let original = vec![100_u8, 200_u8];
        let recompressed = vec![110_u8, 190_u8];
        let cfg = ElaConfig {
            quality: 90,
            scale: 1.0,
        };
        let m = ElaMap::build(&original, &recompressed, 2, 1, &cfg);
        assert!((m.at(0, 0) - 10.0).abs() < 1e-4);
        assert!((m.at(1, 0) - 10.0).abs() < 1e-4);
    }
}
