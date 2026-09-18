//! Color matching between clips or images.
//!
//! Implements Reinhard et al. color transfer, histogram matching,
//! and related statistical color matching methods.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// Per-channel color statistics (mean, std_dev, min, max).
#[derive(Debug, Clone)]
pub struct ColorStats {
    /// Per-channel mean (RGB).
    pub mean: [f32; 3],
    /// Per-channel standard deviation (RGB).
    pub std_dev: [f32; 3],
    /// Per-channel minimum (RGB).
    pub min: [f32; 3],
    /// Per-channel maximum (RGB).
    pub max: [f32; 3],
}

impl ColorStats {
    /// Compute color statistics from a slice of RGB pixel triplets.
    #[must_use]
    pub fn from_pixels(pixels: &[[f32; 3]]) -> Self {
        if pixels.is_empty() {
            return Self {
                mean: [0.0; 3],
                std_dev: [1.0; 3],
                min: [0.0; 3],
                max: [1.0; 3],
            };
        }

        let n = pixels.len() as f32;
        let mut sum = [0.0f32; 3];
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for px in pixels {
            for c in 0..3 {
                sum[c] += px[c];
                if px[c] < min[c] {
                    min[c] = px[c];
                }
                if px[c] > max[c] {
                    max[c] = px[c];
                }
            }
        }

        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

        let mut var = [0.0f32; 3];
        for px in pixels {
            for c in 0..3 {
                let diff = px[c] - mean[c];
                var[c] += diff * diff;
            }
        }

        let std_dev = [
            (var[0] / n).sqrt(),
            (var[1] / n).sqrt(),
            (var[2] / n).sqrt(),
        ];

        Self {
            mean,
            std_dev,
            min,
            max,
        }
    }
}

impl Default for ColorStats {
    fn default() -> Self {
        Self {
            mean: [0.5; 3],
            std_dev: [0.2; 3],
            min: [0.0; 3],
            max: [1.0; 3],
        }
    }
}

/// Reinhard et al. (2001) color transfer algorithm.
///
/// Transfers the color appearance of the target image to the source pixel
/// using Lab space statistics matching.
pub struct ReinhardColorTransfer;

impl ReinhardColorTransfer {
    /// Transfer color from `target` statistics to a source pixel.
    ///
    /// Algorithm (simplified version in RGB):
    /// 1. Shift by mean difference
    /// 2. Scale by std_dev ratio
    ///
    /// A full implementation operates in Lab space; this version uses RGB
    /// as a pragmatic approximation.
    #[must_use]
    pub fn transfer(source: &ColorStats, target: &ColorStats, pixel: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let normalized = pixel[c] - source.mean[c];
            let ratio = if source.std_dev[c] > 1e-8 {
                target.std_dev[c] / source.std_dev[c]
            } else {
                1.0
            };
            out[c] = (normalized * ratio + target.mean[c]).clamp(0.0, 1.0);
        }
        out
    }

    /// Transfer color for an entire image.
    #[must_use]
    pub fn transfer_image(
        source_stats: &ColorStats,
        target_stats: &ColorStats,
        pixels: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        pixels
            .iter()
            .map(|&px| Self::transfer(source_stats, target_stats, px))
            .collect()
    }
}

/// Histogram-based color matching.
///
/// Uses CDF matching to map one image's histogram to another's.
pub struct HistogramMatching;

impl HistogramMatching {
    /// Compute cumulative distribution function (CDF) from a histogram.
    ///
    /// Returns a normalized CDF (values 0..1).
    #[must_use]
    pub fn compute_cdf(histogram: &[u32]) -> Vec<f32> {
        if histogram.is_empty() {
            return Vec::new();
        }
        let total: u64 = histogram.iter().map(|&v| v as u64).sum();
        if total == 0 {
            return vec![0.0; histogram.len()];
        }
        let mut cdf = Vec::with_capacity(histogram.len());
        let mut cumsum = 0u64;
        for &count in histogram {
            cumsum += count as u64;
            cdf.push(cumsum as f32 / total as f32);
        }
        cdf
    }

    /// Match the histogram of `src_pixels` to that of `ref_pixels`.
    ///
    /// Both inputs are single-channel values normalized to 0..1.
    /// Returns the transformed src_pixels.
    #[must_use]
    pub fn match_histograms(src_pixels: &[f32], ref_pixels: &[f32]) -> Vec<f32> {
        const BINS: usize = 256;

        // Build source histogram
        let mut src_hist = vec![0u32; BINS];
        for &v in src_pixels {
            let bin = ((v.clamp(0.0, 1.0) * (BINS - 1) as f32).round() as usize).min(BINS - 1);
            src_hist[bin] += 1;
        }

        // Build reference histogram
        let mut ref_hist = vec![0u32; BINS];
        for &v in ref_pixels {
            let bin = ((v.clamp(0.0, 1.0) * (BINS - 1) as f32).round() as usize).min(BINS - 1);
            ref_hist[bin] += 1;
        }

        let src_cdf = Self::compute_cdf(&src_hist);
        let ref_cdf = Self::compute_cdf(&ref_hist);

        // Build LUT: for each src bin, find the ref bin with closest CDF value
        let mut lut = vec![0usize; BINS];
        for (src_bin, &src_val) in src_cdf.iter().enumerate() {
            // Find closest ref CDF value using binary search
            let mut best_bin = 0;
            let mut best_diff = f32::MAX;
            for (ref_bin, &ref_val) in ref_cdf.iter().enumerate() {
                let diff = (src_val - ref_val).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_bin = ref_bin;
                }
            }
            lut[src_bin] = best_bin;
        }

        // Apply LUT to source pixels
        src_pixels
            .iter()
            .map(|&v| {
                let bin = ((v.clamp(0.0, 1.0) * (BINS - 1) as f32).round() as usize).min(BINS - 1);
                lut[bin] as f32 / (BINS - 1) as f32
            })
            .collect()
    }
}

/// Method for color matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMethod {
    /// Reinhard color transfer (mean/std-dev in Lab/RGB).
    Reinhard,
    /// Histogram-based CDF matching.
    Histogram,
    /// Simple mean and std-dev matching per channel.
    MeanStdDev,
    /// LUT-based matching.
    LutBased,
}

/// Configuration for color matching operations.
#[derive(Debug, Clone)]
pub struct ColorMatchConfig {
    /// Matching method to use.
    pub method: MatchMethod,
    /// Fraction of pixels to sample (0..1). 1.0 = use all pixels.
    pub spatial_sampling: f32,
    /// Apply matching only to shadow regions (luminance < 0.5).
    pub apply_to_shadows_only: bool,
}

impl ColorMatchConfig {
    /// Create a new color match configuration.
    #[must_use]
    pub fn new(method: MatchMethod, spatial_sampling: f32, apply_to_shadows_only: bool) -> Self {
        Self {
            method,
            spatial_sampling: spatial_sampling.clamp(0.001, 1.0),
            apply_to_shadows_only,
        }
    }

    /// Default Reinhard color transfer configuration.
    #[must_use]
    pub fn reinhard() -> Self {
        Self::new(MatchMethod::Reinhard, 1.0, false)
    }

    /// Histogram matching configuration.
    #[must_use]
    pub fn histogram() -> Self {
        Self::new(MatchMethod::Histogram, 1.0, false)
    }
}

impl Default for ColorMatchConfig {
    fn default() -> Self {
        Self::reinhard()
    }
}

/// Perform color matching on an image.
///
/// Returns the source pixels color-matched to the target statistics.
#[must_use]
pub fn color_match(
    src_pixels: &[[f32; 3]],
    target_pixels: &[[f32; 3]],
    config: &ColorMatchConfig,
) -> Vec<[f32; 3]> {
    let src_stats = ColorStats::from_pixels(src_pixels);
    let target_stats = ColorStats::from_pixels(target_pixels);

    match config.method {
        MatchMethod::Reinhard | MatchMethod::MeanStdDev => {
            let result =
                ReinhardColorTransfer::transfer_image(&src_stats, &target_stats, src_pixels);
            if config.apply_to_shadows_only {
                src_pixels
                    .iter()
                    .zip(result.iter())
                    .map(|(&src, &matched)| {
                        let luma = 0.2126 * src[0] + 0.7152 * src[1] + 0.0722 * src[2];
                        if luma < 0.5 {
                            matched
                        } else {
                            src
                        }
                    })
                    .collect()
            } else {
                result
            }
        }
        MatchMethod::Histogram => {
            // Apply per-channel histogram matching
            let channels: Vec<Vec<[f32; 3]>> = (0..3)
                .map(|c| {
                    let src_ch: Vec<f32> = src_pixels.iter().map(|px| px[c]).collect();
                    let tgt_ch: Vec<f32> = target_pixels.iter().map(|px| px[c]).collect();
                    let matched = HistogramMatching::match_histograms(&src_ch, &tgt_ch);
                    matched
                        .iter()
                        .map(|&v| {
                            let mut px = [0.0f32; 3];
                            px[c] = v;
                            px
                        })
                        .collect()
                })
                .collect();

            src_pixels
                .iter()
                .enumerate()
                .map(|(i, _)| [channels[0][i][0], channels[1][i][1], channels[2][i][2]])
                .collect()
        }
        MatchMethod::LutBased => {
            // LUT-based: use mean/std matching with clamping (simplified)
            ReinhardColorTransfer::transfer_image(&src_stats, &target_stats, src_pixels)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixels(n: usize, r: f32, g: f32, b: f32) -> Vec<[f32; 3]> {
        vec![[r, g, b]; n]
    }

    #[test]
    fn test_color_stats_from_pixels_uniform() {
        let pixels = make_pixels(100, 0.5, 0.3, 0.7);
        let stats = ColorStats::from_pixels(&pixels);
        assert!((stats.mean[0] - 0.5).abs() < 1e-5);
        assert!((stats.mean[1] - 0.3).abs() < 1e-5);
        assert!((stats.mean[2] - 0.7).abs() < 1e-5);
        // Uniform → zero std_dev
        assert!(stats.std_dev[0] < 1e-5);
    }

    #[test]
    fn test_color_stats_from_pixels_varying() {
        let pixels: Vec<[f32; 3]> = (0..100)
            .map(|i| [i as f32 / 99.0, 0.5, 1.0 - i as f32 / 99.0])
            .collect();
        let stats = ColorStats::from_pixels(&pixels);
        assert!((stats.mean[0] - 0.5).abs() < 0.01);
        assert!(stats.std_dev[0] > 0.0);
    }

    #[test]
    fn test_color_stats_empty() {
        let stats = ColorStats::from_pixels(&[]);
        assert_eq!(stats.mean, [0.0; 3]);
    }

    #[test]
    fn test_reinhard_transfer_identity() {
        // When source and target have same stats, pixel should be unchanged
        let pixels: Vec<[f32; 3]> = (0..50).map(|i| [i as f32 / 49.0; 3]).collect();
        let stats = ColorStats::from_pixels(&pixels);
        let pixel = [0.6, 0.4, 0.2];
        let result = ReinhardColorTransfer::transfer(&stats, &stats, pixel);
        for c in 0..3 {
            assert!(
                (result[c] - pixel[c]).abs() < 0.01,
                "Channel {c}: {} vs {}",
                result[c],
                pixel[c]
            );
        }
    }

    #[test]
    fn test_reinhard_transfer_shifts_mean() {
        let mut src_stats = ColorStats::default();
        src_stats.mean = [0.3, 0.3, 0.3];
        src_stats.std_dev = [0.1, 0.1, 0.1];

        let mut tgt_stats = ColorStats::default();
        tgt_stats.mean = [0.7, 0.7, 0.7];
        tgt_stats.std_dev = [0.1, 0.1, 0.1];

        let pixel = [0.3, 0.3, 0.3]; // at source mean
        let result = ReinhardColorTransfer::transfer(&src_stats, &tgt_stats, pixel);
        // At source mean, output should be at target mean
        assert!(
            (result[0] - 0.7).abs() < 0.01,
            "Expected ~0.7, got {}",
            result[0]
        );
    }

    #[test]
    fn test_reinhard_transfer_output_clamped() {
        let mut src_stats = ColorStats::default();
        src_stats.mean = [0.5; 3];
        src_stats.std_dev = [0.1; 3];

        let mut tgt_stats = ColorStats::default();
        tgt_stats.mean = [0.9; 3];
        tgt_stats.std_dev = [0.5; 3];

        let pixel = [1.0, 1.0, 1.0];
        let result = ReinhardColorTransfer::transfer(&src_stats, &tgt_stats, pixel);
        for c in 0..3 {
            assert!(
                result[c] <= 1.0 && result[c] >= 0.0,
                "Channel {c} out of range: {}",
                result[c]
            );
        }
    }

    #[test]
    fn test_histogram_compute_cdf_uniform() {
        let hist = vec![1u32; 256];
        let cdf = HistogramMatching::compute_cdf(&hist);
        assert_eq!(cdf.len(), 256);
        assert!((cdf[0] - 1.0 / 256.0).abs() < 1e-5);
        assert!((cdf[255] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_histogram_compute_cdf_empty() {
        let cdf = HistogramMatching::compute_cdf(&[]);
        assert!(cdf.is_empty());
    }

    #[test]
    fn test_histogram_compute_cdf_zeros() {
        let hist = vec![0u32; 10];
        let cdf = HistogramMatching::compute_cdf(&hist);
        assert!(cdf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_match_histograms_same_distribution() {
        let pixels: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let matched = HistogramMatching::match_histograms(&pixels, &pixels);
        assert_eq!(matched.len(), pixels.len());
        // Matching to itself should produce very similar output
        for (&orig, &m) in pixels.iter().zip(matched.iter()) {
            assert!((orig - m).abs() < 0.05, "orig={orig} matched={m}");
        }
    }

    #[test]
    fn test_match_histograms_output_range() {
        let src: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let reference: Vec<f32> = (0..100).map(|i| (i as f32 / 99.0) * 0.5).collect();
        let matched = HistogramMatching::match_histograms(&src, &reference);
        assert_eq!(matched.len(), src.len());
        assert!(matched.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_color_match_config_reinhard() {
        let config = ColorMatchConfig::reinhard();
        assert_eq!(config.method, MatchMethod::Reinhard);
        assert!((config.spatial_sampling - 1.0).abs() < 1e-6);
        assert!(!config.apply_to_shadows_only);
    }

    #[test]
    fn test_color_match_config_histogram() {
        let config = ColorMatchConfig::histogram();
        assert_eq!(config.method, MatchMethod::Histogram);
    }

    #[test]
    fn test_color_match_function_reinhard() {
        let src: Vec<[f32; 3]> = (0..50).map(|i| [i as f32 / 49.0; 3]).collect();
        let target: Vec<[f32; 3]> = (0..50).map(|i| [0.5 + i as f32 / 200.0; 3]).collect();
        let config = ColorMatchConfig::reinhard();
        let result = color_match(&src, &target, &config);
        assert_eq!(result.len(), src.len());
        assert!(result
            .iter()
            .all(|px| px.iter().all(|&v| v.is_finite() && v >= 0.0 && v <= 1.0)));
    }

    #[test]
    fn test_color_match_shadows_only() {
        let src: Vec<[f32; 3]> = vec![[0.1; 3], [0.9; 3]];
        let target: Vec<[f32; 3]> = vec![[0.5; 3]; 2];
        let config = ColorMatchConfig::new(MatchMethod::Reinhard, 1.0, true);
        let result = color_match(&src, &target, &config);
        assert_eq!(result.len(), 2);
        // Bright pixel should be unchanged
        for c in 0..3 {
            assert!(
                (result[1][c] - 0.9).abs() < 0.01,
                "Bright pixel should be unchanged: {}",
                result[1][c]
            );
        }
    }

    #[test]
    fn test_match_method_variants() {
        let _ = MatchMethod::Reinhard;
        let _ = MatchMethod::Histogram;
        let _ = MatchMethod::MeanStdDev;
        let _ = MatchMethod::LutBased;
    }
}
