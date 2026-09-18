#![allow(dead_code)]
//! Lighting analysis for video frames.
//!
//! Analyzes the lighting characteristics of a scene including direction,
//! intensity, contrast ratios, and classification (natural vs. artificial,
//! hard vs. soft). These metrics are useful for color grading decisions,
//! scene matching, and continuity checking.

/// Classification of the dominant light source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightSourceType {
    /// Natural sunlight or daylight.
    Natural,
    /// Artificial indoor lighting (tungsten, fluorescent, LED).
    Artificial,
    /// Mixed natural and artificial lighting.
    Mixed,
    /// Very low light / night scene.
    LowLight,
}

impl LightSourceType {
    /// Return a descriptive label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Artificial => "artificial",
            Self::Mixed => "mixed",
            Self::LowLight => "low_light",
        }
    }
}

/// Classification of light quality (hardness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightQuality {
    /// Hard light with sharp shadows (direct sun, spot).
    Hard,
    /// Soft, diffused light (overcast, bounce).
    Soft,
    /// Flat lighting with minimal shadow.
    Flat,
}

/// Overall result of lighting analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct LightingAnalysis {
    /// Average luminance of the frame (0.0 to 1.0).
    pub mean_luminance: f64,
    /// Standard deviation of luminance.
    pub luminance_std_dev: f64,
    /// Contrast ratio (max luminance / min luminance).
    pub contrast_ratio: f64,
    /// Dominant light direction as (dx, dy) where +x = right, +y = down.
    pub light_direction: (f64, f64),
    /// Estimated light source type.
    pub source_type: LightSourceType,
    /// Estimated light quality.
    pub quality: LightQuality,
    /// Histogram-based exposure score (-1.0 = underexposed, 0.0 = balanced, 1.0 = overexposed).
    pub exposure_bias: f64,
    /// Percentage of pixels that are clipped highlights (above 250).
    pub highlight_clip_pct: f64,
    /// Percentage of pixels that are crushed shadows (below 5).
    pub shadow_crush_pct: f64,
}

/// Configuration for the lighting analyzer.
#[derive(Debug, Clone)]
pub struct LightingAnalyzerConfig {
    /// Threshold below which a frame is considered low-light (0.0-1.0 luminance).
    pub low_light_threshold: f64,
    /// Threshold for hard light detection based on luminance std dev.
    pub hard_light_std_threshold: f64,
    /// Threshold for flat light detection.
    pub flat_light_std_threshold: f64,
}

impl Default for LightingAnalyzerConfig {
    fn default() -> Self {
        Self {
            low_light_threshold: 0.08,
            hard_light_std_threshold: 0.28,
            flat_light_std_threshold: 0.08,
        }
    }
}

/// Analyzer for frame lighting characteristics.
#[derive(Debug)]
pub struct LightingAnalyzer {
    /// Configuration.
    config: LightingAnalyzerConfig,
}

impl LightingAnalyzer {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: LightingAnalyzerConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: LightingAnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyze lighting of a grayscale frame (pixels 0-255, row-major).
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, pixels: &[u8], width: usize, height: usize) -> LightingAnalysis {
        let total = width * height;
        if total == 0 || pixels.len() < total {
            return LightingAnalysis {
                mean_luminance: 0.0,
                luminance_std_dev: 0.0,
                contrast_ratio: 1.0,
                light_direction: (0.0, 0.0),
                source_type: LightSourceType::LowLight,
                quality: LightQuality::Flat,
                exposure_bias: 0.0,
                highlight_clip_pct: 0.0,
                shadow_crush_pct: 0.0,
            };
        }

        let (mean, std_dev) = luminance_stats(&pixels[..total]);
        let contrast_ratio = compute_contrast_ratio(&pixels[..total]);
        let light_direction = estimate_light_direction(&pixels[..total], width, height);
        let (highlight_clip, shadow_crush) = clip_percentages(&pixels[..total]);
        let exposure_bias = compute_exposure_bias(mean);
        let source_type = self.classify_source(mean, std_dev);
        let quality = self.classify_quality(std_dev);

        LightingAnalysis {
            mean_luminance: mean,
            luminance_std_dev: std_dev,
            contrast_ratio,
            light_direction,
            source_type,
            quality,
            exposure_bias,
            highlight_clip_pct: highlight_clip,
            shadow_crush_pct: shadow_crush,
        }
    }

    /// Classify light source type.
    fn classify_source(&self, mean: f64, _std_dev: f64) -> LightSourceType {
        if mean < self.config.low_light_threshold {
            LightSourceType::LowLight
        } else if mean > 0.6 {
            LightSourceType::Natural
        } else if mean > 0.25 {
            LightSourceType::Mixed
        } else {
            LightSourceType::Artificial
        }
    }

    /// Classify light quality.
    fn classify_quality(&self, std_dev: f64) -> LightQuality {
        if std_dev > self.config.hard_light_std_threshold {
            LightQuality::Hard
        } else if std_dev < self.config.flat_light_std_threshold {
            LightQuality::Flat
        } else {
            LightQuality::Soft
        }
    }
}

impl Default for LightingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute mean and standard deviation of luminance (0-255 -> 0.0-1.0).
#[allow(clippy::cast_precision_loss)]
fn luminance_stats(pixels: &[u8]) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 0.0);
    }
    let n = pixels.len() as f64;
    let sum: f64 = pixels.iter().map(|&p| f64::from(p) / 255.0).sum();
    let mean = sum / n;
    let var: f64 = pixels
        .iter()
        .map(|&p| {
            let v = f64::from(p) / 255.0 - mean;
            v * v
        })
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

/// Compute contrast ratio from the luminance range.
#[allow(clippy::cast_precision_loss)]
fn compute_contrast_ratio(pixels: &[u8]) -> f64 {
    if pixels.is_empty() {
        return 1.0;
    }
    // Use 5th and 95th percentile to be robust against outliers
    let mut sorted: Vec<u8> = Vec::with_capacity(pixels.len());
    sorted.extend_from_slice(pixels);
    sorted.sort_unstable();

    let low_idx = sorted.len() * 5 / 100;
    let high_idx = sorted.len() * 95 / 100;
    let low = f64::from(sorted[low_idx]).max(1.0);
    let high = f64::from(sorted[high_idx]).max(1.0);
    high / low
}

/// Estimate light direction by comparing average luminance of image quadrants.
///
/// Returns (dx, dy) where positive dx means light from the right, positive dy means from below.
#[allow(clippy::cast_precision_loss)]
fn estimate_light_direction(pixels: &[u8], width: usize, height: usize) -> (f64, f64) {
    if width < 2 || height < 2 {
        return (0.0, 0.0);
    }

    let mid_x = width / 2;
    let mid_y = height / 2;

    let mut sum_left = 0.0_f64;
    let mut sum_right = 0.0_f64;
    let mut sum_top = 0.0_f64;
    let mut sum_bottom = 0.0_f64;
    let mut count_left = 0_u64;
    let mut count_right = 0_u64;
    let mut count_top = 0_u64;
    let mut count_bottom = 0_u64;

    for y in 0..height {
        for x in 0..width {
            let val = f64::from(pixels[y * width + x]);
            if x < mid_x {
                sum_left += val;
                count_left += 1;
            } else {
                sum_right += val;
                count_right += 1;
            }
            if y < mid_y {
                sum_top += val;
                count_top += 1;
            } else {
                sum_bottom += val;
                count_bottom += 1;
            }
        }
    }

    let avg_left = if count_left > 0 {
        sum_left / count_left as f64
    } else {
        0.0
    };
    let avg_right = if count_right > 0 {
        sum_right / count_right as f64
    } else {
        0.0
    };
    let avg_top = if count_top > 0 {
        sum_top / count_top as f64
    } else {
        0.0
    };
    let avg_bottom = if count_bottom > 0 {
        sum_bottom / count_bottom as f64
    } else {
        0.0
    };

    let dx = (avg_right - avg_left) / 255.0;
    let dy = (avg_bottom - avg_top) / 255.0;

    (dx, dy)
}

/// Compute the percentage of clipped highlights and crushed shadows.
#[allow(clippy::cast_precision_loss)]
fn clip_percentages(pixels: &[u8]) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 0.0);
    }
    let n = pixels.len() as f64;
    let highlights = pixels.iter().filter(|&&p| p >= 250).count() as f64;
    let shadows = pixels.iter().filter(|&&p| p <= 5).count() as f64;
    (highlights / n * 100.0, shadows / n * 100.0)
}

/// Compute exposure bias from mean luminance (-1 to 1).
fn compute_exposure_bias(mean_luminance: f64) -> f64 {
    // Ideal exposure is around 0.45 (middle gray)
    (mean_luminance - 0.45) * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_source_type_label() {
        assert_eq!(LightSourceType::Natural.label(), "natural");
        assert_eq!(LightSourceType::LowLight.label(), "low_light");
    }

    #[test]
    fn test_luminance_stats_uniform() {
        let pixels = vec![128_u8; 100];
        let (mean, std_dev) = luminance_stats(&pixels);
        assert!((mean - 128.0 / 255.0).abs() < 1e-6);
        assert!(std_dev < 1e-10);
    }

    #[test]
    fn test_luminance_stats_binary() {
        let mut pixels = vec![0_u8; 50];
        pixels.extend(vec![255_u8; 50]);
        let (mean, std_dev) = luminance_stats(&pixels);
        assert!((mean - 0.5).abs() < 1e-6);
        assert!(std_dev > 0.4);
    }

    #[test]
    fn test_contrast_ratio_uniform() {
        let pixels = vec![128_u8; 100];
        let ratio = compute_contrast_ratio(&pixels);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_contrast_ratio_high() {
        let mut pixels = vec![10_u8; 500];
        pixels.extend(vec![200_u8; 500]);
        let ratio = compute_contrast_ratio(&pixels);
        assert!(ratio > 5.0);
    }

    #[test]
    fn test_exposure_bias_middle_gray() {
        let bias = compute_exposure_bias(0.45);
        assert!(bias.abs() < 1e-10);
    }

    #[test]
    fn test_exposure_bias_overexposed() {
        let bias = compute_exposure_bias(0.9);
        assert!(bias > 0.5);
    }

    #[test]
    fn test_exposure_bias_underexposed() {
        let bias = compute_exposure_bias(0.1);
        assert!(bias < -0.5);
    }

    #[test]
    fn test_light_direction_left_bright() {
        let width = 20;
        let height = 10;
        let mut pixels = vec![0_u8; width * height];
        // Left half bright
        for y in 0..height {
            for x in 0..width / 2 {
                pixels[y * width + x] = 200;
            }
        }
        let (dx, _dy) = estimate_light_direction(&pixels, width, height);
        // Light should come from left (brighter), so dx < 0
        assert!(dx < 0.0);
    }

    #[test]
    fn test_clip_percentages_no_clipping() {
        let pixels = vec![128_u8; 100];
        let (hi, lo) = clip_percentages(&pixels);
        assert!((hi - 0.0).abs() < 1e-10);
        assert!((lo - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_percentages_all_clipped() {
        let pixels = vec![255_u8; 100];
        let (hi, _lo) = clip_percentages(&pixels);
        assert!((hi - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_dark_frame() {
        let analyzer = LightingAnalyzer::new();
        let pixels = vec![5_u8; 32 * 32];
        let result = analyzer.analyze(&pixels, 32, 32);
        assert_eq!(result.source_type, LightSourceType::LowLight);
        assert!(result.mean_luminance < 0.05);
    }

    #[test]
    fn test_analyze_bright_uniform() {
        let analyzer = LightingAnalyzer::new();
        let pixels = vec![220_u8; 32 * 32];
        let result = analyzer.analyze(&pixels, 32, 32);
        assert_eq!(result.source_type, LightSourceType::Natural);
        assert_eq!(result.quality, LightQuality::Flat);
    }

    #[test]
    fn test_analyze_empty_frame() {
        let analyzer = LightingAnalyzer::new();
        let result = analyzer.analyze(&[], 0, 0);
        assert_eq!(result.source_type, LightSourceType::LowLight);
        assert_eq!(result.quality, LightQuality::Flat);
    }

    #[test]
    fn test_analyzer_with_config() {
        let config = LightingAnalyzerConfig {
            low_light_threshold: 0.2,
            ..Default::default()
        };
        let analyzer = LightingAnalyzer::with_config(config);
        // A dim frame that would normally be "Artificial" becomes "LowLight" with higher threshold
        let pixels = vec![30_u8; 16 * 16];
        let result = analyzer.analyze(&pixels, 16, 16);
        assert_eq!(result.source_type, LightSourceType::LowLight);
    }
}
