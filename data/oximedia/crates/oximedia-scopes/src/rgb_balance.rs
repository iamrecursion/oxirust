#![allow(dead_code)]
//! RGB channel balance analysis scope.
//!
//! Evaluates the relative balance between the red, green, and blue channels
//! of a video frame. Useful for detecting color casts, verifying white balance,
//! and ensuring neutral grading across zones of the image.

/// Summary statistics for a single color channel.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStats {
    /// Mean value (0.0..=255.0 for 8-bit).
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum sample value.
    pub min: u8,
    /// Maximum sample value.
    pub max: u8,
    /// Median value.
    pub median: u8,
}

/// Result of an RGB balance analysis.
#[derive(Debug, Clone)]
pub struct RgbBalanceReport {
    /// Red channel statistics.
    pub red: ChannelStats,
    /// Green channel statistics.
    pub green: ChannelStats,
    /// Blue channel statistics.
    pub blue: ChannelStats,
    /// Difference between the highest and lowest channel means.
    pub max_mean_spread: f64,
    /// Whether the frame appears to have a color cast.
    pub has_color_cast: bool,
    /// Dominant channel if a cast is detected (0=R, 1=G, 2=B), or None.
    pub dominant_channel: Option<usize>,
    /// Gray-world white balance correction gains [R, G, B].
    pub gray_world_gains: [f64; 3],
}

/// Configuration for RGB balance analysis.
#[derive(Debug, Clone)]
pub struct RgbBalanceConfig {
    /// Threshold for mean spread to detect a color cast.
    pub cast_threshold: f64,
    /// Whether to compute gray-world gains.
    pub compute_gray_world: bool,
}

impl Default for RgbBalanceConfig {
    fn default() -> Self {
        Self {
            cast_threshold: 5.0,
            compute_gray_world: true,
        }
    }
}

/// Computes per-channel statistics from an RGB24 frame.
#[allow(clippy::cast_precision_loss)]
fn compute_channel_stats(frame: &[u8], pixel_count: usize, channel: usize) -> ChannelStats {
    if pixel_count == 0 {
        return ChannelStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0,
            max: 0,
            median: 0,
        };
    }

    let mut sum: u64 = 0;
    let mut min_val: u8 = 255;
    let mut max_val: u8 = 0;
    let mut histogram = [0u32; 256];

    for i in 0..pixel_count {
        let v = frame[i * 3 + channel];
        sum += u64::from(v);
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        histogram[v as usize] += 1;
    }

    let mean = sum as f64 / pixel_count as f64;

    // Standard deviation
    let mut var_sum = 0.0_f64;
    for i in 0..pixel_count {
        let v = f64::from(frame[i * 3 + channel]);
        let diff = v - mean;
        var_sum += diff * diff;
    }
    let std_dev = (var_sum / pixel_count as f64).sqrt();

    // Median from histogram
    let half = (pixel_count as u32 + 1) / 2;
    let mut cumulative = 0u32;
    let mut median = 0u8;
    for (val, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= half {
            #[allow(clippy::cast_possible_truncation)]
            {
                median = val as u8;
            }
            break;
        }
    }

    ChannelStats {
        mean,
        std_dev,
        min: min_val,
        max: max_val,
        median,
    }
}

/// Analyzes the RGB balance of an RGB24 video frame.
///
/// # Arguments
///
/// * `frame` - RGB24 pixel data (3 bytes per pixel, row-major).
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `config` - Analysis configuration.
///
/// # Returns
///
/// An `RgbBalanceReport`, or `None` if the frame data is invalid.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyze_rgb_balance(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &RgbBalanceConfig,
) -> Option<RgbBalanceReport> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 || frame.len() < pixel_count * 3 {
        return None;
    }

    let red = compute_channel_stats(frame, pixel_count, 0);
    let green = compute_channel_stats(frame, pixel_count, 1);
    let blue = compute_channel_stats(frame, pixel_count, 2);

    let means = [red.mean, green.mean, blue.mean];
    let max_mean = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_mean = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mean_spread = max_mean - min_mean;

    let has_color_cast = max_mean_spread > config.cast_threshold;

    let dominant_channel = if has_color_cast {
        means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    } else {
        None
    };

    let gray_world_gains = if config.compute_gray_world {
        let avg_of_means = (red.mean + green.mean + blue.mean) / 3.0;
        if red.mean > 0.0 && green.mean > 0.0 && blue.mean > 0.0 {
            [
                avg_of_means / red.mean,
                avg_of_means / green.mean,
                avg_of_means / blue.mean,
            ]
        } else {
            [1.0, 1.0, 1.0]
        }
    } else {
        [1.0, 1.0, 1.0]
    };

    Some(RgbBalanceReport {
        red,
        green,
        blue,
        max_mean_spread,
        has_color_cast,
        dominant_channel,
        gray_world_gains,
    })
}

/// Computes the color temperature bias from RGB means.
///
/// Returns a value where negative = cool (blue bias), positive = warm (red bias).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn color_temperature_bias(red_mean: f64, blue_mean: f64) -> f64 {
    red_mean - blue_mean
}

/// Checks if the gray-world gains are within an acceptable tolerance.
#[must_use]
pub fn gains_within_tolerance(gains: &[f64; 3], tolerance: f64) -> bool {
    gains.iter().all(|&g| (g - 1.0).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut data = vec![0u8; pixel_count * 3];
        for i in 0..pixel_count {
            data[i * 3] = r;
            data[i * 3 + 1] = g;
            data[i * 3 + 2] = b;
        }
        data
    }

    #[test]
    fn test_neutral_frame() {
        let frame = make_uniform_frame(8, 8, 128, 128, 128);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 8, 8, &config).expect("should succeed in test");
        assert!((report.red.mean - 128.0).abs() < f64::EPSILON);
        assert!(!report.has_color_cast);
        assert!(report.dominant_channel.is_none());
    }

    #[test]
    fn test_red_cast() {
        let frame = make_uniform_frame(8, 8, 200, 128, 128);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 8, 8, &config).expect("should succeed in test");
        assert!(report.has_color_cast);
        assert_eq!(report.dominant_channel, Some(0));
    }

    #[test]
    fn test_blue_cast() {
        let frame = make_uniform_frame(8, 8, 100, 100, 200);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 8, 8, &config).expect("should succeed in test");
        assert!(report.has_color_cast);
        assert_eq!(report.dominant_channel, Some(2));
    }

    #[test]
    fn test_green_cast() {
        let frame = make_uniform_frame(8, 8, 100, 200, 100);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 8, 8, &config).expect("should succeed in test");
        assert!(report.has_color_cast);
        assert_eq!(report.dominant_channel, Some(1));
    }

    #[test]
    fn test_max_mean_spread() {
        let frame = make_uniform_frame(4, 4, 100, 150, 200);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 4, 4, &config).expect("should succeed in test");
        assert!((report.max_mean_spread - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gray_world_gains_neutral() {
        let frame = make_uniform_frame(4, 4, 128, 128, 128);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 4, 4, &config).expect("should succeed in test");
        for &g in &report.gray_world_gains {
            assert!((g - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_gray_world_gains_imbalanced() {
        let frame = make_uniform_frame(4, 4, 100, 200, 150);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 4, 4, &config).expect("should succeed in test");
        // Red gain should be > 1.0 (boost), green < 1.0 (reduce)
        assert!(report.gray_world_gains[0] > 1.0);
        assert!(report.gray_world_gains[1] < 1.0);
    }

    #[test]
    fn test_invalid_frame() {
        let config = RgbBalanceConfig::default();
        assert!(analyze_rgb_balance(&[0u8; 5], 10, 10, &config).is_none());
    }

    #[test]
    fn test_zero_dimensions() {
        let config = RgbBalanceConfig::default();
        assert!(analyze_rgb_balance(&[], 0, 0, &config).is_none());
    }

    #[test]
    fn test_channel_stats_std_dev() {
        // Two values: 0 and 255, alternating
        let mut frame = vec![0u8; 4 * 3];
        frame[0] = 0;
        frame[3] = 255;
        frame[6] = 0;
        frame[9] = 255;
        // set G,B to 0 for simplicity
        let stats = compute_channel_stats(&frame, 4, 0);
        assert!((stats.mean - 127.5).abs() < f64::EPSILON);
        assert!(stats.std_dev > 100.0);
    }

    #[test]
    fn test_color_temperature_bias_warm() {
        let bias = color_temperature_bias(200.0, 100.0);
        assert!(bias > 0.0);
    }

    #[test]
    fn test_color_temperature_bias_cool() {
        let bias = color_temperature_bias(100.0, 200.0);
        assert!(bias < 0.0);
    }

    #[test]
    fn test_gains_within_tolerance() {
        assert!(gains_within_tolerance(&[1.0, 1.0, 1.0], 0.01));
        assert!(!gains_within_tolerance(&[1.5, 1.0, 1.0], 0.01));
    }

    #[test]
    fn test_median_value() {
        let frame = make_uniform_frame(4, 4, 42, 42, 42);
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 4, 4, &config).expect("should succeed in test");
        assert_eq!(report.red.median, 42);
    }

    #[test]
    fn test_min_max() {
        let mut frame = make_uniform_frame(4, 1, 100, 100, 100);
        frame[0] = 10; // min red
        frame[9] = 250; // max red
        let config = RgbBalanceConfig::default();
        let report = analyze_rgb_balance(&frame, 4, 1, &config).expect("should succeed in test");
        assert_eq!(report.red.min, 10);
        assert_eq!(report.red.max, 250);
    }
}
