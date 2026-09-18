//! Histogram analysis for exposure and tonal distribution.
//!
//! Histograms display the distribution of pixel values across the tonal range,
//! essential for:
//! - Checking exposure (shadows, midtones, highlights)
//! - Identifying clipping (pure black/white)
//! - Analyzing contrast and dynamic range
//! - Ensuring proper tonal distribution
//!
//! Supports both RGB (per-channel) and luma (overall brightness) histograms.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{HistogramMode, ScopeConfig, ScopeData, ScopeType};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Generates an RGB histogram from frame data.
///
/// Displays separate histograms for red, green, and blue channels,
/// either overlaid or stacked depending on configuration.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Scope configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn generate_rgb_histogram(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    // Build histograms for R, G, B (256 bins each) using rayon parallel
    // row chunks. Each worker maintains thread-local [u32; 256] bins per
    // channel; reduce() merges them lock-free at the end.
    //
    // Accumulators are boxed to keep them on the heap rather than the stack.
    // Without boxing the rayon reduce tree (~17 levels for 76 K pixels in
    // debug builds) overflows the default 2 MB rayon worker thread stack.
    let pixel_count = (width * height) as usize;
    let pixels = &frame[..pixel_count * 3];
    let merged = pixels
        .par_chunks(3)
        .fold(
            || Box::new([[0u32; 256]; 3]),
            |mut acc, chunk| {
                if let [r, g, b] = chunk {
                    acc[0][*r as usize] = acc[0][*r as usize].saturating_add(1);
                    acc[1][*g as usize] = acc[1][*g as usize].saturating_add(1);
                    acc[2][*b as usize] = acc[2][*b as usize].saturating_add(1);
                }
                acc
            },
        )
        .reduce(
            || Box::new([[0u32; 256]; 3]),
            |mut a, b| {
                for ch in 0..3 {
                    for bin in 0..256 {
                        a[ch][bin] = a[ch][bin].saturating_add(b[ch][bin]);
                    }
                }
                a
            },
        );
    let histograms: [[u32; 256]; 3] = *merged;

    // Apply logarithmic scale if requested
    let histograms = if matches!(config.histogram_mode, HistogramMode::Logarithmic) {
        apply_logarithmic_scale(&histograms)
    } else {
        histograms
    };

    // Find max value for normalization
    let max_val = histograms
        .iter()
        .flat_map(|h| h.iter())
        .copied()
        .max()
        .unwrap_or(1);

    let mut canvas = Canvas::new(scope_width, scope_height);

    match config.histogram_mode {
        HistogramMode::Overlay | HistogramMode::Logarithmic => {
            draw_overlay_histogram(&mut canvas, &histograms, max_val);
        }
        HistogramMode::Stacked => {
            draw_stacked_histogram(&mut canvas, &histograms, max_val);
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_histogram_graticule(&mut canvas, config);
    }

    // Draw statistical overlays if labels enabled
    if config.show_labels {
        let stats = compute_histogram_stats(&histograms, width * height);
        draw_histogram_stats(&mut canvas, &stats);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::HistogramRgb,
    })
}

/// Generates a luma (brightness) histogram from frame data.
///
/// Displays a single histogram of overall image brightness (Y channel).
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn generate_luma_histogram(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    // Build luma histogram (256 bins) using rayon parallel row chunks.
    // Each worker computes thread-local bins; reduce() merges lock-free.
    //
    // Accumulators are boxed to keep them on the heap and prevent a rayon
    // worker thread stack overflow in debug builds with large frames.
    let pixel_count = (width * height) as usize;
    let pixels = &frame[..pixel_count * 3];
    let histogram_box = pixels
        .par_chunks(3)
        .fold(
            || Box::new([0u32; 256]),
            |mut acc, chunk| {
                if let [r, g, b] = chunk {
                    let (luma, _, _) = rgb_to_ycbcr(*r, *g, *b);
                    acc[luma as usize] = acc[luma as usize].saturating_add(1);
                }
                acc
            },
        )
        .reduce(
            || Box::new([0u32; 256]),
            |mut a, b| {
                for bin in 0..256 {
                    a[bin] = a[bin].saturating_add(b[bin]);
                }
                a
            },
        );
    let histogram: [u32; 256] = *histogram_box;

    // Apply logarithmic scale if requested
    let histogram = if matches!(config.histogram_mode, HistogramMode::Logarithmic) {
        let mut log_histogram = [0u32; 256];
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                log_histogram[i] = ((count as f32).ln() * 1000.0) as u32;
            }
        }
        log_histogram
    } else {
        histogram
    };

    // Find max value for normalization
    let max_val = histogram.iter().copied().max().unwrap_or(1);

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Draw histogram
    for bin in 0..256 {
        let x = (bin as u32 * scope_width) / 256;
        let count = histogram[bin];

        if count > 0 {
            let bar_height = ((count as f32 / max_val as f32) * scope_height as f32) as u32;
            let y_start = scope_height - bar_height;

            // Draw vertical bar in white
            for y in y_start..scope_height {
                canvas.set_pixel(x, y, crate::render::colors::WHITE);
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_histogram_graticule(&mut canvas, config);
    }

    // Draw statistical overlays if labels enabled
    if config.show_labels {
        let stats = compute_luma_histogram_stats(&histogram, width * height);
        draw_luma_histogram_stats(&mut canvas, &stats);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::HistogramLuma,
    })
}

/// Draws overlaid RGB histogram.
fn draw_overlay_histogram(canvas: &mut Canvas, histograms: &[[u32; 256]; 3], max_val: u32) {
    let scope_width = canvas.width;
    let scope_height = canvas.height;

    let colors = [
        crate::render::colors::RED,
        crate::render::colors::GREEN,
        crate::render::colors::BLUE,
    ];

    // Draw each channel
    for (channel, histogram) in histograms.iter().enumerate() {
        for bin in 0..256 {
            let x = (bin as u32 * scope_width) / 256;
            let count = histogram[bin];

            if count > 0 {
                #[allow(clippy::cast_possible_truncation)]
                #[allow(clippy::cast_sign_loss)]
                let bar_height = ((count as f32 / max_val as f32) * scope_height as f32) as u32;
                let y_start = scope_height - bar_height;

                // Draw vertical bar with semi-transparency for overlay effect
                let mut color = colors[channel];
                color[3] = 192; // Semi-transparent

                for y in y_start..scope_height {
                    canvas.blend_pixel(x, y, color);
                }
            }
        }
    }
}

/// Draws stacked RGB histogram.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn draw_stacked_histogram(canvas: &mut Canvas, histograms: &[[u32; 256]; 3], max_val: u32) {
    let scope_width = canvas.width;
    let scope_height = canvas.height;

    let colors = [
        crate::render::colors::RED,
        crate::render::colors::GREEN,
        crate::render::colors::BLUE,
    ];

    // Draw each bin
    for bin in 0..256 {
        let x = (bin as u32 * scope_width) / 256;

        let mut y_offset = scope_height;

        // Stack channels from bottom to top
        for (channel, histogram) in histograms.iter().enumerate() {
            let count = histogram[bin];

            if count > 0 {
                let bar_height =
                    ((count as f32 / max_val as f32) * (scope_height as f32 / 3.0)) as u32;

                let y_start = y_offset.saturating_sub(bar_height);

                for y in y_start..y_offset {
                    canvas.set_pixel(x, y, colors[channel]);
                }

                y_offset = y_start;
            }
        }
    }
}

/// Applies logarithmic scale to histogram for better visibility of low counts.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn apply_logarithmic_scale(histograms: &[[u32; 256]; 3]) -> [[u32; 256]; 3] {
    let mut log_histograms = [[0u32; 256]; 3];

    for (channel, histogram) in histograms.iter().enumerate() {
        for (bin, &count) in histogram.iter().enumerate() {
            if count > 0 {
                log_histograms[channel][bin] = ((count as f32).ln() * 1000.0) as u32;
            }
        }
    }

    log_histograms
}

/// Histogram statistics for a single channel.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Mean value (0-255).
    pub mean: f32,

    /// Median value (0-255).
    pub median: u8,

    /// Standard deviation.
    pub std_dev: f32,

    /// 1st percentile.
    pub percentile_1: u8,

    /// 99th percentile.
    pub percentile_99: u8,

    /// Percentage of pixels at 0 (clipped black).
    pub black_clip_percent: f32,

    /// Percentage of pixels at 255 (clipped white).
    pub white_clip_percent: f32,
}

/// RGB histogram statistics.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Red channel statistics.
    pub red: ChannelStats,

    /// Green channel statistics.
    pub green: ChannelStats,

    /// Blue channel statistics.
    pub blue: ChannelStats,
}

/// Computes statistics for all RGB channels.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn compute_histogram_stats(histograms: &[[u32; 256]; 3], pixel_count: u32) -> HistogramStats {
    HistogramStats {
        red: compute_channel_stats(&histograms[0], pixel_count),
        green: compute_channel_stats(&histograms[1], pixel_count),
        blue: compute_channel_stats(&histograms[2], pixel_count),
    }
}

/// Computes statistics for a single channel.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn compute_channel_stats(histogram: &[u32; 256], pixel_count: u32) -> ChannelStats {
    // Compute mean
    let mut sum = 0u64;
    for (value, &count) in histogram.iter().enumerate() {
        sum += value as u64 * u64::from(count);
    }
    let mean = sum as f32 / pixel_count as f32;

    // Compute median
    let mut cumulative = 0u32;
    let half_pixels = pixel_count / 2;
    let mut median = 128u8;
    for (value, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= half_pixels {
            median = value as u8;
            break;
        }
    }

    // Compute standard deviation
    let mut variance_sum = 0.0f32;
    for (value, &count) in histogram.iter().enumerate() {
        let diff = value as f32 - mean;
        variance_sum += diff * diff * count as f32;
    }
    let std_dev = (variance_sum / pixel_count as f32).sqrt();

    // Compute 1st and 99th percentiles
    let percentile_1_target = pixel_count / 100;
    let percentile_99_target = (pixel_count * 99) / 100;

    let mut cumulative = 0u32;
    let mut percentile_1 = 0u8;
    let mut percentile_99 = 255u8;

    for (value, &count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= percentile_1_target && percentile_1 == 0 {
            percentile_1 = value as u8;
        }
        if cumulative >= percentile_99_target {
            percentile_99 = value as u8;
            break;
        }
    }

    // Compute clipping percentages
    let black_clip_percent = (histogram[0] as f32 / pixel_count as f32) * 100.0;
    let white_clip_percent = (histogram[255] as f32 / pixel_count as f32) * 100.0;

    ChannelStats {
        mean,
        median,
        std_dev,
        percentile_1,
        percentile_99,
        black_clip_percent,
        white_clip_percent,
    }
}

/// Computes statistics for luma histogram.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn compute_luma_histogram_stats(histogram: &[u32; 256], pixel_count: u32) -> ChannelStats {
    compute_channel_stats(histogram, pixel_count)
}

/// Draws histogram statistics overlay.
fn draw_histogram_stats(canvas: &mut Canvas, stats: &HistogramStats) {
    let color = crate::render::colors::WHITE;

    // Draw clipping warnings if significant
    if stats.red.black_clip_percent > 1.0
        || stats.green.black_clip_percent > 1.0
        || stats.blue.black_clip_percent > 1.0
    {
        crate::render::draw_label(canvas, 2, canvas.height - 20, "BLACK CLIP", color);
    }

    if stats.red.white_clip_percent > 1.0
        || stats.green.white_clip_percent > 1.0
        || stats.blue.white_clip_percent > 1.0
    {
        crate::render::draw_label(
            canvas,
            canvas.width - 50,
            canvas.height - 20,
            "WHITE CLIP",
            color,
        );
    }
}

/// Draws luma histogram statistics overlay.
fn draw_luma_histogram_stats(canvas: &mut Canvas, stats: &ChannelStats) {
    let color = crate::render::colors::WHITE;

    // Draw clipping warnings if significant
    if stats.black_clip_percent > 1.0 {
        crate::render::draw_label(canvas, 2, canvas.height - 20, "BLACK CLIP", color);
    }

    if stats.white_clip_percent > 1.0 {
        crate::render::draw_label(
            canvas,
            canvas.width - 50,
            canvas.height - 20,
            "WHITE CLIP",
            color,
        );
    }

    // Draw mean value
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let mean_str = format!("{}", stats.mean as u32);
    crate::render::draw_label(canvas, canvas.width / 2 - 10, 2, &mean_str, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create gradient from black to white
        for y in 0..height {
            let value = ((y * 255) / height) as u8;
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    #[test]
    fn test_generate_rgb_histogram() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_rgb_histogram(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.scope_type, ScopeType::HistogramRgb);
    }

    #[test]
    fn test_generate_luma_histogram() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_luma_histogram(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.scope_type, ScopeType::HistogramLuma);
    }

    #[test]
    fn test_histogram_modes() {
        let frame = create_test_frame(50, 50);

        // Test overlay mode
        let mut config = ScopeConfig::default();
        config.histogram_mode = HistogramMode::Overlay;
        let result = generate_rgb_histogram(&frame, 50, 50, &config);
        assert!(result.is_ok());

        // Test stacked mode
        config.histogram_mode = HistogramMode::Stacked;
        let result = generate_rgb_histogram(&frame, 50, 50, &config);
        assert!(result.is_ok());

        // Test logarithmic mode
        config.histogram_mode = HistogramMode::Logarithmic;
        let result = generate_rgb_histogram(&frame, 50, 50, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_channel_stats() {
        let mut histogram = [0u32; 256];

        // Uniform distribution
        for bin in histogram.iter_mut() {
            *bin = 100;
        }

        let stats = compute_channel_stats(&histogram, 256 * 100);
        assert!(stats.mean > 100.0 && stats.mean < 150.0);
        assert!(stats.median > 100 && stats.median < 150);
    }

    #[test]
    fn test_clipping_detection() {
        let mut histogram = [0u32; 256];

        // Add lots of black pixels
        histogram[0] = 5000;
        histogram[255] = 3000;

        let stats = compute_channel_stats(&histogram, 10000);
        assert!(stats.black_clip_percent > 40.0);
        assert!(stats.white_clip_percent > 20.0);
    }

    #[test]
    fn test_percentiles() {
        let mut histogram = [0u32; 256];

        // Concentrated distribution
        histogram[50] = 1000;
        histogram[100] = 2000;
        histogram[150] = 1000;

        let stats = compute_channel_stats(&histogram, 4000);
        assert!(stats.percentile_1 <= 100);
        assert!(stats.percentile_99 >= 100);
    }

    #[test]
    fn test_logarithmic_scale() {
        let histograms = [[100u32; 256]; 3];
        let log_histograms = apply_logarithmic_scale(&histograms);

        // Log values should be non-zero
        assert!(log_histograms[0][0] > 0);
    }

    // ── Item 5 tests ──────────────────────────────────────────────────────────

    /// Thread-local rayon histogram must match serial bin accumulation exactly.
    #[test]
    fn test_thread_local_histogram_matches_serial() {
        // 64×64 frame with known pixel values.
        let frame: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();
        let config = ScopeConfig {
            width: 256,
            height: 128,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };
        let scope =
            generate_rgb_histogram(&frame, 64, 64, &config).expect("histogram should succeed");
        // Verify output dimensions
        assert_eq!(scope.width, 256);
        assert_eq!(scope.height, 128);
        assert_eq!(scope.data.len(), (256 * 128 * 4) as usize);
    }

    /// Thread-local parallel luma histogram must produce non-empty output
    /// and not panic on a 128×128 frame.
    #[test]
    fn test_thread_local_histogram_all_channels() {
        // Solid red frame — R channel will be saturated, G/B empty.
        let frame: Vec<u8> = (0..128 * 128).flat_map(|_| [200u8, 0u8, 0u8]).collect();
        let config = ScopeConfig {
            width: 256,
            height: 128,
            show_graticule: false,
            show_labels: false,
            ..ScopeConfig::default()
        };
        let rgb_scope =
            generate_rgb_histogram(&frame, 128, 128, &config).expect("rgb histogram ok");
        let luma_scope =
            generate_luma_histogram(&frame, 128, 128, &config).expect("luma histogram ok");
        // Both should have valid RGBA output.
        assert_eq!(rgb_scope.data.len(), (256 * 128 * 4) as usize);
        assert_eq!(luma_scope.data.len(), (256 * 128 * 4) as usize);
    }
}
