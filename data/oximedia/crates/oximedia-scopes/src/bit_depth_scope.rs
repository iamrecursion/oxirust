#![allow(dead_code)]
//! Bit depth analysis scope for video frames.
//!
//! Analyzes the effective bit depth of video content by examining
//! the distribution of sample values. Useful for detecting whether
//! content is truly 10-bit or has been up-converted from 8-bit,
//! and for identifying banding or quantization artifacts.

/// Number of histogram bins for 8-bit analysis.
const BINS_8BIT: usize = 256;

/// Result of a bit depth analysis.
#[derive(Debug, Clone)]
pub struct BitDepthReport {
    /// Estimated effective bit depth (e.g. 8.0, 9.2, 10.0).
    pub estimated_depth: f64,
    /// Number of unique values found across all channels.
    pub unique_values: usize,
    /// Total number of samples analyzed.
    pub total_samples: usize,
    /// Per-channel unique value counts (R, G, B).
    pub channel_unique: [usize; 3],
    /// Whether the content appears to be up-converted from a lower bit depth.
    pub likely_upconverted: bool,
    /// Ratio of populated bins to total possible bins (0.0..=1.0).
    pub bin_utilization: f64,
}

/// Configuration for bit depth analysis.
#[derive(Debug, Clone)]
pub struct BitDepthConfig {
    /// The nominal bit depth of the source (8 or 10).
    pub nominal_depth: u8,
    /// Threshold for up-conversion detection (fraction of bins that must be empty).
    pub upconvert_threshold: f64,
}

impl Default for BitDepthConfig {
    fn default() -> Self {
        Self {
            nominal_depth: 8,
            upconvert_threshold: 0.4,
        }
    }
}

/// Analyzes the effective bit depth of an RGB24 video frame.
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
/// A `BitDepthReport` with the analysis results, or `None` if the frame is invalid.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyze_bit_depth(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &BitDepthConfig,
) -> Option<BitDepthReport> {
    let expected_len = (width as usize) * (height as usize) * 3;
    if frame.len() < expected_len || width == 0 || height == 0 {
        return None;
    }

    let mut histograms: [Vec<u32>; 3] = [
        vec![0u32; BINS_8BIT],
        vec![0u32; BINS_8BIT],
        vec![0u32; BINS_8BIT],
    ];

    let pixel_count = (width as usize) * (height as usize);
    for i in 0..pixel_count {
        let base = i * 3;
        histograms[0][frame[base] as usize] += 1;
        histograms[1][frame[base + 1] as usize] += 1;
        histograms[2][frame[base + 2] as usize] += 1;
    }

    let mut channel_unique = [0usize; 3];
    for (ch, hist) in histograms.iter().enumerate() {
        channel_unique[ch] = hist.iter().filter(|&&c| c > 0).count();
    }

    let unique_values = channel_unique.iter().copied().max().unwrap_or(0);
    let bin_utilization = unique_values as f64 / BINS_8BIT as f64;

    // Estimate effective bit depth from bin utilization
    let estimated_depth = if bin_utilization > 0.0 {
        (unique_values as f64).log2()
    } else {
        0.0
    };

    let likely_upconverted = if config.nominal_depth > 8 {
        // For 10-bit content presented as 8-bit, many bins would be empty
        // because the original values only map to every 4th bin
        bin_utilization < (1.0 - config.upconvert_threshold)
    } else {
        // For 8-bit, check if there's a regular pattern of empty bins
        let empty_count = histograms[0].iter().filter(|&&c| c == 0).count();
        let empty_fraction = empty_count as f64 / BINS_8BIT as f64;
        empty_fraction > config.upconvert_threshold
    };

    Some(BitDepthReport {
        estimated_depth,
        unique_values,
        total_samples: pixel_count * 3,
        channel_unique,
        likely_upconverted,
        bin_utilization,
    })
}

/// Computes the gap pattern in a histogram to detect quantization steps.
///
/// Returns the most common gap size between non-zero bins.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn detect_quantization_step(histogram: &[u32]) -> u32 {
    let mut last_nonzero: Option<usize> = None;
    let mut gap_counts: Vec<u32> = vec![0; 16]; // gaps up to 15

    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            if let Some(last) = last_nonzero {
                let gap = i - last;
                if gap < gap_counts.len() {
                    gap_counts[gap] += 1;
                }
            }
            last_nonzero = Some(i);
        }
    }

    // The most common gap (ignoring gap=1 which is normal)
    gap_counts
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|&(_, &c)| c)
        .map(|(idx, _)| idx as u32)
        .unwrap_or(1)
}

/// Computes the entropy of a histogram, as a proxy for bit depth.
///
/// Higher entropy means more evenly distributed values (more effective bits).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn histogram_entropy(histogram: &[u32]) -> f64 {
    let total: u64 = histogram.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut entropy = 0.0_f64;
    for &count in histogram {
        if count > 0 {
            let p = f64::from(count) / total_f;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Creates a compact histogram from raw RGB24 frame data for a single channel.
///
/// `channel` must be 0 (R), 1 (G), or 2 (B).
#[must_use]
pub fn channel_histogram(frame: &[u8], width: u32, height: u32, channel: usize) -> Vec<u32> {
    let mut hist = vec![0u32; BINS_8BIT];
    if channel > 2 {
        return hist;
    }
    let pixel_count = (width as usize) * (height as usize);
    let expected = pixel_count * 3;
    if frame.len() < expected {
        return hist;
    }
    for i in 0..pixel_count {
        let val = frame[i * 3 + channel] as usize;
        hist[val] += 1;
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width as usize) * (height as usize) * 3]
    }

    fn make_gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut data = vec![0u8; pixel_count * 3];
        for i in 0..pixel_count {
            #[allow(clippy::cast_possible_truncation)]
            let val = (i % 256) as u8;
            data[i * 3] = val;
            data[i * 3 + 1] = val;
            data[i * 3 + 2] = val;
        }
        data
    }

    #[test]
    fn test_uniform_frame() {
        let frame = make_frame(8, 8, 128);
        let config = BitDepthConfig::default();
        let report = analyze_bit_depth(&frame, 8, 8, &config).expect("should succeed in test");
        assert_eq!(report.unique_values, 1);
        assert!(report.estimated_depth < 1.0);
    }

    #[test]
    fn test_gradient_frame() {
        let frame = make_gradient_frame(256, 1);
        let config = BitDepthConfig::default();
        let report = analyze_bit_depth(&frame, 256, 1, &config).expect("should succeed in test");
        assert_eq!(report.unique_values, 256);
        assert!((report.estimated_depth - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_invalid_frame_too_small() {
        let frame = vec![0u8; 10];
        let config = BitDepthConfig::default();
        assert!(analyze_bit_depth(&frame, 100, 100, &config).is_none());
    }

    #[test]
    fn test_zero_dimensions() {
        let config = BitDepthConfig::default();
        assert!(analyze_bit_depth(&[], 0, 0, &config).is_none());
    }

    #[test]
    fn test_bin_utilization_full() {
        let frame = make_gradient_frame(256, 1);
        let config = BitDepthConfig::default();
        let report = analyze_bit_depth(&frame, 256, 1, &config).expect("should succeed in test");
        assert!((report.bin_utilization - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_channel_unique_counts() {
        let frame = make_gradient_frame(256, 1);
        let config = BitDepthConfig::default();
        let report = analyze_bit_depth(&frame, 256, 1, &config).expect("should succeed in test");
        for &u in &report.channel_unique {
            assert_eq!(u, 256);
        }
    }

    #[test]
    fn test_quantization_step_uniform() {
        let mut hist = vec![0u32; 256];
        // Every 4th bin populated (simulates 6-bit content)
        for i in (0..256).step_by(4) {
            hist[i] = 100;
        }
        let step = detect_quantization_step(&hist);
        assert_eq!(step, 4);
    }

    #[test]
    fn test_quantization_step_full() {
        let hist = vec![100u32; 256];
        let step = detect_quantization_step(&hist);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_entropy_uniform() {
        let hist = vec![100u32; 256];
        let e = histogram_entropy(&hist);
        // For uniform distribution over 256 bins, entropy = 8.0
        assert!((e - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_single_value() {
        let mut hist = vec![0u32; 256];
        hist[128] = 1000;
        let e = histogram_entropy(&hist);
        assert!(e.abs() < f64::EPSILON);
    }

    #[test]
    fn test_entropy_empty() {
        let hist = vec![0u32; 256];
        let e = histogram_entropy(&hist);
        assert!(e.abs() < f64::EPSILON);
    }

    #[test]
    fn test_channel_histogram_red() {
        let mut frame = vec![0u8; 4 * 3]; // 4 pixels
        frame[0] = 10; // pixel 0 red
        frame[3] = 20; // pixel 1 red
        frame[6] = 10; // pixel 2 red
        frame[9] = 30; // pixel 3 red
        let hist = channel_histogram(&frame, 4, 1, 0);
        assert_eq!(hist[10], 2);
        assert_eq!(hist[20], 1);
        assert_eq!(hist[30], 1);
    }

    #[test]
    fn test_channel_histogram_invalid_channel() {
        let frame = vec![0u8; 12];
        let hist = channel_histogram(&frame, 2, 2, 5);
        assert_eq!(hist.iter().sum::<u32>(), 0);
    }

    #[test]
    fn test_upconvert_detection() {
        // Create a frame with only every 4th value populated (simulating upconversion)
        let pixel_count = 256;
        let mut frame = vec![0u8; pixel_count * 3];
        for i in 0..pixel_count {
            #[allow(clippy::cast_possible_truncation)]
            let val = ((i % 64) * 4) as u8;
            frame[i * 3] = val;
            frame[i * 3 + 1] = val;
            frame[i * 3 + 2] = val;
        }
        let config = BitDepthConfig {
            nominal_depth: 8,
            upconvert_threshold: 0.4,
        };
        let report = analyze_bit_depth(&frame, 256, 1, &config).expect("should succeed in test");
        assert!(report.likely_upconverted);
    }

    #[test]
    fn test_total_samples() {
        let frame = make_frame(10, 5, 0);
        let config = BitDepthConfig::default();
        let report = analyze_bit_depth(&frame, 10, 5, &config).expect("should succeed in test");
        assert_eq!(report.total_samples, 10 * 5 * 3);
    }
}
