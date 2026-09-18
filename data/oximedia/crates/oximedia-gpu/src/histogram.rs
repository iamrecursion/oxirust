//! Multi-channel image histogram analysis.
//!
//! Provides per-channel statistics (min, max, mean, std-dev, entropy,
//! percentiles) and whole-image exposure checks.

// ---------------------------------------------------------------------------
// ChannelHistogram
// ---------------------------------------------------------------------------

/// Per-channel histogram data with summary statistics.
#[derive(Debug, Clone)]
pub struct ChannelHistogram {
    /// Raw bin counts; `bins[v]` is the number of pixels with value `v`.
    pub bins: [u32; 256],
    /// Channel index (0-based).
    pub channel: u8,
    /// Minimum pixel value present in this channel.
    pub min_value: u8,
    /// Maximum pixel value present in this channel.
    pub max_value: u8,
    /// Mean pixel value.
    pub mean: f64,
    /// Standard deviation of pixel values.
    pub std_dev: f64,
}

impl ChannelHistogram {
    /// Compute a histogram for one channel of a packed multi-channel image.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw pixel bytes for the whole image.
    /// * `channel` - Zero-based channel index (e.g. 0=R, 1=G, 2=B).
    /// * `stride` - Offset between the start of consecutive rows in bytes
    ///   (use `width * num_channels` for packed images).
    /// * `num_channels` - Total number of interleaved channels per pixel.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn compute(data: &[u8], channel: u8, stride: usize, num_channels: usize) -> Self {
        let ch = channel as usize;
        let mut bins = [0u32; 256];

        if num_channels == 0 || data.is_empty() {
            return Self {
                bins,
                channel,
                min_value: 0,
                max_value: 0,
                mean: 0.0,
                std_dev: 0.0,
            };
        }

        // Iterate over all pixels: stride may differ from num_channels for
        // padded rows, but for packed images stride == width * num_channels.
        // We iterate byte-by-byte and pick channel `ch` from each pixel.
        let total_rows = data.len().checked_div(stride).unwrap_or(0);

        let mut count = 0u64;
        let mut sum = 0u64;

        for row in 0..total_rows {
            let row_start = row * stride;
            let row_end = (row_start + stride).min(data.len());
            let mut byte_idx = row_start + ch;
            while byte_idx < row_end {
                let v = data[byte_idx];
                bins[v as usize] += 1;
                count += 1;
                sum += u64::from(v);
                byte_idx += num_channels;
            }
        }

        // Handle any remaining bytes when data.len() is not a multiple of stride
        // (only relevant when stride == 0, handled above, or non-rectangular).

        let mean = if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        };

        // Variance pass
        let mut sq_sum = 0.0f64;
        for row in 0..total_rows {
            let row_start = row * stride;
            let row_end = (row_start + stride).min(data.len());
            let mut byte_idx = row_start + ch;
            while byte_idx < row_end {
                let v = f64::from(data[byte_idx]);
                let d = v - mean;
                sq_sum += d * d;
                byte_idx += num_channels;
            }
        }
        let std_dev = if count > 1 {
            (sq_sum / count as f64).sqrt()
        } else {
            0.0
        };

        // Min / max from bins
        let mut min_value = 0u8;
        let mut max_value = 0u8;
        let mut found_min = false;
        for (i, &b) in bins.iter().enumerate() {
            if b > 0 {
                if !found_min {
                    min_value = i as u8;
                    found_min = true;
                }
                max_value = i as u8;
            }
        }

        Self {
            bins,
            channel,
            min_value,
            max_value,
            mean,
            std_dev,
        }
    }

    /// Return the pixel value at the given percentile.
    ///
    /// # Arguments
    ///
    /// * `p` - Percentile in `[0.0, 1.0]` (e.g. 0.5 for the median).
    #[must_use]
    pub fn percentile(&self, p: f64) -> u8 {
        let total: u64 = self.bins.iter().map(|&b| u64::from(b)).sum();
        if total == 0 {
            return 0;
        }

        let target = (p.clamp(0.0, 1.0) * total as f64) as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += u64::from(count);
            if cumulative > target {
                return i as u8;
            }
        }
        255
    }

    /// Compute the Shannon entropy of this channel's histogram (in bits).
    ///
    /// Returns `0.0` for a uniform (all-same-value) distribution.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn entropy(&self) -> f64 {
        let total: u64 = self.bins.iter().map(|&b| u64::from(b)).sum();
        if total == 0 {
            return 0.0;
        }

        let total_f = total as f64;
        self.bins
            .iter()
            .filter(|&&b| b > 0)
            .map(|&b| {
                let p = f64::from(b) / total_f;
                -p * p.log2()
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// ImageHistogram
// ---------------------------------------------------------------------------

/// Multi-channel image histogram.
#[derive(Debug, Clone)]
pub struct ImageHistogram {
    /// Per-channel histogram data.
    pub channels: Vec<ChannelHistogram>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl ImageHistogram {
    /// Compute a histogram for a packed RGB or RGBA image.
    ///
    /// The number of channels is inferred from `data.len() / (width * height)`.
    /// For standard packed RGB use 3 channels; for RGBA use 4.
    #[must_use]
    pub fn from_rgb(data: &[u8], width: u32, height: u32) -> Self {
        let pixels = (width * height) as usize;
        let num_channels = data.len().checked_div(pixels).unwrap_or(3);
        let stride = width as usize * num_channels;

        let channels = (0..num_channels as u8)
            .map(|ch| ChannelHistogram::compute(data, ch, stride, num_channels))
            .collect();

        Self {
            channels,
            width,
            height,
        }
    }

    /// Compute a histogram for a single-channel (grayscale) image.
    #[must_use]
    pub fn from_gray(data: &[u8], width: u32, height: u32) -> Self {
        let stride = width as usize;
        let ch = ChannelHistogram::compute(data, 0, stride, 1);

        Self {
            channels: vec![ch],
            width,
            height,
        }
    }

    /// Return `true` if any channel has a mean below 64 (underexposed).
    #[must_use]
    pub fn is_underexposed(&self) -> bool {
        self.channels.iter().any(|ch| ch.mean < 64.0)
    }

    /// Return `true` if any channel has a mean above 192 (overexposed).
    #[must_use]
    pub fn is_overexposed(&self) -> bool {
        self.channels.iter().any(|ch| ch.mean > 192.0)
    }

    /// Return the index of the channel with the highest mean pixel value.
    #[must_use]
    pub fn dominant_channel(&self) -> u8 {
        self.channels
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean
                    .partial_cmp(&b.mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i as u8)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // ChannelHistogram tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_mean_single_channel() {
        // 4 pixels, all value 128.
        let data = vec![128u8; 4];
        let hist = ChannelHistogram::compute(&data, 0, 4, 1);
        assert!((hist.mean - 128.0).abs() < 1e-9);
        assert_eq!(hist.bins[128], 4);
        assert_eq!(hist.min_value, 128);
        assert_eq!(hist.max_value, 128);
    }

    #[test]
    fn test_compute_mean_rgb() {
        // 2 pixels: [255, 0, 0, 255, 0, 0] → red channel mean = 255, green = 0.
        let data = vec![255u8, 0, 0, 255, 0, 0];
        let hist_r = ChannelHistogram::compute(&data, 0, 6, 3);
        let hist_g = ChannelHistogram::compute(&data, 1, 6, 3);
        assert!((hist_r.mean - 255.0).abs() < 1e-9);
        assert!((hist_g.mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_entropy_uniform_image_is_zero() {
        // All pixels the same value → only one bin filled → entropy = 0.
        let data = vec![200u8; 100];
        let hist = ChannelHistogram::compute(&data, 0, 100, 1);
        assert!(
            hist.entropy() < 1e-9,
            "entropy of uniform image should be ~0"
        );
    }

    #[test]
    fn test_entropy_two_equally_likely_values() {
        // 50 pixels at 0 and 50 pixels at 255 → entropy ≈ 1.0 bit.
        let mut data = vec![0u8; 100];
        for b in data.iter_mut().take(50) {
            *b = 255;
        }
        let hist = ChannelHistogram::compute(&data, 0, 100, 1);
        let e = hist.entropy();
        assert!((e - 1.0).abs() < 1e-6, "expected ~1.0 bit entropy, got {e}");
    }

    #[test]
    fn test_percentile_median() {
        // 100 pixels: 50 at 0, 50 at 255.
        let mut data = vec![0u8; 100];
        for i in 50..100 {
            data[i] = 255;
        }
        let hist = ChannelHistogram::compute(&data, 0, 100, 1);
        // 50th percentile should be 0 (cumulative count at 0 reaches target).
        let p50 = hist.percentile(0.5);
        // Exactly half the pixels are 0, so the 50th percentile lands at 0 or 255
        // depending on rounding; just assert it's one of the two values.
        assert!(
            p50 == 0 || p50 == 255,
            "median should be 0 or 255, got {p50}"
        );
    }

    #[test]
    fn test_std_dev_constant_image() {
        let data = vec![100u8; 64];
        let hist = ChannelHistogram::compute(&data, 0, 64, 1);
        assert!(hist.std_dev < 1e-9, "std_dev of constant image should be 0");
    }

    // ------------------------------------------------------------------
    // ImageHistogram tests
    // ------------------------------------------------------------------

    #[test]
    fn test_from_rgb_2x2() {
        // 2×2 image: all pixels (255, 0, 128)
        let data: Vec<u8> = (0..4).flat_map(|_| vec![255u8, 0u8, 128u8]).collect();
        let img = ImageHistogram::from_rgb(&data, 2, 2);

        assert_eq!(img.channels.len(), 3);
        assert!((img.channels[0].mean - 255.0).abs() < 1e-9);
        assert!((img.channels[1].mean - 0.0).abs() < 1e-9);
        assert!((img.channels[2].mean - 128.0).abs() < 1e-9);
    }

    #[test]
    fn test_underexposed_detection() {
        // All pixels at 10 → mean = 10, well below 64.
        let data = vec![10u8; 100];
        let img = ImageHistogram::from_gray(&data, 10, 10);
        assert!(img.is_underexposed());
        assert!(!img.is_overexposed());
    }

    #[test]
    fn test_overexposed_detection() {
        // All pixels at 250 → mean = 250, well above 192.
        let data = vec![250u8; 100];
        let img = ImageHistogram::from_gray(&data, 10, 10);
        assert!(img.is_overexposed());
        assert!(!img.is_underexposed());
    }

    #[test]
    fn test_normal_exposure_neither() {
        // All pixels at 128.
        let data = vec![128u8; 100];
        let img = ImageHistogram::from_gray(&data, 10, 10);
        assert!(!img.is_underexposed());
        assert!(!img.is_overexposed());
    }

    #[test]
    fn test_dominant_channel() {
        // Red=200, Green=50, Blue=100 → dominant = Red (channel 0).
        let data: Vec<u8> = (0..4).flat_map(|_| vec![200u8, 50u8, 100u8]).collect();
        let img = ImageHistogram::from_rgb(&data, 2, 2);
        assert_eq!(img.dominant_channel(), 0);
    }

    #[test]
    fn test_from_gray_single_channel() {
        let data = vec![77u8; 25];
        let img = ImageHistogram::from_gray(&data, 5, 5);
        assert_eq!(img.channels.len(), 1);
        assert!((img.channels[0].mean - 77.0).abs() < 1e-9);
    }

    #[test]
    fn test_underexposed_rgb_one_channel_low() {
        // Red=128, Green=128, Blue=10 → Blue < 64 → underexposed.
        let data: Vec<u8> = (0..4).flat_map(|_| vec![128u8, 128u8, 10u8]).collect();
        let img = ImageHistogram::from_rgb(&data, 2, 2);
        assert!(img.is_underexposed());
    }
}
