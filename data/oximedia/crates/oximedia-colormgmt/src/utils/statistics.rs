//! Color statistics and histogram utilities.

use crate::utils::image::RgbImage;

/// Color histogram with configurable bin count.
#[derive(Clone, Debug)]
pub struct ColorHistogram {
    /// Red channel histogram
    pub red: Vec<u32>,
    /// Green channel histogram
    pub green: Vec<u32>,
    /// Blue channel histogram
    pub blue: Vec<u32>,
    /// Number of bins per channel
    pub bins: usize,
}

impl ColorHistogram {
    /// Creates a new color histogram with the specified number of bins.
    #[must_use]
    pub fn new(bins: usize) -> Self {
        Self {
            red: vec![0; bins],
            green: vec![0; bins],
            blue: vec![0; bins],
            bins,
        }
    }

    /// Computes histogram from an RGB image.
    #[must_use]
    pub fn from_image(image: &RgbImage, bins: usize) -> Self {
        let mut hist = Self::new(bins);

        for y in 0..image.height {
            for x in 0..image.width {
                if let Some(rgb) = image.get_pixel(x, y) {
                    hist.add_sample(rgb);
                }
            }
        }

        hist
    }

    /// Adds a single RGB sample to the histogram.
    pub fn add_sample(&mut self, rgb: [f64; 3]) {
        let r_bin = (rgb[0].clamp(0.0, 0.999_999) * self.bins as f64) as usize;
        let g_bin = (rgb[1].clamp(0.0, 0.999_999) * self.bins as f64) as usize;
        let b_bin = (rgb[2].clamp(0.0, 0.999_999) * self.bins as f64) as usize;

        self.red[r_bin] += 1;
        self.green[g_bin] += 1;
        self.blue[b_bin] += 1;
    }

    /// Normalizes the histogram (returns probability distribution).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let total_r: u32 = self.red.iter().sum();
        let total_g: u32 = self.green.iter().sum();
        let total_b: u32 = self.blue.iter().sum();

        let mut result = Self::new(self.bins);

        if total_r > 0 {
            for (i, &count) in self.red.iter().enumerate() {
                result.red[i] = ((f64::from(count) / f64::from(total_r)) * 1000.0) as u32;
            }
        }

        if total_g > 0 {
            for (i, &count) in self.green.iter().enumerate() {
                result.green[i] = ((f64::from(count) / f64::from(total_g)) * 1000.0) as u32;
            }
        }

        if total_b > 0 {
            for (i, &count) in self.blue.iter().enumerate() {
                result.blue[i] = ((f64::from(count) / f64::from(total_b)) * 1000.0) as u32;
            }
        }

        result
    }

    /// Computes cumulative distribution function.
    #[must_use]
    pub fn cdf(&self) -> Self {
        let mut result = Self::new(self.bins);

        let mut sum_r = 0;
        let mut sum_g = 0;
        let mut sum_b = 0;

        for i in 0..self.bins {
            sum_r += self.red[i];
            sum_g += self.green[i];
            sum_b += self.blue[i];

            result.red[i] = sum_r;
            result.green[i] = sum_g;
            result.blue[i] = sum_b;
        }

        result
    }
}

/// Color statistics for an image.
#[derive(Clone, Debug)]
pub struct ColorStatistics {
    /// Mean RGB values
    pub mean: [f64; 3],
    /// Standard deviation
    pub std_dev: [f64; 3],
    /// Minimum values
    pub min: [f64; 3],
    /// Maximum values
    pub max: [f64; 3],
    /// Number of pixels analyzed
    pub pixel_count: usize,
}

impl ColorStatistics {
    /// Computes statistics from an RGB image.
    #[must_use]
    pub fn from_image(image: &RgbImage) -> Self {
        let mut sum = [0.0; 3];
        let mut sum_sq = [0.0; 3];
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut count = 0;

        for y in 0..image.height {
            for x in 0..image.width {
                if let Some(rgb) = image.get_pixel(x, y) {
                    for i in 0..3 {
                        sum[i] += rgb[i];
                        sum_sq[i] += rgb[i] * rgb[i];
                        min[i] = min[i].min(rgb[i]);
                        max[i] = max[i].max(rgb[i]);
                    }
                    count += 1;
                }
            }
        }

        let mean = if count > 0 {
            [
                sum[0] / count as f64,
                sum[1] / count as f64,
                sum[2] / count as f64,
            ]
        } else {
            [0.0; 3]
        };

        let variance = if count > 0 {
            [
                ((sum_sq[0] / count as f64) - (mean[0] * mean[0])).max(0.0),
                ((sum_sq[1] / count as f64) - (mean[1] * mean[1])).max(0.0),
                ((sum_sq[2] / count as f64) - (mean[2] * mean[2])).max(0.0),
            ]
        } else {
            [0.0; 3]
        };

        let std_dev = [variance[0].sqrt(), variance[1].sqrt(), variance[2].sqrt()];

        Self {
            mean,
            std_dev,
            min,
            max,
            pixel_count: count,
        }
    }

    /// Computes dynamic range (max / min) for each channel.
    #[must_use]
    pub fn dynamic_range(&self) -> [f64; 3] {
        [
            if self.min[0] > 0.0 {
                self.max[0] / self.min[0]
            } else {
                f64::INFINITY
            },
            if self.min[1] > 0.0 {
                self.max[1] / self.min[1]
            } else {
                f64::INFINITY
            },
            if self.min[2] > 0.0 {
                self.max[2] / self.min[2]
            } else {
                f64::INFINITY
            },
        ]
    }

    /// Computes signal-to-noise ratio estimate.
    #[must_use]
    pub fn snr_estimate(&self) -> [f64; 3] {
        [
            if self.std_dev[0] > 0.0 {
                20.0 * (self.mean[0] / self.std_dev[0]).log10()
            } else {
                f64::INFINITY
            },
            if self.std_dev[1] > 0.0 {
                20.0 * (self.mean[1] / self.std_dev[1]).log10()
            } else {
                f64::INFINITY
            },
            if self.std_dev[2] > 0.0 {
                20.0 * (self.mean[2] / self.std_dev[2]).log10()
            } else {
                f64::INFINITY
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let hist = ColorHistogram::new(256);
        assert_eq!(hist.bins, 256);
        assert_eq!(hist.red.len(), 256);
    }

    #[test]
    fn test_histogram_add_sample() {
        let mut hist = ColorHistogram::new(256);
        hist.add_sample([0.5, 0.5, 0.5]);

        assert!(hist.red[128] > 0);
        assert!(hist.green[128] > 0);
        assert!(hist.blue[128] > 0);
    }

    #[test]
    fn test_statistics_from_solid_image() {
        let mut img = RgbImage::new(10, 10);
        img.fill([0.5, 0.6, 0.7]);

        let stats = ColorStatistics::from_image(&img);

        assert!((stats.mean[0] - 0.5).abs() < 1e-10);
        assert!((stats.mean[1] - 0.6).abs() < 1e-10);
        assert!((stats.mean[2] - 0.7).abs() < 1e-10);

        // Solid image should have very small standard deviation (close to zero)
        assert!(stats.std_dev[0] < 0.01, "R std_dev: {}", stats.std_dev[0]);
        assert!(stats.std_dev[1] < 0.01, "G std_dev: {}", stats.std_dev[1]);
        assert!(stats.std_dev[2] < 0.01, "B std_dev: {}", stats.std_dev[2]);
    }

    #[test]
    fn test_statistics_gradient() {
        let img = RgbImage::gradient(100, 100);
        let stats = ColorStatistics::from_image(&img);

        // Gradient should have mean around 0.5 for red channel
        assert!((stats.mean[0] - 0.5).abs() < 0.1);

        // Should have non-zero standard deviation
        assert!(stats.std_dev[0] > 0.1);
    }

    #[test]
    fn test_cdf() {
        let mut hist = ColorHistogram::new(10);
        for _ in 0..10 {
            hist.add_sample([0.5, 0.5, 0.5]);
        }

        let cdf = hist.cdf();
        // CDF should be monotonically increasing
        for i in 1..10 {
            assert!(cdf.red[i] >= cdf.red[i - 1]);
        }
    }
}
