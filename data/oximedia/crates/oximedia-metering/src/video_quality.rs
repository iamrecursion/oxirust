//! Video quality metrics.
//!
//! Implements objective quality metrics including:
//! - PSNR (Peak Signal-to-Noise Ratio)
//! - SSIM (Structural Similarity Index)
//! - MSE (Mean Squared Error)
//! - Blockiness detection

use crate::{MeteringError, MeteringResult};

/// A 2D frame of f64 values stored as a flat Vec with explicit dimensions.
#[derive(Debug, Clone)]
pub struct Frame2D {
    /// Pixel data in row-major order.
    pub data: Vec<f64>,
    /// Frame width.
    pub width: usize,
    /// Frame height.
    pub height: usize,
}

impl Frame2D {
    /// Create a frame filled with a constant value.
    #[must_use]
    pub fn from_elem(height: usize, width: usize, value: f64) -> Self {
        Self {
            data: vec![value; height * width],
            width,
            height,
        }
    }

    /// Create a zero-filled frame.
    #[must_use]
    pub fn zeros(height: usize, width: usize) -> Self {
        Self::from_elem(height, width, 0.0)
    }

    /// Create from a shape function.
    #[must_use]
    pub fn from_shape_fn(height: usize, width: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(height * width);
        for y in 0..height {
            for x in 0..width {
                data.push(f(y, x));
            }
        }
        Self {
            data,
            width,
            height,
        }
    }

    /// Get a reference to the value at (y, x).
    #[inline]
    #[must_use]
    pub fn get(&self, y: usize, x: usize) -> f64 {
        self.data[y * self.width + x]
    }

    /// Get a mutable reference to the value at (y, x).
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, value: f64) {
        self.data[y * self.width + x] = value;
    }

    /// Get dimensions (height, width).
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    /// Iterate over all values.
    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }
}

/// PSNR (Peak Signal-to-Noise Ratio) calculator.
pub struct PsnrCalculator {
    width: usize,
    height: usize,
    max_value: f64,
}

impl PsnrCalculator {
    /// Create a new PSNR calculator.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `max_value` - Maximum possible pixel value (e.g., 1.0 for normalized, 255 for 8-bit)
    pub fn new(width: usize, height: usize, max_value: f64) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            max_value,
        })
    }

    /// Calculate PSNR between reference and distorted frames.
    pub fn calculate(&self, reference: &Frame2D, distorted: &Frame2D) -> MeteringResult<f64> {
        let (ref_h, ref_w) = reference.dim();
        let (dist_h, dist_w) = distorted.dim();

        if ref_w != self.width || ref_h != self.height {
            return Err(MeteringError::InvalidConfig(
                "Reference frame dimensions don't match".to_string(),
            ));
        }

        if dist_w != self.width || dist_h != self.height {
            return Err(MeteringError::InvalidConfig(
                "Distorted frame dimensions don't match".to_string(),
            ));
        }

        let mse = self.calculate_mse(reference, distorted);

        let psnr = if mse > 0.0 {
            20.0 * (self.max_value / mse.sqrt()).log10()
        } else {
            f64::INFINITY
        };

        Ok(psnr)
    }

    /// Calculate MSE (Mean Squared Error).
    pub fn calculate_mse(&self, reference: &Frame2D, distorted: &Frame2D) -> f64 {
        let mut sum = 0.0;

        for (ref_val, dist_val) in reference.iter().zip(distorted.iter()) {
            let diff = ref_val - dist_val;
            sum += diff * diff;
        }

        sum / (self.width * self.height) as f64
    }
}

/// SSIM (Structural Similarity Index) calculator.
pub struct SsimCalculator {
    width: usize,
    height: usize,
    max_value: f64,
    k1: f64,
    k2: f64,
}

impl SsimCalculator {
    /// Create a new SSIM calculator.
    pub fn new(width: usize, height: usize, max_value: f64) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            max_value,
            k1: 0.01,
            k2: 0.03,
        })
    }

    /// Calculate SSIM between reference and distorted frames.
    pub fn calculate(&self, reference: &Frame2D, distorted: &Frame2D) -> MeteringResult<f64> {
        let (ref_h, ref_w) = reference.dim();
        let (dist_h, dist_w) = distorted.dim();

        if ref_w != self.width || ref_h != self.height {
            return Err(MeteringError::InvalidConfig(
                "Reference frame dimensions don't match".to_string(),
            ));
        }

        if dist_w != self.width || dist_h != self.height {
            return Err(MeteringError::InvalidConfig(
                "Distorted frame dimensions don't match".to_string(),
            ));
        }

        let mean_x: f64 = reference.iter().sum::<f64>() / (self.width * self.height) as f64;
        let mean_y: f64 = distorted.iter().sum::<f64>() / (self.width * self.height) as f64;

        let mut var_x = 0.0;
        let mut var_y = 0.0;
        let mut cov_xy = 0.0;

        for (ref_val, dist_val) in reference.iter().zip(distorted.iter()) {
            let diff_x = ref_val - mean_x;
            let diff_y = dist_val - mean_y;

            var_x += diff_x * diff_x;
            var_y += diff_y * diff_y;
            cov_xy += diff_x * diff_y;
        }

        let n = (self.width * self.height) as f64;
        var_x /= n - 1.0;
        var_y /= n - 1.0;
        cov_xy /= n - 1.0;

        let c1 = (self.k1 * self.max_value).powi(2);
        let c2 = (self.k2 * self.max_value).powi(2);

        let numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2);
        let denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2);

        let ssim = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        Ok(ssim.clamp(0.0, 1.0))
    }
}

/// Blockiness detector for compression artifacts.
pub struct BlockinessDetector {
    width: usize,
    height: usize,
    block_size: usize,
}

impl BlockinessDetector {
    /// Create a new blockiness detector.
    pub fn new(width: usize, height: usize, block_size: usize) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        if block_size == 0 {
            return Err(MeteringError::InvalidConfig(
                "Block size must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            block_size,
        })
    }

    /// Detect blockiness in a frame.
    pub fn detect(&self, frame: &Frame2D) -> MeteringResult<f64> {
        let (height, width) = frame.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        let mut horizontal_edges = 0.0;
        let mut vertical_edges = 0.0;
        let mut h_count = 0;
        let mut v_count = 0;

        for y in (self.block_size..height).step_by(self.block_size) {
            for x in 0..width {
                if y > 0 && y < height {
                    let diff = (frame.get(y, x) - frame.get(y - 1, x)).abs();
                    horizontal_edges += diff;
                    h_count += 1;
                }
            }
        }

        for y in 0..height {
            for x in (self.block_size..width).step_by(self.block_size) {
                if x > 0 && x < width {
                    let diff = (frame.get(y, x) - frame.get(y, x - 1)).abs();
                    vertical_edges += diff;
                    v_count += 1;
                }
            }
        }

        let avg_h = if h_count > 0 {
            horizontal_edges / f64::from(h_count)
        } else {
            0.0
        };

        let avg_v = if v_count > 0 {
            vertical_edges / f64::from(v_count)
        } else {
            0.0
        };

        let blockiness = (avg_h + avg_v) / 2.0;

        Ok(blockiness)
    }
}

/// Video quality analyzer combining multiple metrics.
pub struct QualityAnalyzer {
    psnr_calc: PsnrCalculator,
    ssim_calc: SsimCalculator,
    blockiness_detector: BlockinessDetector,
}

impl QualityAnalyzer {
    /// Create a new quality analyzer.
    pub fn new(width: usize, height: usize, max_value: f64) -> MeteringResult<Self> {
        let psnr_calc = PsnrCalculator::new(width, height, max_value)?;
        let ssim_calc = SsimCalculator::new(width, height, max_value)?;
        let blockiness_detector = BlockinessDetector::new(width, height, 8)?;

        Ok(Self {
            psnr_calc,
            ssim_calc,
            blockiness_detector,
        })
    }

    /// Analyze quality metrics.
    pub fn analyze(
        &self,
        reference: &Frame2D,
        distorted: &Frame2D,
    ) -> MeteringResult<QualityMetrics> {
        let psnr = self.psnr_calc.calculate(reference, distorted)?;
        let ssim = self.ssim_calc.calculate(reference, distorted)?;
        let mse = self.psnr_calc.calculate_mse(reference, distorted);
        let blockiness = self.blockiness_detector.detect(distorted)?;

        Ok(QualityMetrics {
            psnr,
            ssim,
            mse,
            blockiness,
        })
    }
}

/// Quality metrics result.
#[derive(Clone, Debug)]
pub struct QualityMetrics {
    /// PSNR in dB.
    pub psnr: f64,
    /// SSIM (0.0 to 1.0).
    pub ssim: f64,
    /// MSE (Mean Squared Error).
    pub mse: f64,
    /// Blockiness score.
    pub blockiness: f64,
}

impl QualityMetrics {
    /// Check if quality is excellent (PSNR > 40 dB, SSIM > 0.95).
    pub fn is_excellent(&self) -> bool {
        self.psnr > 40.0 && self.ssim > 0.95
    }

    /// Check if quality is good (PSNR > 30 dB, SSIM > 0.90).
    pub fn is_good(&self) -> bool {
        self.psnr > 30.0 && self.ssim > 0.90
    }

    /// Check if quality is poor (PSNR < 25 dB, SSIM < 0.80).
    pub fn is_poor(&self) -> bool {
        self.psnr < 25.0 || self.ssim < 0.80
    }

    /// Get a quality rating string.
    pub fn rating(&self) -> &str {
        if self.is_excellent() {
            "Excellent"
        } else if self.is_good() {
            "Good"
        } else if self.is_poor() {
            "Poor"
        } else {
            "Fair"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical_frames() {
        let calc = PsnrCalculator::new(100, 100, 1.0).expect("calc should be valid");

        let frame1 = Frame2D::from_elem(100, 100, 0.5);
        let frame2 = Frame2D::from_elem(100, 100, 0.5);

        let psnr = calc
            .calculate(&frame1, &frame2)
            .expect("psnr should be valid");

        assert!(psnr.is_infinite()); // Perfect match
    }

    #[test]
    fn test_psnr_different_frames() {
        let calc = PsnrCalculator::new(100, 100, 1.0).expect("calc should be valid");

        let frame1 = Frame2D::from_elem(100, 100, 0.5);
        let frame2 = Frame2D::from_elem(100, 100, 0.6);

        let psnr = calc
            .calculate(&frame1, &frame2)
            .expect("psnr should be valid");

        assert!(psnr.is_finite());
        assert!(psnr > 0.0);
    }

    #[test]
    fn test_mse() {
        let calc = PsnrCalculator::new(100, 100, 1.0).expect("calc should be valid");

        let frame1 = Frame2D::from_elem(100, 100, 0.5);
        let frame2 = Frame2D::from_elem(100, 100, 0.6);

        let mse = calc.calculate_mse(&frame1, &frame2);

        // (0.6 - 0.5)^2 = 0.01 (with floating point tolerance)
        assert!((mse - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_ssim_identical_frames() {
        let calc = SsimCalculator::new(100, 100, 1.0).expect("calc should be valid");

        let frame1 = Frame2D::from_elem(100, 100, 0.5);
        let frame2 = Frame2D::from_elem(100, 100, 0.5);

        let ssim = calc
            .calculate(&frame1, &frame2)
            .expect("ssim should be valid");

        assert!((ssim - 1.0).abs() < 0.01); // Should be very close to 1.0
    }

    #[test]
    fn test_blockiness_detector() {
        let detector = BlockinessDetector::new(64, 64, 8).expect("detector should be valid");

        let frame = Frame2D::from_elem(64, 64, 0.5);

        let blockiness = detector.detect(&frame).expect("blockiness should be valid");

        assert_eq!(blockiness, 0.0); // Uniform frame has no blockiness
    }

    #[test]
    fn test_quality_analyzer() {
        let analyzer = QualityAnalyzer::new(100, 100, 1.0).expect("analyzer should be valid");

        let reference = Frame2D::from_elem(100, 100, 0.5);
        let distorted = Frame2D::from_elem(100, 100, 0.52);

        let metrics = analyzer
            .analyze(&reference, &distorted)
            .expect("metrics should be valid");

        assert!(metrics.psnr.is_finite());
        assert!(metrics.ssim >= 0.0 && metrics.ssim <= 1.0);
    }

    #[test]
    fn test_quality_rating() {
        let excellent = QualityMetrics {
            psnr: 45.0,
            ssim: 0.97,
            mse: 0.0001,
            blockiness: 0.01,
        };

        assert!(excellent.is_excellent());
        assert_eq!(excellent.rating(), "Excellent");

        let poor = QualityMetrics {
            psnr: 20.0,
            ssim: 0.75,
            mse: 0.1,
            blockiness: 0.5,
        };

        assert!(poor.is_poor());
        assert_eq!(poor.rating(), "Poor");
    }
}
