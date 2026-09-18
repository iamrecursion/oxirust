//! Video luminance metering.
//!
//! Implements luminance measurement for video signals including:
//! - Peak luminance (nits)
//! - Average luminance
//! - Minimum luminance
//! - Luminance distribution
//! - PQ (HDR10/ST.2084) and HLG (Hybrid Log-Gamma) electro-optical transfer functions

use crate::video_quality::Frame2D;
use crate::{MeteringError, MeteringResult};

/// Transfer function for video luminance encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransferFunction {
    /// Linear (no transfer function, values are already absolute nits).
    Linear,
    /// SMPTE ST 2084 (PQ / HDR10). Signal range 0–1 maps to 0–10 000 cd/m².
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100, BBC/NHK HLG). Signal range 0–1.
    /// Peak luminance is configurable (typically 1000 cd/m² for broadcast).
    Hlg,
    /// ITU-R BT.709 / BT.1886 (SDR). Signal range 0–1 maps to 0–100 cd/m².
    Sdr,
}

impl TransferFunction {
    /// Convert a normalised signal value (0–1) to absolute luminance in cd/m²
    /// (nits) using the specified peak white luminance `peak_nits`.
    ///
    /// # Arguments
    ///
    /// * `signal` - Normalised signal value in [0, 1]
    /// * `peak_nits` - Display peak white luminance in cd/m²
    pub fn to_nits(&self, signal: f64, peak_nits: f64) -> f64 {
        let signal = signal.clamp(0.0, 1.0);
        match self {
            Self::Linear => signal * peak_nits,
            Self::Sdr => {
                // ITU-R BT.1886 gamma 2.4 EOTF, reference white = 100 nits.
                // L = Lw * (E / 1.0)^2.4  (simplified; Lw = peak SDR white = 100 nits)
                signal.powf(2.4) * peak_nits.min(100.0)
            }
            Self::Pq => Self::pq_eotf(signal) * peak_nits.max(1.0),
            Self::Hlg => Self::hlg_eotf(signal, peak_nits),
        }
    }

    /// SMPTE ST 2084 (PQ) EOTF — maps normalised signal to linear scene luminance.
    /// Output is normalised to [0, 1] relative to 10 000 cd/m².
    fn pq_eotf(e: f64) -> f64 {
        // ST 2084 constants
        const M1: f64 = 0.1593_017_578_125;
        const M2: f64 = 78.843_750;
        const C1: f64 = 0.835_937_5;
        const C2: f64 = 18.851_562_5;
        const C3: f64 = 18.687_5;

        let ep = e.powf(1.0 / M2);
        let num = (ep - C1).max(0.0);
        let den = C2 - C3 * ep;
        let linear = if den.abs() > 1e-12 {
            (num / den).powf(1.0 / M1)
        } else {
            0.0
        };
        linear
    }

    /// ITU-R BT.2100 HLG EOTF — maps normalised signal to linear scene luminance
    /// in cd/m², using the system gamma that depends on `peak_nits`.
    ///
    /// Reference: ITU-R BT.2100-2, Table 5.
    fn hlg_eotf(e: f64, peak_nits: f64) -> f64 {
        // HLG constants
        const A: f64 = 0.178_832_77;
        const B: f64 = 0.284_668_66;
        const C: f64 = 0.559_910_73;

        // Compute normalised scene linear value E_s from opto-electronic signal E.
        let e_s = if e <= 0.5 {
            (e * e) / 3.0
        } else {
            ((e - C).exp() / A + B) / 12.0
        };

        // Apply system gamma γ = 1.2 + 0.42 * log10(Lw / 1000).
        // For Lw = 1000 nits, γ = 1.2 (no correction).
        let gamma = 1.2 + 0.42 * (peak_nits / 1000.0).log10();
        let e_d = e_s.powf(gamma);

        // Scale to nits
        e_d * peak_nits
    }
}

/// Convert a frame whose pixel values are encoded as a normalised signal (0–1)
/// into absolute luminance values in nits according to the given transfer function.
///
/// Returns a new `Frame2D` where each pixel value is in cd/m².
pub fn decode_to_nits(frame: &Frame2D, tf: TransferFunction, peak_nits: f64) -> Frame2D {
    Frame2D::from_shape_fn(frame.height, frame.width, |y, x| {
        tf.to_nits(frame.get(y, x), peak_nits)
    })
}

/// Luminance meter for video frames.
pub struct LuminanceMeter {
    width: usize,
    height: usize,
    peak_nits: f64,
    min_nits: f64,
    average_nits: f64,
    histogram: Vec<usize>,
    histogram_bins: usize,
    max_nits: f64,
}

impl LuminanceMeter {
    /// Create a new luminance meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `max_nits` - Maximum expected luminance in nits (e.g., 1000 for HDR, 100 for SDR)
    /// * `histogram_bins` - Number of histogram bins
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        width: usize,
        height: usize,
        max_nits: f64,
        histogram_bins: usize,
    ) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        if max_nits <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Max nits must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            peak_nits: 0.0,
            min_nits: f64::INFINITY,
            average_nits: 0.0,
            histogram: vec![0; histogram_bins],
            histogram_bins,
            max_nits,
        })
    }

    /// Process a luminance frame.
    ///
    /// # Arguments
    ///
    /// * `luminance` - 2D array of luminance values in nits
    ///
    /// # Errors
    ///
    /// Returns error if frame dimensions don't match.
    pub fn process(&mut self, luminance: &Frame2D) -> MeteringResult<()> {
        let (height, width) = luminance.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(format!(
                "Frame dimensions {}x{} don't match expected {}x{}",
                width, height, self.width, self.height
            )));
        }

        // Reset metrics
        self.peak_nits = 0.0;
        self.min_nits = f64::INFINITY;
        let mut sum = 0.0;
        self.histogram.fill(0);

        // Analyze frame
        for &value in luminance.iter() {
            // Update peak and min
            if value > self.peak_nits {
                self.peak_nits = value;
            }
            if value < self.min_nits {
                self.min_nits = value;
            }

            // Update sum for average
            sum += value;

            // Update histogram
            let bin = ((value / self.max_nits) * (self.histogram_bins - 1) as f64)
                .clamp(0.0, (self.histogram_bins - 1) as f64) as usize;
            self.histogram[bin] += 1;
        }

        // Calculate average
        let pixel_count = (self.width * self.height) as f64;
        self.average_nits = sum / pixel_count;

        Ok(())
    }

    /// Get the peak luminance in nits.
    pub fn peak_nits(&self) -> f64 {
        self.peak_nits
    }

    /// Get the minimum luminance in nits.
    pub fn min_nits(&self) -> f64 {
        if self.min_nits.is_infinite() {
            0.0
        } else {
            self.min_nits
        }
    }

    /// Get the average luminance in nits.
    pub fn average_nits(&self) -> f64 {
        self.average_nits
    }

    /// Get the luminance histogram.
    pub fn histogram(&self) -> &[usize] {
        &self.histogram
    }

    /// Get the contrast ratio.
    ///
    /// Contrast ratio = Peak / Min
    pub fn contrast_ratio(&self) -> f64 {
        let min = if self.min_nits.is_infinite() || self.min_nits == 0.0 {
            0.001 // Prevent division by zero
        } else {
            self.min_nits
        };

        self.peak_nits / min
    }

    /// Get the dynamic range in stops.
    ///
    /// Dynamic range (stops) = log2(Peak / Min)
    pub fn dynamic_range_stops(&self) -> f64 {
        self.contrast_ratio().log2()
    }

    /// Check if frame is within SDR range (0-100 nits).
    pub fn is_sdr(&self) -> bool {
        self.peak_nits <= 100.0
    }

    /// Check if frame is HDR10 (up to 1000 nits).
    pub fn is_hdr10(&self) -> bool {
        self.peak_nits > 100.0 && self.peak_nits <= 1000.0
    }

    /// Check if frame is HDR10+ or Dolby Vision (up to 10000 nits).
    pub fn is_extreme_hdr(&self) -> bool {
        self.peak_nits > 1000.0
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_nits = 0.0;
        self.min_nits = f64::INFINITY;
        self.average_nits = 0.0;
        self.histogram.fill(0);
    }
}

/// Black and white level meter for broadcast compliance.
pub struct BlackWhiteLevelMeter {
    width: usize,
    height: usize,
    black_level: f64,
    white_level: f64,
    below_black_count: usize,
    above_white_count: usize,
    black_threshold: f64,
    white_threshold: f64,
}

impl BlackWhiteLevelMeter {
    /// Create a new black/white level meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `black_threshold` - Black level threshold (e.g., 0 for digital, 16/255 for video)
    /// * `white_threshold` - White level threshold (e.g., 1.0 for digital, 235/255 for video)
    pub fn new(
        width: usize,
        height: usize,
        black_threshold: f64,
        white_threshold: f64,
    ) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            black_level: 0.0,
            white_level: 0.0,
            below_black_count: 0,
            above_white_count: 0,
            black_threshold,
            white_threshold,
        })
    }

    /// Process a video frame (normalized 0.0 to 1.0).
    pub fn process(&mut self, frame: &Frame2D) -> MeteringResult<()> {
        let (height, width) = frame.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        self.below_black_count = 0;
        self.above_white_count = 0;
        let mut min_val = f64::INFINITY;
        let mut max_val = 0.0;

        for &value in frame.iter() {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }

            if value < self.black_threshold {
                self.below_black_count += 1;
            }
            if value > self.white_threshold {
                self.above_white_count += 1;
            }
        }

        self.black_level = min_val;
        self.white_level = max_val;

        Ok(())
    }

    /// Get the black level (minimum value).
    pub fn black_level(&self) -> f64 {
        self.black_level
    }

    /// Get the white level (maximum value).
    pub fn white_level(&self) -> f64 {
        self.white_level
    }

    /// Get the number of pixels below black threshold.
    pub fn below_black_count(&self) -> usize {
        self.below_black_count
    }

    /// Get the number of pixels above white threshold.
    pub fn above_white_count(&self) -> usize {
        self.above_white_count
    }

    /// Check if frame is compliant (no pixels outside legal range).
    pub fn is_compliant(&self) -> bool {
        self.below_black_count == 0 && self.above_white_count == 0
    }

    /// Get the percentage of illegal pixels.
    pub fn illegal_pixel_percentage(&self) -> f64 {
        let total_pixels = (self.width * self.height) as f64;
        let illegal_pixels = (self.below_black_count + self.above_white_count) as f64;
        (illegal_pixels / total_pixels) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminance_meter() {
        let mut meter =
            LuminanceMeter::new(1920, 1080, 1000.0, 256).expect("test expectation failed");

        // Create test frame with known values
        let frame = Frame2D::from_shape_fn(1080, 1920, |y, x| {
            (x + y) as f64 / (1920 + 1080) as f64 * 100.0
        });

        meter.process(&frame).expect("process should succeed");

        assert!(meter.peak_nits() > 0.0);
        assert!(meter.average_nits() > 0.0);
        assert!(meter.min_nits() >= 0.0);
    }

    #[test]
    fn test_sdr_detection() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // SDR frame (max 100 nits)
        let frame = Frame2D::from_elem(100, 100, 80.0);
        meter.process(&frame).expect("process should succeed");

        assert!(meter.is_sdr());
        assert!(!meter.is_hdr10());
    }

    #[test]
    fn test_hdr_detection() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // HDR frame (500 nits)
        let frame = Frame2D::from_elem(100, 100, 500.0);
        meter.process(&frame).expect("process should succeed");

        assert!(!meter.is_sdr());
        assert!(meter.is_hdr10());
    }

    #[test]
    fn test_contrast_ratio() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // Frame with known contrast
        let mut frame = Frame2D::zeros(100, 100);
        frame.set(0, 0, 1.0); // Min
        frame.set(99, 99, 100.0); // Max

        meter.process(&frame).expect("process should succeed");

        assert_eq!(meter.peak_nits(), 100.0);
        assert_eq!(meter.min_nits(), 0.0);
    }

    #[test]
    fn test_black_white_level_meter() {
        let mut meter =
            BlackWhiteLevelMeter::new(100, 100, 0.0, 1.0).expect("test expectation failed");

        // Compliant frame
        let frame = Frame2D::from_elem(100, 100, 0.5);
        meter.process(&frame).expect("process should succeed");

        assert!(meter.is_compliant());
        assert_eq!(meter.illegal_pixel_percentage(), 0.0);
    }

    #[test]
    fn test_illegal_pixels() {
        let mut meter =
            BlackWhiteLevelMeter::new(100, 100, 0.0, 1.0).expect("test expectation failed");

        // Frame with illegal pixels
        let mut frame = Frame2D::from_elem(100, 100, 0.5);
        frame.set(0, 0, -0.1); // Below black
        frame.set(99, 99, 1.1); // Above white

        meter.process(&frame).expect("process should succeed");

        assert!(!meter.is_compliant());
        assert!(meter.below_black_count() > 0);
        assert!(meter.above_white_count() > 0);
    }
}
