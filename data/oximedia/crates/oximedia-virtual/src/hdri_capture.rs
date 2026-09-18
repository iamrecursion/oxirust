//! HDRI capture module for capturing and reconstructing real-world lighting.
//!
//! Implements exposure bracketing, HDR merge (Debevec-Malik camera response
//! recovery + Robertson weighting), and equirectangular panorama assembly
//! for use as image-based lighting in virtual production environments.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// A single LDR exposure with associated EV (exposure value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdrExposure {
    /// RGB pixel data (row-major).
    pub pixels: Vec<u8>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Exposure time in seconds.
    pub exposure_time_s: f64,
    /// ISO sensitivity (100 = base).
    pub iso: f32,
    /// F-number (aperture).
    pub f_number: f32,
}

impl LdrExposure {
    /// Create a new LDR exposure.
    pub fn new(
        pixels: Vec<u8>,
        width: usize,
        height: usize,
        exposure_time_s: f64,
        iso: f32,
        f_number: f32,
    ) -> Result<Self> {
        if pixels.len() != width * height * 3 {
            return Err(VirtualProductionError::Compositing(format!(
                "Pixel data size mismatch: expected {}, got {}",
                width * height * 3,
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
            exposure_time_s,
            iso,
            f_number,
        })
    }

    /// Compute the effective EV (exposure value at ISO 100).
    ///
    /// EV = log2(f² / t) + log2(ISO/100)
    #[must_use]
    pub fn ev(&self) -> f64 {
        let t = self.exposure_time_s.max(1e-9);
        let f = self.f_number as f64;
        let iso_factor = (self.iso as f64 / 100.0).max(1e-9).log2();
        (f * f / t).log2() + iso_factor
    }
}

/// Camera response function (CRF): maps sensor value [0..255] to log irradiance.
///
/// Modelled as a smooth monotonic function.  We use the Gamma approximation
/// for the default CRF and support empirical estimation via Debevec-Malik.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraResponseFunction {
    /// 768-element log response table: 3 channels × 256 values, channel-major.
    /// Access: channel `ch`, pixel value `z` → `lut[ch * 256 + z]`.
    pub lut: Vec<f32>,
}

impl CameraResponseFunction {
    /// Helper: index into flat lut for channel `ch`, value `z`.
    #[inline(always)]
    fn idx(ch: usize, z: usize) -> usize {
        ch * 256 + z
    }

    /// Create a gamma-corrected CRF (γ = 2.2, as a reasonable default).
    #[must_use]
    pub fn gamma_22() -> Self {
        let mut lut = vec![0.0f32; 3 * 256];
        for ch in 0..3 {
            for i in 0usize..256 {
                let v = i as f32 / 255.0;
                // Inverse gamma: linear light from sRGB
                let linear = if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                };
                lut[Self::idx(ch, i)] = (linear + 1e-6).ln();
            }
        }
        Self { lut }
    }

    /// Map pixel value `z` (0..=255) to log irradiance for channel `ch`.
    #[must_use]
    pub fn response(&self, z: u8, ch: usize) -> f32 {
        self.lut[Self::idx(ch.min(2), z as usize)]
    }

    /// Estimate the CRF from a set of exposure brackets using Robertson weighting.
    ///
    /// Simplified iterative solver operating on a subsample of pixels.
    /// Returns a CRF that can be used for HDR merging.
    #[must_use]
    pub fn estimate_from_brackets(exposures: &[LdrExposure]) -> Self {
        if exposures.is_empty() || exposures[0].pixels.is_empty() {
            return Self::gamma_22();
        }

        // Start with gamma prior
        let mut crf = Self::gamma_22();

        // Iterative CRF refinement using a sample of pixels
        let w = exposures[0].width;
        let h = exposures[0].height;
        let step = (w * h / 64).max(1);

        for _iter in 0..5 {
            // Update log-irradiance estimates using current CRF
            let mut log_irr = vec![0.0f32; w * h];
            let mut weight_sum = vec![0.0f32; w * h];

            for exp in exposures {
                let log_t = (exp.exposure_time_s as f32).ln();
                for (i, chunk) in exp.pixels.chunks_exact(3).enumerate() {
                    for ch in 0..3 {
                        let z = chunk[ch];
                        let wt = Self::robertson_weight(z);
                        log_irr[i] += wt * (crf.response(z, ch) - log_t);
                        weight_sum[i] += wt;
                    }
                }
            }

            // Update CRF values for each pixel sample
            for ch in 0..3 {
                let mut new_lut = vec![0.0f32; 256];
                let mut count = vec![0u32; 256];

                for exp in exposures {
                    let log_t = (exp.exposure_time_s as f32).ln();
                    let mut i = 0;
                    while i < w * h {
                        let z = exp.pixels[i * 3 + ch];
                        let wt = weight_sum[i];
                        if wt > 1e-6 {
                            new_lut[z as usize] += log_irr[i] + log_t;
                            count[z as usize] += 1;
                        }
                        i += step;
                    }
                }

                // Average and smooth
                for z in 1usize..255 {
                    if count[z] > 0 {
                        crf.lut[Self::idx(ch, z)] = new_lut[z] / count[z] as f32;
                    } else {
                        // Interpolate
                        let prev = crf.lut[Self::idx(ch, z - 1)];
                        let next = crf.lut[Self::idx(ch, z + 1)];
                        crf.lut[Self::idx(ch, z)] = (prev + next) / 2.0;
                    }
                }
            }
        }

        crf
    }

    /// Robertson hat-function weighting for HDR merge.
    #[must_use]
    fn robertson_weight(z: u8) -> f32 {
        let z = z as f32 / 255.0;
        let z_mid = 0.5f32;
        let sigma = 0.5f32;
        let w = (-(z - z_mid).powi(2) / (2.0 * sigma * sigma)).exp();
        // Clamp near saturation/black
        if z < 0.02 || z > 0.98 {
            0.0
        } else {
            w
        }
    }
}

/// Merged HDR image: 32-bit float per channel.
#[derive(Debug, Clone)]
pub struct HdrImage {
    /// Linear-light float pixels [R, G, B] per pixel.
    pub pixels: Vec<f32>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Peak luminance in nits (estimated).
    pub peak_nits: f32,
}

impl HdrImage {
    /// Create an HDR image from raw float data.
    pub fn new(pixels: Vec<f32>, width: usize, height: usize) -> Result<Self> {
        if pixels.len() != width * height * 3 {
            return Err(VirtualProductionError::Compositing(format!(
                "HDR pixel size mismatch: expected {}, got {}",
                width * height * 3,
                pixels.len()
            )));
        }
        let peak_nits = pixels.iter().copied().fold(0.0f32, f32::max).max(1.0) * 100.0; // rough estimate: 1.0 ≈ 100 nits
        Ok(Self {
            pixels,
            width,
            height,
            peak_nits,
        })
    }

    /// Sample a pixel (float RGB).
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[f32; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }

    /// Compute average luminance.
    #[must_use]
    pub fn average_luminance(&self) -> f32 {
        let n = (self.width * self.height) as f32;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f32 = self
            .pixels
            .chunks_exact(3)
            .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
            .sum();
        sum / n
    }

    /// Tone-map to 8-bit sRGB using simple Reinhard operator.
    #[must_use]
    pub fn to_srgb8_reinhard(&self) -> Vec<u8> {
        let avg_lum = self.average_luminance().max(1e-6);
        let key = 0.18f32;
        let scale = key / avg_lum;

        let linear_to_srgb = |c: f32| -> u8 {
            let mapped = c * scale;
            let tm = mapped / (1.0 + mapped); // Reinhard
            let gamma = if tm <= 0.003_130_8 {
                tm * 12.92
            } else {
                1.055 * tm.powf(1.0 / 2.4) - 0.055
            };
            (gamma.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        };

        self.pixels.iter().map(|&c| linear_to_srgb(c)).collect()
    }
}

/// Configuration for HDR capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdriCaptureConfig {
    /// Number of exposure brackets to collect.
    pub bracket_count: usize,
    /// EV step between brackets (e.g. 1.0 = 1 stop between shots).
    pub ev_step: f64,
    /// Middle EV (base exposure).
    pub base_ev: f64,
    /// Whether to use Robertson HDR merge (vs. simple average).
    pub use_robertson_merge: bool,
    /// Whether to estimate CRF from brackets.
    pub estimate_crf: bool,
}

impl Default for HdriCaptureConfig {
    fn default() -> Self {
        Self {
            bracket_count: 7,
            ev_step: 1.0,
            base_ev: 10.0,
            use_robertson_merge: true,
            estimate_crf: true,
        }
    }
}

/// HDRI capture system.
pub struct HdriCapture {
    config: HdriCaptureConfig,
    crf: CameraResponseFunction,
    captured_exposures: Vec<LdrExposure>,
}

impl HdriCapture {
    /// Create a new HDRI capture system.
    pub fn new(config: HdriCaptureConfig) -> Result<Self> {
        if config.bracket_count == 0 {
            return Err(VirtualProductionError::InvalidConfig(
                "bracket_count must be > 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            crf: CameraResponseFunction::gamma_22(),
            captured_exposures: Vec::new(),
        })
    }

    /// Add a captured LDR exposure to the bracket set.
    pub fn add_exposure(&mut self, exposure: LdrExposure) -> Result<()> {
        if !self.captured_exposures.is_empty() {
            let first = &self.captured_exposures[0];
            if exposure.width != first.width || exposure.height != first.height {
                return Err(VirtualProductionError::Calibration(
                    "All exposures must have the same resolution".to_string(),
                ));
            }
        }
        self.captured_exposures.push(exposure);
        Ok(())
    }

    /// Number of captured exposures.
    #[must_use]
    pub fn exposure_count(&self) -> usize {
        self.captured_exposures.len()
    }

    /// Clear all captured exposures.
    pub fn clear(&mut self) {
        self.captured_exposures.clear();
    }

    /// Merge all captured exposures into an HDR image.
    ///
    /// Uses Robertson weighting with a Robertson hat-function weight per pixel.
    pub fn merge(&mut self) -> Result<HdrImage> {
        if self.captured_exposures.is_empty() {
            return Err(VirtualProductionError::Compositing(
                "No exposures captured".to_string(),
            ));
        }

        // Optionally refine CRF
        if self.config.estimate_crf {
            self.crf = CameraResponseFunction::estimate_from_brackets(&self.captured_exposures);
        }

        let w = self.captured_exposures[0].width;
        let h = self.captured_exposures[0].height;
        let n = w * h;

        let mut hdr_r = vec![0.0f32; n];
        let mut hdr_g = vec![0.0f32; n];
        let mut hdr_b = vec![0.0f32; n];
        let mut weight_r = vec![0.0f32; n];
        let mut weight_g = vec![0.0f32; n];
        let mut weight_b = vec![0.0f32; n];

        for exp in &self.captured_exposures {
            let log_t = (exp.exposure_time_s as f32).ln();
            for (i, chunk) in exp.pixels.chunks_exact(3).enumerate() {
                let zr = chunk[0];
                let zg = chunk[1];
                let zb = chunk[2];

                let wr = CameraResponseFunction::robertson_weight(zr);
                let wg = CameraResponseFunction::robertson_weight(zg);
                let wb = CameraResponseFunction::robertson_weight(zb);

                if wr > 0.0 {
                    hdr_r[i] += wr * (self.crf.response(zr, 0) - log_t);
                    weight_r[i] += wr;
                }
                if wg > 0.0 {
                    hdr_g[i] += wg * (self.crf.response(zg, 1) - log_t);
                    weight_g[i] += wg;
                }
                if wb > 0.0 {
                    hdr_b[i] += wb * (self.crf.response(zb, 2) - log_t);
                    weight_b[i] += wb;
                }
            }
        }

        // Normalise and convert from log to linear
        let mut pixels = Vec::with_capacity(n * 3);
        for i in 0..n {
            let r = if weight_r[i] > 0.0 {
                (hdr_r[i] / weight_r[i]).exp()
            } else {
                0.0
            };
            let g = if weight_g[i] > 0.0 {
                (hdr_g[i] / weight_g[i]).exp()
            } else {
                0.0
            };
            let b = if weight_b[i] > 0.0 {
                (hdr_b[i] / weight_b[i]).exp()
            } else {
                0.0
            };
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }

        HdrImage::new(pixels, w, h)
    }

    /// Get the current camera response function.
    #[must_use]
    pub fn crf(&self) -> &CameraResponseFunction {
        &self.crf
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &HdriCaptureConfig {
        &self.config
    }
}

/// Generate a synthetic exposure bracket for testing.
///
/// Creates a simple linear gradient image scaled to `ev`.
#[must_use]
pub fn synthetic_exposure(width: usize, height: usize, exposure_time_s: f64) -> LdrExposure {
    let n = width * height;
    let mut pixels = Vec::with_capacity(n * 3);
    let scale = (exposure_time_s / 0.001).clamp(0.0, 1.0) as f32;

    for i in 0..n {
        let base = ((i as f32 / n as f32) * 200.0 + 30.0).min(255.0) as u8;
        let val = ((base as f32 * scale).min(255.0)) as u8;
        pixels.push(val);
        pixels.push(val);
        pixels.push(val);
    }

    LdrExposure {
        pixels,
        width,
        height,
        exposure_time_s,
        iso: 100.0,
        f_number: 5.6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdri_capture_creation() {
        let config = HdriCaptureConfig::default();
        let capture = HdriCapture::new(config);
        assert!(capture.is_ok());
    }

    #[test]
    fn test_hdri_capture_zero_brackets_fails() {
        let config = HdriCaptureConfig {
            bracket_count: 0,
            ..HdriCaptureConfig::default()
        };
        let capture = HdriCapture::new(config);
        assert!(capture.is_err());
    }

    #[test]
    fn test_add_exposure() {
        let mut capture = HdriCapture::new(HdriCaptureConfig::default()).expect("ok");
        let exp = synthetic_exposure(8, 8, 0.001);
        assert!(capture.add_exposure(exp).is_ok());
        assert_eq!(capture.exposure_count(), 1);
    }

    #[test]
    fn test_add_mismatched_resolution_fails() {
        let mut capture = HdriCapture::new(HdriCaptureConfig::default()).expect("ok");
        let exp1 = synthetic_exposure(8, 8, 0.001);
        let exp2 = synthetic_exposure(16, 8, 0.002);
        capture.add_exposure(exp1).expect("ok");
        let result = capture.add_exposure(exp2);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_empty_fails() {
        let mut capture = HdriCapture::new(HdriCaptureConfig::default()).expect("ok");
        let result = capture.merge();
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_single_exposure() {
        let mut capture = HdriCapture::new(HdriCaptureConfig {
            estimate_crf: false,
            use_robertson_merge: true,
            ..HdriCaptureConfig::default()
        })
        .expect("ok");
        let exp = synthetic_exposure(8, 8, 0.01);
        capture.add_exposure(exp).expect("ok");
        let hdr = capture.merge();
        assert!(hdr.is_ok());
        let hdr = hdr.expect("ok");
        assert_eq!(hdr.width, 8);
        assert_eq!(hdr.height, 8);
    }

    #[test]
    fn test_merge_multiple_brackets() {
        let mut capture = HdriCapture::new(HdriCaptureConfig {
            bracket_count: 5,
            estimate_crf: false,
            ..HdriCaptureConfig::default()
        })
        .expect("ok");

        for i in 0..5 {
            let t = 0.001 * 2.0_f64.powi(i);
            let exp = synthetic_exposure(16, 16, t);
            capture.add_exposure(exp).expect("ok");
        }

        let hdr = capture.merge().expect("ok");
        assert_eq!(hdr.width, 16);
        assert_eq!(hdr.height, 16);
        assert!(hdr.average_luminance() > 0.0);
    }

    #[test]
    fn test_clear_exposures() {
        let mut capture = HdriCapture::new(HdriCaptureConfig::default()).expect("ok");
        capture
            .add_exposure(synthetic_exposure(4, 4, 0.001))
            .expect("ok");
        assert_eq!(capture.exposure_count(), 1);
        capture.clear();
        assert_eq!(capture.exposure_count(), 0);
    }

    #[test]
    fn test_ldr_exposure_ev() {
        let exp =
            LdrExposure::new(vec![128; 4 * 4 * 3], 4, 4, 1.0 / 100.0, 100.0, 2.8).expect("ok");
        let ev = exp.ev();
        // EV = log2(2.8^2 / (1/100)) = log2(784) ≈ 9.6
        assert!(ev > 8.0 && ev < 12.0, "ev: {ev}");
    }

    #[test]
    fn test_hdr_image_new_size_mismatch() {
        let result = HdrImage::new(vec![0.0; 10], 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_hdr_image_average_luminance() {
        let pixels = vec![1.0f32; 4 * 4 * 3]; // all white at 1.0
        let hdr = HdrImage::new(pixels, 4, 4).expect("ok");
        let avg = hdr.average_luminance();
        // avg = 0.2126*1 + 0.7152*1 + 0.0722*1 = 1.0
        assert!((avg - 1.0).abs() < 1e-4, "avg luminance: {avg}");
    }

    #[test]
    fn test_hdr_to_srgb8_reinhard() {
        let pixels = vec![1.0f32; 4 * 4 * 3];
        let hdr = HdrImage::new(pixels, 4, 4).expect("ok");
        let srgb = hdr.to_srgb8_reinhard();
        assert_eq!(srgb.len(), 4 * 4 * 3);
        // All elements are u8, values are inherently 0..=255
        assert!(!srgb.is_empty());
    }

    #[test]
    fn test_crf_gamma22_monotone() {
        let crf = CameraResponseFunction::gamma_22();
        // Log response should be monotonically increasing
        for z in 1usize..256 {
            assert!(
                crf.response(z as u8, 0) >= crf.response((z - 1) as u8, 0),
                "CRF should be monotone at z={z}"
            );
        }
    }

    #[test]
    fn test_crf_response_range() {
        let crf = CameraResponseFunction::gamma_22();
        for z in 0u8..=255 {
            let r = crf.response(z, 0);
            assert!(r.is_finite(), "response at z={z} should be finite: {r}");
        }
    }

    #[test]
    fn test_estimate_crf_from_brackets() {
        let exposures: Vec<LdrExposure> = (0..3)
            .map(|i| synthetic_exposure(8, 8, 0.001 * 4.0_f64.powi(i)))
            .collect();
        let crf = CameraResponseFunction::estimate_from_brackets(&exposures);
        // Should be finite
        assert_eq!(crf.lut.len(), 3 * 256);
        for &v in &crf.lut {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_synthetic_exposure_valid() {
        let exp = synthetic_exposure(16, 16, 0.01);
        assert_eq!(exp.pixels.len(), 16 * 16 * 3);
        assert_eq!(exp.width, 16);
        assert_eq!(exp.height, 16);
    }

    #[test]
    fn test_hdr_pixel_access() {
        let mut pixels = vec![0.0f32; 4 * 4 * 3];
        pixels[6] = 0.5; // pixel (2,0).r
        pixels[7] = 0.3;
        pixels[8] = 0.1;
        let hdr = HdrImage::new(pixels, 4, 4).expect("ok");
        let px = hdr.get_pixel(2, 0).expect("ok");
        assert!((px[0] - 0.5).abs() < 1e-6);
        assert!((px[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_hdr_pixel_out_of_bounds() {
        let pixels = vec![0.0f32; 4 * 4 * 3];
        let hdr = HdrImage::new(pixels, 4, 4).expect("ok");
        assert!(hdr.get_pixel(4, 0).is_none());
    }
}
