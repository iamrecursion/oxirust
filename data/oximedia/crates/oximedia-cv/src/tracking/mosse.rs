//! MOSSE (Minimum Output Sum of Squared Error) tracker.
//!
//! MOSSE is a fast correlation filter-based tracker that uses simple
//! frequency domain operations for robust tracking.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::mosse::MosseTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
//! let tracker = MosseTracker::new(bbox);
//! ```

use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};
use std::f64::consts::PI;

/// MOSSE tracker configuration.
#[derive(Debug, Clone)]
pub struct MosseTracker {
    /// Current bounding box.
    bbox: BoundingBox,
    /// Filter numerator (frequency domain).
    filter_num: Vec<f64>,
    /// Filter denominator (frequency domain).
    filter_den: Vec<f64>,
    /// Template size.
    template_size: (usize, usize),
    /// Learning rate.
    learning_rate: f64,
    /// PSR threshold for quality.
    psr_threshold: f64,
    /// Current confidence.
    confidence: f64,
    /// Gaussian response.
    gaussian_response: Vec<f64>,
}

impl MosseTracker {
    /// Create a new MOSSE tracker.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Initial bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::mosse::MosseTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
    /// let tracker = MosseTracker::new(bbox);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox) -> Self {
        let template_size = (64, 64);
        let gaussian_response = create_gaussian_response(template_size, 2.0);

        Self {
            bbox,
            filter_num: Vec::new(),
            filter_den: Vec::new(),
            template_size,
            learning_rate: 0.125,
            psr_threshold: 8.0,
            confidence: 1.0,
            gaussian_response,
        }
    }

    /// Set learning rate.
    #[must_use]
    pub const fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set PSR threshold.
    #[must_use]
    pub const fn with_psr_threshold(mut self, threshold: f64) -> Self {
        self.psr_threshold = threshold;
        self
    }

    /// Initialize the tracker with the first frame.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions are invalid.
    pub fn initialize(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<()> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        // Extract and preprocess initial patch
        let patch = extract_and_preprocess(frame, width, height, &self.bbox, self.template_size)?;

        // Compute FFT of patch
        let patch_fft = compute_fft(&patch, self.template_size);

        // Compute FFT of Gaussian response
        let response_fft = compute_fft(&self.gaussian_response, self.template_size);

        // Initialize filter: H = F* ⊙ G / (F* ⊙ F + ε)
        self.filter_num = vec![0.0; patch_fft.len()];
        self.filter_den = vec![0.0; patch_fft.len()];

        for i in 0..(patch_fft.len() / 2) {
            let f_real = patch_fft[2 * i];
            let f_imag = patch_fft[2 * i + 1];
            let g_real = response_fft[2 * i];
            let g_imag = response_fft[2 * i + 1];

            // F* ⊙ G (conjugate of F times G)
            self.filter_num[2 * i] = f_real * g_real + f_imag * g_imag;
            self.filter_num[2 * i + 1] = f_real * g_imag - f_imag * g_real;

            // F* ⊙ F
            self.filter_den[2 * i] = f_real * f_real + f_imag * f_imag;
            self.filter_den[2 * i + 1] = 0.0;
        }

        Ok(())
    }

    /// Update tracker with a new frame.
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails or dimensions are invalid.
    pub fn update(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<BoundingBox> {
        if self.filter_num.is_empty() {
            return Err(CvError::tracking_error("Tracker not initialized"));
        }

        // Extract patch at current location
        let patch = extract_and_preprocess(frame, width, height, &self.bbox, self.template_size)?;

        // Compute FFT of patch
        let patch_fft = compute_fft(&patch, self.template_size);

        // Correlate with filter: G = H ⊙ F
        let mut response_fft = vec![0.0; patch_fft.len()];

        for i in 0..(patch_fft.len() / 2) {
            let h_num_real = self.filter_num[2 * i];
            let h_num_imag = self.filter_num[2 * i + 1];
            let h_den = self.filter_den[2 * i] + 0.01; // Add regularization

            let f_real = patch_fft[2 * i];
            let f_imag = patch_fft[2 * i + 1];

            // H ⊙ F
            let h_real = h_num_real / h_den;
            let h_imag = h_num_imag / h_den;

            response_fft[2 * i] = h_real * f_real - h_imag * f_imag;
            response_fft[2 * i + 1] = h_real * f_imag + h_imag * f_real;
        }

        // Compute inverse FFT to get response
        let response = compute_ifft(&response_fft, self.template_size);

        // Find peak
        let (peak_y, peak_x, psr) = find_peak_with_psr(&response, self.template_size);

        // Update confidence
        self.confidence = (psr / 20.0).clamp(0.0, 1.0);

        // Compute displacement
        let (tw, th) = self.template_size;
        let dy = peak_y as f64 - th as f64 / 2.0;
        let dx = peak_x as f64 - tw as f64 / 2.0;

        // Update bounding box
        self.bbox.x += dx as f32;
        self.bbox.y += dy as f32;

        // Clamp to image bounds
        self.bbox = self.bbox.clamp(width as f32, height as f32);

        // Update filter if PSR is good
        if psr > self.psr_threshold {
            let new_patch =
                extract_and_preprocess(frame, width, height, &self.bbox, self.template_size)?;
            let new_fft = compute_fft(&new_patch, self.template_size);
            let response_fft = compute_fft(&self.gaussian_response, self.template_size);

            // Update filter with learning rate
            let lr = self.learning_rate;
            for i in 0..(new_fft.len() / 2) {
                let f_real = new_fft[2 * i];
                let f_imag = new_fft[2 * i + 1];
                let g_real = response_fft[2 * i];
                let g_imag = response_fft[2 * i + 1];

                // Update numerator
                let new_num_real = f_real * g_real + f_imag * g_imag;
                let new_num_imag = f_real * g_imag - f_imag * g_real;

                self.filter_num[2 * i] = lr * new_num_real + (1.0 - lr) * self.filter_num[2 * i];
                self.filter_num[2 * i + 1] =
                    lr * new_num_imag + (1.0 - lr) * self.filter_num[2 * i + 1];

                // Update denominator
                let new_den = f_real * f_real + f_imag * f_imag;
                self.filter_den[2 * i] = lr * new_den + (1.0 - lr) * self.filter_den[2 * i];
            }
        }

        Ok(self.bbox)
    }

    /// Get current bounding box.
    #[must_use]
    pub const fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Get current confidence.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Reset tracker with new bounding box.
    pub fn reset(&mut self, bbox: BoundingBox) {
        self.bbox = bbox;
        self.filter_num.clear();
        self.filter_den.clear();
        self.confidence = 1.0;
    }
}

/// Extract and preprocess image patch.
fn extract_and_preprocess(
    frame: &[u8],
    width: u32,
    height: u32,
    bbox: &BoundingBox,
    size: (usize, usize),
) -> CvResult<Vec<f64>> {
    let (tw, th) = size;

    // Clamp bbox to image bounds
    let clamped = bbox.clamp(width as f32, height as f32);

    let x0 = clamped.x.max(0.0) as usize;
    let y0 = clamped.y.max(0.0) as usize;
    let x1 = (clamped.x + clamped.width).min(width as f32) as usize;
    let y1 = (clamped.y + clamped.height).min(height as f32) as usize;

    if x1 <= x0 || y1 <= y0 {
        return Err(CvError::tracking_error("Invalid bounding box"));
    }

    let mut patch = vec![0.0; tw * th];

    // Bilinear interpolation for resizing
    for y in 0..th {
        for x in 0..tw {
            let src_x = x0 as f64 + (x as f64 + 0.5) * (x1 - x0) as f64 / tw as f64;
            let src_y = y0 as f64 + (y as f64 + 0.5) * (y1 - y0) as f64 / th as f64;

            let x_floor = src_x.floor() as usize;
            let y_floor = src_y.floor() as usize;
            let x_frac = src_x - x_floor as f64;
            let y_frac = src_y - y_floor as f64;

            if x_floor + 1 < width as usize && y_floor + 1 < height as usize {
                let idx00 = y_floor * width as usize + x_floor;
                let idx01 = y_floor * width as usize + x_floor + 1;
                let idx10 = (y_floor + 1) * width as usize + x_floor;
                let idx11 = (y_floor + 1) * width as usize + x_floor + 1;

                if idx11 < frame.len() {
                    let v00 = frame[idx00] as f64;
                    let v01 = frame[idx01] as f64;
                    let v10 = frame[idx10] as f64;
                    let v11 = frame[idx11] as f64;

                    let v0 = v00 * (1.0 - x_frac) + v01 * x_frac;
                    let v1 = v10 * (1.0 - x_frac) + v11 * x_frac;
                    patch[y * tw + x] = v0 * (1.0 - y_frac) + v1 * y_frac;
                }
            }
        }
    }

    // Apply preprocessing: log transform and normalize
    preprocess_patch(&mut patch);

    // Apply cosine window to reduce boundary effects
    apply_cosine_window(&mut patch, size);

    Ok(patch)
}

/// Preprocess patch with log transform and normalization.
fn preprocess_patch(patch: &mut [f64]) {
    // Log transform
    for val in patch.iter_mut() {
        *val = (*val + 1.0).ln();
    }

    // Normalize to zero mean and unit variance
    let n = patch.len() as f64;
    let mean = patch.iter().sum::<f64>() / n;
    let variance = patch.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std = (variance + 1e-5).sqrt();

    for val in patch.iter_mut() {
        *val = (*val - mean) / std;
    }
}

/// Apply cosine window to patch.
fn apply_cosine_window(patch: &mut [f64], size: (usize, usize)) {
    let (w, h) = size;

    for y in 0..h {
        for x in 0..w {
            let wx = 0.5 * (1.0 - (2.0 * PI * x as f64 / w as f64).cos());
            let wy = 0.5 * (1.0 - (2.0 * PI * y as f64 / h as f64).cos());
            patch[y * w + x] *= wx * wy;
        }
    }
}

/// Create Gaussian response for training.
fn create_gaussian_response(size: (usize, usize), sigma: f64) -> Vec<f64> {
    let (w, h) = size;
    let mut response = vec![0.0; w * h];

    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let sigma2 = sigma * sigma;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            response[y * w + x] = (-0.5 * (dx * dx + dy * dy) / sigma2).exp();
        }
    }

    response
}

/// Compute 2D FFT (simplified DFT for now).
fn compute_fft(data: &[f64], size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let mut result = vec![0.0; 2 * w * h]; // Complex numbers: [real, imag, real, imag, ...]

    for v in 0..h {
        for u in 0..w {
            let mut real = 0.0;
            let mut imag = 0.0;

            for y in 0..h {
                for x in 0..w {
                    let angle = -2.0
                        * PI
                        * (u as f64 * x as f64 / w as f64 + v as f64 * y as f64 / h as f64);
                    real += data[y * w + x] * angle.cos();
                    imag += data[y * w + x] * angle.sin();
                }
            }

            result[v * w * 2 + u * 2] = real;
            result[v * w * 2 + u * 2 + 1] = imag;
        }
    }

    result
}

/// Compute 2D inverse FFT.
fn compute_ifft(data: &[f64], size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let mut result = vec![0.0; w * h];
    let n = (w * h) as f64;

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;

            for v in 0..h {
                for u in 0..w {
                    let angle = 2.0
                        * PI
                        * (u as f64 * x as f64 / w as f64 + v as f64 * y as f64 / h as f64);
                    let real = data[v * w * 2 + u * 2];
                    let imag = data[v * w * 2 + u * 2 + 1];
                    sum += real * angle.cos() - imag * angle.sin();
                }
            }

            result[y * w + x] = sum / n;
        }
    }

    result
}

/// Find peak in response and compute PSR.
fn find_peak_with_psr(response: &[f64], size: (usize, usize)) -> (usize, usize, f64) {
    let (w, h) = size;

    // Find maximum
    let mut max_idx = 0;
    let mut max_val = f64::NEG_INFINITY;

    for (i, &val) in response.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    let peak_x = max_idx % w;
    let peak_y = max_idx / w;

    // Compute PSR (Peak-to-Sidelobe Ratio)
    let mut sidelobe_sum = 0.0;
    let mut sidelobe_sq_sum = 0.0;
    let mut sidelobe_count = 0.0;

    for y in 0..h {
        for x in 0..w {
            let dx = (x as i32 - peak_x as i32).abs();
            let dy = (y as i32 - peak_y as i32).abs();

            // Exclude 11x11 window around peak
            if dx > 5 || dy > 5 {
                let val = response[y * w + x];
                sidelobe_sum += val;
                sidelobe_sq_sum += val * val;
                sidelobe_count += 1.0;
            }
        }
    }

    let psr = if sidelobe_count > 0.0 {
        let mean = sidelobe_sum / sidelobe_count;
        let variance = (sidelobe_sq_sum / sidelobe_count) - (mean * mean);
        let std = (variance + 1e-10).sqrt();
        (max_val - mean) / std
    } else {
        0.0
    };

    (peak_y, peak_x, psr)
}
