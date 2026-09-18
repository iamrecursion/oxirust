//! KCF (Kernelized Correlation Filter) tracker.
//!
//! High-speed tracking with kernels using circulant matrices and FFT.
//! This implementation uses the Gaussian kernel for robust tracking.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::kcf::KcfTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
//! let tracker = KcfTracker::new(bbox);
//! ```

use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};
use std::f64::consts::PI;

/// KCF tracker configuration.
#[derive(Debug, Clone)]
pub struct KcfTracker {
    /// Current bounding box.
    bbox: BoundingBox,
    /// Alpha coefficients (frequency domain).
    alpha: Vec<f64>,
    /// Template size.
    template_size: (usize, usize),
    /// Learning rate.
    learning_rate: f64,
    /// Regularization parameter.
    lambda: f64,
    /// Gaussian kernel bandwidth.
    sigma: f64,
    /// Padding ratio around bbox.
    padding: f64,
    /// Scale adaptation enabled.
    scale_adaptation: bool,
    /// Scale factors to test.
    scale_factors: Vec<f64>,
    /// Current confidence.
    confidence: f64,
    /// Model features (HOG-like).
    model_x: Vec<f64>,
    /// Kernel matrix.
    kernel_matrix: Vec<f64>,
}

impl KcfTracker {
    /// Create a new KCF tracker.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Initial bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::kcf::KcfTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
    /// let tracker = KcfTracker::new(bbox);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox) -> Self {
        Self {
            bbox,
            alpha: Vec::new(),
            template_size: (64, 64),
            learning_rate: 0.02,
            lambda: 0.0001,
            sigma: 0.5,
            padding: 1.5,
            scale_adaptation: true,
            scale_factors: vec![0.95, 1.0, 1.05],
            confidence: 1.0,
            model_x: Vec::new(),
            kernel_matrix: Vec::new(),
        }
    }

    /// Enable or disable scale adaptation.
    #[must_use]
    pub const fn with_scale_adaptation(mut self, enabled: bool) -> Self {
        self.scale_adaptation = enabled;
        self
    }

    /// Set learning rate.
    #[must_use]
    pub const fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set regularization parameter.
    #[must_use]
    pub const fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
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

        // Extract features from initial patch
        let patch = self.get_padded_patch(frame, width, height)?;
        let features = extract_features(&patch, self.template_size);

        // Create Gaussian labels
        let labels = create_gaussian_labels(self.template_size, 2.0);

        // Compute kernel matrix
        let kernel = gaussian_correlation(&features, &features, self.sigma);

        // Solve for alpha coefficients in frequency domain
        self.alpha = train_filter(&kernel, &labels, self.lambda, self.template_size);

        // Store model
        self.model_x = features;
        self.kernel_matrix = kernel;

        Ok(())
    }

    /// Update tracker with a new frame.
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails or dimensions are invalid.
    #[allow(unused_assignments)]
    pub fn update(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<BoundingBox> {
        if self.alpha.is_empty() {
            return Err(CvError::tracking_error("Tracker not initialized"));
        }

        let mut best_response = f64::NEG_INFINITY;
        let mut best_bbox = self.bbox;
        let mut best_scale = 1.0;

        // Test different scales
        let default_scales = vec![1.0];
        let scales = if self.scale_adaptation {
            &self.scale_factors
        } else {
            &default_scales
        };

        for &scale in scales {
            // Scale the bounding box
            let scaled_bbox = BoundingBox::new(
                self.bbox.x,
                self.bbox.y,
                self.bbox.width * scale as f32,
                self.bbox.height * scale as f32,
            );

            // Temporarily update bbox for patch extraction
            let original_bbox = self.bbox;
            self.bbox = scaled_bbox;

            // Extract features at current location
            let patch = self.get_padded_patch(frame, width, height)?;
            let features = extract_features(&patch, self.template_size);

            // Compute kernel with model
            let kernel = gaussian_correlation(&features, &self.model_x, self.sigma);

            // Detect: response = IFFT(alpha * FFT(kernel))
            let response = detect_with_filter(&kernel, &self.alpha, self.template_size);

            // Find peak
            let (peak_y, peak_x, max_response) = find_peak(&response, self.template_size);

            // Restore original bbox
            self.bbox = original_bbox;

            if max_response > best_response {
                best_response = max_response;
                best_scale = scale;

                // Compute displacement
                let (tw, th) = self.template_size;
                let dy = peak_y as f64 - th as f64 / 2.0;
                let dx = peak_x as f64 - tw as f64 / 2.0;

                // Scale displacement by padding factor
                let cell_size = self.bbox.width as f64 * self.padding / tw as f64;
                let actual_dx = dx * cell_size;
                let actual_dy = dy * cell_size;

                best_bbox = BoundingBox::new(
                    self.bbox.x + actual_dx as f32,
                    self.bbox.y + actual_dy as f32,
                    self.bbox.width * best_scale as f32,
                    self.bbox.height * best_scale as f32,
                );
            }
        }

        // Update confidence based on response
        self.confidence = (best_response / 10.0).clamp(0.0, 1.0);

        // Update bounding box
        self.bbox = best_bbox.clamp(width as f32, height as f32);

        // Update model
        if self.confidence > 0.5 {
            let patch = self.get_padded_patch(frame, width, height)?;
            let new_features = extract_features(&patch, self.template_size);
            let new_kernel = gaussian_correlation(&new_features, &new_features, self.sigma);

            // Create labels
            let labels = create_gaussian_labels(self.template_size, 2.0);

            // Train new filter
            let new_alpha = train_filter(&new_kernel, &labels, self.lambda, self.template_size);

            // Update with learning rate
            let lr = self.learning_rate;
            for i in 0..self.alpha.len().min(new_alpha.len()) {
                self.alpha[i] = lr * new_alpha[i] + (1.0 - lr) * self.alpha[i];
            }

            // Update model features
            for i in 0..self.model_x.len().min(new_features.len()) {
                self.model_x[i] = lr * new_features[i] + (1.0 - lr) * self.model_x[i];
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
        self.alpha.clear();
        self.model_x.clear();
        self.confidence = 1.0;
    }

    /// Get padded patch around current bbox.
    fn get_padded_patch(&self, frame: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
        let padded_w = (self.bbox.width * self.padding as f32) as usize;
        let padded_h = (self.bbox.height * self.padding as f32) as usize;

        let cx = self.bbox.x + self.bbox.width / 2.0;
        let cy = self.bbox.y + self.bbox.height / 2.0;

        let x0 = (cx - padded_w as f32 / 2.0).max(0.0) as usize;
        let y0 = (cy - padded_h as f32 / 2.0).max(0.0) as usize;
        let x1 = (cx + padded_w as f32 / 2.0).min(width as f32) as usize;
        let y1 = (cy + padded_h as f32 / 2.0).min(height as f32) as usize;

        if x1 <= x0 || y1 <= y0 {
            return Err(CvError::tracking_error("Invalid padded region"));
        }

        // Extract and resize to template size
        let (tw, th) = self.template_size;
        let mut patch = vec![0.0; tw * th];

        for y in 0..th {
            for x in 0..tw {
                let src_x = x0 + (x * (x1 - x0)) / tw;
                let src_y = y0 + (y * (y1 - y0)) / th;

                if src_x < width as usize && src_y < height as usize {
                    let idx = src_y * width as usize + src_x;
                    if idx < frame.len() {
                        patch[y * tw + x] = frame[idx] as f64;
                    }
                }
            }
        }

        Ok(patch)
    }
}

/// Extract features from patch (simplified HOG-like features).
fn extract_features(patch: &[f64], size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let mut features = vec![0.0; w * h * 3]; // Gray + Grad_X + Grad_Y

    // Copy grayscale
    for i in 0..(w * h) {
        features[i] = patch[i];
    }

    // Compute gradients
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;

            // Gradient in X direction
            let gx = patch[idx + 1] - patch[idx - 1];
            features[w * h + idx] = gx;

            // Gradient in Y direction
            let gy = patch[idx + w] - patch[idx - w];
            features[2 * w * h + idx] = gy;
        }
    }

    // Normalize features
    normalize_features(&mut features);

    // Apply cosine window
    apply_cosine_window(&mut features, size);

    features
}

/// Normalize features to zero mean and unit variance.
fn normalize_features(features: &mut [f64]) {
    let n = features.len() as f64;
    let mean = features.iter().sum::<f64>() / n;
    let variance = features
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n;
    let std = (variance + 1e-5).sqrt();

    for val in features {
        *val = (*val - mean) / std;
    }
}

/// Apply cosine window to features.
fn apply_cosine_window(features: &mut [f64], size: (usize, usize)) {
    let (w, h) = size;
    let num_channels = features.len() / (w * h);

    for ch in 0..num_channels {
        for y in 0..h {
            for x in 0..w {
                let wx = 0.5 * (1.0 - (2.0 * PI * x as f64 / w as f64).cos());
                let wy = 0.5 * (1.0 - (2.0 * PI * y as f64 / h as f64).cos());
                let idx = ch * w * h + y * w + x;
                features[idx] *= wx * wy;
            }
        }
    }
}

/// Create Gaussian labels for training.
fn create_gaussian_labels(size: (usize, usize), sigma: f64) -> Vec<f64> {
    let (w, h) = size;
    let mut labels = vec![0.0; w * h];

    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let sigma2 = sigma * sigma;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            labels[y * w + x] = (-0.5 * (dx * dx + dy * dy) / sigma2).exp();
        }
    }

    labels
}

/// Compute Gaussian correlation kernel.
fn gaussian_correlation(x: &[f64], y: &[f64], sigma: f64) -> Vec<f64> {
    let n = x.len();
    let size = (n as f64).sqrt() as usize;

    // Compute ||x||^2
    let norm_x: f64 = x.iter().map(|&v| v * v).sum();

    // Compute ||y||^2
    let norm_y: f64 = y.iter().map(|&v| v * v).sum();

    // Compute x ⊙ y (element-wise product) via FFT
    let mut correlation = vec![0.0; size * size];

    // Simplified: compute direct correlation
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for dy in 0..size {
                for dx in 0..size {
                    let x_idx = dy * size + dx;
                    let y_idx = ((dy + i) % size) * size + ((dx + j) % size);
                    if x_idx < x.len() && y_idx < y.len() {
                        sum += x[x_idx] * y[y_idx];
                    }
                }
            }
            correlation[i * size + j] = sum;
        }
    }

    // Apply Gaussian kernel: exp(-(||x||^2 + ||y||^2 - 2*corr) / (2*sigma^2))
    let sigma2 = sigma * sigma;
    for val in &mut correlation {
        let dist = norm_x + norm_y - 2.0 * (*val);
        *val = (-dist / (2.0 * sigma2)).exp();
    }

    correlation
}

/// Train filter coefficients.
fn train_filter(kernel: &[f64], labels: &[f64], lambda: f64, size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let n = w * h;

    // Compute FFT of kernel and labels
    let kernel_fft = compute_fft_real(kernel, size);
    let labels_fft = compute_fft_real(labels, size);

    // Solve: alpha = Y / (K + lambda) in frequency domain
    let mut alpha_fft = vec![0.0; 2 * n];

    for i in 0..n {
        let k_real = kernel_fft[2 * i];
        let k_imag = kernel_fft[2 * i + 1];
        let y_real = labels_fft[2 * i];
        let y_imag = labels_fft[2 * i + 1];

        let denom_real = k_real + lambda;
        let denom_imag = k_imag;
        let denom_norm = denom_real * denom_real + denom_imag * denom_imag;

        if denom_norm > 1e-10 {
            alpha_fft[2 * i] = (y_real * denom_real + y_imag * denom_imag) / denom_norm;
            alpha_fft[2 * i + 1] = (y_imag * denom_real - y_real * denom_imag) / denom_norm;
        }
    }

    alpha_fft
}

/// Detect using trained filter.
fn detect_with_filter(kernel: &[f64], alpha: &[f64], size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let n = w * h;

    // Compute FFT of kernel
    let kernel_fft = compute_fft_real(kernel, size);

    // Multiply: response = alpha * kernel in frequency domain
    let mut response_fft = vec![0.0; 2 * n];

    for i in 0..n.min(alpha.len() / 2) {
        let a_real = alpha[2 * i];
        let a_imag = alpha[2 * i + 1];
        let k_real = kernel_fft[2 * i];
        let k_imag = kernel_fft[2 * i + 1];

        response_fft[2 * i] = a_real * k_real - a_imag * k_imag;
        response_fft[2 * i + 1] = a_real * k_imag + a_imag * k_real;
    }

    // Compute inverse FFT
    compute_ifft_real(&response_fft, size)
}

/// Compute FFT of real signal.
fn compute_fft_real(data: &[f64], size: (usize, usize)) -> Vec<f64> {
    let (w, h) = size;
    let mut result = vec![0.0; 2 * w * h];

    for v in 0..h {
        for u in 0..w {
            let mut real = 0.0;
            let mut imag = 0.0;

            for y in 0..h {
                for x in 0..w {
                    let angle = -2.0
                        * PI
                        * (u as f64 * x as f64 / w as f64 + v as f64 * y as f64 / h as f64);
                    if y * w + x < data.len() {
                        real += data[y * w + x] * angle.cos();
                        imag += data[y * w + x] * angle.sin();
                    }
                }
            }

            result[v * w * 2 + u * 2] = real;
            result[v * w * 2 + u * 2 + 1] = imag;
        }
    }

    result
}

/// Compute inverse FFT to real signal.
fn compute_ifft_real(data: &[f64], size: (usize, usize)) -> Vec<f64> {
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
                    let idx = v * w * 2 + u * 2;
                    if idx + 1 < data.len() {
                        let real = data[idx];
                        let imag = data[idx + 1];
                        sum += real * angle.cos() - imag * angle.sin();
                    }
                }
            }

            result[y * w + x] = sum / n;
        }
    }

    result
}

/// Find peak in response map.
fn find_peak(response: &[f64], size: (usize, usize)) -> (usize, usize, f64) {
    let (w, _h) = size;
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

    (peak_y, peak_x, max_val)
}
