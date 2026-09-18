//! Phase correlation for image registration.
//!
//! This module implements FFT-based phase correlation for sub-pixel accurate translation estimation,
//! as well as rotation and scale invariant matching using log-polar transforms.

use crate::error::{CvError, CvResult};
use crate::registration::{RegistrationQuality, TransformMatrix};
use std::f64::consts::PI;

/// Complex number representation.
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn div(&self, other: &Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        if denom < f64::EPSILON {
            return Self::new(0.0, 0.0);
        }
        Self::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }
}

/// Window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// No windowing.
    None,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}

/// Phase correlation parameters.
pub struct PhaseCorrelation {
    window_type: WindowType,
    upsampling_factor: usize,
    use_log_polar: bool,
}

impl PhaseCorrelation {
    /// Create a new phase correlation instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            window_type: WindowType::Hann,
            upsampling_factor: 10,
            use_log_polar: false,
        }
    }

    /// Set window function.
    #[must_use]
    pub const fn with_window(mut self, window: WindowType) -> Self {
        self.window_type = window;
        self
    }

    /// Set upsampling factor for sub-pixel accuracy.
    #[must_use]
    pub const fn with_upsampling(mut self, factor: usize) -> Self {
        self.upsampling_factor = factor;
        self
    }

    /// Enable log-polar transform for rotation/scale invariance.
    #[must_use]
    pub const fn with_log_polar(mut self, enabled: bool) -> Self {
        self.use_log_polar = enabled;
        self
    }

    /// Compute phase correlation between two images.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn compute(
        &self,
        reference: &[u8],
        target: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(f64, f64, f64)> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        if !is_power_of_two(width) || !is_power_of_two(height) {
            return Err(CvError::matrix_error(
                "phase correlation requires power-of-two dimensions",
            ));
        }

        // Apply windowing
        let ref_windowed = self.apply_window(reference, width, height);
        let tgt_windowed = self.apply_window(target, width, height);

        // Compute FFT
        let fft_ref = fft_2d(&ref_windowed, width, height)?;
        let fft_tgt = fft_2d(&tgt_windowed, width, height)?;

        // Compute cross-power spectrum
        let cross_power = compute_cross_power_spectrum(&fft_ref, &fft_tgt);

        // Inverse FFT
        let correlation = ifft_2d(&cross_power, width, height)?;

        // Find peak
        let (dx, dy, peak_value) = find_peak(&correlation, width, height);

        // Sub-pixel refinement
        let (dx_refined, dy_refined) =
            self.refine_sub_pixel(&correlation, width, height, dx as usize, dy as usize);

        Ok((dx_refined, dy_refined, peak_value))
    }

    /// Apply window function.
    fn apply_window(&self, image: &[u8], width: u32, height: u32) -> Vec<f64> {
        let size = (width * height) as usize;
        let mut result = vec![0.0; size];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let val = image[idx] as f64 / 255.0;

                let window_val = match self.window_type {
                    WindowType::None => 1.0,
                    WindowType::Hann => {
                        hann_window(x as f64, width as f64) * hann_window(y as f64, height as f64)
                    }
                    WindowType::Hamming => {
                        hamming_window(x as f64, width as f64)
                            * hamming_window(y as f64, height as f64)
                    }
                    WindowType::Blackman => {
                        blackman_window(x as f64, width as f64)
                            * blackman_window(y as f64, height as f64)
                    }
                };

                result[idx] = val * window_val;
            }
        }

        result
    }

    /// Refine peak location to sub-pixel accuracy.
    #[allow(clippy::too_many_arguments)]
    fn refine_sub_pixel(
        &self,
        correlation: &[f64],
        width: u32,
        height: u32,
        peak_x: usize,
        peak_y: usize,
    ) -> (f64, f64) {
        // Parabolic interpolation for sub-pixel accuracy
        let w = width as usize;

        let x = peak_x;
        let y = peak_y;

        if x == 0 || x >= w - 1 || y == 0 || y >= height as usize - 1 {
            return (peak_x as f64, peak_y as f64);
        }

        // X direction
        let c_x = correlation[y * w + x];
        let c_x1 = correlation[y * w + x - 1];
        let c_x2 = correlation[y * w + x + 1];

        let dx = if (c_x2 - c_x1).abs() > f64::EPSILON {
            0.5 * (c_x1 - c_x2) / (c_x1 - 2.0 * c_x + c_x2)
        } else {
            0.0
        };

        // Y direction
        let c_y1 = correlation[(y - 1) * w + x];
        let c_y2 = correlation[(y + 1) * w + x];

        let dy = if (c_y2 - c_y1).abs() > f64::EPSILON {
            0.5 * (c_y1 - c_y2) / (c_y1 - 2.0 * c_x + c_y2)
        } else {
            0.0
        };

        let refined_x = peak_x as f64 + dx;
        let refined_y = peak_y as f64 + dy;

        // Handle wraparound for negative shifts
        let final_x = if refined_x > width as f64 / 2.0 {
            refined_x - width as f64
        } else {
            refined_x
        };

        let final_y = if refined_y > height as f64 / 2.0 {
            refined_y - height as f64
        } else {
            refined_y
        };

        (final_x, final_y)
    }
}

impl Default for PhaseCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

/// Register images using phase correlation.
///
/// # Errors
///
/// Returns an error if registration fails.
pub fn register_phase_correlation(
    reference: &[u8],
    target: &[u8],
    width: u32,
    height: u32,
) -> CvResult<(TransformMatrix, RegistrationQuality)> {
    // Pad to power of two if necessary
    let (padded_width, padded_height) = next_power_of_two_dims(width, height);

    let ref_padded = pad_image(reference, width, height, padded_width, padded_height);
    let tgt_padded = pad_image(target, width, height, padded_width, padded_height);

    let pc = PhaseCorrelation::new();
    let (dx, dy, peak) = pc.compute(&ref_padded, &tgt_padded, padded_width, padded_height)?;

    let transform = TransformMatrix::translation(-dx, -dy);

    let quality = RegistrationQuality {
        success: peak > 0.3,
        rmse: (dx * dx + dy * dy).sqrt(),
        inliers: 0,
        confidence: peak.min(1.0),
        iterations: 0,
    };

    Ok((transform, quality))
}

/// Log-polar transform for rotation and scale invariance.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn log_polar_transform(
    image: &[u8],
    width: u32,
    height: u32,
    angles: usize,
    scales: usize,
) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let mut result = vec![0u8; angles * scales];

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let max_radius = (cx.min(cy)).max(1.0);

    for theta_idx in 0..angles {
        let theta = (theta_idx as f64 / angles as f64) * 2.0 * PI;

        for rho_idx in 0..scales {
            // Logarithmic radial sampling
            let log_rho = (rho_idx as f64 / scales as f64) * max_radius.ln();
            let rho = log_rho.exp();

            let x = cx + rho * theta.cos();
            let y = cy + rho * theta.sin();

            let val = bilinear_interpolate(image, width, height, x, y);
            result[theta_idx * scales + rho_idx] = val;
        }
    }

    Ok(result)
}

/// Fourier-Mellin transform for rotation and scale invariant matching.
///
/// # Errors
///
/// Returns an error if estimation fails.
pub fn estimate_rotation_and_scale(
    reference: &[u8],
    target: &[u8],
    width: u32,
    height: u32,
) -> CvResult<(f64, f64)> {
    let angles = 360;
    let scales = 128;

    // Transform to log-polar space
    let ref_lp = log_polar_transform(reference, width, height, angles, scales)?;
    let tgt_lp = log_polar_transform(target, width, height, angles, scales)?;

    // Pad to power of two
    let lp_width = next_power_of_two(angles as u32);
    let lp_height = next_power_of_two(scales as u32);

    let ref_lp_padded = pad_image(&ref_lp, angles as u32, scales as u32, lp_width, lp_height);
    let tgt_lp_padded = pad_image(&tgt_lp, angles as u32, scales as u32, lp_width, lp_height);

    // Phase correlation in log-polar space
    let pc = PhaseCorrelation::new();
    let (d_angle, d_scale, _) = pc.compute(&ref_lp_padded, &tgt_lp_padded, lp_width, lp_height)?;

    // Convert back to rotation and scale
    let rotation = (d_angle / angles as f64) * 2.0 * PI;
    let scale = (d_scale / scales as f64).exp();

    Ok((rotation, scale))
}

// Helper functions

fn is_power_of_two(n: u32) -> bool {
    n.is_power_of_two()
}

fn next_power_of_two(n: u32) -> u32 {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

fn next_power_of_two_dims(width: u32, height: u32) -> (u32, u32) {
    (next_power_of_two(width), next_power_of_two(height))
}

fn pad_image(image: &[u8], width: u32, height: u32, new_width: u32, new_height: u32) -> Vec<u8> {
    let mut result = vec![0u8; (new_width * new_height) as usize];

    for y in 0..height {
        for x in 0..width {
            let src_idx = (y * width + x) as usize;
            let dst_idx = (y * new_width + x) as usize;
            if src_idx < image.len() && dst_idx < result.len() {
                result[dst_idx] = image[src_idx];
            }
        }
    }

    result
}

fn hann_window(n: f64, size: f64) -> f64 {
    0.5 * (1.0 - ((2.0 * PI * n) / (size - 1.0)).cos())
}

fn hamming_window(n: f64, size: f64) -> f64 {
    0.54 - 0.46 * ((2.0 * PI * n) / (size - 1.0)).cos()
}

fn blackman_window(n: f64, size: f64) -> f64 {
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    a0 - a1 * ((2.0 * PI * n) / (size - 1.0)).cos() + a2 * ((4.0 * PI * n) / (size - 1.0)).cos()
}

/// 2D FFT using row-column decomposition.
fn fft_2d(data: &[f64], width: u32, height: u32) -> CvResult<Vec<Complex>> {
    let size = (width * height) as usize;
    let mut result = vec![Complex::new(0.0, 0.0); size];

    // Convert to complex
    for i in 0..size {
        result[i] = Complex::new(data[i], 0.0);
    }

    // FFT rows
    for y in 0..height {
        let row_start = (y * width) as usize;
        let row_end = row_start + width as usize;
        let mut row: Vec<_> = result[row_start..row_end].to_vec();
        fft_1d(&mut row);
        result[row_start..row_end].copy_from_slice(&row);
    }

    // FFT columns
    for x in 0..width {
        let mut col = vec![Complex::new(0.0, 0.0); height as usize];
        for y in 0..height {
            col[y as usize] = result[(y * width + x) as usize];
        }
        fft_1d(&mut col);
        for y in 0..height {
            result[(y * width + x) as usize] = col[y as usize];
        }
    }

    Ok(result)
}

/// 2D inverse FFT.
fn ifft_2d(data: &[Complex], width: u32, height: u32) -> CvResult<Vec<f64>> {
    let size = (width * height) as usize;
    let mut result = data.to_vec();

    // Conjugate
    for val in &mut result {
        *val = val.conj();
    }

    // IFFT rows
    for y in 0..height {
        let row_start = (y * width) as usize;
        let row_end = row_start + width as usize;
        let mut row: Vec<_> = result[row_start..row_end].to_vec();
        fft_1d(&mut row);
        result[row_start..row_end].copy_from_slice(&row);
    }

    // IFFT columns
    for x in 0..width {
        let mut col = vec![Complex::new(0.0, 0.0); height as usize];
        for y in 0..height {
            col[y as usize] = result[(y * width + x) as usize];
        }
        fft_1d(&mut col);
        for y in 0..height {
            result[(y * width + x) as usize] = col[y as usize];
        }
    }

    // Conjugate and normalize
    let norm = 1.0 / size as f64;
    let mut real_result = vec![0.0; size];
    for i in 0..size {
        real_result[i] = result[i].conj().re * norm;
    }

    Ok(real_result)
}

/// 1D FFT using Cooley-Tukey algorithm.
fn fft_1d(data: &mut [Complex]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n - 1 {
        if i < j {
            data.swap(i, j);
        }
        let mut k = n / 2;
        while k <= j {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = -2.0 * PI / len as f64;

        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            let w_step = Complex::new(angle.cos(), angle.sin());

            for j in 0..half_len {
                let t = w.mul(&data[i + j + half_len]);
                let u = data[i + j];

                data[i + j] = Complex::new(u.re + t.re, u.im + t.im);
                data[i + j + half_len] = Complex::new(u.re - t.re, u.im - t.im);

                w = w.mul(&w_step);
            }
        }
        len *= 2;
    }
}

fn compute_cross_power_spectrum(fft1: &[Complex], fft2: &[Complex]) -> Vec<Complex> {
    let mut result = Vec::with_capacity(fft1.len());

    for i in 0..fft1.len() {
        let product = fft1[i].mul(&fft2[i].conj());
        let magnitude = product.magnitude();

        if magnitude > f64::EPSILON {
            result.push(Complex::new(product.re / magnitude, product.im / magnitude));
        } else {
            result.push(Complex::new(0.0, 0.0));
        }
    }

    result
}

fn find_peak(data: &[f64], width: u32, height: u32) -> (i32, i32, f64) {
    let mut max_val = f64::MIN;
    let mut max_x = 0;
    let mut max_y = 0;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if data[idx] > max_val {
                max_val = data[idx];
                max_x = x as i32;
                max_y = y as i32;
            }
        }
    }

    (max_x, max_y, max_val)
}

fn bilinear_interpolate(image: &[u8], width: u32, height: u32, x: f64, y: f64) -> u8 {
    if x < 0.0 || x >= width as f64 - 1.0 || y < 0.0 || y >= height as f64 - 1.0 {
        return 0;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let idx00 = (y0 * width + x0) as usize;
    let idx01 = (y0 * width + x1) as usize;
    let idx10 = (y1 * width + x0) as usize;
    let idx11 = (y1 * width + x1) as usize;

    if idx11 >= image.len() {
        return 0;
    }

    let v00 = image[idx00] as f64;
    let v01 = image[idx01] as f64;
    let v10 = image[idx10] as f64;
    let v11 = image[idx11] as f64;

    let v0 = v00 * (1.0 - fx) + v01 * fx;
    let v1 = v10 * (1.0 - fx) + v11 * fx;
    let v = v0 * (1.0 - fy) + v1 * fy;

    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let c = a.mul(&b);
        assert!((c.re - (-5.0)).abs() < 1e-6);
        assert!((c.im - 10.0).abs() < 1e-6);

        let mag = a.magnitude();
        assert!((mag - (5.0_f64).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(128));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(100));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(128), 128);
    }

    #[test]
    fn test_hann_window() {
        let w0 = hann_window(0.0, 100.0);
        let w50 = hann_window(50.0, 100.0);
        assert!(w0 < w50);
        assert!(w0 >= 0.0 && w0 <= 1.0);
        assert!(w50 >= 0.0 && w50 <= 1.0);
    }

    #[test]
    fn test_phase_correlation_new() {
        let pc = PhaseCorrelation::new();
        assert_eq!(pc.window_type, WindowType::Hann);
        assert_eq!(pc.upsampling_factor, 10);
    }

    #[test]
    fn test_pad_image() {
        let img = vec![1u8; 16];
        let padded = pad_image(&img, 4, 4, 8, 8);
        assert_eq!(padded.len(), 64);
        assert_eq!(padded[0], 1);
    }

    #[test]
    fn test_fft_1d() {
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        fft_1d(&mut data);
        assert!((data[0].re - 4.0).abs() < 1e-6);
        assert!(data[0].im.abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_interpolate() {
        let img = vec![0u8, 100, 0, 100, 200, 100, 0, 100, 0];
        let val = bilinear_interpolate(&img, 3, 3, 1.5, 1.5);
        assert!(val > 0 && val < 255);
    }
}
