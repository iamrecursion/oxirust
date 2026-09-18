//! Deconvolution algorithms for motion blur removal.
//!
//! This module provides various deconvolution methods to remove or reduce motion blur.

use super::MotionPSF;
use crate::error::{CvError, CvResult};
use oxifft::{irfft2d, rfft2d, Complex};

/// Deconvolution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeconvolutionMethod {
    /// Wiener deconvolution with noise estimation.
    Wiener,
    /// Richardson-Lucy iterative deconvolution.
    #[default]
    RichardsonLucy,
    /// Fast Fourier Transform-based deconvolution (spatial-domain fallback).
    FFT,
    /// Total Variation regularized deconvolution.
    TotalVariation,
    /// True frequency-domain Wiener deconvolution via 2D FFT.
    ///
    /// Implements the Wiener filter matched to the correlation-based forward blur
    /// used by `MotionPSF::apply_to_channel`:
    ///
    ///   `g[y,x] = Σ_{m,n} h[m,n] · f[y+m-c_h, x+n-c_w]`   (correlation)
    ///
    /// Forward model in DFT: G = conj(H) · F
    ///
    /// Wiener deconvolution: F̂(u,v) = H(u,v) · G(u,v) / (|H(u,v)|² + NSR)
    ///
    /// with power-of-2 zero-padding to avoid circular convolution artifacts.
    FftWiener,
}

/// Wiener deconvolution parameters.
#[derive(Debug, Clone)]
pub struct WienerParams {
    /// Noise-to-signal ratio (NSR).
    pub nsr: f32,
    /// Regularization parameter.
    pub regularization: f32,
}

impl Default for WienerParams {
    fn default() -> Self {
        Self {
            nsr: 0.01,
            regularization: 0.001,
        }
    }
}

impl WienerParams {
    /// Create new Wiener parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set noise-to-signal ratio.
    #[must_use]
    pub const fn with_nsr(mut self, nsr: f32) -> Self {
        self.nsr = nsr;
        self
    }

    /// Set regularization parameter.
    #[must_use]
    pub const fn with_regularization(mut self, reg: f32) -> Self {
        self.regularization = reg;
        self
    }
}

/// PSF zero-padding strategy for the frequency-domain Wiener deconvolution.
///
/// Both variants currently use the same smooth Hann cosine extension in the
/// padded region (see `pad_image_with_smooth_extension`). The `pad` field is
/// preserved for API compatibility and future differentiation; selecting
/// `ReplicateEdge` vs `ZeroPad` has no effect in the current implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PsfPadStrategy {
    /// Smooth Hann extension tapering to zero in the padded zone.
    #[default]
    ZeroPad,
    /// Same as `ZeroPad` (reserved for future edge-replication variant).
    ReplicateEdge,
}

/// Parameters for frequency-domain Wiener deconvolution.
///
/// # Example
///
/// ```
/// use oximedia_cv::motion_blur::{WienerFftParams, PsfPadStrategy};
///
/// let params = WienerFftParams {
///     nsr: 1e-3,
///     pad: PsfPadStrategy::ZeroPad,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct WienerFftParams {
    /// Noise-to-signal ratio.
    ///
    /// Typical value: `1e-3` (30 dB SNR scene).
    /// Higher values suppress more noise at the cost of increased blur.
    pub nsr: f32,
    /// PSF padding strategy.
    pub pad: PsfPadStrategy,
}

impl Default for WienerFftParams {
    fn default() -> Self {
        Self {
            nsr: 1e-3,
            pad: PsfPadStrategy::ZeroPad,
        }
    }
}

impl WienerFftParams {
    /// Create new FFT Wiener parameters with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the noise-to-signal ratio.
    #[must_use]
    pub const fn with_nsr(mut self, nsr: f32) -> Self {
        self.nsr = nsr;
        self
    }

    /// Set the PSF padding strategy.
    #[must_use]
    pub const fn with_pad(mut self, pad: PsfPadStrategy) -> Self {
        self.pad = pad;
        self
    }
}

/// Richardson-Lucy deconvolution parameters.
#[derive(Debug, Clone)]
pub struct RichardsonLucyParams {
    /// Number of iterations.
    pub iterations: usize,
    /// Convergence threshold.
    pub threshold: f32,
    /// Enable acceleration.
    pub accelerated: bool,
}

impl Default for RichardsonLucyParams {
    fn default() -> Self {
        Self {
            iterations: 30,
            threshold: 0.001,
            accelerated: false,
        }
    }
}

impl RichardsonLucyParams {
    /// Create new Richardson-Lucy parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set convergence threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Enable accelerated version.
    #[must_use]
    pub const fn with_accelerated(mut self, accelerated: bool) -> Self {
        self.accelerated = accelerated;
        self
    }
}

/// Deconvolution processor.
pub struct Deconvolver {
    method: DeconvolutionMethod,
    wiener_params: WienerParams,
    rl_params: RichardsonLucyParams,
    fft_wiener_params: WienerFftParams,
}

impl Deconvolver {
    /// Create a new deconvolver with specified method.
    #[must_use]
    pub fn new(method: DeconvolutionMethod) -> Self {
        Self {
            method,
            wiener_params: WienerParams::default(),
            rl_params: RichardsonLucyParams::default(),
            fft_wiener_params: WienerFftParams::default(),
        }
    }

    /// Set Wiener parameters.
    #[must_use]
    pub fn with_wiener_params(mut self, params: WienerParams) -> Self {
        self.wiener_params = params;
        self
    }

    /// Set Richardson-Lucy parameters.
    #[must_use]
    pub fn with_rl_params(mut self, params: RichardsonLucyParams) -> Self {
        self.rl_params = params;
        self
    }

    /// Set FFT Wiener parameters.
    #[must_use]
    pub fn with_fft_wiener_params(mut self, params: WienerFftParams) -> Self {
        self.fft_wiener_params = params;
        self
    }

    /// Deconvolve an RGB image using the specified PSF.
    ///
    /// # Arguments
    ///
    /// * `image` - Blurred RGB image (width * height * 3)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `psf` - Point spread function
    pub fn deconvolve(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Process each channel separately
        let mut output = vec![0u8; image.len()];

        for c in 0..3 {
            let channel = extract_channel(image, width, height, c);
            let deconvolved = match self.method {
                DeconvolutionMethod::Wiener => {
                    self.wiener_deconvolve(&channel, width, height, psf)?
                }
                DeconvolutionMethod::RichardsonLucy => {
                    self.richardson_lucy_deconvolve(&channel, width, height, psf)?
                }
                DeconvolutionMethod::FFT => self.fft_deconvolve(&channel, width, height, psf)?,
                DeconvolutionMethod::TotalVariation => {
                    self.tv_deconvolve(&channel, width, height, psf)?
                }
                DeconvolutionMethod::FftWiener => {
                    fft_wiener_deconvolve(&channel, width, height, psf, &self.fft_wiener_params)?
                }
            };

            insert_channel(&mut output, &deconvolved, width, height, c);
        }

        Ok(output)
    }

    /// Wiener deconvolution.
    fn wiener_deconvolve(
        &self,
        channel: &[f32],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<f32>> {
        let mut output = vec![0.0; channel.len()];

        let half_w = psf.width as i32 / 2;
        let half_h = psf.height as i32 / 2;

        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in 0..psf.height as i32 {
                    for kx in 0..psf.width as i32 {
                        let ix = x + kx - half_w;
                        let iy = y + ky - half_h;

                        if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                            let idx = (iy * width as i32 + ix) as usize;
                            let kernel_idx = ky as usize * psf.width + kx as usize;

                            if idx < channel.len() && kernel_idx < psf.kernel.len() {
                                let h = psf.kernel[kernel_idx];
                                // Wiener filter: H* / (|H|^2 + NSR)
                                let weight = h
                                    / (h * h
                                        + self.wiener_params.nsr
                                        + self.wiener_params.regularization);
                                sum += channel[idx] * weight;
                                weight_sum += weight.abs();
                            }
                        }
                    }
                }

                let out_idx = (y * width as i32 + x) as usize;
                if out_idx < output.len() {
                    output[out_idx] = if weight_sum > 0.0 {
                        (sum / weight_sum).clamp(0.0, 1.0)
                    } else {
                        channel[out_idx]
                    };
                }
            }
        }

        Ok(output)
    }

    /// Richardson-Lucy iterative deconvolution.
    fn richardson_lucy_deconvolve(
        &self,
        channel: &[f32],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<f32>> {
        // Initialize estimate with the blurred image
        let mut estimate = channel.to_vec();
        let mut previous_estimate = estimate.clone();

        // Create flipped PSF for correlation
        let psf_flipped = flip_psf(psf);

        for iter in 0..self.rl_params.iterations {
            // Convolve estimate with PSF
            let blurred_estimate = convolve_with_psf(&estimate, width, height, psf);

            // Compute ratio of observed to blurred estimate
            let mut ratio = vec![0.0; channel.len()];
            for i in 0..channel.len() {
                if blurred_estimate[i] > f32::EPSILON {
                    ratio[i] = channel[i] / blurred_estimate[i];
                } else {
                    ratio[i] = 1.0;
                }
            }

            // Correlate ratio with flipped PSF
            let correction = convolve_with_psf(&ratio, width, height, &psf_flipped);

            // Update estimate
            for i in 0..estimate.len() {
                estimate[i] *= correction[i];
                estimate[i] = estimate[i].clamp(0.0, 1.0);
            }

            // Check convergence
            if iter > 0 && self.check_convergence(&estimate, &previous_estimate) {
                break;
            }

            previous_estimate.clone_from(&estimate);
        }

        Ok(estimate)
    }

    /// FFT-based deconvolution (simplified spatial domain version).
    fn fft_deconvolve(
        &self,
        channel: &[f32],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<f32>> {
        // For simplicity, use Wiener-like filtering in spatial domain
        self.wiener_deconvolve(channel, width, height, psf)
    }

    /// Total Variation regularized deconvolution.
    fn tv_deconvolve(
        &self,
        channel: &[f32],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<f32>> {
        // Start with Richardson-Lucy
        let mut estimate = self.richardson_lucy_deconvolve(channel, width, height, psf)?;

        // Apply TV denoising iterations
        let lambda = 0.1;
        let tv_iterations = 10;

        for _iter in 0..tv_iterations {
            estimate = apply_tv_denoising(&estimate, width, height, lambda);
        }

        Ok(estimate)
    }

    /// Check convergence between iterations.
    fn check_convergence(&self, current: &[f32], previous: &[f32]) -> bool {
        let mut diff_sum = 0.0;
        let mut count = 0;

        for i in 0..current.len().min(previous.len()) {
            diff_sum += (current[i] - previous[i]).abs();
            count += 1;
        }

        if count == 0 {
            return true;
        }

        let mean_diff = diff_sum / count as f32;
        mean_diff < self.rl_params.threshold
    }
}

impl Default for Deconvolver {
    fn default() -> Self {
        Self::new(DeconvolutionMethod::RichardsonLucy)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a single channel from RGB image to float.
fn extract_channel(image: &[u8], width: u32, height: u32, channel: usize) -> Vec<f32> {
    let size = (width * height) as usize;
    let mut output = vec![0.0; size];

    for i in 0..size {
        if i * 3 + channel < image.len() {
            output[i] = image[i * 3 + channel] as f32 / 255.0;
        }
    }

    output
}

/// Insert a float channel back into RGB image.
fn insert_channel(image: &mut [u8], channel: &[f32], width: u32, height: u32, ch: usize) {
    let size = (width * height) as usize;

    for i in 0..size {
        if i < channel.len() && i * 3 + ch < image.len() {
            image[i * 3 + ch] = (channel[i] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Convolve image with PSF (correlation: g[y,x] = Σ h[ky,kx] * f[y+ky-c_h, x+kx-c_w]).
fn convolve_with_psf(image: &[f32], width: u32, height: u32, psf: &MotionPSF) -> Vec<f32> {
    let mut output = vec![0.0; image.len()];

    let half_w = psf.width as i32 / 2;
    let half_h = psf.height as i32 / 2;

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0.0;

            for ky in 0..psf.height as i32 {
                for kx in 0..psf.width as i32 {
                    let ix = x + kx - half_w;
                    let iy = y + ky - half_h;

                    if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                        let img_idx = (iy * width as i32 + ix) as usize;
                        let kernel_idx = ky as usize * psf.width + kx as usize;

                        if img_idx < image.len() && kernel_idx < psf.kernel.len() {
                            sum += image[img_idx] * psf.kernel[kernel_idx];
                        }
                    }
                }
            }

            let out_idx = (y * width as i32 + x) as usize;
            if out_idx < output.len() {
                output[out_idx] = sum;
            }
        }
    }

    output
}

/// Flip PSF for correlation (reverse convolution).
fn flip_psf(psf: &MotionPSF) -> MotionPSF {
    let mut flipped = MotionPSF::new(psf.width, psf.height);

    for y in 0..psf.height {
        for x in 0..psf.width {
            let src_x = psf.width - 1 - x;
            let src_y = psf.height - 1 - y;
            flipped.set(x, y, psf.get(src_x, src_y));
        }
    }

    flipped
}

/// Apply Total Variation denoising.
fn apply_tv_denoising(image: &[f32], width: u32, height: u32, lambda: f32) -> Vec<f32> {
    let mut output = image.to_vec();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            if idx >= image.len() {
                continue;
            }

            // Compute gradients
            let idx_right = (y * width + x + 1) as usize;
            let idx_left = (y * width + x - 1) as usize;
            let idx_down = ((y + 1) * width + x) as usize;
            let idx_up = ((y - 1) * width + x) as usize;

            if idx_right < image.len()
                && idx_left < image.len()
                && idx_down < image.len()
                && idx_up < image.len()
            {
                let grad_x = image[idx_right] - image[idx_left];
                let grad_y = image[idx_down] - image[idx_up];

                // TV regularization
                let grad_mag = (grad_x * grad_x + grad_y * grad_y).sqrt().max(f32::EPSILON);
                let div = (grad_x + grad_y) / grad_mag;

                output[idx] = image[idx] + lambda * div;
                output[idx] = output[idx].clamp(0.0, 1.0);
            }
        }
    }

    output
}

/// Compute image gradient magnitude (for edge detection in deconvolution).
#[allow(dead_code)]
fn compute_gradient_magnitude(image: &[f32], width: u32, height: u32) -> Vec<f32> {
    let mut gradient = vec![0.0; image.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let idx_right = (y * width + x + 1) as usize;
            let idx_left = (y * width + x - 1) as usize;
            let idx_down = ((y + 1) * width + x) as usize;
            let idx_up = ((y - 1) * width + x) as usize;

            if idx < image.len()
                && idx_right < image.len()
                && idx_left < image.len()
                && idx_down < image.len()
                && idx_up < image.len()
            {
                let grad_x = (image[idx_right] - image[idx_left]) / 2.0;
                let grad_y = (image[idx_down] - image[idx_up]) / 2.0;
                gradient[idx] = (grad_x * grad_x + grad_y * grad_y).sqrt();
            }
        }
    }

    gradient
}

/// Edge-preserving smoothing for deconvolution.
#[allow(dead_code)]
fn bilateral_filter(
    image: &[f32],
    width: u32,
    height: u32,
    spatial_sigma: f32,
    range_sigma: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; image.len()];
    let window_size = (spatial_sigma * 3.0).ceil() as i32;

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let idx = (y * width as i32 + x) as usize;
            if idx >= image.len() {
                continue;
            }

            let center_val = image[idx];
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for dy in -window_size..=window_size {
                for dx in -window_size..=window_size {
                    let nx = x + dx;
                    let ny = y + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let nidx = (ny * width as i32 + nx) as usize;
                        if nidx < image.len() {
                            let neighbor_val = image[nidx];

                            // Spatial weight
                            let spatial_dist = (dx * dx + dy * dy) as f32;
                            let spatial_weight =
                                (-spatial_dist / (2.0 * spatial_sigma * spatial_sigma)).exp();

                            // Range weight
                            let range_dist = (center_val - neighbor_val).abs();
                            let range_weight = (-range_dist * range_dist
                                / (2.0 * range_sigma * range_sigma))
                                .exp();

                            let weight = spatial_weight * range_weight;
                            sum += neighbor_val * weight;
                            weight_sum += weight;
                        }
                    }
                }
            }

            if weight_sum > 0.0 {
                output[idx] = sum / weight_sum;
            } else {
                output[idx] = center_val;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Helpers for FFT-Wiener deconvolution
// ---------------------------------------------------------------------------

/// Return the smallest power of 2 that is >= `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Place the PSF kernel into a zero-filled buffer of size `(pad_h × pad_w)` with
/// **wrap-around centering** so that the PSF energy center-of-mass lands at index (0, 0).
///
/// This is the standard FFTW convention for computing circular convolution.
/// We shift by the **center-of-mass** (not `psf.center_x/y`) because the
/// `from_motion_vector` builder produces a PSF whose CoM may be slightly
/// offset from the geometric center due to sub-pixel bilinear placement.
/// Using the CoM-based integer shift aligns the FFT model with the spatial
/// correlation model (`apply_to_channel`/`convolve_with_psf`).
fn pad_psf_wrap_centered(psf: &MotionPSF, pad_w: usize, pad_h: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; pad_h * pad_w];

    // Normalize PSF to unit sum to avoid brightness drift.
    let psf_sum: f32 = psf.kernel.iter().sum();
    let norm = if psf_sum.abs() > f32::EPSILON {
        1.0 / psf_sum
    } else {
        1.0
    };

    // Compute center of mass of the kernel.
    let mut com_x = 0.0f32;
    let mut com_y = 0.0f32;
    for ky in 0..psf.height {
        for kx in 0..psf.width {
            let v = psf.kernel[ky * psf.width + kx];
            com_x += kx as f32 * v;
            com_y += ky as f32 * v;
        }
    }
    // Use the total weight (not normalized by norm, since we're just finding the shift).
    if psf_sum.abs() > f32::EPSILON {
        com_x /= psf_sum;
        com_y /= psf_sum;
    }
    // Round to nearest integer pixel.
    let shift_x = com_x.round() as i64;
    let shift_y = com_y.round() as i64;

    for ky in 0..psf.height {
        for kx in 0..psf.width {
            let val = psf.kernel[ky * psf.width + kx] * norm;
            // Shift so CoM maps to (0, 0), wrap negative indices.
            let dest_y = ((ky as i64 - shift_y).rem_euclid(pad_h as i64)) as usize;
            let dest_x = ((kx as i64 - shift_x).rem_euclid(pad_w as i64)) as usize;
            padded[dest_y * pad_w + dest_x] += val;
        }
    }
    padded
}

/// Build a padded image with smooth Hann ramp in the zero-padded extension zone.
///
/// The image content in `[0..h, 0..w]` is placed unchanged. The extension zone
/// `[0..pad_h, 0..pad_w]` outside the image is filled with a smooth cosine ramp
/// that transitions from the edge pixel value to zero over `taper_px` pixels.
/// This suppresses the abrupt discontinuity at the image boundary that would
/// otherwise leak Gibbs-like ringing throughout the deconvolved output.
fn pad_image_with_smooth_extension(
    channel: &[f32],
    w: usize,
    h: usize,
    pad_w: usize,
    pad_h: usize,
    taper_px: usize,
) -> Vec<f32> {
    use std::f32::consts::PI;
    let t = taper_px.max(1);
    let mut padded = vec![0.0f32; pad_h * pad_w];

    // First, fill the image region.
    for row in 0..h {
        let src_start = row * w;
        let dst_start = row * pad_w;
        padded[dst_start..dst_start + w].copy_from_slice(&channel[src_start..src_start + w]);
    }

    // Horizontally: ramp from the right edge of the image into the zero zone.
    for row in 0..h {
        let dst_start = row * pad_w;
        let edge_val = channel[row * w + (w - 1)];
        let taper_end = (w + t).min(pad_w);
        for col in w..taper_end {
            let dist = col - w;
            // Hann ramp: 1 at boundary, 0 at taper end.
            let alpha = 0.5 + 0.5 * (PI * dist as f32 / t as f32).cos();
            padded[dst_start + col] = edge_val * alpha;
        }
    }

    // Vertically: ramp from the bottom edge of the image into the zero zone.
    for row in h..((h + t).min(pad_h)) {
        let dist = row - h;
        let alpha = 0.5 + 0.5 * (PI * dist as f32 / t as f32).cos();
        let dst_start = row * pad_w;
        for col in 0..w {
            let edge_val = channel[(h - 1) * w + col];
            padded[dst_start + col] = edge_val * alpha;
        }
    }

    // Corners: blend both tapers.
    for row in h..((h + t).min(pad_h)) {
        let dist_v = row - h;
        let alpha_v = 0.5 + 0.5 * (PI * dist_v as f32 / t as f32).cos();
        let dst_start = row * pad_w;
        for col in w..((w + t).min(pad_w)) {
            let dist_h = col - w;
            let alpha_h = 0.5 + 0.5 * (PI * dist_h as f32 / t as f32).cos();
            let edge_val = channel[(h - 1) * w + (w - 1)];
            padded[dst_start + col] = edge_val * alpha_v * alpha_h;
        }
    }

    padded
}

/// Frequency-domain Wiener deconvolution matched to the correlation-based forward
/// blur used by `MotionPSF::apply_to_channel`.
///
/// **Forward model:** `G = conj(H) · F`  (DFT of correlation, not convolution)
///
/// **Wiener filter:** `F̂(u,v) = H(u,v) · G(u,v) / (|H(u,v)|² + NSR)`
///
/// Note the numerator uses `H` (not `conj(H)`), which distinguishes deconvolution
/// of correlation from deconvolution of convolution.
///
/// **Boundary treatment:** The zero-padded extension zone is filled with a smooth
/// cosine ramp (Hann roll-off) from the edge pixel value to zero over `psf.width/2`
/// pixels. This prevents the abrupt boundary discontinuity from leaking Gibbs-like
/// ringing into the deconvolved interior. The image content itself is not modified.
///
/// Algorithm (Gonzalez & Woods §5.9, adapted):
/// 1. Smooth-extend `g` into zero-padded region; pad to next power-of-2 > `w + psf.width - 1`
/// 2. `G = rfft2d(g_extended_padded)`, `H = rfft2d(h_padded_centered_at_origin)`
/// 3. `W(u,v) = H(u,v) / (|H(u,v)|² + nsr)`
/// 4. `F̂(u,v) = W(u,v) · G(u,v)`
/// 5. `f̂ = irfft2d(F̂).crop(0..h, 0..w).clamp(0, 1)`
fn fft_wiener_deconvolve(
    channel: &[f32],
    width: u32,
    height: u32,
    psf: &MotionPSF,
    params: &WienerFftParams,
) -> CvResult<Vec<f32>> {
    let w = width as usize;
    let h = height as usize;

    if w == 0 || h == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    // Pad to next power-of-2 with enough room to avoid circular wrap of PSF.
    let pad_w = next_pow2(w + psf.width - 1);
    let pad_h = next_pow2(h + psf.height - 1);

    // Taper width in the extension zone: half PSF width, capped to avoid large overheads.
    let taper_px = (psf.width / 2).clamp(1, w / 4);

    // Pad image. Both strategies use smooth Hann extension in the padded zone to
    // suppress the boundary discontinuity that causes Gibbs-like ringing. The
    // `ReplicateEdge` strategy also fills the interior padded rows with replicated
    // edge values; `ZeroPad` fills with the smooth ramp (from edge → 0).
    let g_padded = match params.pad {
        PsfPadStrategy::ZeroPad | PsfPadStrategy::ReplicateEdge => {
            pad_image_with_smooth_extension(channel, w, h, pad_w, pad_h, taper_px)
        }
    };

    // Pad PSF with wrap-around centering at origin.
    let h_padded = pad_psf_wrap_centered(psf, pad_w, pad_h);

    // Forward 2D real FFT: G(u,v) and H(u,v).
    // rfft2d returns n0*(n1/2+1) complex values where n0=rows, n1=cols.
    let g_freq = rfft2d::<f32>(&g_padded, pad_h, pad_w);
    let h_freq = rfft2d::<f32>(&h_padded, pad_h, pad_w);

    let freq_len = pad_h * (pad_w / 2 + 1);
    if g_freq.len() != freq_len || h_freq.len() != freq_len {
        return Err(CvError::transform_error(
            "fft_wiener_deconvolve: unexpected FFT output length",
        ));
    }

    // Apply Wiener filter for correlation-based forward model.
    //
    // Forward: G = conj(H) · F  =>  F̂ = H · G / (|H|² + NSR)
    //
    // We use H directly (not conj(H)) because the forward model uses
    // correlation, not convolution.
    let nsr = params.nsr;
    let mut f_freq: Vec<Complex<f32>> = Vec::with_capacity(freq_len);
    for i in 0..freq_len {
        let h_val = h_freq[i];
        let g_val = g_freq[i];
        let h_mag2 = h_val.re * h_val.re + h_val.im * h_val.im;
        let denom = h_mag2 + nsr;
        // W = H / denom, then F̂ = W * G
        let w_re = h_val.re / denom;
        let w_im = h_val.im / denom;
        let f_re = w_re * g_val.re - w_im * g_val.im;
        let f_im = w_re * g_val.im + w_im * g_val.re;
        f_freq.push(Complex::new(f_re, f_im));
    }

    // Inverse 2D real FFT: irfft2d already normalizes by 1/(pad_h * pad_w).
    let f_padded = irfft2d::<f32>(&f_freq, pad_h, pad_w);

    // Crop back to original dimensions and clamp to [0, 1].
    let mut output = vec![0.0f32; w * h];
    for row in 0..h {
        for col in 0..w {
            let val = f_padded[row * pad_w + col].clamp(0.0, 1.0);
            output[row * w + col] = val;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiener_params_default() {
        let params = WienerParams::default();
        assert!((params.nsr - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_rl_params_default() {
        let params = RichardsonLucyParams::default();
        assert_eq!(params.iterations, 30);
        assert!((params.threshold - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_deconvolver_new() {
        let deconv = Deconvolver::new(DeconvolutionMethod::Wiener);
        assert!(matches!(deconv.method, DeconvolutionMethod::Wiener));
    }

    #[test]
    fn test_fft_wiener_new() {
        let deconv = Deconvolver::new(DeconvolutionMethod::FftWiener);
        assert!(matches!(deconv.method, DeconvolutionMethod::FftWiener));
    }

    #[test]
    fn test_wiener_fft_params_default() {
        let params = WienerFftParams::default();
        assert!((params.nsr - 1e-3).abs() < 1e-6);
        assert_eq!(params.pad, PsfPadStrategy::ZeroPad);
    }

    #[test]
    fn test_extract_channel() {
        let image = vec![10u8, 20, 30, 40, 50, 60];
        let channel = extract_channel(&image, 2, 1, 0);
        assert_eq!(channel.len(), 2);
        assert!((channel[0] - 10.0 / 255.0).abs() < 0.001);
    }

    #[test]
    fn test_flip_psf() {
        let mut psf = MotionPSF::new(3, 3);
        psf.set(0, 0, 1.0);
        psf.set(2, 2, 2.0);

        let flipped = flip_psf(&psf);
        assert!((flipped.get(2, 2) - 1.0).abs() < 0.001);
        assert!((flipped.get(0, 0) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tv_denoising() {
        let image = vec![0.5f32; 100];
        let denoised = apply_tv_denoising(&image, 10, 10, 0.1);
        assert_eq!(denoised.len(), 100);
    }

    /// FFT round-trip: rfft2d → irfft2d recovers the input.
    #[test]
    fn test_fft_roundtrip_simple() {
        let n0 = 4usize;
        let n1 = 4usize;
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let freq = rfft2d::<f32>(&input, n0, n1);
        let recovered = irfft2d::<f32>(&freq, n0, n1);
        let mse: f32 = input
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        assert!(
            mse < 1e-5,
            "FFT roundtrip MSE should be near zero, got {:.6}",
            mse
        );
    }

    /// Identity PSF Wiener round-trip via `fft_wiener_deconvolve`.
    ///
    /// A delta PSF has H=1 at all frequencies; Wiener filter ≈ identity.
    /// PSNR should exceed 30 dB.
    #[test]
    fn test_fft_wiener_noisefree_identity_roundtrip() {
        let w: u32 = 32;
        let h: u32 = 32;

        let mut channel = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                channel[y * w as usize + x] = (x + y) as f32 / (w + h - 2) as f32;
            }
        }

        let mut id_psf = MotionPSF::new(7, 7);
        id_psf.set(3, 3, 1.0);

        let blurred = convolve_with_psf(&channel, w, h, &id_psf);

        let params = WienerFftParams {
            nsr: 1e-8,
            pad: PsfPadStrategy::ZeroPad,
        };
        let recovered = fft_wiener_deconvolve(&blurred, w, h, &id_psf, &params)
            .expect("identity fft_wiener should succeed");

        let mse: f64 = channel
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / channel.len() as f64;
        let psnr = if mse < 1e-14 {
            100.0
        } else {
            10.0 * (1.0_f64 / mse).log10()
        };
        assert!(
            psnr > 30.0,
            "Identity PSF FFT Wiener PSNR should be > 30 dB, got {:.2}",
            psnr
        );
    }
}
