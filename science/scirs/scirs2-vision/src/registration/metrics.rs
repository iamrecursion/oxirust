//! Registration quality metrics

use crate::error::Result;
use image::GrayImage;

/// Registration quality metrics
#[derive(Debug, Clone)]
pub struct RegistrationMetrics {
    /// Mean squared error between images
    pub mean_squared_error: f64,
    /// Peak signal-to-noise ratio in dB
    pub peak_signal_to_noise_ratio: f64,
    /// Structural similarity index (SSIM)
    pub structural_similarity_index: f64,
    /// Normalized cross-correlation coefficient
    pub normalized_cross_correlation: f64,
}

/// Compute registration quality metrics
#[allow(dead_code)]
pub fn compute_registration_metrics(
    reference: &GrayImage,
    registered: &GrayImage,
) -> Result<RegistrationMetrics> {
    use crate::error::VisionError;

    // Check that images have the same dimensions
    if reference.dimensions() != registered.dimensions() {
        return Err(VisionError::InvalidInput(
            "Reference and registered images must have the same dimensions".to_string(),
        ));
    }

    // Compute MSE
    let mean_squared_error = compute_mse_metric(reference, registered)?;

    // Compute PSNR
    let peak_signal_to_noise_ratio = compute_psnr_metric(reference, registered)?;

    // Compute SSIM (simplified version)
    let structural_similarity_index = compute_ssim_metric(reference, registered)?;

    // Compute NCC
    let normalized_cross_correlation = compute_ncc_metric(reference, registered)?;

    Ok(RegistrationMetrics {
        mean_squared_error,
        peak_signal_to_noise_ratio,
        structural_similarity_index,
        normalized_cross_correlation,
    })
}

/// Compute Mean Squared Error
#[allow(dead_code)]
pub fn compute_mse_metric(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    use crate::error::VisionError;

    // Check that images have the same dimensions
    if image1.dimensions() != image2.dimensions() {
        return Err(VisionError::InvalidInput(
            "Images must have the same dimensions for MSE computation".to_string(),
        ));
    }

    let (width, height) = image1.dimensions();
    let n_pixels = (width * height) as f64;

    // Compute sum of squared differences
    let mut sum_squared_diff = 0.0;
    for (p1, p2) in image1.pixels().zip(image2.pixels()) {
        let diff = p1[0] as f64 - p2[0] as f64;
        sum_squared_diff += diff * diff;
    }

    // Return mean squared error
    Ok(sum_squared_diff / n_pixels)
}

/// Compute Peak Signal-to-Noise Ratio
#[allow(dead_code)]
pub fn compute_psnr_metric(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    // First compute MSE
    let mse = compute_mse_metric(image1, image2)?;

    // Handle case where images are identical (MSE = 0)
    if mse == 0.0 {
        // Return a very high PSNR value (commonly used convention)
        return Ok(100.0);
    }

    // For 8-bit grayscale images, MAX_I is 255
    const MAX_PIXEL_VALUE: f64 = 255.0;

    // PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    // or equivalently: PSNR = 10 * log10(MAX_I^2 / MSE)
    let psnr = 10.0 * ((MAX_PIXEL_VALUE * MAX_PIXEL_VALUE) / mse).log10();

    Ok(psnr)
}

/// Compute Structural Similarity Index (simplified version)
#[allow(dead_code)]
pub fn compute_ssim_metric(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    use crate::error::VisionError;

    // Check that images have the same dimensions
    if image1.dimensions() != image2.dimensions() {
        return Err(VisionError::InvalidInput(
            "Images must have the same dimensions for SSIM computation".to_string(),
        ));
    }

    let (width, height) = image1.dimensions();
    let n_pixels = (width * height) as f64;

    // Compute means
    let mut mean1 = 0.0;
    let mut mean2 = 0.0;
    for (p1, p2) in image1.pixels().zip(image2.pixels()) {
        mean1 += p1[0] as f64;
        mean2 += p2[0] as f64;
    }
    mean1 /= n_pixels;
    mean2 /= n_pixels;

    // Compute variances and covariance
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    let mut covar = 0.0;
    for (p1, p2) in image1.pixels().zip(image2.pixels()) {
        let v1 = p1[0] as f64 - mean1;
        let v2 = p2[0] as f64 - mean2;
        var1 += v1 * v1;
        var2 += v2 * v2;
        covar += v1 * v2;
    }
    var1 /= n_pixels - 1.0;
    var2 /= n_pixels - 1.0;
    covar /= n_pixels - 1.0;

    // SSIM constants (from the original paper)
    const K1: f64 = 0.01;
    const K2: f64 = 0.03;
    const L: f64 = 255.0; // Dynamic range for 8-bit images
    let c1 = (K1 * L) * (K1 * L);
    let c2 = (K2 * L) * (K2 * L);

    // Compute SSIM
    let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

    Ok(numerator / denominator)
}

/// Compute Normalized Cross-Correlation
#[allow(dead_code)]
pub fn compute_ncc_metric(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    use crate::error::VisionError;

    // Check that images have the same dimensions
    if image1.dimensions() != image2.dimensions() {
        return Err(VisionError::InvalidInput(
            "Images must have the same dimensions for NCC computation".to_string(),
        ));
    }

    let (width, height) = image1.dimensions();
    let n_pixels = (width * height) as f64;

    // Compute means
    let mut mean1 = 0.0;
    let mut mean2 = 0.0;
    for (p1, p2) in image1.pixels().zip(image2.pixels()) {
        mean1 += p1[0] as f64;
        mean2 += p2[0] as f64;
    }
    mean1 /= n_pixels;
    mean2 /= n_pixels;

    // Compute normalized cross-correlation
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for (p1, p2) in image1.pixels().zip(image2.pixels()) {
        let v1 = p1[0] as f64 - mean1;
        let v2 = p2[0] as f64 - mean2;
        numerator += v1 * v2;
        sum_sq1 += v1 * v1;
        sum_sq2 += v2 * v2;
    }

    let denominator = (sum_sq1 * sum_sq2).sqrt();

    // Handle case where one or both images have zero variance
    if denominator == 0.0 {
        if sum_sq1 == 0.0 && sum_sq2 == 0.0 {
            // Both images are constant - they're perfectly correlated if equal
            return Ok(if mean1 == mean2 { 1.0 } else { 0.0 });
        } else {
            // One image is constant, the other isn't - no correlation
            return Ok(0.0);
        }
    }

    Ok(numerator / denominator)
}
