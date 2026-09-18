//! GPU-accelerated 2D convolution for filter kernels.
//!
//! Provides spatial convolution operations with configurable kernels:
//!
//! - **Gaussian blur**: Separable 2D Gaussian blur with configurable radius and sigma.
//! - **Sharpen**: Unsharp masking and Laplacian sharpening.
//! - **Edge detection**: Sobel, Prewitt, and Laplacian edge detectors.
//! - **Emboss**: Directional emboss filters.
//! - **Custom kernels**: Arbitrary N x N convolution kernels.
//!
//! All operations use CPU-parallel (rayon) fallback paths.

use crate::error::{AccelError, AccelResult};
use rayon::prelude::*;

/// Predefined convolution filter types.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvolutionFilter {
    /// Gaussian blur with given radius (kernel size = 2*radius+1) and sigma.
    GaussianBlur {
        /// Kernel half-size. Full kernel is (2*radius+1) x (2*radius+1).
        radius: u32,
        /// Standard deviation of the Gaussian. If 0, auto-calculated from radius.
        sigma: f32,
    },
    /// Box blur (uniform averaging) with given radius.
    BoxBlur {
        /// Kernel half-size.
        radius: u32,
    },
    /// Sharpen filter (3x3 Laplacian-based sharpening).
    Sharpen {
        /// Strength multiplier (1.0 = standard, higher = more aggressive).
        strength: f32,
    },
    /// Unsharp mask: sharpen by subtracting a blurred version.
    UnsharpMask {
        /// Blur radius for the mask.
        radius: u32,
        /// Blur sigma for the mask.
        sigma: f32,
        /// Sharpening amount (0.0 = none, 1.0 = full difference added).
        amount: f32,
    },
    /// Sobel edge detection.
    SobelEdge,
    /// Prewitt edge detection.
    PrewittEdge,
    /// Laplacian edge detection.
    LaplacianEdge,
    /// Emboss filter with configurable direction angle (degrees).
    Emboss {
        /// Direction angle in degrees (0 = right, 90 = down).
        angle_degrees: f32,
    },
    /// Custom kernel (row-major, must be square with odd side length).
    Custom {
        /// Side length of the kernel.
        size: u32,
        /// Kernel weights in row-major order.
        weights: Vec<f32>,
    },
}

/// Edge handling mode for convolution boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Clamp coordinates to image boundaries (repeat edge pixels).
    Clamp,
    /// Treat out-of-bounds pixels as zero (black).
    Zero,
    /// Mirror/reflect at the boundary.
    Mirror,
}

/// Configuration for a convolution operation.
#[derive(Debug, Clone)]
pub struct ConvolutionConfig {
    /// The filter to apply.
    pub filter: ConvolutionFilter,
    /// How to handle edges.
    pub edge_mode: EdgeMode,
    /// Whether to normalize the kernel weights to sum to 1.0.
    pub normalize: bool,
}

impl Default for ConvolutionConfig {
    fn default() -> Self {
        Self {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 2,
                sigma: 0.0,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        }
    }
}

/// Generates a 2D Gaussian kernel.
fn generate_gaussian_kernel(radius: u32, sigma: f32) -> (u32, Vec<f32>) {
    let size = 2 * radius + 1;
    let s = if sigma <= 0.0 {
        radius as f32 / 3.0_f32.max(0.5)
    } else {
        sigma
    };
    let mut kernel = vec![0.0f32; (size * size) as usize];
    let center = radius as f32;
    let two_sigma_sq = 2.0 * s * s;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            kernel[(y * size + x) as usize] = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
        }
    }

    // Normalize
    let sum: f32 = kernel.iter().sum();
    if sum.abs() > 1e-6 {
        for w in &mut kernel {
            *w /= sum;
        }
    }

    (size, kernel)
}

/// Generates a box blur kernel.
fn generate_box_kernel(radius: u32) -> (u32, Vec<f32>) {
    let size = 2 * radius + 1;
    let count = (size * size) as f32;
    let kernel = vec![1.0 / count; (size * size) as usize];
    (size, kernel)
}

/// Generates a sharpen kernel with given strength.
fn generate_sharpen_kernel(strength: f32) -> (u32, Vec<f32>) {
    let center = 1.0 + 4.0 * strength;
    let side = -strength;
    #[rustfmt::skip]
    let kernel = vec![
        0.0,  side,   0.0,
        side,  center, side,
        0.0,  side,   0.0,
    ];
    (3, kernel)
}

/// Generates a Sobel kernel (horizontal and vertical).
fn generate_sobel_kernels() -> [(u32, Vec<f32>); 2] {
    #[rustfmt::skip]
    let gx = vec![
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let gy = vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ];
    [(3, gx), (3, gy)]
}

/// Generates a Prewitt kernel (horizontal and vertical).
fn generate_prewitt_kernels() -> [(u32, Vec<f32>); 2] {
    #[rustfmt::skip]
    let gx = vec![
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let gy = vec![
        -1.0, -1.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  1.0,  1.0,
    ];
    [(3, gx), (3, gy)]
}

/// Generates a Laplacian kernel.
fn generate_laplacian_kernel() -> (u32, Vec<f32>) {
    #[rustfmt::skip]
    let kernel = vec![
        0.0,  1.0, 0.0,
        1.0, -4.0, 1.0,
        0.0,  1.0, 0.0,
    ];
    (3, kernel)
}

/// Generates an emboss kernel based on angle.
fn generate_emboss_kernel(angle_degrees: f32) -> (u32, Vec<f32>) {
    let angle_rad = angle_degrees.to_radians();
    let dx = angle_rad.cos();
    let dy = angle_rad.sin();

    // Build a 3x3 emboss kernel from the direction
    #[rustfmt::skip]
    let kernel = vec![
        -2.0 * (-dx - dy).max(0.0).min(1.0), -(-dy).max(0.0).min(1.0), -2.0 * (dx - dy).max(0.0).min(1.0),
        -(-dx).max(0.0).min(1.0),       1.0,                            -dx.max(0.0).min(1.0),
        -2.0 * (-dx + dy).max(0.0).min(1.0), -dy.max(0.0).min(1.0),     -2.0 * (dx + dy).max(0.0).min(1.0),
    ];
    // Simple directional emboss fallback
    let _ = kernel;
    #[rustfmt::skip]
    let kernel = if angle_degrees.abs() < 45.0 || (angle_degrees - 360.0).abs() < 45.0 {
        vec![
            -1.0, -1.0, 0.0,
            -1.0,  1.0, 1.0,
             0.0,  1.0, 1.0,
        ]
    } else if (angle_degrees - 90.0).abs() < 45.0 {
        vec![
             0.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  0.0,
        ]
    } else if (angle_degrees - 180.0).abs() < 45.0 {
        vec![
             1.0,  1.0,  0.0,
             1.0,  1.0, -1.0,
             0.0, -1.0, -1.0,
        ]
    } else {
        vec![
             0.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  0.0,
        ]
    };
    (3, kernel)
}

/// Samples a pixel from the input, applying the edge mode.
#[inline]
fn sample_pixel(
    input: &[u8],
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    channels: u32,
    channel: u32,
    edge_mode: EdgeMode,
) -> f32 {
    let (sx, sy) = match edge_mode {
        EdgeMode::Zero => {
            if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                return 0.0;
            }
            (x as u32, y as u32)
        }
        EdgeMode::Clamp => {
            let cx = x.clamp(0, width as i32 - 1) as u32;
            let cy = y.clamp(0, height as i32 - 1) as u32;
            (cx, cy)
        }
        EdgeMode::Mirror => {
            let cx = if x < 0 {
                (-x).min(width as i32 - 1) as u32
            } else if x >= width as i32 {
                (2 * width as i32 - x - 2).max(0) as u32
            } else {
                x as u32
            };
            let cy = if y < 0 {
                (-y).min(height as i32 - 1) as u32
            } else if y >= height as i32 {
                (2 * height as i32 - y - 2).max(0) as u32
            } else {
                y as u32
            };
            (cx, cy)
        }
    };

    let idx = ((sy * width + sx) * channels + channel) as usize;
    if idx < input.len() {
        f32::from(input[idx])
    } else {
        0.0
    }
}

/// Applies a single convolution kernel to the input image.
fn apply_kernel(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    kernel: &[f32],
    edge_mode: EdgeMode,
) -> Vec<u8> {
    let stride = (width * channels) as usize;
    let radius = (kernel_size / 2) as i32;
    let mut output = vec![0u8; (width * height * channels) as usize];

    output
        .par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                for c in 0..channels {
                    let mut sum = 0.0f32;

                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let sx = x as i32 + kx as i32 - radius;
                            let sy = y as i32 + ky as i32 - radius;
                            let weight = kernel[(ky * kernel_size + kx) as usize];
                            sum +=
                                sample_pixel(input, sx, sy, width, height, channels, c, edge_mode)
                                    * weight;
                        }
                    }

                    let idx = (x * channels + c) as usize;
                    row[idx] = sum.clamp(0.0, 255.0) as u8;
                }
            }
        });

    output
}

/// Applies a gradient magnitude operation from two directional kernels.
fn apply_gradient_magnitude(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    kernels: &[(u32, Vec<f32>); 2],
    edge_mode: EdgeMode,
) -> Vec<u8> {
    let stride = (width * channels) as usize;
    let radius = (kernels[0].0 / 2) as i32;
    let kernel_size = kernels[0].0;
    let mut output = vec![0u8; (width * height * channels) as usize];

    output
        .par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                for c in 0..channels {
                    let mut sum_x = 0.0f32;
                    let mut sum_y = 0.0f32;

                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let sx = x as i32 + kx as i32 - radius;
                            let sy = y as i32 + ky as i32 - radius;
                            let pixel =
                                sample_pixel(input, sx, sy, width, height, channels, c, edge_mode);
                            sum_x += pixel * kernels[0].1[(ky * kernel_size + kx) as usize];
                            sum_y += pixel * kernels[1].1[(ky * kernel_size + kx) as usize];
                        }
                    }

                    let magnitude = (sum_x * sum_x + sum_y * sum_y).sqrt();
                    let idx = (x * channels + c) as usize;
                    row[idx] = magnitude.clamp(0.0, 255.0) as u8;
                }
            }
        });

    output
}

/// Applies a 2D convolution filter to the input image.
///
/// # Arguments
///
/// * `input` - Input image data (row-major, `width * height * channels` bytes)
/// * `width` - Image width
/// * `height` - Image height
/// * `channels` - Number of channels per pixel (1, 3, or 4)
/// * `config` - Convolution configuration
///
/// # Errors
///
/// Returns an error if input size does not match dimensions, or if
/// a custom kernel has invalid dimensions.
pub fn convolve(
    input: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    config: &ConvolutionConfig,
) -> AccelResult<Vec<u8>> {
    let expected = (width * height * channels) as usize;
    if input.len() != expected {
        return Err(AccelError::BufferSizeMismatch {
            expected,
            actual: input.len(),
        });
    }

    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    match &config.filter {
        ConvolutionFilter::GaussianBlur { radius, sigma } => {
            let (size, kernel) = generate_gaussian_kernel(*radius, *sigma);
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::BoxBlur { radius } => {
            let (size, kernel) = generate_box_kernel(*radius);
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::Sharpen { strength } => {
            let (size, kernel) = generate_sharpen_kernel(*strength);
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::UnsharpMask {
            radius,
            sigma,
            amount,
        } => {
            let (size, kernel) = generate_gaussian_kernel(*radius, *sigma);
            let blurred = apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            );
            // Unsharp mask: result = input + amount * (input - blurred)
            let mut output = input.to_vec();
            for i in 0..output.len() {
                let orig = f32::from(input[i]);
                let blur = f32::from(blurred[i]);
                let sharpened = orig + amount * (orig - blur);
                output[i] = sharpened.clamp(0.0, 255.0) as u8;
            }
            Ok(output)
        }
        ConvolutionFilter::SobelEdge => {
            let kernels = generate_sobel_kernels();
            Ok(apply_gradient_magnitude(
                input,
                width,
                height,
                channels,
                &kernels,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::PrewittEdge => {
            let kernels = generate_prewitt_kernels();
            Ok(apply_gradient_magnitude(
                input,
                width,
                height,
                channels,
                &kernels,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::LaplacianEdge => {
            let (size, kernel) = generate_laplacian_kernel();
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::Emboss { angle_degrees } => {
            let (size, kernel) = generate_emboss_kernel(*angle_degrees);
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                size,
                &kernel,
                config.edge_mode,
            ))
        }
        ConvolutionFilter::Custom { size, weights } => {
            if *size == 0 || size % 2 == 0 {
                return Err(AccelError::InvalidDimensions(
                    "Custom kernel size must be odd and > 0".to_string(),
                ));
            }
            let expected_weights = (size * size) as usize;
            if weights.len() != expected_weights {
                return Err(AccelError::InvalidDimensions(format!(
                    "Custom kernel expects {} weights for size {}, got {}",
                    expected_weights,
                    size,
                    weights.len()
                )));
            }
            let mut kernel = weights.clone();
            if config.normalize {
                let sum: f32 = kernel.iter().sum();
                if sum.abs() > 1e-6 {
                    for w in &mut kernel {
                        *w /= sum;
                    }
                }
            }
            Ok(apply_kernel(
                input,
                width,
                height,
                channels,
                *size,
                &kernel,
                config.edge_mode,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_image(width: u32, height: u32, channels: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * channels) as usize]
    }

    fn gradient_image(width: u32, height: u32) -> Vec<u8> {
        let mut img = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                img[(y * width + x) as usize] = (x % 256) as u8;
            }
        }
        img
    }

    #[test]
    fn test_gaussian_blur_uniform() {
        let img = uniform_image(8, 8, 1, 128);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 1,
                sigma: 1.0,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        // Uniform input should produce uniform output
        for &v in &result {
            assert!((v as i32 - 128).abs() <= 1);
        }
    }

    #[test]
    fn test_box_blur() {
        let img = uniform_image(4, 4, 1, 200);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::BoxBlur { radius: 1 },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 4, 4, 1, &config).expect("convolve should succeed");
        for &v in &result {
            assert!((v as i32 - 200).abs() <= 1);
        }
    }

    #[test]
    fn test_sharpen() {
        let img = uniform_image(8, 8, 1, 100);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::Sharpen { strength: 1.0 },
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_sobel_edge_uniform() {
        let img = uniform_image(8, 8, 1, 128);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::SobelEdge,
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        // Uniform image should have zero edges
        for &v in &result {
            assert!(v <= 1, "Expected near-zero edge for uniform input, got {v}");
        }
    }

    #[test]
    fn test_sobel_edge_gradient() {
        let img = gradient_image(16, 16);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::SobelEdge,
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 16, 16, 1, &config).expect("convolve should succeed");
        // Gradient image should produce non-zero edges
        let max_val = result.iter().copied().max().unwrap_or(0);
        assert!(max_val > 0, "Gradient image should produce edges");
    }

    #[test]
    fn test_prewitt_edge() {
        let img = gradient_image(8, 8);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::PrewittEdge,
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_laplacian_edge() {
        let img = gradient_image(8, 8);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::LaplacianEdge,
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_emboss() {
        let img = gradient_image(8, 8);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::Emboss {
                angle_degrees: 45.0,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_unsharp_mask() {
        let img = gradient_image(8, 8);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::UnsharpMask {
                radius: 1,
                sigma: 1.0,
                amount: 0.5,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_custom_kernel_identity() {
        let img = gradient_image(8, 8);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::Custom {
                size: 3,
                weights: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            },
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
        assert_eq!(result, img);
    }

    #[test]
    fn test_custom_kernel_invalid_size() {
        let img = vec![0u8; 16];
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::Custom {
                size: 4, // even - invalid
                weights: vec![0.0; 16],
            },
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 4, 4, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_kernel_weight_mismatch() {
        let img = vec![0u8; 16];
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::Custom {
                size: 3,
                weights: vec![1.0; 5], // wrong count
            },
            edge_mode: EdgeMode::Clamp,
            normalize: false,
        };
        let result = convolve(&img, 4, 4, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_size_mismatch() {
        let config = ConvolutionConfig::default();
        let result = convolve(&[0u8; 10], 4, 4, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_mode_zero() {
        let img = vec![255u8; 9]; // 3x3 single channel
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 1,
                sigma: 1.0,
            },
            edge_mode: EdgeMode::Zero,
            normalize: true,
        };
        let result = convolve(&img, 3, 3, 1, &config).expect("convolve should succeed");
        // Corner pixel should be less than 255 due to zero padding
        assert!(result[0] < 255);
    }

    #[test]
    fn test_edge_mode_mirror() {
        let img = uniform_image(4, 4, 1, 100);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 1,
                sigma: 1.0,
            },
            edge_mode: EdgeMode::Mirror,
            normalize: true,
        };
        let result = convolve(&img, 4, 4, 1, &config).expect("convolve should succeed");
        for &v in &result {
            assert!((v as i32 - 100).abs() <= 1);
        }
    }

    #[test]
    fn test_multichannel_blur() {
        let img = uniform_image(4, 4, 3, 150);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 1,
                sigma: 1.0,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 4, 4, 3, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 48);
        for &v in &result {
            assert!((v as i32 - 150).abs() <= 1);
        }
    }

    #[test]
    fn test_rgba_channel_blur() {
        let img = uniform_image(4, 4, 4, 200);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::BoxBlur { radius: 1 },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 4, 4, 4, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_empty_image() {
        let config = ConvolutionConfig::default();
        let result = convolve(&[], 0, 0, 1, &config).expect("convolve should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_gaussian_kernel_generation() {
        let (size, kernel) = generate_gaussian_kernel(1, 1.0);
        assert_eq!(size, 3);
        assert_eq!(kernel.len(), 9);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sobel_kernels_sum_to_zero() {
        let kernels = generate_sobel_kernels();
        for (_, k) in &kernels {
            let sum: f32 = k.iter().sum();
            assert!(
                sum.abs() < 1e-6,
                "Sobel kernel should sum to zero, got {sum}"
            );
        }
    }

    #[test]
    fn test_emboss_different_angles() {
        let img = gradient_image(8, 8);
        for angle in &[0.0f32, 90.0, 180.0, 270.0] {
            let config = ConvolutionConfig {
                filter: ConvolutionFilter::Emboss {
                    angle_degrees: *angle,
                },
                edge_mode: EdgeMode::Clamp,
                normalize: false,
            };
            let result = convolve(&img, 8, 8, 1, &config).expect("convolve should succeed");
            assert_eq!(result.len(), 64);
        }
    }

    #[test]
    fn test_large_radius_blur() {
        let img = gradient_image(16, 16);
        let config = ConvolutionConfig {
            filter: ConvolutionFilter::GaussianBlur {
                radius: 5,
                sigma: 2.0,
            },
            edge_mode: EdgeMode::Clamp,
            normalize: true,
        };
        let result = convolve(&img, 16, 16, 1, &config).expect("convolve should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_default_config() {
        let config = ConvolutionConfig::default();
        assert_eq!(config.edge_mode, EdgeMode::Clamp);
        assert!(config.normalize);
    }
}
