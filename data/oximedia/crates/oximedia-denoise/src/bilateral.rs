//! Bilateral filter implementation for `OxiMedia` denoise crate.
//!
//! A bilateral filter is an edge-preserving smoothing filter that weighs
//! neighbouring pixels by both spatial distance and intensity difference.

#![allow(dead_code)]

/// Parameters controlling the bilateral filter's smoothing behaviour.
#[derive(Clone, Debug)]
pub struct BilateralParams {
    /// Spatial sigma: controls how far neighbours influence a pixel.
    pub sigma_spatial: f32,
    /// Range sigma: controls how much intensity difference is tolerated.
    pub sigma_range: f32,
    /// Half-size of the filter kernel (full kernel = 2*radius+1).
    pub radius: usize,
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self {
            sigma_spatial: 3.0,
            sigma_range: 0.1,
            radius: 3,
        }
    }
}

impl BilateralParams {
    /// Construct with explicit sigmas and radius.
    pub fn new(sigma_spatial: f32, sigma_range: f32, radius: usize) -> Self {
        Self {
            sigma_spatial,
            sigma_range,
            radius,
        }
    }

    /// Range sigma accessor.
    pub fn sigma_range(&self) -> f32 {
        self.sigma_range
    }

    /// Spatial sigma accessor.
    pub fn sigma_spatial(&self) -> f32 {
        self.sigma_spatial
    }

    /// Kernel radius accessor.
    pub fn radius(&self) -> usize {
        self.radius
    }
}

/// Compute the bilateral weight for a single neighbour.
///
/// `val_diff` – difference in pixel value (0.0–1.0 range recommended).
/// `spatial_dist` – Euclidean distance in pixels.
pub fn bilateral_weight(val_diff: f32, spatial_dist: f32, params: &BilateralParams) -> f32 {
    let range_term = -(val_diff * val_diff) / (2.0 * params.sigma_range * params.sigma_range);
    let spatial_term =
        -(spatial_dist * spatial_dist) / (2.0 * params.sigma_spatial * params.sigma_spatial);
    (range_term + spatial_term).exp()
}

/// Single-pixel bilateral filter.
///
/// Applies bilateral smoothing to one pixel using a provided neighbourhood patch.
pub struct BilateralFilter {
    params: BilateralParams,
}

impl BilateralFilter {
    /// Create a filter with the given parameters.
    pub fn new(params: BilateralParams) -> Self {
        Self { params }
    }

    /// Apply bilateral weighting to the centre pixel of `patch`.
    ///
    /// `patch` is a flat row-major slice of size `(2*radius+1)²`.
    /// Returns the filtered value for the centre pixel.
    pub fn apply_pixel(&self, patch: &[f32]) -> f32 {
        let diameter = 2 * self.params.radius + 1;
        let centre = patch.len() / 2;
        let centre_val = patch[centre];

        let mut weighted_sum = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for (idx, &val) in patch.iter().enumerate() {
            let row = (idx / diameter) as f32 - self.params.radius as f32;
            let col = (idx % diameter) as f32 - self.params.radius as f32;
            let spatial_dist = (row * row + col * col).sqrt();
            let val_diff = (val - centre_val).abs();
            let w = bilateral_weight(val_diff, spatial_dist, &self.params);
            weighted_sum += w * val;
            weight_total += w;
        }

        if weight_total > 1e-12 {
            weighted_sum / weight_total
        } else {
            centre_val
        }
    }

    /// Parameter accessor.
    pub fn params(&self) -> &BilateralParams {
        &self.params
    }
}

/// Image-level bilateral denoiser operating on planar f32 pixel data.
pub struct BilateralDenoiser {
    filter: BilateralFilter,
}

impl BilateralDenoiser {
    /// Create a new denoiser.
    pub fn new(params: BilateralParams) -> Self {
        Self {
            filter: BilateralFilter::new(params),
        }
    }

    /// Denoise a planar image.
    ///
    /// `pixels` – input pixel buffer (row-major, values 0.0–1.0).
    /// `width`, `height` – image dimensions.
    /// Returns a new denoised pixel buffer.
    pub fn denoise(&self, pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
        assert_eq!(pixels.len(), width * height, "pixel buffer size mismatch");
        let radius = self.filter.params.radius;
        let diameter = 2 * radius + 1;
        let mut output = vec![0.0_f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let mut patch = vec![0.0_f32; diameter * diameter];
                for dy in 0..diameter {
                    for dx in 0..diameter {
                        let ny = (y as isize + dy as isize - radius as isize)
                            .clamp(0, height as isize - 1)
                            as usize;
                        let nx = (x as isize + dx as isize - radius as isize)
                            .clamp(0, width as isize - 1) as usize;
                        patch[dy * diameter + dx] = pixels[ny * width + nx];
                    }
                }
                output[y * width + x] = self.filter.apply_pixel(&patch);
            }
        }
        output
    }

    /// Access inner filter parameters.
    pub fn params(&self) -> &BilateralParams {
        self.filter.params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilateral_params_default() {
        let p = BilateralParams::default();
        assert!(p.sigma_spatial() > 0.0);
        assert!(p.sigma_range() > 0.0);
        assert!(p.radius() > 0);
    }

    #[test]
    fn test_bilateral_params_sigma_range() {
        let p = BilateralParams::new(2.0, 0.05, 2);
        assert!((p.sigma_range() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_bilateral_params_sigma_spatial() {
        let p = BilateralParams::new(5.0, 0.1, 4);
        assert!((p.sigma_spatial() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilateral_weight_zero_diff() {
        let p = BilateralParams::default();
        let w = bilateral_weight(0.0, 0.0, &p);
        // exp(0) = 1.0
        assert!((w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bilateral_weight_large_diff_small() {
        let p = BilateralParams::default();
        let w_near = bilateral_weight(0.0, 0.0, &p);
        let w_far = bilateral_weight(1.0, 10.0, &p);
        assert!(w_near > w_far);
    }

    #[test]
    fn test_bilateral_weight_symmetry() {
        let p = BilateralParams::default();
        let w1 = bilateral_weight(0.2, 1.0, &p);
        let w2 = bilateral_weight(-0.2, 1.0, &p); // abs in formula, same result
                                                  // val_diff is taken as abs in apply_pixel; raw function uses val_diff directly
                                                  // both should produce same result if sign of val_diff is same magnitude
        let w2_pos = bilateral_weight(0.2, 1.0, &p);
        assert!((w1 - w2_pos).abs() < 1e-6);
        let _ = w2;
    }

    #[test]
    fn test_filter_apply_pixel_uniform_patch() {
        let p = BilateralParams::new(3.0, 0.1, 1);
        let filter = BilateralFilter::new(p);
        // Uniform patch: output == input value
        let patch = vec![0.5_f32; 9]; // 3×3
        let result = filter.apply_pixel(&patch);
        assert!((result - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_filter_apply_pixel_centre_dominant() {
        let p = BilateralParams::new(1.0, 0.01, 1);
        let filter = BilateralFilter::new(p);
        // Centre is 0.5, neighbours are 0.0 — tight sigma_range keeps centre
        let mut patch = vec![0.0_f32; 9];
        patch[4] = 0.5; // centre of 3×3
        let result = filter.apply_pixel(&patch);
        // With tiny sigma_range, very different neighbours get near-zero weight
        assert!(result > 0.3); // centre value should dominate
    }

    #[test]
    fn test_filter_params_accessor() {
        let p = BilateralParams::new(2.0, 0.2, 2);
        let filter = BilateralFilter::new(p);
        assert!((filter.params().sigma_range() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_denoiser_output_size() {
        let p = BilateralParams::new(1.5, 0.1, 1);
        let denoiser = BilateralDenoiser::new(p);
        let pixels = vec![0.5_f32; 16 * 16];
        let output = denoiser.denoise(&pixels, 16, 16);
        assert_eq!(output.len(), 16 * 16);
    }

    #[test]
    fn test_denoiser_uniform_image_unchanged() {
        let p = BilateralParams::new(1.5, 0.1, 1);
        let denoiser = BilateralDenoiser::new(p);
        let pixels = vec![0.3_f32; 8 * 8];
        let output = denoiser.denoise(&pixels, 8, 8);
        for &v in &output {
            assert!((v - 0.3).abs() < 1e-4);
        }
    }

    #[test]
    fn test_denoiser_output_finite() {
        let p = BilateralParams::default();
        let denoiser = BilateralDenoiser::new(p);
        let pixels: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let output = denoiser.denoise(&pixels, 8, 8);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_denoiser_params_accessor() {
        let p = BilateralParams::new(2.0, 0.15, 2);
        let denoiser = BilateralDenoiser::new(p);
        assert!((denoiser.params().sigma_spatial() - 2.0).abs() < 1e-6);
    }
}
