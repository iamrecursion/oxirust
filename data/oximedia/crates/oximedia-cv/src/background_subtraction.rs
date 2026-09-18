//! Background subtraction via a per-pixel Mixture-of-Gaussians (MOG2-inspired) model.
//!
//! Each pixel is modelled by up to `num_components` Gaussian distributions in
//! intensity space.  On each call to `apply`, the model is updated with the
//! new frame and a binary foreground mask is returned (255 = foreground,
//! 0 = background).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::background_subtraction::{MixtureOfGaussians, MogConfig};
//!
//! let width = 4u32;
//! let height = 4u32;
//! let bg_frame = vec![128u8; (width * height) as usize];
//!
//! let mut mog = MixtureOfGaussians::new(MogConfig::default(), width, height);
//! // Warm up model with background frames
//! for _ in 0..10 {
//!     mog.apply(&bg_frame);
//! }
//! // A bright foreground object
//! let fg_frame: Vec<u8> = (0..(width * height) as usize)
//!     .map(|i| if i == 5 { 255 } else { 128 })
//!     .collect();
//! let mask = mog.apply(&fg_frame);
//! assert_eq!(mask.len(), (width * height) as usize);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`MixtureOfGaussians`].
#[derive(Debug, Clone)]
pub struct MogConfig {
    /// Maximum number of Gaussian components per pixel (K in the literature).
    pub num_components: u32,
    /// Learning rate α ∈ (0, 1]: how quickly new frames influence the model.
    pub learning_rate: f32,
    /// Initial variance for a newly created component.
    pub initial_variance: f32,
    /// Mahalanobis distance threshold (squared, in σ units) for component membership.
    pub match_threshold_sq: f32,
    /// Ratio of the background weight sum required to classify a component as background.
    pub background_ratio: f32,
    /// Minimum variance to prevent numerical collapse.
    pub min_variance: f32,
    /// Maximum variance to prevent runaway components.
    pub max_variance: f32,
}

impl Default for MogConfig {
    fn default() -> Self {
        Self {
            num_components: 5,
            learning_rate: 0.005,
            initial_variance: 225.0, // σ² = 15² for 8-bit imagery
            match_threshold_sq: 9.0, // 3σ match radius
            background_ratio: 0.9,
            min_variance: 4.0,
            max_variance: 10_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-pixel Gaussian component
// ---------------------------------------------------------------------------

/// A single Gaussian component in a pixel's mixture model.
#[derive(Debug, Clone)]
struct GaussianComponent {
    /// Mean intensity value.
    mean: f32,
    /// Variance σ².
    variance: f32,
    /// Mixing weight (all weights in a pixel sum to ~1).
    weight: f32,
}

impl GaussianComponent {
    fn new(mean: f32, variance: f32, weight: f32) -> Self {
        Self {
            mean,
            variance,
            weight,
        }
    }

    /// Mahalanobis distance squared from `val` to this component.
    fn maha_sq(&self, val: f32) -> f32 {
        let diff = val - self.mean;
        diff * diff / self.variance.max(1e-6)
    }
}

// ---------------------------------------------------------------------------
// MixtureOfGaussians
// ---------------------------------------------------------------------------

/// Per-pixel Mixture-of-Gaussians background subtractor.
///
/// The model stores `num_components` Gaussian components for every pixel of
/// the image.  Pixels whose intensity is well-explained by the background
/// components are classified as background; the rest as foreground.
pub struct MixtureOfGaussians {
    cfg: MogConfig,
    width: u32,
    height: u32,
    /// Flat array: `num_pixels × num_components` components, row-major.
    components: Vec<Vec<GaussianComponent>>,
    /// How many frames have been processed.
    frame_count: u64,
}

impl MixtureOfGaussians {
    /// Create a new MOG model for images of `width × height` pixels.
    ///
    /// Components are initialised lazily on the first frame.
    #[must_use]
    pub fn new(cfg: MogConfig, width: u32, height: u32) -> Self {
        let n_pixels = (width as usize) * (height as usize);
        let components = vec![Vec::new(); n_pixels];
        Self {
            cfg,
            width,
            height,
            components,
            frame_count: 0,
        }
    }

    /// Process a new frame and return a binary foreground mask.
    ///
    /// `frame` must be a grayscale u8 image in row-major order with exactly
    /// `width × height` pixels.  Returns a `Vec<u8>` of the same length where
    /// 255 = foreground and 0 = background.
    ///
    /// If `frame` is a different size the method returns an all-zero mask of
    /// the same length as `frame` without modifying the model.
    #[must_use]
    pub fn apply(&mut self, frame: &[u8]) -> Vec<u8> {
        let n_pixels = (self.width as usize) * (self.height as usize);
        if frame.len() != n_pixels {
            return vec![0u8; frame.len()];
        }

        self.frame_count += 1;
        let alpha = self.cfg.learning_rate;
        let k = self.cfg.num_components as usize;

        let mut mask = vec![0u8; n_pixels];

        for (i, (&pixel, comps)) in frame.iter().zip(self.components.iter_mut()).enumerate() {
            let val = pixel as f32;
            let is_fg = update_pixel_model(val, comps, k, alpha, &self.cfg);
            if is_fg {
                mask[i] = 255;
            }
        }

        mask
    }

    /// Reset the model (clear all learned components).
    pub fn reset(&mut self) {
        for c in self.components.iter_mut() {
            c.clear();
        }
        self.frame_count = 0;
    }

    /// Number of frames processed so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Width of the image the model was created for.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the image the model was created for.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Return the number of active Gaussian components for pixel at (x, y),
    /// or 0 if out of bounds.
    #[must_use]
    pub fn components_at(&self, x: u32, y: u32) -> usize {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        self.components[idx].len()
    }

    /// Compute the fraction of foreground pixels in the most recently returned
    /// mask (for testing / diagnostics).
    #[must_use]
    #[allow(clippy::naive_bytecount)]
    pub fn foreground_fraction(mask: &[u8]) -> f32 {
        if mask.is_empty() {
            return 0.0;
        }
        let fg = mask.iter().filter(|&&v| v == 255).count();
        fg as f32 / mask.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Core per-pixel update logic
// ---------------------------------------------------------------------------

/// Update the Gaussian mixture model for a single pixel and return `true` if
/// the pixel is classified as foreground.
fn update_pixel_model(
    val: f32,
    comps: &mut Vec<GaussianComponent>,
    max_k: usize,
    alpha: f32,
    cfg: &MogConfig,
) -> bool {
    // --- 1. Find a matching component ---
    let mut matched_idx: Option<usize> = None;
    for (i, c) in comps.iter().enumerate() {
        if c.maha_sq(val) <= cfg.match_threshold_sq {
            matched_idx = Some(i);
            break;
        }
    }

    // --- 2. Update weights (decay all, boost matched) ---
    let one_minus_alpha = 1.0 - alpha;
    for c in comps.iter_mut() {
        c.weight *= one_minus_alpha;
    }

    if let Some(mi) = matched_idx {
        // Update the matched component
        let c = &mut comps[mi];
        let rho = alpha / (c.weight + alpha); // per-component learning rate
        let diff = val - c.mean;
        c.mean += rho * diff;
        c.variance = (one_minus_alpha * c.variance + rho * diff * diff)
            .clamp(cfg.min_variance, cfg.max_variance);
        c.weight += alpha;
    } else {
        // No match: create a new component or replace the least-weighted one
        let new_comp = GaussianComponent::new(val, cfg.initial_variance, alpha);
        if comps.len() < max_k {
            comps.push(new_comp);
        } else if !comps.is_empty() {
            // Replace least-weighted component
            let min_idx = comps
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.weight
                        .partial_cmp(&b.1.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            comps[min_idx] = new_comp;
        }
    }

    // --- 3. Renormalise weights ---
    let weight_sum: f32 = comps.iter().map(|c| c.weight).sum();
    if weight_sum > 1e-12 {
        for c in comps.iter_mut() {
            c.weight /= weight_sum;
        }
    }

    // --- 4. Sort by weight/variance ratio (background-first ordering) ---
    comps.sort_by(|a, b| {
        let ra = a.weight / a.variance.max(1e-6);
        let rb = b.weight / b.variance.max(1e-6);
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- 5. Determine foreground ---
    // Components are already sorted: accumulate until background_ratio is reached.
    let mut bg_weight = 0.0_f32;
    let is_bg = 'bg: {
        for c in comps.iter() {
            if c.maha_sq(val) <= cfg.match_threshold_sq {
                // This component matches AND is within the background set
                break 'bg true;
            }
            bg_weight += c.weight;
            if bg_weight >= cfg.background_ratio {
                break;
            }
        }
        false
    };

    !is_bg
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::naive_bytecount)]
mod tests {
    use super::*;

    fn uniform_frame(val: u8, w: u32, h: u32) -> Vec<u8> {
        vec![val; (w * h) as usize]
    }

    // --- MogConfig ---

    #[test]
    fn test_default_config() {
        let cfg = MogConfig::default();
        assert_eq!(cfg.num_components, 5);
        assert!(cfg.learning_rate > 0.0 && cfg.learning_rate <= 1.0);
        assert!(cfg.background_ratio > 0.0 && cfg.background_ratio <= 1.0);
    }

    // --- MixtureOfGaussians construction ---

    #[test]
    fn test_new_mog() {
        let mog = MixtureOfGaussians::new(MogConfig::default(), 8, 8);
        assert_eq!(mog.width(), 8);
        assert_eq!(mog.height(), 8);
        assert_eq!(mog.frame_count(), 0);
    }

    #[test]
    fn test_apply_returns_correct_length() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 4, 4);
        let frame = uniform_frame(100, 4, 4);
        let mask = mog.apply(&frame);
        assert_eq!(mask.len(), 16);
    }

    #[test]
    fn test_apply_increments_frame_count() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 2, 2);
        mog.apply(&uniform_frame(50, 2, 2));
        mog.apply(&uniform_frame(50, 2, 2));
        assert_eq!(mog.frame_count(), 2);
    }

    #[test]
    fn test_mask_values_binary() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 4, 4);
        let frame = uniform_frame(128, 4, 4);
        let mask = mog.apply(&frame);
        for &v in &mask {
            assert!(v == 0 || v == 255, "mask pixel must be 0 or 255, got {v}");
        }
    }

    #[test]
    fn test_wrong_size_frame_returns_zeros() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 4, 4);
        let mask = mog.apply(&[128u8; 10]); // wrong size
        assert!(mask.iter().all(|&v| v == 0));
        assert_eq!(mog.frame_count(), 0); // should not have been updated
    }

    #[test]
    fn test_background_converges_to_zero_fg() {
        let w = 6u32;
        let h = 6u32;
        let mut cfg = MogConfig::default();
        cfg.learning_rate = 0.05;
        let mut mog = MixtureOfGaussians::new(cfg, w, h);
        let bg = uniform_frame(128, w, h);
        // Warm up with many background frames
        for _ in 0..200 {
            mog.apply(&bg);
        }
        let mask = mog.apply(&bg);
        let fg_count = mask.iter().filter(|&&v| v == 255).count();
        assert!(
            fg_count == 0,
            "After convergence, pure bg frame should produce 0 fg pixels, got {fg_count}"
        );
    }

    #[test]
    fn test_sudden_foreground_detected() {
        let w = 4u32;
        let h = 4u32;
        let mut cfg = MogConfig::default();
        cfg.learning_rate = 0.05;
        let mut mog = MixtureOfGaussians::new(cfg, w, h);
        let bg = uniform_frame(100, w, h);
        for _ in 0..100 {
            mog.apply(&bg);
        }
        // Introduce a very different frame (simulate foreground)
        let fg = uniform_frame(250, w, h);
        let mask = mog.apply(&fg);
        let fg_count = mask.iter().filter(|&&v| v == 255).count();
        assert!(
            fg_count > 0,
            "Sudden bright change should produce foreground pixels"
        );
    }

    #[test]
    fn test_reset_clears_model() {
        let w = 2u32;
        let h = 2u32;
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), w, h);
        let bg = uniform_frame(80, w, h);
        for _ in 0..20 {
            mog.apply(&bg);
        }
        mog.reset();
        assert_eq!(mog.frame_count(), 0);
        assert_eq!(mog.components_at(0, 0), 0);
    }

    #[test]
    fn test_components_grow_on_new_values() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 1, 1);
        // Send distinctly different values so new components are created
        for v in [10u8, 100, 200] {
            mog.apply(&[v]);
        }
        let n = mog.components_at(0, 0);
        assert!(n >= 1, "Should have at least 1 component, got {n}");
    }

    #[test]
    fn test_components_capped_at_max_k() {
        let mut cfg = MogConfig::default();
        cfg.num_components = 3;
        let mut mog = MixtureOfGaussians::new(cfg, 1, 1);
        for v in 0..=50u8 {
            mog.apply(&[v * 5]);
        }
        let n = mog.components_at(0, 0);
        assert!(n <= 3, "Components must be capped at max_k=3, got {n}");
    }

    #[test]
    fn test_foreground_fraction_helper() {
        let mask = vec![255u8, 0, 255, 0];
        let frac = MixtureOfGaussians::foreground_fraction(&mask);
        assert!((frac - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_foreground_fraction_empty() {
        let frac = MixtureOfGaussians::foreground_fraction(&[]);
        assert!((frac - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_components_at_out_of_bounds() {
        let mog = MixtureOfGaussians::new(MogConfig::default(), 4, 4);
        assert_eq!(mog.components_at(10, 10), 0);
    }

    #[test]
    fn test_apply_single_pixel() {
        let mut mog = MixtureOfGaussians::new(MogConfig::default(), 1, 1);
        let mask = mog.apply(&[128u8]);
        assert_eq!(mask.len(), 1);
        assert!(mask[0] == 0 || mask[0] == 255);
    }

    #[test]
    fn test_model_adapts_to_background_shift() {
        let w = 2u32;
        let h = 2u32;
        let mut cfg = MogConfig::default();
        cfg.learning_rate = 0.1;
        let mut mog = MixtureOfGaussians::new(cfg, w, h);
        // Warm up at value 100
        for _ in 0..100 {
            mog.apply(&uniform_frame(100, w, h));
        }
        // Shift background to value 150; model should adapt
        for _ in 0..200 {
            mog.apply(&uniform_frame(150, w, h));
        }
        let mask = mog.apply(&uniform_frame(150, w, h));
        let fg_count = mask.iter().filter(|&&v| v == 255).count();
        assert!(
            fg_count == 0,
            "After adaptation, new bg value should not be foreground: fg_count={fg_count}"
        );
    }
}
