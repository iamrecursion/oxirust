//! Image pyramid structures for multi-scale image processing.
//!
//! Provides `ImagePyramid` (generic multi-scale storage) and `GaussianPyramid`
//! (successive Gaussian down-sampling).

#![allow(dead_code)]

/// A single level within an image pyramid.
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    /// Level index (0 = finest / original resolution).
    pub level: usize,
    /// Width in pixels at this level.
    pub width: usize,
    /// Height in pixels at this level.
    pub height: usize,
    /// Pixel data in row-major f32 format (single channel).
    pub data: Vec<f32>,
}

impl PyramidLevel {
    /// Creates a new pyramid level.
    #[must_use]
    pub fn new(level: usize, width: usize, height: usize, data: Vec<f32>) -> Self {
        Self {
            level,
            width,
            height,
            data,
        }
    }

    /// Returns the down-sampling scale factor relative to the original image.
    ///
    /// Level 0 has scale factor 1.0, level 1 has 0.5, level 2 has 0.25, etc.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        0.5_f64.powi(self.level as i32)
    }

    /// Returns the number of pixels at this level.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Returns true if the data slice length matches the expected `width * height`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.width * self.height
    }
}

/// A multi-scale image pyramid.
///
/// Levels are stored in order from finest (index 0) to coarsest.
#[derive(Debug, Clone, Default)]
pub struct ImagePyramid {
    levels: Vec<PyramidLevel>,
}

impl ImagePyramid {
    /// Creates an empty pyramid.
    #[must_use]
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Adds a level.  Levels should be added in order (finest first).
    pub fn add_level(&mut self, level: PyramidLevel) {
        self.levels.push(level);
    }

    /// Returns all levels as a slice.
    #[must_use]
    pub fn levels(&self) -> &[PyramidLevel] {
        &self.levels
    }

    /// Returns the number of levels.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Returns the finest (original) level, if any.
    #[must_use]
    pub fn finest(&self) -> Option<&PyramidLevel> {
        self.levels.first()
    }

    /// Returns the coarsest level, if any.
    #[must_use]
    pub fn coarsest(&self) -> Option<&PyramidLevel> {
        self.levels.last()
    }

    /// Finds the level whose scale factor is closest to `target_scale`.
    ///
    /// Returns `None` if the pyramid is empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn find_by_scale(&self, target_scale: f64) -> Option<&PyramidLevel> {
        self.levels.iter().min_by(|a, b| {
            let da = (a.scale_factor() - target_scale).abs();
            let db = (b.scale_factor() - target_scale).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the level at `index`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&PyramidLevel> {
        self.levels.get(index)
    }
}

/// Builds a Gaussian pyramid by successively down-sampling with a 2×2 box filter.
#[derive(Debug, Clone)]
pub struct GaussianPyramid {
    pyramid: ImagePyramid,
}

impl GaussianPyramid {
    /// Creates an empty Gaussian pyramid.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pyramid: ImagePyramid::new(),
        }
    }

    /// Returns the underlying `ImagePyramid`.
    #[must_use]
    pub fn pyramid(&self) -> &ImagePyramid {
        &self.pyramid
    }

    /// Builds a Gaussian pyramid from the given base image data.
    ///
    /// `max_levels` is capped so that the smallest level is at least 1×1.
    /// The base image (level 0) is stored verbatim; subsequent levels are
    /// produced by 2×2 average down-sampling (a simple box filter).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn build(
        base_data: Vec<f32>,
        base_width: usize,
        base_height: usize,
        max_levels: usize,
    ) -> Self {
        let mut gp = Self::new();
        if base_width == 0 || base_height == 0 || max_levels == 0 {
            return gp;
        }

        let base = PyramidLevel::new(0, base_width, base_height, base_data);
        gp.pyramid.add_level(base);

        let mut prev_level = 0usize;
        while gp.pyramid.depth() < max_levels {
            let prev = &gp.pyramid.levels[prev_level];
            let new_w = (prev.width / 2).max(1);
            let new_h = (prev.height / 2).max(1);

            if new_w == prev.width && new_h == prev.height {
                break; // Cannot down-sample further
            }

            let down = downsample_2x2(&prev.data, prev.width, prev.height, new_w, new_h);
            let new_idx = prev_level + 1;
            gp.pyramid
                .add_level(PyramidLevel::new(new_idx, new_w, new_h, down));
            prev_level = new_idx;

            if new_w == 1 && new_h == 1 {
                break;
            }
        }

        gp
    }
}

impl Default for GaussianPyramid {
    fn default() -> Self {
        Self::new()
    }
}

/// Downsamples `src` (width × height) to `new_w × new_h` using a 2×2 box filter.
#[allow(clippy::cast_precision_loss)]
fn downsample_2x2(src: &[f32], src_w: usize, src_h: usize, new_w: usize, new_h: usize) -> Vec<f32> {
    let mut dst = vec![0.0_f32; new_w * new_h];
    for dy in 0..new_h {
        for dx in 0..new_w {
            let sx = (dx * 2).min(src_w.saturating_sub(1));
            let sy = (dy * 2).min(src_h.saturating_sub(1));
            let sx1 = (sx + 1).min(src_w.saturating_sub(1));
            let sy1 = (sy + 1).min(src_h.saturating_sub(1));
            let sum = src[sy * src_w + sx]
                + src[sy * src_w + sx1]
                + src[sy1 * src_w + sx]
                + src[sy1 * src_w + sx1];
            dst[dy * new_w + dx] = sum / 4.0;
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_level_scale_factor_level0() {
        let lv = PyramidLevel::new(0, 64, 64, vec![0.0; 64 * 64]);
        assert!((lv.scale_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pyramid_level_scale_factor_level1() {
        let lv = PyramidLevel::new(1, 32, 32, vec![0.0; 32 * 32]);
        assert!((lv.scale_factor() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn pyramid_level_scale_factor_level2() {
        let lv = PyramidLevel::new(2, 16, 16, vec![0.0; 16 * 16]);
        assert!((lv.scale_factor() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn pyramid_level_pixel_count() {
        let lv = PyramidLevel::new(0, 4, 8, vec![0.0; 32]);
        assert_eq!(lv.pixel_count(), 32);
    }

    #[test]
    fn pyramid_level_is_valid() {
        let lv = PyramidLevel::new(0, 3, 3, vec![0.0; 9]);
        assert!(lv.is_valid());
    }

    #[test]
    fn pyramid_level_is_invalid() {
        let lv = PyramidLevel::new(0, 3, 3, vec![0.0; 8]);
        assert!(!lv.is_valid());
    }

    #[test]
    fn image_pyramid_add_and_depth() {
        let mut p = ImagePyramid::new();
        p.add_level(PyramidLevel::new(0, 8, 8, vec![1.0; 64]));
        p.add_level(PyramidLevel::new(1, 4, 4, vec![1.0; 16]));
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn image_pyramid_finest_coarsest() {
        let mut p = ImagePyramid::new();
        p.add_level(PyramidLevel::new(0, 8, 8, vec![1.0; 64]));
        p.add_level(PyramidLevel::new(1, 4, 4, vec![1.0; 16]));
        assert_eq!(p.finest().expect("should succeed in test").level, 0);
        assert_eq!(p.coarsest().expect("should succeed in test").level, 1);
    }

    #[test]
    fn image_pyramid_find_by_scale() {
        let mut p = ImagePyramid::new();
        p.add_level(PyramidLevel::new(0, 8, 8, vec![0.0; 64]));
        p.add_level(PyramidLevel::new(1, 4, 4, vec![0.0; 16]));
        p.add_level(PyramidLevel::new(2, 2, 2, vec![0.0; 4]));
        let found = p.find_by_scale(0.26).expect("should succeed in test");
        assert_eq!(found.level, 2); // scale 0.25 is nearest to 0.26
    }

    #[test]
    fn gaussian_pyramid_build_levels() {
        let data = vec![0.5_f32; 64 * 64];
        let gp = GaussianPyramid::build(data, 64, 64, 4);
        assert_eq!(gp.pyramid().depth(), 4);
    }

    #[test]
    fn gaussian_pyramid_width_halves() {
        let data = vec![1.0_f32; 16 * 16];
        let gp = GaussianPyramid::build(data, 16, 16, 3);
        let levels = gp.pyramid().levels();
        assert_eq!(levels[0].width, 16);
        assert_eq!(levels[1].width, 8);
        assert_eq!(levels[2].width, 4);
    }

    #[test]
    fn gaussian_pyramid_uniform_image_preserves_value() {
        // For a uniform image, every level should also be uniform at the same value
        let data = vec![0.8_f32; 8 * 8];
        let gp = GaussianPyramid::build(data, 8, 8, 3);
        for lv in gp.pyramid().levels() {
            assert!(lv.data.iter().all(|&v| (v - 0.8).abs() < 1e-5));
        }
    }

    #[test]
    fn gaussian_pyramid_empty_on_zero_levels() {
        let gp = GaussianPyramid::build(vec![1.0; 4], 2, 2, 0);
        assert_eq!(gp.pyramid().depth(), 0);
    }
}
