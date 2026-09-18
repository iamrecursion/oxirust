//! Image inpainting — fill masked regions using surrounding pixels.
#![allow(dead_code)]

/// Describes which pixels in an image are masked for inpainting.
#[derive(Debug, Clone)]
pub struct InpaintMask {
    width: u32,
    height: u32,
    /// `true` means the pixel needs to be filled.
    mask: Vec<bool>,
}

impl InpaintMask {
    /// Create a new mask with all pixels unmasked.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mask: vec![false; (width * height) as usize],
        }
    }

    /// Mark a single pixel as needing inpainting.
    pub fn mark(&mut self, x: u32, y: u32) {
        if x < self.width && y < self.height {
            self.mask[(y * self.width + x) as usize] = true;
        }
    }

    /// Returns the fraction (0.0–1.0) of pixels that are masked.
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage_pct(&self) -> f32 {
        let total = self.mask.len();
        if total == 0 {
            return 0.0;
        }
        let marked = self.mask.iter().filter(|&&v| v).count();
        marked as f32 / total as f32
    }

    /// Returns `true` if the pixel at (x, y) is masked.
    pub fn is_masked(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.mask[(y * self.width + x) as usize]
    }

    /// Width of the mask in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the mask in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Algorithm used to fill masked regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InpaintMethod {
    /// Simple nearest-neighbour diffusion.
    NearestNeighbour,
    /// Patch-based exemplar inpainting.
    PatchBased,
    /// Fast Marching Method (Telea).
    FastMarching,
    /// Navier–Stokes fluid-based filling.
    NavierStokes,
}

impl InpaintMethod {
    /// A numeric quality ranking — higher is generally better but slower.
    pub fn quality(&self) -> u8 {
        match self {
            Self::NearestNeighbour => 1,
            Self::FastMarching => 2,
            Self::NavierStokes => 3,
            Self::PatchBased => 4,
        }
    }

    /// Returns `true` if this method uses patch matching.
    pub fn is_patch_based(&self) -> bool {
        matches!(self, Self::PatchBased)
    }
}

/// A rectangular region of interest for inpainting.
#[derive(Debug, Clone, Copy)]
pub struct InpaintRegion {
    /// Left edge of the region in pixels.
    pub x: u32,
    /// Top edge of the region in pixels.
    pub y: u32,
    /// Width of the region in pixels.
    pub w: u32,
    /// Height of the region in pixels.
    pub h: u32,
}

impl InpaintRegion {
    /// Create a new region.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Returns `true` when the region has non-zero area.
    pub fn is_valid(&self) -> bool {
        self.w > 0 && self.h > 0
    }

    /// Area of the region in pixels.
    pub fn area(&self) -> u64 {
        u64::from(self.w) * u64::from(self.h)
    }
}

/// Result of an inpaint operation.
#[derive(Debug, Clone)]
pub struct InpaintResult {
    /// Pixel data after inpainting (RGBA, 8-bit per channel).
    pub pixels: Vec<u8>,
    /// Width of the output image in pixels.
    pub width: u32,
    /// Height of the output image in pixels.
    pub height: u32,
    /// Number of pixels that were actually filled.
    filled: u32,
}

impl InpaintResult {
    /// Create a new result.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, filled: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            filled,
        }
    }

    /// Number of pixels that were filled during inpainting.
    pub fn filled_pixels(&self) -> u32 {
        self.filled
    }

    /// Returns `true` if at least one pixel was filled.
    pub fn any_filled(&self) -> bool {
        self.filled > 0
    }
}

/// High-level processor that combines a mask, method, and image data.
pub struct InpaintProcessor {
    method: InpaintMethod,
    patch_size: u32,
}

impl InpaintProcessor {
    /// Create a processor with the given method and patch size.
    pub fn new(method: InpaintMethod, patch_size: u32) -> Self {
        Self {
            method,
            patch_size: patch_size.max(3),
        }
    }

    /// Fill masked pixels in `pixels` (RGBA u8) according to `mask`.
    ///
    /// This is a minimal nearest-neighbour implementation used as a
    /// fallback regardless of the chosen method (full implementations
    /// would call into native libraries).
    pub fn fill_region(
        &self,
        pixels: &[u8],
        mask: &InpaintMask,
        _region: &InpaintRegion,
    ) -> InpaintResult {
        let w = mask.width();
        let h = mask.height();
        let mut out = pixels.to_vec();
        let mut filled = 0u32;

        for y in 0..h {
            for x in 0..w {
                if !mask.is_masked(x, y) {
                    continue;
                }
                // Find nearest unmasked pixel (simple scan).
                'found: for r in 1..w.max(h) {
                    for dy in -(r as i64)..=(r as i64) {
                        for dx in -(r as i64)..=(r as i64) {
                            let nx = x as i64 + dx;
                            let ny = y as i64 + dy;
                            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                                continue;
                            }
                            let (nx, ny) = (nx as u32, ny as u32);
                            if !mask.is_masked(nx, ny) {
                                let src = ((ny * w + nx) * 4) as usize;
                                let dst = ((y * w + x) * 4) as usize;
                                if dst + 3 < out.len() && src + 3 < out.len() {
                                    // Copy src bytes before mutably borrowing dst.
                                    let copied =
                                        [out[src], out[src + 1], out[src + 2], out[src + 3]];
                                    out[dst..dst + 4].copy_from_slice(&copied);
                                }
                                filled += 1;
                                break 'found;
                            }
                        }
                    }
                }
            }
        }
        InpaintResult::new(out, w, h, filled)
    }

    /// Returns the currently configured method.
    pub fn method(&self) -> InpaintMethod {
        self.method
    }

    /// Returns the patch size used for patch-based methods.
    pub fn patch_size(&self) -> u32 {
        self.patch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- InpaintMask ---

    #[test]
    fn test_mask_new_all_unmasked() {
        let m = InpaintMask::new(4, 4);
        assert!(!m.is_masked(0, 0));
        assert!(!m.is_masked(3, 3));
    }

    #[test]
    fn test_mask_mark_and_query() {
        let mut m = InpaintMask::new(4, 4);
        m.mark(1, 2);
        assert!(m.is_masked(1, 2));
        assert!(!m.is_masked(0, 0));
    }

    #[test]
    fn test_mask_out_of_bounds_mark() {
        let mut m = InpaintMask::new(4, 4);
        m.mark(10, 10); // should not panic
        assert_eq!(m.coverage_pct(), 0.0);
    }

    #[test]
    fn test_mask_coverage_pct_zero() {
        let m = InpaintMask::new(10, 10);
        assert_eq!(m.coverage_pct(), 0.0);
    }

    #[test]
    fn test_mask_coverage_pct_full() {
        let mut m = InpaintMask::new(2, 2);
        for y in 0..2 {
            for x in 0..2 {
                m.mark(x, y);
            }
        }
        assert!((m.coverage_pct() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mask_coverage_pct_half() {
        let mut m = InpaintMask::new(4, 1);
        m.mark(0, 0);
        m.mark(1, 0);
        assert!((m.coverage_pct() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mask_dimensions() {
        let m = InpaintMask::new(8, 6);
        assert_eq!(m.width(), 8);
        assert_eq!(m.height(), 6);
    }

    // --- InpaintMethod ---

    #[test]
    fn test_method_quality_ordering() {
        assert!(InpaintMethod::NearestNeighbour.quality() < InpaintMethod::PatchBased.quality());
    }

    #[test]
    fn test_method_patch_based_flag() {
        assert!(InpaintMethod::PatchBased.is_patch_based());
        assert!(!InpaintMethod::FastMarching.is_patch_based());
    }

    #[test]
    fn test_method_quality_values() {
        assert_eq!(InpaintMethod::NearestNeighbour.quality(), 1);
        assert_eq!(InpaintMethod::NavierStokes.quality(), 3);
    }

    // --- InpaintRegion ---

    #[test]
    fn test_region_valid() {
        let r = InpaintRegion::new(0, 0, 10, 10);
        assert!(r.is_valid());
    }

    #[test]
    fn test_region_zero_width_invalid() {
        let r = InpaintRegion::new(0, 0, 0, 10);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_region_area() {
        let r = InpaintRegion::new(0, 0, 5, 4);
        assert_eq!(r.area(), 20);
    }

    // --- InpaintProcessor / InpaintResult ---

    #[test]
    fn test_fill_region_no_mask() {
        let w = 2u32;
        let h = 2u32;
        let pixels = vec![255u8; (w * h * 4) as usize];
        let mask = InpaintMask::new(w, h);
        let region = InpaintRegion::new(0, 0, w, h);
        let proc = InpaintProcessor::new(InpaintMethod::NearestNeighbour, 5);
        let result = proc.fill_region(&pixels, &mask, &region);
        assert_eq!(result.filled_pixels(), 0);
    }

    #[test]
    fn test_fill_region_single_masked_pixel() {
        let w = 3u32;
        let h = 3u32;
        let mut pixels = vec![128u8; (w * h * 4) as usize];
        // Make pixel (1,1) different.
        let idx = ((1 * w + 1) * 4) as usize;
        pixels[idx..idx + 4].copy_from_slice(&[0, 0, 0, 0]);
        let mut mask = InpaintMask::new(w, h);
        mask.mark(1, 1);
        let region = InpaintRegion::new(0, 0, w, h);
        let proc = InpaintProcessor::new(InpaintMethod::NearestNeighbour, 3);
        let result = proc.fill_region(&pixels, &mask, &region);
        assert_eq!(result.filled_pixels(), 1);
        assert!(result.any_filled());
    }

    #[test]
    fn test_processor_method_accessor() {
        let p = InpaintProcessor::new(InpaintMethod::PatchBased, 7);
        assert_eq!(p.method(), InpaintMethod::PatchBased);
    }

    #[test]
    fn test_processor_patch_size_minimum() {
        let p = InpaintProcessor::new(InpaintMethod::FastMarching, 1);
        assert_eq!(p.patch_size(), 3); // clamped to minimum 3
    }
}
