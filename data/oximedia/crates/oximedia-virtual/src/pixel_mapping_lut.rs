//! Lookup-table cached LED panel pixel-to-UV mapping.
//!
//! Provides `PixelMappingLut` which pre-computes and caches UV coordinates for
//! every pixel on an LED panel.  Cache-friendly remap operations avoid
//! per-pixel recomputation across frames.

use serde::{Deserialize, Serialize};

/// UV coordinate pair [u, v] in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Uv {
    pub u: f32,
    pub v: f32,
}

impl Uv {
    #[must_use]
    pub const fn new(u: f32, v: f32) -> Self {
        Self { u, v }
    }

    /// Bilinear interpolation of four corners.
    #[must_use]
    pub fn bilinear(tl: Self, tr: Self, bl: Self, br: Self, tx: f32, ty: f32) -> Self {
        let u = tl.u * (1.0 - tx) * (1.0 - ty)
            + tr.u * tx * (1.0 - ty)
            + bl.u * (1.0 - tx) * ty
            + br.u * tx * ty;
        let v = tl.v * (1.0 - tx) * (1.0 - ty)
            + tr.v * tx * (1.0 - ty)
            + bl.v * (1.0 - tx) * ty
            + br.v * tx * ty;
        Self { u, v }
    }
}

/// Distortion warp model for a single LED panel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WarpModel {
    /// Perfect flat panel — UV maps linearly to pixel coordinates.
    Identity,
    /// Barrel distortion applied to the UV map.
    Barrel {
        /// Radial coefficient k1 (negative = barrel).
        k1: f32,
    },
    /// Pincushion distortion.
    Pincushion { k1: f32 },
    /// Four-corner pin-prick warp (projective transform).
    PerspectiveWarp {
        /// Normalized offsets [dx, dy] at each corner: TL, TR, BR, BL.
        offsets: [[f32; 2]; 4],
    },
}

impl WarpModel {
    /// Apply the warp to normalised UV `[0,1]` and return the warped UV.
    #[must_use]
    pub fn apply(&self, u: f32, v: f32) -> Uv {
        match *self {
            Self::Identity => Uv::new(u, v),
            Self::Barrel { k1 } => {
                let cu = u - 0.5;
                let cv = v - 0.5;
                let r2 = cu * cu + cv * cv;
                let scale = 1.0 + k1 * r2;
                Uv::new(
                    (cu * scale + 0.5).clamp(0.0, 1.0),
                    (cv * scale + 0.5).clamp(0.0, 1.0),
                )
            }
            Self::Pincushion { k1 } => Self::Barrel { k1: -k1 }.apply(u, v),
            Self::PerspectiveWarp { offsets } => {
                // Bilinear interpolation of corner offsets
                let [tl, tr, br, bl] = offsets;
                let wu = tl[0] * (1.0 - u) * (1.0 - v)
                    + tr[0] * u * (1.0 - v)
                    + br[0] * u * v
                    + bl[0] * (1.0 - u) * v;
                let wv = tl[1] * (1.0 - u) * (1.0 - v)
                    + tr[1] * u * (1.0 - v)
                    + br[1] * u * v
                    + bl[1] * (1.0 - u) * v;
                Uv::new((u + wu).clamp(0.0, 1.0), (v + wv).clamp(0.0, 1.0))
            }
        }
    }
}

/// Pre-computed UV lookup table for a single LED panel.
///
/// The LUT stores one UV entry per LED pixel.  After construction the table
/// can be queried in O(1) for any pixel.
pub struct PixelMappingLut {
    /// Stored UVs in row-major order.
    table: Vec<Uv>,
    /// Panel width in pixels.
    pub width: usize,
    /// Panel height in pixels.
    pub height: usize,
    /// Whether this table is valid (all UVs in `[0,1]`).
    pub is_valid: bool,
    /// Warp model used to build this table.
    pub warp_model: WarpModel,
}

impl PixelMappingLut {
    /// Build a UV LUT for a panel of `width × height` pixels using the given warp model.
    #[must_use]
    pub fn build(width: usize, height: usize, warp: WarpModel) -> Self {
        let n = width * height;
        let mut table = Vec::with_capacity(n);
        let mut valid = true;

        for row in 0..height {
            for col in 0..width {
                let u = if width > 1 {
                    col as f32 / (width - 1) as f32
                } else {
                    0.5
                };
                let v = if height > 1 {
                    row as f32 / (height - 1) as f32
                } else {
                    0.5
                };
                let uv = warp.apply(u, v);
                if !(0.0..=1.0).contains(&uv.u) || !(0.0..=1.0).contains(&uv.v) {
                    valid = false;
                }
                table.push(uv);
            }
        }

        Self {
            table,
            width,
            height,
            is_valid: valid,
            warp_model: warp,
        }
    }

    /// Get UV for pixel (col, row).  Returns None if out of bounds.
    #[must_use]
    pub fn get(&self, col: usize, row: usize) -> Option<Uv> {
        if col >= self.width || row >= self.height {
            return None;
        }
        Some(self.table[row * self.width + col])
    }

    /// Sample a texture using the cached UV for pixel (col, row).
    ///
    /// Performs bilinear filtering on the texture.  `tex` is RGB row-major,
    /// `tex_w × tex_h`.  Returns `[r, g, b]` or black if out of bounds.
    #[must_use]
    pub fn sample_texture(
        &self,
        col: usize,
        row: usize,
        tex: &[u8],
        tex_w: usize,
        tex_h: usize,
    ) -> [u8; 3] {
        let Some(uv) = self.get(col, row) else {
            return [0, 0, 0];
        };

        // Map UV to texture coords
        let fx = uv.u * (tex_w as f32 - 1.0);
        let fy = uv.v * (tex_h as f32 - 1.0);

        let x0 = fx as usize;
        let y0 = fy as usize;
        let x1 = (x0 + 1).min(tex_w - 1);
        let y1 = (y0 + 1).min(tex_h - 1);

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let fetch = |x: usize, y: usize| -> [f32; 3] {
            if tex.len() < (y * tex_w + x) * 3 + 3 {
                return [0.0; 3];
            }
            let i = (y * tex_w + x) * 3;
            [tex[i] as f32, tex[i + 1] as f32, tex[i + 2] as f32]
        };

        let tl = fetch(x0, y0);
        let tr = fetch(x1, y0);
        let bl = fetch(x0, y1);
        let br = fetch(x1, y1);

        let r = tl[0] * (1.0 - tx) * (1.0 - ty)
            + tr[0] * tx * (1.0 - ty)
            + bl[0] * (1.0 - tx) * ty
            + br[0] * tx * ty;
        let g = tl[1] * (1.0 - tx) * (1.0 - ty)
            + tr[1] * tx * (1.0 - ty)
            + bl[1] * (1.0 - tx) * ty
            + br[1] * tx * ty;
        let b = tl[2] * (1.0 - tx) * (1.0 - ty)
            + tr[2] * tx * (1.0 - ty)
            + bl[2] * (1.0 - tx) * ty
            + br[2] * tx * ty;

        [r as u8, g as u8, b as u8]
    }

    /// Remap a source texture to the LED panel using the pre-built LUT.
    ///
    /// `source` is RGB row-major `src_w × src_h`.  Returns an RGB buffer of
    /// `width × height` with pixels filled from the warped UV map.
    #[must_use]
    pub fn remap(&self, source: &[u8], src_w: usize, src_h: usize) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = vec![0u8; n * 3];

        for row in 0..self.height {
            for col in 0..self.width {
                let rgb = self.sample_texture(col, row, source, src_w, src_h);
                let i = (row * self.width + col) * 3;
                out[i] = rgb[0];
                out[i + 1] = rgb[1];
                out[i + 2] = rgb[2];
            }
        }

        out
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Maximum UV distortion magnitude (max distance between warped and identity UV).
    #[must_use]
    pub fn max_distortion(&self) -> f32 {
        let mut max_d = 0.0f32;
        for row in 0..self.height {
            for col in 0..self.width {
                let u = if self.width > 1 {
                    col as f32 / (self.width - 1) as f32
                } else {
                    0.5
                };
                let v = if self.height > 1 {
                    row as f32 / (self.height - 1) as f32
                } else {
                    0.5
                };
                let uv = self.table[row * self.width + col];
                let du = uv.u - u;
                let dv = uv.v - v;
                let d = (du * du + dv * dv).sqrt();
                max_d = max_d.max(d);
            }
        }
        max_d
    }
}

/// A cache of per-panel LUTs keyed by panel index.
pub struct PanelLutCache {
    luts: Vec<Option<PixelMappingLut>>,
    capacity: usize,
}

impl PanelLutCache {
    /// Create a cache with capacity for `capacity` panels.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut luts = Vec::with_capacity(capacity);
        luts.resize_with(capacity, || None);
        Self { luts, capacity }
    }

    /// Get or build a LUT for panel `index`.
    ///
    /// If the LUT is not cached, it is built from the given dimensions and warp model.
    pub fn get_or_build(
        &mut self,
        index: usize,
        width: usize,
        height: usize,
        warp: WarpModel,
    ) -> Option<&PixelMappingLut> {
        if index >= self.capacity {
            return None;
        }
        if self.luts[index].is_none() {
            self.luts[index] = Some(PixelMappingLut::build(width, height, warp));
        }
        self.luts[index].as_ref()
    }

    /// Invalidate (drop) the LUT for panel `index`.
    pub fn invalidate(&mut self, index: usize) {
        if index < self.capacity {
            self.luts[index] = None;
        }
    }

    /// Check whether a LUT is cached for panel `index`.
    #[must_use]
    pub fn is_cached(&self, index: usize) -> bool {
        self.luts.get(index).map_or(false, |l| l.is_some())
    }

    /// Number of panels with cached LUTs.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.luts.iter().filter(|l| l.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lut_uv_correct() {
        let lut = PixelMappingLut::build(4, 4, WarpModel::Identity);
        let uv = lut.get(0, 0).expect("should have pixel");
        assert!((uv.u - 0.0).abs() < 1e-5);
        assert!((uv.v - 0.0).abs() < 1e-5);

        let uv = lut.get(3, 3).expect("should have pixel");
        assert!((uv.u - 1.0).abs() < 1e-5);
        assert!((uv.v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_identity_lut_is_valid() {
        let lut = PixelMappingLut::build(16, 16, WarpModel::Identity);
        assert!(lut.is_valid);
    }

    #[test]
    fn test_identity_lut_zero_distortion() {
        let lut = PixelMappingLut::build(8, 8, WarpModel::Identity);
        assert!(
            lut.max_distortion() < 1e-5,
            "identity should have zero distortion"
        );
    }

    #[test]
    fn test_barrel_distortion_lut() {
        let lut = PixelMappingLut::build(16, 16, WarpModel::Barrel { k1: -0.3 });
        assert!(
            lut.max_distortion() > 0.0,
            "barrel should have nonzero distortion"
        );
    }

    #[test]
    fn test_pincushion_distortion_lut() {
        let lut = PixelMappingLut::build(16, 16, WarpModel::Pincushion { k1: 0.3 });
        assert!(lut.max_distortion() > 0.0);
    }

    #[test]
    fn test_perspective_warp_lut() {
        let lut = PixelMappingLut::build(
            16,
            16,
            WarpModel::PerspectiveWarp {
                offsets: [[0.0, 0.0], [0.05, 0.0], [0.05, 0.05], [0.0, 0.05]],
            },
        );
        assert!(lut.max_distortion() > 0.0);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let lut = PixelMappingLut::build(8, 8, WarpModel::Identity);
        assert!(lut.get(8, 0).is_none());
        assert!(lut.get(0, 8).is_none());
    }

    #[test]
    fn test_sample_texture_identity() {
        // 2×2 texture: red, green, blue, white
        let tex = vec![
            255u8, 0, 0, // (0,0) red
            0, 255, 0, // (1,0) green
            0, 0, 255, // (0,1) blue
            255, 255, 255, // (1,1) white
        ];
        let lut = PixelMappingLut::build(2, 2, WarpModel::Identity);
        let px00 = lut.sample_texture(0, 0, &tex, 2, 2);
        assert_eq!(px00, [255, 0, 0], "top-left should be red");

        let px10 = lut.sample_texture(1, 0, &tex, 2, 2);
        assert_eq!(px10, [0, 255, 0], "top-right should be green");
    }

    #[test]
    fn test_remap_identity_copies_source() {
        let source = vec![100u8; 8 * 8 * 3];
        let lut = PixelMappingLut::build(8, 8, WarpModel::Identity);
        let out = lut.remap(&source, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
        // Identity should reproduce source pixels closely
        assert!(out.iter().all(|&v| (v as i32 - 100).abs() <= 2));
    }

    #[test]
    fn test_pixel_count() {
        let lut = PixelMappingLut::build(12, 8, WarpModel::Identity);
        assert_eq!(lut.pixel_count(), 96);
    }

    #[test]
    fn test_uv_bilinear() {
        let tl = Uv::new(0.0, 0.0);
        let tr = Uv::new(1.0, 0.0);
        let bl = Uv::new(0.0, 1.0);
        let br = Uv::new(1.0, 1.0);
        let mid = Uv::bilinear(tl, tr, bl, br, 0.5, 0.5);
        assert!((mid.u - 0.5).abs() < 1e-5);
        assert!((mid.v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_warp_model_identity() {
        let uv = WarpModel::Identity.apply(0.3, 0.7);
        assert!((uv.u - 0.3).abs() < 1e-5);
        assert!((uv.v - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_panel_lut_cache_get_or_build() {
        let mut cache = PanelLutCache::new(4);
        assert!(!cache.is_cached(0));

        let lut = cache.get_or_build(0, 8, 8, WarpModel::Identity);
        assert!(lut.is_some());
        assert!(cache.is_cached(0));
        assert_eq!(cache.cached_count(), 1);
    }

    #[test]
    fn test_panel_lut_cache_invalidate() {
        let mut cache = PanelLutCache::new(4);
        cache.get_or_build(0, 8, 8, WarpModel::Identity);
        assert!(cache.is_cached(0));
        cache.invalidate(0);
        assert!(!cache.is_cached(0));
    }

    #[test]
    fn test_panel_lut_cache_out_of_bounds() {
        let mut cache = PanelLutCache::new(4);
        let result = cache.get_or_build(10, 8, 8, WarpModel::Identity);
        assert!(result.is_none());
    }

    #[test]
    fn test_panel_lut_cache_multiple_panels() {
        let mut cache = PanelLutCache::new(8);
        for i in 0..8 {
            cache.get_or_build(i, 4, 4, WarpModel::Identity);
        }
        assert_eq!(cache.cached_count(), 8);
    }

    #[test]
    fn test_barrel_warp_center_unchanged() {
        let uv = WarpModel::Barrel { k1: -0.5 }.apply(0.5, 0.5);
        // Center should be unchanged by barrel distortion
        assert!((uv.u - 0.5).abs() < 1e-5, "center u: {}", uv.u);
        assert!((uv.v - 0.5).abs() < 1e-5, "center v: {}", uv.v);
    }
}
