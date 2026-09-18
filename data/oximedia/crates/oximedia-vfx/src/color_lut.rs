#![allow(dead_code)]
//! Color look-up table (LUT) application for VFX.
//!
//! Provides 1-D and 3-D LUT structures that can be applied to RGBA frames
//! for colour grading, log-to-linear conversion, and creative looks.

/// A 1-D look-up table mapping a single channel through 256 entries.
#[derive(Debug, Clone)]
pub struct Lut1D {
    /// Red channel curve (256 entries, u8).
    pub red: Vec<u8>,
    /// Green channel curve (256 entries, u8).
    pub green: Vec<u8>,
    /// Blue channel curve (256 entries, u8).
    pub blue: Vec<u8>,
}

impl Lut1D {
    /// Create an identity (pass-through) 1-D LUT.
    pub fn identity() -> Self {
        let curve: Vec<u8> = (0..=255).map(|i| i as u8).collect();
        Self {
            red: curve.clone(),
            green: curve.clone(),
            blue: curve,
        }
    }

    /// Create a 1-D LUT from a brightness / contrast adjustment.
    ///
    /// `brightness` is added to each value, `contrast` scales around midpoint.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_brightness_contrast(brightness: f32, contrast: f32) -> Self {
        let build = |_channel: usize| -> Vec<u8> {
            (0..=255u16)
                .map(|i| {
                    let v = i as f32 / 255.0;
                    let v = (v - 0.5) * contrast + 0.5 + brightness;
                    (v.clamp(0.0, 1.0) * 255.0) as u8
                })
                .collect()
        };
        Self {
            red: build(0),
            green: build(1),
            blue: build(2),
        }
    }

    /// Create a gamma correction LUT.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_gamma(gamma: f32) -> Self {
        let inv = if gamma.abs() > 1e-6 { 1.0 / gamma } else { 1.0 };
        let curve: Vec<u8> = (0..=255u16)
            .map(|i| {
                let v = (i as f32 / 255.0).powf(inv);
                (v.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect();
        Self {
            red: curve.clone(),
            green: curve.clone(),
            blue: curve,
        }
    }

    /// Create an inverted (negative) LUT.
    pub fn invert() -> Self {
        let curve: Vec<u8> = (0..=255u16).map(|i| (255 - i) as u8).collect();
        Self {
            red: curve.clone(),
            green: curve.clone(),
            blue: curve,
        }
    }

    /// Apply this 1-D LUT to a pixel.
    pub fn apply_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        (
            self.red[r as usize],
            self.green[g as usize],
            self.blue[b as usize],
        )
    }

    /// Apply this 1-D LUT to an RGBA buffer in place. Alpha is preserved.
    pub fn apply_buffer(&self, data: &mut [u8]) {
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = self.red[chunk[0] as usize];
            chunk[1] = self.green[chunk[1] as usize];
            chunk[2] = self.blue[chunk[2] as usize];
        }
    }

    /// Compose two 1-D LUTs (apply `self` then `other`).
    pub fn compose(&self, other: &Self) -> Self {
        let red: Vec<u8> = self.red.iter().map(|&v| other.red[v as usize]).collect();
        let green: Vec<u8> = self
            .green
            .iter()
            .map(|&v| other.green[v as usize])
            .collect();
        let blue: Vec<u8> = self.blue.iter().map(|&v| other.blue[v as usize]).collect();
        Self { red, green, blue }
    }
}

/// A 3-D colour look-up table.
///
/// The table maps (R, G, B) triplets through a cube of size `size x size x size`.
/// Each entry is an (R, G, B) output triplet stored as `[u8; 3]`.
#[derive(Debug, Clone)]
pub struct Lut3D {
    /// Cube side length.
    size: usize,
    /// Flattened table of `[r, g, b]` entries, length `size^3`.
    data: Vec<[u8; 3]>,
}

impl Lut3D {
    /// Create an identity 3-D LUT of the given size (typically 17 or 33).
    #[allow(clippy::cast_precision_loss)]
    pub fn identity(size: usize) -> Self {
        let size = size.max(2);
        let n = size * size * size;
        let mut data = Vec::with_capacity(n);
        for b_idx in 0..size {
            for g_idx in 0..size {
                for r_idx in 0..size {
                    let r = (r_idx as f32 / (size - 1) as f32 * 255.0) as u8;
                    let g = (g_idx as f32 / (size - 1) as f32 * 255.0) as u8;
                    let b = (b_idx as f32 / (size - 1) as f32 * 255.0) as u8;
                    data.push([r, g, b]);
                }
            }
        }
        Self { size, data }
    }

    /// Get the cube side length.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get entry count.
    pub fn entry_count(&self) -> usize {
        self.data.len()
    }

    /// Index into the 3-D table.
    fn index(&self, ri: usize, gi: usize, bi: usize) -> &[u8; 3] {
        &self.data[bi * self.size * self.size + gi * self.size + ri]
    }

    /// Look up a colour with trilinear interpolation.
    #[allow(clippy::cast_precision_loss)]
    pub fn lookup(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let max_idx = (self.size - 1) as f32;

        let rf = r as f32 / 255.0 * max_idx;
        let gf = g as f32 / 255.0 * max_idx;
        let bf = b as f32 / 255.0 * max_idx;

        let r0 = (rf.floor() as usize).min(self.size - 2);
        let g0 = (gf.floor() as usize).min(self.size - 2);
        let b0 = (bf.floor() as usize).min(self.size - 2);

        let rd = rf - r0 as f32;
        let gd = gf - g0 as f32;
        let bd = bf - b0 as f32;

        // Trilinear interpolation
        let c000 = self.index(r0, g0, b0);
        let c100 = self.index(r0 + 1, g0, b0);
        let c010 = self.index(r0, g0 + 1, b0);
        let c110 = self.index(r0 + 1, g0 + 1, b0);
        let c001 = self.index(r0, g0, b0 + 1);
        let c101 = self.index(r0 + 1, g0, b0 + 1);
        let c011 = self.index(r0, g0 + 1, b0 + 1);
        let c111 = self.index(r0 + 1, g0 + 1, b0 + 1);

        let mut out = [0u8; 3];
        for ch in 0..3 {
            let v000 = c000[ch] as f32;
            let v100 = c100[ch] as f32;
            let v010 = c010[ch] as f32;
            let v110 = c110[ch] as f32;
            let v001 = c001[ch] as f32;
            let v101 = c101[ch] as f32;
            let v011 = c011[ch] as f32;
            let v111 = c111[ch] as f32;

            let c00 = v000 * (1.0 - rd) + v100 * rd;
            let c10 = v010 * (1.0 - rd) + v110 * rd;
            let c01 = v001 * (1.0 - rd) + v101 * rd;
            let c11 = v011 * (1.0 - rd) + v111 * rd;

            let c0 = c00 * (1.0 - gd) + c10 * gd;
            let c1 = c01 * (1.0 - gd) + c11 * gd;

            let val = c0 * (1.0 - bd) + c1 * bd;
            out[ch] = val.clamp(0.0, 255.0) as u8;
        }

        (out[0], out[1], out[2])
    }

    /// Apply this 3-D LUT to an RGBA buffer in place.
    pub fn apply_buffer(&self, data: &mut [u8]) {
        for chunk in data.chunks_exact_mut(4) {
            let (r, g, b) = self.lookup(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
    }
}

/// Blend mode for combining the LUT result with the original.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutBlendMode {
    /// Full replacement.
    Replace,
    /// Linear interpolation with the original.
    Mix,
}

/// A LUT applicator that wraps either a 1-D or 3-D LUT with blending.
#[derive(Debug, Clone)]
pub struct LutApplicator {
    /// Blend strength (0.0 = original, 1.0 = full LUT).
    pub strength: f32,
    /// Blend mode.
    pub mode: LutBlendMode,
    /// Inner LUT kind.
    kind: LutKind,
}

/// Inner enum for holding either 1-D or 3-D LUT.
#[derive(Debug, Clone)]
enum LutKind {
    /// 1-D channel curves.
    OneDim(Lut1D),
    /// 3-D colour cube.
    ThreeDim(Lut3D),
}

impl LutApplicator {
    /// Create an applicator from a 1-D LUT.
    pub fn from_1d(lut: Lut1D) -> Self {
        Self {
            strength: 1.0,
            mode: LutBlendMode::Replace,
            kind: LutKind::OneDim(lut),
        }
    }

    /// Create an applicator from a 3-D LUT.
    pub fn from_3d(lut: Lut3D) -> Self {
        Self {
            strength: 1.0,
            mode: LutBlendMode::Replace,
            kind: LutKind::ThreeDim(lut),
        }
    }

    /// Set the blend strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the blend mode.
    pub fn with_mode(mut self, mode: LutBlendMode) -> Self {
        self.mode = mode;
        self
    }

    /// Apply the LUT to an RGBA buffer in place with blending.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_buffer(&self, data: &mut [u8]) {
        if self.strength <= 0.0 {
            return;
        }
        let full = self.strength >= 1.0 && self.mode == LutBlendMode::Replace;

        for chunk in data.chunks_exact_mut(4) {
            let (orig_r, orig_g, orig_b) = (chunk[0], chunk[1], chunk[2]);
            let (lr, lg, lb) = match &self.kind {
                LutKind::OneDim(lut) => lut.apply_pixel(orig_r, orig_g, orig_b),
                LutKind::ThreeDim(lut) => lut.lookup(orig_r, orig_g, orig_b),
            };

            if full {
                chunk[0] = lr;
                chunk[1] = lg;
                chunk[2] = lb;
            } else {
                let s = self.strength;
                let inv = 1.0 - s;
                chunk[0] = (orig_r as f32 * inv + lr as f32 * s).clamp(0.0, 255.0) as u8;
                chunk[1] = (orig_g as f32 * inv + lg as f32 * s).clamp(0.0, 255.0) as u8;
                chunk[2] = (orig_b as f32 * inv + lb as f32 * s).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut1d_identity() {
        let lut = Lut1D::identity();
        assert_eq!(lut.red.len(), 256);
        for i in 0..=255u8 {
            assert_eq!(lut.red[i as usize], i);
            assert_eq!(lut.green[i as usize], i);
            assert_eq!(lut.blue[i as usize], i);
        }
    }

    #[test]
    fn test_lut1d_apply_pixel_identity() {
        let lut = Lut1D::identity();
        assert_eq!(lut.apply_pixel(128, 64, 200), (128, 64, 200));
    }

    #[test]
    fn test_lut1d_invert() {
        let lut = Lut1D::invert();
        assert_eq!(lut.apply_pixel(0, 0, 0), (255, 255, 255));
        assert_eq!(lut.apply_pixel(255, 255, 255), (0, 0, 0));
    }

    #[test]
    fn test_lut1d_gamma() {
        let lut = Lut1D::from_gamma(2.2);
        // Black stays black, white stays white
        assert_eq!(lut.apply_pixel(0, 0, 0), (0, 0, 0));
        assert_eq!(lut.apply_pixel(255, 255, 255), (255, 255, 255));
        // Mid-grey should be shifted
        let (r, _g, _b) = lut.apply_pixel(128, 128, 128);
        assert_ne!(r, 128);
    }

    #[test]
    fn test_lut1d_brightness_contrast() {
        let lut = Lut1D::from_brightness_contrast(0.1, 1.0);
        // Values should be shifted up
        let (r, _, _) = lut.apply_pixel(100, 100, 100);
        assert!(r > 100);
    }

    #[test]
    fn test_lut1d_compose() {
        let invert = Lut1D::invert();
        let double_invert = invert.compose(&Lut1D::invert());
        // Double inversion should yield identity
        for i in 0..=255u8 {
            assert_eq!(double_invert.red[i as usize], i);
        }
    }

    #[test]
    fn test_lut1d_apply_buffer() {
        let lut = Lut1D::invert();
        let mut buf = vec![100, 150, 200, 255]; // RGBA
        lut.apply_buffer(&mut buf);
        assert_eq!(buf[0], 155);
        assert_eq!(buf[1], 105);
        assert_eq!(buf[2], 55);
        assert_eq!(buf[3], 255); // alpha unchanged
    }

    #[test]
    fn test_lut3d_identity_corners() {
        let lut = Lut3D::identity(17);
        assert_eq!(lut.size(), 17);
        // Black corner
        let (r, g, b) = lut.lookup(0, 0, 0);
        assert_eq!((r, g, b), (0, 0, 0));
        // White corner
        let (r, g, b) = lut.lookup(255, 255, 255);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_lut3d_identity_passthrough() {
        let lut = Lut3D::identity(33);
        // Arbitrary color should be approximately preserved
        let (r, g, b) = lut.lookup(128, 64, 200);
        assert!((r as i16 - 128).unsigned_abs() <= 1);
        assert!((g as i16 - 64).unsigned_abs() <= 1);
        assert!((b as i16 - 200).unsigned_abs() <= 1);
    }

    #[test]
    fn test_lut3d_entry_count() {
        let lut = Lut3D::identity(17);
        assert_eq!(lut.entry_count(), 17 * 17 * 17);
    }

    #[test]
    fn test_lut3d_apply_buffer() {
        let lut = Lut3D::identity(17);
        let mut buf = vec![100, 150, 200, 255];
        lut.apply_buffer(&mut buf);
        // Identity should approximately preserve values
        assert!((buf[0] as i16 - 100).unsigned_abs() <= 1);
        assert!((buf[1] as i16 - 150).unsigned_abs() <= 1);
        assert!((buf[2] as i16 - 200).unsigned_abs() <= 1);
        assert_eq!(buf[3], 255);
    }

    #[test]
    fn test_applicator_1d_full_strength() {
        let lut = Lut1D::invert();
        let app = LutApplicator::from_1d(lut);
        let mut buf = vec![100, 100, 100, 255];
        app.apply_buffer(&mut buf);
        assert_eq!(buf[0], 155);
    }

    #[test]
    fn test_applicator_zero_strength() {
        let lut = Lut1D::invert();
        let app = LutApplicator::from_1d(lut).with_strength(0.0);
        let mut buf = vec![100, 100, 100, 255];
        app.apply_buffer(&mut buf);
        assert_eq!(buf[0], 100); // unchanged
    }

    #[test]
    fn test_applicator_half_strength() {
        let lut = Lut1D::invert();
        let app = LutApplicator::from_1d(lut)
            .with_strength(0.5)
            .with_mode(LutBlendMode::Mix);
        let mut buf = vec![100, 100, 100, 255];
        app.apply_buffer(&mut buf);
        // Should be roughly midway between 100 and 155
        assert!(buf[0] > 110 && buf[0] < 145);
    }

    #[test]
    fn test_applicator_3d() {
        let lut = Lut3D::identity(17);
        let app = LutApplicator::from_3d(lut);
        let mut buf = vec![50, 100, 200, 255];
        app.apply_buffer(&mut buf);
        assert!((buf[0] as i16 - 50).unsigned_abs() <= 1);
    }
}
