//! Hald CLUT format — identity and custom LUT encoded as an image.
//!
//! A Hald CLUT stores a full 3D colour cube as a flat image. Level N creates
//! a cube of size N² (so level 8 → 64³ = 262 144 entries, stored as a
//! 512 × 512 image).  The data layout is sequential: R varies fastest, then G,
//! then B (inner → outer).

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

/// Raw 3-D LUT data (f32 precision, `[r; g; b]` triplets).
///
/// Flat indexing: `index = r + g * size + b * size * size`
#[derive(Debug, Clone)]
pub struct Lut3DData {
    /// Number of lattice divisions per axis.
    pub size: usize,
    /// Flat `size³` array of `[r, g, b]` triplets (values in `[0, 1]`).
    pub data: Vec<[f32; 3]>,
}

impl Lut3DData {
    /// Create an identity 3-D LUT of the given `size`.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let mut data = Vec::with_capacity(size * size * size);
        let scale = (size - 1) as f32;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push([r as f32 / scale, g as f32 / scale, b as f32 / scale]);
                }
            }
        }
        Self { size, data }
    }

    /// Trilinear lookup for a normalised `(r, g, b)` coordinate.
    #[must_use]
    pub fn lookup(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let size = self.size;
        if size < 2 {
            return [r, g, b];
        }
        let scale = (size - 1) as f32;
        let rv = (r.clamp(0.0, 1.0) * scale).min(scale);
        let gv = (g.clamp(0.0, 1.0) * scale).min(scale);
        let bv = (b.clamp(0.0, 1.0) * scale).min(scale);

        let r0 = rv.floor() as usize;
        let g0 = gv.floor() as usize;
        let b0 = bv.floor() as usize;
        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        let tr = rv - r0 as f32;
        let tg = gv - g0 as f32;
        let tb = bv - b0 as f32;

        let idx = |ri: usize, gi: usize, bi: usize| ri + gi * size + bi * size * size;

        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        let mut out = [0.0_f32; 3];
        for ch in 0..3 {
            out[ch] = c000[ch] * (1.0 - tr) * (1.0 - tg) * (1.0 - tb)
                + c100[ch] * tr * (1.0 - tg) * (1.0 - tb)
                + c010[ch] * (1.0 - tr) * tg * (1.0 - tb)
                + c110[ch] * tr * tg * (1.0 - tb)
                + c001[ch] * (1.0 - tr) * (1.0 - tg) * tb
                + c101[ch] * tr * (1.0 - tg) * tb
                + c011[ch] * (1.0 - tr) * tg * tb
                + c111[ch] * tr * tg * tb;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// HaldClut
// ---------------------------------------------------------------------------

/// Hald CLUT — a 3-D colour cube stored as a flat image.
///
/// Level `L` defines `size = L²` divisions per axis.  The total entry count
/// is `size³`.  The image dimensions are `(size * L) × (size * L)` pixels, or
/// equivalently `size² × size` for a square image — but here we store the
/// cube directly as `Vec<[f32; 3]>`.
#[derive(Debug, Clone)]
pub struct HaldClut {
    /// Hald level (e.g. 8 → 512 × 512 image).
    pub level: u32,
    /// Cube size per axis (`level²`).
    pub size: u32,
    /// `size³` RGB triplets (values in `[0, 1]`).
    pub data: Vec<[f32; 3]>,
}

impl HaldClut {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create an identity Hald CLUT for the given `level`.
    ///
    /// * Level 8 → `size = 64`, 262 144 entries.
    /// * Level 12 → `size = 144`, 2 985 984 entries.
    #[must_use]
    pub fn identity(level: u32) -> Self {
        let size = level * level;
        let n = size as usize;
        let scale = (n - 1) as f32;
        let mut data = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    data.push([r as f32 / scale, g as f32 / scale, b as f32 / scale]);
                }
            }
        }
        Self { level, size, data }
    }

    /// Build a Hald CLUT from raw 8-bit RGB pixel data.
    ///
    /// `pixels` must contain exactly `size³ * 3` bytes (R, G, B interleaved).
    /// Returns `None` if the length does not match.
    #[must_use]
    pub fn from_raw_rgb(pixels: &[u8], level: u32) -> Option<Self> {
        let size = level * level;
        let n = size as usize;
        let expected = n * n * n * 3;
        if pixels.len() != expected {
            return None;
        }
        let data: Vec<[f32; 3]> = pixels
            .chunks_exact(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect();
        Some(Self { level, size, data })
    }

    /// Convert to raw 8-bit RGB bytes (multiply each channel by 255 and round).
    #[must_use]
    pub fn to_raw_rgb(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 3);
        for &[r, g, b] in &self.data {
            out.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
            out.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        out
    }

    /// Convert from a [`Lut3DData`].
    ///
    /// The source LUT's `size` is used to derive the Hald level by finding the
    /// nearest integer square root so that `level² ≈ lut_size`.  If the sizes
    /// do not match exactly the LUT is resampled via trilinear interpolation.
    #[must_use]
    pub fn from_lut3d(lut: &Lut3DData) -> Self {
        // Find the Hald level whose cube size is closest to the LUT's size.
        let lut_size = lut.size;
        let level = (lut_size as f32).sqrt().round() as u32;
        let level = level.max(2);
        let size = level * level;
        let n = size as usize;
        let scale = (n - 1) as f32;

        let mut data = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let rf = r as f32 / scale;
                    let gf = g as f32 / scale;
                    let bf = b as f32 / scale;
                    data.push(lut.lookup(rf, gf, bf));
                }
            }
        }
        Self { level, size, data }
    }

    /// Parse an Adobe `.cube` text and convert to a Hald CLUT of the given `level`.
    ///
    /// Returns `Err` with a description if the cube text is invalid.
    pub fn parse_cube_to_hald(text: &str, level: u32) -> Result<Self, String> {
        let mut cube_size: Option<usize> = None;
        let mut entries: Vec<[f32; 3]> = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                let s = rest
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid LUT_3D_SIZE: {e}"))?;
                cube_size = Some(s);
                continue;
            }
            // Skip TITLE / DOMAIN_* keywords
            if line.starts_with("TITLE")
                || line.starts_with("DOMAIN_MIN")
                || line.starts_with("DOMAIN_MAX")
                || line.starts_with("LUT_1D_SIZE")
            {
                continue;
            }
            let mut parts = line.split_ascii_whitespace();
            let parse_next = |p: &mut std::str::SplitAsciiWhitespace<'_>| {
                p.next()
                    .ok_or_else(|| "Missing value".to_string())
                    .and_then(|s| s.parse::<f32>().map_err(|e| format!("Parse error: {e}")))
            };
            if let (Ok(r), Ok(g), Ok(b)) = (
                parse_next(&mut parts),
                parse_next(&mut parts),
                parse_next(&mut parts),
            ) {
                entries.push([r, g, b]);
            }
        }

        let cube_size = cube_size.ok_or_else(|| "Missing LUT_3D_SIZE".to_string())?;
        let expected = cube_size * cube_size * cube_size;
        if entries.len() != expected {
            return Err(format!(
                "Expected {expected} entries but found {}",
                entries.len()
            ));
        }

        // The .cube format is B-major (B varies slowest, R fastest).
        // Reindex to our R-major layout: index = r + g*size + b*size²
        let src_lut = Lut3DData {
            size: cube_size,
            data: {
                // cube entries: index in file = b*size² + g*size + r → maps to r,g,b
                // We need r-major storage: out[r + g*size + b*size²] = entries[b*size² + g*size + r]
                let mut cube_rmajor = vec![[0.0_f32; 3]; expected];
                for bi in 0..cube_size {
                    for gi in 0..cube_size {
                        for ri in 0..cube_size {
                            let src_idx = bi * cube_size * cube_size + gi * cube_size + ri;
                            let dst_idx = ri + gi * cube_size + bi * cube_size * cube_size;
                            cube_rmajor[dst_idx] = entries[src_idx];
                        }
                    }
                }
                cube_rmajor
            },
        };

        Ok(Self::from_lut3d_with_level(&src_lut, level))
    }

    // -----------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------

    /// Apply the Hald CLUT to a single `(r, g, b)` pixel via trilinear interpolation.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let size = self.size as usize;
        if size < 2 {
            return (r, g, b);
        }
        let scale = (size - 1) as f32;
        let rv = (r.clamp(0.0, 1.0) * scale).min(scale);
        let gv = (g.clamp(0.0, 1.0) * scale).min(scale);
        let bv = (b.clamp(0.0, 1.0) * scale).min(scale);

        let r0 = rv.floor() as usize;
        let g0 = gv.floor() as usize;
        let b0 = bv.floor() as usize;
        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        let tr = rv - r0 as f32;
        let tg = gv - g0 as f32;
        let tb = bv - b0 as f32;

        let idx = |ri: usize, gi: usize, bi: usize| ri + gi * size + bi * size * size;

        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        let interp = |c000v: f32,
                      c100v: f32,
                      c010v: f32,
                      c110v: f32,
                      c001v: f32,
                      c101v: f32,
                      c011v: f32,
                      c111v: f32| {
            c000v * (1.0 - tr) * (1.0 - tg) * (1.0 - tb)
                + c100v * tr * (1.0 - tg) * (1.0 - tb)
                + c010v * (1.0 - tr) * tg * (1.0 - tb)
                + c110v * tr * tg * (1.0 - tb)
                + c001v * (1.0 - tr) * (1.0 - tg) * tb
                + c101v * tr * (1.0 - tg) * tb
                + c011v * (1.0 - tr) * tg * tb
                + c111v * tr * tg * tb
        };

        (
            interp(
                c000[0], c100[0], c010[0], c110[0], c001[0], c101[0], c011[0], c111[0],
            ),
            interp(
                c000[1], c100[1], c010[1], c110[1], c001[1], c101[1], c011[1], c111[1],
            ),
            interp(
                c000[2], c100[2], c010[2], c110[2], c001[2], c101[2], c011[2], c111[2],
            ),
        )
    }

    /// Apply the Hald CLUT to an RGB-interleaved `f32` frame in-place.
    ///
    /// `pixels` is expected to be `[r0, g0, b0, r1, g1, b1, …]`.
    /// The length must be a multiple of 3; extra trailing values are ignored.
    #[must_use]
    pub fn apply_frame(&self, pixels: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pixels.len());
        let mut i = 0;
        while i + 2 < pixels.len() {
            let (r, g, b) = self.apply(pixels[i], pixels[i + 1], pixels[i + 2]);
            out.push(r);
            out.push(g);
            out.push(b);
            i += 3;
        }
        // Handle any trailing values that don't form a complete triplet
        while i < pixels.len() {
            out.push(pixels[i]);
            i += 1;
        }
        out
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Like [`from_lut3d`] but forces a specific `level`.
    fn from_lut3d_with_level(lut: &Lut3DData, level: u32) -> Self {
        let size = level * level;
        let n = size as usize;
        let scale = if n > 1 { (n - 1) as f32 } else { 1.0 };

        let mut data = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let rf = r as f32 / scale;
                    let gf = g as f32 / scale;
                    let bf = b as f32 / scale;
                    data.push(lut.lookup(rf, gf, bf));
                }
            }
        }
        Self { level, size, data }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_level8_entry_count() {
        let h = HaldClut::identity(8);
        assert_eq!(h.size, 64);
        assert_eq!(h.data.len(), 64 * 64 * 64);
    }

    #[test]
    fn test_identity_level4_spot_check() {
        let h = HaldClut::identity(4);
        let size = h.size as usize;
        let scale = (size - 1) as f32;
        // First entry should be (0,0,0)
        assert!((h.data[0][0]).abs() < 1e-5);
        // Last entry should be (1,1,1)
        let last = h.data[size * size * size - 1];
        assert!((last[0] - 1.0).abs() < 1e-5);
        // Middle red channel
        let mid = size / 2;
        let idx = mid + 0 * size + 0 * size * size;
        assert!((h.data[idx][0] - mid as f32 / scale).abs() < 1e-5);
    }

    #[test]
    fn test_apply_identity_passthrough() {
        let h = HaldClut::identity(4);
        let (r, g, b) = h.apply(0.5, 0.3, 0.8);
        assert!((r - 0.5).abs() < 0.02, "r={r}");
        assert!((g - 0.3).abs() < 0.02, "g={g}");
        assert!((b - 0.8).abs() < 0.02, "b={b}");
    }

    #[test]
    fn test_apply_clamps_out_of_range() {
        let h = HaldClut::identity(4);
        let (r, g, b) = h.apply(-0.1, 1.5, 0.5);
        assert!(r >= 0.0 && r <= 1.0, "r={r}");
        assert!(g >= 0.0 && g <= 1.0, "g={g}");
        assert!(b >= 0.0 && b <= 1.0, "b={b}");
    }

    #[test]
    fn test_to_raw_rgb_roundtrip() {
        let h = HaldClut::identity(2);
        let raw = h.to_raw_rgb();
        assert_eq!(raw.len(), h.data.len() * 3);
        // Reconstruct
        let h2 = HaldClut::from_raw_rgb(&raw, 2).expect("should reconstruct");
        for (a, b) in h.data.iter().zip(h2.data.iter()) {
            assert!((a[0] - b[0]).abs() < 0.005, "r mismatch {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_from_raw_rgb_wrong_length() {
        let result = HaldClut::from_raw_rgb(&[0u8; 10], 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_frame_identity() {
        let h = HaldClut::identity(4);
        let pixels = vec![0.2_f32, 0.4, 0.6, 0.8, 0.1, 0.3];
        let out = h.apply_frame(&pixels);
        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.2).abs() < 0.02, "out[0]={}", out[0]);
        assert!((out[3] - 0.8).abs() < 0.02, "out[3]={}", out[3]);
    }

    #[test]
    fn test_from_lut3d_identity() {
        let lut = Lut3DData::identity(16);
        let h = HaldClut::from_lut3d(&lut);
        // apply should pass through near-identity
        let (r, g, b) = h.apply(0.5, 0.5, 0.5);
        assert!((r - 0.5).abs() < 0.05, "r={r}");
        assert!((g - 0.5).abs() < 0.05, "g={g}");
        assert!((b - 0.5).abs() < 0.05, "b={b}");
    }

    #[test]
    fn test_parse_cube_to_hald_identity() {
        // Build a small 2-point cube (identity)
        let size = 2_usize;
        let mut cube = format!("LUT_3D_SIZE {size}\n");
        for bi in 0..size {
            for gi in 0..size {
                for ri in 0..size {
                    cube.push_str(&format!(
                        "{} {} {}\n",
                        ri as f32 / (size - 1) as f32,
                        gi as f32 / (size - 1) as f32,
                        bi as f32 / (size - 1) as f32
                    ));
                }
            }
        }
        let h = HaldClut::parse_cube_to_hald(&cube, 2).expect("parse should succeed");
        assert_eq!(h.level, 2);
        let (r, g, b) = h.apply(0.5, 0.5, 0.5);
        assert!((r - 0.5).abs() < 0.1, "r={r}");
        assert!((g - 0.5).abs() < 0.1, "g={g}");
        assert!((b - 0.5).abs() < 0.1, "b={b}");
    }

    #[test]
    fn test_parse_cube_missing_size() {
        let result = HaldClut::parse_cube_to_hald("0.5 0.5 0.5\n", 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_lut3d_data_identity_lookup_corners() {
        let lut = Lut3DData::identity(8);
        let black = lut.lookup(0.0, 0.0, 0.0);
        assert!((black[0]).abs() < 1e-5);
        let white = lut.lookup(1.0, 1.0, 1.0);
        assert!((white[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hald_identity_level2() {
        let h = HaldClut::identity(2);
        assert_eq!(h.size, 4);
        assert_eq!(h.data.len(), 64); // 4³
    }

    #[test]
    fn test_apply_frame_empty() {
        let h = HaldClut::identity(4);
        let out = h.apply_frame(&[]);
        assert!(out.is_empty());
    }
}
