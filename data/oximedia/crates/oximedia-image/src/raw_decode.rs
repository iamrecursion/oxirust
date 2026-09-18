//! RAW image decoding: demosaicing, white balance, and metadata extraction.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::collections::HashMap;

// ── Bayer pattern ─────────────────────────────────────────────────────────────

/// Bayer colour filter array layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BayerPattern {
    /// RGGB – red top-left.
    Rggb,
    /// BGGR – blue top-left.
    Bggr,
    /// GRBG – green-red top row.
    Grbg,
    /// GBRG – green-blue top row.
    Gbrg,
}

impl BayerPattern {
    /// Returns the (r, g1, g2, b) offsets within a 2×2 quad (row-major).
    #[must_use]
    pub const fn quad_offsets(&self) -> (usize, usize, usize, usize) {
        match self {
            Self::Rggb => (0, 1, 2, 3), // R G / G B
            Self::Bggr => (3, 2, 1, 0), // B G / G R  (r at [3])
            Self::Grbg => (1, 0, 3, 2), // G R / B G
            Self::Gbrg => (2, 3, 0, 1), // G B / R G
        }
    }
}

// ── White balance ─────────────────────────────────────────────────────────────

/// Per-channel white-balance multipliers (R, G, B).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteBalance {
    /// Red channel multiplier.
    pub red: f32,
    /// Green channel multiplier.
    pub green: f32,
    /// Blue channel multiplier.
    pub blue: f32,
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
        }
    }
}

impl WhiteBalance {
    /// Daylight (~5500 K) preset.
    #[must_use]
    pub fn daylight() -> Self {
        Self {
            red: 2.0,
            green: 1.0,
            blue: 1.5,
        }
    }

    /// Tungsten (~3200 K) preset.
    #[must_use]
    pub fn tungsten() -> Self {
        Self {
            red: 2.5,
            green: 1.0,
            blue: 1.1,
        }
    }

    /// Apply multipliers to a linear RGB triplet.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        (r * self.red, g * self.green, b * self.blue)
    }

    /// Normalise multipliers so that green == 1.0.
    #[must_use]
    pub fn normalised(&self) -> Self {
        let inv = 1.0 / self.green;
        Self {
            red: self.red * inv,
            green: 1.0,
            blue: self.blue * inv,
        }
    }
}

// ── RAW metadata ──────────────────────────────────────────────────────────────

/// Metadata extracted from a RAW image header.
#[derive(Clone, Debug, Default)]
pub struct RawMetadata {
    /// Camera make (e.g. "Canon").
    pub make: Option<String>,
    /// Camera model (e.g. "EOS R5").
    pub model: Option<String>,
    /// ISO sensitivity.
    pub iso: Option<u32>,
    /// Shutter speed in seconds.
    pub shutter_speed: Option<f32>,
    /// Aperture f-number.
    pub aperture: Option<f32>,
    /// Focal length in mm.
    pub focal_length_mm: Option<f32>,
    /// Arbitrary key-value tags.
    pub tags: HashMap<String, String>,
}

impl RawMetadata {
    /// Create an empty metadata record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an arbitrary tag.
    pub fn insert_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

// ── Demosaicing ───────────────────────────────────────────────────────────────

/// Bilinear demosaic of a 16-bit Bayer mosaic.
///
/// Returns an interleaved RGB buffer with dimensions `width × height`.
/// Each pixel is represented as three consecutive `u16` values.
///
/// # Panics
///
/// Panics if `data.len() != width * height`.
#[must_use]
pub fn demosaic_bilinear(
    data: &[u16],
    width: usize,
    height: usize,
    pattern: BayerPattern,
) -> Vec<u16> {
    assert_eq!(data.len(), width * height, "data length mismatch");

    let mut rgb = vec![0u16; width * height * 3];

    // Simple nearest-neighbour copy for single-pixel border; bilinear interior.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            // Determine which channel this pixel belongs to.
            let (quad_r, quad_g1, quad_g2, quad_b) = pattern.quad_offsets();
            let quad_pos = (y % 2) * 2 + (x % 2);

            let (r, g, b) = if quad_pos == quad_r {
                let rval = data[idx];
                let gval = average_neighbours(data, x, y, width, height);
                let bval = average_diagonal(data, x, y, width, height);
                (rval, gval, bval)
            } else if quad_pos == quad_g1 || quad_pos == quad_g2 {
                let gval = data[idx];
                let rval = average_axis(data, x, y, width, height, true);
                let bval = average_axis(data, x, y, width, height, false);
                (rval, gval, bval)
            } else {
                debug_assert_eq!(quad_pos, quad_b);
                let bval = data[idx];
                let gval = average_neighbours(data, x, y, width, height);
                let rval = average_diagonal(data, x, y, width, height);
                (rval, gval, bval)
            };

            rgb[idx * 3] = r;
            rgb[idx * 3 + 1] = g;
            rgb[idx * 3 + 2] = b;
        }
    }

    rgb
}

fn clamp_coord(v: isize, max: usize) -> usize {
    v.clamp(0, max as isize - 1) as usize
}

fn sample(data: &[u16], x: usize, y: usize, width: usize) -> u32 {
    u32::from(data[y * width + x])
}

fn average_neighbours(data: &[u16], x: usize, y: usize, width: usize, height: usize) -> u16 {
    let xi = x as isize;
    let yi = y as isize;
    let sum: u32 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        .iter()
        .map(|(dx, dy)| {
            let nx = clamp_coord(xi + dx, width);
            let ny = clamp_coord(yi + dy, height);
            sample(data, nx, ny, width)
        })
        .sum();
    (sum / 4) as u16
}

fn average_diagonal(data: &[u16], x: usize, y: usize, width: usize, height: usize) -> u16 {
    let xi = x as isize;
    let yi = y as isize;
    let sum: u32 = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        .iter()
        .map(|(dx, dy)| {
            let nx = clamp_coord(xi + dx, width);
            let ny = clamp_coord(yi + dy, height);
            sample(data, nx, ny, width)
        })
        .sum();
    (sum / 4) as u16
}

fn average_axis(
    data: &[u16],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    horizontal: bool,
) -> u16 {
    let xi = x as isize;
    let yi = y as isize;
    let (a, b) = if horizontal {
        ((-1, 0), (1, 0))
    } else {
        ((0, -1), (0, 1))
    };
    let na_x = clamp_coord(xi + a.0, width);
    let na_y = clamp_coord(yi + a.1, height);
    let nb_x = clamp_coord(xi + b.0, width);
    let nb_y = clamp_coord(yi + b.1, height);
    let sum = sample(data, na_x, na_y, width) + sample(data, nb_x, nb_y, width);
    (sum / 2) as u16
}

// ── Apply white balance to RGB buffer ────────────────────────────────────────

/// Apply white-balance multipliers to an interleaved u16 RGB buffer in-place.
pub fn apply_white_balance_u16(rgb: &mut [u16], wb: &WhiteBalance) {
    let nb = wb.normalised();
    for chunk in rgb.chunks_exact_mut(3) {
        let r = (chunk[0] as f32 * nb.red).min(65535.0) as u16;
        let g = (chunk[1] as f32 * nb.green).min(65535.0) as u16;
        let b = (chunk[2] as f32 * nb.blue).min(65535.0) as u16;
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayer_pattern_quad_offsets_rggb() {
        let (r, g1, g2, b) = BayerPattern::Rggb.quad_offsets();
        assert_eq!(r, 0);
        assert_eq!(g1, 1);
        assert_eq!(g2, 2);
        assert_eq!(b, 3);
    }

    #[test]
    fn test_bayer_pattern_quad_offsets_bggr() {
        let (r, _g1, _g2, b) = BayerPattern::Bggr.quad_offsets();
        assert_eq!(b, 0);
        assert_eq!(r, 3);
    }

    #[test]
    fn test_white_balance_default() {
        let wb = WhiteBalance::default();
        assert_eq!(wb.red, 1.0);
        assert_eq!(wb.green, 1.0);
        assert_eq!(wb.blue, 1.0);
    }

    #[test]
    fn test_white_balance_apply() {
        let wb = WhiteBalance {
            red: 2.0,
            green: 1.0,
            blue: 0.5,
        };
        let (r, g, b) = wb.apply(1.0, 1.0, 1.0);
        assert!((r - 2.0).abs() < 1e-6);
        assert!((g - 1.0).abs() < 1e-6);
        assert!((b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_white_balance_normalised() {
        let wb = WhiteBalance {
            red: 2.0,
            green: 2.0,
            blue: 1.0,
        };
        let nb = wb.normalised();
        assert!((nb.green - 1.0).abs() < 1e-6);
        assert!((nb.red - 1.0).abs() < 1e-6);
        assert!((nb.blue - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_white_balance_daylight() {
        let wb = WhiteBalance::daylight();
        assert!(wb.red > 1.0);
    }

    #[test]
    fn test_white_balance_tungsten() {
        let wb = WhiteBalance::tungsten();
        assert!(wb.red > wb.blue);
    }

    #[test]
    fn test_raw_metadata_new() {
        let m = RawMetadata::new();
        assert!(m.make.is_none());
        assert!(m.tags.is_empty());
    }

    #[test]
    fn test_raw_metadata_insert_tag() {
        let mut m = RawMetadata::new();
        m.insert_tag("Lens", "50mm f/1.4");
        assert_eq!(
            m.tags.get("Lens").expect("should succeed in test"),
            "50mm f/1.4"
        );
    }

    #[test]
    fn test_demosaic_bilinear_dimensions() {
        let w = 4;
        let h = 4;
        let data = vec![1000u16; w * h];
        let rgb = demosaic_bilinear(&data, w, h, BayerPattern::Rggb);
        assert_eq!(rgb.len(), w * h * 3);
    }

    #[test]
    fn test_demosaic_bilinear_uniform() {
        // Uniform mosaic → uniform output
        let w = 4;
        let h = 4;
        let data = vec![8000u16; w * h];
        let rgb = demosaic_bilinear(&data, w, h, BayerPattern::Grbg);
        for v in &rgb {
            assert!(*v > 0, "uniform mosaic should produce non-zero output");
        }
    }

    #[test]
    fn test_apply_white_balance_identity() {
        let mut rgb = vec![1000u16, 2000, 3000, 500, 600, 700];
        let orig = rgb.clone();
        let wb = WhiteBalance::default();
        apply_white_balance_u16(&mut rgb, &wb);
        assert_eq!(rgb, orig);
    }

    #[test]
    fn test_apply_white_balance_scales() {
        let mut rgb = vec![1000u16, 1000, 1000];
        let wb = WhiteBalance {
            red: 2.0,
            green: 1.0,
            blue: 0.5,
        };
        apply_white_balance_u16(&mut rgb, &wb);
        assert_eq!(rgb[0], 2000);
        assert_eq!(rgb[1], 1000);
        assert_eq!(rgb[2], 500);
    }

    #[test]
    fn test_apply_white_balance_clamping() {
        let mut rgb = vec![60000u16, 1000, 1000];
        let wb = WhiteBalance {
            red: 2.0,
            green: 1.0,
            blue: 1.0,
        };
        apply_white_balance_u16(&mut rgb, &wb);
        assert_eq!(rgb[0], 65535);
    }
}
