//! Color space conversion operations optimized for SIMD-friendly access patterns.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// An RGB pixel with floating-point components in the range [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbPixel {
    /// Red channel [0, 1]
    pub r: f32,
    /// Green channel [0, 1]
    pub g: f32,
    /// Blue channel [0, 1]
    pub b: f32,
}

impl RgbPixel {
    /// Create a new `RgbPixel`.
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Clamp all components to [0, 1].
    #[must_use]
    pub fn clamp(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Compute the luminance using Rec.709 coefficients:
    /// Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    #[must_use]
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

/// An HSV pixel.
///
/// - `h`: Hue in degrees [0, 360)
/// - `s`: Saturation [0, 1]
/// - `v`: Value [0, 1]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HsvPixel {
    /// Hue [0, 360)
    pub h: f32,
    /// Saturation [0, 1]
    pub s: f32,
    /// Value [0, 1]
    pub v: f32,
}

impl HsvPixel {
    /// Create a new `HsvPixel`.
    #[must_use]
    pub fn new(h: f32, s: f32, v: f32) -> Self {
        Self { h, s, v }
    }
}

/// Convert an `RgbPixel` (components in [0, 1]) to `HsvPixel`.
#[must_use]
pub fn rgb_to_hsv(rgb: RgbPixel) -> HsvPixel {
    let r = rgb.r;
    let g = rgb.g;
    let b = rgb.b;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    let s = if max < 1e-10 { 0.0 } else { delta / max };

    let h = if delta < 1e-10 {
        0.0
    } else if (max - r).abs() < 1e-10 {
        let mut h = 60.0 * ((g - b) / delta);
        if h < 0.0 {
            h += 360.0;
        }
        h
    } else if (max - g).abs() < 1e-10 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    HsvPixel { h, s, v }
}

/// Convert an `HsvPixel` to `RgbPixel` (components in [0, 1]).
#[must_use]
pub fn hsv_to_rgb(hsv: HsvPixel) -> RgbPixel {
    let h = hsv.h;
    let s = hsv.s;
    let v = hsv.v;

    if s < 1e-10 {
        return RgbPixel { r: v, g: v, b: v };
    }

    let h_sector = h / 60.0;
    let i = h_sector.floor() as i32 % 6;
    let f = h_sector - h_sector.floor();

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    RgbPixel { r, g, b }
}

/// Apply a 1D LUT to every element in `pixels`.
///
/// The LUT maps an index derived from each value to a new value.
/// Values are clamped to [0, 1] before indexing, and the LUT is
/// indexed by `(value * (lut.len() - 1)).round() as usize`.
///
/// If `lut` is empty, no change is made.
pub fn apply_lut_1d(pixels: &mut [f32], lut: &[f32]) {
    if lut.is_empty() {
        return;
    }
    let n = lut.len() - 1;
    for pixel in pixels.iter_mut() {
        let clamped = pixel.clamp(0.0, 1.0);
        let idx = (clamped * n as f32).round() as usize;
        let idx = idx.min(n);
        *pixel = lut[idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_pixel_new() {
        let p = RgbPixel::new(0.1, 0.5, 0.9);
        assert!((p.r - 0.1).abs() < 1e-6);
        assert!((p.g - 0.5).abs() < 1e-6);
        assert!((p.b - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_pixel_clamp() {
        let p = RgbPixel::new(-0.5, 1.5, 0.5).clamp();
        assert!((p.r - 0.0).abs() < 1e-6);
        assert!((p.g - 1.0).abs() < 1e-6);
        assert!((p.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_luminance_white() {
        let p = RgbPixel::new(1.0, 1.0, 1.0);
        let lum = p.luminance();
        assert!((lum - 1.0).abs() < 1e-4, "white luminance: {lum}");
    }

    #[test]
    fn test_rgb_luminance_black() {
        let p = RgbPixel::new(0.0, 0.0, 0.0);
        assert!((p.luminance() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_luminance_rec709_coefficients() {
        let r = RgbPixel::new(1.0, 0.0, 0.0);
        assert!((r.luminance() - 0.2126).abs() < 1e-4);
        let g = RgbPixel::new(0.0, 1.0, 0.0);
        assert!((g.luminance() - 0.7152).abs() < 1e-4);
        let b = RgbPixel::new(0.0, 0.0, 1.0);
        assert!((b.luminance() - 0.0722).abs() < 1e-4);
    }

    #[test]
    fn test_rgb_to_hsv_white() {
        let hsv = rgb_to_hsv(RgbPixel::new(1.0, 1.0, 1.0));
        assert!((hsv.v - 1.0).abs() < 1e-5);
        assert!((hsv.s - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsv_black() {
        let hsv = rgb_to_hsv(RgbPixel::new(0.0, 0.0, 0.0));
        assert!((hsv.v - 0.0).abs() < 1e-5);
        assert!((hsv.s - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsv_red() {
        let hsv = rgb_to_hsv(RgbPixel::new(1.0, 0.0, 0.0));
        assert!((hsv.h - 0.0).abs() < 1.0, "hue of red: {}", hsv.h);
        assert!((hsv.s - 1.0).abs() < 1e-5);
        assert!((hsv.v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsv_green() {
        let hsv = rgb_to_hsv(RgbPixel::new(0.0, 1.0, 0.0));
        assert!((hsv.h - 120.0).abs() < 1.0, "hue of green: {}", hsv.h);
        assert!((hsv.s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_hsv_blue() {
        let hsv = rgb_to_hsv(RgbPixel::new(0.0, 0.0, 1.0));
        assert!((hsv.h - 240.0).abs() < 1.0, "hue of blue: {}", hsv.h);
    }

    #[test]
    fn test_hsv_to_rgb_white() {
        let rgb = hsv_to_rgb(HsvPixel::new(0.0, 0.0, 1.0));
        assert!((rgb.r - 1.0).abs() < 1e-5);
        assert!((rgb.g - 1.0).abs() < 1e-5);
        assert!((rgb.b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hsv_to_rgb_black() {
        let rgb = hsv_to_rgb(HsvPixel::new(0.0, 0.0, 0.0));
        assert!((rgb.r - 0.0).abs() < 1e-5);
        assert!((rgb.g - 0.0).abs() < 1e-5);
        assert!((rgb.b - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let original = RgbPixel::new(0.4, 0.6, 0.8);
        let hsv = rgb_to_hsv(original);
        let recovered = hsv_to_rgb(hsv);
        assert!((original.r - recovered.r).abs() < 1e-4, "R roundtrip");
        assert!((original.g - recovered.g).abs() < 1e-4, "G roundtrip");
        assert!((original.b - recovered.b).abs() < 1e-4, "B roundtrip");
    }

    #[test]
    fn test_apply_lut_1d_identity() {
        let lut: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        let mut pixels = vec![0.0f32, 0.5, 1.0];
        apply_lut_1d(&mut pixels, &lut);
        assert!((pixels[0] - 0.0).abs() < 0.01);
        assert!((pixels[1] - 0.5).abs() < 0.01);
        assert!((pixels[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_lut_1d_invert() {
        let lut: Vec<f32> = (0..=255).map(|i| 1.0 - i as f32 / 255.0).collect();
        let mut pixels = vec![0.0f32, 1.0];
        apply_lut_1d(&mut pixels, &lut);
        assert!((pixels[0] - 1.0).abs() < 0.01);
        assert!((pixels[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_lut_1d_empty_lut() {
        let mut pixels = vec![0.5f32, 0.8];
        apply_lut_1d(&mut pixels, &[]);
        // No change
        assert!((pixels[0] - 0.5).abs() < 1e-6);
        assert!((pixels[1] - 0.8).abs() < 1e-6);
    }
}
