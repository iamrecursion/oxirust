//! LUT builder for convenient LUT generation.
//!
//! This module provides a builder pattern for creating LUTs with various
//! transformations and effects.

use crate::aces::{AcesOdt, AcesSpace};
use crate::colorspace::{ColorSpace, TransferFunction};
use crate::error::LutResult;
use crate::gamut::{GamutMethod, GamutParams};
use crate::tonemap::{ToneMapOp, ToneMapParams};
use crate::{Lut1d, Lut3d, LutSize, Matrix3x3, Rgb};

/// Builder for creating 3D LUTs.
#[derive(Clone)]
pub struct Lut3dBuilder {
    size: LutSize,
    title: Option<String>,
    transforms: Vec<Transform>,
}

/// Transform operation to apply during LUT generation.
#[derive(Clone)]
enum Transform {
    ColorSpaceConvert {
        from: ColorSpace,
        to: ColorSpace,
    },
    TransferFunction {
        tf: TransferFunction,
        inverse: bool,
    },
    Matrix(Matrix3x3),
    GamutMap {
        method: GamutMethod,
        params: GamutParams,
    },
    ToneMap {
        op: ToneMapOp,
        params: ToneMapParams,
    },
    AcesOdt(AcesOdt),
    AcesSpaceConvert {
        from: AcesSpace,
        to: AcesSpace,
    },
    Custom(fn(&Rgb) -> Rgb),
    Exposure(f64),
    Saturation(f64),
    Contrast {
        contrast: f64,
        pivot: f64,
    },
    Hue(f64),
    Vibrance(f64),
}

impl Lut3dBuilder {
    /// Create a new builder with the specified size.
    #[must_use]
    pub fn new(size: LutSize) -> Self {
        Self {
            size,
            title: None,
            transforms: Vec::new(),
        }
    }

    /// Set the LUT title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add a color space conversion.
    #[must_use]
    pub fn color_space(mut self, from: ColorSpace, to: ColorSpace) -> Self {
        self.transforms
            .push(Transform::ColorSpaceConvert { from, to });
        self
    }

    /// Add a transfer function (EOTF or OETF).
    #[must_use]
    pub fn transfer_function(mut self, tf: TransferFunction, inverse: bool) -> Self {
        self.transforms
            .push(Transform::TransferFunction { tf, inverse });
        self
    }

    /// Add a color matrix transform.
    #[must_use]
    pub fn matrix(mut self, matrix: Matrix3x3) -> Self {
        self.transforms.push(Transform::Matrix(matrix));
        self
    }

    /// Add gamut mapping.
    #[must_use]
    pub fn gamut_map(mut self, method: GamutMethod, params: GamutParams) -> Self {
        self.transforms.push(Transform::GamutMap { method, params });
        self
    }

    /// Add tone mapping.
    #[must_use]
    pub fn tone_map(mut self, op: ToneMapOp, params: ToneMapParams) -> Self {
        self.transforms.push(Transform::ToneMap { op, params });
        self
    }

    /// Add ACES ODT.
    #[must_use]
    pub fn aces_odt(mut self, odt: AcesOdt) -> Self {
        self.transforms.push(Transform::AcesOdt(odt));
        self
    }

    /// Add ACES color space conversion.
    #[must_use]
    pub fn aces_space(mut self, from: AcesSpace, to: AcesSpace) -> Self {
        self.transforms
            .push(Transform::AcesSpaceConvert { from, to });
        self
    }

    /// Add a custom transform function.
    #[must_use]
    pub fn custom(mut self, f: fn(&Rgb) -> Rgb) -> Self {
        self.transforms.push(Transform::Custom(f));
        self
    }

    /// Add exposure adjustment (in stops).
    #[must_use]
    pub fn exposure(mut self, stops: f64) -> Self {
        self.transforms.push(Transform::Exposure(stops));
        self
    }

    /// Add saturation adjustment (1.0 = no change, 0.0 = grayscale, >1.0 = more saturated).
    #[must_use]
    pub fn saturation(mut self, amount: f64) -> Self {
        self.transforms.push(Transform::Saturation(amount));
        self
    }

    /// Add contrast adjustment (1.0 = no change, >1.0 = more contrast).
    #[must_use]
    pub fn contrast(mut self, contrast: f64, pivot: f64) -> Self {
        self.transforms
            .push(Transform::Contrast { contrast, pivot });
        self
    }

    /// Add hue rotation (in degrees).
    #[must_use]
    pub fn hue(mut self, degrees: f64) -> Self {
        self.transforms.push(Transform::Hue(degrees));
        self
    }

    /// Add vibrance adjustment (selective saturation).
    #[must_use]
    pub fn vibrance(mut self, amount: f64) -> Self {
        self.transforms.push(Transform::Vibrance(amount));
        self
    }

    /// Build the LUT.
    ///
    /// # Errors
    ///
    /// Returns an error if any transform fails.
    pub fn build(self) -> LutResult<Lut3d> {
        let mut lut = Lut3d::identity(self.size);
        if let Some(title) = self.title {
            lut.title = Some(title);
        }

        // Generate LUT by applying all transforms
        let size = self.size.as_usize();
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let mut rgb = [
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    ];

                    // Apply each transform in order
                    for transform in &self.transforms {
                        rgb = apply_transform(&rgb, transform)?;
                    }

                    lut.set(r, g, b, rgb);
                }
            }
        }

        Ok(lut)
    }
}

/// Apply a single transform to an RGB value.
fn apply_transform(rgb: &Rgb, transform: &Transform) -> LutResult<Rgb> {
    match transform {
        Transform::ColorSpaceConvert { from, to } => from.convert(*to, rgb),
        Transform::TransferFunction { tf, inverse } => {
            if *inverse {
                Ok(tf.eotf_rgb(rgb))
            } else {
                Ok(tf.oetf_rgb(rgb))
            }
        }
        Transform::Matrix(matrix) => Ok(crate::matrix::apply_matrix3x3(matrix, rgb)),
        Transform::GamutMap { method, params } => Ok(crate::gamut::map_gamut(rgb, *method, params)),
        Transform::ToneMap { op, params } => Ok(crate::tonemap::tonemap(rgb, *op, params)),
        Transform::AcesOdt(odt) => odt.apply(rgb),
        Transform::AcesSpaceConvert { from, to } => {
            let aces2065 = from.to_aces2065(rgb);
            Ok(to.from_aces2065(&aces2065))
        }
        Transform::Custom(f) => Ok(f(rgb)),
        Transform::Exposure(stops) => {
            let scale = 2.0_f64.powf(*stops);
            Ok([rgb[0] * scale, rgb[1] * scale, rgb[2] * scale])
        }
        Transform::Saturation(amount) => Ok(adjust_saturation(rgb, *amount)),
        Transform::Contrast { contrast, pivot } => Ok(adjust_contrast(rgb, *contrast, *pivot)),
        Transform::Hue(degrees) => Ok(adjust_hue(rgb, *degrees)),
        Transform::Vibrance(amount) => Ok(adjust_vibrance(rgb, *amount)),
    }
}

/// Adjust saturation.
#[must_use]
fn adjust_saturation(rgb: &Rgb, amount: f64) -> Rgb {
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    [
        luma + (rgb[0] - luma) * amount,
        luma + (rgb[1] - luma) * amount,
        luma + (rgb[2] - luma) * amount,
    ]
}

/// Adjust contrast around a pivot point.
#[must_use]
fn adjust_contrast(rgb: &Rgb, contrast: f64, pivot: f64) -> Rgb {
    [
        pivot + (rgb[0] - pivot) * contrast,
        pivot + (rgb[1] - pivot) * contrast,
        pivot + (rgb[2] - pivot) * contrast,
    ]
}

/// Rotate hue in HSV space.
#[must_use]
fn adjust_hue(rgb: &Rgb, degrees: f64) -> Rgb {
    let (h, s, v) = rgb_to_hsv(rgb);
    let h = (h + degrees / 360.0).rem_euclid(1.0);
    hsv_to_rgb(h, s, v)
}

/// Adjust vibrance (selective saturation that preserves skin tones).
#[must_use]
fn adjust_vibrance(rgb: &Rgb, amount: f64) -> Rgb {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };

    // Vibrance affects low-saturation colors more
    let factor = (1.0 - saturation) * amount;
    adjust_saturation(rgb, 1.0 + factor)
}

/// Convert RGB to HSV.
#[must_use]
fn rgb_to_hsv(rgb: &Rgb) -> (f64, f64, f64) {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta == 0.0 {
        0.0
    } else if (max - rgb[0]).abs() < 1e-10 {
        ((rgb[1] - rgb[2]) / delta).rem_euclid(6.0) / 6.0
    } else if (max - rgb[1]).abs() < 1e-10 {
        ((rgb[2] - rgb[0]) / delta + 2.0) / 6.0
    } else {
        ((rgb[0] - rgb[1]) / delta + 4.0) / 6.0
    };

    (h, s, v)
}

/// Convert HSV to RGB.
#[must_use]
#[allow(clippy::many_single_char_names)]
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> Rgb {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}

/// Builder for creating 1D LUTs.
pub struct Lut1dBuilder {
    size: usize,
    r_curve: Vec<f64>,
    g_curve: Vec<f64>,
    b_curve: Vec<f64>,
}

impl Lut1dBuilder {
    /// Create a new 1D LUT builder.
    #[must_use]
    pub fn new(size: usize) -> Self {
        let identity: Vec<f64> = (0..size).map(|i| i as f64 / (size - 1) as f64).collect();
        Self {
            size,
            r_curve: identity.clone(),
            g_curve: identity.clone(),
            b_curve: identity,
        }
    }

    /// Apply a gamma curve to all channels.
    #[must_use]
    pub fn gamma(mut self, gamma: f64) -> Self {
        for i in 0..self.size {
            let t = i as f64 / (self.size - 1) as f64;
            let v = t.powf(gamma);
            self.r_curve[i] = v;
            self.g_curve[i] = v;
            self.b_curve[i] = v;
        }
        self
    }

    /// Apply different gamma to each channel.
    #[must_use]
    pub fn gamma_rgb(mut self, r: f64, g: f64, b: f64) -> Self {
        for i in 0..self.size {
            let t = i as f64 / (self.size - 1) as f64;
            self.r_curve[i] = t.powf(r);
            self.g_curve[i] = t.powf(g);
            self.b_curve[i] = t.powf(b);
        }
        self
    }

    /// Build the 1D LUT.
    #[must_use]
    pub fn build(self) -> Lut1d {
        let mut lut = Lut1d::new(self.size);
        lut.r = self.r_curve;
        lut.g = self.g_curve;
        lut.b = self.b_curve;
        lut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut3d_builder() {
        let lut = Lut3dBuilder::new(LutSize::Size17)
            .title("Test LUT")
            .exposure(1.0)
            .build()
            .expect("should succeed in test");

        assert_eq!(lut.title, Some("Test LUT".to_string()));
        assert_eq!(lut.size(), 17);
    }

    #[test]
    fn test_lut1d_builder() {
        let lut = Lut1dBuilder::new(256).gamma(2.2).build();
        assert_eq!(lut.size(), 256);

        // Check gamma curve
        let mid = lut.r[128];
        let expected = (0.5_f64).powf(2.2);
        assert!((mid - expected).abs() < 0.01);
    }

    #[test]
    fn test_saturation() {
        let rgb = [0.5, 0.3, 0.7];
        let desaturated = adjust_saturation(&rgb, 0.0);
        // All channels should be equal (gray)
        assert!((desaturated[0] - desaturated[1]).abs() < 1e-10);
        assert!((desaturated[1] - desaturated[2]).abs() < 1e-10);
    }

    #[test]
    fn test_hsv_round_trip() {
        let rgb = [0.5, 0.3, 0.7];
        let (h, s, v) = rgb_to_hsv(&rgb);
        let back = hsv_to_rgb(h, s, v);
        assert!((rgb[0] - back[0]).abs() < 1e-10);
        assert!((rgb[1] - back[1]).abs() < 1e-10);
        assert!((rgb[2] - back[2]).abs() < 1e-10);
    }
}
