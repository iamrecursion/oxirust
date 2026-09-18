#![allow(dead_code)]
//! Lens distortion correction for professional image workflows.
//!
//! Provides barrel/pincushion distortion correction, vignetting compensation,
//! and chromatic aberration reduction using standard lens models.

use std::fmt;

/// Lens distortion model type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DistortionModel {
    /// Brown-Conrady radial distortion model (k1, k2, k3 coefficients).
    BrownConrady,
    /// Division model (single parameter).
    Division,
    /// Fisheye equidistant projection.
    FisheyeEquidistant,
    /// Fisheye stereographic projection.
    FisheyeStereographic,
}

/// Parameters for radial lens distortion correction.
#[derive(Clone, Debug)]
pub struct RadialDistortionParams {
    /// First radial distortion coefficient.
    pub k1: f64,
    /// Second radial distortion coefficient.
    pub k2: f64,
    /// Third radial distortion coefficient.
    pub k3: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
    /// Image center X as fraction of width (0.5 = center).
    pub center_x: f64,
    /// Image center Y as fraction of height (0.5 = center).
    pub center_y: f64,
}

impl Default for RadialDistortionParams {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl RadialDistortionParams {
    /// Creates new params with zero distortion (identity).
    pub fn identity() -> Self {
        Self::default()
    }

    /// Creates barrel distortion params (positive k1).
    pub fn barrel(k1: f64) -> Self {
        Self {
            k1,
            ..Self::default()
        }
    }

    /// Creates pincushion distortion params (negative k1).
    pub fn pincushion(k1: f64) -> Self {
        Self {
            k1: -k1.abs(),
            ..Self::default()
        }
    }

    /// Computes the distorted radius for a given normalized radius.
    ///
    /// Uses the Brown-Conrady model: `r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn distorted_radius(&self, r: f64) -> f64 {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        r * (1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6)
    }

    /// Maps a point from distorted coordinates to undistorted coordinates.
    ///
    /// Returns `(x_undistorted, y_undistorted)` in normalized coordinates.
    #[allow(clippy::cast_precision_loss)]
    pub fn undistort_point(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.center_x;
        let dy = y - self.center_y;
        let r2 = dx * dx + dy * dy;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let tang_x = 2.0 * self.p1 * dx * dy + self.p2 * (r2 + 2.0 * dx * dx);
        let tang_y = self.p1 * (r2 + 2.0 * dy * dy) + 2.0 * self.p2 * dx * dy;
        let ux = self.center_x + dx * radial + tang_x;
        let uy = self.center_y + dy * radial + tang_y;
        (ux, uy)
    }
}

/// Vignetting compensation model.
#[derive(Clone, Debug)]
pub struct VignetteParams {
    /// Vignetting strength (0.0 = no vignetting, 1.0 = maximum).
    pub strength: f64,
    /// Radial falloff exponent (typically 2.0-4.0).
    pub falloff: f64,
    /// Center X as fraction of width.
    pub center_x: f64,
    /// Center Y as fraction of height.
    pub center_y: f64,
}

impl Default for VignetteParams {
    fn default() -> Self {
        Self {
            strength: 0.0,
            falloff: 2.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl VignetteParams {
    /// Computes the vignetting gain at a given normalized position.
    ///
    /// Returns a multiplier >= 1.0 that compensates for vignetting when applied
    /// to pixel values.
    #[allow(clippy::cast_precision_loss)]
    pub fn compensation_gain(&self, norm_x: f64, norm_y: f64) -> f64 {
        let dx = norm_x - self.center_x;
        let dy = norm_y - self.center_y;
        let r2 = dx * dx + dy * dy;
        // Maximum possible r2 from center is 0.5 (corner to center in normalized coords)
        let r_norm = (r2 * 2.0).min(1.0);
        let vignette = 1.0 - self.strength * r_norm.powf(self.falloff / 2.0);
        if vignette <= 0.01 {
            return 1.0 / 0.01;
        }
        1.0 / vignette
    }
}

/// Chromatic aberration parameters per channel.
#[derive(Clone, Debug)]
pub struct ChromaticAberrationParams {
    /// Radial scale for the red channel relative to green.
    pub red_scale: f64,
    /// Radial scale for the blue channel relative to green.
    pub blue_scale: f64,
    /// Center X as fraction of width.
    pub center_x: f64,
    /// Center Y as fraction of height.
    pub center_y: f64,
}

impl Default for ChromaticAberrationParams {
    fn default() -> Self {
        Self {
            red_scale: 1.0,
            blue_scale: 1.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl ChromaticAberrationParams {
    /// Computes the corrected position for the red channel.
    pub fn red_offset(&self, norm_x: f64, norm_y: f64) -> (f64, f64) {
        let dx = norm_x - self.center_x;
        let dy = norm_y - self.center_y;
        (
            self.center_x + dx * self.red_scale,
            self.center_y + dy * self.red_scale,
        )
    }

    /// Computes the corrected position for the blue channel.
    pub fn blue_offset(&self, norm_x: f64, norm_y: f64) -> (f64, f64) {
        let dx = norm_x - self.center_x;
        let dy = norm_y - self.center_y;
        (
            self.center_x + dx * self.blue_scale,
            self.center_y + dy * self.blue_scale,
        )
    }
}

/// A complete lens correction profile combining all corrections.
#[derive(Clone, Debug)]
pub struct LensCorrectionProfile {
    /// Profile name.
    pub name: String,
    /// Lens make.
    pub lens_make: String,
    /// Lens model.
    pub lens_model: String,
    /// Focal length in mm.
    pub focal_length_mm: f64,
    /// Distortion model type.
    pub model: DistortionModel,
    /// Radial distortion parameters.
    pub distortion: RadialDistortionParams,
    /// Vignetting parameters.
    pub vignette: VignetteParams,
    /// Chromatic aberration parameters.
    pub chromatic_aberration: ChromaticAberrationParams,
}

impl fmt::Display for LensCorrectionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LensProfile({} {} @ {}mm)",
            self.lens_make, self.lens_model, self.focal_length_mm
        )
    }
}

impl LensCorrectionProfile {
    /// Creates a new lens correction profile with default (identity) parameters.
    pub fn new(name: &str, make: &str, model_name: &str, focal_mm: f64) -> Self {
        Self {
            name: name.to_string(),
            lens_make: make.to_string(),
            lens_model: model_name.to_string(),
            focal_length_mm: focal_mm,
            model: DistortionModel::BrownConrady,
            distortion: RadialDistortionParams::default(),
            vignette: VignetteParams::default(),
            chromatic_aberration: ChromaticAberrationParams::default(),
        }
    }

    /// Corrects a single pixel coordinate, returning the source coordinate.
    pub fn correct_point(&self, norm_x: f64, norm_y: f64) -> (f64, f64) {
        self.distortion.undistort_point(norm_x, norm_y)
    }
}

/// Bilinear interpolation for sub-pixel sampling.
///
/// `data` is a flat row-major buffer of `width * height` elements.
/// Coordinates are in pixel space (0-based). Out-of-bounds returns the border value.
#[allow(clippy::cast_precision_loss)]
pub fn bilinear_sample(data: &[f64], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x.floor();
    let fy = y - y.floor();

    let sample = |sx: isize, sy: isize| -> f64 {
        let cx = sx.clamp(0, (width as isize) - 1) as usize;
        let cy = sy.clamp(0, (height as isize) - 1) as usize;
        data[cy * width + cx]
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x0 + 1, y0);
    let v01 = sample(x0, y0 + 1);
    let v11 = sample(x0 + 1, y0 + 1);

    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_distortion() {
        let params = RadialDistortionParams::identity();
        assert_eq!(params.k1, 0.0);
        assert_eq!(params.k2, 0.0);
        assert_eq!(params.k3, 0.0);
    }

    #[test]
    fn test_distorted_radius_identity() {
        let params = RadialDistortionParams::identity();
        assert!((params.distorted_radius(0.5) - 0.5).abs() < 1e-10);
        assert!((params.distorted_radius(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_barrel_distortion_expands() {
        let params = RadialDistortionParams::barrel(0.1);
        // Barrel distortion: distorted radius should be > original
        let r = 0.5;
        assert!(params.distorted_radius(r) > r);
    }

    #[test]
    fn test_pincushion_distortion_contracts() {
        let params = RadialDistortionParams::pincushion(0.1);
        // Pincushion: negative k1, distorted radius < original
        let r = 0.5;
        assert!(params.distorted_radius(r) < r);
    }

    #[test]
    fn test_undistort_center_unchanged() {
        let params = RadialDistortionParams::barrel(0.2);
        let (ux, uy) = params.undistort_point(0.5, 0.5);
        assert!((ux - 0.5).abs() < 1e-10);
        assert!((uy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vignette_center_no_correction() {
        let params = VignetteParams {
            strength: 0.5,
            falloff: 2.0,
            center_x: 0.5,
            center_y: 0.5,
        };
        let gain = params.compensation_gain(0.5, 0.5);
        assert!((gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vignette_corner_higher_gain() {
        let params = VignetteParams {
            strength: 0.5,
            falloff: 2.0,
            center_x: 0.5,
            center_y: 0.5,
        };
        let corner_gain = params.compensation_gain(0.0, 0.0);
        let center_gain = params.compensation_gain(0.5, 0.5);
        assert!(corner_gain > center_gain);
    }

    #[test]
    fn test_vignette_zero_strength() {
        let params = VignetteParams {
            strength: 0.0,
            ..VignetteParams::default()
        };
        let gain = params.compensation_gain(0.0, 0.0);
        assert!((gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chromatic_aberration_identity() {
        let params = ChromaticAberrationParams::default();
        let (rx, ry) = params.red_offset(0.7, 0.3);
        assert!((rx - 0.7).abs() < 1e-10);
        assert!((ry - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_chromatic_aberration_red_shift() {
        let params = ChromaticAberrationParams {
            red_scale: 1.01,
            blue_scale: 0.99,
            center_x: 0.5,
            center_y: 0.5,
        };
        let (rx, _ry) = params.red_offset(0.8, 0.5);
        // Red channel shifted outward
        assert!(rx > 0.8);
    }

    #[test]
    fn test_lens_profile_creation() {
        let profile = LensCorrectionProfile::new("test", "Canon", "EF 50mm f/1.4", 50.0);
        assert_eq!(profile.focal_length_mm, 50.0);
        assert_eq!(profile.lens_make, "Canon");
        let display = format!("{profile}");
        assert!(display.contains("Canon"));
        assert!(display.contains("50"));
    }

    #[test]
    fn test_bilinear_sample_exact() {
        // 2x2 image
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let val = bilinear_sample(&data, 2, 2, 0.0, 0.0);
        assert!((val - 0.0).abs() < 1e-10);
        let val = bilinear_sample(&data, 2, 2, 1.0, 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_sample_interpolated() {
        // 2x2: [0, 2; 0, 2]
        let data = vec![0.0, 2.0, 0.0, 2.0];
        let val = bilinear_sample(&data, 2, 2, 0.5, 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_profile_correct_center() {
        let profile = LensCorrectionProfile::new("test", "Nikon", "AF-S 85mm", 85.0);
        let (cx, cy) = profile.correct_point(0.5, 0.5);
        assert!((cx - 0.5).abs() < 1e-10);
        assert!((cy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_distortion_model_variants() {
        assert_ne!(DistortionModel::BrownConrady, DistortionModel::Division);
        assert_ne!(
            DistortionModel::FisheyeEquidistant,
            DistortionModel::FisheyeStereographic
        );
    }
}
