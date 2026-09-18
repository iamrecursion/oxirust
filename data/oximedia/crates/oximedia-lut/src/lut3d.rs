//! 3D LUT (Look-Up Table) implementation.
//!
//! 3D LUTs map input RGB values to output RGB values using a 3-dimensional cube
//! of color transformations. They are more powerful than 1D LUTs and can handle
//! complex color grading and correction that affects color relationships.
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::{Lut3d, LutSize, LutInterpolation};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an identity 3D LUT
//! let lut = Lut3d::identity(LutSize::Size33);
//!
//! // Apply to a color with tetrahedral interpolation
//! let input = [0.5, 0.3, 0.7];
//! let output = lut.apply(&input, LutInterpolation::Tetrahedral);
//! # Ok(())
//! # }
//! ```

use crate::error::{LutError, LutResult};
use crate::interpolation::{self, LutInterpolation};
use crate::{LutSize, Rgb};
use std::path::Path;

/// 3D LUT for complex color transformations.
#[derive(Clone, Debug)]
pub struct Lut3d {
    /// LUT data stored as a flat array in R-G-B order.
    /// Index calculation: `(r * size * size + g * size + b) * 3 + channel`
    data: Vec<f64>,
    /// Size of the LUT (number of entries per dimension).
    size: usize,
    /// Input range minimum (usually [0, 0, 0]).
    pub input_min: Rgb,
    /// Input range maximum (usually [1, 1, 1]).
    pub input_max: Rgb,
    /// Title/name of the LUT (optional, from file metadata).
    pub title: Option<String>,
}

impl Lut3d {
    /// Create a new 3D LUT with the specified size.
    ///
    /// All values are initialized to zero.
    #[must_use]
    pub fn new(size: LutSize) -> Self {
        let s = size.as_usize();
        let data_size = s * s * s * 3;
        Self {
            data: vec![0.0; data_size],
            size: s,
            input_min: [0.0, 0.0, 0.0],
            input_max: [1.0, 1.0, 1.0],
            title: None,
        }
    }

    /// Create an identity 3D LUT.
    ///
    /// An identity LUT maps each input color to itself.
    #[must_use]
    pub fn identity(size: LutSize) -> Self {
        let mut lut = Self::new(size);
        let s = lut.size;

        for r in 0..s {
            for g in 0..s {
                for b in 0..s {
                    let rf = r as f64 / (s - 1) as f64;
                    let gf = g as f64 / (s - 1) as f64;
                    let bf = b as f64 / (s - 1) as f64;
                    lut.set(r, g, b, [rf, gf, bf]);
                }
            }
        }

        lut
    }

    /// Get the size of the LUT.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Get the total number of entries in the LUT.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.size * self.size * self.size
    }

    /// Set an RGB value at the specified indices.
    pub fn set(&mut self, r: usize, g: usize, b: usize, value: Rgb) {
        let index = (r * self.size * self.size + g * self.size + b) * 3;
        self.data[index] = value[0];
        self.data[index + 1] = value[1];
        self.data[index + 2] = value[2];
    }

    /// Get an RGB value at the specified indices.
    #[must_use]
    pub fn get(&self, r: usize, g: usize, b: usize) -> Rgb {
        let index = (r * self.size * self.size + g * self.size + b) * 3;
        [self.data[index], self.data[index + 1], self.data[index + 2]]
    }

    /// Apply the 3D LUT to an RGB color.
    #[must_use]
    pub fn apply(&self, rgb: &Rgb, interpolation: LutInterpolation) -> Rgb {
        // Normalize input to 0-1 range based on input min/max
        let normalized = [
            (rgb[0] - self.input_min[0]) / (self.input_max[0] - self.input_min[0]),
            (rgb[1] - self.input_min[1]) / (self.input_max[1] - self.input_min[1]),
            (rgb[2] - self.input_min[2]) / (self.input_max[2] - self.input_min[2]),
        ];

        // Clamp to valid range
        let clamped = [
            normalized[0].clamp(0.0, 1.0),
            normalized[1].clamp(0.0, 1.0),
            normalized[2].clamp(0.0, 1.0),
        ];

        // Map to LUT index space
        let index_f = [
            clamped[0] * (self.size - 1) as f64,
            clamped[1] * (self.size - 1) as f64,
            clamped[2] * (self.size - 1) as f64,
        ];

        match interpolation {
            LutInterpolation::Nearest => self.apply_nearest(&index_f),
            LutInterpolation::Tetrahedral => self.apply_tetrahedral(&index_f),
            _ => self.apply_trilinear(&index_f), // Default to trilinear
        }
    }

    /// Apply nearest neighbor interpolation.
    #[must_use]
    fn apply_nearest(&self, index_f: &[f64; 3]) -> Rgb {
        let r = index_f[0].round() as usize;
        let g = index_f[1].round() as usize;
        let b = index_f[2].round() as usize;
        self.get(
            r.min(self.size - 1),
            g.min(self.size - 1),
            b.min(self.size - 1),
        )
    }

    /// Apply trilinear interpolation.
    #[must_use]
    fn apply_trilinear(&self, index_f: &[f64; 3]) -> Rgb {
        let r0 = index_f[0].floor() as usize;
        let g0 = index_f[1].floor() as usize;
        let b0 = index_f[2].floor() as usize;

        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = index_f[0] - r0 as f64;
        let fg = index_f[1] - g0 as f64;
        let fb = index_f[2] - b0 as f64;

        // Get all 8 corners of the cube
        let c000 = self.get(r0, g0, b0);
        let c100 = self.get(r1, g0, b0);
        let c010 = self.get(r0, g1, b0);
        let c110 = self.get(r1, g1, b0);
        let c001 = self.get(r0, g0, b1);
        let c101 = self.get(r1, g0, b1);
        let c011 = self.get(r0, g1, b1);
        let c111 = self.get(r1, g1, b1);

        interpolation::trilerp_rgb(
            &c000, &c100, &c010, &c110, &c001, &c101, &c011, &c111, fr, fg, fb,
        )
    }

    /// Apply tetrahedral interpolation.
    #[must_use]
    fn apply_tetrahedral(&self, index_f: &[f64; 3]) -> Rgb {
        let r0 = index_f[0].floor() as usize;
        let g0 = index_f[1].floor() as usize;
        let b0 = index_f[2].floor() as usize;

        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = index_f[0] - r0 as f64;
        let fg = index_f[1] - g0 as f64;
        let fb = index_f[2] - b0 as f64;

        // Get all 8 corners of the cube
        let c000 = self.get(r0, g0, b0);
        let c100 = self.get(r1, g0, b0);
        let c010 = self.get(r0, g1, b0);
        let c110 = self.get(r1, g1, b0);
        let c001 = self.get(r0, g0, b1);
        let c101 = self.get(r1, g0, b1);
        let c011 = self.get(r0, g1, b1);
        let c111 = self.get(r1, g1, b1);

        interpolation::tetrahedral_interp(
            &c000, &c100, &c010, &c110, &c001, &c101, &c011, &c111, fr, fg, fb,
        )
    }

    /// Load a 3D LUT from cube format text string.
    ///
    /// Parses `.cube` format data from a string (not a file path).
    ///
    /// # Errors
    ///
    /// Returns an error if the text is not valid cube format.
    pub fn load_cube(text: &str) -> LutResult<Self> {
        let mut title = None;
        let mut size = None;
        let mut domain_min = [0.0_f64, 0.0, 0.0];
        let mut domain_max = [1.0_f64, 1.0, 1.0];
        let mut data: Vec<[f64; 3]> = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("TITLE") {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() >= 2 {
                    title = Some(parts[1].trim().trim_matches('"').to_string());
                }
            } else if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    size = parts[1].parse::<usize>().ok();
                }
            } else if line.starts_with("DOMAIN_MIN") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        domain_min = [r, g, b];
                    }
                }
            } else if line.starts_with("DOMAIN_MAX") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        domain_max = [r, g, b];
                    }
                }
            } else {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                    ) {
                        data.push([r, g, b]);
                    }
                }
            }
        }

        let size = size.ok_or_else(|| LutError::Parse("Missing LUT_3D_SIZE".to_string()))?;
        let expected = size * size * size;
        if data.len() != expected {
            return Err(LutError::InvalidSize {
                expected,
                actual: data.len(),
            });
        }

        let mut lut = Self::new(LutSize::from(size));
        lut.title = title;
        lut.input_min = domain_min;
        lut.input_max = domain_max;

        let mut index = 0;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    lut.set(r, g, b, data[index]);
                    index += 1;
                }
            }
        }

        Ok(lut)
    }

    /// Apply the 3D LUT to individual r, g, b f32 values.
    ///
    /// Returns a tuple of (r, g, b) f32 values.
    #[must_use]
    pub fn apply_rgb(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let rgb = [f64::from(r), f64::from(g), f64::from(b)];
        let result = self.apply(&rgb, crate::interpolation::LutInterpolation::Trilinear);
        (result[0] as f32, result[1] as f32, result[2] as f32)
    }

    /// Apply the 3D LUT to all pixels in a frame.
    ///
    /// Each pixel is a `[f32; 3]` representing `[r, g, b]` in 0.0-1.0 range.
    pub fn apply_frame(&self, pixels: &mut [[f32; 3]]) {
        for pixel in pixels.iter_mut() {
            let (r, g, b) = self.apply_rgb(pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
    }

    /// Create a 3D LUT from a function.
    ///
    /// The function takes normalized RGB input (0.0-1.0) and returns RGB output.
    #[must_use]
    pub fn from_fn<F>(size: LutSize, f: F) -> Self
    where
        F: Fn(Rgb) -> Rgb,
    {
        let mut lut = Self::new(size);
        let s = lut.size;

        for r in 0..s {
            for g in 0..s {
                for b in 0..s {
                    let rf = r as f64 / (s - 1) as f64;
                    let gf = g as f64 / (s - 1) as f64;
                    let bf = b as f64 / (s - 1) as f64;
                    let output = f([rf, gf, bf]);
                    lut.set(r, g, b, output);
                }
            }
        }

        lut
    }

    /// Compose this LUT with another LUT.
    ///
    /// Returns a new LUT that is equivalent to applying `self` followed by `other`.
    #[must_use]
    pub fn compose(&self, other: &Self, interpolation: LutInterpolation) -> Self {
        let size = LutSize::from(self.size);
        Self::from_fn(size, |rgb| {
            let intermediate = self.apply(&rgb, interpolation);
            other.apply(&intermediate, interpolation)
        })
    }

    /// Validate the LUT for common issues.
    ///
    /// Returns a list of warning messages if any issues are found.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for NaN or infinite values
        for &value in &self.data {
            if value.is_nan() {
                warnings.push("LUT contains NaN values".to_string());
                break;
            }
            if value.is_infinite() {
                warnings.push("LUT contains infinite values".to_string());
                break;
            }
        }

        // Check if values are in reasonable range
        let mut has_negative = false;
        let mut has_large = false;
        for &value in &self.data {
            if value < 0.0 {
                has_negative = true;
            }
            if value > 1.0 {
                has_large = true;
            }
        }
        if has_negative {
            warnings.push("LUT contains negative values".to_string());
        }
        if has_large {
            warnings.push("LUT contains values greater than 1.0".to_string());
        }

        warnings
    }

    /// Load a 3D LUT from a file.
    ///
    /// Automatically detects the file format based on the extension.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> LutResult<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| LutError::UnsupportedFormat("No file extension".to_string()))?;

        match extension.to_lowercase().as_str() {
            "cube" => crate::formats::cube::parse_cube_file(path),
            "3dl" => crate::formats::threedl::parse_3dl_file(path),
            "csp" => crate::formats::csp::parse_csp_file(path),
            _ => Err(LutError::UnsupportedFormat(format!(
                "Unsupported format: {extension}"
            ))),
        }
    }

    /// Save the 3D LUT to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> LutResult<()> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| LutError::UnsupportedFormat("No file extension".to_string()))?;

        match extension.to_lowercase().as_str() {
            "cube" => crate::formats::cube::write_cube_file(path, self),
            "3dl" => crate::formats::threedl::write_3dl_file(path, self),
            "csp" => crate::formats::csp::write_csp_file(path, self),
            _ => Err(LutError::UnsupportedFormat(format!(
                "Unsupported format: {extension}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lut() {
        let lut = Lut3d::identity(LutSize::Size17);
        let input = [0.5, 0.3, 0.7];
        let output = lut.apply(&input, LutInterpolation::Trilinear);
        assert!((output[0] - input[0]).abs() < 1e-6);
        assert!((output[1] - input[1]).abs() < 1e-6);
        assert!((output[2] - input[2]).abs() < 1e-6);
    }

    #[test]
    fn test_get_set() {
        let mut lut = Lut3d::new(LutSize::Size17);
        lut.set(0, 0, 0, [0.1, 0.2, 0.3]);
        let value = lut.get(0, 0, 0);
        assert_eq!(value[0], 0.1);
        assert_eq!(value[1], 0.2);
        assert_eq!(value[2], 0.3);
    }

    #[test]
    fn test_from_fn() {
        let lut = Lut3d::from_fn(LutSize::Size17, |rgb| [rgb[0] * 2.0, rgb[1] * 0.5, rgb[2]]);
        let input = [0.5, 0.5, 0.5];
        let output = lut.apply(&input, LutInterpolation::Trilinear);
        assert!((output[0] - 1.0).abs() < 0.05);
        assert!((output[1] - 0.25).abs() < 0.05);
        assert!((output[2] - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_interpolation_consistency() {
        let lut = Lut3d::identity(LutSize::Size33);
        let input = [0.5, 0.3, 0.7];

        let nearest = lut.apply(&input, LutInterpolation::Nearest);
        let trilinear = lut.apply(&input, LutInterpolation::Trilinear);
        let tetrahedral = lut.apply(&input, LutInterpolation::Tetrahedral);

        // For an identity LUT, all methods should give similar results
        assert!((nearest[0] - input[0]).abs() < 0.05);
        assert!((trilinear[0] - input[0]).abs() < 1e-6);
        assert!((tetrahedral[0] - input[0]).abs() < 1e-6);
    }

    #[test]
    fn test_compose() {
        // Create a LUT that doubles values (clamped to valid range)
        let lut1 = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [
                (rgb[0] * 2.0).min(1.0),
                (rgb[1] * 2.0).min(1.0),
                (rgb[2] * 2.0).min(1.0),
            ]
        });
        // Create a LUT that halves values
        let lut2 = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });

        let composed = lut1.compose(&lut2, LutInterpolation::Trilinear);
        let input = [0.4, 0.4, 0.4]; // Use values that won't clip
        let output = composed.apply(&input, LutInterpolation::Trilinear);

        // Should be close to identity
        assert!((output[0] - input[0]).abs() < 0.15);
        assert!((output[1] - input[1]).abs() < 0.15);
        assert!((output[2] - input[2]).abs() < 0.15);
    }

    #[test]
    fn test_validate() {
        let lut = Lut3d::identity(LutSize::Size17);
        let warnings = lut.validate();
        assert_eq!(warnings.len(), 0);
    }

    #[test]
    fn test_size() {
        let lut = Lut3d::new(LutSize::Size33);
        assert_eq!(lut.size(), 33);
        assert_eq!(lut.entry_count(), 33 * 33 * 33);
    }

    #[test]
    fn test_apply_rgb_f32_identity() {
        let lut = Lut3d::identity(LutSize::Size17);
        let (r, g, b) = lut.apply_rgb(0.5, 0.3, 0.7);
        assert!((r - 0.5_f32).abs() < 0.001);
        assert!((g - 0.3_f32).abs() < 0.001);
        assert!((b - 0.7_f32).abs() < 0.001);
    }

    #[test]
    fn test_apply_frame_identity() {
        let lut = Lut3d::identity(LutSize::Size17);
        let mut pixels = vec![[0.5_f32, 0.3, 0.7], [1.0, 0.0, 0.5]];
        lut.apply_frame(&mut pixels);
        assert!((pixels[0][0] - 0.5).abs() < 0.001);
        assert!((pixels[1][0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_cube_minimal() {
        // Build a 17x17x17 cube text (standard size)
        let size = 17_usize;
        let mut cube = format!("LUT_3D_SIZE {size}\n");
        for _ in 0..size * size * size {
            cube.push_str("0.5 0.5 0.5\n");
        }
        let lut = Lut3d::load_cube(&cube).expect("should succeed in test");
        assert_eq!(lut.size(), 17);
        let val = lut.get(0, 0, 0);
        assert!((val[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_load_cube_with_title() {
        let mut cube = "TITLE \"My LUT\"\nLUT_3D_SIZE 2\n".to_string();
        for _ in 0..8 {
            cube.push_str("0.5 0.5 0.5\n");
        }
        let lut = Lut3d::load_cube(&cube).expect("should succeed in test");
        assert_eq!(lut.title.as_deref(), Some("My LUT"));
    }

    #[test]
    fn test_load_cube_missing_size_fails() {
        let cube = "0.5 0.5 0.5\n";
        assert!(Lut3d::load_cube(cube).is_err());
    }

    #[test]
    fn test_apply_frame_empty() {
        let lut = Lut3d::identity(LutSize::Size17);
        let mut pixels: Vec<[f32; 3]> = vec![];
        lut.apply_frame(&mut pixels); // Should not panic
        assert!(pixels.is_empty());
    }
}
