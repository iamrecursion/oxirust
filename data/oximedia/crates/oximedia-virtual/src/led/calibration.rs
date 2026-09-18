//! LED wall color calibration with matrix correction and gamma LUT.
//!
//! Provides tools for measuring and correcting color accuracy on LED wall panels,
//! including 3x3 color correction matrices and per-channel gamma LUTs.

#![allow(dead_code)]

/// A physical LED wall panel descriptor.
#[derive(Debug, Clone)]
pub struct LedWallPanel {
    /// Panel identifier.
    pub id: u32,
    /// Number of tiles in the horizontal direction.
    pub x_panels: u32,
    /// Number of tiles in the vertical direction.
    pub y_panels: u32,
    /// Panel native resolution (width, height in pixels).
    pub resolution: (u32, u32),
    /// Panel gamma value (typically 2.2 for standard LED panels).
    pub gamma: f32,
}

impl LedWallPanel {
    /// Creates a new LED wall panel descriptor.
    #[must_use]
    pub const fn new(
        id: u32,
        x_panels: u32,
        y_panels: u32,
        resolution: (u32, u32),
        gamma: f32,
    ) -> Self {
        Self {
            id,
            x_panels,
            y_panels,
            resolution,
            gamma,
        }
    }

    /// Returns the total number of pixels across all tiles.
    #[must_use]
    pub fn total_pixels(&self) -> u32 {
        self.resolution.0 * self.resolution.1 * self.x_panels * self.y_panels
    }
}

impl Default for LedWallPanel {
    fn default() -> Self {
        Self {
            id: 0,
            x_panels: 1,
            y_panels: 1,
            resolution: (1920, 1080),
            gamma: 2.2,
        }
    }
}

/// A calibration color patch: a target color and a human-readable label.
#[derive(Debug, Clone)]
pub struct CalibrationPatch {
    /// Target RGB color in linear 0..1 range.
    pub target_rgb: [f32; 3],
    /// Human-readable label (e.g. "White", "Red Primary").
    pub label: String,
}

impl CalibrationPatch {
    /// Creates a new calibration patch.
    #[must_use]
    pub fn new(target_rgb: [f32; 3], label: impl Into<String>) -> Self {
        Self {
            target_rgb,
            label: label.into(),
        }
    }
}

/// A measured calibration target: a patch plus what the camera actually measured.
#[derive(Debug, Clone)]
pub struct LedCalibrationTarget {
    /// The intended color patch.
    pub patch: CalibrationPatch,
    /// The RGB values measured by a camera or colorimeter (linear 0..1).
    pub measured_rgb: [f32; 3],
}

impl LedCalibrationTarget {
    /// Creates a new calibration target.
    #[must_use]
    pub fn new(patch: CalibrationPatch, measured_rgb: [f32; 3]) -> Self {
        Self {
            patch,
            measured_rgb,
        }
    }

    /// Returns the per-channel ratio of target to measured (correction factors).
    #[must_use]
    pub fn correction_ratios(&self) -> [f32; 3] {
        [
            Self::safe_div(self.patch.target_rgb[0], self.measured_rgb[0]),
            Self::safe_div(self.patch.target_rgb[1], self.measured_rgb[1]),
            Self::safe_div(self.patch.target_rgb[2], self.measured_rgb[2]),
        ]
    }

    fn safe_div(num: f32, den: f32) -> f32 {
        if den.abs() < 1e-9 {
            1.0
        } else {
            (num / den).clamp(0.0, 4.0)
        }
    }
}

/// A 3x3 color correction matrix.
///
/// Applied as: `output = matrix * input`, where input/output are column vectors [R, G, B].
/// The matrix is stored in row-major order: `inner[row][col]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LedColorMatrix(pub [[f32; 3]; 3]);

impl LedColorMatrix {
    /// Returns the identity matrix (no color correction).
    #[must_use]
    pub const fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Computes a correction matrix from a set of calibration targets using a simplified
    /// ratio-based least-squares approach.
    ///
    /// For each channel, the correction is the average of per-target correction ratios.
    /// This produces a diagonal correction matrix that compensates for systematic
    /// per-channel gains.
    #[must_use]
    pub fn compute(targets: &[LedCalibrationTarget]) -> Self {
        if targets.is_empty() {
            return Self::identity();
        }

        // Average per-channel correction ratios across all targets
        let mut sum = [0.0f32; 3];
        let n = targets.len() as f32;

        for target in targets {
            let ratios = target.correction_ratios();
            sum[0] += ratios[0];
            sum[1] += ratios[1];
            sum[2] += ratios[2];
        }

        let avg_r = (sum[0] / n).clamp(0.0, 4.0);
        let avg_g = (sum[1] / n).clamp(0.0, 4.0);
        let avg_b = (sum[2] / n).clamp(0.0, 4.0);

        // Build diagonal correction matrix
        Self([[avg_r, 0.0, 0.0], [0.0, avg_g, 0.0], [0.0, 0.0, avg_b]])
    }

    /// Applies the matrix to an RGB color vector.
    ///
    /// The result is clamped to [0.0, 1.0].
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            (m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2]).clamp(0.0, 1.0),
            (m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2]).clamp(0.0, 1.0),
            (m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2]).clamp(0.0, 1.0),
        ]
    }

    /// Multiplies two matrices (self * other).
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let a = &self.0;
        let b = &other.0;
        let mut result = [[0.0f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                result[row][col] =
                    a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
            }
        }
        Self(result)
    }
}

impl Default for LedColorMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Per-channel gamma correction via a precomputed 8-bit LUT.
#[derive(Debug, Clone)]
pub struct LedGammaCorrection {
    /// Look-up table: maps input value [0..=255] to gamma-corrected output.
    pub lut: Vec<u8>,
}

impl LedGammaCorrection {
    /// Builds a gamma correction LUT for the given gamma value.
    ///
    /// Output = (input / 255)^gamma * 255.
    #[must_use]
    pub fn build(gamma: f32) -> Self {
        let lut = (0u32..=255)
            .map(|i| {
                let normalized = i as f32 / 255.0;
                let corrected = normalized.powf(gamma);
                (corrected * 255.0).round().clamp(0.0, 255.0) as u8
            })
            .collect();
        Self { lut }
    }

    /// Applies gamma correction to an 8-bit value.
    #[must_use]
    pub fn apply(&self, value: u8) -> u8 {
        self.lut[value as usize]
    }

    /// Applies gamma correction to an RGB triplet.
    #[must_use]
    pub fn apply_rgb(&self, rgb: [u8; 3]) -> [u8; 3] {
        [self.apply(rgb[0]), self.apply(rgb[1]), self.apply(rgb[2])]
    }
}

impl Default for LedGammaCorrection {
    fn default() -> Self {
        Self::build(1.0) // Linear (no correction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_led_wall_panel_default() {
        let panel = LedWallPanel::default();
        assert_eq!(panel.id, 0);
        assert_eq!(panel.gamma, 2.2);
    }

    #[test]
    fn test_led_wall_panel_total_pixels() {
        let panel = LedWallPanel::new(0, 2, 3, (100, 50), 2.2);
        assert_eq!(panel.total_pixels(), 2 * 3 * 100 * 50);
    }

    #[test]
    fn test_calibration_patch_creation() {
        let patch = CalibrationPatch::new([1.0, 0.0, 0.0], "Red Primary");
        assert_eq!(patch.label, "Red Primary");
        assert_eq!(patch.target_rgb, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_calibration_target_correction_ratios_perfect() {
        let patch = CalibrationPatch::new([1.0, 1.0, 1.0], "White");
        let target = LedCalibrationTarget::new(patch, [1.0, 1.0, 1.0]);
        let ratios = target.correction_ratios();
        assert!((ratios[0] - 1.0).abs() < 1e-6);
        assert!((ratios[1] - 1.0).abs() < 1e-6);
        assert!((ratios[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_target_correction_ratios_off() {
        let patch = CalibrationPatch::new([1.0, 1.0, 1.0], "White");
        let target = LedCalibrationTarget::new(patch, [0.5, 0.8, 1.0]);
        let ratios = target.correction_ratios();
        assert!((ratios[0] - 2.0).abs() < 1e-4); // 1.0 / 0.5
        assert!((ratios[1] - 1.25).abs() < 1e-4); // 1.0 / 0.8
        assert!((ratios[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_led_color_matrix_identity() {
        let m = LedColorMatrix::identity();
        let input = [0.5, 0.3, 0.8];
        let output = m.apply(input);
        assert!((output[0] - input[0]).abs() < 1e-6);
        assert!((output[1] - input[1]).abs() < 1e-6);
        assert!((output[2] - input[2]).abs() < 1e-6);
    }

    #[test]
    fn test_led_color_matrix_compute_empty() {
        let m = LedColorMatrix::compute(&[]);
        assert_eq!(m, LedColorMatrix::identity());
    }

    #[test]
    fn test_led_color_matrix_compute_perfect_targets() {
        let targets = vec![
            LedCalibrationTarget::new(
                CalibrationPatch::new([1.0, 1.0, 1.0], "White"),
                [1.0, 1.0, 1.0],
            ),
            LedCalibrationTarget::new(
                CalibrationPatch::new([0.5, 0.5, 0.5], "Gray"),
                [0.5, 0.5, 0.5],
            ),
        ];
        let m = LedColorMatrix::compute(&targets);
        // With perfect targets, matrix should be identity
        assert!((m.0[0][0] - 1.0).abs() < 1e-4);
        assert!((m.0[1][1] - 1.0).abs() < 1e-4);
        assert!((m.0[2][2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_led_color_matrix_apply_clamps() {
        let m = LedColorMatrix([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
        let output = m.apply([1.0, 1.0, 1.0]);
        assert_eq!(output, [1.0, 1.0, 1.0]); // clamped to 1.0
    }

    #[test]
    fn test_led_color_matrix_mul() {
        let identity = LedColorMatrix::identity();
        let m = LedColorMatrix([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]);
        let result = identity.mul(&m);
        assert!((result.0[0][0] - 2.0).abs() < 1e-6);
        assert!((result.0[1][1] - 0.5).abs() < 1e-6);
        assert!((result.0[2][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_correction_linear() {
        let gc = LedGammaCorrection::build(1.0);
        // With gamma=1.0, LUT should be identity
        assert_eq!(gc.apply(0), 0);
        assert_eq!(gc.apply(128), 128);
        assert_eq!(gc.apply(255), 255);
    }

    #[test]
    fn test_gamma_correction_lut_size() {
        let gc = LedGammaCorrection::build(2.2);
        assert_eq!(gc.lut.len(), 256);
    }

    #[test]
    fn test_gamma_correction_monotonic() {
        let gc = LedGammaCorrection::build(2.2);
        // Gamma-corrected LUT should be monotonically non-decreasing
        for i in 1..256 {
            assert!(gc.lut[i] >= gc.lut[i - 1]);
        }
    }

    #[test]
    fn test_gamma_correction_apply_rgb() {
        let gc = LedGammaCorrection::build(1.0);
        let result = gc.apply_rgb([100, 150, 200]);
        assert_eq!(result, [100, 150, 200]);
    }
}
