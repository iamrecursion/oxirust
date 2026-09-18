//! White balance calibration for professional camera workflows.
//!
//! Provides illuminant definitions, white balance matrices, grey-world white
//! balance estimation, and calibration reporting.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Illuminant
// ---------------------------------------------------------------------------

/// Standard CIE illuminant definitions with colour temperature and
/// CIE 1931 xy chromaticity coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Illuminant {
    /// D50 – Horizon daylight (5000 K).
    D50,
    /// D55 – Mid-morning/afternoon daylight (5500 K).
    D55,
    /// D65 – Noon daylight (6500 K).
    D65,
    /// D75 – North sky daylight (7500 K).
    D75,
    /// Standard Illuminant A – tungsten/incandescent (2856 K).
    A,
    /// Standard Illuminant B – direct sunlight (~4874 K).
    B,
    /// Standard Illuminant C – overcast sky (~6774 K).
    C,
    /// Fluorescent F2 – cool white (4200 K).
    F2,
    /// Fluorescent F7 – broad-band daylight (6500 K).
    F7,
    /// Fluorescent F11 – narrow-band white (4000 K).
    F11,
}

impl Illuminant {
    /// Colour temperature in Kelvin.
    #[must_use]
    pub const fn color_temperature_k(self) -> u32 {
        match self {
            Self::D50 => 5000,
            Self::D55 => 5500,
            Self::D65 => 6500,
            Self::D75 => 7500,
            Self::A => 2856,
            Self::B => 4874,
            Self::C => 6774,
            Self::F2 => 4200,
            Self::F7 => 6500,
            Self::F11 => 4000,
        }
    }

    /// CIE 1931 xy chromaticity coordinates `(x, y)` for the 2° observer.
    #[must_use]
    pub const fn xy_chromaticity(self) -> (f64, f64) {
        match self {
            Self::D50 => (0.3457, 0.3585),
            Self::D55 => (0.3324, 0.3474),
            Self::D65 => (0.3127, 0.3290),
            Self::D75 => (0.2990, 0.3149),
            Self::A => (0.4476, 0.4074),
            Self::B => (0.3484, 0.3516),
            Self::C => (0.3101, 0.3162),
            Self::F2 => (0.3721, 0.3751),
            Self::F7 => (0.3129, 0.3292),
            Self::F11 => (0.3805, 0.3769),
        }
    }
}

// ---------------------------------------------------------------------------
// WhiteBalanceMatrix
// ---------------------------------------------------------------------------

/// Per-channel RGB white balance gains.
///
/// Multiply each channel of a raw linear image pixel by the corresponding
/// gain to achieve white balance.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WhiteBalanceMatrix {
    /// Per-channel gains `[R, G, B]`.
    pub gains: [f32; 3],
}

impl WhiteBalanceMatrix {
    /// Unity gain – no colour shift.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            gains: [1.0, 1.0, 1.0],
        }
    }

    /// Apply the white balance gains to a linear RGB pixel.
    #[must_use]
    pub fn apply(&self, pixel: [f32; 3]) -> [f32; 3] {
        [
            pixel[0] * self.gains[0],
            pixel[1] * self.gains[1],
            pixel[2] * self.gains[2],
        ]
    }

    /// Normalise so that the green channel gain equals 1.0.
    ///
    /// Green is conventionally used as the reference channel.  If the green
    /// gain is zero the original matrix is returned unchanged.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let g = self.gains[1];
        if g.abs() < f32::EPSILON {
            return self.clone();
        }
        Self {
            gains: [self.gains[0] / g, 1.0, self.gains[2] / g],
        }
    }
}

// ---------------------------------------------------------------------------
// White balance computation helpers
// ---------------------------------------------------------------------------

/// Compute a white balance matrix from a reference patch and its measured
/// value under the camera.
///
/// Each channel gain is `reference[c] / measured[c]`.  Channels where the
/// measured value is zero or negative get a gain of 1.0.
#[must_use]
pub fn compute_wb_from_patch(reference: [f32; 3], measured: [f32; 3]) -> WhiteBalanceMatrix {
    let gain = |r: f32, m: f32| {
        if m > f32::EPSILON {
            r / m
        } else {
            1.0
        }
    };
    WhiteBalanceMatrix {
        gains: [
            gain(reference[0], measured[0]),
            gain(reference[1], measured[1]),
            gain(reference[2], measured[2]),
        ],
    }
}

/// Estimate white balance using the **grey-world assumption**.
///
/// Computes per-channel gains so that each channel average equals the
/// overall luminance average of the image:
/// `gain[c] = mean_all / mean[c]`.
///
/// Returns identity if `pixels` is empty or a channel mean is zero.
#[must_use]
pub fn grey_world_wb(pixels: &[[f32; 3]]) -> WhiteBalanceMatrix {
    if pixels.is_empty() {
        return WhiteBalanceMatrix::identity();
    }

    let n = pixels.len() as f64;
    let mut sum = [0.0f64; 3];
    for px in pixels {
        sum[0] += f64::from(px[0]);
        sum[1] += f64::from(px[1]);
        sum[2] += f64::from(px[2]);
    }
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
    let overall = (mean[0] + mean[1] + mean[2]) / 3.0;

    let gain = |m: f64| {
        if m > 1e-10 {
            (overall / m) as f32
        } else {
            1.0
        }
    };

    WhiteBalanceMatrix {
        gains: [gain(mean[0]), gain(mean[1]), gain(mean[2])],
    }
}

// ---------------------------------------------------------------------------
// WbCalibrationReport
// ---------------------------------------------------------------------------

/// Summary of a white balance calibration run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WbCalibrationReport {
    /// Target illuminant for this calibration.
    pub target_illuminant: Illuminant,
    /// Computed white balance matrix.
    pub matrix: WhiteBalanceMatrix,
    /// Colour accuracy metric – ΔE 76 between target and corrected patch.
    pub delta_e: f32,
}

impl WbCalibrationReport {
    /// Returns `true` when the calibration is within an acceptable ΔE
    /// threshold (< 3.0, which is considered just noticeable difference).
    #[must_use]
    pub fn is_accurate(&self) -> bool {
        self.delta_e < 3.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Illuminant ──────────────────────────────────────────────────────────

    #[test]
    fn test_illuminant_d65_temperature() {
        assert_eq!(Illuminant::D65.color_temperature_k(), 6500);
    }

    #[test]
    fn test_illuminant_a_temperature() {
        assert_eq!(Illuminant::A.color_temperature_k(), 2856);
    }

    #[test]
    fn test_illuminant_f11_temperature() {
        assert_eq!(Illuminant::F11.color_temperature_k(), 4000);
    }

    #[test]
    fn test_illuminant_d65_chromaticity() {
        let (x, y) = Illuminant::D65.xy_chromaticity();
        assert!((x - 0.3127).abs() < 1e-4, "x={x}");
        assert!((y - 0.3290).abs() < 1e-4, "y={y}");
    }

    #[test]
    fn test_illuminant_a_chromaticity_warm() {
        // Illuminant A is a warm source: x > 0.4
        let (x, _) = Illuminant::A.xy_chromaticity();
        assert!(
            x > 0.4,
            "Illuminant A should have warm x chromaticity, got {x}"
        );
    }

    // ── WhiteBalanceMatrix ──────────────────────────────────────────────────

    #[test]
    fn test_wb_matrix_identity() {
        let m = WhiteBalanceMatrix::identity();
        assert_eq!(m.gains, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_wb_matrix_apply_identity() {
        let m = WhiteBalanceMatrix::identity();
        let pixel = [0.5, 0.6, 0.7];
        let out = m.apply(pixel);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_wb_matrix_apply_scales_channels() {
        let m = WhiteBalanceMatrix {
            gains: [2.0, 1.0, 0.5],
        };
        let out = m.apply([1.0, 1.0, 1.0]);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wb_matrix_normalize_green_to_one() {
        let m = WhiteBalanceMatrix {
            gains: [3.0, 2.0, 1.0],
        };
        let n = m.normalize();
        assert!((n.gains[0] - 1.5).abs() < 1e-6, "R gain={}", n.gains[0]);
        assert!((n.gains[1] - 1.0).abs() < 1e-6, "G gain={}", n.gains[1]);
        assert!((n.gains[2] - 0.5).abs() < 1e-6, "B gain={}", n.gains[2]);
    }

    #[test]
    fn test_wb_matrix_normalize_zero_green_returns_self() {
        let m = WhiteBalanceMatrix {
            gains: [1.0, 0.0, 1.0],
        };
        let n = m.normalize();
        assert_eq!(n.gains, m.gains);
    }

    // ── compute_wb_from_patch ───────────────────────────────────────────────

    #[test]
    fn test_compute_wb_from_patch_basic() {
        let reference = [1.0_f32, 1.0, 1.0];
        let measured = [0.5_f32, 1.0, 2.0];
        let m = compute_wb_from_patch(reference, measured);
        assert!((m.gains[0] - 2.0).abs() < 1e-5, "R gain={}", m.gains[0]);
        assert!((m.gains[1] - 1.0).abs() < 1e-5, "G gain={}", m.gains[1]);
        assert!((m.gains[2] - 0.5).abs() < 1e-5, "B gain={}", m.gains[2]);
    }

    #[test]
    fn test_compute_wb_from_patch_zero_measured_channel() {
        let reference = [1.0_f32, 1.0, 1.0];
        let measured = [0.0_f32, 1.0, 1.0];
        let m = compute_wb_from_patch(reference, measured);
        // Zero measured channel → gain defaults to 1.0
        assert!((m.gains[0] - 1.0).abs() < 1e-5);
    }

    // ── grey_world_wb ───────────────────────────────────────────────────────

    #[test]
    fn test_grey_world_wb_neutral_image() {
        // All channels equal → gains should all equal 1.0
        let pixels: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 100];
        let m = grey_world_wb(&pixels);
        for g in m.gains {
            assert!((g - 1.0).abs() < 1e-5, "Expected 1.0, got {g}");
        }
    }

    #[test]
    fn test_grey_world_wb_red_dominant() {
        // Red channel is twice as bright → red gain should be < 1
        let pixels: Vec<[f32; 3]> = vec![[1.0, 0.5, 0.5]; 100];
        let m = grey_world_wb(&pixels);
        assert!(
            m.gains[0] < 1.0,
            "Red gain should be < 1.0 for red-dominant image"
        );
        assert!(m.gains[1] > m.gains[0], "Green gain should exceed red gain");
    }

    #[test]
    fn test_grey_world_wb_empty_pixels_returns_identity() {
        let m = grey_world_wb(&[]);
        assert_eq!(m.gains, [1.0, 1.0, 1.0]);
    }

    // ── WbCalibrationReport ─────────────────────────────────────────────────

    #[test]
    fn test_wb_report_accurate_below_threshold() {
        let report = WbCalibrationReport {
            target_illuminant: Illuminant::D65,
            matrix: WhiteBalanceMatrix::identity(),
            delta_e: 1.5,
        };
        assert!(report.is_accurate());
    }

    #[test]
    fn test_wb_report_inaccurate_above_threshold() {
        let report = WbCalibrationReport {
            target_illuminant: Illuminant::D65,
            matrix: WhiteBalanceMatrix::identity(),
            delta_e: 5.0,
        };
        assert!(!report.is_accurate());
    }
}
