//! Display profile management for color-managed rendering.
//!
//! Provides structures and utilities for describing display device
//! colour characteristics, tone response curves, and calibration state.

#![allow(dead_code)]

use std::fmt;

/// Display colour gamut primaries and white point in CIE xy chromaticity.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayGamut {
    /// Red primary chromaticity (x, y).
    pub red: [f64; 2],
    /// Green primary chromaticity (x, y).
    pub green: [f64; 2],
    /// Blue primary chromaticity (x, y).
    pub blue: [f64; 2],
    /// White point chromaticity (x, y).
    pub white: [f64; 2],
}

impl DisplayGamut {
    /// sRGB / Rec.709 gamut.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            red: [0.6400, 0.3300],
            green: [0.3000, 0.6000],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// DCI-P3 gamut (D65 white point variant).
    #[must_use]
    pub fn dci_p3_d65() -> Self {
        Self {
            red: [0.6800, 0.3200],
            green: [0.2650, 0.6900],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Rec.2020 gamut.
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            red: [0.7080, 0.2920],
            green: [0.1700, 0.7970],
            blue: [0.1310, 0.0460],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Adobe RGB (1998) gamut.
    #[must_use]
    pub fn adobe_rgb() -> Self {
        Self {
            red: [0.6400, 0.3300],
            green: [0.2100, 0.7100],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Compute the gamut area in xy chromaticity using the shoelace formula.
    ///
    /// Larger values indicate wider gamuts.
    #[must_use]
    pub fn chromaticity_area(&self) -> f64 {
        let r = self.red;
        let g = self.green;
        let b = self.blue;
        // Shoelace formula for triangle area
        0.5 * ((g[0] - r[0]) * (b[1] - r[1]) - (b[0] - r[0]) * (g[1] - r[1])).abs()
    }
}

/// Tone response curve type for a display device.
#[derive(Debug, Clone, PartialEq)]
pub enum ToneResponseCurve {
    /// Pure power-law gamma (e.g., 2.2 for sRGB nominal).
    Gamma(f64),
    /// sRGB piecewise transfer function.
    Srgb,
    /// BT.1886 EOTF (reference display gamma).
    Bt1886 {
        /// Black level in cd/m² (typically 0.01).
        black_level: f64,
        /// White level in cd/m² (typically 100.0).
        white_level: f64,
    },
    /// Perceptual Quantizer (ST 2084 / PQ).
    Pq,
    /// Hybrid Log-Gamma (HLG).
    Hlg,
    /// Linear (no gamma encoding).
    Linear,
    /// Custom LUT points (evenly spaced, input 0-1).
    CustomLut(Vec<f64>),
}

impl ToneResponseCurve {
    /// Apply the electro-optical transfer function (encoded -> display linear).
    ///
    /// Input `v` is expected in [0, 1].  Output is in linear light [0, 1]
    /// (or cd/m² units for absolute TRCs like PQ).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_eotf(&self, v: f64) -> f64 {
        match self {
            Self::Gamma(g) => v.abs().powf(*g).copysign(v),
            Self::Linear => v,
            Self::Srgb => {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            }
            Self::Bt1886 {
                black_level,
                white_level,
            } => {
                // Simplified BT.1886: (v)^2.4 mapped to [black, white]
                let linear = v.powf(2.4);
                black_level + (white_level - black_level) * linear
            }
            Self::Pq => {
                // ST 2084 EOTF
                const M1: f64 = 0.1593017578125;
                const M2: f64 = 78.84375;
                const C1: f64 = 0.8359375;
                const C2: f64 = 18.8515625;
                const C3: f64 = 18.6875;
                let vm2 = v.abs().powf(1.0 / M2);
                let num = (vm2 - C1).max(0.0);
                let den = C2 - C3 * vm2;
                (num / den).powf(1.0 / M1) * 10000.0 * if v < 0.0 { -1.0 } else { 1.0 }
            }
            Self::Hlg => {
                // ARIB STD-B67 OETF inverse (simplified)
                const A: f64 = 0.17883277;
                const B: f64 = 0.28466892;
                const C: f64 = 0.55991073;
                if v <= 0.5 {
                    (v * v) / 3.0
                } else {
                    ((v - C).exp() / A + B) / 12.0
                }
            }
            Self::CustomLut(lut) => {
                if lut.is_empty() {
                    return v;
                }
                let n = lut.len() - 1;
                let pos = v.clamp(0.0, 1.0) * n as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(n);
                let t = pos - lo as f64;
                lut[lo] * (1.0 - t) + lut[hi] * t
            }
        }
    }

    /// Apply the opto-electronic transfer function (linear -> encoded).
    ///
    /// This is the inverse of `apply_eotf` for simple gamma/sRGB cases.
    #[must_use]
    pub fn apply_oetf(&self, v: f64) -> f64 {
        match self {
            Self::Gamma(g) => v.abs().powf(1.0 / g).copysign(v),
            Self::Linear => v,
            Self::Srgb => {
                if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            // For other TRCs a full inverse is complex; return v as fallback
            _ => v,
        }
    }
}

/// Calibration state of a display device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationState {
    /// Display has never been calibrated.
    Uncalibrated,
    /// Display has been calibrated, profile may be valid.
    Calibrated,
    /// Calibration date is known (ISO 8601 date string).
    CalibratedAt(String),
    /// Calibration has expired.
    Expired,
}

impl fmt::Display for CalibrationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uncalibrated => write!(f, "Uncalibrated"),
            Self::Calibrated => write!(f, "Calibrated"),
            Self::CalibratedAt(d) => write!(f, "Calibrated at {d}"),
            Self::Expired => write!(f, "Expired"),
        }
    }
}

/// A complete display profile description.
#[derive(Debug, Clone)]
pub struct DisplayProfile {
    /// Human-readable display name.
    pub name: String,
    /// Colour gamut of the display.
    pub gamut: DisplayGamut,
    /// Tone response curve.
    pub trc: ToneResponseCurve,
    /// Peak luminance in cd/m².
    pub peak_luminance: f64,
    /// Black level in cd/m².
    pub black_level: f64,
    /// Calibration state.
    pub calibration: CalibrationState,
    /// Whether the display supports HDR.
    pub hdr_capable: bool,
}

impl DisplayProfile {
    /// Create a standard sRGB display profile.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            name: "sRGB IEC61966-2.1".to_string(),
            gamut: DisplayGamut::srgb(),
            trc: ToneResponseCurve::Srgb,
            peak_luminance: 80.0,
            black_level: 0.0,
            calibration: CalibrationState::Calibrated,
            hdr_capable: false,
        }
    }

    /// Create a Rec.2020 HDR10 display profile.
    #[must_use]
    pub fn hdr10() -> Self {
        Self {
            name: "HDR10 (Rec.2020/PQ)".to_string(),
            gamut: DisplayGamut::rec2020(),
            trc: ToneResponseCurve::Pq,
            peak_luminance: 1000.0,
            black_level: 0.0001,
            calibration: CalibrationState::Calibrated,
            hdr_capable: true,
        }
    }

    /// Dynamic contrast ratio (peak / black level).
    ///
    /// Returns `None` if black level is zero.
    #[must_use]
    pub fn dynamic_contrast(&self) -> Option<f64> {
        if self.black_level <= 0.0 {
            None
        } else {
            Some(self.peak_luminance / self.black_level)
        }
    }

    /// Whether this profile covers at least `coverage` fraction of the sRGB gamut.
    #[must_use]
    pub fn covers_srgb(&self, coverage: f64) -> bool {
        let srgb_area = DisplayGamut::srgb().chromaticity_area();
        let this_area = self.gamut.chromaticity_area();
        this_area >= srgb_area * coverage
    }
}

// ── Automatic color space detection from ICC profile ─────────────────────────

/// Known colour space detected from an ICC profile's primary chromaticities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectedColorSpace {
    /// sRGB IEC 61966-2-1.
    Srgb,
    /// Adobe RGB (1998).
    AdobeRgb,
    /// Display P3 (DCI-P3 with D65).
    DisplayP3,
    /// Rec.2020 / BT.2020.
    Rec2020,
    /// ProPhoto RGB (ROMM RGB).
    ProPhoto,
    /// DCI-P3 (theatre white point).
    DciP3Theatre,
    /// Unknown colour space.
    Unknown,
}

impl DetectedColorSpace {
    /// Human-readable name.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB IEC 61966-2-1",
            Self::AdobeRgb => "Adobe RGB (1998)",
            Self::DisplayP3 => "Display P3",
            Self::Rec2020 => "Rec.2020 / BT.2020",
            Self::ProPhoto => "ProPhoto RGB (ROMM)",
            Self::DciP3Theatre => "DCI-P3 (Theatre)",
            Self::Unknown => "Unknown",
        }
    }
}

/// Reference primary chromaticities for well-known colour spaces.
///
/// Each entry is `[[rx, ry], [gx, gy], [bx, by], [wx, wy]]`.
const KNOWN_PRIMARIES: &[(DetectedColorSpace, [[f64; 2]; 4])] = &[
    (
        DetectedColorSpace::Srgb,
        [
            [0.6400, 0.3300],
            [0.3000, 0.6000],
            [0.1500, 0.0600],
            [0.3127, 0.3290], // D65
        ],
    ),
    (
        DetectedColorSpace::AdobeRgb,
        [
            [0.6400, 0.3300],
            [0.2100, 0.7100],
            [0.1500, 0.0600],
            [0.3127, 0.3290], // D65
        ],
    ),
    (
        DetectedColorSpace::DisplayP3,
        [
            [0.6800, 0.3200],
            [0.2650, 0.6900],
            [0.1500, 0.0600],
            [0.3127, 0.3290], // D65
        ],
    ),
    (
        DetectedColorSpace::Rec2020,
        [
            [0.7080, 0.2920],
            [0.1700, 0.7970],
            [0.1310, 0.0460],
            [0.3127, 0.3290], // D65
        ],
    ),
    (
        DetectedColorSpace::ProPhoto,
        [
            [0.7347, 0.2653],
            [0.1596, 0.8404],
            [0.0366, 0.0001],
            [0.3457, 0.3585], // D50
        ],
    ),
    (
        DetectedColorSpace::DciP3Theatre,
        [
            [0.6800, 0.3200],
            [0.2650, 0.6900],
            [0.1500, 0.0600],
            [0.3140, 0.3510], // DCI white
        ],
    ),
];

/// Tolerance for primary chromaticity matching (CIE xy units).
const PRIMARY_MATCH_TOLERANCE: f64 = 0.002;

/// Detect the colour space from ICC profile primary chromaticities.
///
/// Given measured or parsed red/green/blue and white-point chromaticities,
/// this function identifies which well-known colour space they correspond to
/// by comparing against a table of known standards.
///
/// # Arguments
///
/// * `red_xy` - Red primary CIE xy chromaticity.
/// * `green_xy` - Green primary CIE xy chromaticity.
/// * `blue_xy` - Blue primary CIE xy chromaticity.
/// * `white_xy` - White point CIE xy chromaticity.
///
/// # Returns
///
/// The best matching [`DetectedColorSpace`], or [`DetectedColorSpace::Unknown`].
#[must_use]
pub fn detect_color_space_from_primaries(
    red_xy: [f64; 2],
    green_xy: [f64; 2],
    blue_xy: [f64; 2],
    white_xy: [f64; 2],
) -> DetectedColorSpace {
    let primaries = [red_xy, green_xy, blue_xy, white_xy];

    let mut best_match = DetectedColorSpace::Unknown;
    let mut best_distance = f64::MAX;

    for &(cs, ref ref_primaries) in KNOWN_PRIMARIES {
        let mut total_distance = 0.0;
        for i in 0..4 {
            let dx = primaries[i][0] - ref_primaries[i][0];
            let dy = primaries[i][1] - ref_primaries[i][1];
            total_distance += (dx * dx + dy * dy).sqrt();
        }
        let avg_distance = total_distance / 4.0;

        if avg_distance < best_distance && avg_distance < PRIMARY_MATCH_TOLERANCE {
            best_distance = avg_distance;
            best_match = cs;
        }
    }

    best_match
}

/// Display characterisation and profiling utilities.
///
/// Provides the ability to characterise a display by measuring patches and
/// fitting a display model. Supports both additive (RGB LCD/OLED) and
/// projector characterisation.
#[derive(Debug, Clone)]
pub struct DisplayCharacterizer {
    /// Measured black point in XYZ.
    pub black_xyz: [f64; 3],
    /// Measured white point in XYZ.
    pub white_xyz: [f64; 3],
    /// Measured red primary in XYZ (at 100% R, 0% G, 0% B).
    pub red_xyz: [f64; 3],
    /// Measured green primary in XYZ.
    pub green_xyz: [f64; 3],
    /// Measured blue primary in XYZ.
    pub blue_xyz: [f64; 3],
    /// Tone response curve samples for each channel (R, G, B).
    pub trc_samples: [Vec<f64>; 3],
}

impl DisplayCharacterizer {
    /// Create a new display characterizer with measured data.
    ///
    /// All XYZ values should be in absolute cd/m² units.
    #[must_use]
    pub fn new(
        black_xyz: [f64; 3],
        white_xyz: [f64; 3],
        red_xyz: [f64; 3],
        green_xyz: [f64; 3],
        blue_xyz: [f64; 3],
    ) -> Self {
        Self {
            black_xyz,
            white_xyz,
            red_xyz,
            green_xyz,
            blue_xyz,
            trc_samples: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Add tone response curve samples for a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - 0=R, 1=G, 2=B
    /// * `samples` - Y (luminance) values at evenly-spaced code value steps
    pub fn set_trc_samples(&mut self, channel: usize, samples: Vec<f64>) {
        if channel < 3 {
            self.trc_samples[channel] = samples;
        }
    }

    /// Returns the estimated display gamut from the measured primaries.
    ///
    /// Converts measured XYZ primaries to CIE xy chromaticity for the
    /// detected gamut.
    #[must_use]
    pub fn measured_gamut(&self) -> DisplayGamut {
        DisplayGamut {
            red: xyz_to_xy(self.red_xyz),
            green: xyz_to_xy(self.green_xyz),
            blue: xyz_to_xy(self.blue_xyz),
            white: xyz_to_xy(self.white_xyz),
        }
    }

    /// Returns the detected colour space from measured primaries.
    #[must_use]
    pub fn detected_color_space(&self) -> DetectedColorSpace {
        let gamut = self.measured_gamut();
        detect_color_space_from_primaries(gamut.red, gamut.green, gamut.blue, gamut.white)
    }

    /// Returns the peak luminance (white point Y in cd/m²).
    #[must_use]
    pub fn peak_luminance_nits(&self) -> f64 {
        self.white_xyz[1]
    }

    /// Returns the black level (black point Y in cd/m²).
    #[must_use]
    pub fn black_level_nits(&self) -> f64 {
        self.black_xyz[1]
    }

    /// Returns the static contrast ratio (peak/black).
    ///
    /// Returns `None` if the black level is zero.
    #[must_use]
    pub fn static_contrast_ratio(&self) -> Option<f64> {
        if self.black_xyz[1] <= 0.0 {
            None
        } else {
            Some(self.white_xyz[1] / self.black_xyz[1])
        }
    }

    /// Fit a power-law (gamma) model to the TRC samples for a given channel.
    ///
    /// Uses the least-squares log-log regression to estimate gamma.
    ///
    /// # Arguments
    ///
    /// * `channel` - 0=R, 1=G, 2=B
    ///
    /// # Returns
    ///
    /// Estimated gamma exponent, or `None` if insufficient samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit_gamma(&self, channel: usize) -> Option<f64> {
        if channel >= 3 {
            return None;
        }
        let samples = &self.trc_samples[channel];
        if samples.len() < 4 {
            return None;
        }

        let _n = samples.len() as f64;
        let y_max = *samples.last().unwrap_or(&1.0);
        if y_max <= 0.0 {
            return None;
        }

        // Log-log regression: log(Y) = gamma * log(x) + offset
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut count = 0.0;

        for (i, &y_abs) in samples.iter().enumerate() {
            let x = i as f64 / (samples.len() - 1) as f64;
            let y = y_abs / y_max;
            if x > 0.01 && y > 0.001 {
                let lx = x.ln();
                let ly = y.ln();
                sum_xx += lx * lx;
                sum_xy += lx * ly;
                sum_x += lx;
                count += 1.0;
            }
        }

        if count < 4.0 || (sum_xx - sum_x * sum_x / count).abs() < f64::EPSILON {
            return None;
        }

        let gamma = (sum_xy - sum_x * (sum_xy / sum_xx)) / (sum_xx - sum_x * sum_x / count);
        Some(gamma.abs().clamp(1.0, 4.0))
    }

    /// Generate a `DisplayProfile` from the characterisation data.
    #[must_use]
    pub fn build_profile(&self, name: &str) -> DisplayProfile {
        let gamut = self.measured_gamut();
        let gamma = self.fit_gamma(1).unwrap_or(2.2); // Use green channel

        DisplayProfile {
            name: name.to_string(),
            gamut,
            trc: ToneResponseCurve::Gamma(gamma),
            peak_luminance: self.peak_luminance_nits(),
            black_level: self.black_level_nits(),
            calibration: CalibrationState::Calibrated,
            hdr_capable: self.peak_luminance_nits() > 200.0,
        }
    }
}

/// Convert XYZ to CIE xy chromaticity.
fn xyz_to_xy(xyz: [f64; 3]) -> [f64; 2] {
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum < f64::EPSILON {
        return [0.0, 0.0];
    }
    [xyz[0] / sum, xyz[1] / sum]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_display_gamut_srgb_white() {
        let g = DisplayGamut::srgb();
        assert!(approx_eq(g.white[0], 0.3127, 1e-4));
        assert!(approx_eq(g.white[1], 0.3290, 1e-4));
    }

    #[test]
    fn test_gamut_area_rec2020_wider_than_srgb() {
        let srgb = DisplayGamut::srgb().chromaticity_area();
        let r2020 = DisplayGamut::rec2020().chromaticity_area();
        assert!(r2020 > srgb);
    }

    #[test]
    fn test_gamut_area_p3_wider_than_srgb() {
        let srgb = DisplayGamut::srgb().chromaticity_area();
        let p3 = DisplayGamut::dci_p3_d65().chromaticity_area();
        assert!(p3 > srgb);
    }

    #[test]
    fn test_tone_response_curve_linear() {
        let trc = ToneResponseCurve::Linear;
        assert!(approx_eq(trc.apply_eotf(0.5), 0.5, 1e-10));
        assert!(approx_eq(trc.apply_oetf(0.5), 0.5, 1e-10));
    }

    #[test]
    fn test_tone_response_curve_gamma_round_trip() {
        let trc = ToneResponseCurve::Gamma(2.2);
        let v = 0.6;
        let encoded = trc.apply_oetf(v);
        let decoded = trc.apply_eotf(encoded);
        assert!(approx_eq(decoded, v, 1e-8));
    }

    #[test]
    fn test_srgb_trc_round_trip() {
        let trc = ToneResponseCurve::Srgb;
        for &v in &[0.0, 0.01, 0.18, 0.5, 0.9, 1.0] {
            let enc = trc.apply_oetf(v);
            let dec = trc.apply_eotf(enc);
            assert!(approx_eq(dec, v, 1e-8), "v={v} enc={enc} dec={dec}");
        }
    }

    #[test]
    fn test_srgb_eotf_black() {
        let trc = ToneResponseCurve::Srgb;
        assert!(approx_eq(trc.apply_eotf(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_srgb_eotf_white() {
        let trc = ToneResponseCurve::Srgb;
        assert!(approx_eq(trc.apply_eotf(1.0), 1.0, 1e-8));
    }

    #[test]
    fn test_calibration_state_display() {
        assert_eq!(CalibrationState::Uncalibrated.to_string(), "Uncalibrated");
        let at = CalibrationState::CalibratedAt("2024-01-01".to_string());
        assert!(at.to_string().contains("2024-01-01"));
    }

    #[test]
    fn test_display_profile_srgb_not_hdr() {
        let p = DisplayProfile::srgb();
        assert!(!p.hdr_capable);
        assert!(approx_eq(p.peak_luminance, 80.0, 1e-10));
    }

    #[test]
    fn test_display_profile_hdr10() {
        let p = DisplayProfile::hdr10();
        assert!(p.hdr_capable);
        assert!(p.peak_luminance >= 1000.0);
    }

    #[test]
    fn test_dynamic_contrast_none_for_zero_black() {
        let p = DisplayProfile::srgb();
        // sRGB black_level = 0.0 -> None
        assert!(p.dynamic_contrast().is_none());
    }

    #[test]
    fn test_dynamic_contrast_hdr10() {
        let p = DisplayProfile::hdr10();
        let cr = p
            .dynamic_contrast()
            .expect("dynamic contrast should be available");
        assert!(cr > 1_000_000.0);
    }

    #[test]
    fn test_covers_srgb() {
        let p_srgb = DisplayProfile::srgb();
        let p_hdr10 = DisplayProfile::hdr10();
        assert!(p_srgb.covers_srgb(1.0));
        assert!(p_hdr10.covers_srgb(1.0));
    }

    #[test]
    fn test_custom_lut_trc_interpolates() {
        let lut = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let trc = ToneResponseCurve::CustomLut(lut);
        // Midpoint should interpolate to ~0.5
        assert!(approx_eq(trc.apply_eotf(0.5), 0.5, 1e-6));
    }
}
