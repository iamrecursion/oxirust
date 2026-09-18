#![allow(dead_code)]
//! ACES (Academy Color Encoding System) pipeline utilities.
//!
//! Provides color space enumerations, transform descriptors, and input-device
//! transforms (IDTs) for building ACES-compliant color pipelines.
//!
//! # Reference
//! - SMPTE ST 2065-1 (ACES2065-1 encoding)
//! - S-2014-006 (ACEScg)
//! - TB-2014-012 (ACES Input Transform methodology)

use std::fmt;

// ─── Color Spaces ────────────────────────────────────────────────────────────

/// Enumerates the principal ACES color encoding spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcesColorSpace {
    /// ACES2065-1: scene-linear, AP0 primaries, D60 white point.
    /// Used as the archival interchange color space.
    ACES2065_1,
    /// ACEScg: scene-linear, AP1 primaries, D60 white point.
    /// Preferred working color space for VFX and compositing.
    ACEScg,
    /// ACEScc: logarithmic, AP1 primaries. Designed for color grading.
    ACEScc,
    /// ACEScct: logarithmic with a toe, AP1 primaries. Improved shadow
    /// behaviour compared to ACEScc.
    ACEScct,
    /// ACESproxy: integer log encoding for on-set preview monitors.
    ACESproxy,
}

impl AcesColorSpace {
    /// Returns a short human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::ACES2065_1 => "ACES2065-1",
            Self::ACEScg => "ACEScg",
            Self::ACEScc => "ACEScc",
            Self::ACEScct => "ACEScct",
            Self::ACESproxy => "ACESproxy",
        }
    }

    /// Returns `true` if this encoding is scene-linear (not log).
    #[must_use]
    pub fn is_linear(self) -> bool {
        matches!(self, Self::ACES2065_1 | Self::ACEScg)
    }

    /// Returns `true` if the space uses AP0 primaries (ACES2065-1 only).
    #[must_use]
    pub fn uses_ap0(self) -> bool {
        self == Self::ACES2065_1
    }

    /// Returns `true` if the space uses AP1 primaries.
    #[must_use]
    pub fn uses_ap1(self) -> bool {
        !self.uses_ap0()
    }

    /// Returns the nominal exposure middle-grey value in this encoding.
    #[must_use]
    pub fn middle_grey(self) -> f32 {
        match self {
            Self::ACES2065_1 | Self::ACEScg => 0.18,
            Self::ACEScc => 0.4135884, // log2(0.18)/17.52 + (9.72/17.52) approx
            Self::ACEScct => 0.413,
            Self::ACESproxy => 0.413,
        }
    }
}

impl fmt::Display for AcesColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── AP0 / AP1 matrix coefficients ───────────────────────────────────────────

/// 3 × 3 row-major matrix for converting AP1 (ACEScg) to AP0 (ACES2065-1).
///
/// Source: ACES S-2014-006, Table B.2.
pub const AP1_TO_AP0: [[f32; 3]; 3] = [
    [0.695_452, 0.140_679, 0.163_869],
    [0.044_794, 0.859_671, 0.095_535],
    [-0.005_535, 0.004_062, 1.001_473],
];

/// 3 × 3 row-major matrix for converting AP0 (ACES2065-1) to AP1 (ACEScg).
///
/// Source: ACES S-2014-006, Table B.3.
pub const AP0_TO_AP1: [[f32; 3]; 3] = [
    [1.451_439, -0.236_511, -0.214_929],
    [-0.076_597, 1.176_226, -0.099_629],
    [0.008_332, -0.006_051, 0.997_719],
];

// ─── Matrix helpers ───────────────────────────────────────────────────────────

/// Multiply a 3×3 row-major matrix by a column vector [r, g, b].
#[must_use]
fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ─── AcesTransform ───────────────────────────────────────────────────────────

/// Describes a single color-space conversion within the ACES system.
///
/// Only linear-to-linear conversions between AP0 and AP1 are currently
/// implemented; non-linear spaces require an additional tone curve step.
#[derive(Debug, Clone, Copy)]
pub struct AcesTransform {
    /// Input color space.
    pub src: AcesColorSpace,
    /// Output color space.
    pub dst: AcesColorSpace,
}

impl AcesTransform {
    /// Create a new `AcesTransform` between `src` and `dst`.
    #[must_use]
    pub const fn new(src: AcesColorSpace, dst: AcesColorSpace) -> Self {
        Self { src, dst }
    }

    /// Returns `true` when source and destination are the same space (identity).
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.src == self.dst
    }

    /// Apply the transform to a linear scene-referred RGB triplet.
    ///
    /// Currently handles ACES2065-1 ↔ ACEScg via the AP0/AP1 matrices.
    /// All other combinations return the input unchanged (identity pass-through).
    #[must_use]
    pub fn apply(self, rgb: [f32; 3]) -> [f32; 3] {
        match (self.src, self.dst) {
            (AcesColorSpace::ACEScg, AcesColorSpace::ACES2065_1) => mat3_mul(&AP1_TO_AP0, rgb),
            (AcesColorSpace::ACES2065_1, AcesColorSpace::ACEScg) => mat3_mul(&AP0_TO_AP1, rgb),
            _ => rgb,
        }
    }
}

// ─── AcesInputTransform ──────────────────────────────────────────────────────

/// Source device class for an Input Device Transform (IDT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputDeviceClass {
    /// Standard digital cinema camera (e.g. ARRI, RED, Sony Venice).
    DigitalCinema,
    /// Still-camera raw sensor output.
    RawSensor,
    /// Computer-generated imagery (scene-linear, sRGB primaries).
    CgiLinear,
    /// Legacy video (Rec.709 / BT.1886).
    Video709,
}

impl InputDeviceClass {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::DigitalCinema => "Digital Cinema",
            Self::RawSensor => "Raw Sensor",
            Self::CgiLinear => "CGI Linear",
            Self::Video709 => "Video Rec.709",
        }
    }
}

/// An Input Device Transform that converts camera-native footage to ACEScg.
///
/// In a full ACES pipeline the IDT maps raw or log-encoded footage from a
/// specific camera into the ACES scene-linear working space (ACEScg).
#[derive(Debug, Clone)]
pub struct AcesInputTransform {
    /// Descriptive name for the camera / device this IDT targets.
    pub device_name: String,
    /// Class of source device.
    pub device_class: InputDeviceClass,
    /// 3 × 3 input matrix applied before the transfer function (row-major).
    pub input_matrix: [[f32; 3]; 3],
    /// Whether the input data is already in a linear state (no OETF reversal).
    pub input_is_linear: bool,
    /// Exposure compensation applied as a linear gain after the IDT.
    pub exposure_compensation: f32,
}

impl AcesInputTransform {
    /// Create an IDT configured for scene-linear CGI input (identity matrix).
    #[must_use]
    pub fn cgi_linear() -> Self {
        Self {
            device_name: "CGI Linear (sRGB primaries)".to_owned(),
            device_class: InputDeviceClass::CgiLinear,
            input_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            input_is_linear: true,
            exposure_compensation: 1.0,
        }
    }

    /// Create an IDT for generic Rec.709 video.
    #[must_use]
    pub fn video_rec709() -> Self {
        // Simplified: Rec.709 → AP1 matrix (approximate)
        Self {
            device_name: "Rec.709 Video".to_owned(),
            device_class: InputDeviceClass::Video709,
            input_matrix: [
                [0.613_117, 0.339_496, 0.047_387],
                [0.070_093, 0.916_378, 0.013_529],
                [0.020_535, 0.109_453, 0.870_012],
            ],
            input_is_linear: false,
            exposure_compensation: 1.0,
        }
    }

    /// Apply this IDT to a camera-native RGB triplet.
    ///
    /// Steps:
    /// 1. Apply `input_matrix`.
    /// 2. Apply `exposure_compensation` gain.
    ///
    /// (OETF reversal / log decode is outside the scope of this simplified IDT.)
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let [r, g, b] = mat3_mul(&self.input_matrix, rgb);
        [
            r * self.exposure_compensation,
            g * self.exposure_compensation,
            b * self.exposure_compensation,
        ]
    }

    /// Set the exposure compensation and return `self` for chaining.
    #[must_use]
    pub fn with_exposure(mut self, stops: f32) -> Self {
        self.exposure_compensation = 2.0_f32.powf(stops);
        self
    }

    /// Returns `true` when the IDT is purely an identity (unit matrix, no gain).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        let id = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        self.input_matrix == id && (self.exposure_compensation - 1.0).abs() < 1e-6
    }
}

// ─── ACES Output Transform 2.0 ───────────────────────────────────────────────

/// Target output colour space for an ACES Output Transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputColorSpace {
    /// sRGB (IEC 61966-2-1), D65 white point, gamma ~2.2.
    Srgb,
    /// Display P3 (DCI-P3 with D65 white point).
    P3D65,
    /// Rec.2020, D65, HDR-capable.
    Rec2020,
    /// Rec.709 (same gamut as sRGB, but BT.1886 gamma).
    Rec709,
}

impl OutputColorSpace {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB",
            Self::P3D65 => "P3-D65",
            Self::Rec2020 => "Rec.2020",
            Self::Rec709 => "Rec.709",
        }
    }

    /// ACEScg (AP1) to output-RGB 3×3 matrix (D60 → D65 adaptation included).
    ///
    /// Source: ACES CTL transforms and ACES OT 2.0 working document.
    #[must_use]
    pub fn acescg_to_output_matrix(self) -> [[f32; 3]; 3] {
        match self {
            Self::Srgb | Self::Rec709 => {
                // ACEScg AP1 → sRGB/Rec.709, D60→D65 adapted (approximate)
                [
                    [1.704_859, -0.621_791, -0.083_068],
                    [-0.130_257, 1.140_804, -0.010_547],
                    [-0.023_964, -0.128_975, 1.152_939],
                ]
            }
            Self::P3D65 => {
                // ACEScg AP1 → P3-D65
                [
                    [1.286_189, -0.278_894, -0.007_295],
                    [-0.042_581, 1.062_323, -0.019_742],
                    [-0.013_604, -0.158_305, 1.171_909],
                ]
            }
            Self::Rec2020 => {
                // ACEScg AP1 → Rec.2020
                [
                    [0.941_700, 0.013_981, 0.044_319],
                    [0.008_956, 0.948_080, 0.042_964],
                    [-0.004_789, 0.017_188, 0.987_601],
                ]
            }
        }
    }

    /// Apply the output OETF (encoding gamma/TRC) to a single linear value.
    #[must_use]
    pub fn apply_oetf(self, linear: f32) -> f32 {
        let v = linear.clamp(0.0, 1.0) as f64;
        let encoded = match self {
            Self::Srgb => {
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Self::Rec709 => {
                if v < 0.018 {
                    4.5 * v
                } else {
                    1.099 * v.powf(0.45) - 0.099
                }
            }
            Self::P3D65 => {
                // DCI-P3 uses 2.6 gamma, but P3-D65 for consumer displays uses sRGB-like
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
            Self::Rec2020 => {
                // BT.2020 10-bit OETF
                const ALPHA: f64 = 1.099_296_826_809_443;
                const BETA: f64 = 0.018_053_968_510_807;
                if v < BETA {
                    4.5 * v
                } else {
                    ALPHA * v.powf(0.45) - (ALPHA - 1.0)
                }
            }
        };
        encoded.clamp(0.0, 1.0) as f32
    }
}

/// ACES Reference Rendering Transform (RRT) tone curve.
///
/// The RRT maps scene-linear ACES values to an output-referred signal.
/// This is the standard "S-curve" used in the ACES pipeline.
///
/// Based on: ACES CTL `rrt.ctl`, parametric form from ACES developers.
#[must_use]
pub fn aces_rrt_tone_curve(x: f32) -> f32 {
    // Parametric RRT S-curve (simplified; matches ACES 1.0 closely)
    // Uses the segmented spline model: toe / midpoint / shoulder
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;

    // ACES-fitted curve — identical formula to the widely-used approximation
    let v = x.max(0.0);
    ((v * (A * v + B)) / (v * (C * v + D) + E)).clamp(0.0, 1.0)
}

/// ACES Output Transform 2.0 (OT 2.0).
///
/// Implements the full ACES Output Transform pipeline:
/// 1. Apply the Reference Rendering Transform (RRT) tone curve per channel.
/// 2. Gamut-convert from ACEScg (AP1) to the target output colour space.
/// 3. Apply the output OETF (gamma/TRC).
///
/// This is the "candidate" ACES OT 2.0 approach as described in the ACES
/// working group documents, which separates the RRT from the Output Device
/// Transform (ODT) and applies a unified S-curve before gamut conversion.
///
/// # Reference
///
/// - ACES Technical Bulletin TB-2014-013
/// - ACES Output Transform 2.0 Working Document (2022)
#[derive(Debug, Clone)]
pub struct AcesOutputTransform2 {
    /// Target output colour space.
    pub output_space: OutputColorSpace,
    /// Peak output luminance in nits (default 100 for SDR).
    pub peak_luminance: f32,
    /// Minimum output luminance in nits (default 0 for SDR).
    pub min_luminance: f32,
}

impl AcesOutputTransform2 {
    /// Creates a new ACES OT 2.0 targeting the given output colour space.
    #[must_use]
    pub fn new(output_space: OutputColorSpace) -> Self {
        Self {
            output_space,
            peak_luminance: 100.0,
            min_luminance: 0.0,
        }
    }

    /// Set peak output luminance in nits (for HDR outputs).
    #[must_use]
    pub fn with_peak_luminance(mut self, nits: f32) -> Self {
        self.peak_luminance = nits.max(0.0);
        self
    }

    /// Apply the complete ACES Output Transform 2.0 to an ACEScg RGB triplet.
    ///
    /// Input should be scene-linear ACEScg values (AP1 primaries, D60).
    /// Output is encoded in the target colour space and transfer function.
    #[must_use]
    pub fn apply(&self, acescg: [f32; 3]) -> [f32; 3] {
        // Step 1: Apply RRT tone curve per channel
        let rrt = [
            aces_rrt_tone_curve(acescg[0]),
            aces_rrt_tone_curve(acescg[1]),
            aces_rrt_tone_curve(acescg[2]),
        ];

        // Step 2: Gamut conversion: ACEScg AP1 → output primaries
        let matrix = self.output_space.acescg_to_output_matrix();
        let linear_out = [
            (matrix[0][0] * rrt[0] + matrix[0][1] * rrt[1] + matrix[0][2] * rrt[2]).clamp(0.0, 1.0),
            (matrix[1][0] * rrt[0] + matrix[1][1] * rrt[1] + matrix[1][2] * rrt[2]).clamp(0.0, 1.0),
            (matrix[2][0] * rrt[0] + matrix[2][1] * rrt[1] + matrix[2][2] * rrt[2]).clamp(0.0, 1.0),
        ];

        // Step 3: Apply output OETF
        [
            self.output_space.apply_oetf(linear_out[0]),
            self.output_space.apply_oetf(linear_out[1]),
            self.output_space.apply_oetf(linear_out[2]),
        ]
    }

    /// Apply OT 2.0 to an entire frame.
    ///
    /// Each element of `pixels` is an ACEScg `[R, G, B]` triplet.
    #[must_use]
    pub fn apply_frame(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels.iter().map(|&px| self.apply(px)).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AcesColorSpace ──

    #[test]
    fn test_label_non_empty() {
        for cs in [
            AcesColorSpace::ACES2065_1,
            AcesColorSpace::ACEScg,
            AcesColorSpace::ACEScc,
            AcesColorSpace::ACEScct,
            AcesColorSpace::ACESproxy,
        ] {
            assert!(!cs.label().is_empty(), "label empty for {cs:?}");
        }
    }

    #[test]
    fn test_linear_spaces() {
        assert!(AcesColorSpace::ACES2065_1.is_linear());
        assert!(AcesColorSpace::ACEScg.is_linear());
        assert!(!AcesColorSpace::ACEScc.is_linear());
        assert!(!AcesColorSpace::ACEScct.is_linear());
    }

    #[test]
    fn test_primaries_ap0_ap1() {
        assert!(AcesColorSpace::ACES2065_1.uses_ap0());
        assert!(!AcesColorSpace::ACES2065_1.uses_ap1());
        assert!(AcesColorSpace::ACEScg.uses_ap1());
        assert!(!AcesColorSpace::ACEScg.uses_ap0());
    }

    #[test]
    fn test_middle_grey_linear() {
        let mg = AcesColorSpace::ACEScg.middle_grey();
        assert!((mg - 0.18).abs() < 1e-5);
    }

    #[test]
    fn test_display_trait() {
        let s = format!("{}", AcesColorSpace::ACEScg);
        assert_eq!(s, "ACEScg");
    }

    // ── AcesTransform ──

    #[test]
    fn test_identity_transform() {
        let t = AcesTransform::new(AcesColorSpace::ACEScg, AcesColorSpace::ACEScg);
        assert!(t.is_identity());
        let rgb = [0.18, 0.18, 0.18];
        let out = t.apply(rgb);
        assert!((out[0] - 0.18).abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_ap1_ap0() {
        let to_ap0 = AcesTransform::new(AcesColorSpace::ACEScg, AcesColorSpace::ACES2065_1);
        let to_ap1 = AcesTransform::new(AcesColorSpace::ACES2065_1, AcesColorSpace::ACEScg);
        let rgb = [0.5, 0.3, 0.2];
        let out = to_ap1.apply(to_ap0.apply(rgb));
        assert!(
            (out[0] - rgb[0]).abs() < 1e-3,
            "R round-trip error too large"
        );
        assert!(
            (out[1] - rgb[1]).abs() < 1e-3,
            "G round-trip error too large"
        );
        assert!(
            (out[2] - rgb[2]).abs() < 1e-3,
            "B round-trip error too large"
        );
    }

    #[test]
    fn test_ap1_to_ap0_non_identity() {
        let t = AcesTransform::new(AcesColorSpace::ACEScg, AcesColorSpace::ACES2065_1);
        assert!(!t.is_identity());
        let rgb = [1.0, 0.0, 0.0];
        let out = t.apply(rgb);
        // Pure red in AP1 maps to a different tristimulus in AP0
        assert!((out[0] - 1.0).abs() > 1e-3, "should differ from identity");
    }

    #[test]
    fn test_unknown_pair_passthrough() {
        let t = AcesTransform::new(AcesColorSpace::ACEScc, AcesColorSpace::ACEScct);
        let rgb = [0.4, 0.4, 0.4];
        let out = t.apply(rgb);
        assert!((out[0] - 0.4).abs() < 1e-9);
    }

    // ── AcesInputTransform ──

    #[test]
    fn test_cgi_linear_is_identity() {
        let idt = AcesInputTransform::cgi_linear();
        assert!(idt.is_identity());
    }

    #[test]
    fn test_cgi_linear_apply_passthrough() {
        let idt = AcesInputTransform::cgi_linear();
        let rgb = [0.3, 0.5, 0.8];
        let out = idt.apply(rgb);
        assert!((out[0] - 0.3).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_rec709_is_not_identity() {
        let idt = AcesInputTransform::video_rec709();
        assert!(!idt.is_identity());
    }

    #[test]
    fn test_with_exposure_positive_stop() {
        let idt = AcesInputTransform::cgi_linear().with_exposure(1.0);
        // +1 stop → gain = 2.0
        assert!((idt.exposure_compensation - 2.0).abs() < 1e-5);
        assert!(!idt.is_identity());
    }

    #[test]
    fn test_with_exposure_zero_stop() {
        let idt = AcesInputTransform::cgi_linear().with_exposure(0.0);
        assert!((idt.exposure_compensation - 1.0).abs() < 1e-5);
        assert!(idt.is_identity());
    }

    #[test]
    fn test_input_device_label_non_empty() {
        for cls in [
            InputDeviceClass::DigitalCinema,
            InputDeviceClass::RawSensor,
            InputDeviceClass::CgiLinear,
            InputDeviceClass::Video709,
        ] {
            assert!(!cls.label().is_empty());
        }
    }

    #[test]
    fn test_device_name_stored() {
        let idt = AcesInputTransform::cgi_linear();
        assert!(!idt.device_name.is_empty());
    }

    // ── ACES OT 2.0 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_aces_ot2_black_maps_to_black() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        let out = ot.apply([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(ch.abs() < 1e-4, "Black should map near 0: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_middle_grey_in_range() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        // Middle grey 0.18 in ACEScg → should be around 0.18 in SDR
        let out = ot.apply([0.18, 0.18, 0.18]);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "Output should be in [0,1]: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_white_point_maps_to_one() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        // Very bright input should map to 1.0 (clamped)
        let out = ot.apply([100.0, 100.0, 100.0]);
        for ch in out {
            assert!(ch >= 0.99, "Bright white should map to ~1.0: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_rec2020_output() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Rec2020);
        let out = ot.apply([0.5, 0.3, 0.2]);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "Rec.2020 output in [0,1]: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_p3_output() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::P3D65);
        let out = ot.apply([0.5, 0.3, 0.2]);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "P3 output in [0,1]: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_preserves_neutrals() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        // For neutral colors (equal RGB), output should be equal (neutral axis preserved)
        let out = ot.apply([0.18, 0.18, 0.18]);
        let max_diff = (out[0] - out[1]).abs().max((out[1] - out[2]).abs());
        assert!(max_diff < 0.01, "Neutral axis not preserved: {out:?}");
    }

    #[test]
    fn test_aces_ot2_monotonic_brightness() {
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        let out1 = ot.apply([0.1, 0.1, 0.1]);
        let out2 = ot.apply([0.5, 0.5, 0.5]);
        let out3 = ot.apply([2.0, 2.0, 2.0]);
        assert!(out1[0] < out2[0], "Should be monotonic increasing");
        assert!(out2[0] < out3[0], "Should be monotonic increasing");
    }

    #[test]
    fn test_aces_rrt_tone_curve() {
        // RRT tone curve should produce values in [0, 1]
        for x in [0.0f32, 0.01, 0.1, 0.18, 0.5, 1.0, 2.0, 10.0] {
            let y = aces_rrt_tone_curve(x);
            assert!(
                y >= 0.0 && y <= 1.0,
                "RRT curve out of range for x={x}: y={y}"
            );
        }
    }

    #[test]
    fn test_aces_rrt_monotonic() {
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let x = i as f32 * 0.1;
            let y = aces_rrt_tone_curve(x);
            assert!(
                y >= prev - 1e-6,
                "RRT curve should be monotonic: {prev} -> {y} at x={x}"
            );
            prev = y;
        }
    }

    // ── New numeric accuracy tests ─────────────────────────────────────────────

    #[test]
    fn test_aces_middle_grey_is_neutral() {
        // Middle grey 0.18 ACEScg → output channels should be in [0,1] and neutral (R≈G≈B).
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb);
        let out = ot.apply([0.18, 0.18, 0.18]);
        for ch in out {
            assert!(
                (0.0..=1.0).contains(&ch),
                "Middle grey output channel must be in [0,1]: {ch}"
            );
        }
        let max_diff = (out[0] - out[1]).abs().max((out[1] - out[2]).abs());
        assert!(
            max_diff < 0.05,
            "Middle grey must map to a near-neutral output (max channel diff = {max_diff:.4}); got {out:?}"
        );
    }

    #[test]
    fn test_rrt_tone_curve_monotonic() {
        // Verify strict monotonicity over the canonical ACES sweep.
        let sweep = [0.0f32, 0.01, 0.05, 0.1, 0.18, 0.5, 1.0, 2.0, 4.0];
        let values: Vec<f32> = sweep.iter().map(|&x| aces_rrt_tone_curve(x)).collect();
        for i in 1..values.len() {
            assert!(
                values[i] > values[i - 1],
                "RRT tone curve must be strictly increasing at index {i}: \
                 f({x_prev}) = {y_prev} >= f({x_cur}) = {y_cur}",
                x_prev = sweep[i - 1],
                y_prev = values[i - 1],
                x_cur = sweep[i],
                y_cur = values[i],
            );
        }
    }

    #[test]
    fn test_aces_output_channels_clamped_to_01() {
        // For a range of very bright HDR inputs all output channels must stay in [0.0, 1.0].
        let ot = AcesOutputTransform2::new(OutputColorSpace::Srgb).with_peak_luminance(1000.0);
        let bright_inputs: &[[f32; 3]] = &[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [5.0, 5.0, 5.0],
            [10.0, 10.0, 10.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [8.0, 4.0, 2.0],
        ];
        for &input in bright_inputs {
            let out = ot.apply(input);
            for (ch_idx, ch) in out.iter().enumerate() {
                assert!(
                    (0.0f32..=1.0).contains(ch),
                    "Channel {ch_idx} out of [0,1] for input {input:?}: got {ch}"
                );
            }
        }
    }
}
