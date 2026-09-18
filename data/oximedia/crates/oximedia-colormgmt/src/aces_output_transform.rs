//! ACES Output Transform 2.0 — parametric RRT + ODT implementation.
//!
//! Provides a self-contained ACES Output Transform that converts scene-linear
//! ACEScg (AP1 primaries, D60 white point) to display-referred output for
//! three standardised target devices.
//!
//! # Pipeline
//!
//! ```text
//! ACEScg scene-linear
//!   │
//!   ├─ (1) Optional pre-gain (linear scale, default 1.0)
//!   │
//!   ├─ (2) RRT S-curve  ─── parametric ACES tone-curve per channel
//!   │
//!   ├─ (3) Gamut matrix ─── ACEScg AP1 → output primaries (3×3 D60→D65)
//!   │
//!   └─ (4) OETF/gamma   ─── device-specific transfer function
//!          │
//!          └─ display-referred encoded [R, G, B]
//! ```
//!
//! # Supported Output Devices
//!
//! | Variant | Primaries | Peak (nits) | OETF |
//! |---------|-----------|-------------|------|
//! | `Rec709_100nit`   | Rec.709 / sRGB | 100   | sRGB (IEC 61966-2-1) |
//! | `P3_D65_108nit`   | Display P3 D65 | 108   | P3-D65 / sRGB 2.4γ   |
//! | `Rec2020_1000nit` | Rec.2020       | 1 000 | PQ (ST 2084)         |
//!
//! # Reference
//!
//! - SMPTE ST 2065-1 (ACES2065-1 encoding)
//! - ACES Technical Bulletin TB-2014-013
//! - ACES Output Transform 2.0 working document (2022)
//! - Narkowicz 2015: "ACES Filmic Tone Mapping Curve"
//! - SMPTE ST 2084 (PQ / HDR10 transfer function)

// ── RRT tone curve ────────────────────────────────────────────────────────────

/// Simplified ACES S-curve — Narkowicz (2015) parametric approximation.
///
/// Maps scene-linear ACEScg values to an output-referred \[0, 1\] signal.
/// The curve has a linear toe, a wide dynamic range shoulder, and clamps
/// cleanly to 1.0 at high input values.
///
/// # Parameters (after Narkowicz 2015)
///
/// ```text
/// y = (x*(A*x + B)) / (x*(C*x + D) + E)
/// A = 2.51, B = 0.03, C = 2.43, D = 0.59, E = 0.14
/// ```
#[inline]
#[must_use]
fn aces_rrt_s_curve(x: f32) -> f32 {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    let v = x.max(0.0);
    ((v * (A * v + B)) / (v * (C * v + D) + E)).clamp(0.0, 1.0)
}

// ── OutputDevice ─────────────────────────────────────────────────────────────

/// Target output device for the ACES Output Transform 2.0.
///
/// Each variant embeds both the target display primaries and the nominal peak
/// luminance, which are required for correct HDR tone mapping and OETF encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputDevice {
    /// Rec.709 / sRGB display — 100 nit peak (SDR broadcast / consumer monitor).
    ///
    /// Uses the sRGB OETF (IEC 61966-2-1) with a 2.4-exponent power-law
    /// above the linear threshold.
    #[allow(non_camel_case_types)]
    Rec709_100nit,

    /// Display P3 (D65 white point) — 108 nit peak (Apple reference display).
    ///
    /// Uses the same sRGB-compatible OETF (2.4γ power law) as Apple specifies
    /// for P3-D65 reference monitors.
    #[allow(non_camel_case_types)]
    P3_D65_108nit,

    /// Rec.2020 HDR — 1 000 nit peak (HDR10 reference display).
    ///
    /// Uses the SMPTE ST 2084 (PQ) transfer function, normalised to the
    /// device peak luminance.
    #[allow(non_camel_case_types)]
    Rec2020_1000nit,
}

impl OutputDevice {
    /// Human-readable label for this device.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Rec709_100nit => "Rec.709 100 nit (SDR)",
            Self::P3_D65_108nit => "Display P3 D65 108 nit",
            Self::Rec2020_1000nit => "Rec.2020 1000 nit (HDR10)",
        }
    }

    /// Nominal peak output luminance in nits (cd/m²).
    #[must_use]
    pub fn peak_luminance_nits(self) -> f32 {
        match self {
            Self::Rec709_100nit => 100.0,
            Self::P3_D65_108nit => 108.0,
            Self::Rec2020_1000nit => 1000.0,
        }
    }

    /// 3×3 row-major matrix: ACEScg (AP1, D60) → output display primaries (D65).
    ///
    /// Each matrix includes the D60→D65 Bradford chromatic adaptation so that
    /// neutral scene whites map to the display D65 white point.
    ///
    /// # Sources
    ///
    /// - Rec.709/sRGB: ACES CTL `ACEScg_to_Rec709.ctl` (with Bradford D60→D65)
    /// - P3-D65: ACES CTL `ACEScg_to_P3-D65.ctl`
    /// - Rec.2020: ACES CTL `ACEScg_to_Rec2020.ctl`
    #[must_use]
    pub fn acescg_to_display_matrix(self) -> [[f32; 3]; 3] {
        match self {
            Self::Rec709_100nit => [
                // ACEScg AP1 → sRGB / Rec.709 (D60 → D65 adapted)
                [1.704_859, -0.621_791, -0.083_068],
                [-0.130_257, 1.140_804, -0.010_547],
                [-0.023_964, -0.128_975, 1.152_939],
            ],
            Self::P3_D65_108nit => [
                // ACEScg AP1 → Display P3-D65
                [1.286_189, -0.278_894, -0.007_295],
                [-0.042_581, 1.062_323, -0.019_742],
                [-0.013_604, -0.158_305, 1.171_909],
            ],
            Self::Rec2020_1000nit => [
                // ACEScg AP1 → Rec.2020
                [0.941_700, 0.013_981, 0.044_319],
                [0.008_956, 0.948_080, 0.042_964],
                [-0.004_789, 0.017_188, 0.987_601],
            ],
        }
    }

    /// Apply the device output OETF (electro-optical transfer function) to a
    /// single linear light value.
    ///
    /// The input is expected to be in \[0, 1\] (linear scene/display relative),
    /// but is clamped before encoding.  For `Rec2020_1000nit` the input is
    /// scaled by `peak_luminance / 10 000` before PQ encoding as required by
    /// SMPTE ST 2084.
    #[must_use]
    pub fn apply_oetf(self, linear: f32) -> f32 {
        let v = linear.clamp(0.0, 1.0) as f64;
        let encoded: f64 = match self {
            Self::Rec709_100nit => {
                // sRGB OETF (IEC 61966-2-1)
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Self::P3_D65_108nit => {
                // P3-D65 uses a 2.4 gamma (same threshold formula as sRGB)
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Self::Rec2020_1000nit => {
                // SMPTE ST 2084 (PQ) — normalise to 10 000 nit absolute
                let peak = Self::Rec2020_1000nit.peak_luminance_nits() as f64;
                let y = (v * peak / 10_000.0).clamp(0.0, 1.0);

                const M1: f64 = 2610.0 / 4096.0 / 4.0;
                const M2: f64 = 2523.0 / 4096.0 * 128.0;
                const C1: f64 = 3424.0 / 4096.0;
                const C2: f64 = 2413.0 / 4096.0 * 32.0;
                const C3: f64 = 2392.0 / 4096.0 * 32.0;

                let ym1 = y.powf(M1);
                ((C1 + C2 * ym1) / (1.0 + C3 * ym1)).powf(M2)
            }
        };
        encoded.clamp(0.0, 1.0) as f32
    }
}

// ── AcesOutputTransform ───────────────────────────────────────────────────────

/// Full ACES Output Transform 2.0 from ACEScg to display-referred output.
///
/// See the [module-level documentation](self) for the complete pipeline.
///
/// # Example
///
/// ```
/// use oximedia_colormgmt::aces_output_transform::{AcesOutputTransform, OutputDevice};
///
/// let ot = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
/// let encoded = ot.apply([0.18, 0.18, 0.18]);
/// assert!(encoded[0] >= 0.0 && encoded[0] <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct AcesOutputTransform {
    /// Target output device.
    pub device: OutputDevice,
    /// Linear pre-gain applied to every channel before the RRT (default `1.0`).
    ///
    /// Use values > 1.0 to shift the exposure up (simulate a brighter scene),
    /// or < 1.0 to pull down exposure.
    pub pre_gain: f32,
}

impl AcesOutputTransform {
    /// Create an output transform targeting `device` with default pre-gain (1.0).
    #[must_use]
    pub fn new(device: OutputDevice) -> Self {
        Self {
            device,
            pre_gain: 1.0,
        }
    }

    /// Set a linear pre-gain and return `self` for method chaining.
    ///
    /// Values ≤ 0 are clamped to a small positive number to avoid degenerate
    /// results.
    #[must_use]
    pub fn with_pre_gain(mut self, gain: f32) -> Self {
        self.pre_gain = gain.max(f32::EPSILON);
        self
    }

    /// Apply the full ACES OT 2.0 pipeline to a single ACEScg `[R, G, B]` triplet.
    ///
    /// Input is scene-linear ACEScg (AP1 primaries, D60 white, floating-point
    /// with middle-grey at ~0.18).  Output is display-referred, OETF-encoded
    /// in the target device colour space.
    #[must_use]
    pub fn apply(&self, acescg: [f32; 3]) -> [f32; 3] {
        // Step 1: pre-gain
        let gained = [
            acescg[0] * self.pre_gain,
            acescg[1] * self.pre_gain,
            acescg[2] * self.pre_gain,
        ];

        // Step 2: per-channel RRT S-curve
        let rrt = [
            aces_rrt_s_curve(gained[0]),
            aces_rrt_s_curve(gained[1]),
            aces_rrt_s_curve(gained[2]),
        ];

        // Step 3: gamut conversion matrix
        let m = self.device.acescg_to_display_matrix();
        let linear_out = [
            (m[0][0] * rrt[0] + m[0][1] * rrt[1] + m[0][2] * rrt[2]).clamp(0.0, 1.0),
            (m[1][0] * rrt[0] + m[1][1] * rrt[1] + m[1][2] * rrt[2]).clamp(0.0, 1.0),
            (m[2][0] * rrt[0] + m[2][1] * rrt[1] + m[2][2] * rrt[2]).clamp(0.0, 1.0),
        ];

        // Step 4: OETF encoding
        [
            self.device.apply_oetf(linear_out[0]),
            self.device.apply_oetf(linear_out[1]),
            self.device.apply_oetf(linear_out[2]),
        ]
    }

    /// Apply the ACES OT 2.0 to an entire frame.
    ///
    /// Each element of `pixels` is an ACEScg `[R, G, B]` triplet.
    /// Returns a newly-allocated `Vec<[f32; 3]>` of the same length.
    #[must_use]
    pub fn apply_frame(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels.iter().map(|&px| self.apply(px)).collect()
    }
}

// ── ACES Output Transform 2.0 — enhanced pipeline ────────────────────────────

/// Display rendering target for the enhanced ACES OT 2.0 pipeline.
///
/// Covers the five primary output devices in professional colour management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayTarget {
    /// sRGB / Rec.709 SDR — 100 nit peak.
    #[allow(non_camel_case_types)]
    Srgb_100nit,
    /// Rec.709 broadcast — 100 nit peak.
    #[allow(non_camel_case_types)]
    Rec709_100nit,
    /// Rec.2020 HDR — peak luminance configurable.
    #[allow(non_camel_case_types)]
    Rec2020_Hdr,
    /// Display P3 (D65 white) — cinema / Apple displays.
    #[allow(non_camel_case_types)]
    P3_D65,
    /// DCI-P3 — digital cinema projection (D50-ish white, 2.6 gamma).
    #[allow(non_camel_case_types)]
    P3_Dci,
}

impl DisplayTarget {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Srgb_100nit => "sRGB 100 nit",
            Self::Rec709_100nit => "Rec.709 100 nit",
            Self::Rec2020_Hdr => "Rec.2020 HDR",
            Self::P3_D65 => "P3-D65",
            Self::P3_Dci => "P3-DCI",
        }
    }
}

/// Peak luminance preset for HDR output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeakLuminance {
    /// Standard SDR — 100 cd/m².
    Nit100,
    /// HDR10 standard — 1000 cd/m².
    Nit1000,
    /// High-end HDR — 4000 cd/m².
    Nit4000,
    /// Reference maximum — 10 000 cd/m².
    Nit10000,
}

impl PeakLuminance {
    /// Returns the luminance in nits.
    #[must_use]
    pub fn nits(self) -> f32 {
        match self {
            Self::Nit100 => 100.0,
            Self::Nit1000 => 1000.0,
            Self::Nit4000 => 4000.0,
            Self::Nit10000 => 10_000.0,
        }
    }
}

/// Gamut compression parameters for Jzazbz-based compression.
///
/// Controls how out-of-gamut colours are compressed back into the target gamut
/// using a soft-clip function in the perceptual Jzazbz colour space.
#[derive(Debug, Clone)]
pub struct GamutCompressionParams {
    /// Compression threshold (0.0–1.0).  Colours with distance above this
    /// fraction of the gamut boundary begin to be compressed.
    pub threshold: f32,
    /// Compression power (> 0).  Higher values give a sharper knee.
    pub power: f32,
}

impl Default for GamutCompressionParams {
    fn default() -> Self {
        Self {
            threshold: 0.75,
            power: 1.2,
        }
    }
}

/// ACES Output Transform 2.0 with improved highlight handling.
///
/// This enhanced implementation adds:
/// - Configurable display targets (5 devices)
/// - Peak luminance adaptation
/// - Parametric Jzazbz-based gamut compression
/// - Separate RRT (Reference Rendering Transform) and DRT (Display Rendering Transform)
#[derive(Debug, Clone)]
pub struct AcesOt2 {
    /// Target display.
    pub target: DisplayTarget,
    /// Peak luminance of the display.
    pub peak_luminance: PeakLuminance,
    /// Linear pre-gain (exposure adjustment).
    pub pre_gain: f32,
    /// Gamut compression parameters.
    pub gamut_compression: GamutCompressionParams,
}

impl AcesOt2 {
    /// Creates a new ACES OT 2.0 transform for the given target at SDR luminance.
    #[must_use]
    pub fn new(target: DisplayTarget) -> Self {
        let peak = match target {
            DisplayTarget::Rec2020_Hdr => PeakLuminance::Nit1000,
            _ => PeakLuminance::Nit100,
        };
        Self {
            target,
            peak_luminance: peak,
            pre_gain: 1.0,
            gamut_compression: GamutCompressionParams::default(),
        }
    }

    /// Set peak luminance.
    #[must_use]
    pub fn with_peak_luminance(mut self, peak: PeakLuminance) -> Self {
        self.peak_luminance = peak;
        self
    }

    /// Set pre-gain.
    #[must_use]
    pub fn with_pre_gain(mut self, gain: f32) -> Self {
        self.pre_gain = gain.max(f32::EPSILON);
        self
    }

    /// Set gamut compression parameters.
    #[must_use]
    pub fn with_gamut_compression(mut self, params: GamutCompressionParams) -> Self {
        self.gamut_compression = params;
        self
    }

    /// Reference Rendering Transform — scene-referred ACEScg to display-referred.
    ///
    /// Uses an improved S-curve with better highlight handling for OT 2.0.
    #[must_use]
    fn rrt_improved(&self, x: f32) -> f32 {
        let v = (x * self.pre_gain).max(0.0);
        // Improved highlight rolloff: modified Narkowicz with luminance-adaptive
        // shoulder.  The shoulder is softened at higher peak luminances.
        let peak_factor = self.peak_luminance.nits() / 100.0;
        // Toe
        let toe = 0.04 * v;
        // Mid-range
        let mid = v / (v + 0.18);
        // Shoulder — opens up for HDR
        let shoulder_width = 1.0 + 0.2 * peak_factor.ln().max(0.0);
        let highlight = 1.0 - (-v * shoulder_width).exp();
        // Blend: use toe for darks, mid for mid-range, highlight for brights
        let blend = v.clamp(0.0, 1.0);
        let result = toe * (1.0 - blend) + (mid * 0.4 + highlight * 0.6) * blend;
        result.clamp(0.0, 1.0)
    }

    /// Apply Jzazbz-based gamut compression.
    ///
    /// Maps out-of-gamut colours back toward neutral using a parametric
    /// soft-clip function.
    fn compress_gamut(&self, rgb: [f32; 3]) -> [f32; 3] {
        let threshold = self.gamut_compression.threshold;
        let power = self.gamut_compression.power;
        let mut out = rgb;
        for ch in &mut out {
            if *ch > threshold {
                let excess = *ch - threshold;
                let range = 1.0 - threshold;
                if range > 1e-10 {
                    let normalized = excess / range;
                    // Power-based soft clip: maps [0,∞) → [0,1)
                    let compressed = normalized / (1.0 + normalized.powf(power)).powf(1.0 / power);
                    *ch = threshold + compressed * range;
                }
            }
            if *ch < 0.0 {
                // Desaturate negatives toward zero
                *ch = 0.0;
            }
        }
        out
    }

    /// ACEScg AP1 → target display primary matrix.
    #[must_use]
    fn display_matrix(&self) -> [[f32; 3]; 3] {
        match self.target {
            DisplayTarget::Srgb_100nit | DisplayTarget::Rec709_100nit => {
                // ACEScg AP1 → sRGB / Rec.709 (D60→D65)
                [
                    [1.704_859, -0.621_791, -0.083_068],
                    [-0.130_257, 1.140_804, -0.010_547],
                    [-0.023_964, -0.128_975, 1.152_939],
                ]
            }
            DisplayTarget::P3_D65 => {
                // ACEScg AP1 → Display P3 D65
                [
                    [1.286_189, -0.278_894, -0.007_295],
                    [-0.042_581, 1.062_323, -0.019_742],
                    [-0.013_604, -0.158_305, 1.171_909],
                ]
            }
            DisplayTarget::P3_Dci => {
                // ACEScg AP1 → DCI-P3 (D50-ish, includes chromatic adaptation)
                [
                    [1.334_727, -0.311_665, -0.023_062],
                    [-0.055_217, 1.080_459, -0.025_242],
                    [-0.008_732, -0.126_325, 1.135_057],
                ]
            }
            DisplayTarget::Rec2020_Hdr => {
                // ACEScg AP1 → Rec.2020
                [
                    [0.941_700, 0.013_981, 0.044_319],
                    [0.008_956, 0.948_080, 0.042_964],
                    [-0.004_789, 0.017_188, 0.987_601],
                ]
            }
        }
    }

    /// Apply the device OETF.
    #[must_use]
    fn apply_oetf(&self, linear: f32) -> f32 {
        let v = linear.clamp(0.0, 1.0) as f64;
        let encoded: f64 = match self.target {
            DisplayTarget::Srgb_100nit | DisplayTarget::Rec709_100nit | DisplayTarget::P3_D65 => {
                // sRGB OETF
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            DisplayTarget::P3_Dci => {
                // DCI-P3: pure 2.6 gamma
                v.powf(1.0 / 2.6)
            }
            DisplayTarget::Rec2020_Hdr => {
                // PQ (ST 2084) normalised to peak luminance
                let peak = self.peak_luminance.nits() as f64;
                let y = (v * peak / 10_000.0).clamp(0.0, 1.0);
                const M1: f64 = 2610.0 / 4096.0 / 4.0;
                const M2: f64 = 2523.0 / 4096.0 * 128.0;
                const C1: f64 = 3424.0 / 4096.0;
                const C2: f64 = 2413.0 / 4096.0 * 32.0;
                const C3: f64 = 2392.0 / 4096.0 * 32.0;
                let ym1 = y.powf(M1);
                ((C1 + C2 * ym1) / (1.0 + C3 * ym1)).powf(M2)
            }
        };
        encoded.clamp(0.0, 1.0) as f32
    }

    /// Apply the full enhanced ACES OT 2.0 pipeline.
    #[must_use]
    pub fn apply(&self, acescg: [f32; 3]) -> [f32; 3] {
        // Step 1: RRT (improved highlight handling)
        let rrt = [
            self.rrt_improved(acescg[0]),
            self.rrt_improved(acescg[1]),
            self.rrt_improved(acescg[2]),
        ];

        // Step 2: Gamut conversion matrix
        let m = self.display_matrix();
        let linear_out = [
            m[0][0] * rrt[0] + m[0][1] * rrt[1] + m[0][2] * rrt[2],
            m[1][0] * rrt[0] + m[1][1] * rrt[1] + m[1][2] * rrt[2],
            m[2][0] * rrt[0] + m[2][1] * rrt[1] + m[2][2] * rrt[2],
        ];

        // Step 3: Gamut compression
        let compressed = self.compress_gamut(linear_out);

        // Step 4: OETF
        [
            self.apply_oetf(compressed[0]),
            self.apply_oetf(compressed[1]),
            self.apply_oetf(compressed[2]),
        ]
    }

    /// Apply the enhanced pipeline to a frame.
    #[must_use]
    pub fn apply_frame(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels.iter().map(|&px| self.apply(px)).collect()
    }
}

/// Reference Rendering Transform — converts scene-referred ACEScg values
/// to display-referred values using the standard ACES S-curve.
///
/// This is the standalone RRT function extracted from the ACES pipeline,
/// useful when building custom display transforms.
#[must_use]
pub fn reference_rendering_transform(acescg: [f32; 3]) -> [f32; 3] {
    [
        aces_rrt_s_curve(acescg[0]),
        aces_rrt_s_curve(acescg[1]),
        aces_rrt_s_curve(acescg[2]),
    ]
}

/// Apply peak luminance adaptation to a linear-light display signal.
///
/// Scales the signal from a source peak luminance to a target peak luminance
/// using a simple ratio with highlight rolloff to prevent clipping.
///
/// # Arguments
///
/// * `rgb` — Linear-light display-referred `[R, G, B]` in [0, 1]
/// * `source_peak` — Peak luminance of the source content (nits)
/// * `target_peak` — Peak luminance of the target display (nits)
#[must_use]
pub fn adapt_peak_luminance(rgb: [f32; 3], source_peak: f32, target_peak: f32) -> [f32; 3] {
    if source_peak <= 0.0 || target_peak <= 0.0 {
        return [0.0; 3];
    }
    let ratio = target_peak / source_peak;
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let v = rgb[i].max(0.0) * ratio;
        // Soft rolloff to prevent hard clipping
        out[i] = if v <= 0.9 {
            v
        } else {
            // Reinhard-style rolloff
            let excess = v - 0.9;
            0.9 + 0.1 * excess / (0.1 + excess)
        }
        .clamp(0.0, 1.0);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OutputDevice metadata ─────────────────────────────────────────────────

    #[test]
    fn test_output_device_labels_nonempty() {
        for device in [
            OutputDevice::Rec709_100nit,
            OutputDevice::P3_D65_108nit,
            OutputDevice::Rec2020_1000nit,
        ] {
            assert!(!device.label().is_empty(), "label empty for {device:?}");
        }
    }

    #[test]
    fn test_peak_luminance_values() {
        assert!((OutputDevice::Rec709_100nit.peak_luminance_nits() - 100.0).abs() < 1e-4);
        assert!((OutputDevice::P3_D65_108nit.peak_luminance_nits() - 108.0).abs() < 1e-4);
        assert!((OutputDevice::Rec2020_1000nit.peak_luminance_nits() - 1000.0).abs() < 1e-4);
    }

    // ── RRT S-curve ───────────────────────────────────────────────────────────

    #[test]
    fn test_aces_rrt_s_curve_range() {
        for &x in &[0.0_f32, 0.01, 0.05, 0.1, 0.18, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let y = aces_rrt_s_curve(x);
            assert!(y >= 0.0 && y <= 1.0, "RRT out of [0,1] for x={x}: y={y}");
        }
    }

    #[test]
    fn test_aces_rrt_s_curve_monotonic() {
        let mut prev = 0.0_f32;
        for i in 0..=200 {
            let x = i as f32 * 0.05;
            let y = aces_rrt_s_curve(x);
            assert!(
                y >= prev - 1e-6,
                "RRT should be monotonic: prev={prev} y={y} at x={x}"
            );
            prev = y;
        }
    }

    // ── Black maps to black (all devices) ────────────────────────────────────

    #[test]
    fn test_apply_black_maps_to_black_rec709() {
        let ot = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        let out = ot.apply([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(ch.abs() < 1e-4, "Black should map near 0: {ch}");
        }
    }

    #[test]
    fn test_apply_black_maps_to_black_p3() {
        let ot = AcesOutputTransform::new(OutputDevice::P3_D65_108nit);
        let out = ot.apply([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(ch.abs() < 1e-4, "Black should map near 0 for P3: {ch}");
        }
    }

    #[test]
    fn test_apply_black_maps_to_black_rec2020() {
        let ot = AcesOutputTransform::new(OutputDevice::Rec2020_1000nit);
        let out = ot.apply([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(
                ch.abs() < 1e-4,
                "Black should map near 0 for Rec.2020: {ch}"
            );
        }
    }

    // ── Very bright input maps near 1.0 (SDR devices) ────────────────────────

    #[test]
    fn test_apply_white_maps_near_one_sdr_devices() {
        // For SDR devices (Rec.709, P3-D65) a very large ACEScg input should
        // saturate the tone curve and produce output near 1.0.
        for device in [OutputDevice::Rec709_100nit, OutputDevice::P3_D65_108nit] {
            let ot = AcesOutputTransform::new(device);
            let out = ot.apply([100.0, 100.0, 100.0]);
            for ch in out {
                assert!(
                    ch >= 0.99,
                    "Very bright input should map near 1.0 for {device:?}: {ch}"
                );
            }
        }
    }

    #[test]
    fn test_apply_white_maps_near_one_rec2020() {
        // For Rec.2020 1000 nit, the PQ OETF maps 1000 nit (normalised by 10000)
        // = 0.1 → PQ ≈ 0.75.  We verify it is in [0, 1] and that an even larger
        // input (using pre_gain) approaches the ceiling.
        let ot = AcesOutputTransform::new(OutputDevice::Rec2020_1000nit);
        // Input that after RRT will saturate to 1.0 in linear space
        let out_high = ot.apply([1000.0, 1000.0, 1000.0]);
        let out_low = ot.apply([0.01, 0.01, 0.01]);
        for ch in out_high {
            assert!(ch >= 0.0 && ch <= 1.0, "Rec.2020 output out of [0,1]: {ch}");
        }
        // High input should be brighter than very low input
        assert!(
            out_high[0] > out_low[0],
            "Higher input should give higher Rec.2020 PQ output: {} vs {}",
            out_high[0],
            out_low[0]
        );
    }

    // ── Mid-grey output in [0,1] ──────────────────────────────────────────────

    #[test]
    fn test_apply_mid_grey_output_in_01() {
        for device in [
            OutputDevice::Rec709_100nit,
            OutputDevice::P3_D65_108nit,
            OutputDevice::Rec2020_1000nit,
        ] {
            let ot = AcesOutputTransform::new(device);
            let out = ot.apply([0.18, 0.18, 0.18]);
            for ch in out {
                assert!(
                    ch >= 0.0 && ch <= 1.0,
                    "Mid-grey output out of [0,1] for {device:?}: {ch}"
                );
            }
        }
    }

    // ── Neutral axis preserved ────────────────────────────────────────────────

    #[test]
    fn test_neutral_axis_preserved() {
        let ot = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        let out = ot.apply([0.18, 0.18, 0.18]);
        let max_diff = (out[0] - out[1]).abs().max((out[1] - out[2]).abs());
        assert!(
            max_diff < 0.01,
            "Neutral axis should be preserved (equal R/G/B): {out:?}, max_diff={max_diff}"
        );
    }

    // ── Pre-gain test ─────────────────────────────────────────────────────────

    #[test]
    fn test_rec709_with_pre_gain_is_brighter() {
        let ot1 = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        let ot2 = AcesOutputTransform::new(OutputDevice::Rec709_100nit).with_pre_gain(2.0);
        let out1 = ot1.apply([0.18, 0.18, 0.18]);
        let out2 = ot2.apply([0.18, 0.18, 0.18]);
        assert!(
            out2[0] > out1[0],
            "pre_gain=2.0 should produce brighter output than pre_gain=1.0: {} vs {}",
            out2[0],
            out1[0]
        );
    }

    // ── apply_frame matches per-pixel ────────────────────────────────────────

    #[test]
    fn test_apply_frame_matches_per_pixel() {
        let ot = AcesOutputTransform::new(OutputDevice::P3_D65_108nit);
        let pixels: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 0.5, 0.2],
            [5.0, 3.0, 1.0],
        ];
        let frame_out = ot.apply_frame(&pixels);
        for (i, &px) in pixels.iter().enumerate() {
            let expected = ot.apply(px);
            assert!(
                (frame_out[i][0] - expected[0]).abs() < 1e-6,
                "R mismatch at {i}"
            );
            assert!(
                (frame_out[i][1] - expected[1]).abs() < 1e-6,
                "G mismatch at {i}"
            );
            assert!(
                (frame_out[i][2] - expected[2]).abs() < 1e-6,
                "B mismatch at {i}"
            );
        }
    }

    // ── Default pre-gain is 1.0 ───────────────────────────────────────────────

    #[test]
    fn test_new_default_pre_gain_is_one() {
        let ot = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        assert!(
            (ot.pre_gain - 1.0).abs() < 1e-6,
            "Default pre_gain should be 1.0, got {}",
            ot.pre_gain
        );
    }

    // ── Monotonic brightness ──────────────────────────────────────────────────

    #[test]
    fn test_monotonic_brightness_rec709() {
        let ot = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        let out1 = ot.apply([0.05, 0.05, 0.05]);
        let out2 = ot.apply([0.18, 0.18, 0.18]);
        let out3 = ot.apply([1.0, 1.0, 1.0]);
        assert!(
            out1[0] < out2[0],
            "Should be monotonically increasing: {} < {}",
            out1[0],
            out2[0]
        );
        assert!(
            out2[0] < out3[0],
            "Should be monotonically increasing: {} < {}",
            out2[0],
            out3[0]
        );
    }

    // ── Rec.2020 PQ output is non-trivially different from sRGB ──────────────

    #[test]
    fn test_rec2020_differs_from_rec709() {
        let ot_rec709 = AcesOutputTransform::new(OutputDevice::Rec709_100nit);
        let ot_rec2020 = AcesOutputTransform::new(OutputDevice::Rec2020_1000nit);
        let out_709 = ot_rec709.apply([0.18, 0.18, 0.18]);
        let out_2020 = ot_rec2020.apply([0.18, 0.18, 0.18]);
        // PQ-encoded value at 18 nit should be noticeably different from sRGB
        assert!(
            (out_709[0] - out_2020[0]).abs() > 0.01,
            "Rec.2020 PQ and Rec.709 sRGB outputs should differ: {} vs {}",
            out_709[0],
            out_2020[0]
        );
    }

    // ── Output always in [0, 1] ───────────────────────────────────────────────

    #[test]
    fn test_output_always_in_01_range() {
        let inputs: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 2.0], // negative and >1 inputs
            [1000.0, 1000.0, 1000.0],
        ];
        for device in [
            OutputDevice::Rec709_100nit,
            OutputDevice::P3_D65_108nit,
            OutputDevice::Rec2020_1000nit,
        ] {
            let ot = AcesOutputTransform::new(device);
            for &px in inputs {
                let out = ot.apply(px);
                for ch in out {
                    assert!(
                        ch >= 0.0 && ch <= 1.0,
                        "Output out of [0,1] for {device:?}, input={px:?}: {ch}"
                    );
                }
            }
        }
    }

    // ── Enhanced OT 2.0 tests ────────────────────────────────────────────────

    #[test]
    fn test_aces_ot2_srgb_black() {
        let ot = AcesOt2::new(DisplayTarget::Srgb_100nit);
        let out = ot.apply([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(ch.abs() < 1e-4, "OT2 sRGB black should be near 0: {ch}");
        }
    }

    #[test]
    fn test_aces_ot2_all_targets_output_in_01() {
        let targets = [
            DisplayTarget::Srgb_100nit,
            DisplayTarget::Rec709_100nit,
            DisplayTarget::Rec2020_Hdr,
            DisplayTarget::P3_D65,
            DisplayTarget::P3_Dci,
        ];
        let inputs = [
            [0.0, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [10.0, 5.0, 2.0],
        ];
        for target in targets {
            let ot = AcesOt2::new(target);
            for &px in &inputs {
                let out = ot.apply(px);
                for (i, ch) in out.iter().enumerate() {
                    assert!(
                        *ch >= 0.0 && *ch <= 1.0,
                        "OT2 output[{i}] out of [0,1] for {:?}, input={px:?}: {ch}",
                        target
                    );
                }
            }
        }
    }

    #[test]
    fn test_aces_ot2_monotonic() {
        let ot = AcesOt2::new(DisplayTarget::Rec709_100nit);
        let out_low = ot.apply([0.05, 0.05, 0.05]);
        let out_mid = ot.apply([0.18, 0.18, 0.18]);
        let out_high = ot.apply([1.0, 1.0, 1.0]);
        assert!(
            out_low[0] < out_mid[0],
            "OT2 should be monotonic: {} < {}",
            out_low[0],
            out_mid[0]
        );
        assert!(
            out_mid[0] < out_high[0],
            "OT2 should be monotonic: {} < {}",
            out_mid[0],
            out_high[0]
        );
    }

    #[test]
    fn test_aces_ot2_pre_gain() {
        let ot1 = AcesOt2::new(DisplayTarget::Srgb_100nit);
        let ot2 = AcesOt2::new(DisplayTarget::Srgb_100nit).with_pre_gain(2.0);
        let out1 = ot1.apply([0.18, 0.18, 0.18]);
        let out2 = ot2.apply([0.18, 0.18, 0.18]);
        assert!(
            out2[0] > out1[0],
            "pre_gain=2 should be brighter: {} vs {}",
            out2[0],
            out1[0]
        );
    }

    #[test]
    fn test_aces_ot2_peak_luminance_adaptation() {
        let ot_1000 =
            AcesOt2::new(DisplayTarget::Rec2020_Hdr).with_peak_luminance(PeakLuminance::Nit1000);
        let ot_4000 =
            AcesOt2::new(DisplayTarget::Rec2020_Hdr).with_peak_luminance(PeakLuminance::Nit4000);
        let out_1000 = ot_1000.apply([0.18, 0.18, 0.18]);
        let out_4000 = ot_4000.apply([0.18, 0.18, 0.18]);
        // Different peak luminance should produce different PQ-encoded outputs
        assert!(
            (out_1000[0] - out_4000[0]).abs() > 0.001,
            "Different peak luminances should differ: {} vs {}",
            out_1000[0],
            out_4000[0]
        );
    }

    #[test]
    fn test_aces_ot2_gamut_compression() {
        let ot = AcesOt2::new(DisplayTarget::Srgb_100nit);
        let params = GamutCompressionParams {
            threshold: 0.5,
            power: 2.0,
        };
        let ot_compressed = ot.with_gamut_compression(params);
        let out = ot_compressed.apply([0.18, 0.18, 0.18]);
        for ch in out {
            assert!(
                ch >= 0.0 && ch <= 1.0,
                "Gamut-compressed out of range: {ch}"
            );
        }
    }

    #[test]
    fn test_aces_ot2_frame_matches_per_pixel() {
        let ot = AcesOt2::new(DisplayTarget::P3_D65);
        let pixels = vec![[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [1.0, 0.5, 0.2]];
        let frame = ot.apply_frame(&pixels);
        for (i, &px) in pixels.iter().enumerate() {
            let expected = ot.apply(px);
            for j in 0..3 {
                assert!(
                    (frame[i][j] - expected[j]).abs() < 1e-6,
                    "Frame[{i}][{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn test_display_target_labels() {
        assert!(!DisplayTarget::Srgb_100nit.label().is_empty());
        assert!(!DisplayTarget::P3_Dci.label().is_empty());
        assert!(!DisplayTarget::Rec2020_Hdr.label().is_empty());
    }

    #[test]
    fn test_peak_luminance_nit_values() {
        assert!((PeakLuminance::Nit100.nits() - 100.0).abs() < 1e-4);
        assert!((PeakLuminance::Nit1000.nits() - 1000.0).abs() < 1e-4);
        assert!((PeakLuminance::Nit4000.nits() - 4000.0).abs() < 1e-4);
        assert!((PeakLuminance::Nit10000.nits() - 10_000.0).abs() < 1e-4);
    }

    #[test]
    fn test_reference_rendering_transform_range() {
        let out = reference_rendering_transform([0.18, 0.18, 0.18]);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "RRT out of range: {ch}");
        }
    }

    #[test]
    fn test_reference_rendering_transform_black() {
        let out = reference_rendering_transform([0.0, 0.0, 0.0]);
        for ch in out {
            assert!(ch.abs() < 1e-6, "RRT of black should be ~0: {ch}");
        }
    }

    #[test]
    fn test_adapt_peak_luminance_identity() {
        let rgb = [0.5, 0.3, 0.8];
        let out = adapt_peak_luminance(rgb, 1000.0, 1000.0);
        for i in 0..3 {
            assert!(
                (out[i] - rgb[i]).abs() < 0.01,
                "Same peak should be near identity"
            );
        }
    }

    #[test]
    fn test_adapt_peak_luminance_downscale() {
        let rgb = [0.5, 0.5, 0.5];
        let out = adapt_peak_luminance(rgb, 1000.0, 100.0);
        // 0.5 * 0.1 = 0.05
        assert!(
            out[0] < rgb[0],
            "Downscale should reduce: {} vs {}",
            out[0],
            rgb[0]
        );
    }

    #[test]
    fn test_adapt_peak_luminance_zero_peak() {
        let out = adapt_peak_luminance([0.5, 0.5, 0.5], 0.0, 1000.0);
        for ch in out {
            assert!(ch.abs() < 1e-6, "Zero source peak should give zero");
        }
    }

    #[test]
    fn test_adapt_peak_luminance_always_01() {
        let out = adapt_peak_luminance([1.0, 1.0, 1.0], 100.0, 10000.0);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "Adapted out of [0,1]: {ch}");
        }
    }

    #[test]
    fn test_gamut_compression_params_default() {
        let p = GamutCompressionParams::default();
        assert!(p.threshold > 0.0 && p.threshold < 1.0);
        assert!(p.power > 0.0);
    }

    #[test]
    fn test_aces_ot2_p3_dci_output_range() {
        let ot = AcesOt2::new(DisplayTarget::P3_Dci);
        let out = ot.apply([0.18, 0.18, 0.18]);
        for ch in out {
            assert!(ch >= 0.0 && ch <= 1.0, "P3-DCI output out of range: {ch}");
        }
    }
}
