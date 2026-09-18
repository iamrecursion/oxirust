//! Automatic colour grading from frame statistics.
//!
//! Analyses a raw RGB frame to extract photometric statistics, then derives
//! lift/gamma/gain + saturation/contrast adjustments appropriate for the
//! requested [`GradeStyle`].  All computation is pure-Rust, no external
//! colour-science libraries are required.

#![allow(dead_code)]

// ─── GradeStyle ──────────────────────────────────────────────────────────────

/// Target look / aesthetic style for the auto grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradeStyle {
    /// Desaturated, lifted blacks, filmic curve — classic cinema look.
    Cinematic,
    /// Natural, accurate skin tones, mild contrast — broadcast documentary.
    Documentary,
    /// Clean, neutral, slight cool cast — news / corporate.
    News,
    /// Vivid, punchy, high saturation — social-media / UGC.
    Social,
    /// Warm, faded, reduced saturation — vintage / film emulation.
    Vintage,
    /// Teal shadows, orange highlights — popular Hollywood orange-teal look.
    TealOrange,
}

impl GradeStyle {
    /// Return a human-readable name for this style.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cinematic => "Cinematic",
            Self::Documentary => "Documentary",
            Self::News => "News",
            Self::Social => "Social",
            Self::Vintage => "Vintage",
            Self::TealOrange => "Teal-Orange",
        }
    }
}

// ─── AutoGradeConfig ─────────────────────────────────────────────────────────

/// Configuration for [`AutoGrader`].
#[derive(Debug, Clone)]
pub struct AutoGradeConfig {
    /// Target look preset.
    pub target_style: GradeStyle,
    /// Intensity of the grade in [0.0, 1.0].
    ///
    /// 0.0 = no grade applied (pass-through), 1.0 = full effect.
    pub intensity: f32,
}

impl AutoGradeConfig {
    /// Create a new config with the given style and intensity.
    #[must_use]
    pub const fn new(target_style: GradeStyle, intensity: f32) -> Self {
        Self {
            target_style,
            intensity,
        }
    }

    /// Intensity clamped to [0.0, 1.0].
    #[must_use]
    pub fn clamped_intensity(&self) -> f32 {
        self.intensity.clamp(0.0, 1.0)
    }
}

impl Default for AutoGradeConfig {
    fn default() -> Self {
        Self::new(GradeStyle::Cinematic, 0.75)
    }
}

// ─── ColorGradeParams ────────────────────────────────────────────────────────

/// Lift / Gamma / Gain colour grading parameters plus global saturation and
/// contrast adjustments.
///
/// Each `[f32; 3]` array is ordered `[R, G, B]`.
///
/// ## Semantics
///
/// The classic three-way colour-corrector model is:
/// ```text
/// output = (input + lift) ^ (1/gamma) * gain
/// ```
/// where exponentiation is applied per-channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorGradeParams {
    /// Shadow colour offset in [-0.5, 0.5].  0.0 = no adjustment.
    pub lift: [f32; 3],
    /// Mid-tone power curve (reciprocal of gamma exponent).
    ///
    /// Values > 1.0 brighten mid-tones; < 1.0 darken them.  1.0 = neutral.
    pub gamma: [f32; 3],
    /// Highlight multiplier.  1.0 = neutral, > 1.0 = brighter.
    pub gain: [f32; 3],
    /// Global saturation multiplier.  1.0 = neutral.
    pub saturation: f32,
    /// Global contrast multiplier around mid-grey.  1.0 = neutral.
    pub contrast: f32,
}

impl ColorGradeParams {
    /// Identity / pass-through grade (no change).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            saturation: 1.0,
            contrast: 1.0,
        }
    }

    /// Blend `self` towards `identity` so that `t = 0` → identity and
    /// `t = 1` → the full grade.
    #[must_use]
    pub fn blend_with_identity(&self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let id = Self::identity();
        Self {
            lift: lerp_arr3(id.lift, self.lift, t),
            gamma: lerp_arr3(id.gamma, self.gamma, t),
            gain: lerp_arr3(id.gain, self.gain, t),
            saturation: lerp(id.saturation, self.saturation, t),
            contrast: lerp(id.contrast, self.contrast, t),
        }
    }
}

// ─── FrameStats ──────────────────────────────────────────────────────────────

/// Per-channel and global statistics computed from a raw RGB frame.
#[derive(Debug, Clone)]
struct FrameStats {
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    /// Global mean luminance (BT.601 weighted).
    mean_luma: f32,
    /// RMS contrast.
    rms_contrast: f32,
    /// Mean saturation (Hue-Saturation-Value model).
    mean_saturation: f32,
}

impl FrameStats {
    /// Compute statistics from a raw RGB byte slice.
    ///
    /// Returns `None` if the buffer is empty or its size is not a multiple of 3.
    fn from_frame(frame: &[u8]) -> Option<Self> {
        if frame.is_empty() || frame.len() % 3 != 0 {
            return None;
        }

        let n = frame.len() / 3;
        let inv_n = 1.0 / n as f64;

        let mut sum_r = 0_u64;
        let mut sum_g = 0_u64;
        let mut sum_b = 0_u64;
        let mut sum_luma = 0_f64;
        let mut sum_luma_sq = 0_f64;
        let mut sum_sat = 0_f64;

        for chunk in frame.chunks_exact(3) {
            let r = chunk[0] as f64 / 255.0;
            let g = chunk[1] as f64 / 255.0;
            let b = chunk[2] as f64 / 255.0;

            sum_r += chunk[0] as u64;
            sum_g += chunk[1] as u64;
            sum_b += chunk[2] as u64;

            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            sum_luma += luma;
            sum_luma_sq += luma * luma;

            // HSV saturation = (max - min) / max, with max > 0
            let c_max = r.max(g).max(b);
            let c_min = r.min(g).min(b);
            let s = if c_max > 0.0 {
                (c_max - c_min) / c_max
            } else {
                0.0
            };
            sum_sat += s;
        }

        let mean_luma = (sum_luma * inv_n) as f32;
        let mean_luma_sq = (sum_luma_sq * inv_n) as f32;
        let variance = (mean_luma_sq - mean_luma * mean_luma).max(0.0);

        Some(Self {
            mean_r: (sum_r as f64 * inv_n / 255.0) as f32,
            mean_g: (sum_g as f64 * inv_n / 255.0) as f32,
            mean_b: (sum_b as f64 * inv_n / 255.0) as f32,
            mean_luma,
            rms_contrast: variance.sqrt(),
            mean_saturation: (sum_sat * inv_n) as f32,
        })
    }
}

// ─── AutoGrader ──────────────────────────────────────────────────────────────

/// Automatic colour grader.
///
/// Analyses a frame and produces style-appropriate [`ColorGradeParams`]
/// that can be fed into a downstream three-way colour-corrector.
pub struct AutoGrader;

impl AutoGrader {
    /// Analyse `frame` (raw interleaved RGB, `w × h × 3` bytes) and return
    /// colour grade parameters for the requested style.
    ///
    /// If the frame buffer is empty or malformed the grader falls back to
    /// the identity grade.
    #[must_use]
    pub fn analyze_and_grade(
        frame: &[u8],
        _w: u32,
        _h: u32,
        config: &AutoGradeConfig,
    ) -> ColorGradeParams {
        let stats = match FrameStats::from_frame(frame) {
            Some(s) => s,
            None => return ColorGradeParams::identity(),
        };

        let intensity = config.clamped_intensity();

        // Base grade for the requested style (at full intensity).
        let full_grade = Self::style_grade(&stats, config.target_style);

        // Blend between identity and the full grade according to intensity.
        full_grade.blend_with_identity(intensity)
    }

    // ── Style-specific grade derivation ──────────────────────────────────────

    fn style_grade(stats: &FrameStats, style: GradeStyle) -> ColorGradeParams {
        match style {
            GradeStyle::Cinematic => Self::cinematic(stats),
            GradeStyle::Documentary => Self::documentary(stats),
            GradeStyle::News => Self::news(stats),
            GradeStyle::Social => Self::social(stats),
            GradeStyle::Vintage => Self::vintage(stats),
            GradeStyle::TealOrange => Self::teal_orange(stats),
        }
    }

    /// Cinematic: lifted blacks, cool shadows, reduced saturation, filmic contrast.
    fn cinematic(stats: &FrameStats) -> ColorGradeParams {
        // Shadow lift: slight blue-green tint in shadows
        let lift_amount = 0.04_f32;
        let lift = [lift_amount * 0.8, lift_amount * 0.95, lift_amount * 1.15];

        // Gamma: slightly warm midtones if scene is cool, neutral otherwise
        let gamma_correction = if stats.mean_b > stats.mean_r + 0.05 {
            1.05_f32 // warm up blue-biased scene
        } else {
            1.0
        };
        let gamma = [gamma_correction, 1.0, 1.0 / gamma_correction.max(0.01)];

        // Gain: very slight highlight roll-off
        let gain = [0.97, 0.97, 0.97];

        ColorGradeParams {
            lift,
            gamma,
            gain,
            saturation: 0.82,
            contrast: 1.18,
        }
    }

    /// Documentary: accurate, minimal grade, natural contrast.
    fn documentary(stats: &FrameStats) -> ColorGradeParams {
        // Mild exposure correction towards neutral grey (0.5)
        let exposure_delta = 0.5_f32 - stats.mean_luma;
        let gain_adj = 1.0 + exposure_delta * 0.2;

        ColorGradeParams {
            lift: [0.0; 3],
            gamma: [1.0, 1.0, 1.0],
            gain: [gain_adj, gain_adj, gain_adj],
            saturation: 1.0,
            contrast: 1.05,
        }
    }

    /// News: clean, slightly cool, high clarity.
    fn news(stats: &FrameStats) -> ColorGradeParams {
        // Cool the image slightly
        let gain = [0.98, 0.99, 1.02];

        // Correct exposure if over/underexposed
        let luma_corr = 1.0 + (0.5 - stats.mean_luma) * 0.15;

        ColorGradeParams {
            lift: [0.0; 3],
            gamma: [1.0, 1.0, 1.0],
            gain: [
                gain[0] * luma_corr,
                gain[1] * luma_corr,
                gain[2] * luma_corr,
            ],
            saturation: 0.95,
            contrast: 1.0,
        }
    }

    /// Social: vivid, saturated, high contrast.
    fn social(stats: &FrameStats) -> ColorGradeParams {
        // Boost each channel proportional to its deviation from grey
        let gain_r = 1.0 + (stats.mean_r - stats.mean_luma) * 0.3;
        let gain_g = 1.0 + (stats.mean_g - stats.mean_luma) * 0.2;
        let gain_b = 1.0 + (stats.mean_b - stats.mean_luma) * 0.3;

        ColorGradeParams {
            lift: [0.0; 3],
            gamma: [0.95, 0.95, 0.95], // slightly lift mid-tones (brighter)
            gain: [gain_r.max(0.8), gain_g.max(0.8), gain_b.max(0.8)],
            saturation: 1.35,
            contrast: 1.25,
        }
    }

    /// Vintage: warm, faded, soft.
    fn vintage(_stats: &FrameStats) -> ColorGradeParams {
        // Lift blacks (faded look)
        let lift = [0.06, 0.04, 0.01];

        // Warm gamma
        let gamma = [1.08, 1.0, 0.92];

        // Slight highlight roll-off + warm highlights
        let gain = [1.02, 0.99, 0.88];

        ColorGradeParams {
            lift,
            gamma,
            gain,
            saturation: 0.72,
            contrast: 0.90,
        }
    }

    /// Teal-orange: shadows pushed teal, highlights pushed orange.
    fn teal_orange(stats: &FrameStats) -> ColorGradeParams {
        // Teal shadows: subtract red/green slightly, boost blue in lift
        let lift = [-0.02, 0.0, 0.04];

        // Neutral gamma
        let gamma = [1.0, 1.0, 1.0];

        // Orange highlights: boost red/green in highlights, reduce blue
        // (strength proportional to mean highlight exposure)
        let highlight_warmth = (stats.mean_luma - 0.5).max(0.0);
        let gain = [
            1.0 + 0.10 * highlight_warmth,
            1.0 + 0.05 * highlight_warmth,
            1.0 - 0.12 * highlight_warmth,
        ];

        ColorGradeParams {
            lift,
            gamma,
            gain,
            saturation: 1.1,
            contrast: 1.15,
        }
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp_arr3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GradeStyle ────────────────────────────────────────────────────────────

    #[test]
    fn grade_style_names() {
        assert_eq!(GradeStyle::Cinematic.name(), "Cinematic");
        assert_eq!(GradeStyle::Documentary.name(), "Documentary");
        assert_eq!(GradeStyle::News.name(), "News");
        assert_eq!(GradeStyle::Social.name(), "Social");
        assert_eq!(GradeStyle::Vintage.name(), "Vintage");
        assert_eq!(GradeStyle::TealOrange.name(), "Teal-Orange");
    }

    // ── AutoGradeConfig ───────────────────────────────────────────────────────

    #[test]
    fn auto_grade_config_default() {
        let cfg = AutoGradeConfig::default();
        assert_eq!(cfg.target_style, GradeStyle::Cinematic);
        assert!((cfg.clamped_intensity() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn auto_grade_config_intensity_clamp() {
        let cfg = AutoGradeConfig::new(GradeStyle::Social, 2.5);
        assert!((cfg.clamped_intensity() - 1.0).abs() < 1e-6);
        let cfg2 = AutoGradeConfig::new(GradeStyle::Social, -1.0);
        assert!((cfg2.clamped_intensity()).abs() < 1e-6);
    }

    // ── ColorGradeParams identity ─────────────────────────────────────────────

    #[test]
    fn identity_grade_values() {
        let id = ColorGradeParams::identity();
        assert_eq!(id.lift, [0.0; 3]);
        assert_eq!(id.gamma, [1.0; 3]);
        assert_eq!(id.gain, [1.0; 3]);
        assert!((id.saturation - 1.0).abs() < 1e-6);
        assert!((id.contrast - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blend_with_identity_at_zero_returns_identity() {
        let grade = AutoGrader::analyze_and_grade(
            &make_grey_frame(4, 4, 128),
            4,
            4,
            &AutoGradeConfig::new(GradeStyle::Cinematic, 0.0),
        );
        // At intensity 0, should be identical to identity.
        let id = ColorGradeParams::identity();
        assert!((grade.saturation - id.saturation).abs() < 1e-5);
        assert!((grade.contrast - id.contrast).abs() < 1e-5);
    }

    #[test]
    fn blend_with_identity_at_one_returns_full_grade() {
        let frame = make_grey_frame(4, 4, 128);
        let full = AutoGrader::analyze_and_grade(
            &frame,
            4,
            4,
            &AutoGradeConfig::new(GradeStyle::Vintage, 1.0),
        );
        // Vintage at full intensity should have saturation < 1.0
        assert!(full.saturation < 1.0, "Vintage should reduce saturation");
    }

    // ── AutoGrader — empty / malformed frame ──────────────────────────────────

    #[test]
    fn empty_frame_returns_identity() {
        let grade = AutoGrader::analyze_and_grade(&[], 0, 0, &AutoGradeConfig::default());
        let id = ColorGradeParams::identity();
        assert!((grade.saturation - id.saturation).abs() < 1e-6);
    }

    #[test]
    fn misaligned_frame_returns_identity() {
        let grade = AutoGrader::analyze_and_grade(
            &[255, 128], // not a multiple of 3
            1,
            1,
            &AutoGradeConfig::default(),
        );
        let id = ColorGradeParams::identity();
        assert!((grade.saturation - id.saturation).abs() < 1e-6);
    }

    // ── Style-specific properties ─────────────────────────────────────────────

    #[test]
    fn cinematic_has_lifted_blacks() {
        let frame = make_grey_frame(8, 8, 64);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Cinematic, 1.0),
        );
        // All lift values should be > 0 for cinematic
        for &l in &grade.lift {
            assert!(l > 0.0, "Cinematic lift should be positive, got {l}");
        }
    }

    #[test]
    fn cinematic_reduced_saturation() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Cinematic, 1.0),
        );
        assert!(grade.saturation < 1.0, "Cinematic should reduce saturation");
    }

    #[test]
    fn cinematic_increased_contrast() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Cinematic, 1.0),
        );
        assert!(grade.contrast > 1.0, "Cinematic should increase contrast");
    }

    #[test]
    fn social_increased_saturation() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Social, 1.0),
        );
        assert!(grade.saturation > 1.0, "Social should boost saturation");
    }

    #[test]
    fn vintage_reduced_saturation() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Vintage, 1.0),
        );
        assert!(grade.saturation < 1.0, "Vintage should reduce saturation");
    }

    #[test]
    fn vintage_has_positive_lift_red() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::Vintage, 1.0),
        );
        assert!(grade.lift[0] > 0.0, "Vintage should have warm lift in R");
    }

    #[test]
    fn teal_orange_shadow_blue_lift() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::TealOrange, 1.0),
        );
        // Blue lift should be positive (teal shadows)
        assert!(
            grade.lift[2] > 0.0,
            "Teal-Orange should have positive blue lift"
        );
        // Red lift should be negative (teal = away from orange)
        assert!(
            grade.lift[0] < 0.0,
            "Teal-Orange should have negative red lift"
        );
    }

    #[test]
    fn news_near_neutral_saturation() {
        let frame = make_grey_frame(8, 8, 128);
        let grade = AutoGrader::analyze_and_grade(
            &frame,
            8,
            8,
            &AutoGradeConfig::new(GradeStyle::News, 1.0),
        );
        // News should be close to neutral saturation
        assert!(
            grade.saturation > 0.8 && grade.saturation <= 1.0,
            "News saturation out of expected range: {}",
            grade.saturation
        );
    }

    // ── lerp helper ───────────────────────────────────────────────────────────

    #[test]
    fn lerp_midpoint() {
        let v = lerp(0.0, 1.0, 0.5);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_arr3_midpoint() {
        let a = [0.0_f32; 3];
        let b = [1.0_f32; 3];
        let mid = lerp_arr3(a, b, 0.5);
        for v in mid {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /// Create an `w × h` grey frame (all channels equal to `value`).
    fn make_grey_frame(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h * 3) as usize]
    }
}
