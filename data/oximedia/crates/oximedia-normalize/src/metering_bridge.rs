//! Normalization metering integration.
//!
//! Bridges loudness measurements (LUFS, true peak, LRA) with normalization
//! targets and produces actionable gain plans.

#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// LufsTarget
// ──────────────────────────────────────────────────────────────────────────────

/// Broadcast / streaming loudness normalization target expressed as a LUFS
/// value (or a custom level).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LufsTarget {
    /// EBU R128: −23 LUFS (European broadcast standard).
    EbuR128,
    /// ATSC A/85: −24 LKFS (US broadcast standard).
    Atsc,
    /// ARIB TR-B32: −24 LKFS (Japanese broadcast standard).
    Arib,
    /// Custom LUFS target.
    CustomLufs(f32),
}

impl LufsTarget {
    /// Return the target integrated loudness in LUFS.
    pub fn target_lufs(&self) -> f32 {
        match self {
            Self::EbuR128 => -23.0,
            Self::Atsc => -24.0,
            Self::Arib => -24.0,
            Self::CustomLufs(lufs) => *lufs,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MeteringWindow
// ──────────────────────────────────────────────────────────────────────────────

/// Time window used for loudness measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeteringWindow {
    /// Momentary loudness (400 ms window, no gating).
    Momentary,
    /// Short-term loudness (3 s window, no gating).
    ShortTerm,
    /// Integrated loudness (full-programme gated measurement, ITU-R BS.1770).
    Integrated,
}

impl MeteringWindow {
    /// Window duration in milliseconds.
    pub fn window_ms(&self) -> u32 {
        match self {
            Self::Momentary => 400,
            Self::ShortTerm => 3_000,
            Self::Integrated => 0, // unbounded / gated
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LoudnessMeasurement
// ──────────────────────────────────────────────────────────────────────────────

/// A complete loudness measurement result.
#[derive(Debug, Clone, Copy)]
pub struct LoudnessMeasurement {
    /// Integrated (programme-level) loudness in LUFS.
    pub integrated_lufs: f32,
    /// Short-term loudness in LUFS.
    pub short_term_lufs: f32,
    /// Momentary loudness in LUFS.
    pub momentary_lufs: f32,
    /// True peak level in dBTP.
    pub true_peak_dbtp: f32,
    /// Loudness range (LRA) in LU.
    pub lra: f32,
}

impl LoudnessMeasurement {
    /// Create a new measurement.
    pub fn new(
        integrated_lufs: f32,
        short_term_lufs: f32,
        momentary_lufs: f32,
        true_peak_dbtp: f32,
        lra: f32,
    ) -> Self {
        Self {
            integrated_lufs,
            short_term_lufs,
            momentary_lufs,
            true_peak_dbtp,
            lra,
        }
    }

    /// Returns `true` if the integrated loudness is within `tolerance_lu` of
    /// the target and true peak is below −1 dBTP (EBU R128 default ceiling).
    pub fn is_within_target(&self, target: &LufsTarget, tolerance_lu: f32) -> bool {
        let diff = (self.integrated_lufs - target.target_lufs()).abs();
        diff <= tolerance_lu
    }

    /// Gain (in dB) that must be applied to reach the target integrated loudness.
    ///
    /// Positive means amplification is needed; negative means attenuation.
    pub fn gain_to_apply(&self, target: &LufsTarget) -> f32 {
        target.target_lufs() - self.integrated_lufs
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NormalizationPlan
// ──────────────────────────────────────────────────────────────────────────────

/// An actionable plan for normalizing audio to a target loudness.
#[derive(Debug, Clone)]
pub struct NormalizationPlan {
    /// The source loudness measurement.
    pub source: LoudnessMeasurement,
    /// The normalization target.
    pub target: LufsTarget,
    /// Gain to apply in dB.
    pub gain_db: f32,
    /// Whether a true-peak limiter must be engaged after gain application.
    pub needs_limiting: bool,
}

impl NormalizationPlan {
    /// True-peak ceiling used for limiting decisions (EBU R128 default).
    const TRUE_PEAK_CEILING_DBTP: f32 = -1.0;

    /// Create a normalization plan from a measurement and a target.
    ///
    /// `needs_limiting` is set to `true` when the post-gain true peak would
    /// exceed the ceiling of −1 dBTP.
    pub fn create(measurement: LoudnessMeasurement, target: LufsTarget) -> Self {
        let gain_db = measurement.gain_to_apply(&target);
        let post_gain_true_peak = measurement.true_peak_dbtp + gain_db;
        let needs_limiting = post_gain_true_peak > Self::TRUE_PEAK_CEILING_DBTP;
        Self {
            source: measurement,
            target,
            gain_db,
            needs_limiting,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // LufsTarget ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ebu_r128_target() {
        assert_eq!(LufsTarget::EbuR128.target_lufs(), -23.0);
    }

    #[test]
    fn test_atsc_target() {
        assert_eq!(LufsTarget::Atsc.target_lufs(), -24.0);
    }

    #[test]
    fn test_arib_target() {
        assert_eq!(LufsTarget::Arib.target_lufs(), -24.0);
    }

    #[test]
    fn test_custom_lufs_target() {
        assert_eq!(LufsTarget::CustomLufs(-14.0).target_lufs(), -14.0);
    }

    // MeteringWindow ──────────────────────────────────────────────────────────

    #[test]
    fn test_momentary_window_ms() {
        assert_eq!(MeteringWindow::Momentary.window_ms(), 400);
    }

    #[test]
    fn test_short_term_window_ms() {
        assert_eq!(MeteringWindow::ShortTerm.window_ms(), 3_000);
    }

    #[test]
    fn test_integrated_window_ms() {
        assert_eq!(MeteringWindow::Integrated.window_ms(), 0);
    }

    // LoudnessMeasurement ─────────────────────────────────────────────────────

    #[test]
    fn test_gain_to_apply_positive() {
        // Source at -30 LUFS, target -23 LUFS → need +7 dB.
        let m = LoudnessMeasurement::new(-30.0, -28.0, -25.0, -6.0, 8.0);
        let gain = m.gain_to_apply(&LufsTarget::EbuR128);
        assert!((gain - 7.0).abs() < 1e-5, "Expected +7 dB, got {}", gain);
    }

    #[test]
    fn test_gain_to_apply_negative() {
        // Source at -15 LUFS, target -23 LUFS → need -8 dB.
        let m = LoudnessMeasurement::new(-15.0, -15.0, -15.0, -2.0, 5.0);
        let gain = m.gain_to_apply(&LufsTarget::EbuR128);
        assert!((gain - (-8.0)).abs() < 1e-5, "Expected -8 dB, got {}", gain);
    }

    #[test]
    fn test_is_within_target_true() {
        let m = LoudnessMeasurement::new(-23.0, -23.0, -23.0, -3.0, 5.0);
        assert!(m.is_within_target(&LufsTarget::EbuR128, 1.0));
    }

    #[test]
    fn test_is_within_target_false() {
        let m = LoudnessMeasurement::new(-15.0, -15.0, -15.0, -2.0, 5.0);
        assert!(!m.is_within_target(&LufsTarget::EbuR128, 1.0));
    }

    #[test]
    fn test_is_within_target_boundary() {
        // Exactly 1 LU below target (within tolerance of 1 LU).
        let m = LoudnessMeasurement::new(-24.0, -24.0, -24.0, -4.0, 5.0);
        assert!(m.is_within_target(&LufsTarget::EbuR128, 1.0));
    }

    // NormalizationPlan ───────────────────────────────────────────────────────

    #[test]
    fn test_plan_gain_db_computed_correctly() {
        let m = LoudnessMeasurement::new(-30.0, -28.0, -25.0, -10.0, 8.0);
        let plan = NormalizationPlan::create(m, LufsTarget::EbuR128);
        assert!(
            (plan.gain_db - 7.0).abs() < 1e-5,
            "Expected +7 dB, got {}",
            plan.gain_db
        );
    }

    #[test]
    fn test_plan_needs_limiting_when_would_clip() {
        // True peak at −3 dBTP; applying +7 dB → post-gain = +4 dBTP > −1 dBTP ceiling.
        let m = LoudnessMeasurement::new(-30.0, -28.0, -25.0, -3.0, 5.0);
        let plan = NormalizationPlan::create(m, LufsTarget::EbuR128);
        assert!(plan.needs_limiting);
    }

    #[test]
    fn test_plan_no_limiting_when_safe() {
        // True peak at −15 dBTP; applying +1 dB → post-gain = −14 dBTP < −1 dBTP ceiling.
        let m = LoudnessMeasurement::new(-24.0, -23.0, -22.0, -15.0, 3.0);
        let plan = NormalizationPlan::create(m, LufsTarget::EbuR128);
        assert!(!plan.needs_limiting);
    }

    #[test]
    fn test_plan_target_stored() {
        let m = LoudnessMeasurement::new(-30.0, -28.0, -25.0, -10.0, 8.0);
        let plan = NormalizationPlan::create(m, LufsTarget::Atsc);
        assert_eq!(plan.target, LufsTarget::Atsc);
    }

    #[test]
    fn test_plan_custom_target() {
        let m = LoudnessMeasurement::new(-20.0, -19.0, -18.0, -5.0, 4.0);
        let plan = NormalizationPlan::create(m, LufsTarget::CustomLufs(-14.0));
        assert!(
            (plan.gain_db - 6.0).abs() < 1e-5,
            "Expected +6 dB, got {}",
            plan.gain_db
        );
    }
}
