#![allow(dead_code)]
//! Display verification: white point, black level, and target compliance.

/// Standard display targets with characteristic peak luminance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayTarget {
    /// Standard dynamic range consumer display (~100 nits).
    Sdr100,
    /// HLG broadcast display (~1000 nits peak).
    HlgBroadcast,
    /// HDR10 consumer display (~1000 nits peak).
    Hdr10,
    /// HDR10+ or Dolby Vision consumer OLED (~4000 nits peak).
    Hdr4000,
    /// Cinema reference (~48 nits for DCI-P3).
    CinemaReference,
    /// Custom target with a user-supplied peak luminance.
    Custom(u32),
}

impl DisplayTarget {
    /// Returns the reference peak luminance in nits (cd/m²) for this target.
    #[must_use]
    pub fn peak_luminance_nits(&self) -> u32 {
        match self {
            Self::Sdr100 => 100,
            Self::HlgBroadcast => 1000,
            Self::Hdr10 => 1000,
            Self::Hdr4000 => 4000,
            Self::CinemaReference => 48,
            Self::Custom(nits) => *nits,
        }
    }

    /// Returns `true` if this is an HDR target (peak > 200 nits).
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.peak_luminance_nits() > 200
    }
}

/// CIE xy chromaticity coordinate pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaXy {
    /// CIE x chromaticity coordinate.
    pub x: f64,
    /// CIE y chromaticity coordinate.
    pub y: f64,
}

impl ChromaXy {
    /// Creates a new `ChromaXy`.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the Euclidean distance to another chromaticity point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// The result of a single display verification run.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Measured peak white luminance in nits.
    pub measured_white_nits: f64,
    /// Measured black level luminance in nits.
    pub measured_black_nits: f64,
    /// Measured white point chromaticity.
    pub measured_white_point: ChromaXy,
    /// Target that was verified against.
    pub target: DisplayTarget,
    /// Maximum allowable ΔE (chromaticity deviation tolerance).
    pub delta_e_tolerance: f64,
    /// Actual chromaticity deviation from D65.
    pub white_point_delta: f64,
}

impl VerifyResult {
    /// Reference D65 white point chromaticity.
    pub const D65: ChromaXy = ChromaXy {
        x: 0.3127,
        y: 0.3290,
    };

    /// Returns `true` if the measured white luminance is within 10 % of the target.
    #[must_use]
    pub fn passes_luminance(&self) -> bool {
        let target = self.target.peak_luminance_nits() as f64;
        let ratio = self.measured_white_nits / target;
        (0.90..=1.10).contains(&ratio)
    }

    /// Returns `true` if the white point chromaticity is within `delta_e_tolerance`.
    #[must_use]
    pub fn passes_white_point(&self) -> bool {
        self.white_point_delta <= self.delta_e_tolerance
    }

    /// Returns `true` if the black level is ≤ 0.1 nits (SDR) or ≤ 0.005 nits (HDR).
    #[must_use]
    pub fn passes_black_level(&self) -> bool {
        let limit = if self.target.is_hdr() { 0.005 } else { 0.1 };
        self.measured_black_nits <= limit
    }

    /// Returns `true` if the display passes all checks.
    #[must_use]
    pub fn passes_target(&self) -> bool {
        self.passes_luminance() && self.passes_white_point() && self.passes_black_level()
    }

    /// Returns a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "White: {:.1} nits, Black: {:.4} nits, ΔE: {:.4}, Pass: {}",
            self.measured_white_nits,
            self.measured_black_nits,
            self.white_point_delta,
            self.passes_target(),
        )
    }
}

/// Verifies a display against a given `DisplayTarget`.
#[derive(Debug, Clone)]
pub struct DisplayVerifier {
    /// The target to verify against.
    pub target: DisplayTarget,
    /// Tolerance for white-point ΔE (default 0.01).
    pub delta_e_tolerance: f64,
}

impl DisplayVerifier {
    /// Creates a new `DisplayVerifier`.
    #[must_use]
    pub fn new(target: DisplayTarget) -> Self {
        Self {
            target,
            delta_e_tolerance: 0.01,
        }
    }

    /// Sets the white-point ΔE tolerance and returns `self` for chaining.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.delta_e_tolerance = tol;
        self
    }

    /// Simulates measuring the white point by accepting externally provided chromaticity.
    ///
    /// In a real implementation this would drive a colorimeter.
    #[must_use]
    pub fn measure_white_point(&self, measured: ChromaXy) -> f64 {
        measured.distance_to(&VerifyResult::D65)
    }

    /// Simulates measuring the black level luminance.
    ///
    /// In production this would read from a colorimeter measurement.
    #[must_use]
    pub fn measure_black_level(&self, measured_nits: f64) -> f64 {
        measured_nits.max(0.0)
    }

    /// Runs a full display verification given externally measured values.
    ///
    /// `white_nits` — measured peak white luminance.
    /// `black_nits` — measured black level luminance.
    /// `white_point` — measured CIE xy chromaticity of white.
    #[must_use]
    pub fn verify(&self, white_nits: f64, black_nits: f64, white_point: ChromaXy) -> VerifyResult {
        let delta = self.measure_white_point(white_point);
        VerifyResult {
            measured_white_nits: white_nits,
            measured_black_nits: self.measure_black_level(black_nits),
            measured_white_point: white_point,
            target: self.target,
            delta_e_tolerance: self.delta_e_tolerance,
            white_point_delta: delta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DisplayTarget ---

    #[test]
    fn sdr_peak_luminance() {
        assert_eq!(DisplayTarget::Sdr100.peak_luminance_nits(), 100);
    }

    #[test]
    fn hdr10_peak_luminance() {
        assert_eq!(DisplayTarget::Hdr10.peak_luminance_nits(), 1000);
    }

    #[test]
    fn custom_target_nits() {
        assert_eq!(DisplayTarget::Custom(500).peak_luminance_nits(), 500);
    }

    #[test]
    fn sdr_not_hdr() {
        assert!(!DisplayTarget::Sdr100.is_hdr());
    }

    #[test]
    fn hdr10_is_hdr() {
        assert!(DisplayTarget::Hdr10.is_hdr());
    }

    #[test]
    fn cinema_reference_not_hdr() {
        assert!(!DisplayTarget::CinemaReference.is_hdr());
    }

    // --- ChromaXy ---

    #[test]
    fn chroma_distance_to_self_is_zero() {
        let p = ChromaXy::new(0.3127, 0.3290);
        assert!(p.distance_to(&p) < 1e-12);
    }

    #[test]
    fn chroma_distance_nonzero() {
        let a = ChromaXy::new(0.3127, 0.3290);
        let b = ChromaXy::new(0.3000, 0.3100);
        assert!(a.distance_to(&b) > 0.0);
    }

    // --- VerifyResult ---

    fn good_result() -> VerifyResult {
        let verifier = DisplayVerifier::new(DisplayTarget::Sdr100).with_tolerance(0.02);
        verifier.verify(100.0, 0.05, ChromaXy::new(0.3127, 0.3290))
    }

    #[test]
    fn passing_result_passes_target() {
        assert!(good_result().passes_target());
    }

    #[test]
    fn luminance_10pct_low_still_passes() {
        let v = DisplayVerifier::new(DisplayTarget::Sdr100).with_tolerance(0.02);
        let r = v.verify(91.0, 0.05, ChromaXy::new(0.3127, 0.3290));
        assert!(r.passes_luminance());
    }

    #[test]
    fn luminance_too_low_fails() {
        let v = DisplayVerifier::new(DisplayTarget::Sdr100).with_tolerance(0.02);
        let r = v.verify(50.0, 0.05, ChromaXy::new(0.3127, 0.3290));
        assert!(!r.passes_luminance());
    }

    #[test]
    fn bad_white_point_fails() {
        let v = DisplayVerifier::new(DisplayTarget::Sdr100).with_tolerance(0.001);
        let r = v.verify(100.0, 0.05, ChromaXy::new(0.35, 0.35));
        assert!(!r.passes_white_point());
    }

    #[test]
    fn hdr_black_level_strict() {
        let v = DisplayVerifier::new(DisplayTarget::Hdr10).with_tolerance(0.02);
        // 0.01 nits fails HDR (<= 0.005 required)
        let r = v.verify(1000.0, 0.01, ChromaXy::new(0.3127, 0.3290));
        assert!(!r.passes_black_level());
    }

    #[test]
    fn summary_contains_pass() {
        let s = good_result().summary();
        assert!(s.contains("Pass: true"));
    }

    #[test]
    fn measure_white_point_d65_gives_zero() {
        let v = DisplayVerifier::new(DisplayTarget::Sdr100);
        let delta = v.measure_white_point(ChromaXy::new(0.3127, 0.3290));
        assert!(delta < 1e-10);
    }

    #[test]
    fn measure_black_level_clamps_negative() {
        let v = DisplayVerifier::new(DisplayTarget::Sdr100);
        assert_eq!(v.measure_black_level(-1.0), 0.0);
    }
}
