//! Advanced HLG (Hybrid Log-Gamma) processing per ITU-R BT.2100.
//!
//! Implements the complete HLG system including:
//! - OETF (scene to signal)
//! - EOTF (signal to display)
//! - OOTF (scene to display, the "system gamma" chain)
//! - SDR conversion via BT.709 gamut mapping
//! - HLG-to-PQ cross-format conversion

use crate::{HdrError, Result};

// ── HLG system constants (ITU-R BT.2100) ─────────────────────────────────────

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.559_910_7;

/// Scene-linear threshold below which the square-root segment applies.
const SCENE_LINEAR_THRESHOLD: f32 = 1.0 / 12.0;

/// Signal-domain threshold corresponding to `SCENE_LINEAR_THRESHOLD`.
/// `sqrt(3 * 1/12) = 0.5`
const SIGNAL_THRESHOLD: f32 = 0.5;

// ── BT.709 display gamma ──────────────────────────────────────────────────────

/// BT.709 OETF transfer gamma exponent.
const BT709_GAMMA: f32 = 1.0 / 2.2;

// ── PQ constants (SMPTE ST 2084) ─────────────────────────────────────────────

const PQ_M1: f64 = 0.159_301_757_812_5;
const PQ_M2: f64 = 78.843_75;
const PQ_C1: f64 = 0.835_937_5;
const PQ_C2: f64 = 18.851_562_5;
const PQ_C3: f64 = 18.687_5;

/// Encode a linear value (normalised 0–1, where 1 = 10 000 nits) as PQ signal.
fn pq_oetf_f32(lin: f32) -> f32 {
    let lin = lin.max(0.0) as f64;
    let y = lin.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * y;
    let den = 1.0 + PQ_C3 * y;
    (num / den).powf(PQ_M2) as f32
}

// ── HlgSystem ─────────────────────────────────────────────────────────────────

/// Complete HLG system model per ITU-R BT.2100.
///
/// Encapsulates the three transfer functions that together define
/// the end-to-end pipeline from scene light to displayed light:
///
/// | Abbreviation | Direction                 |
/// |--------------|---------------------------|
/// | OETF         | scene linear → HLG signal |
/// | EOTF         | HLG signal → display light|
/// | OOTF         | scene linear → display light (via system gamma) |
#[derive(Debug, Clone)]
pub struct HlgSystem {
    /// System gamma (γ_s). Default 1.2 per BT.2100.
    pub system_gamma: f32,
    /// Nominal peak luminance of the reference display (L_W), nits. Default 1000.
    pub nominal_peak_luminance: f32,
    /// Reference white luminance (L_w), nits. Default 203 (BT.2100-2).
    pub reference_white: f32,
}

impl Default for HlgSystem {
    /// ITU-R BT.2100-2 reference system: γ=1.2, L_W=1000, L_w=203.
    fn default() -> Self {
        Self {
            system_gamma: 1.2,
            nominal_peak_luminance: 1000.0,
            reference_white: 203.0,
        }
    }
}

impl HlgSystem {
    /// Create an `HlgSystem` with custom parameters.
    pub fn new(system_gamma: f32, nominal_peak_luminance: f32, reference_white: f32) -> Self {
        Self {
            system_gamma,
            nominal_peak_luminance,
            reference_white,
        }
    }

    /// HLG OETF — scene-linear normalised signal E' → HLG signal [0, 1].
    ///
    /// Segments:
    /// - E' ≤ 1/12 → `sqrt(3 * E')`
    /// - E' > 1/12 → `a * ln(12*E' - b) + c`
    ///
    /// # Errors
    /// Returns [`HdrError::InvalidLuminance`] if `scene_linear` is negative.
    pub fn oetf(&self, scene_linear: f32) -> Result<f32> {
        if scene_linear < 0.0 {
            return Err(HdrError::InvalidLuminance(scene_linear));
        }
        if scene_linear <= SCENE_LINEAR_THRESHOLD {
            Ok((3.0 * scene_linear).sqrt())
        } else {
            Ok(HLG_A * (12.0 * scene_linear - HLG_B).ln() + HLG_C)
        }
    }

    /// HLG EOTF — HLG signal [0, 1] → display-linear light (before OOTF).
    ///
    /// This is the inverse of the OETF.  The result is then composed with
    /// the OOTF to yield the final displayed luminance.
    ///
    /// # Errors
    /// Returns [`HdrError::InvalidLuminance`] if `signal` is outside [0, 1].
    pub fn eotf(&self, signal: f32) -> Result<f32> {
        if !(0.0..=1.0).contains(&signal) {
            return Err(HdrError::InvalidLuminance(signal));
        }
        // Invert OETF to recover scene linear.
        let scene_linear = if signal <= SIGNAL_THRESHOLD {
            (signal * signal) / 3.0
        } else {
            (((signal - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
        };
        // Apply OOTF with the display's nominal peak luminance.
        self.ootf(scene_linear, self.nominal_peak_luminance)
    }

    /// HLG OOTF — scene-linear E' → display light F_d.
    ///
    /// Formula: `F_d = α * E'^γ_s * L_w`
    /// where α = `display_peak / reference_white`.
    ///
    /// The result is the display-referred luminance (not normalised).
    ///
    /// # Errors
    /// Returns [`HdrError::InvalidLuminance`] if `normalized_signal` is negative.
    pub fn ootf(&self, normalized_signal: f32, display_peak_luma: f32) -> Result<f32> {
        if normalized_signal < 0.0 {
            return Err(HdrError::InvalidLuminance(normalized_signal));
        }
        let alpha = display_peak_luma / self.reference_white;
        Ok(alpha * normalized_signal.powf(self.system_gamma) * self.reference_white)
    }
}

// ── HlgSdrConvert ─────────────────────────────────────────────────────────────

/// Converts HLG-encoded RGB (BT.2020 primaries) to BT.709 SDR RGB.
///
/// Pipeline:
/// 1. Apply `HlgSystem::eotf` per channel.
/// 2. Convert BT.2020 display-linear → BT.709 display-linear (3×3 matrix).
/// 3. Apply BT.709 OETF (gamma 1/2.2).
/// 4. Clamp output to [0, 1].
#[derive(Debug, Clone, Default)]
pub struct HlgSdrConvert {
    system: HlgSystem,
}

impl HlgSdrConvert {
    /// Create a converter with the given HLG system parameters.
    pub fn new(system: HlgSystem) -> Self {
        Self { system }
    }

    /// Convert a single HLG-encoded BT.2020 RGB triplet to BT.709 SDR.
    ///
    /// # Errors
    /// Propagates `HdrError::InvalidLuminance` if any channel is out of [0, 1].
    pub fn hdr_to_sdr(&self, r: f32, g: f32, b: f32) -> Result<(f32, f32, f32)> {
        // Step 1 — HLG EOTF per channel.
        let r_lin = self.system.eotf(r)?;
        let g_lin = self.system.eotf(g)?;
        let b_lin = self.system.eotf(b)?;

        // Step 2 — BT.2020 → BT.709 linear matrix.
        // Values from ITU-R BT.2087 Table 2.
        let (r_709, g_709, b_709) = bt2020_to_bt709_linear(r_lin, g_lin, b_lin);

        // Step 3 — BT.709 OETF (gamma 1/2.2) + clamp.
        let r_sdr = r_709.max(0.0).powf(BT709_GAMMA).min(1.0);
        let g_sdr = g_709.max(0.0).powf(BT709_GAMMA).min(1.0);
        let b_sdr = b_709.max(0.0).powf(BT709_GAMMA).min(1.0);

        Ok((r_sdr, g_sdr, b_sdr))
    }
}

// ── HlgHdr10Convert ───────────────────────────────────────────────────────────

/// Converts HLG-encoded RGB to PQ-encoded (HDR10) RGB at a given display peak.
///
/// Pipeline:
/// 1. Apply HLG EOTF (scene → display linear, normalised to `display_peak` nits).
/// 2. Normalise to PQ reference (10 000 nits).
/// 3. Apply PQ OETF to produce [0, 1] PQ signal.
#[derive(Debug, Clone, Default)]
pub struct HlgHdr10Convert {
    system: HlgSystem,
}

impl HlgHdr10Convert {
    /// Create with custom system parameters.
    pub fn new(system: HlgSystem) -> Self {
        Self { system }
    }

    /// Convert HLG RGB (BT.2020) to PQ RGB (BT.2020, HDR10).
    ///
    /// `display_peak` is the target display peak in nits (e.g. 1000.0).
    ///
    /// # Errors
    /// Propagates `HdrError::InvalidLuminance` from the EOTF.
    pub fn hlg_to_pq(&self, r: f32, g: f32, b: f32, display_peak: f32) -> Result<(f32, f32, f32)> {
        // Step 1 — HLG EOTF → display-linear, then normalise to display peak.
        let r_sys = self.system.eotf(r)?;
        let g_sys = self.system.eotf(g)?;
        let b_sys = self.system.eotf(b)?;

        // eotf() returns values in units where reference_white ≈ 203 nits and
        // the display peak is `nominal_peak_luminance`. We re-scale from
        // nominal_peak_luminance to the caller-supplied display_peak first,
        // then normalise to 10 000 nits for PQ.
        let scale = display_peak / (self.system.nominal_peak_luminance * 10_000.0);
        let r_pq_lin = r_sys * scale;
        let g_pq_lin = g_sys * scale;
        let b_pq_lin = b_sys * scale;

        // Step 2 — PQ OETF.
        let r_pq = pq_oetf_f32(r_pq_lin);
        let g_pq = pq_oetf_f32(g_pq_lin);
        let b_pq = pq_oetf_f32(b_pq_lin);

        Ok((r_pq, g_pq, b_pq))
    }
}

// ── System gamma adjustment per BT.2100 Annex 1 ─────────────────────────────

/// Compute the HLG system gamma adjusted for a given display peak luminance.
///
/// ITU-R BT.2100-2 Annex 1 specifies that the system gamma should be
/// adapted to the display's peak luminance to maintain consistent apparent
/// brightness across displays of different capabilities:
///
/// ```text
///     γ = 1.2 + 0.42 * log10(L_W / 1000)
/// ```
///
/// where L_W is the display's nominal peak luminance in nits.
///
/// This means:
/// - At 1000 nits (the reference): γ = 1.2 (the BT.2100 default)
/// - At 400 nits (a typical HDR laptop): γ ≈ 1.03 (less contrast boost)
/// - At 4000 nits (a mastering display): γ ≈ 1.45 (more contrast boost)
///
/// # Arguments
/// - `display_peak_nits`: the display's nominal peak luminance in nits (must be > 0)
///
/// # Returns
/// The adapted system gamma. Returns 1.2 for invalid inputs.
pub fn hlg_adapted_system_gamma(display_peak_nits: f32) -> f32 {
    if display_peak_nits <= 0.0 {
        return 1.2; // fallback to BT.2100 default
    }

    // BT.2100-2, Table 5, Note 3b:
    //   γ = 1.2 + 0.42 * log10(L_W / 1000)
    let log_ratio = (display_peak_nits / 1000.0).log10();
    let gamma = 1.2 + 0.42 * log_ratio;

    // Clamp to a physically reasonable range.
    gamma.clamp(0.6, 2.0)
}

/// Create an [`HlgSystem`] automatically adapted to the target display.
///
/// The system gamma is computed from the peak luminance using the
/// BT.2100 Annex 1 formula, and the reference white is kept at the
/// BT.2100-2 standard of 203 nits.
///
/// # Arguments
/// - `display_peak_nits`: the display's nominal peak luminance in nits
pub fn hlg_system_for_display(display_peak_nits: f32) -> HlgSystem {
    let gamma = hlg_adapted_system_gamma(display_peak_nits);
    HlgSystem::new(gamma, display_peak_nits, 203.0)
}

/// Batch-convert an interleaved HLG RGB frame to SDR, adapting the system
/// gamma for the specified display peak luminance.
///
/// # Arguments
/// - `pixels`: interleaved HLG-encoded BT.2020 RGB (length must be divisible by 3)
/// - `display_peak_nits`: the display's nominal peak luminance in nits
///
/// # Errors
/// - `HdrError::InvalidLuminance` if any pixel is outside [0, 1]
/// - Returns an error if the pixel buffer length is not divisible by 3
pub fn hlg_to_sdr_adaptive(pixels: &[f32], display_peak_nits: f32) -> Result<Vec<f32>> {
    if !pixels.len().is_multiple_of(3) {
        return Err(HdrError::ToneMappingError(format!(
            "pixel buffer length {} is not divisible by 3",
            pixels.len()
        )));
    }

    let system = hlg_system_for_display(display_peak_nits);
    let converter = HlgSdrConvert::new(system);

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(3) {
        let (r, g, b) = converter.hdr_to_sdr(chunk[0], chunk[1], chunk[2])?;
        out.push(r);
        out.push(g);
        out.push(b);
    }
    Ok(out)
}

// ── Standalone system gamma / OOTF / HLG→PQ helpers ─────────────────────────

/// HLG system gamma per BT.2100-2 Annex 1.
///
/// ```text
/// γ = 1.2 + 0.42 × log₁₀(Lₒ / 1000)
/// ```
///
/// # Arguments
/// - `l_display_nits`: target display peak luminance in nits (e.g., 1000, 400, 300)
///
/// Returns the system gamma; the formula is applied without clamping to allow
/// full precision at any luminance level.
pub fn hlg_system_gamma(l_display_nits: f32) -> f32 {
    1.2 + 0.42 * (l_display_nits / 1000.0).log10()
}

/// Apply HLG OOTF (Optical-to-Optical Transfer Function) with display-dependent gamma.
///
/// Implements the scene-to-display mapping:
/// ```text
/// L_d = E_s^(γ-1) × E_s  =  E_s^γ
/// ```
/// where γ is the BT.2100 system gamma for the given display peak luminance.
///
/// # Arguments
/// - `signal`: scene-referred HLG signal in [0, 1]
/// - `l_display_nits`: target display peak luminance in nits
///
/// Returns display-linear light, normalised to [0, 1] for the given peak.
pub fn hlg_ootf(signal: f32, l_display_nits: f32) -> f32 {
    let gamma = hlg_system_gamma(l_display_nits);
    signal.max(0.0).powf(gamma - 1.0) * signal.max(0.0)
}

/// HLG to HDR10/PQ conversion with display-adaptive system gamma.
///
/// Pipeline:
/// 1. Apply display-adaptive OOTF (`hlg_ootf`) to convert scene HLG → display-linear.
/// 2. Normalise to PQ reference peak (10 000 nits) using `l_display_nits`.
/// 3. Encode as PQ signal (SMPTE ST 2084 OETF).
///
/// # Arguments
/// - `hlg`: HLG scene-referred signal in [0, 1]
/// - `l_display_nits`: target display peak luminance in nits
///
/// Returns a PQ-encoded signal in [0, 1].
pub fn hlg_to_pq_with_gamma(hlg: f32, l_display_nits: f32) -> f32 {
    let ootf_out = hlg_ootf(hlg, l_display_nits);
    // Normalise display-linear output to PQ reference (10 000 nits).
    let pq_lin = (ootf_out * l_display_nits / 10_000.0).max(0.0);
    pq_oetf_f32(pq_lin)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// BT.2020 display-linear → BT.709 display-linear colour matrix.
///
/// Derived from ITU-R BT.2087-0, Table 2 (row for BT.2020 → BT.709).
#[inline]
fn bt2020_to_bt709_linear(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r_out = 1.660_49 * r - 0.587_79 * g - 0.072_69 * b;
    let g_out = -0.124_55 * r + 1.132_90 * g - 0.008_36 * b;
    let b_out = -0.018_15 * r - 0.100_56 * g + 1.118_72 * b;
    (r_out, g_out, b_out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const LOOSE: f32 = 1e-3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // 1. Default constructor
    #[test]
    fn test_hlg_system_default() {
        let s = HlgSystem::default();
        assert!(approx(s.system_gamma, 1.2, EPS));
        assert!(approx(s.nominal_peak_luminance, 1000.0, EPS));
        assert!(approx(s.reference_white, 203.0, EPS));
    }

    // 2. OETF: zero → zero
    #[test]
    fn test_oetf_zero() {
        let s = HlgSystem::default();
        let v = s.oetf(0.0).expect("oetf(0)");
        assert!(approx(v, 0.0, EPS));
    }

    // 3. OETF: threshold boundary (E' = 1/12, should hit low-end branch)
    #[test]
    fn test_oetf_threshold_boundary() {
        let s = HlgSystem::default();
        let threshold = 1.0_f32 / 12.0;
        let v = s.oetf(threshold).expect("oetf(1/12)");
        // sqrt(3 * 1/12) = 0.5
        assert!(approx(v, 0.5, LOOSE));
    }

    // 4. OETF: above threshold
    #[test]
    fn test_oetf_above_threshold() {
        let s = HlgSystem::default();
        let v = s.oetf(0.5).expect("oetf(0.5)");
        assert!(v > 0.5 && v <= 1.0, "oetf(0.5) = {v}");
    }

    // 5. OETF: negative error
    #[test]
    fn test_oetf_negative_error() {
        let s = HlgSystem::default();
        assert!(s.oetf(-0.01).is_err());
    }

    // 6. EOTF: zero → zero (via OOTF)
    #[test]
    fn test_eotf_zero() {
        let s = HlgSystem::default();
        let v = s.eotf(0.0).expect("eotf(0)");
        assert!(approx(v, 0.0, EPS));
    }

    // 7. EOTF: out-of-range error
    #[test]
    fn test_eotf_out_of_range() {
        let s = HlgSystem::default();
        assert!(s.eotf(1.1).is_err());
        assert!(s.eotf(-0.1).is_err());
    }

    // 8. EOTF: mid-range produces finite result
    #[test]
    fn test_eotf_mid_range() {
        let s = HlgSystem::default();
        let v = s.eotf(0.5).expect("eotf(0.5)");
        assert!(v.is_finite() && v >= 0.0);
    }

    // 9. OOTF: linearity check (E'=0 → F_d=0)
    #[test]
    fn test_ootf_zero() {
        let s = HlgSystem::default();
        let v = s.ootf(0.0, 1000.0).expect("ootf(0)");
        assert!(approx(v, 0.0, EPS));
    }

    // 10. OOTF: negative signal error
    #[test]
    fn test_ootf_negative_error() {
        let s = HlgSystem::default();
        assert!(s.ootf(-0.5, 1000.0).is_err());
    }

    // 11. OOTF: scale proportionality (double display_peak → higher output)
    #[test]
    fn test_ootf_scaling() {
        let s = HlgSystem::default();
        let v1 = s.ootf(0.5, 1000.0).expect("ootf v1");
        let v2 = s.ootf(0.5, 2000.0).expect("ootf v2");
        assert!(v2 > v1, "higher display peak should give higher output");
    }

    // 12. SDR convert: black stays black
    #[test]
    fn test_sdr_black() {
        let c = HlgSdrConvert::default();
        let (r, g, b) = c.hdr_to_sdr(0.0, 0.0, 0.0).expect("black");
        assert!(approx(r, 0.0, EPS));
        assert!(approx(g, 0.0, EPS));
        assert!(approx(b, 0.0, EPS));
    }

    // 13. SDR convert: out-of-range error
    #[test]
    fn test_sdr_out_of_range() {
        let c = HlgSdrConvert::default();
        assert!(c.hdr_to_sdr(1.5, 0.0, 0.0).is_err());
    }

    // 14. SDR convert: output is in [0, 1]
    #[test]
    fn test_sdr_output_range() {
        let c = HlgSdrConvert::default();
        let (r, g, b) = c.hdr_to_sdr(0.5, 0.4, 0.3).expect("mid");
        assert!((0.0..=1.0).contains(&r));
        assert!((0.0..=1.0).contains(&g));
        assert!((0.0..=1.0).contains(&b));
    }

    // 15. HLG-to-PQ: black stays black
    #[test]
    fn test_hlg_to_pq_black() {
        let c = HlgHdr10Convert::default();
        let (r, g, b) = c.hlg_to_pq(0.0, 0.0, 0.0, 1000.0).expect("black");
        assert!(approx(r, 0.0, LOOSE));
        assert!(approx(g, 0.0, LOOSE));
        assert!(approx(b, 0.0, LOOSE));
    }

    // 16. HLG-to-PQ: output is in [0, 1]
    #[test]
    fn test_hlg_to_pq_output_range() {
        let c = HlgHdr10Convert::default();
        let (r, g, b) = c.hlg_to_pq(0.5, 0.3, 0.2, 1000.0).expect("mid");
        assert!((0.0..=1.0).contains(&r), "r = {r}");
        assert!((0.0..=1.0).contains(&g), "g = {g}");
        assert!((0.0..=1.0).contains(&b), "b = {b}");
    }

    // 17. HLG-to-PQ: out-of-range input error
    #[test]
    fn test_hlg_to_pq_oor_error() {
        let c = HlgHdr10Convert::default();
        assert!(c.hlg_to_pq(-0.1, 0.0, 0.0, 1000.0).is_err());
    }

    // 18. Custom system: custom gamma changes OOTF output
    #[test]
    fn test_custom_system_gamma() {
        let s1 = HlgSystem::new(1.0, 1000.0, 203.0);
        let s2 = HlgSystem::new(1.4, 1000.0, 203.0);
        let v1 = s1.ootf(0.5, 1000.0).expect("ootf gamma=1");
        let v2 = s2.ootf(0.5, 1000.0).expect("ootf gamma=1.4");
        assert!(
            (v1 - v2).abs() > 1e-3,
            "different gammas should differ: {v1} vs {v2}"
        );
    }

    // 19. OETF: unity at E'=1.0 is well-defined
    #[test]
    fn test_oetf_unity_approx() {
        let s = HlgSystem::default();
        let v = s.oetf(1.0).expect("oetf(1)");
        // HLG OETF(1.0) ≈ 1.0 (BT.2100 design property)
        assert!(approx(v, 1.0, LOOSE), "oetf(1.0) = {v}");
    }

    // 20. bt2020→bt709 matrix: grey stays grey (approx)
    #[test]
    fn test_bt2020_to_bt709_grey() {
        let grey = 0.5_f32;
        let (r, g, b) = bt2020_to_bt709_linear(grey, grey, grey);
        // For achromatic (equal R=G=B) the matrix should preserve neutrality.
        let expected = grey;
        assert!(approx(r, expected, LOOSE), "r={r}");
        assert!(approx(g, expected, LOOSE), "g={g}");
        assert!(approx(b, expected, LOOSE), "b={b}");
    }

    // ── System gamma adjustment tests ───────────────────────────────────

    #[test]
    fn test_adapted_gamma_reference_1000_nits() {
        // At the reference 1000 nits, gamma should be exactly 1.2.
        let gamma = hlg_adapted_system_gamma(1000.0);
        assert!(approx(gamma, 1.2, 0.01), "1000 nit gamma: {gamma}");
    }

    #[test]
    fn test_adapted_gamma_400_nits() {
        // At 400 nits: γ = 1.2 + 0.42 * log10(0.4) ≈ 1.2 + 0.42*(-0.3979) ≈ 1.033
        let gamma = hlg_adapted_system_gamma(400.0);
        assert!(gamma < 1.2, "400 nit gamma should be < 1.2: {gamma}");
        assert!(gamma > 0.9, "400 nit gamma should be > 0.9: {gamma}");
    }

    #[test]
    fn test_adapted_gamma_4000_nits() {
        // At 4000 nits: γ = 1.2 + 0.42 * log10(4.0) ≈ 1.2 + 0.42*0.602 ≈ 1.453
        let gamma = hlg_adapted_system_gamma(4000.0);
        assert!(gamma > 1.2, "4000 nit gamma should be > 1.2: {gamma}");
        assert!(gamma < 1.6, "4000 nit gamma should be < 1.6: {gamma}");
    }

    #[test]
    fn test_adapted_gamma_monotonic_with_luminance() {
        // Higher display peak should yield higher system gamma.
        let g100 = hlg_adapted_system_gamma(100.0);
        let g400 = hlg_adapted_system_gamma(400.0);
        let g1000 = hlg_adapted_system_gamma(1000.0);
        let g4000 = hlg_adapted_system_gamma(4000.0);
        assert!(g100 < g400, "100 < 400: {g100} < {g400}");
        assert!(g400 < g1000, "400 < 1000: {g400} < {g1000}");
        assert!(g1000 < g4000, "1000 < 4000: {g1000} < {g4000}");
    }

    #[test]
    fn test_adapted_gamma_zero_nits_fallback() {
        let gamma = hlg_adapted_system_gamma(0.0);
        assert!(
            approx(gamma, 1.2, EPS),
            "zero nits should fallback: {gamma}"
        );
    }

    #[test]
    fn test_adapted_gamma_negative_nits_fallback() {
        let gamma = hlg_adapted_system_gamma(-500.0);
        assert!(
            approx(gamma, 1.2, EPS),
            "negative nits should fallback: {gamma}"
        );
    }

    #[test]
    fn test_adapted_gamma_clamped_low() {
        // Very low luminance should not produce unreasonably low gamma.
        let gamma = hlg_adapted_system_gamma(1.0);
        assert!(
            gamma >= 0.6,
            "very low nits gamma should be >= 0.6: {gamma}"
        );
    }

    #[test]
    fn test_adapted_gamma_clamped_high() {
        // Very high luminance should not produce unreasonably high gamma.
        let gamma = hlg_adapted_system_gamma(100_000.0);
        assert!(
            gamma <= 2.0,
            "very high nits gamma should be <= 2.0: {gamma}"
        );
    }

    // ── hlg_system_for_display tests ────────────────────────────────────

    #[test]
    fn test_system_for_display_1000() {
        let sys = hlg_system_for_display(1000.0);
        assert!(approx(sys.system_gamma, 1.2, 0.01));
        assert!(approx(sys.nominal_peak_luminance, 1000.0, EPS));
        assert!(approx(sys.reference_white, 203.0, EPS));
    }

    #[test]
    fn test_system_for_display_600() {
        let sys = hlg_system_for_display(600.0);
        // γ = 1.2 + 0.42 * log10(0.6) ≈ 1.2 + 0.42 * (-0.2218) ≈ 1.107
        assert!(
            sys.system_gamma > 1.0 && sys.system_gamma < 1.2,
            "600 nit system gamma: {}",
            sys.system_gamma
        );
        assert!(approx(sys.nominal_peak_luminance, 600.0, EPS));
    }

    #[test]
    fn test_system_for_display_eotf_output_differs() {
        // Two displays with different peaks should produce different EOTF output.
        let sys_400 = hlg_system_for_display(400.0);
        let sys_1000 = hlg_system_for_display(1000.0);
        let v_400 = sys_400.eotf(0.5).expect("eotf 400");
        let v_1000 = sys_1000.eotf(0.5).expect("eotf 1000");
        assert!(
            (v_400 - v_1000).abs() > 1e-2,
            "different display peaks should yield different EOTF: {v_400} vs {v_1000}"
        );
    }

    // ── hlg_to_sdr_adaptive tests ───────────────────────────────────────

    #[test]
    fn test_adaptive_sdr_black() {
        let pixels = vec![0.0_f32; 6]; // 2 black pixels
        let out = hlg_to_sdr_adaptive(&pixels, 1000.0).expect("adaptive black");
        assert_eq!(out.len(), 6);
        for v in &out {
            assert!(approx(*v, 0.0, EPS), "black pixel: {v}");
        }
    }

    #[test]
    fn test_adaptive_sdr_output_range() {
        let pixels = vec![0.3_f32, 0.5, 0.7, 0.1, 0.2, 0.4];
        let out = hlg_to_sdr_adaptive(&pixels, 600.0).expect("adaptive mid");
        for v in &out {
            assert!(
                (0.0..=1.0).contains(v),
                "adaptive SDR pixel out of range: {v}"
            );
        }
    }

    #[test]
    fn test_adaptive_sdr_invalid_length() {
        let pixels = vec![0.5_f32; 5]; // not divisible by 3
        assert!(hlg_to_sdr_adaptive(&pixels, 1000.0).is_err());
    }

    #[test]
    fn test_adaptive_sdr_differs_from_default() {
        // Adaptive conversion at 400 nits should produce different output
        // than the default system (1000 nits). Use low signal values to
        // avoid clamp-to-1.0 masking the difference.
        let pixels = vec![0.15_f32, 0.10, 0.05];
        let out_default = {
            let c = HlgSdrConvert::default();
            let (r, g, b) = c.hdr_to_sdr(0.15, 0.10, 0.05).expect("default");
            vec![r, g, b]
        };
        let out_400 = hlg_to_sdr_adaptive(&pixels, 400.0).expect("adaptive 400");
        let diff: f32 = out_default
            .iter()
            .zip(out_400.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "adaptive 400 nit should differ from default: diff={diff}"
        );
    }

    #[test]
    fn test_adaptive_sdr_higher_peak_different() {
        // Different display peaks should produce different SDR output.
        // Use low signal values to avoid saturation at the gamma clamp.
        let pixels = vec![0.10_f32, 0.08, 0.05];
        let out_400 = hlg_to_sdr_adaptive(&pixels, 400.0).expect("400");
        let out_4000 = hlg_to_sdr_adaptive(&pixels, 4000.0).expect("4000");
        let diff: f32 = out_400
            .iter()
            .zip(out_4000.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "different peaks should produce different SDR output: diff={diff}"
        );
    }

    // ── hlg_system_gamma / hlg_ootf / hlg_to_pq_with_gamma tests ────────────

    #[test]
    fn test_hlg_system_gamma_at_1000_nits() {
        // At 1000 nits, log10(1) = 0, so gamma = 1.2
        let g = hlg_system_gamma(1000.0);
        assert!(
            (g - 1.2).abs() < 1e-5,
            "gamma at 1000 nits should be 1.2, got {g}"
        );
    }

    #[test]
    fn test_hlg_system_gamma_at_100_nits() {
        // gamma = 1.2 + 0.42 * log10(0.1) = 1.2 + 0.42 * (-1) = 0.78
        let g = hlg_system_gamma(100.0);
        assert!(
            (g - 0.78).abs() < 1e-4,
            "gamma at 100 nits should be ~0.78, got {g}"
        );
    }

    #[test]
    fn test_hlg_system_gamma_at_4000_nits() {
        // gamma = 1.2 + 0.42 * log10(4) ≈ 1.2 + 0.42 * 0.602 ≈ 1.453
        let g = hlg_system_gamma(4000.0);
        assert!(g > 1.2, "gamma at 4000 nits should be > 1.2, got {g}");
    }

    #[test]
    fn test_hlg_ootf_zero_signal() {
        let out = hlg_ootf(0.0, 1000.0);
        assert!(
            out.abs() < 1e-6,
            "ootf of zero signal should be zero, got {out}"
        );
    }

    #[test]
    fn test_hlg_ootf_mid_signal_positive() {
        let out = hlg_ootf(0.5, 1000.0);
        assert!(
            out > 0.0 && out <= 1.0,
            "ootf(0.5, 1000) should be in (0,1], got {out}"
        );
    }

    #[test]
    fn test_hlg_ootf_higher_peak_differs() {
        let out_400 = hlg_ootf(0.5, 400.0);
        let out_1000 = hlg_ootf(0.5, 1000.0);
        assert!(
            (out_400 - out_1000).abs() > 1e-4,
            "different display peaks should yield different ootf output: {out_400} vs {out_1000}"
        );
    }

    #[test]
    fn test_hlg_to_pq_with_gamma_zero() {
        let pq = hlg_to_pq_with_gamma(0.0, 1000.0);
        assert!(pq.abs() < 1e-4, "pq of zero should be ~0, got {pq}");
    }

    #[test]
    fn test_hlg_to_pq_with_gamma_range() {
        let pq = hlg_to_pq_with_gamma(0.5, 1000.0);
        assert!(
            (0.0..=1.0).contains(&pq),
            "pq output should be in [0,1], got {pq}"
        );
    }

    #[test]
    fn test_hlg_to_pq_with_gamma_differs_by_peak() {
        let pq_400 = hlg_to_pq_with_gamma(0.5, 400.0);
        let pq_4000 = hlg_to_pq_with_gamma(0.5, 4000.0);
        assert!(
            (pq_400 - pq_4000).abs() > 1e-4,
            "different peaks should yield different PQ output: {pq_400} vs {pq_4000}"
        );
    }
}
