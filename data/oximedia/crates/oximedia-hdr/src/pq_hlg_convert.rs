//! PQ ↔ HLG bidirectional conversion with correct OOTF / system gamma.
//!
//! Implements the cross-format conversion pipeline between SMPTE ST 2084 (PQ)
//! and ARIB STD-B67 (HLG) as recommended by ITU-R BT.2408 / BT.2100.
//!
//! ## Pipeline overview
//!
//! ### PQ → HLG
//! ```text
//! PQ signal [0,1]
//!   → PQ EOTF → linear display-referred nits
//!   → normalise by peak_luminance → [0,1] display-referred
//!   → [opt] inverse OOTF: display^(1/γ) → scene-referred [0,1]
//!   → HLG OETF → HLG signal [0,1]
//! ```
//!
//! ### HLG → PQ
//! ```text
//! HLG signal [0,1]
//!   → HLG EOTF → scene-referred [0,1]
//!   → [opt] OOTF: scene^γ → display-referred [0,1]
//!   → scale × (peak_luminance / 10 000) → PQ-normalised [0,1]
//!   → PQ OETF → PQ signal [0,1]
//! ```

use crate::transfer_function::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf};
use crate::{HdrError, Result};

// ── PqHlgConverterConfig ──────────────────────────────────────────────────────

/// Configuration for PQ ↔ HLG conversion.
#[derive(Debug, Clone)]
pub struct PqHlgConverterConfig {
    /// Peak luminance of the PQ content in nits.
    ///
    /// Common values: 1000, 4000, 10 000 nits.  The default is 1000 nits.
    pub peak_luminance: f32,

    /// HLG system gamma γ.
    ///
    /// When `None` (default) the value is derived from `peak_luminance` via
    /// the BT.2100-2 formula: `γ = 1.2 + 0.42 × log₁₀(L_w / 1000)`.
    pub system_gamma: Option<f32>,

    /// Scene-referred mode.
    ///
    /// When `true` the OOTF (scene → display) is skipped; the conversion
    /// operates purely on scene-referred signals.  Default: `false`.
    pub scene_referred: bool,
}

impl Default for PqHlgConverterConfig {
    fn default() -> Self {
        Self {
            peak_luminance: 1000.0,
            system_gamma: None,
            scene_referred: false,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the effective HLG system gamma for `config`.
///
/// If `config.system_gamma` is `Some(g)` that value is returned directly
/// (clamped to `[1.0, 1.6]`).  Otherwise the BT.2100-2 formula is applied:
///
/// ```text
/// γ = 1.2 + 0.42 × log₁₀(peak_luminance / 1000)
/// ```
///
/// The result is clamped to `[1.0, 1.6]`.
pub fn effective_system_gamma(config: &PqHlgConverterConfig) -> f32 {
    let raw = config.system_gamma.unwrap_or_else(|| {
        let ratio = (config.peak_luminance / 1000.0).max(f32::MIN_POSITIVE);
        1.2_f32 + 0.42_f32 * ratio.log10()
    });
    raw.clamp(1.0, 1.6)
}

// ── PQ → HLG ─────────────────────────────────────────────────────────────────

/// Convert a PQ-encoded RGB frame to an HLG-encoded RGB frame.
///
/// `frame` must be interleaved RGB (length divisible by 3), with each sample
/// in `[0.0, 1.0]` representing the normalised PQ code value (where 1.0 ≡
/// 10 000 nits per SMPTE ST 2084).
///
/// The output is an interleaved RGB frame with HLG signal values in `[0.0, 1.0]`.
///
/// # Errors
///
/// - [`HdrError::ToneMappingError`] if `frame.len()` is not divisible by 3.
/// - [`HdrError::InvalidLuminance`] if a PQ sample is outside `[0.0, 1.0]`.
pub fn pq_to_hlg_convert(frame: &[f32], config: &PqHlgConverterConfig) -> Result<Vec<f32>> {
    if !frame.len().is_multiple_of(3) {
        return Err(HdrError::ToneMappingError(format!(
            "frame length {} is not divisible by 3",
            frame.len()
        )));
    }
    if frame.is_empty() {
        return Ok(Vec::new());
    }

    let gamma = effective_system_gamma(config);
    let inv_gamma = 1.0_f64 / f64::from(gamma);
    let peak = f64::from(config.peak_luminance);

    let mut out = Vec::with_capacity(frame.len());

    let mut i = 0usize;
    while i + 2 < frame.len() {
        let r_pq = frame[i];
        let g_pq = frame[i + 1];
        let b_pq = frame[i + 2];

        // 1. PQ EOTF: PQ signal → linear-light normalised to [0, 1] (1.0 = 10 000 nits).
        let r_lin = pq_eotf(f64::from(r_pq))?;
        let g_lin = pq_eotf(f64::from(g_pq))?;
        let b_lin = pq_eotf(f64::from(b_pq))?;

        // 2. Convert to nits, then normalise by peak_luminance → [0, 1].
        let r_disp = (r_lin * 10_000.0 / peak).clamp(0.0, 1.0);
        let g_disp = (g_lin * 10_000.0 / peak).clamp(0.0, 1.0);
        let b_disp = (b_lin * 10_000.0 / peak).clamp(0.0, 1.0);

        // 3. Inverse OOTF: display-referred → scene-referred (unless scene_referred).
        let (r_scene, g_scene, b_scene) = if config.scene_referred {
            (r_disp, g_disp, b_disp)
        } else {
            (
                r_disp.powf(inv_gamma),
                g_disp.powf(inv_gamma),
                b_disp.powf(inv_gamma),
            )
        };

        // 4. HLG OETF: scene-linear → HLG signal.
        let r_hlg = hlg_oetf(r_scene)?;
        let g_hlg = hlg_oetf(g_scene)?;
        let b_hlg = hlg_oetf(b_scene)?;

        out.push(r_hlg as f32);
        out.push(g_hlg as f32);
        out.push(b_hlg as f32);

        i += 3;
    }

    Ok(out)
}

// ── HLG → PQ ─────────────────────────────────────────────────────────────────

/// Convert an HLG-encoded RGB frame to a PQ-encoded RGB frame.
///
/// `frame` must be interleaved RGB (length divisible by 3), with each sample
/// in `[0.0, 1.0]` representing the HLG signal value.
///
/// The output is an interleaved RGB frame with PQ signal values in `[0.0, 1.0]`.
///
/// # Errors
///
/// - [`HdrError::ToneMappingError`] if `frame.len()` is not divisible by 3.
/// - [`HdrError::InvalidLuminance`] if an HLG sample is outside `[0.0, 1.0]`.
pub fn hlg_to_pq_convert(frame: &[f32], config: &PqHlgConverterConfig) -> Result<Vec<f32>> {
    if !frame.len().is_multiple_of(3) {
        return Err(HdrError::ToneMappingError(format!(
            "frame length {} is not divisible by 3",
            frame.len()
        )));
    }
    if frame.is_empty() {
        return Ok(Vec::new());
    }

    let gamma = effective_system_gamma(config);
    let peak = f64::from(config.peak_luminance);

    let mut out = Vec::with_capacity(frame.len());

    let mut i = 0usize;
    while i + 2 < frame.len() {
        let r_hlg = frame[i];
        let g_hlg = frame[i + 1];
        let b_hlg = frame[i + 2];

        // 1. HLG EOTF: HLG signal → scene-linear [0, 1].
        let r_scene = hlg_eotf(f64::from(r_hlg))?;
        let g_scene = hlg_eotf(f64::from(g_hlg))?;
        let b_scene = hlg_eotf(f64::from(b_hlg))?;

        // 2. OOTF: scene-referred → display-referred (unless scene_referred).
        let (r_disp, g_disp, b_disp) = if config.scene_referred {
            (r_scene, g_scene, b_scene)
        } else {
            let gamma_f64 = f64::from(gamma);
            (
                r_scene.powf(gamma_f64),
                g_scene.powf(gamma_f64),
                b_scene.powf(gamma_f64),
            )
        };

        // 3. Scale to PQ normalisation (divide by 10 000 nits, multiply by peak).
        let scale = peak / 10_000.0;
        let r_pq_in = (r_disp * scale).clamp(0.0, 1.0);
        let g_pq_in = (g_disp * scale).clamp(0.0, 1.0);
        let b_pq_in = (b_disp * scale).clamp(0.0, 1.0);

        // 4. PQ OETF: linear → PQ signal.
        let r_pq = pq_oetf(r_pq_in)?.clamp(0.0, 1.0) as f32;
        let g_pq = pq_oetf(g_pq_in)?.clamp(0.0, 1.0) as f32;
        let b_pq = pq_oetf(b_pq_in)?.clamp(0.0, 1.0) as f32;

        out.push(r_pq);
        out.push(g_pq);
        out.push(b_pq);

        i += 3;
    }

    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // 1. Default config peak_luminance is 1000.
    #[test]
    fn test_default_config_peak_luminance() {
        let cfg = PqHlgConverterConfig::default();
        assert!((cfg.peak_luminance - 1000.0).abs() < f32::EPSILON);
        assert!(cfg.system_gamma.is_none());
        assert!(!cfg.scene_referred);
    }

    // 2. effective_gamma at 1000 nits = 1.2.
    #[test]
    fn test_effective_gamma_at_1000_nits() {
        let cfg = PqHlgConverterConfig {
            peak_luminance: 1000.0,
            ..Default::default()
        };
        let g = effective_system_gamma(&cfg);
        assert!(
            approx(g, 1.2, 1e-4),
            "γ at 1000 nits should be 1.2, got {}",
            g
        );
    }

    // 3. Custom system_gamma is returned as-is (within clamp).
    #[test]
    fn test_effective_gamma_custom() {
        let cfg = PqHlgConverterConfig {
            peak_luminance: 1000.0,
            system_gamma: Some(1.5),
            scene_referred: false,
        };
        let g = effective_system_gamma(&cfg);
        assert!(approx(g, 1.5, 1e-4), "Custom γ should be 1.5, got {}", g);
    }

    // 4. effective_gamma at 4000 nits = 1.2 + 0.42*log10(4) ≈ 1.453.
    #[test]
    fn test_effective_gamma_at_4000_nits() {
        let cfg = PqHlgConverterConfig {
            peak_luminance: 4000.0,
            ..Default::default()
        };
        let g = effective_system_gamma(&cfg);
        let expected = 1.2_f32 + 0.42_f32 * 4.0_f32.log10();
        assert!(
            approx(g, expected, 1e-3),
            "γ at 4000 nits: expected {:.4}, got {:.4}",
            expected,
            g
        );
    }

    // 5. pq_to_hlg_convert output has same length as input.
    #[test]
    fn test_pq_to_hlg_output_length() {
        let cfg = PqHlgConverterConfig::default();
        let frame = vec![0.5f32; 12];
        let out = pq_to_hlg_convert(&frame, &cfg).expect("pq_to_hlg");
        assert_eq!(out.len(), frame.len());
    }

    // 6. hlg_to_pq_convert output has same length as input.
    #[test]
    fn test_hlg_to_pq_output_length() {
        let cfg = PqHlgConverterConfig::default();
        let frame = vec![0.5f32; 12];
        let out = hlg_to_pq_convert(&frame, &cfg).expect("hlg_to_pq");
        assert_eq!(out.len(), frame.len());
    }

    // 7. Round-trip PQ→HLG→PQ mid-grey within ±0.01.
    #[test]
    fn test_roundtrip_pq_hlg_pq_midgrey() {
        let cfg = PqHlgConverterConfig {
            scene_referred: true,
            ..Default::default()
        };
        let orig = vec![0.5f32, 0.5, 0.5];
        let hlg = pq_to_hlg_convert(&orig, &cfg).expect("pq_to_hlg");
        let pq2 = hlg_to_pq_convert(&hlg, &cfg).expect("hlg_to_pq");
        assert_eq!(pq2.len(), 3);
        assert!(
            approx(pq2[0], orig[0], 0.01),
            "Round-trip PQ→HLG→PQ: expected {:.4}, got {:.4}",
            orig[0],
            pq2[0]
        );
    }

    // 8. Round-trip HLG→PQ→HLG mid-grey within ±0.01.
    #[test]
    fn test_roundtrip_hlg_pq_hlg_midgrey() {
        let cfg = PqHlgConverterConfig {
            scene_referred: true,
            ..Default::default()
        };
        let orig = vec![0.5f32, 0.5, 0.5];
        let pq = hlg_to_pq_convert(&orig, &cfg).expect("hlg_to_pq");
        let hlg2 = pq_to_hlg_convert(&pq, &cfg).expect("pq_to_hlg");
        assert_eq!(hlg2.len(), 3);
        assert!(
            approx(hlg2[0], orig[0], 0.01),
            "Round-trip HLG→PQ→HLG: expected {:.4}, got {:.4}",
            orig[0],
            hlg2[0]
        );
    }

    // 9. PQ black (0.0) → HLG ≈ 0.0.
    #[test]
    fn test_pq_black_stays_black() {
        let cfg = PqHlgConverterConfig::default();
        let frame = vec![0.0f32, 0.0, 0.0];
        let out = pq_to_hlg_convert(&frame, &cfg).expect("pq_to_hlg");
        for &v in &out {
            assert!(
                approx(v, 0.0, 0.001),
                "PQ black → HLG should be 0.0, got {}",
                v
            );
        }
    }

    // 10. HLG black (0.0) → PQ ≈ 0.0.
    #[test]
    fn test_hlg_black_stays_black() {
        let cfg = PqHlgConverterConfig::default();
        let frame = vec![0.0f32, 0.0, 0.0];
        let out = hlg_to_pq_convert(&frame, &cfg).expect("hlg_to_pq");
        for &v in &out {
            assert!(
                approx(v, 0.0, 0.001),
                "HLG black → PQ should be 0.0, got {}",
                v
            );
        }
    }

    // 11. Empty frame returns empty without error.
    #[test]
    fn test_empty_frame_returns_empty() {
        let cfg = PqHlgConverterConfig::default();
        let out1 = pq_to_hlg_convert(&[], &cfg).expect("pq_to_hlg empty");
        assert!(out1.is_empty());
        let out2 = hlg_to_pq_convert(&[], &cfg).expect("hlg_to_pq empty");
        assert!(out2.is_empty());
    }

    // 12. Frame length not divisible by 3 returns Err.
    #[test]
    fn test_invalid_frame_length_error() {
        let cfg = PqHlgConverterConfig::default();
        let frame = vec![0.5f32, 0.5];
        assert!(pq_to_hlg_convert(&frame, &cfg).is_err());
        assert!(hlg_to_pq_convert(&frame, &cfg).is_err());
    }

    // 13. scene_referred flag affects output for non-black input.
    #[test]
    fn test_scene_referred_flag_affects_output() {
        let frame = vec![0.4f32, 0.4, 0.4];
        let cfg_scene = PqHlgConverterConfig {
            scene_referred: true,
            ..Default::default()
        };
        let cfg_disp = PqHlgConverterConfig {
            scene_referred: false,
            ..Default::default()
        };
        let out_scene = pq_to_hlg_convert(&frame, &cfg_scene).expect("scene");
        let out_disp = pq_to_hlg_convert(&frame, &cfg_disp).expect("display");
        // With γ=1.2, the OOTF changes the signal; they should differ.
        assert!(
            (out_scene[0] - out_disp[0]).abs() > 0.001,
            "scene_referred=true vs false should produce different HLG values"
        );
    }

    // 14. pq_to_hlg output values in [0, 1].
    #[test]
    fn test_pq_to_hlg_output_in_range() {
        let cfg = PqHlgConverterConfig::default();
        let frame: Vec<f32> = (0..=10)
            .map(|i| i as f32 * 0.1)
            .flat_map(|v| [v, v, v])
            .collect();
        let out = pq_to_hlg_convert(&frame, &cfg).expect("pq_to_hlg");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "HLG output value {} is outside [0, 1]",
                v
            );
        }
    }

    // 15. hlg_to_pq output values in [0, 1].
    #[test]
    fn test_hlg_to_pq_output_in_range() {
        let cfg = PqHlgConverterConfig::default();
        let frame: Vec<f32> = (0..=10)
            .map(|i| i as f32 * 0.1)
            .flat_map(|v| [v, v, v])
            .collect();
        let out = hlg_to_pq_convert(&frame, &cfg).expect("hlg_to_pq");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "PQ output value {} is outside [0, 1]",
                v
            );
        }
    }

    // 16. Round-trip at 4000-nit peak within ±0.02.
    #[test]
    fn test_roundtrip_at_4000_nit_peak() {
        let cfg = PqHlgConverterConfig {
            peak_luminance: 4000.0,
            scene_referred: true,
            ..Default::default()
        };
        let orig = vec![0.3f32, 0.5, 0.7];
        let hlg = pq_to_hlg_convert(&orig, &cfg).expect("pq_to_hlg");
        let pq2 = hlg_to_pq_convert(&hlg, &cfg).expect("hlg_to_pq");
        for (i, (&a, &b)) in orig.iter().zip(pq2.iter()).enumerate() {
            assert!(
                approx(a, b, 0.02),
                "channel {}: expected {:.4}, got {:.4}",
                i,
                a,
                b
            );
        }
    }
}
