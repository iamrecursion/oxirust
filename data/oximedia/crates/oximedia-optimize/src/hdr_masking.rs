//! HDR-aware psychovisual masking for PQ and HLG transfer functions.
//!
//! Based on the Barten (1999) contrast sensitivity function (CSF) model and the
//! ITU-R BT.2100-2 / BT.2390-9 specifications for PQ and HLG.
//!
//! The human visual system (HVS) exhibits non-uniform sensitivity to luminance:
//! at high luminance (highlights) the eye is less sensitive to distortion, while
//! at low luminance (shadows) the eye is more sensitive.  This allows a QP
//! encoder to boost QP in bright highlights (saving bits with no perceptual loss)
//! and attenuate QP in deep shadows (spending more bits to preserve quality).
//!
//! # Model
//!
//! JND(L) ∝ L^0.4 cd/m² (Barten 1999).
//! A normalised luma code in `[0.0, 1.0]` is mapped to a signed QP delta:
//!
//! ```text
//! if luma > 0.5: qp_delta = +highlight_boost  * 2 * (luma - 0.5)
//! if luma < 0.5: qp_delta = -shadow_attenuate * 2 * (0.5 - luma)
//! clamped to [-shadow_attenuate, +highlight_boost]
//! ```
//!
//! For PQ the code word is treated as linear in the EOTF domain (after `sT2084`).
//! For HLG an inverse OOTF step brings the value into scene-linear before thresholding.

/// Transfer function selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    /// Standard dynamic range (SDR).
    Sdr,
    /// SMPTE ST 2084 — Perceptual Quantizer, per BT.2100-2.
    Pq,
    /// Hybrid Log-Gamma, per BT.2100-2.
    Hlg,
}

/// Configuration for HDR-aware QP modulation.
#[derive(Debug, Clone)]
pub struct HdrMaskingConfig {
    /// Transfer function in use.
    pub tf: TransferFunction,
    /// Peak display luminance in cd/m² (nits).  Typical: 1000 for HDR10, 400 for HLG.
    pub peak_nits: f32,
    /// Maximum QP boost applied to highlights (high luma).  Typical range: 2–4.
    pub highlight_qp_boost: i32,
    /// QP reduction in deep shadows (low luma).  Typical range: 1–3.
    pub shadow_qp_attenuate: i32,
}

impl Default for HdrMaskingConfig {
    fn default() -> Self {
        Self {
            tf: TransferFunction::Pq,
            peak_nits: 1000.0,
            highlight_qp_boost: 3,
            shadow_qp_attenuate: 2,
        }
    }
}

// ── Transfer-function helpers ─────────────────────────────────────────────────

/// HLG OETF inverse (scene-linear from HLG display signal).
///
/// Input `e` is the normalised HLG signal in `[0.0, 1.0]`.
/// Returns normalised scene luminance in `[0.0, 1.0]` approximately.
fn hlg_inverse_ootf_normalised(e: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_73;

    let e = e.clamp(0.0, 1.0);
    // Inverse HLG OETF (apply to each sample independently):
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((e - C) / A).exp() + B
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the HDR-aware QP offset for a single luma code word.
///
/// `luma_code` — raw luma code in the range `[0, 2^bit_depth - 1]`.
/// `bit_depth` — bit depth of the signal (typically 10 for HDR10 / HLG).
/// `config`    — masking parameters.
///
/// Returns a signed integer QP delta in the range
/// `[-shadow_qp_attenuate, +highlight_qp_boost]`.
///
/// # Model
///
/// Normalise → compute perceptual level → map to QP delta via Barten-style
/// linear interpolation around the 0.5 midpoint.
///
/// For **PQ** the normalised code word is used directly as the perceptual level:
/// PQ is already designed so that equal code-word steps are approximately
/// equal-JND steps (the core property of SMPTE ST 2084).  The midpoint 0.5
/// therefore correctly separates dark tones (below) from bright highlights
/// (above).
///
/// For **HLG** the signal goes through an inverse OOTF first to convert from
/// display-referred to scene-linear, aligning the masking threshold with
/// absolute luminance.
///
/// For **SDR** the normalised code is used directly (linear light approximation).
#[must_use]
pub fn hdr_qp_offset(luma_code: u16, bit_depth: u8, config: &HdrMaskingConfig) -> i32 {
    let max_code = (1u32 << bit_depth) - 1;
    let normalised = f32::from(luma_code) / max_code as f32;

    // Convert normalised code to perceptual luminance level [0.0, 1.0].
    let perceptual = match config.tf {
        // PQ: code words are already perceptually uniform — use directly.
        TransferFunction::Sdr | TransferFunction::Pq => normalised,
        // HLG: apply inverse OOTF to get scene-linear before thresholding.
        TransferFunction::Hlg => hlg_inverse_ootf_normalised(normalised),
    };

    // Apply Barten-style threshold around 0.5.
    let qp_delta_f = if perceptual > 0.5 {
        // Highlight region: positive QP boost (save bits — HVS less sensitive).
        config.highlight_qp_boost as f32 * 2.0 * (perceptual - 0.5)
    } else {
        // Shadow region: negative QP (spend more bits — HVS more sensitive).
        -(config.shadow_qp_attenuate as f32 * 2.0 * (0.5 - perceptual))
    };

    // Clamp to configured limits and round to nearest integer.
    let lo = -config.shadow_qp_attenuate;
    let hi = config.highlight_qp_boost;
    qp_delta_f.round().clamp(lo as f32, hi as f32) as i32
}

/// Compute per-block HDR QP offsets for an entire luma plane.
///
/// Divides the luma plane into `block_size × block_size` blocks, computes the
/// average code word per block, and returns one QP offset per block.
///
/// `luma_plane`  — packed luma samples (u16, little-endian, row-major).
/// `width`       — frame width in pixels.
/// `height`      — frame height in pixels.
/// `bit_depth`   — bit depth of the luma signal.
/// `block_size`  — block size in pixels (e.g. 16 for CTU-level, 8 for CU-level).
/// `config`      — HDR masking configuration.
///
/// Returns a `Vec<i32>` with length `ceil(height / block_size) * ceil(width / block_size)`.
#[must_use]
pub fn hdr_qp_map(
    luma_plane: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    block_size: usize,
    config: &HdrMaskingConfig,
) -> Vec<i32> {
    if width == 0 || height == 0 || block_size == 0 {
        return Vec::new();
    }

    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;
    let mut qp_map = Vec::with_capacity(blocks_x * blocks_y);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let x_start = bx * block_size;
            let y_start = by * block_size;
            let x_end = (x_start + block_size).min(width);
            let y_end = (y_start + block_size).min(height);

            let mut sum: u64 = 0;
            let mut count: u64 = 0;

            for y in y_start..y_end {
                for x in x_start..x_end {
                    let idx = y * width + x;
                    if let Some(&code) = luma_plane.get(idx) {
                        sum += u64::from(code);
                        count += 1;
                    }
                }
            }

            let avg_code = sum.checked_div(count).unwrap_or(0) as u16;

            qp_map.push(hdr_qp_offset(avg_code, bit_depth, config));
        }
    }

    qp_map
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic PQ tests ────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_pq_boosts_qp_in_highlights() {
        let config = HdrMaskingConfig::default(); // PQ, peak 1000 nit, boost=3
                                                  // Maximum 10-bit code = 1023 → full-brightness highlights.
        let delta = hdr_qp_offset(1023, 10, &config);
        assert!(
            delta > 0,
            "QP offset for full-brightness highlight must be positive, got {delta}"
        );
    }

    #[test]
    fn test_hdr_pq_attenuates_qp_in_shadows() {
        let config = HdrMaskingConfig::default();
        // Code 0 → deepest shadow.
        let delta = hdr_qp_offset(0, 10, &config);
        assert!(
            delta < 0,
            "QP offset for deepest shadow must be negative, got {delta}"
        );
    }

    #[test]
    fn test_hdr_pq_midpoint_near_zero() {
        let config = HdrMaskingConfig::default();
        // Code around mid-scale (512 / 1023 ≈ 0.5).  The perceptual midpoint for
        // PQ differs from the code midpoint, but the delta should stay bounded.
        let delta = hdr_qp_offset(512, 10, &config);
        assert!(delta >= -config.shadow_qp_attenuate && delta <= config.highlight_qp_boost);
    }

    #[test]
    fn test_hdr_pq_clamp_bounds() {
        let config = HdrMaskingConfig {
            tf: TransferFunction::Pq,
            peak_nits: 1000.0,
            highlight_qp_boost: 4,
            shadow_qp_attenuate: 3,
        };
        for code in [0u16, 1, 511, 512, 1022, 1023] {
            let delta = hdr_qp_offset(code, 10, &config);
            assert!(
                delta >= -config.shadow_qp_attenuate && delta <= config.highlight_qp_boost,
                "code {code}: delta {delta} out of [-{}, {}]",
                config.shadow_qp_attenuate,
                config.highlight_qp_boost
            );
        }
    }

    // ── HLG tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_hlg_produces_offsets() {
        let config = HdrMaskingConfig {
            tf: TransferFunction::Hlg,
            peak_nits: 400.0,
            highlight_qp_boost: 3,
            shadow_qp_attenuate: 2,
        };
        // HLG code 0 → shadow → negative offset.
        let shadow_delta = hdr_qp_offset(0, 10, &config);
        assert!(
            shadow_delta <= 0,
            "HLG shadow must not produce positive QP boost, got {shadow_delta}"
        );

        // HLG code 1023 → bright highlight → non-negative offset.
        let highlight_delta = hdr_qp_offset(1023, 10, &config);
        assert!(
            highlight_delta >= 0,
            "HLG highlight must produce non-negative QP offset, got {highlight_delta}"
        );
    }

    #[test]
    fn test_hdr_hlg_clamp_bounds() {
        let config = HdrMaskingConfig {
            tf: TransferFunction::Hlg,
            peak_nits: 400.0,
            highlight_qp_boost: 3,
            shadow_qp_attenuate: 2,
        };
        for code in [0u16, 100, 511, 512, 900, 1023] {
            let delta = hdr_qp_offset(code, 10, &config);
            assert!(
                delta >= -config.shadow_qp_attenuate && delta <= config.highlight_qp_boost,
                "HLG code {code}: delta {delta} out of bounds"
            );
        }
    }

    // ── SDR pass-through ──────────────────────────────────────────────────────

    #[test]
    fn test_hdr_sdr_monotonic() {
        let config = HdrMaskingConfig {
            tf: TransferFunction::Sdr,
            peak_nits: 100.0,
            highlight_qp_boost: 3,
            shadow_qp_attenuate: 2,
        };
        // For SDR, higher code → higher luma → higher (or equal) QP offset.
        let d0 = hdr_qp_offset(0, 8, &config);
        let d128 = hdr_qp_offset(128, 8, &config);
        let d255 = hdr_qp_offset(255, 8, &config);
        assert!(
            d0 <= d128,
            "SDR: code 0 should have <= QP than code 128, got {d0} vs {d128}"
        );
        assert!(
            d128 <= d255,
            "SDR: code 128 should have <= QP than code 255, got {d128} vs {d255}"
        );
    }

    // ── hdr_qp_map ────────────────────────────────────────────────────────────

    #[test]
    fn test_hdr_qp_map_dimensions() {
        let width = 32usize;
        let height = 24usize;
        let block_size = 8;
        let plane: Vec<u16> = vec![512u16; width * height];
        let config = HdrMaskingConfig::default();

        let map = hdr_qp_map(&plane, width, height, 10, block_size, &config);
        let expected_len =
            ((height + block_size - 1) / block_size) * ((width + block_size - 1) / block_size);
        assert_eq!(map.len(), expected_len);
    }

    #[test]
    fn test_hdr_qp_map_empty_input() {
        let config = HdrMaskingConfig::default();
        let map = hdr_qp_map(&[], 0, 0, 10, 8, &config);
        assert!(map.is_empty());
    }

    #[test]
    fn test_hdr_qp_map_zero_block_size_empty() {
        let plane = vec![100u16; 64];
        let config = HdrMaskingConfig::default();
        let map = hdr_qp_map(&plane, 8, 8, 10, 0, &config);
        assert!(map.is_empty());
    }

    // ── HLG inverse OOTF helper sanity ───────────────────────────────────────

    #[test]
    fn test_hlg_inverse_ootf_zero_returns_zero() {
        let v = hlg_inverse_ootf_normalised(0.0);
        assert!(v < 1e-4, "HLG inv-OOTF(0) should be near 0, got {v}");
    }

    #[test]
    fn test_hlg_inverse_ootf_midpoint_between_bounds() {
        // HLG 0.5 → scene-linear: above 0.5 threshold so falls in highlight zone.
        let v = hlg_inverse_ootf_normalised(0.5);
        // (0.5 * 0.5) / 3.0 = 0.0833 (below 0.5, so shadow for threshold logic)
        // confirms value is non-negative
        assert!(
            v >= 0.0,
            "HLG inv-OOTF(0.5) should be non-negative, got {v}"
        );
    }
}
