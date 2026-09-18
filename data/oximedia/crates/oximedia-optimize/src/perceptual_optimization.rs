//! Perceptual encoding optimization.
//!
//! Applies Human Visual System (HVS) models to guide quantization decisions:
//! Just-Noticeable-Difference (JND) maps, texture classification, and
//! psychovisual QP adjustment.

#![allow(dead_code)]

use crate::bitrate_controller::FrameType;
use crate::hdr_masking::{HdrMaskingConfig, TransferFunction};

/// Perceptual weighting parameters for the HVS model.
#[derive(Debug, Clone)]
pub struct PerceptualWeight {
    /// Sensitivity to luma distortion (higher → penalise luma errors more).
    pub luma_sensitivity: f32,
    /// Sensitivity to chroma distortion.
    pub chroma_sensitivity: f32,
    /// QP boost applied to edge regions (edges need fidelity).
    pub edge_boost: f32,
    /// QP relaxation applied to textured regions (textures mask distortion).
    pub texture_reduction: f32,
}

impl PerceptualWeight {
    /// Default Human Visual System model weights.
    ///
    /// Values are calibrated to typical broadcast / streaming use-cases.
    #[must_use]
    pub fn default_hvs() -> Self {
        Self {
            luma_sensitivity: 1.0,
            chroma_sensitivity: 0.5,
            edge_boost: 0.8,
            texture_reduction: 1.2,
        }
    }
}

impl Default for PerceptualWeight {
    fn default() -> Self {
        Self::default_hvs()
    }
}

/// Just-Noticeable Difference map computation.
///
/// Returns a per-pixel visibility threshold: higher values mean more distortion
/// can be tolerated (the HVS will not notice).
pub struct JndMap;

impl JndMap {
    /// Compute JND for every pixel of a luma frame.
    ///
    /// High-frequency / textured areas receive a high JND (tolerant).
    /// Smooth / flat areas receive a low JND (sensitive).
    #[must_use]
    pub fn compute(frame: &[f32], width: u32, height: u32) -> Vec<f32> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        if n == 0 || frame.len() < n {
            return vec![0.0; n];
        }

        let mut jnd = vec![0.0f32; n];

        // Base JND from local luminance masking (Weber law approximation)
        for y in 0..h {
            for x in 0..w {
                let lum = frame[y * w + x].clamp(0.0, 255.0);
                // Weber model: JND ≈ max(17 - 0.05*lum, 3) for dark;
                //             JND ≈ 3 + 0.025*(lum-128) for bright
                let base_jnd = if lum < 60.0 {
                    (17.0 - 0.05 * lum).max(3.0)
                } else {
                    3.0 + 0.025 * (lum - 128.0).max(0.0)
                };
                jnd[y * w + x] = base_jnd;
            }
        }

        // Add texture masking: high local variance → raise JND
        let bs = 8usize;
        if w >= bs && h >= bs {
            let blocks_x = w / bs;
            let blocks_y = h / bs;

            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    let n_pix = (bs * bs) as f32;

                    for row in 0..bs {
                        for col in 0..bs {
                            let v = frame[(by * bs + row) * w + bx * bs + col];
                            sum += v;
                            sum_sq += v * v;
                        }
                    }
                    let mean = sum / n_pix;
                    let variance = (sum_sq / n_pix - mean * mean).max(0.0);

                    // Texture boost: up to +20 JND units for very textured blocks
                    let texture_bonus = (variance.sqrt() / 255.0 * 20.0).min(20.0);

                    for row in 0..bs {
                        for col in 0..bs {
                            let idx = (by * bs + row) * w + bx * bs + col;
                            jnd[idx] += texture_bonus;
                        }
                    }
                }
            }
        }

        jnd
    }
}

/// Psychovisual QP adjuster.
///
/// Adjusts a base QP value up or down according to the local JND threshold
/// and the frame type.
pub struct PsychovisualQp;

impl PsychovisualQp {
    /// Adjust `base_qp` for a pixel/block with a given `jnd` value.
    ///
    /// - High JND (textured, masked) → raise QP (less bits needed).
    /// - Low JND (smooth, flat) → lower QP (need higher fidelity).
    /// - B frames get a mild additional QP raise.
    #[must_use]
    pub fn adjust(base_qp: u8, jnd: f32, frame_type: FrameType) -> u8 {
        // Convert JND to a QP delta: JND range roughly 3..40, map to -4..+4
        let normalised = ((jnd - 3.0) / 37.0).clamp(0.0, 1.0);
        let delta = (normalised * 8.0 - 4.0) as i8; // -4 to +4

        let type_offset: i8 = match frame_type {
            FrameType::I => -1, // I frames: slightly lower QP (higher quality)
            FrameType::P => 0,
            FrameType::B => 2, // B frames: raise QP a bit
        };

        let adjusted = base_qp as i16 + delta as i16 + type_offset as i16;
        adjusted.clamp(0, 51) as u8
    }

    /// Adjust `base_qp` incorporating both HVS psychovisual masking and HDR
    /// transfer-function-aware luminance masking.
    ///
    /// This composes two independent QP deltas:
    /// 1. The standard psychovisual delta from `PsychovisualQp::adjust`.
    /// 2. An HDR luminance masking delta from `hdr_masking::hdr_qp_offset`.
    ///
    /// The combined delta is clamped to a valid QP range `[0, 51]`.
    ///
    /// # Parameters
    ///
    /// - `base_qp`    — baseline QP value (0–51).
    /// - `jnd`        — local JND threshold from `JndMap::compute`.
    /// - `frame_type` — I/P/B frame type.
    /// - `luma_code`  — raw luma code word for the block (10-bit = 0–1023).
    /// - `bit_depth`  — bit depth of the luma signal (typically 10 for HDR).
    /// - `tf`         — transfer function (Sdr / Pq / Hlg).
    ///
    /// Returns the adjusted QP value clamped to `[0, 51]`.
    #[must_use]
    pub fn compute_hdr_qp(
        base_qp: u8,
        jnd: f32,
        frame_type: FrameType,
        luma_code: u16,
        bit_depth: u8,
        tf: TransferFunction,
    ) -> u8 {
        // Step 1: apply standard psychovisual masking.
        let psycho_qp = Self::adjust(base_qp, jnd, frame_type) as i16;

        // Step 2: compute HDR luminance masking delta.
        let hdr_config = HdrMaskingConfig {
            tf,
            ..HdrMaskingConfig::default()
        };
        let hdr_delta = crate::hdr_masking::hdr_qp_offset(luma_code, bit_depth, &hdr_config);

        // Step 3: compose and clamp.
        (psycho_qp + hdr_delta as i16).clamp(0, 51) as u8
    }
}

/// Texture classification for 8×8 blocks.
pub struct TextureClassifier;

impl TextureClassifier {
    /// Classify a flat 64-element 8×8 block of f32 luma samples.
    #[must_use]
    pub fn classify(block: &[f32; 64]) -> TextureType {
        let mean = block.iter().sum::<f32>() / 64.0;
        let variance = block.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 64.0;
        let std_dev = variance.sqrt();

        // Gradient statistics across the block
        let (max_grad, avg_grad) = Self::gradient_stats(block);

        if std_dev < 2.0 {
            TextureType::Flat
        } else if max_grad > 40.0 && avg_grad < max_grad * 0.5 && std_dev < 120.0 {
            // Strong localised gradient with low average gradient → edge structure.
            // A checkerboard has uniformly high gradients everywhere, so avg ≈ max.
            TextureType::Edge
        } else if std_dev > 80.0 {
            TextureType::Complex
        } else if std_dev > 20.0 {
            TextureType::HighFreq
        } else {
            TextureType::Texture
        }
    }

    /// Returns (max_gradient, average_gradient) over the 7x7 inner grid.
    fn gradient_stats(block: &[f32; 64]) -> (f32, f32) {
        let mut max_g = 0.0f32;
        let mut sum_g = 0.0f32;
        let count = (7 * 7) as f32;
        for row in 0..7usize {
            for col in 0..7usize {
                let gx = (block[row * 8 + col + 1] - block[row * 8 + col]).abs();
                let gy = (block[(row + 1) * 8 + col] - block[row * 8 + col]).abs();
                let g = (gx * gx + gy * gy).sqrt();
                sum_g += g;
                if g > max_g {
                    max_g = g;
                }
            }
        }
        (max_g, sum_g / count)
    }
}

/// Texture type classification for an 8×8 block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    /// Flat, smooth region (very low variance).
    Flat,
    /// Dominant edge structure.
    Edge,
    /// Mid-frequency texture.
    Texture,
    /// High-frequency noise or detail.
    HighFreq,
    /// Complex mixed content.
    Complex,
}

impl TextureType {
    /// QP delta suggested by this texture type.
    ///
    /// - Flat: lower QP (need fidelity, banding is visible).
    /// - Edge: no change (already handled by edge_boost weight).
    /// - Texture: raise QP slightly (masking).
    /// - HighFreq: raise QP more (heavily masked).
    /// - Complex: raise QP a little.
    #[must_use]
    pub fn qp_delta(&self) -> i8 {
        match self {
            Self::Flat => -2,
            Self::Edge => 0,
            Self::Texture => 2,
            Self::HighFreq => 4,
            Self::Complex => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptual_weight_default_hvs() {
        let w = PerceptualWeight::default_hvs();
        assert!((w.luma_sensitivity - 1.0).abs() < f32::EPSILON);
        assert!(w.chroma_sensitivity < w.luma_sensitivity);
        assert!(w.edge_boost > 0.0);
        assert!(w.texture_reduction > 1.0);
    }

    #[test]
    fn test_jnd_map_length() {
        let frame: Vec<f32> = vec![128.0; 64 * 64];
        let jnd = JndMap::compute(&frame, 64, 64);
        assert_eq!(jnd.len(), 64 * 64);
    }

    #[test]
    fn test_jnd_map_flat_frame_low_values() {
        // Flat grey frame → only base JND, no texture bonus
        let frame: Vec<f32> = vec![128.0; 32 * 32];
        let jnd = JndMap::compute(&frame, 32, 32);
        // All values should be relatively small (no texture bonus > 0)
        for &v in &jnd {
            assert!(v >= 0.0 && v < 25.0, "unexpected jnd={v}");
        }
    }

    #[test]
    fn test_jnd_map_textured_frame_higher() {
        // Alternating 0/255 pattern → high variance → texture bonus
        let frame: Vec<f32> = (0..32 * 32)
            .map(|i| if i % 2 == 0 { 0.0 } else { 255.0 })
            .collect();
        let flat = vec![128.0f32; 32 * 32];
        let jnd_tex = JndMap::compute(&frame, 32, 32);
        let jnd_flat = JndMap::compute(&flat, 32, 32);
        let avg_tex: f32 = jnd_tex.iter().sum::<f32>() / jnd_tex.len() as f32;
        let avg_flat: f32 = jnd_flat.iter().sum::<f32>() / jnd_flat.len() as f32;
        assert!(
            avg_tex > avg_flat,
            "textured avg={avg_tex}, flat avg={avg_flat}"
        );
    }

    #[test]
    fn test_psychovisual_qp_high_jnd_raises_qp() {
        let base = 26u8;
        let low_jnd = PsychovisualQp::adjust(base, 3.0, FrameType::P);
        let high_jnd = PsychovisualQp::adjust(base, 40.0, FrameType::P);
        assert!(
            high_jnd > low_jnd,
            "high_jnd qp={high_jnd}, low_jnd qp={low_jnd}"
        );
    }

    #[test]
    fn test_psychovisual_qp_b_frame_higher_than_i() {
        let i_qp = PsychovisualQp::adjust(26, 20.0, FrameType::I);
        let b_qp = PsychovisualQp::adjust(26, 20.0, FrameType::B);
        assert!(b_qp > i_qp);
    }

    #[test]
    fn test_psychovisual_qp_clamped() {
        let qp_max = PsychovisualQp::adjust(51, 40.0, FrameType::B);
        assert!(qp_max <= 51);
        let qp_min = PsychovisualQp::adjust(0, 3.0, FrameType::I);
        assert_eq!(qp_min, 0);
    }

    #[test]
    fn test_texture_classifier_flat() {
        let block = [128.0f32; 64];
        assert_eq!(TextureClassifier::classify(&block), TextureType::Flat);
    }

    #[test]
    fn test_texture_classifier_edge() {
        let mut block = [0.0f32; 64];
        // Left half dark, right half bright → strong edge
        for row in 0..8 {
            for col in 4..8 {
                block[row * 8 + col] = 200.0;
            }
        }
        let t = TextureClassifier::classify(&block);
        assert!(t == TextureType::Edge || t == TextureType::HighFreq);
    }

    #[test]
    fn test_texture_classifier_high_freq() {
        // Checkerboard at max amplitude → very high std_dev
        let block: [f32; 64] =
            std::array::from_fn(|i| if (i / 8 + i % 8) % 2 == 0 { 0.0 } else { 255.0 });
        let t = TextureClassifier::classify(&block);
        assert!(t == TextureType::Complex || t == TextureType::HighFreq);
    }

    #[test]
    fn test_texture_type_qp_delta_flat_negative() {
        assert!(TextureType::Flat.qp_delta() < 0);
    }

    #[test]
    fn test_texture_type_qp_delta_high_freq_positive() {
        assert!(TextureType::HighFreq.qp_delta() > 0);
    }

    #[test]
    fn test_jnd_empty_frame() {
        let jnd = JndMap::compute(&[], 0, 0);
        assert!(jnd.is_empty());
    }

    // ── compute_hdr_qp tests ──────────────────────────────────────────────────

    #[test]
    fn test_compute_hdr_qp_pq_highlight_raises_qp() {
        // At full-brightness PQ code, HDR masking should push QP up relative to base.
        let base_qp = 26u8;
        let jnd_mid = 20.0f32;
        let qp_shadow = PsychovisualQp::compute_hdr_qp(
            base_qp,
            jnd_mid,
            FrameType::P,
            0, // code 0 → deepest shadow
            10,
            TransferFunction::Pq,
        );
        let qp_highlight = PsychovisualQp::compute_hdr_qp(
            base_qp,
            jnd_mid,
            FrameType::P,
            1023, // code 1023 → full brightness
            10,
            TransferFunction::Pq,
        );
        assert!(
            qp_highlight > qp_shadow,
            "PQ highlight QP ({qp_highlight}) should exceed shadow QP ({qp_shadow})"
        );
    }

    #[test]
    fn test_compute_hdr_qp_clamped_to_valid_range() {
        // Result must always be in [0, 51].
        for &luma_code in &[0u16, 511, 512, 1023] {
            for &tf in &[
                TransferFunction::Pq,
                TransferFunction::Hlg,
                TransferFunction::Sdr,
            ] {
                let qp = PsychovisualQp::compute_hdr_qp(26, 15.0, FrameType::P, luma_code, 10, tf);
                assert!(qp <= 51, "QP {qp} exceeds 51 for luma_code={luma_code}");
            }
        }
    }

    #[test]
    fn test_compute_hdr_qp_sdr_equals_adjust() {
        // For SDR at a mid-code (127/255 ≈ 0.5), HDR delta ≈ 0, so result ≈ adjust().
        let base_qp = 26u8;
        let jnd = 20.0f32;
        let sdr_qp = PsychovisualQp::compute_hdr_qp(
            base_qp,
            jnd,
            FrameType::P,
            127, // ≈ midpoint for 8-bit
            8,
            TransferFunction::Sdr,
        );
        let plain_qp = PsychovisualQp::adjust(base_qp, jnd, FrameType::P);
        // Should be close (within ±1) since HDR delta at mid-code is small.
        let diff = (sdr_qp as i16 - plain_qp as i16).abs();
        assert!(
            diff <= 1,
            "SDR compute_hdr_qp ({sdr_qp}) should be near adjust ({plain_qp})"
        );
    }
}
