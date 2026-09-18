//! VMAF feature computation: VIF, DLM/ADM, and motion.
//!
//! Netflix's VMAF is ML-trained, but the two primary feature groups —
//! **VIF** (Visual Information Fidelity) and **ADM/DLM** (Additive Distortion
//! Measure / Detail Loss Metric) — are deterministic signal-processing
//! computations that can be reproduced in pure Rust.
//!
//! ## VIF (Sheikh & Bovik, 2006)
//!
//! VIF measures the ratio of mutual information extractable from the distorted
//! image relative to the reference across four Gaussian scale-space levels.
//! Simplified formulation:
//!
//! ```text
//! vif_i = Σ_blocks  log2(1 + g² σ_r² / σ_n²)
//!       / Σ_blocks  log2(1 + σ_r² / σ_n²)
//! ```
//!
//! where `g` is the local gain (regression coefficient), `σ_r²` is reference
//! block variance, and `σ_n²` is viewer-noise variance.
//!
//! ## ADM/DLM (Li et al., 2011)
//!
//! ADM decomposes each scale into detail (high-frequency) and masking
//! (low-frequency) components using a 5-tap Laplacian of Gaussian (LoG)
//! approximation, then computes the ratio of reference-visible detail
//! surviving in the distorted image.
//!
//! ## Motion
//!
//! Inter-frame motion is estimated as the mean absolute difference (MAD)
//! between consecutive luma planes.  If no previous frame is supplied
//! `motion = 0`.
//!
//! ## References
//!
//! * Sheikh & Bovik, "Image Information and Visual Quality," IEEE TIP 2006.
//! * Li et al., "A Visual Saliency-Based Method for Automatic Cinematic
//!   Video Editing," 2011.
//! * Netflix VMAF open-source implementation, github.com/Netflix/vmaf.

/// All VMAF feature values for one frame pair.
///
/// Each `vif_scaleN` and `adm_scaleN` field is in `[0.0, 1.0]` where `1.0`
/// means no distortion at that scale.  `motion` is in `[0.0, 255.0]` (mean
/// absolute luma difference).
#[derive(Debug, Clone, PartialEq)]
pub struct VmafFeatures {
    /// VIF at full resolution (scale 0).
    pub vif_scale0: f32,
    /// VIF at 1/2 resolution (scale 1).
    pub vif_scale1: f32,
    /// VIF at 1/4 resolution (scale 2).
    pub vif_scale2: f32,
    /// VIF at 1/8 resolution (scale 3).
    pub vif_scale3: f32,
    /// ADM at full resolution (scale 0).
    pub adm_scale0: f32,
    /// ADM at 1/2 resolution (scale 1).
    pub adm_scale1: f32,
    /// ADM at 1/4 resolution (scale 2).
    pub adm_scale2: f32,
    /// ADM at 1/8 resolution (scale 3).
    pub adm_scale3: f32,
    /// Mean absolute inter-frame luma difference.
    pub motion: f32,
}

// ── Viewer noise variance (HVS model parameter from Sheikh & Bovik) ───────────
const SIGMA_NSQ: f32 = 2.0;
// ── 8×8 block size for local statistics ──────────────────────────────────────
const BLOCK: usize = 8;
// ── LoG 5-tap kernel for ADM detail extraction (σ = 1.0) ─────────────────────
const LOG5: [f32; 5] = [-0.0239, 0.2434, 0.5610, 0.2434, -0.0239];

// ── Public API ────────────────────────────────────────────────────────────────

/// Computes the full set of VMAF features for one frame pair.
///
/// # Parameters
///
/// * `ref_frame`  – packed f32 luma plane (row-major, `w × h` elements, range `[0, 255]`).
/// * `dist_frame` – same layout; must have the same length as `ref_frame`.
/// * `prev_ref`   – optional previous reference luma plane for motion estimation.
/// * `w`, `h`     – frame dimensions in pixels.
///
/// # Returns
///
/// A [`VmafFeatures`] struct with per-scale VIF, ADM, and motion values.
#[must_use]
pub fn compute_vmaf_features(
    ref_frame: &[f32],
    dist_frame: &[f32],
    prev_ref: Option<&[f32]>,
    w: u32,
    h: u32,
) -> VmafFeatures {
    let motion = match prev_ref {
        Some(prev) => mean_absolute_difference(prev, ref_frame),
        None => 0.0,
    };

    let (ref_f0, dist_f0) = (ref_frame, dist_frame);
    let (w0, h0) = (w as usize, h as usize);

    // Scale 0 — full resolution
    let vif0 = vif_spatial(ref_f0, dist_f0, w0, h0);
    let (adm0, _) = dlm_spatial(ref_f0, dist_f0, w0, h0);

    // Scale 1 — 1/2 resolution
    let (ref_f1, w1, h1) = downsample_f32(ref_f0, w0, h0);
    let (dist_f1, _, _) = downsample_f32(dist_f0, w0, h0);
    let vif1 = vif_spatial(&ref_f1, &dist_f1, w1, h1);
    let (adm1, _) = dlm_spatial(&ref_f1, &dist_f1, w1, h1);

    // Scale 2 — 1/4 resolution
    let (ref_f2, w2, h2) = downsample_f32(&ref_f1, w1, h1);
    let (dist_f2, _, _) = downsample_f32(&dist_f1, w1, h1);
    let vif2 = vif_spatial(&ref_f2, &dist_f2, w2, h2);
    let (adm2, _) = dlm_spatial(&ref_f2, &dist_f2, w2, h2);

    // Scale 3 — 1/8 resolution
    let (ref_f3, w3, h3) = downsample_f32(&ref_f2, w2, h2);
    let (dist_f3, _, _) = downsample_f32(&dist_f2, w2, h2);
    let vif3 = vif_spatial(&ref_f3, &dist_f3, w3, h3);
    let (adm3, _) = dlm_spatial(&ref_f3, &dist_f3, w3, h3);

    VmafFeatures {
        vif_scale0: vif0,
        vif_scale1: vif1,
        vif_scale2: vif2,
        vif_scale3: vif3,
        adm_scale0: adm0,
        adm_scale1: adm1,
        adm_scale2: adm2,
        adm_scale3: adm3,
        motion,
    }
}

// ── VIF (simplified Sheikh & Bovik formulation) ───────────────────────────────

/// Computes VIF on a single scale (f32 luma plane).
///
/// Uses non-overlapping `BLOCK×BLOCK` patches to estimate local statistics via
/// the Sheikh & Bovik (2006) information-theoretic formulation.  The score is
/// the ratio of visual information surviving in the distorted channel versus
/// the reference channel:
///
/// ```text
/// vif = Σ_b log(1 + g²σ_r² / σ_v²) / Σ_b log(1 + σ_r² / σ_n²)
/// ```
///
/// A variance-preservation bonus is blended in so that blurring (which
/// destroys local variance without shifting block means) also lowers the score.
///
/// Returns a value in `[0.0, 1.0]` (1.0 = reference quality preserved).
#[must_use]
pub fn vif_spatial(ref_frame: &[f32], dist_frame: &[f32], w: usize, h: usize) -> f32 {
    if ref_frame.is_empty() || w < BLOCK || h < BLOCK {
        return 1.0;
    }

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    // Variance preservation: ratio of dist block variance to ref block variance.
    let mut var_ratio_sum = 0.0f64;
    let mut block_count = 0usize;

    let blocks_y = h / BLOCK;
    let blocks_x = w / BLOCK;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let (ref_mean, ref_var) = block_stats(ref_frame, bx, by, w, BLOCK);
            let (dist_mean, dist_var_raw) = block_stats(dist_frame, bx, by, w, BLOCK);

            // Local gain g = Cov(ref, dist) / Var(ref)
            let cov =
                block_covariance(ref_frame, dist_frame, bx, by, w, BLOCK, ref_mean, dist_mean);
            let sigma_r2 = ref_var.max(1e-6) as f64;
            let sigma_n2 = SIGMA_NSQ as f64;

            let g = if sigma_r2 > sigma_n2 {
                (cov as f64) / sigma_r2
            } else {
                1.0
            };

            // Effective noise variance in distorted channel.
            let sigma_d2 = (dist_var_raw as f64).max(0.0);
            let sv2 = (sigma_d2 - g * g * sigma_r2).max(0.0) + sigma_n2;

            // Log-ratio contributions (information theoretic).
            if sigma_r2 > sigma_n2 {
                let num_i = (1.0 + g * g * sigma_r2 / sv2).ln();
                let den_i = (1.0 + sigma_r2 / sigma_n2).ln();
                num += num_i;
                den += den_i;
            }

            // Variance preservation ratio: compares distorted variance to reference.
            // When both are near-zero (both uniform), score = 1 (no distortion).
            // When ref is high-variance but dist is low-variance (blur), score → 0.
            let vp = if sigma_r2 < 1.0 {
                // Both near-zero variance → indistinguishable / no content to compare.
                1.0f64
            } else {
                (sigma_d2 / sigma_r2).min(1.0)
            };
            var_ratio_sum += vp;
            block_count += 1;
        }
    }

    // Variance preservation score across all blocks (captures blur).
    let var_preservation = if block_count > 0 {
        (var_ratio_sum / block_count as f64) as f32
    } else {
        1.0
    };

    if den < 1e-10 {
        // All blocks are below the noise threshold (uniform / flat content).
        // In this regime the information-theoretic path gives no signal, so
        // rely entirely on the variance-preservation score, which is 1.0 when
        // both ref and dist are uniformly flat (identical) and 0.0 when ref
        // has structure that the distorted frame lacks.
        return var_preservation;
    }

    let vif_ac = ((num / den) as f32).clamp(0.0, 1.0);

    // Final VIF: weight AC-information fidelity and variance preservation.
    // The 0.70/0.30 split ensures that either pure blurring or pixel-noise
    // distortion is reflected in the score.
    (0.70 * vif_ac + 0.30 * var_preservation).clamp(0.0, 1.0)
}

/// Computes DLM/ADM on a single scale.
///
/// Returns `(detail_loss_measure, additive_impairment)` in `[0.0, 1.0]`.
///
/// ADM measures the fraction of visible reference detail preserved in the
/// distorted image.  A value of 1.0 means all detail is preserved; 0.0 means
/// all detail has been suppressed or replaced by additive noise.
#[must_use]
pub fn dlm_spatial(ref_frame: &[f32], dist_frame: &[f32], w: usize, h: usize) -> (f32, f32) {
    if ref_frame.is_empty() || w < 5 || h < 5 {
        return (1.0, 0.0);
    }

    // Extract detail bands via LoG filter.
    let ref_detail = log_filter(ref_frame, w, h);
    let dist_detail = log_filter(dist_frame, w, h);

    let len = ref_detail.len();
    if len == 0 {
        return (1.0, 0.0);
    }

    let mut ref_energy = 0.0f64;
    let mut preserved = 0.0f64;
    let mut additive = 0.0f64;

    for i in 0..len {
        let r = ref_detail[i] as f64;
        let d = dist_detail[i] as f64;
        ref_energy += r.abs();

        // ADM: how much reference detail is preserved?
        let pres = r.abs().min(d.abs());
        preserved += pres;

        // Additive impairment: noise added by distortion.
        let add = (d.abs() - r.abs()).max(0.0);
        additive += add;
    }

    let adm = if ref_energy < 1e-8 {
        1.0
    } else {
        ((preserved / ref_energy) as f32).clamp(0.0, 1.0)
    };

    let total_signal = ref_energy + 1e-8;
    let imp = ((additive / total_signal) as f32).clamp(0.0, 1.0);

    (adm, imp)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Computes mean and variance for a `BLOCK×BLOCK` patch at block position `(bx, by)`.
fn block_stats(plane: &[f32], bx: usize, by: usize, stride: usize, block: usize) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0usize;

    for dy in 0..block {
        let y = by * block + dy;
        for dx in 0..block {
            let x = bx * block + dx;
            let idx = y * stride + x;
            if idx < plane.len() {
                let v = plane[idx] as f64;
                sum += v;
                sum2 += v * v;
                n += 1;
            }
        }
    }

    if n == 0 {
        return (0.0, 0.0);
    }

    let mean = (sum / n as f64) as f32;
    let var = ((sum2 / n as f64) - (sum / n as f64).powi(2)).max(0.0) as f32;
    (mean, var)
}

/// Computes covariance for a block pair.
fn block_covariance(
    ref_plane: &[f32],
    dist_plane: &[f32],
    bx: usize,
    by: usize,
    stride: usize,
    block: usize,
    ref_mean: f32,
    dist_mean: f32,
) -> f32 {
    let mut cov = 0.0f64;
    let mut n = 0usize;

    for dy in 0..block {
        let y = by * block + dy;
        for dx in 0..block {
            let x = bx * block + dx;
            let idx = y * stride + x;
            if idx < ref_plane.len() && idx < dist_plane.len() {
                let r = ref_plane[idx] as f64 - ref_mean as f64;
                let d = dist_plane[idx] as f64 - dist_mean as f64;
                cov += r * d;
                n += 1;
            }
        }
    }

    if n == 0 {
        0.0
    } else {
        (cov / n as f64) as f32
    }
}

/// Separable 5-tap LoG filter (σ ≈ 1.0) for detail extraction.
fn log_filter(plane: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w < 5 || h < 5 {
        return Vec::new();
    }

    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 2..w - 2 {
            let mut val = 0.0f32;
            for (k, &coef) in LOG5.iter().enumerate() {
                val += coef * plane[y * w + (x + k).wrapping_sub(2).min(w - 1)];
            }
            tmp[y * w + x] = val;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; w * h];
    for y in 2..h - 2 {
        for x in 0..w {
            let mut val = 0.0f32;
            for (k, &coef) in LOG5.iter().enumerate() {
                val += coef * tmp[((y + k).wrapping_sub(2).min(h - 1)) * w + x];
            }
            out[y * w + x] = val;
        }
    }

    out
}

/// Box-average 2× downsampler for f32 planes.
fn downsample_f32(plane: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut out = vec![0.0f32; nw * nh];

    for y in 0..nh {
        for x in 0..nw {
            let sy = y * 2;
            let sx = x * 2;
            let i00 = sy * w + sx;
            let i01 = sy * w + (sx + 1).min(w - 1);
            let i10 = (sy + 1).min(h - 1) * w + sx;
            let i11 = (sy + 1).min(h - 1) * w + (sx + 1).min(w - 1);
            out[y * nw + x] = (plane[i00] + plane[i01] + plane[i10] + plane[i11]) * 0.25;
        }
    }

    (out, nw, nh)
}

/// Mean absolute difference between two planes (for motion estimation).
fn mean_absolute_difference(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .sum();
    sum / n as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_f32(w: usize, h: usize, v: f32) -> Vec<f32> {
        vec![v; w * h]
    }

    /// High-frequency test image: pixel-level alternating bright/dark pattern.
    ///
    /// Uses a 3-pixel period (not 2) so that `sin(2π/3 * x)` is non-zero and
    /// produces strong local variance in every 8×8 block.  Blurring this
    /// pattern with a 3×3 box filter severely attenuates the signal because the
    /// filter spans exactly one period.
    fn checkerboard_f32(w: usize, h: usize) -> Vec<f32> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    // 3-pixel period sinusoid: values cycle approx 128±100
                    let phase = (x + y * 3) % 3;
                    match phase {
                        0 => 230.0_f32,
                        1 => 60.0_f32,
                        _ => 128.0_f32,
                    }
                })
            })
            .collect()
    }

    fn blur_frame(src: &[f32], w: usize, h: usize) -> Vec<f32> {
        // Simple 3×3 box blur repeated 4× for a heavily blurred result.
        let mut cur = src.to_vec();
        for _ in 0..4 {
            let mut next = cur.clone();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let sum = cur[(y - 1) * w + (x - 1)]
                        + cur[(y - 1) * w + x]
                        + cur[(y - 1) * w + (x + 1)]
                        + cur[y * w + (x - 1)]
                        + cur[y * w + x]
                        + cur[y * w + (x + 1)]
                        + cur[(y + 1) * w + (x - 1)]
                        + cur[(y + 1) * w + x]
                        + cur[(y + 1) * w + (x + 1)];
                    next[y * w + x] = sum / 9.0;
                }
            }
            cur = next;
        }
        cur
    }

    // ── Task 3 tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_vmaf_features_identical() {
        let frame = checkerboard_f32(64, 64);
        let feats = compute_vmaf_features(&frame, &frame, None, 64, 64);

        // VIF for identical frames should be ≈ 1.0
        assert!(
            feats.vif_scale0 >= 0.9,
            "vif_scale0 = {} (expected ≈ 1.0)",
            feats.vif_scale0
        );
        assert!(feats.vif_scale1 >= 0.9, "vif_scale1 = {}", feats.vif_scale1);

        // ADM for identical frames should be ≈ 1.0
        assert!(
            feats.adm_scale0 >= 0.9,
            "adm_scale0 = {} (expected ≈ 1.0)",
            feats.adm_scale0
        );
        assert!(feats.adm_scale1 >= 0.9, "adm_scale1 = {}", feats.adm_scale1);

        // Motion with no prev_ref must be 0
        assert!(
            feats.motion.abs() < 1e-4,
            "motion without prev_ref must be 0, got {}",
            feats.motion
        );
    }

    #[test]
    fn test_vmaf_features_degraded() {
        // Use a larger frame (128×128) so that all 4 scales (64→32→16→8)
        // still contain enough blocks for meaningful VIF measurement.
        let ref_frame = checkerboard_f32(128, 128);
        let dist_frame = blur_frame(&ref_frame, 128, 128);

        let feats = compute_vmaf_features(&ref_frame, &dist_frame, None, 128, 128);

        // At scales with enough resolution to compute VIF, blurring must
        // degrade the score below 0.8.
        assert!(
            feats.vif_scale0 < 0.8,
            "vif_scale0 = {} (should be < 0.8 for blurred frame)",
            feats.vif_scale0
        );
        assert!(
            feats.vif_scale1 < 0.8,
            "vif_scale1 = {} (should be < 0.8 for blurred frame)",
            feats.vif_scale1
        );
    }

    #[test]
    fn test_vmaf_features_motion() {
        let ref_frame = flat_f32(64, 64, 128.0);
        let prev_ref = flat_f32(64, 64, 100.0); // different from ref_frame

        let feats = compute_vmaf_features(&ref_frame, &ref_frame, Some(&prev_ref), 64, 64);

        assert!(
            feats.motion > 0.0,
            "motion must be > 0 when prev_ref differs from ref_frame, got {}",
            feats.motion
        );
        assert!(
            (feats.motion - 28.0).abs() < 1.0,
            "expected MAD ≈ 28, got {}",
            feats.motion
        );
    }

    #[test]
    fn test_vif_spatial_identical() {
        let frame = checkerboard_f32(32, 32);
        let vif = vif_spatial(&frame, &frame, 32, 32);
        assert!(vif >= 0.9, "identical frames → VIF ≈ 1.0, got {vif}");
    }

    #[test]
    fn test_dlm_spatial_identical() {
        let frame = checkerboard_f32(32, 32);
        let (adm, imp) = dlm_spatial(&frame, &frame, 32, 32);
        assert!(adm >= 0.9, "identical → ADM ≈ 1.0, got {adm}");
        assert!(imp < 0.1, "identical → additive_imp ≈ 0, got {imp}");
    }

    #[test]
    fn test_vmaf_features_all_scales_range() {
        let ref_frame = checkerboard_f32(128, 128);
        let dist_frame = blur_frame(&ref_frame, 128, 128);
        let feats = compute_vmaf_features(&ref_frame, &dist_frame, None, 128, 128);

        for (name, val) in [
            ("vif0", feats.vif_scale0),
            ("vif1", feats.vif_scale1),
            ("vif2", feats.vif_scale2),
            ("vif3", feats.vif_scale3),
            ("adm0", feats.adm_scale0),
            ("adm1", feats.adm_scale1),
            ("adm2", feats.adm_scale2),
            ("adm3", feats.adm_scale3),
        ] {
            assert!(
                (0.0..=1.0).contains(&val),
                "{name} out of [0,1] range: {val}"
            );
        }
    }
}
