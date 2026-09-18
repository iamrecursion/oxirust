//! Subsampled SSIM for fast quality assessment on large frames.
//!
//! Full-resolution SSIM on 4K content (3840×2160) processes ~8 million
//! windows.  By downsampling 2× or 4× before computing SSIM, throughput
//! improves ≈4× or ≈16× respectively with a typical accuracy loss of < 5%
//! in absolute SSIM.
//!
//! ## Algorithm
//!
//! 1. Box-average downsample the luma planes by `subsample` factor.
//! 2. Run the standard 11×11 Gaussian-windowed SSIM on the smaller planes.
//!
//! ## Reference
//!
//! Z. Wang et al., "Image Quality Assessment: From Error Visibility to
//! Structural Similarity," IEEE TIP 2004.

use rayon::prelude::*;

/// Configuration for subsampled SSIM.
#[derive(Debug, Clone)]
pub struct SsimConfig {
    /// Gaussian window radius.  Full window = `window_size × window_size`.
    /// Default: 11.
    pub window_size: u32,
    /// Spatial subsampling factor.
    ///   * `1` → full resolution (no downsampling)
    ///   * `2` → half-resolution  (≈4× speedup)
    ///   * `4` → quarter-resolution (≈16× speedup)
    ///
    /// Values outside {1, 2, 4} are clamped to the nearest supported factor.
    /// Default: 1.
    pub subsample: u32,
    /// SSIM stability constant K1 for luminance.  Default: 0.01.
    pub k1: f32,
    /// SSIM stability constant K2 for contrast.  Default: 0.03.
    pub k2: f32,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window_size: 11,
            subsample: 1,
            k1: 0.01,
            k2: 0.03,
        }
    }
}

/// Computes SSIM between `ref_frame` and `dist_frame`, optionally downsampling
/// first for speed.
///
/// # Parameters
///
/// * `ref_frame`  – packed 8-bit luma data, row-major, `w × h` bytes.
/// * `dist_frame` – same layout and dimensions as `ref_frame`.
/// * `w`, `h`     – frame dimensions in pixels.
/// * `cfg`        – SSIM and subsampling configuration.
///
/// # Returns
///
/// SSIM in `[0.0, 1.0]` (higher → better quality).  Returns `1.0` when both
/// planes are empty.
#[must_use]
pub fn ssim_subsampled(
    ref_frame: &[u8],
    dist_frame: &[u8],
    w: u32,
    h: u32,
    cfg: &SsimConfig,
) -> f32 {
    let factor = effective_factor(cfg.subsample);

    if factor == 1 {
        compute_ssim_plane(ref_frame, dist_frame, w as usize, h as usize, cfg)
    } else {
        let sw = (w / factor).max(1);
        let sh = (h / factor).max(1);
        let ref_ds = box_downsample(ref_frame, w as usize, h as usize, factor as usize);
        let dist_ds = box_downsample(dist_frame, w as usize, h as usize, factor as usize);
        compute_ssim_plane(&ref_ds, &dist_ds, sw as usize, sh as usize, cfg)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Clamps subsample factor to {1, 2, 4}.
fn effective_factor(requested: u32) -> u32 {
    if requested >= 4 {
        4
    } else if requested >= 2 {
        2
    } else {
        1
    }
}

/// Box-average downsample of a packed 8-bit plane.
///
/// Output size is `(w/factor) × (h/factor)`.  Each output pixel is the
/// mean of a `factor×factor` block of input pixels.
fn box_downsample(plane: &[u8], w: usize, h: usize, factor: usize) -> Vec<u8> {
    let out_w = w / factor;
    let out_h = h / factor;
    let mut out = vec![0u8; out_w * out_h];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum: u32 = 0;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sy = oy * factor + dy;
                    let sx = ox * factor + dx;
                    sum += u32::from(plane[sy * w + sx]);
                }
            }
            let count = (factor * factor) as u32;
            out[oy * out_w + ox] = (sum / count) as u8;
        }
    }

    out
}

/// Full SSIM computation on already-sized planes.
///
/// Uses a `window_size × window_size` Gaussian window and the standard SSIM
/// formula (Wang et al., 2004).  Parallelised row-wise via rayon.
fn compute_ssim_plane(
    ref_plane: &[u8],
    dist_plane: &[u8],
    width: usize,
    height: usize,
    cfg: &SsimConfig,
) -> f32 {
    if ref_plane.is_empty() || dist_plane.is_empty() {
        return 1.0;
    }

    let ws = cfg.window_size as usize;
    let half = ws / 2;

    // Luminance range for 8-bit (255).
    let l: f32 = 255.0;
    let c1 = (cfg.k1 * l) * (cfg.k1 * l);
    let c2 = (cfg.k2 * l) * (cfg.k2 * l);

    let window = create_gaussian_window_f32(ws);

    if height <= 2 * half || width <= 2 * half {
        // Frame too small for even one full window — return perfect score.
        return 1.0;
    }

    let row_start = half;
    let row_end = height - half;

    let ssim_sum: f32 = (row_start..row_end)
        .into_par_iter()
        .map(|cy| {
            let mut row_sum = 0.0f32;
            for cx in half..width - half {
                row_sum += ssim_at(
                    ref_plane, dist_plane, cx, cy, width, ws, half, &window, c1, c2,
                );
            }
            row_sum
        })
        .sum();

    let count = ((row_end - row_start) * (width - 2 * half)) as f32;
    if count <= 0.0 {
        1.0
    } else {
        ssim_sum / count
    }
}

/// Gaussian window (sigma = 1.5), normalised to sum to 1.
fn create_gaussian_window_f32(size: usize) -> Vec<f32> {
    let sigma = 1.5_f32;
    let center = (size - 1) as f32 / 2.0;
    let mut window = Vec::with_capacity(size * size);
    let mut sum = 0.0f32;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let v = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            window.push(v);
            sum += v;
        }
    }
    for v in &mut window {
        *v /= sum;
    }
    window
}

/// SSIM value for one window centred at `(cx, cy)`.
#[allow(clippy::too_many_arguments)]
fn ssim_at(
    ref_plane: &[u8],
    dist_plane: &[u8],
    cx: usize,
    cy: usize,
    stride: usize,
    ws: usize,
    half: usize,
    window: &[f32],
    c1: f32,
    c2: f32,
) -> f32 {
    let mut sum_r = 0.0f32;
    let mut sum_d = 0.0f32;
    let mut sum_rr = 0.0f32;
    let mut sum_dd = 0.0f32;
    let mut sum_rd = 0.0f32;

    for dy in 0..ws {
        let y = cy - half + dy;
        for dx in 0..ws {
            let x = cx - half + dx;
            let w = window[dy * ws + dx];
            let r = ref_plane[y * stride + x] as f32;
            let d = dist_plane[y * stride + x] as f32;
            sum_r += w * r;
            sum_d += w * d;
            sum_rr += w * r * r;
            sum_dd += w * d * d;
            sum_rd += w * r * d;
        }
    }

    let mu_r = sum_r;
    let mu_d = sum_d;
    let sigma_rr = sum_rr - mu_r * mu_r;
    let sigma_dd = sum_dd - mu_d * mu_d;
    let sigma_rd = sum_rd - mu_r * mu_d;

    let num = (2.0 * mu_r * mu_d + c1) * (2.0 * sigma_rd + c2);
    let den = (mu_r * mu_r + mu_d * mu_d + c1) * (sigma_rr + sigma_dd + c2);

    num / den
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, value: u8) -> Vec<u8> {
        vec![value; w * h]
    }

    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| if (x + y) % 2 == 0 { 200u8 } else { 50u8 }))
            .collect()
    }

    #[test]
    fn test_ssim_subsampled_identical() {
        let cfg = SsimConfig {
            subsample: 2,
            ..Default::default()
        };
        let frame = make_frame(256, 256, 128);
        let result = ssim_subsampled(&frame, &frame, 256, 256, &cfg);
        assert!(
            (result - 1.0).abs() < 1e-4,
            "identical frames → SSIM ≈ 1.0, got {result}"
        );
    }

    #[test]
    fn test_ssim_subsampled_vs_full_correlation() {
        let cfg_full = SsimConfig {
            subsample: 1,
            ..Default::default()
        };
        let cfg_sub = SsimConfig {
            subsample: 2,
            ..Default::default()
        };

        let ref_frame = checkerboard(256, 256);
        // Distorted: add a mild uniform offset.
        let dist_frame: Vec<u8> = ref_frame.iter().map(|&v| v.saturating_add(20)).collect();

        let full = ssim_subsampled(&ref_frame, &dist_frame, 256, 256, &cfg_full);
        let sub = ssim_subsampled(&ref_frame, &dist_frame, 256, 256, &cfg_sub);

        assert!(
            (full - sub).abs() < 0.05,
            "subsampled SSIM {sub} should be within ±0.05 of full-res SSIM {full}"
        );
    }

    #[test]
    fn test_ssim_subsampled_quarter_identical() {
        let cfg = SsimConfig {
            subsample: 4,
            ..Default::default()
        };
        let frame = make_frame(256, 256, 200);
        let result = ssim_subsampled(&frame, &frame, 256, 256, &cfg);
        assert!(
            (result - 1.0).abs() < 1e-4,
            "identical 4× subsampled → SSIM ≈ 1.0, got {result}"
        );
    }

    #[test]
    fn test_ssim_subsampled_low_quality() {
        let cfg = SsimConfig {
            subsample: 2,
            ..Default::default()
        };
        // Use a gradient reference (dark top-half, bright bottom-half) vs uniform mid-grey.
        // After 2× downsampling the structural difference is still large enough.
        let ref_frame: Vec<u8> = (0..256_u32)
            .flat_map(|y| {
                (0..256_u32).map(move |_x| {
                    // Top half dark, bottom half bright
                    if y < 128 {
                        20u8
                    } else {
                        235u8
                    }
                })
            })
            .collect();
        let dist_frame = make_frame(256, 256, 128);
        let result = ssim_subsampled(&ref_frame, &dist_frame, 256, 256, &cfg);
        assert!(result < 0.9, "dissimilar frames → SSIM < 0.9, got {result}");
    }

    #[test]
    fn test_box_downsample_uniform() {
        let plane = vec![100u8; 16 * 16];
        let ds = box_downsample(&plane, 16, 16, 2);
        assert_eq!(ds.len(), 8 * 8);
        assert!(ds.iter().all(|&v| v == 100));
    }
}
