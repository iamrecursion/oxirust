//! HDR image merging from bracketed exposures.
//!
//! Implements Mertens exposure fusion and a global Reinhard tone-mapping
//! operator for merging multiple LDR exposures into a single HDR result.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── Data structures ──────────────────────────────────────────────────────────

/// A single exposure from a bracketed set.
///
/// `pixels` contains linearised RGB values in `[0.0, 1.0]` stored as
/// interleaved `[R, G, B, R, G, B, …]`.
#[derive(Debug, Clone)]
pub struct ExposedFrame {
    /// Exposure value (EV).  More negative → darker frame.
    pub ev: f64,
    /// Linearised pixel data (interleaved RGB, `[0.0, 1.0]`).
    pub pixels: Vec<f64>,
}

impl ExposedFrame {
    /// Creates a new `ExposedFrame`.
    #[must_use]
    pub fn new(ev: f64, pixels: Vec<f64>) -> Self {
        Self { ev, pixels }
    }

    /// Returns the number of pixels (triplets) in the frame.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.pixels.len() / 3
    }
}

// ── Weight functions ─────────────────────────────────────────────────────────

/// Gaussian well-exposedness weight centred at 0.5.
///
/// Returns a value in `(0.0, 1.0]`; pixels close to 0.5 receive weight ≈ 1.0,
/// under- and over-exposed pixels receive weight approaching 0.
#[must_use]
pub fn well_exposedness_weight(value: f64) -> f64 {
    // σ = 0.2 gives a reasonable half-width
    let sigma = 0.2_f64;
    let x = value - 0.5;
    (-(x * x) / (2.0 * sigma * sigma)).exp()
}

/// Laplacian-magnitude contrast weight for a local patch.
///
/// `patch` is a small grayscale neighbourhood (e.g. 3×3 luminance values).
/// Returns the absolute value of the discrete Laplacian at the centre pixel.
///
/// # Panics
/// Does not panic; returns 0.0 if `patch` has fewer than 5 elements.
#[must_use]
pub fn contrast_weight(patch: &[f64]) -> f64 {
    // Require at least a 1D 5-tap neighbourhood; for a 3×3 grid we use the
    // standard 4-connected Laplacian approximation on elements [1,3,4,5,7].
    if patch.len() < 5 {
        return 0.0;
    }
    let center = patch[4];

    (patch[1] + patch[3] + patch[5] + patch[7] - 4.0 * center).abs()
}

// ── Core algorithms ──────────────────────────────────────────────────────────

/// Computes BT.709 luminance from linear RGB components.
#[must_use]
pub fn compute_luminance(r: f64, g: f64, b: f64) -> f64 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Mertens exposure fusion.
///
/// Merges `frames` using per-pixel well-exposedness weights.  No HDR assembly
/// step is needed—the result is a directly displayable LDR image.
///
/// # Arguments
/// * `frames`  – Bracketed exposures with linearised pixels.
/// * `width`   – Image width.
/// * `height`  – Image height (used for validation; not strictly required for
///               the computation).
///
/// Returns an interleaved RGB `f64` vector of length `width * height * 3`.
///
/// # Panics
/// Does not panic on empty `frames`; returns a black image instead.
#[must_use]
pub fn merge_hdr_mertens(frames: &[ExposedFrame], width: u32, height: u32) -> Vec<f64> {
    let pixel_count = (width as usize) * (height as usize);
    let channel_count = pixel_count * 3;

    if frames.is_empty() || pixel_count == 0 {
        return vec![0.0; channel_count];
    }

    // Per-pixel weight accumulators
    let mut weight_sum = vec![0.0_f64; pixel_count];
    let mut weighted_r = vec![0.0_f64; pixel_count];
    let mut weighted_g = vec![0.0_f64; pixel_count];
    let mut weighted_b = vec![0.0_f64; pixel_count];

    for frame in frames {
        let pix = &frame.pixels;
        if pix.len() < channel_count {
            continue;
        }
        for p in 0..pixel_count {
            let r = pix[p * 3];
            let g = pix[p * 3 + 1];
            let b = pix[p * 3 + 2];

            // Well-exposedness: average across channels
            let we = (well_exposedness_weight(r)
                + well_exposedness_weight(g)
                + well_exposedness_weight(b))
                / 3.0;

            // Saturation: std-dev of RGB channels
            let mean = (r + g + b) / 3.0;
            let variance = ((r - mean).powi(2) + (g - mean).powi(2) + (b - mean).powi(2)) / 3.0;
            let saturation = variance.sqrt();

            // Combined weight (avoid exact zero so all frames contribute)
            let w = (we * (saturation + 1e-6)).max(1e-10);

            weight_sum[p] += w;
            weighted_r[p] += w * r;
            weighted_g[p] += w * g;
            weighted_b[p] += w * b;
        }
    }

    // Normalise
    let mut result = vec![0.0_f64; channel_count];
    for p in 0..pixel_count {
        let ws = weight_sum[p].max(1e-10);
        result[p * 3] = (weighted_r[p] / ws).clamp(0.0, 1.0);
        result[p * 3 + 1] = (weighted_g[p] / ws).clamp(0.0, 1.0);
        result[p * 3 + 2] = (weighted_b[p] / ws).clamp(0.0, 1.0);
    }
    result
}

/// Global Reinhard tone-mapping operator.
///
/// Maps HDR luminance values to `[0.0, 1.0]` using:
/// `L_d = L_scaled / (1 + L_scaled)` where `L_scaled = key * L / L_avg`.
///
/// # Arguments
/// * `hdr`   – Interleaved linear RGB input.
/// * `key`   – Key value controlling overall brightness (typically 0.18).
/// * `lwmax` – Maximum luminance in the scene (for highlight compression).
///             If `≤ 0`, auto-computed from input.
#[must_use]
pub fn tone_map_reinhard(hdr: &[f64], key: f64, lwmax: f64) -> Vec<f64> {
    if hdr.is_empty() {
        return Vec::new();
    }

    let pixel_count = hdr.len() / 3;
    if pixel_count == 0 {
        return Vec::new();
    }

    // Compute log-average luminance
    let delta = 1e-6_f64;
    let log_sum: f64 = (0..pixel_count)
        .map(|p| {
            let r = hdr[p * 3];
            let g = hdr[p * 3 + 1];
            let b = hdr[p * 3 + 2];
            let lum = compute_luminance(r, g, b);
            (lum + delta).ln()
        })
        .sum();
    let lw_avg = (log_sum / pixel_count as f64).exp();

    // Effective maximum luminance
    let effective_max = if lwmax > 0.0 {
        lwmax
    } else {
        (0..pixel_count)
            .map(|p| compute_luminance(hdr[p * 3], hdr[p * 3 + 1], hdr[p * 3 + 2]))
            .fold(0.0_f64, f64::max)
            .max(1.0)
    };

    let scale = key / lw_avg;
    let lwhite = scale * effective_max;

    let mut out = vec![0.0_f64; hdr.len()];
    for p in 0..pixel_count {
        let r = hdr[p * 3];
        let g = hdr[p * 3 + 1];
        let b = hdr[p * 3 + 2];
        let lum = compute_luminance(r, g, b);
        let l_scaled = scale * lum;
        // Extended Reinhard with white-point
        let l_out = if lwhite > 0.0 {
            l_scaled * (1.0 + l_scaled / (lwhite * lwhite)) / (1.0 + l_scaled)
        } else {
            l_scaled / (1.0 + l_scaled)
        };

        // Scale RGB by the luminance ratio
        let ratio = if lum > 1e-10 { l_out / lum } else { 0.0 };
        out[p * 3] = (r * ratio).clamp(0.0, 1.0);
        out[p * 3 + 1] = (g * ratio).clamp(0.0, 1.0);
        out[p * 3 + 2] = (b * ratio).clamp(0.0, 1.0);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_well_exposedness_midpoint_is_max() {
        let mid = well_exposedness_weight(0.5);
        let low = well_exposedness_weight(0.0);
        let high = well_exposedness_weight(1.0);
        assert!(mid > low, "midpoint should be brighter than black");
        assert!(mid > high, "midpoint should be brighter than white");
        assert!((mid - 1.0).abs() < 1e-9, "w(0.5) should be 1.0");
    }

    #[test]
    fn test_well_exposedness_symmetry() {
        let w1 = well_exposedness_weight(0.3);
        let w2 = well_exposedness_weight(0.7);
        assert!(
            (w1 - w2).abs() < 1e-9,
            "weight must be symmetric around 0.5"
        );
    }

    #[test]
    fn test_contrast_weight_flat() {
        let patch = vec![0.5; 9];
        let cw = contrast_weight(&patch);
        assert!(cw < 1e-9, "flat patch should have near-zero contrast");
    }

    #[test]
    fn test_contrast_weight_spike() {
        let mut patch = vec![0.0; 9];
        patch[4] = 1.0; // bright centre on dark background
        let cw = contrast_weight(&patch);
        assert!(cw > 3.0, "spike should have high contrast weight");
    }

    #[test]
    fn test_contrast_weight_short_patch() {
        let patch = vec![0.5, 0.5, 0.5];
        assert_eq!(contrast_weight(&patch), 0.0);
    }

    #[test]
    fn test_compute_luminance_white() {
        let lum = compute_luminance(1.0, 1.0, 1.0);
        assert!((lum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_luminance_black() {
        let lum = compute_luminance(0.0, 0.0, 0.0);
        assert_eq!(lum, 0.0);
    }

    #[test]
    fn test_compute_luminance_bt709_coefficients() {
        // Pure red should give BT.709 red coefficient
        let lum = compute_luminance(1.0, 0.0, 0.0);
        assert!((lum - 0.2126).abs() < 1e-9);
    }

    #[test]
    fn test_merge_hdr_mertens_output_size() {
        let pixels = vec![0.5_f64; 4 * 4 * 3];
        let frames = vec![
            ExposedFrame::new(-1.0, pixels.clone()),
            ExposedFrame::new(0.0, pixels.clone()),
            ExposedFrame::new(1.0, pixels),
        ];
        let result = merge_hdr_mertens(&frames, 4, 4);
        assert_eq!(result.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_merge_hdr_mertens_empty_frames() {
        let result = merge_hdr_mertens(&[], 4, 4);
        assert_eq!(result.len(), 4 * 4 * 3);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_merge_hdr_mertens_values_in_range() {
        let pixels = vec![0.4_f64; 8 * 8 * 3];
        let frames = vec![
            ExposedFrame::new(-1.0, pixels.clone()),
            ExposedFrame::new(0.0, pixels),
        ];
        let result = merge_hdr_mertens(&frames, 8, 8);
        for v in &result {
            assert!(*v >= 0.0 && *v <= 1.0, "output must be in [0, 1]");
        }
    }

    #[test]
    fn test_tone_map_reinhard_output_size() {
        let hdr = vec![0.5_f64; 10 * 10 * 3];
        let out = tone_map_reinhard(&hdr, 0.18, 1.0);
        assert_eq!(out.len(), hdr.len());
    }

    #[test]
    fn test_tone_map_reinhard_values_in_range() {
        let hdr: Vec<f64> = (0..100 * 3)
            .map(|i| (i as f64 / (100 * 3) as f64) * 10.0)
            .collect();
        let out = tone_map_reinhard(&hdr, 0.18, 0.0);
        for v in &out {
            assert!(*v >= 0.0 && *v <= 1.0, "output out of range: {}", v);
        }
    }

    #[test]
    fn test_tone_map_reinhard_empty() {
        let out = tone_map_reinhard(&[], 0.18, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_exposed_frame_pixel_count() {
        let frame = ExposedFrame::new(0.0, vec![0.0; 30]);
        assert_eq!(frame.pixel_count(), 10);
    }
}
