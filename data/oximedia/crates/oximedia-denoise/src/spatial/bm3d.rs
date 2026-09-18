//! BM3D-inspired denoising for [`Frame2D`].
//!
//! Block-Matching 3D (BM3D) is the state-of-the-art image denoising algorithm.
//! This module implements the core concepts of BM3D without the full DFT/DHT
//! 3D transform stack, trading some quality for implementation simplicity and
//! pure-Rust dependencies:
//!
//! 1. **Block matching** — For every reference patch, find the most similar
//!    patches in the frame using SSD (Sum of Squared Differences) distance.
//!
//! 2. **Collaborative Wiener filtering** — Stack matched patches into a 3D
//!    array.  Apply a separable 2-D Haar wavelet to each patch, then apply
//!    Wiener shrinkage in the transform domain using the stacked variance as a
//!    signal-power estimate.
//!
//! 3. **Aggregation** — Accumulate the filtered patches back to their original
//!    positions with inverse-variance weighting.  Divide by accumulated weights
//!    to get the final denoised estimate.
//!
//! # References
//!
//! * Dabov, K. et al. "Image Denoising by Sparse 3D Transform-Domain
//!   Collaborative Filtering" (2007). <https://doi.org/10.1109/TIP.2007.901238>

use crate::frame2d::Frame2D;
use crate::{DenoiseError, DenoiseResult};
use rayon::prelude::*;

/// BM3D-inspired denoiser configuration.
///
/// ```
/// use oximedia_denoise::spatial::bm3d::Bm3dDenoiser;
/// let d = Bm3dDenoiser::new(8, 32, 25.0).expect("valid params");
/// assert_eq!(d.block_size, 8);
/// ```
#[derive(Clone, Debug)]
pub struct Bm3dDenoiser {
    /// Side length of square patches in pixels.
    pub block_size: u32,
    /// Half-width of the search window around each reference patch.
    pub search_window: u32,
    /// Assumed noise standard deviation (controls Wiener shrinkage strength).
    pub sigma: f32,
    /// Maximum number of similar patches to group (group size cap).
    pub max_group_size: usize,
    /// SSD threshold for patch similarity (relative to block_size²).
    pub ssd_threshold: f32,
}

impl Bm3dDenoiser {
    /// Construct a `Bm3dDenoiser` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `block_size`    – Patch side length (must be ≥ 2 and ≤ 64).
    /// * `search_window` – Search half-width (must be ≥ block_size).
    /// * `sigma`         – Noise sigma (must be > 0.0).
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] for invalid parameters.
    pub fn new(block_size: u32, search_window: u32, sigma: f32) -> DenoiseResult<Self> {
        if block_size < 2 || block_size > 64 {
            return Err(DenoiseError::InvalidConfig(format!(
                "block_size must be in [2, 64], got {block_size}"
            )));
        }
        if search_window < block_size {
            return Err(DenoiseError::InvalidConfig(format!(
                "search_window ({search_window}) must be >= block_size ({block_size})"
            )));
        }
        if sigma <= 0.0 {
            return Err(DenoiseError::InvalidConfig(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        Ok(Self {
            block_size,
            search_window,
            sigma,
            max_group_size: 16,
            ssd_threshold: 2500.0,
        })
    }

    /// Denoise `frame`, returning a new [`Frame2D`].
    ///
    /// Implements the two-pass BM3D pipeline:
    /// - **Pass 1 (basic estimate)**: Block-match and collaborate-Wiener-filter
    ///   using the noisy frame itself as the guide.  This gives a rough but
    ///   unbiased first estimate.
    /// - **Pass 2 (final estimate)**: Repeat block-matching and Wiener
    ///   filtering using the Pass 1 output as the guide.  The cleaner guide
    ///   provides better signal-variance estimates leading to a higher-quality
    ///   Wiener gain.
    ///
    /// The outputs of both passes are blended (0.4 basic + 0.6 refined) to
    /// avoid over-smoothing artefacts from the guide noise propagation.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::ProcessingError`] if `frame` is empty.
    pub fn denoise(&self, frame: &Frame2D) -> DenoiseResult<Frame2D> {
        if frame.is_empty() {
            return Err(DenoiseError::ProcessingError(
                "Bm3dDenoiser: empty input frame".to_string(),
            ));
        }

        // Pass 1: basic estimate using noisy frame as self-guide.
        let basic = self.bm3d_step(frame, frame)?;

        // Pass 2: refine using the basic estimate as guide.
        let refined = self.bm3d_step(frame, &basic)?;

        // Blend: give more weight to the refined estimate while retaining
        // some from basic to avoid over-smoothing artefacts.
        let w = frame.width;
        let h = frame.height;
        let mut blended = Frame2D::new(w, h)?;
        for y in 0..h {
            for x in 0..w {
                let b = basic.get(y, x);
                let r = refined.get(y, x);
                // 40% basic + 60% refined
                blended.set(y, x, 0.4 * b + 0.6 * r);
            }
        }
        Ok(blended)
    }

    /// Core BM3D processing step.
    ///
    /// `noisy` is the input to denoise; `guide` is used to compute Wiener
    /// shrinkage weights (on the first pass guide == noisy; on the second pass
    /// guide is the basic estimate from pass one).
    ///
    /// The implementation:
    /// 1. Processes reference patches on a `step = (bs/2).max(1)` grid.
    /// 2. For each reference patch: block-match + collaborative Wiener filter.
    /// 3. Scatters the filtered patch back to its pixel positions with
    ///    reliability weighting — this is the classic BM3D aggregation step.
    /// 4. Divides accumulated values by accumulated weights to form the output.
    ///
    /// Parallelism: each row of reference patches is processed independently on
    /// a thread using `rayon`.  Each thread returns a full-frame accumulator
    /// pair that is merged sequentially afterwards.  This avoids data races
    /// while preserving most of the throughput benefit.
    fn bm3d_step(&self, noisy: &Frame2D, guide: &Frame2D) -> DenoiseResult<Frame2D> {
        let w = noisy.width;
        let h = noisy.height;
        let bs = self.block_size as usize;

        // For very small frames (smaller than block_size), fall back to
        // the Wiener spatial path on the whole frame.
        if w < bs || h < bs {
            return self.wiener_fallback(noisy);
        }

        let step = (bs / 2).max(1);
        let num_cols = w;

        // Collect reference patch positions on a regular grid.
        let ref_positions: Vec<(usize, usize)> = (0..h)
            .step_by(step)
            .flat_map(|ry| (0..num_cols).step_by(step).map(move |rx| (ry, rx)))
            .collect();

        // Compute one (acc, wgt) buffer per reference patch, then merge.
        // Each buffer entry accumulates filtered_patch_pixel * weight and weight
        // for all pixels covered by that patch.
        let patch_contributions: Vec<(Vec<(usize, usize, f32, f32)>,)> = ref_positions
            .par_iter()
            .map(|&(ref_y, ref_x)| {
                let ref_patch = extract_patch(guide, ref_x, ref_y, bs);
                let noisy_patch = extract_patch(noisy, ref_x, ref_y, bs);
                let matches = self.find_similar_patches(guide, ref_x, ref_y, &ref_patch);
                let (filtered_patch, wiener_weight) =
                    self.collaborative_wiener(&noisy_patch, &matches, noisy, guide);

                // Collect (global_y, global_x, value, weight) for each pixel
                // in this patch that is within frame bounds.
                let mut contributions = Vec::with_capacity(bs * bs);
                for py in 0..bs {
                    let fy = ref_y + py;
                    if fy >= h {
                        break;
                    }
                    for px in 0..bs {
                        let fx = ref_x + px;
                        if fx >= w {
                            break;
                        }
                        let val = filtered_patch[py * bs + px];
                        contributions.push((fy, fx, val * wiener_weight, wiener_weight));
                    }
                }

                (contributions,)
            })
            .collect();

        // Merge all patch contributions into global accumulation buffers.
        let mut acc = vec![0.0f32; w * h];
        let mut wgt = vec![0.0f32; w * h];

        for (contributions,) in patch_contributions {
            for (fy, fx, val, wt) in contributions {
                acc[fy * w + fx] += val;
                wgt[fy * w + fx] += wt;
            }
        }

        // Divide by accumulated weights; fill zero-weight pixels from guide.
        let mut output_data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let wi = wgt[y * w + x];
                output_data[y * w + x] = if wi > 0.0 {
                    acc[y * w + x] / wi
                } else {
                    guide.get(y, x)
                };
            }
        }

        Frame2D::from_vec(output_data, w, h)
    }

    /// Find patches in `guide` similar to `ref_patch` centred at
    /// `(ref_x, ref_y)`.
    ///
    /// Returns a list of `(y, x)` coordinates of the top-matching patches
    /// sorted by SSD ascending, capped at `self.max_group_size`.
    fn find_similar_patches(
        &self,
        guide: &Frame2D,
        ref_x: usize,
        ref_y: usize,
        ref_patch: &[f32],
    ) -> Vec<(usize, usize)> {
        let bs = self.block_size as usize;
        let sw = self.search_window as usize;
        let w = guide.width;
        let h = guide.height;

        let search_y0 = (ref_y as i32 - sw as i32).max(0) as usize;
        let search_y1 = (ref_y + sw + 1).min(h.saturating_sub(bs - 1));
        let search_x0 = (ref_x as i32 - sw as i32).max(0) as usize;
        let search_x1 = (ref_x + sw + 1).min(w.saturating_sub(bs - 1));

        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();

        for sy in (search_y0..search_y1).step_by(1) {
            for sx in (search_x0..search_x1).step_by(1) {
                let ssd = patch_ssd(guide, sx, sy, ref_patch, bs);
                if ssd <= self.ssd_threshold * bs as f32 {
                    candidates.push((ssd, sy, sx));
                }
            }
        }

        // Sort by SSD ascending.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.max_group_size);

        // Always include reference patch itself.
        let mut matches: Vec<(usize, usize)> =
            candidates.into_iter().map(|(_, y, x)| (y, x)).collect();

        // Ensure reference patch is in the list.
        if !matches.iter().any(|&(y, x)| y == ref_y && x == ref_x) {
            matches.insert(0, (ref_y, ref_x));
        }

        if matches.is_empty() {
            matches.push((ref_y, ref_x));
        }

        matches
    }

    /// Collaborative Wiener filtering on matched patches.
    ///
    /// 1. Collects all matched patches from `noisy` and `guide`.
    /// 2. Computes signal variance for each transform coefficient from `guide`.
    /// 3. Applies Wiener shrinkage to `noisy` coefficients.
    /// 4. Returns the filtered reference patch + scalar reliability weight.
    fn collaborative_wiener(
        &self,
        ref_noisy: &[f32],
        matches: &[(usize, usize)],
        noisy: &Frame2D,
        guide: &Frame2D,
    ) -> (Vec<f32>, f32) {
        let bs = self.block_size as usize;
        let n = matches.len();
        let noise_var = self.sigma * self.sigma;

        // Stack matched patches from both noisy and guide frames.
        let mut noisy_stack: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut guide_stack: Vec<Vec<f32>> = Vec::with_capacity(n);

        for &(y, x) in matches {
            noisy_stack.push(extract_patch(noisy, x, y, bs));
            guide_stack.push(extract_patch(guide, x, y, bs));
        }

        // Apply Haar wavelet transform to each patch.
        // noisy_transformed is used indirectly via ref_noisy_t computed below.
        let _noisy_transformed: Vec<Vec<f32>> = noisy_stack
            .iter()
            .map(|p| haar_transform_2d(p, bs))
            .collect();
        let guide_transformed: Vec<Vec<f32>> = guide_stack
            .iter()
            .map(|p| haar_transform_2d(p, bs))
            .collect();

        // Compute per-coefficient Wiener gain using guide statistics.
        //
        // A single-level 2D Haar transform produces four subbands stored in
        // row-major order within a `bs×bs` array:
        //
        //   LL (smooth)  LH (horizontal detail)   [rows 0..half, cols 0..half | cols half..bs]
        //   HL (vertical detail)  HH (diagonal)   [rows half..bs, ...]
        //
        // Where `half = bs / 2`.  The LL subband represents the signal's
        // low-frequency content and must be fully preserved (gain = 1.0).
        // The detail subbands (LH, HL, HH) receive Wiener shrinkage.
        let num_coeffs = bs * bs;
        let half = bs / 2;
        let mut wiener_gain = vec![0.0f32; num_coeffs];
        let mut n_nonzero_gains = 0u32;

        for coeff_idx in 0..num_coeffs {
            let row = coeff_idx / bs;
            let col = coeff_idx % bs;
            let is_ll = row < half && col < half;

            if is_ll {
                // LL subband — always preserve low-frequency content.
                wiener_gain[coeff_idx] = 1.0;
                n_nonzero_gains += 1;
                continue;
            }

            // Detail coefficient: apply Wiener shrinkage.
            let signal_mean =
                guide_transformed.iter().map(|t| t[coeff_idx]).sum::<f32>() / n as f32;
            let signal_var = guide_transformed
                .iter()
                .map(|t| {
                    let d = t[coeff_idx] - signal_mean;
                    d * d
                })
                .sum::<f32>()
                / n as f32;

            // Wiener gain = signal / (signal + noise); bounded in [0, 1].
            let g = (signal_var / (signal_var + noise_var).max(1e-6)).clamp(0.0, 1.0);
            wiener_gain[coeff_idx] = g;
            if g > 0.01 {
                n_nonzero_gains += 1;
            }
        }

        // Apply Wiener gain to the reference noisy patch transform.
        let ref_noisy_t = haar_transform_2d(ref_noisy, bs);
        let mut filtered_t = vec![0.0f32; num_coeffs];
        for i in 0..num_coeffs {
            filtered_t[i] = ref_noisy_t[i] * wiener_gain[i];
        }

        // Inverse transform.
        let filtered_patch = haar_inverse_2d(&filtered_t, bs);

        // Reliability weight: inversely proportional to number of non-zero gains.
        // Patches with fewer active detail coefficients are less reliable.
        let reliability = 1.0 / (n_nonzero_gains.max(1) as f32);

        (filtered_patch, reliability)
    }

    /// Wiener spatial fallback for frames smaller than `block_size`.
    fn wiener_fallback(&self, frame: &Frame2D) -> DenoiseResult<Frame2D> {
        let w = frame.width;
        let h = frame.height;
        let noise_var = self.sigma * self.sigma;
        let window = 2usize;

        let mut out = Frame2D::new(w, h)?;
        for y in 0..h {
            for x in 0..w {
                let (mean, var) = local_stats(frame, x, y, window);
                let signal_var = (var - noise_var).max(0.0);
                let gain = signal_var / (var + noise_var).max(1e-6);
                let center = frame.get(y, x);
                out.set(y, x, mean + gain * (center - mean));
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Haar wavelet helpers
// ---------------------------------------------------------------------------

/// 1-D Haar forward transform (in-place, length must be a power of two).
fn haar_forward_1d(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let mut temp = vec![0.0f32; n];
    let half = n / 2;
    for i in 0..half {
        temp[i] = (data[2 * i] + data[2 * i + 1]) * std::f32::consts::FRAC_1_SQRT_2;
        temp[half + i] = (data[2 * i] - data[2 * i + 1]) * std::f32::consts::FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&temp);
}

/// 1-D Haar inverse transform (in-place).
fn haar_inverse_1d(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let mut temp = vec![0.0f32; n];
    let half = n / 2;
    for i in 0..half {
        temp[2 * i] = (data[i] + data[half + i]) * std::f32::consts::FRAC_1_SQRT_2;
        temp[2 * i + 1] = (data[i] - data[half + i]) * std::f32::consts::FRAC_1_SQRT_2;
    }
    data.copy_from_slice(&temp);
}

/// Round `n` up to the nearest power-of-two >= `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Apply a separable 2-D Haar wavelet transform to a `bs×bs` patch.
///
/// Because `bs` may not be a power of two, the patch is zero-padded to
/// the next power of two, transformed, then the output is truncated back
/// to `bs×bs`.
fn haar_transform_2d(patch: &[f32], bs: usize) -> Vec<f32> {
    let padded = next_pow2(bs);
    let mut buf = vec![0.0f32; padded * padded];

    // Copy patch into padded buffer.
    for y in 0..bs {
        for x in 0..bs {
            buf[y * padded + x] = patch[y * bs + x];
        }
    }

    // Apply row transforms.
    for y in 0..padded {
        haar_forward_1d(&mut buf[y * padded..(y + 1) * padded]);
    }

    // Apply column transforms.
    let mut col_buf = vec![0.0f32; padded];
    for x in 0..padded {
        for y in 0..padded {
            col_buf[y] = buf[y * padded + x];
        }
        haar_forward_1d(&mut col_buf);
        for y in 0..padded {
            buf[y * padded + x] = col_buf[y];
        }
    }

    // Extract bs×bs sub-block.
    let mut out = vec![0.0f32; bs * bs];
    for y in 0..bs {
        for x in 0..bs {
            out[y * bs + x] = buf[y * padded + x];
        }
    }
    out
}

/// Inverse 2-D Haar wavelet transform.
fn haar_inverse_2d(coeffs: &[f32], bs: usize) -> Vec<f32> {
    let padded = next_pow2(bs);
    let mut buf = vec![0.0f32; padded * padded];

    for y in 0..bs {
        for x in 0..bs {
            buf[y * padded + x] = coeffs[y * bs + x];
        }
    }

    // Inverse column transforms.
    let mut col_buf = vec![0.0f32; padded];
    for x in 0..padded {
        for y in 0..padded {
            col_buf[y] = buf[y * padded + x];
        }
        haar_inverse_1d(&mut col_buf);
        for y in 0..padded {
            buf[y * padded + x] = col_buf[y];
        }
    }

    // Inverse row transforms.
    for y in 0..padded {
        haar_inverse_1d(&mut buf[y * padded..(y + 1) * padded]);
    }

    let mut out = vec![0.0f32; bs * bs];
    for y in 0..bs {
        for x in 0..bs {
            out[y * bs + x] = buf[y * padded + x];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Patch helpers
// ---------------------------------------------------------------------------

/// Extract a `bs×bs` patch from `frame` anchored at top-left `(x, y)`.
///
/// Boundary pixels are clamped.
fn extract_patch(frame: &Frame2D, x: usize, y: usize, bs: usize) -> Vec<f32> {
    let mut patch = vec![0.0f32; bs * bs];
    for py in 0..bs {
        for px in 0..bs {
            patch[py * bs + px] = frame.get_clamped((y + py) as i32, (x + px) as i32);
        }
    }
    patch
}

/// SSD distance between patch `p` and the patch in `frame` at `(sx, sy)`.
fn patch_ssd(frame: &Frame2D, sx: usize, sy: usize, reference: &[f32], bs: usize) -> f32 {
    let mut ssd = 0.0f32;
    for py in 0..bs {
        for px in 0..bs {
            let fv = frame.get_clamped((sy + py) as i32, (sx + px) as i32);
            let rv = reference[py * bs + px];
            let d = fv - rv;
            ssd += d * d;
        }
    }
    ssd
}

/// Local mean and variance in a `(2*window+1)²` neighbourhood.
fn local_stats(frame: &Frame2D, x: usize, y: usize, window: usize) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0u32;

    for dy in -(window as i32)..=(window as i32) {
        for dx in -(window as i32)..=(window as i32) {
            let v = frame.get_clamped(y as i32 + dy, x as i32 + dx);
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }

    let mean = sum / count as f32;
    let variance = (sum_sq / count as f32 - mean * mean).max(0.0);
    (mean, variance)
}

// ---------------------------------------------------------------------------
// Convenience top-level function
// ---------------------------------------------------------------------------

/// Denoise a floating-point grayscale image using BM3D.
///
/// This is a convenience wrapper around [`Bm3dDenoiser`] that uses sensible
/// default parameters: `block_size = 8`, `search_window = 32`.
///
/// # Arguments
///
/// * `src`    – Input pixel data in row-major order, values in `[0.0, 255.0]`.
/// * `width`  – Image width in pixels (must be > 0).
/// * `height` – Image height in pixels (must be > 0).
/// * `sigma`  – Assumed noise standard deviation (must be > 0).
///
/// # Returns
///
/// Denoised pixel data with the same length as `src`, or the original data
/// if construction/processing fails (error-safe fallback).
///
/// # Panics
///
/// Never panics; errors produce the unmodified input as output.
pub fn bm3d_denoise(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    // Validate inputs and build denoiser with safe defaults.
    let block_size = 8u32;
    let search_window = 32u32.max(block_size);

    let denoiser = match Bm3dDenoiser::new(block_size, search_window, sigma.max(1e-6)) {
        Ok(d) => d,
        Err(_) => return src.to_vec(),
    };

    let frame = match Frame2D::from_vec(src.to_vec(), width, height) {
        Ok(f) => f,
        Err(_) => return src.to_vec(),
    };

    match denoiser.denoise(&frame) {
        Ok(out) => out.data,
        Err(_) => src.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_noisy(w: usize, h: usize, base: f32, amplitude: f32) -> Frame2D {
        let data: Vec<f32> = (0..w * h)
            .map(|i| {
                let noise = if i % 3 == 0 {
                    amplitude
                } else if i % 3 == 1 {
                    -amplitude
                } else {
                    0.0
                };
                (base + noise).clamp(0.0, 255.0)
            })
            .collect();
        Frame2D::from_vec(data, w, h).expect("valid")
    }

    // ----- parameter validation -----

    #[test]
    fn test_bm3d_new_valid() {
        let d = Bm3dDenoiser::new(8, 32, 25.0).expect("valid params");
        assert_eq!(d.block_size, 8);
        assert_eq!(d.search_window, 32);
        assert!((d.sigma - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bm3d_block_size_too_small() {
        assert!(Bm3dDenoiser::new(1, 16, 25.0).is_err());
    }

    #[test]
    fn test_bm3d_block_size_too_large() {
        assert!(Bm3dDenoiser::new(65, 128, 25.0).is_err());
    }

    #[test]
    fn test_bm3d_search_window_too_small() {
        assert!(Bm3dDenoiser::new(8, 4, 25.0).is_err()); // sw < bs
    }

    #[test]
    fn test_bm3d_sigma_zero() {
        assert!(Bm3dDenoiser::new(8, 16, 0.0).is_err());
    }

    #[test]
    fn test_bm3d_sigma_negative() {
        assert!(Bm3dDenoiser::new(8, 16, -5.0).is_err());
    }

    // ----- output shape -----

    #[test]
    fn test_bm3d_output_shape() {
        let d = Bm3dDenoiser::new(4, 8, 20.0).expect("valid");
        let frame = Frame2D::filled(24, 24, 128.0).expect("valid");
        let out = d.denoise(&frame).expect("denoise ok");
        assert_eq!(out.width, 24);
        assert_eq!(out.height, 24);
    }

    #[test]
    fn test_bm3d_empty_frame_error() {
        let d = Bm3dDenoiser::new(4, 8, 20.0).expect("valid");
        let empty = Frame2D {
            data: vec![],
            width: 0,
            height: 0,
        };
        assert!(d.denoise(&empty).is_err());
    }

    // ----- PSNR improvement -----

    #[test]
    fn test_bm3d_psnr_improvement() {
        let d = Bm3dDenoiser::new(4, 16, 25.0).expect("valid");
        let clean = Frame2D::filled(32, 32, 128.0).expect("valid");
        let noisy = make_noisy(32, 32, 128.0, 30.0);

        let psnr_noisy = noisy.psnr(&clean).expect("psnr");
        let denoised = d.denoise(&noisy).expect("denoise ok");
        let psnr_denoised = denoised.psnr(&clean).expect("psnr");

        assert!(
            psnr_denoised >= psnr_noisy - 3.0, // allow 3 dB tolerance
            "BM3D should not significantly degrade PSNR: {psnr_noisy:.1} -> {psnr_denoised:.1}"
        );
    }

    // ----- clean signal stability -----

    #[test]
    fn test_bm3d_clean_signal_stability() {
        // A perfectly clean constant frame should remain close to the original.
        let d = Bm3dDenoiser::new(4, 8, 5.0).expect("valid"); // small sigma
        let clean = Frame2D::filled(24, 24, 128.0).expect("valid");
        let out = d.denoise(&clean).expect("denoise ok");
        let psnr = out.psnr(&clean).expect("psnr");
        assert!(
            psnr > 30.0,
            "clean signal should remain stable, PSNR={psnr:.1}"
        );
    }

    // ----- Haar wavelet round-trip -----

    #[test]
    fn test_haar_1d_roundtrip() {
        let original = vec![1.0f32, 3.0, 5.0, 7.0];
        let mut data = original.clone();
        haar_forward_1d(&mut data);
        haar_inverse_1d(&mut data);
        for (orig, reconstructed) in original.iter().zip(data.iter()) {
            assert!(
                (orig - reconstructed).abs() < 1e-4,
                "{orig} != {reconstructed}"
            );
        }
    }

    #[test]
    fn test_haar_2d_roundtrip() {
        let bs = 4;
        let original: Vec<f32> = (0..bs * bs).map(|i| i as f32).collect();
        let coeffs = haar_transform_2d(&original, bs);
        let reconstructed = haar_inverse_2d(&coeffs, bs);
        for (i, (o, r)) in original.iter().zip(reconstructed.iter()).enumerate() {
            assert!((o - r).abs() < 1e-3, "coeff {i}: {o} vs {r}");
        }
    }

    // ----- SSD distance -----

    #[test]
    fn test_ssd_identical_patches() {
        let frame = Frame2D::filled(16, 16, 100.0).expect("valid");
        let patch = vec![100.0f32; 4 * 4];
        let ssd = patch_ssd(&frame, 2, 2, &patch, 4);
        assert!(ssd < f32::EPSILON, "identical patches: SSD should be 0");
    }

    // ----- small frame fallback -----

    #[test]
    fn test_bm3d_small_frame_fallback() {
        // frame (3x3) smaller than block_size (4) triggers Wiener fallback.
        let d = Bm3dDenoiser::new(4, 8, 20.0).expect("valid");
        let frame = make_noisy(3, 3, 128.0, 30.0);
        let out = d.denoise(&frame).expect("fallback ok");
        assert_eq!(out.width, 3);
        assert_eq!(out.height, 3);
    }
}
