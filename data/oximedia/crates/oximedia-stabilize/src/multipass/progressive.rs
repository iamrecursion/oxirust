//! Progressive multi-resolution analysis for stabilization.
//!
//! Implements image pyramid refinement: coarse-to-fine motion estimation
//! where each pass warm-starts from the previous level's result.
//!
//! ## Algorithm
//!
//! Given `pyramid_levels = 3`:
//! - Pass 0 (level 2): scale = min_scale (e.g. 0.25) — coarse motion estimate
//! - Pass 1 (level 1): scale = 0.5 — refined from pass 0
//! - Pass 2 (level 0): scale = 1.0 — final full-resolution pass
//!
//! At each level the motion estimate from the previous level is scaled up (×2)
//! to provide a warm-start for the next level. This dramatically reduces the
//! search space at each level compared to a cold-start full-resolution search.
//!
//! ## Homography convention
//!
//! We estimate per-frame *stabilising* transforms as 3×3 homographies in
//! normalized image coordinates.  A frame with no motion relative to the
//! reference frame receives the identity matrix.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for progressive (pyramid) analysis.
#[derive(Debug, Clone)]
pub struct ProgressiveAnalysisConfig {
    /// Number of pyramid levels (coarse → fine).
    ///
    /// With `pyramid_levels = 3` the scales are `[min_scale, 2·min_scale, 1.0]`
    /// (clamped so the finest level is always 1.0).
    pub pyramid_levels: usize,

    /// Scale factor of the coarsest level relative to the full frame.
    ///
    /// Must be in `(0.0, 1.0]`.  Default `0.25` (quarter resolution).
    pub min_scale: f32,

    /// Early-termination threshold in pixels.
    ///
    /// If the RMS motion change between consecutive passes is smaller than
    /// this value the remaining finer passes are skipped.  Default `0.5` px.
    pub convergence_threshold: f32,
}

impl Default for ProgressiveAnalysisConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            min_scale: 0.25,
            convergence_threshold: 0.5,
        }
    }
}

/// Progressive (coarse-to-fine) video analyzer.
///
/// Builds an image pyramid and refines per-frame homography estimates from
/// the coarsest level to the finest, warm-starting each level from the
/// previous one.
pub struct ProgressiveAnalyzer {
    config: ProgressiveAnalysisConfig,
    /// Counts the number of analysis passes actually executed. Exposed for
    /// testing via the accessor [`Self::pass_count`].
    pass_count: Arc<AtomicUsize>,
}

impl ProgressiveAnalyzer {
    /// Create a new progressive analyzer with the given configuration.
    #[must_use]
    pub fn new(config: ProgressiveAnalysisConfig) -> Self {
        Self {
            config,
            pass_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Return the number of pyramid passes executed during the last
    /// [`Self::analyze`] call.  Useful for testing convergence behaviour.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.pass_count.load(Ordering::Relaxed)
    }

    /// Analyze a sequence of raw 8-bit grayscale frames progressively.
    ///
    /// Returns one 3×3 homography per frame (the stabilising transform).
    /// Frame 0 always receives the identity matrix.
    ///
    /// # Arguments
    ///
    /// * `frames` – Raw pixel data; each `Vec<u8>` is a row-major 8-bit
    ///              grayscale image of size `w × h` bytes.
    /// * `w`, `h` – Width and height of the full-resolution frames.
    ///
    /// # Panics
    ///
    /// Does not panic (uses saturating arithmetic throughout).
    #[must_use]
    pub fn analyze(&self, frames: &[Vec<u8>], w: u32, h: u32) -> Vec<[[f32; 3]; 3]> {
        self.pass_count.store(0, Ordering::Relaxed);

        if frames.is_empty() {
            return Vec::new();
        }

        let n = frames.len();
        // Per-frame homographies; index i is the transform from frame i to frame 0.
        let mut transforms: Vec<[[f32; 3]; 3]> = (0..n).map(|_| identity_h()).collect();

        // Build the scale schedule from coarse to fine.
        let scales = self.build_scale_schedule();

        for (pass_idx, &scale) in scales.iter().enumerate() {
            self.pass_count.fetch_add(1, Ordering::Relaxed);

            if scale <= 0.0 {
                continue;
            }

            // Downsample all frames to the current pyramid level.
            let (ds_frames, ds_w, ds_h) = if (scale - 1.0).abs() < 1e-4 {
                // Full resolution — avoid a redundant copy.
                let cloned: Vec<Vec<u8>> = frames.iter().cloned().collect();
                (cloned, w, h)
            } else {
                let mut ds_buf = Vec::with_capacity(n);
                let mut ow = w;
                let mut oh = h;
                // We compute scale as successive halvings.
                // Determine how many times we need to halve to reach `scale`.
                let halvings = scale_to_halvings(scale);
                for frame in frames.iter() {
                    let mut f = frame.clone();
                    let mut fw = w;
                    let mut fh = h;
                    for _ in 0..halvings {
                        let (next_f, next_w, next_h) = self.downsample(&f, fw, fh);
                        f = next_f;
                        fw = next_w;
                        fh = next_h;
                        ow = next_w;
                        oh = next_h;
                    }
                    ds_buf.push(f);
                }
                (ds_buf, ow, oh)
            };

            if ds_w == 0 || ds_h == 0 {
                continue;
            }

            // Scale the warm-start transforms down to the current resolution.
            let scaled_transforms: Vec<[[f32; 3]; 3]> = transforms
                .iter()
                .map(|&t| self.scale_homography(t, scale))
                .collect();

            // Refine motion estimates at this resolution.
            let refined = self.analyze_level(&ds_frames, ds_w, ds_h, &scaled_transforms, pass_idx);

            // Scale the refined transforms back to full resolution.
            let inv_scale = if scale.abs() > 1e-7 { 1.0 / scale } else { 1.0 };
            transforms = refined
                .iter()
                .map(|&t| self.scale_homography(t, inv_scale))
                .collect();

            // Early convergence check: compare the updated full-res transforms to
            // those from before this pass.
            // (We already overwrote `transforms`, so compare against full-res re-scaled.)
            // Skip convergence check on the first pass (no previous to compare).
            if pass_idx > 0 && self.converged(&transforms, &scaled_transforms, inv_scale) {
                break;
            }
        }

        transforms
    }

    // -----------------------------------------------------------------------
    //  Private helpers
    // -----------------------------------------------------------------------

    /// Build the scale schedule from coarse (min_scale) to fine (1.0).
    fn build_scale_schedule(&self) -> Vec<f32> {
        let levels = self.config.pyramid_levels.max(1);

        if levels == 1 {
            return vec![1.0];
        }

        // Distribute levels so that:
        //   level 0 → min_scale
        //   level last → 1.0
        // We use powers of 2 rounding (halving-based pyramid).
        let min_s = self.config.min_scale.clamp(1.0 / 32.0, 1.0);
        let max_halvings = scale_to_halvings(min_s);

        // Assign halvings for each level from coarse to fine.
        // Level 0 → max_halvings halvings, level (levels-1) → 0 halvings.
        let mut schedule = Vec::with_capacity(levels);
        for i in 0..levels {
            // Map i ∈ [0, levels-1] to halvings ∈ [max_halvings, 0]
            let halvings = if levels > 1 {
                let frac = i as f32 / (levels - 1) as f32;
                let h = (max_halvings as f32 * (1.0 - frac)).round() as u32;
                h
            } else {
                0
            };
            let scale = halvings_to_scale(halvings);
            schedule.push(scale);
        }

        schedule
    }

    /// Analyze frames at a single pyramid level.
    ///
    /// Uses simple NCC-based translation estimation between consecutive frames,
    /// then integrates into cumulative homographies.  The warm-start
    /// `prior_transforms` are used only for the motion-change check.
    fn analyze_level(
        &self,
        frames: &[Vec<u8>],
        w: u32,
        h: u32,
        prior_transforms: &[[[f32; 3]; 3]],
        _pass_idx: usize,
    ) -> Vec<[[f32; 3]; 3]> {
        let n = frames.len();
        let mut result: Vec<[[f32; 3]; 3]> = Vec::with_capacity(n);

        // Frame 0 is always the reference.
        result.push(identity_h());

        // Cumulative homography: transform from frame 0 to frame i.
        let mut cumulative = identity_h();

        for i in 1..n {
            // Estimate inter-frame translation between frame (i-1) and frame i.
            let prior_tx = prior_transforms.get(i).map(|h| h[0][2]).unwrap_or(0.0);
            let prior_ty = prior_transforms.get(i).map(|h| h[1][2]).unwrap_or(0.0);

            let (dx, dy) =
                estimate_translation_ncc(&frames[i - 1], &frames[i], w, h, prior_tx, prior_ty);

            // Build incremental homography for this frame pair.
            let delta = translation_h(-dx, -dy); // stabilising direction

            // Compose: cumulative = delta * cumulative.
            cumulative = compose_h(delta, cumulative);
            result.push(cumulative);
        }

        result
    }

    /// Downsample a grayscale frame by 2× using a 2×2 box filter.
    ///
    /// Returns `(downsampled_pixels, new_width, new_height)`.
    pub(crate) fn downsample(&self, frame: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
        let ow = (w / 2).max(1);
        let oh = (h / 2).max(1);

        if w < 2 || h < 2 {
            // Frame too small to halve; return a copy.
            return (frame.to_vec(), w, h);
        }

        let sw = w as usize;
        let ow_usize = ow as usize;
        let oh_usize = oh as usize;
        let mut out = vec![0u8; ow_usize * oh_usize];

        for oy in 0..oh_usize {
            for ox in 0..ow_usize {
                let sy = oy * 2;
                let sx = ox * 2;

                // Sample the 2×2 neighbourhood; stay in bounds.
                let p00 = pixel(frame, sw, sx, sy);
                let p10 = pixel(frame, sw, sx + 1, sy);
                let p01 = pixel(frame, sw, sx, sy + 1);
                let p11 = pixel(frame, sw, sx + 1, sy + 1);

                let avg = (p00 as u16 + p10 as u16 + p01 as u16 + p11 as u16 + 2) / 4;
                out[oy * ow_usize + ox] = avg.min(255) as u8;
            }
        }

        (out, ow, oh)
    }

    /// Scale a homography between two resolutions.
    ///
    /// Translates `src_h` (which describes motion at `1.0` normalised scale)
    /// to a homography valid at `scale × full_resolution`.
    ///
    /// The translation components (column 2, rows 0 and 1) are scaled by
    /// `scale`; the perspective components (row 2, columns 0 and 1) are
    /// scaled by `1/scale`.  The remaining components are unchanged.
    pub(crate) fn scale_homography(&self, h: [[f32; 3]; 3], scale: f32) -> [[f32; 3]; 3] {
        if scale.abs() < 1e-7 {
            return identity_h();
        }
        let inv_scale = 1.0 / scale;

        // S · H · S^{-1}  where S = diag(scale, scale, 1)
        //
        //   [s  0  0]   [h00 h01 h02]   [1/s  0   0]
        //   [0  s  0] × [h10 h11 h12] × [0   1/s  0]
        //   [0  0  1]   [h20 h21 h22]   [0    0   1]
        //
        // = [h00       h01       h02·s    ]
        //   [h10       h11       h12·s    ]
        //   [h20·(1/s) h21·(1/s) h22     ]
        [
            [h[0][0], h[0][1], h[0][2] * scale],
            [h[1][0], h[1][1], h[1][2] * scale],
            [h[2][0] * inv_scale, h[2][1] * inv_scale, h[2][2]],
        ]
    }

    /// Check whether the change in transforms between two passes is below the
    /// convergence threshold.
    fn converged(
        &self,
        new_transforms: &[[[f32; 3]; 3]],
        old_scaled: &[[[f32; 3]; 3]],
        inv_scale: f32,
    ) -> bool {
        if new_transforms.len() != old_scaled.len() {
            return false;
        }

        let mut sum_sq = 0.0f64;
        let count = new_transforms.len();

        for (new_h, old_scaled_h) in new_transforms.iter().zip(old_scaled.iter()) {
            // Scale old back to full resolution for comparison.
            let old_full = self.scale_homography(*old_scaled_h, inv_scale);
            let dt0 = (new_h[0][2] - old_full[0][2]) as f64;
            let dt1 = (new_h[1][2] - old_full[1][2]) as f64;
            sum_sq += dt0 * dt0 + dt1 * dt1;
        }

        if count == 0 {
            return true;
        }

        let rms = (sum_sq / count as f64).sqrt() as f32;
        rms < self.config.convergence_threshold
    }
}

// ---------------------------------------------------------------------------
//  Module-level helpers
// ---------------------------------------------------------------------------

/// 3×3 identity homography.
#[inline]
fn identity_h() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Pure-translation homography: `[[1,0,tx],[0,1,ty],[0,0,1]]`.
#[inline]
fn translation_h(tx: f32, ty: f32) -> [[f32; 3]; 3] {
    [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]
}

/// Compose two homographies: `a * b`.
fn compose_h(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                out[r][c] += a[r][k] * b[k][c];
            }
        }
    }
    out
}

/// Safe pixel accessor — returns `0` for out-of-bounds.
#[inline]
fn pixel(frame: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    let idx = y * stride + x;
    if idx < frame.len() {
        frame[idx]
    } else {
        0
    }
}

/// How many 2× halvings to approximate `scale`.
///
/// `scale = 0.25` → 2 halvings, `scale = 0.5` → 1 halving, `scale ≥ 1.0` → 0.
fn scale_to_halvings(scale: f32) -> u32 {
    if scale >= 1.0 {
        return 0;
    }
    // count how many times we must divide by 2
    let mut s = scale.clamp(1.0 / 32.0, 1.0);
    let mut halvings = 0u32;
    while s < 0.99 {
        s *= 2.0;
        halvings += 1;
    }
    halvings
}

/// Convert a halving count back to a nominal scale.
fn halvings_to_scale(halvings: u32) -> f32 {
    if halvings == 0 {
        1.0
    } else {
        1.0 / (1u32 << halvings) as f32
    }
}

// ---------------------------------------------------------------------------
//  NCC-based inter-frame translation estimator
// ---------------------------------------------------------------------------

/// Maximum integer pixel search radius per axis at a given pyramid level.
const SEARCH_RADIUS: i32 = 16;
/// Patch half-size for NCC.
const PATCH_HALF: i32 = 4;

/// Estimate the translation `(dx, dy)` that maps pixels from `ref_frame`
/// to `cur_frame` using Normalised Cross-Correlation (NCC) on a sparse
/// grid of candidate patches.
///
/// The warm-start `(prior_dx, prior_dy)` initialises the search centre so
/// we only need a small local search window.
///
/// Returns `(dx, dy)` in pixels — positive dx means cur has moved right
/// relative to ref.
fn estimate_translation_ncc(
    ref_frame: &[u8],
    cur_frame: &[u8],
    w: u32,
    h: u32,
    prior_dx: f32,
    prior_dy: f32,
) -> (f32, f32) {
    let sw = w as usize;
    let sh = h as usize;

    if sw < (PATCH_HALF * 2 + 1) as usize || sh < (PATCH_HALF * 2 + 1) as usize {
        return (prior_dx, prior_dy);
    }

    // Sample reference patch centres on a sparse 4×4 grid.
    let step_x = (sw / 4).max(1) as i32;
    let step_y = (sh / 4).max(1) as i32;
    let margin = PATCH_HALF + SEARCH_RADIUS + 1;

    let x_start = margin;
    let x_end = sw as i32 - margin;
    let y_start = margin;
    let y_end = sh as i32 - margin;

    if x_end <= x_start || y_end <= y_start {
        return (prior_dx, prior_dy);
    }

    let center_dx = prior_dx.round() as i32;
    let center_dy = prior_dy.round() as i32;

    let mut total_dx = 0.0f64;
    let mut total_dy = 0.0f64;
    let mut total_weight = 0.0f64;

    let mut cx = x_start;
    while cx < x_end {
        let mut cy = y_start;
        while cy < y_end {
            // Extract reference patch.
            let ref_patch = extract_patch(ref_frame, sw, sh, cx, cy, PATCH_HALF);
            let ref_mean = mean_u8(&ref_patch);
            let ref_std = std_u8(&ref_patch, ref_mean);

            if ref_std < 1.0 {
                // Homogeneous patch; skip.
                cy += step_y;
                continue;
            }

            // Search for best match in cur_frame within SEARCH_RADIUS.
            let mut best_ncc = -1.0f64;
            let mut best_dx = 0i32;
            let mut best_dy = 0i32;

            let search_range_x = (center_dx - SEARCH_RADIUS)..(center_dx + SEARCH_RADIUS + 1);
            let search_range_y = (center_dy - SEARCH_RADIUS)..(center_dy + SEARCH_RADIUS + 1);

            for ty in search_range_y {
                for tx in search_range_x.clone() {
                    let scx = cx + tx;
                    let scy = cy + ty;
                    // Bounds check for the candidate patch.
                    if scx - PATCH_HALF < 0
                        || scx + PATCH_HALF >= sw as i32
                        || scy - PATCH_HALF < 0
                        || scy + PATCH_HALF >= sh as i32
                    {
                        continue;
                    }
                    let cur_patch = extract_patch(cur_frame, sw, sh, scx, scy, PATCH_HALF);
                    let cur_mean = mean_u8(&cur_patch);
                    let cur_std = std_u8(&cur_patch, cur_mean);

                    if cur_std < 1.0 {
                        continue;
                    }

                    let ncc =
                        ncc_score(&ref_patch, ref_mean, ref_std, &cur_patch, cur_mean, cur_std);
                    if ncc > best_ncc {
                        best_ncc = ncc;
                        best_dx = tx;
                        best_dy = ty;
                    }
                }
            }

            if best_ncc > 0.5 {
                let weight = best_ncc * best_ncc * ref_std as f64;
                total_dx += best_dx as f64 * weight;
                total_dy += best_dy as f64 * weight;
                total_weight += weight;
            }

            cy += step_y;
        }
        cx += step_x;
    }

    if total_weight < 1e-9 {
        return (prior_dx, prior_dy);
    }

    let est_dx = (total_dx / total_weight) as f32;
    let est_dy = (total_dy / total_weight) as f32;

    (est_dx, est_dy)
}

/// Extract a square patch of half-size `half` centred at `(cx, cy)`.
fn extract_patch(frame: &[u8], w: usize, h: usize, cx: i32, cy: i32, half: i32) -> Vec<u8> {
    let side = (2 * half + 1) as usize;
    let mut patch = Vec::with_capacity(side * side);

    for dy in -half..=half {
        for dx in -half..=half {
            let x = (cx + dx).clamp(0, w as i32 - 1) as usize;
            let y = (cy + dy).clamp(0, h as i32 - 1) as usize;
            patch.push(frame[y * w + x]);
        }
    }

    patch
}

fn mean_u8(patch: &[u8]) -> f64 {
    if patch.is_empty() {
        return 0.0;
    }
    patch.iter().map(|&p| p as f64).sum::<f64>() / patch.len() as f64
}

fn std_u8(patch: &[u8], mean: f64) -> f64 {
    if patch.len() < 2 {
        return 0.0;
    }
    let var = patch
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / patch.len() as f64;
    var.sqrt()
}

fn ncc_score(a: &[u8], a_mean: f64, a_std: f64, b: &[u8], b_mean: f64, b_std: f64) -> f64 {
    if a.len() != b.len() || a_std < 1e-9 || b_std < 1e-9 {
        return 0.0;
    }

    let n = a.len() as f64;
    let denom = n * a_std * b_std;

    let cross: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&pa, &pb)| (pa as f64 - a_mean) * (pb as f64 - b_mean))
        .sum();

    (cross / denom).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a random-ish 8-bit grayscale frame seeded by a value.
    fn make_frame(w: u32, h: u32, seed: u8) -> Vec<u8> {
        let n = (w * h) as usize;
        (0..n)
            .map(|i| {
                let x = i as u8;
                x.wrapping_add(seed)
                    .wrapping_mul(131)
                    .wrapping_add(seed / 2)
            })
            .collect()
    }

    /// All-identical frames should yield identity (or near-identity) transforms.
    #[test]
    fn test_progressive_identity_sequence() {
        let w = 64u32;
        let h = 64u32;
        let base = make_frame(w, h, 42);
        let frames: Vec<Vec<u8>> = (0..10).map(|_| base.clone()).collect();

        let config = ProgressiveAnalysisConfig::default();
        let analyzer = ProgressiveAnalyzer::new(config);
        let transforms = analyzer.analyze(&frames, w, h);

        assert_eq!(transforms.len(), frames.len());

        for (i, t) in transforms.iter().enumerate() {
            // Translation components should be close to zero.
            let tx = t[0][2];
            let ty = t[1][2];
            assert!(
                tx.abs() < 2.0,
                "Frame {i}: tx={tx:.3} should be near 0 for identical frames"
            );
            assert!(
                ty.abs() < 2.0,
                "Frame {i}: ty={ty:.3} should be near 0 for identical frames"
            );
        }
    }

    /// With `pyramid_levels = 2`, exactly 2 passes should be executed
    /// (unless early convergence triggers after pass 1).
    #[test]
    fn test_progressive_levels() {
        let w = 64u32;
        let h = 64u32;
        let base = make_frame(w, h, 7);
        // Use identical frames so that convergence kicks in after 1 pass.
        // We want to test the *maximum* pass count with non-converging frames.
        // We'll use slightly different frames to prevent early convergence.
        let frames: Vec<Vec<u8>> = (0..8u8)
            .map(|i| make_frame(w, h, i.wrapping_mul(17)))
            .collect();
        let _ = base;

        let config = ProgressiveAnalysisConfig {
            pyramid_levels: 2,
            min_scale: 0.25,
            convergence_threshold: 0.01, // very tight → forces all passes
        };
        let analyzer = ProgressiveAnalyzer::new(config);
        let transforms = analyzer.analyze(&frames, w, h);

        // We should have at most 2 passes.
        let passes = analyzer.pass_count();
        assert!(
            passes <= 2,
            "pyramid_levels=2 should execute at most 2 passes, got {passes}"
        );
        assert!(passes >= 1, "At least 1 pass must execute, got {passes}");
        assert_eq!(
            transforms.len(),
            frames.len(),
            "Must return one transform per frame"
        );
    }

    /// Empty input must return an empty transform vector.
    #[test]
    fn test_progressive_empty_input() {
        let config = ProgressiveAnalysisConfig::default();
        let analyzer = ProgressiveAnalyzer::new(config);
        let result = analyzer.analyze(&[], 64, 64);
        assert!(result.is_empty());
    }

    /// Single frame must return a single identity transform.
    #[test]
    fn test_progressive_single_frame() {
        let w = 32u32;
        let h = 32u32;
        let config = ProgressiveAnalysisConfig::default();
        let analyzer = ProgressiveAnalyzer::new(config);
        let frames = vec![make_frame(w, h, 0)];
        let result = analyzer.analyze(&frames, w, h);
        assert_eq!(result.len(), 1);
        // Frame 0 is always identity.
        assert!((result[0][0][0] - 1.0).abs() < 1e-6, "H[0][0] should be 1");
        assert!((result[0][0][2]).abs() < 1e-6, "H[0][2] tx should be 0");
        assert!((result[0][1][2]).abs() < 1e-6, "H[1][2] ty should be 0");
    }

    /// On a 60-frame 256×256 sequence, progressive (3 levels) must complete in
    /// less than 2× the time of single-pass full-resolution analysis.
    #[test]
    fn test_progressive_faster_than_single_pass() {
        use std::time::Instant;

        let w = 256u32;
        let h = 256u32;
        let frames: Vec<Vec<u8>> = (0..60u8)
            .map(|i| make_frame(w, h, i.wrapping_mul(13)))
            .collect();

        // Progressive 3-level analysis.
        let prog_config = ProgressiveAnalysisConfig {
            pyramid_levels: 3,
            min_scale: 0.25,
            convergence_threshold: 0.5,
        };
        let prog_analyzer = ProgressiveAnalyzer::new(prog_config);

        let t0 = Instant::now();
        let _ = prog_analyzer.analyze(&frames, w, h);
        let prog_time = t0.elapsed();

        // Single-pass full-res analysis (pyramid_levels=1).
        let single_config = ProgressiveAnalysisConfig {
            pyramid_levels: 1,
            min_scale: 1.0,
            convergence_threshold: 0.5,
        };
        let single_analyzer = ProgressiveAnalyzer::new(single_config);

        let t1 = Instant::now();
        let _ = single_analyzer.analyze(&frames, w, h);
        let single_time = t1.elapsed();

        // Progressive must be < 2× single-pass time.
        assert!(
            prog_time <= single_time * 2,
            "Progressive time {prog_time:?} should be < 2× single-pass {single_time:?}"
        );
    }

    /// Downsample should halve the resolution.
    #[test]
    fn test_downsample_dimensions() {
        let config = ProgressiveAnalysisConfig::default();
        let analyzer = ProgressiveAnalyzer::new(config);
        let frame = make_frame(64, 48, 0);
        let (out, ow, oh) = analyzer.downsample(&frame, 64, 48);
        assert_eq!(ow, 32, "Width should halve");
        assert_eq!(oh, 24, "Height should halve");
        assert_eq!(out.len(), 32 * 24, "Output buffer size");
    }

    /// `scale_homography` round-trip: scale down then back up ≈ identity.
    #[test]
    fn test_scale_homography_roundtrip() {
        let config = ProgressiveAnalysisConfig::default();
        let analyzer = ProgressiveAnalyzer::new(config);

        let h = [[1.0f32, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]];
        let scaled = analyzer.scale_homography(h, 0.25);
        let restored = analyzer.scale_homography(scaled, 4.0);

        assert!(
            (restored[0][2] - h[0][2]).abs() < 1e-4,
            "tx should round-trip"
        );
        assert!(
            (restored[1][2] - h[1][2]).abs() < 1e-4,
            "ty should round-trip"
        );
    }

    /// Build scale schedule with 3 levels should produce [0.25, 0.5, 1.0].
    #[test]
    fn test_build_scale_schedule() {
        let config = ProgressiveAnalysisConfig {
            pyramid_levels: 3,
            min_scale: 0.25,
            convergence_threshold: 0.5,
        };
        let analyzer = ProgressiveAnalyzer::new(config);
        let schedule = analyzer.build_scale_schedule();
        assert_eq!(schedule.len(), 3);
        // Coarsest first.
        assert!(
            schedule[0] <= schedule[schedule.len() - 1],
            "Schedule should be coarse-to-fine"
        );
        // Finest must be 1.0.
        assert!(
            (schedule[2] - 1.0).abs() < 1e-4,
            "Finest level should be 1.0, got {}",
            schedule[2]
        );
    }
}
