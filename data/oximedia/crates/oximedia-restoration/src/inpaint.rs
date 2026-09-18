//! Content-aware inpainting using a simplified PatchMatch algorithm.
//!
//! Inpainting fills damaged or missing image regions (indicated by a boolean
//! `mask`) by synthesising texture from the surrounding undamaged area.  The
//! algorithm is based on the Approximate Nearest-Neighbour (ANN) field
//! constructed by Barnes et al. "PatchMatch: A Randomized Correspondence
//! Algorithm for Structural Image Editing" (SIGGRAPH 2009).
//!
//! # Algorithm outline
//!
//! 1. **Initialisation** – for every masked pixel, choose a random source
//!    patch from the unmasked region.
//! 2. **Propagation** – in alternating forward/backward scan passes, propagate
//!    good matches from neighbours.
//! 3. **Random search** – from the current best-match position, search random
//!    candidate offsets at exponentially decreasing radii.
//! 4. **Reconstruction** – fill each masked pixel with the weighted average of
//!    all patches that cover it.  Each contributing patch has weight inversely
//!    proportional to its SSD (sum-of-squared differences) distance.  A
//!    Gaussian-weighted blend over `blend_radius` smooths seam discontinuities.
//!
//! # Example
//!
//! ```rust
//! use oximedia_restoration::inpaint::{inpaint_patchmatch, InpaintConfig};
//!
//! let w = 10u32;
//! let h = 10u32;
//! let image: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
//! let mask = vec![false; (w * h) as usize];
//! let cfg = InpaintConfig::default();
//! let restored = inpaint_patchmatch(&image, &mask, w, h, &cfg);
//! assert_eq!(restored.len(), image.len());
//! ```

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// ─── Public API ───────────────────────────────────────────────────────────────

/// Configuration for PatchMatch-based inpainting.
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    /// Patch half-size in pixels; the patch window is `(2*patch_size+1)²`.
    /// Default: 3.
    pub patch_size: u32,
    /// Number of PatchMatch propagation+search iterations.  Default: 5.
    pub iterations: u32,
    /// Blending radius for seam smoothing.  Each reconstructed pixel is the
    /// weighted average of up to `(2*blend_radius+1)²` overlapping patches.
    /// Default: 1.
    pub blend_radius: u32,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            patch_size: 3,
            iterations: 5,
            blend_radius: 1,
        }
    }
}

/// Inpaint masked pixels using a simplified PatchMatch algorithm.
///
/// # Parameters
/// - `image` – packed RGB bytes (`[R,G,B, R,G,B, …]`), length `w * h * 3`.
/// - `mask`  – `true` at pixels that need inpainting, length `w * h`.
/// - `w`, `h` – image dimensions.
/// - `cfg`   – algorithm configuration.
///
/// # Returns
/// A new `Vec<u8>` of length `w * h * 3` with masked pixels filled in.
///
/// # Panics
/// Panics if `image.len() != w * h * 3` or `mask.len() != w * h`.
pub fn inpaint_patchmatch(
    image: &[u8],
    mask: &[bool],
    w: u32,
    h: u32,
    cfg: &InpaintConfig,
) -> Vec<u8> {
    let n_px = (w as usize) * (h as usize);
    assert_eq!(image.len(), n_px * 3, "image.len() must equal w * h * 3");
    assert_eq!(mask.len(), n_px, "mask.len() must equal w * h");

    // If nothing is masked, return a copy of the input immediately.
    if !mask.iter().any(|&m| m) {
        return image.to_vec();
    }

    let ctx = InpaintContext::new(image, mask, w, h, cfg);
    ctx.run()
}

// ─── Internal implementation ──────────────────────────────────────────────────

/// All state needed during a single inpainting run.
struct InpaintContext<'a> {
    image: &'a [u8],
    mask: &'a [bool],
    w: usize,
    h: usize,
    cfg: &'a InpaintConfig,
    /// Nearest-neighbour field: for each pixel (including masked), stores
    /// the (source_row, source_col) of the best matching patch centre.
    /// Source patches are always drawn from unmasked regions.
    nnf_row: Vec<i32>,
    nnf_col: Vec<i32>,
    /// SSD distance of the current best patch for each pixel.
    nnf_dist: Vec<f32>,
}

impl<'a> InpaintContext<'a> {
    fn new(image: &'a [u8], mask: &'a [bool], w: u32, h: u32, cfg: &'a InpaintConfig) -> Self {
        let w = w as usize;
        let h = h as usize;
        let n = w * h;

        // Build list of unmasked source pixels (valid patch centres).
        let ps = cfg.patch_size as usize;
        let unmasked: Vec<(usize, usize)> = (0..h)
            .flat_map(|r| {
                (0..w).filter_map(move |c| {
                    // A valid source must itself be unmasked, and all pixels in
                    // its patch window must also be unmasked.
                    if !mask[r * w + c] && is_patch_clean(mask, r, c, w, h, ps) {
                        Some((r, c))
                    } else {
                        None
                    }
                })
            })
            .collect();

        // If no fully-clean patches exist, fall back to any unmasked pixel.
        let unmasked = if unmasked.is_empty() {
            (0..n)
                .filter(|&i| !mask[i])
                .map(|i| (i / w, i % w))
                .collect()
        } else {
            unmasked
        };

        // Initialise NNF with random assignments for masked pixels.
        // Unmasked pixels get identity mapping (themselves) with dist = 0.
        let mut rng = SmallRng::seed_from_u64(0x4f786d6564);
        let mut nnf_row = vec![0i32; n];
        let mut nnf_col = vec![0i32; n];
        let mut nnf_dist = vec![f32::MAX; n];

        let n_src = unmasked.len();
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                if mask[idx] {
                    // Random source from unmasked pool
                    let src = unmasked[rng.random_range(0..n_src)];
                    nnf_row[idx] = src.0 as i32;
                    nnf_col[idx] = src.1 as i32;
                    nnf_dist[idx] = f32::MAX; // will be evaluated lazily
                } else {
                    nnf_row[idx] = r as i32;
                    nnf_col[idx] = c as i32;
                    nnf_dist[idx] = 0.0;
                }
            }
        }

        Self {
            image,
            mask,
            w,
            h,
            cfg,
            nnf_row,
            nnf_col,
            nnf_dist,
        }
    }

    /// Execute the PatchMatch iterations and reconstruct the output image.
    fn run(mut self) -> Vec<u8> {
        let ps = self.cfg.patch_size as usize;
        let iterations = self.cfg.iterations as usize;

        // ── Evaluate initial SSD for masked pixels ────────────────────────────
        for r in 0..self.h {
            for c in 0..self.w {
                let idx = r * self.w + c;
                if self.mask[idx] && self.nnf_dist[idx] == f32::MAX {
                    let sr = self.nnf_row[idx] as usize;
                    let sc = self.nnf_col[idx] as usize;
                    self.nnf_dist[idx] = self.patch_ssd(r, c, sr, sc, ps);
                }
            }
        }

        // ── Alternating PatchMatch iterations ────────────────────────────────
        let mut rng = SmallRng::seed_from_u64(0xb10c_deada);
        for iter in 0..iterations {
            if iter % 2 == 0 {
                // Forward pass: top-left → bottom-right
                for r in 0..self.h {
                    for c in 0..self.w {
                        if self.mask[r * self.w + c] {
                            self.propagate_and_search(r, c, ps, true, &mut rng);
                        }
                    }
                }
            } else {
                // Backward pass: bottom-right → top-left
                for r in (0..self.h).rev() {
                    for c in (0..self.w).rev() {
                        if self.mask[r * self.w + c] {
                            self.propagate_and_search(r, c, ps, false, &mut rng);
                        }
                    }
                }
            }
        }

        // ── Reconstruct: weighted average of best patches ─────────────────────
        self.reconstruct()
    }

    /// Propagation + random search for pixel `(row, col)`.
    fn propagate_and_search(
        &mut self,
        row: usize,
        col: usize,
        ps: usize,
        forward: bool,
        rng: &mut SmallRng,
    ) {
        let idx = row * self.w + col;

        // ── Propagation from left/top (forward) or right/bottom (backward) ───
        let neighbours: [(i32, i32); 2] = if forward {
            [(-1, 0), (0, -1)]
        } else {
            [(1, 0), (0, 1)]
        };

        for &(dr, dc) in &neighbours {
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if nr < 0 || nr >= self.h as i32 || nc < 0 || nc >= self.w as i32 {
                continue;
            }
            let nidx = nr as usize * self.w + nc as usize;
            // Candidate: neighbour's source offset, shifted by (-dr, -dc)
            let cand_r = self.nnf_row[nidx] - dr;
            let cand_c = self.nnf_col[nidx] - dc;
            if cand_r < 0 || cand_r >= self.h as i32 || cand_c < 0 || cand_c >= self.w as i32 {
                continue;
            }
            let cr = cand_r as usize;
            let cc = cand_c as usize;
            // Source must be unmasked
            if self.mask[cr * self.w + cc] {
                continue;
            }
            let d = self.patch_ssd(row, col, cr, cc, ps);
            if d < self.nnf_dist[idx] {
                self.nnf_row[idx] = cand_r;
                self.nnf_col[idx] = cand_c;
                self.nnf_dist[idx] = d;
            }
        }

        // ── Random search at exponentially shrinking radii ────────────────────
        let max_radius = self.w.max(self.h) as f32;
        let mut radius = max_radius;
        while radius >= 1.0 {
            let r_int = radius as i32;
            let lo_r = (self.nnf_row[idx] - r_int).max(0) as usize;
            let hi_r = (self.nnf_row[idx] + r_int).min(self.h as i32 - 1) as usize;
            let lo_c = (self.nnf_col[idx] - r_int).max(0) as usize;
            let hi_c = (self.nnf_col[idx] + r_int).min(self.w as i32 - 1) as usize;

            let cand_r = if lo_r < hi_r {
                rng.random_range(lo_r..=hi_r)
            } else {
                lo_r
            };
            let cand_c = if lo_c < hi_c {
                rng.random_range(lo_c..=hi_c)
            } else {
                lo_c
            };

            if !self.mask[cand_r * self.w + cand_c] {
                let d = self.patch_ssd(row, col, cand_r, cand_c, ps);
                if d < self.nnf_dist[idx] {
                    self.nnf_row[idx] = cand_r as i32;
                    self.nnf_col[idx] = cand_c as i32;
                    self.nnf_dist[idx] = d;
                }
            }

            radius *= 0.5;
        }
    }

    /// Compute the patch SSD between query patch centred at `(qr, qc)` and
    /// source patch centred at `(sr, sc)`.  Only unmasked pixels in the query
    /// patch contribute; if no pixels contribute, return `f32::MAX`.
    fn patch_ssd(&self, qr: usize, qc: usize, sr: usize, sc: usize, ps: usize) -> f32 {
        let qr = qr as i32;
        let qc = qc as i32;
        let sr = sr as i32;
        let sc = sc as i32;
        let ps = ps as i32;
        let w = self.w as i32;
        let h = self.h as i32;

        let mut ssd = 0.0f32;
        let mut count = 0u32;

        for dr in -ps..=ps {
            for dc in -ps..=ps {
                let qrow = qr + dr;
                let qcol = qc + dc;
                let srow = sr + dr;
                let scol = sc + dc;

                if qrow < 0 || qrow >= h || qcol < 0 || qcol >= w {
                    continue;
                }
                if srow < 0 || srow >= h || scol < 0 || scol >= w {
                    continue;
                }

                let qi = qrow as usize * self.w + qcol as usize;
                // Only compare unmasked query pixels
                if self.mask[qi] {
                    continue;
                }
                let si = srow as usize * self.w + scol as usize;

                for ch in 0..3 {
                    let q = self.image[qi * 3 + ch] as f32;
                    let s = self.image[si * 3 + ch] as f32;
                    ssd += (q - s) * (q - s);
                }
                count += 1;
            }
        }

        if count == 0 {
            f32::MAX
        } else {
            ssd / count as f32
        }
    }

    /// Reconstruct the output: unmasked pixels are copied verbatim; masked
    /// pixels are filled by accumulating weighted patch contributions, then
    /// blending over `blend_radius` using a simple box average.
    fn reconstruct(self) -> Vec<u8> {
        let n = self.w * self.h;
        let blend_r = self.cfg.blend_radius as usize;

        // Accumulate weighted colour sums over each masked pixel.
        // For each masked destination pixel `(r,c)`, we iterate over every
        // source pixel `(sr,sc)` whose patch overlaps `(r,c)`, weight by
        // `exp(-dist)` and accumulate.
        let mut acc_rgb = vec![[0.0f32; 3]; n];
        let mut acc_w = vec![0.0f32; n];

        for row in 0..self.h {
            for col in 0..self.w {
                let idx = row * self.w + col;
                if !self.mask[idx] {
                    continue;
                }

                let sr = self.nnf_row[idx] as usize;
                let sc = self.nnf_col[idx] as usize;
                let dist = self.nnf_dist[idx];

                // Weight by inverse distance (clamp for numerics)
                let weight = (-dist / (255.0 * 255.0 * 3.0 + 1.0)).exp().max(1e-6);

                // The best source patch covers pixels in [sr-br..sr+br, sc-br..sc+br].
                // Spread contribution over the blend neighbourhood.
                let br = blend_r as i32;
                for dr in -br..=br {
                    for dc in -br..=br {
                        let dst_r = row as i32 + dr;
                        let dst_c = col as i32 + dc;
                        if dst_r < 0
                            || dst_r >= self.h as i32
                            || dst_c < 0
                            || dst_c >= self.w as i32
                        {
                            continue;
                        }
                        let dst_idx = dst_r as usize * self.w + dst_c as usize;
                        if !self.mask[dst_idx] {
                            continue;
                        }
                        // Source pixel at corresponding offset
                        let src_r = (sr as i32 + dr).clamp(0, self.h as i32 - 1) as usize;
                        let src_c = (sc as i32 + dc).clamp(0, self.w as i32 - 1) as usize;
                        let src_idx = src_r * self.w + src_c;

                        for ch in 0..3 {
                            acc_rgb[dst_idx][ch] += weight * self.image[src_idx * 3 + ch] as f32;
                        }
                        acc_w[dst_idx] += weight;
                    }
                }
            }
        }

        // Build output: copy unmasked pixels; write averaged colour for masked.
        let mut out = image_to_vec(self.image);
        for r in 0..self.h {
            for c in 0..self.w {
                let idx = r * self.w + c;
                if !self.mask[idx] {
                    continue;
                }
                let w_sum = acc_w[idx];
                if w_sum > 0.0 {
                    for ch in 0..3 {
                        out[idx * 3 + ch] =
                            (acc_rgb[idx][ch] / w_sum).round().clamp(0.0, 255.0) as u8;
                    }
                } else {
                    // Fallback: copy from NNF source
                    let sr = self.nnf_row[idx].clamp(0, self.h as i32 - 1) as usize;
                    let sc = self.nnf_col[idx].clamp(0, self.w as i32 - 1) as usize;
                    let si = sr * self.w + sc;
                    out[idx * 3] = self.image[si * 3];
                    out[idx * 3 + 1] = self.image[si * 3 + 1];
                    out[idx * 3 + 2] = self.image[si * 3 + 2];
                }
            }
        }

        out
    }
}

// ─── Small helpers ────────────────────────────────────────────────────────────

/// Returns `true` if all pixels in the `ps`-radius patch centred at `(r,c)`
/// are unmasked.
fn is_patch_clean(mask: &[bool], r: usize, c: usize, w: usize, h: usize, ps: usize) -> bool {
    let r = r as i32;
    let c = c as i32;
    let ps = ps as i32;
    let w = w as i32;
    let h = h as i32;
    for dr in -ps..=ps {
        for dc in -ps..=ps {
            let nr = r + dr;
            let nc = c + dc;
            if nr < 0 || nr >= h || nc < 0 || nc >= w {
                continue;
            }
            if mask[nr as usize * w as usize + nc as usize] {
                return false;
            }
        }
    }
    true
}

/// Clone the image bytes into a new `Vec<u8>`.
fn image_to_vec(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty mask: output must equal input exactly.
    #[test]
    fn test_inpaint_no_mask() {
        let w = 8u32;
        let h = 8u32;
        let image: Vec<u8> = (0..(w * h * 3) as usize)
            .map(|i| (i * 7 % 256) as u8)
            .collect();
        let mask = vec![false; (w * h) as usize];
        let cfg = InpaintConfig::default();
        let out = inpaint_patchmatch(&image, &mask, w, h, &cfg);
        assert_eq!(out, image, "Empty mask should produce identical output");
    }

    /// A 3×3 hole in the centre of a solid-colour image should be filled with
    /// that same colour (within ±2 LSB to allow rounding).
    #[test]
    fn test_inpaint_small_hole() {
        let w = 16u32;
        let h = 16u32;
        let fill: [u8; 3] = [120, 80, 200];
        // Solid colour image
        let image: Vec<u8> = (0..(w * h) as usize)
            .flat_map(|_| fill.iter().cloned())
            .collect();
        // Mask the 3×3 centre
        let mut mask = vec![false; (w * h) as usize];
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        for r in cy - 1..=cy + 1 {
            for c in cx - 1..=cx + 1 {
                mask[r * w as usize + c] = true;
            }
        }

        let cfg = InpaintConfig::default();
        let out = inpaint_patchmatch(&image, &mask, w, h, &cfg);

        // Check that all masked pixels are filled with the correct colour
        for r in cy - 1..=cy + 1 {
            for c in cx - 1..=cx + 1 {
                let idx = r * w as usize + c;
                for ch in 0..3 {
                    let got = out[idx * 3 + ch];
                    let exp = fill[ch];
                    assert!(
                        got.abs_diff(exp) <= 2,
                        "Pixel ({r},{c}) ch{ch}: expected ~{exp} got {got}"
                    );
                }
            }
        }
    }

    /// Output must have same total byte count as input.
    #[test]
    fn test_inpaint_output_dimensions() {
        let w = 24u32;
        let h = 18u32;
        let n = (w * h) as usize;
        let image = vec![128u8; n * 3];
        let mask = vec![false; n];
        let cfg = InpaintConfig::default();
        let out = inpaint_patchmatch(&image, &mask, w, h, &cfg);
        assert_eq!(out.len(), n * 3, "Output length must equal w * h * 3");
    }
}
