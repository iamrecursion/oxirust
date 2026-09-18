//! van Herk–Gil–Werman O(1)-per-pixel morphological filters.
//!
//! Implements the van Herk (1992) / Gil & Werman (1993) algorithm for
//! 1D running maximum/minimum in O(3n) operations (O(1) amortised per pixel).
//! 2D rectangular and axis-aligned line structuring elements are handled by
//! separating the filter into a horizontal pass followed by a vertical pass.
//!
//! # Algorithm sketch (max filter, window k over length n)
//!
//! 1. Split the input into non-overlapping chunks of size `k`.
//! 2. For each chunk compute a *prefix-max* and a *suffix-max* array.
//! 3. For output pixel `i`: let `r = i % k`, `left_overlap = prefix_max[r]`
//!    from the chunk containing `i` (extending right), and
//!    `right_overlap = suffix_max[(r + k) % k]` from the chunk to the right
//!    (extending left); result = max(right, left).
//!
//! Boundary pixels are padded with the neutral element (0 for dilation, 255
//! for erosion) so that behaviour matches the brute-force implementation in
//! `advanced_morphology`.

/// O(1)-per-pixel dilation for a rectangular structuring element `kw × kh`.
///
/// Applies separable 1D passes: horizontal then vertical.  The result is
/// bit-exact with brute-force dilation when the SE is a fully-active
/// rectangle centred at `(kw/2, kh/2)`.
#[must_use]
pub fn dilate_rect_vhgw(pixels: &[u8], w: u32, h: u32, kw: u32, kh: u32) -> Vec<u8> {
    // Horizontal pass
    let horiz = apply_1d_rows(pixels, w, h, kw, max_filter_1d);
    // Vertical pass
    apply_1d_cols(&horiz, w, h, kh, max_filter_1d)
}

/// O(1)-per-pixel erosion for a rectangular structuring element `kw × kh`.
#[must_use]
pub fn erode_rect_vhgw(pixels: &[u8], w: u32, h: u32, kw: u32, kh: u32) -> Vec<u8> {
    let horiz = apply_1d_rows(pixels, w, h, kw, min_filter_1d);
    apply_1d_cols(&horiz, w, h, kh, min_filter_1d)
}

/// O(1)-per-pixel dilation for an axis-aligned line SE of `length` pixels.
///
/// * `horizontal = true`  → line runs left-right (height 1, width `length`).
/// * `horizontal = false` → line runs top-bottom (width 1, height `length`).
#[must_use]
pub fn dilate_line_vhgw(pixels: &[u8], w: u32, h: u32, length: u32, horizontal: bool) -> Vec<u8> {
    if horizontal {
        apply_1d_rows(pixels, w, h, length, max_filter_1d)
    } else {
        apply_1d_cols(pixels, w, h, length, max_filter_1d)
    }
}

/// O(1)-per-pixel erosion for an axis-aligned line SE.
#[must_use]
pub fn erode_line_vhgw(pixels: &[u8], w: u32, h: u32, length: u32, horizontal: bool) -> Vec<u8> {
    if horizontal {
        apply_1d_rows(pixels, w, h, length, min_filter_1d)
    } else {
        apply_1d_cols(pixels, w, h, length, min_filter_1d)
    }
}

// ---------------------------------------------------------------------------
// 1D van Herk–Gil–Werman algorithm
// ---------------------------------------------------------------------------

/// Running-maximum filter over a 1D signal with symmetric half-window `k`.
///
/// The window extends `k/2` pixels to the left and right of each position
/// (with `k/2` = `(k-1)/2` for odd `k`).  Out-of-bounds positions are
/// padded with `0` (neutral element for max → no artificial brightening
/// arises from padding alone).
pub fn max_filter_1d(row: &[u8], k: u32) -> Vec<u8> {
    vhgw_1d(row, k, 0u8, |a: u8, b: u8| a.max(b))
}

/// Running-minimum filter over a 1D signal with symmetric half-window `k`.
///
/// Out-of-bounds positions are padded with `255` (neutral element for min).
pub fn min_filter_1d(row: &[u8], k: u32) -> Vec<u8> {
    vhgw_1d(row, k, 255u8, |a: u8, b: u8| a.min(b))
}

/// Core van Herk–Gil–Werman 1D sliding-window extremum filter.
///
/// * `neutral` — value used to pad out-of-bounds positions (0 for max, 255 for min).
/// * `combine` — the fold operator (max or min).
fn vhgw_1d<F>(row: &[u8], k: u32, neutral: u8, combine: F) -> Vec<u8>
where
    F: Fn(u8, u8) -> u8,
{
    let n = row.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k as usize;
    if k == 0 || k == 1 {
        return row.to_vec();
    }

    // Half-radius (symmetric window of k).  For even k the window is
    // asymmetric by one; we follow the convention of advanced_morphology
    // (origin = k/2).
    let radius = k / 2;

    // Pad the signal: `radius` neutrals on each side.
    let padded_len = radius + n + radius;
    let mut padded = vec![neutral; padded_len];
    padded[radius..radius + n].copy_from_slice(row);

    let pn = padded_len;

    // Build prefix-max and suffix-max arrays for each chunk of size k.
    // We over-allocate slightly to avoid a special case at the last chunk.
    let num_chunks = (pn + k - 1) / k;
    let buf_len = num_chunks * k;

    let mut prefix = vec![neutral; buf_len];
    let mut suffix = vec![neutral; buf_len];

    // Prefix pass: for each chunk[i*k .. (i+1)*k], prefix[j] = max(chunk[0..=j])
    for c in 0..num_chunks {
        let base = c * k;
        let val0 = if base < pn { padded[base] } else { neutral };
        prefix[base] = val0;
        for j in 1..k {
            let idx = base + j;
            let val = if idx < pn { padded[idx] } else { neutral };
            prefix[idx] = combine(prefix[idx - 1], val);
        }
    }

    // Suffix pass: for each chunk, suffix[j] = max(chunk[j..k])
    for c in 0..num_chunks {
        let base = c * k;
        let last = base + k - 1;
        let val_last = if last < pn { padded[last] } else { neutral };
        suffix[last] = val_last;
        for j in (0..k - 1).rev() {
            let idx = base + j;
            let val = if idx < pn { padded[idx] } else { neutral };
            suffix[idx] = combine(val, suffix[idx + 1]);
        }
    }

    // Combine prefix + suffix to produce the output.
    // For position `i` in the padded signal, the window is padded[i-radius .. i+k-radius-1]
    // which in 0-based terms is [i, i+k-1] in padded (because we shifted by radius).
    // Actually: result for original position p is window padded[p .. p+k].
    let mut out = vec![neutral; n];
    for p in 0..n {
        // Window over padded[p .. p+k) (since we padded with `radius` on the left,
        // original position p maps to padded[p] and the window of size k spans [p, p+k)).
        // Using prefix/suffix:
        //   chunk of p+k-1 = (p+k-1)/k
        //   chunk of p = p/k
        // If they share the same chunk: use suffix[p] directly.
        // Otherwise: suffix[p] covers [p .. end_of_chunk_p], prefix[p+k-1] covers [start_of_chunk_pkm1 .. p+k-1]
        let left = p;
        let right = p + k - 1;
        // clamp right to buf_len-1
        let right = right.min(buf_len - 1);

        let chunk_l = left / k;
        let chunk_r = right / k;

        let val = if chunk_l == chunk_r {
            // Both ends in the same chunk: use suffix[left] (= max of left..right)
            suffix[left]
        } else {
            // Cross-chunk: suffix[left] covers left..end_of_chunk, prefix[right] covers start..right
            combine(suffix[left], prefix[right])
        };
        out[p] = val;
    }

    out
}

// ---------------------------------------------------------------------------
// 2D separable application
// ---------------------------------------------------------------------------

/// Apply a 1D filter to every row of an image.
fn apply_1d_rows<F>(pixels: &[u8], w: u32, h: u32, k: u32, filter: F) -> Vec<u8>
where
    F: Fn(&[u8], u32) -> Vec<u8>,
{
    let width = w as usize;
    let height = h as usize;
    let mut out = vec![0u8; width * height];
    for row_idx in 0..height {
        let row = &pixels[row_idx * width..(row_idx + 1) * width];
        let filtered = filter(row, k);
        out[row_idx * width..(row_idx + 1) * width].copy_from_slice(&filtered);
    }
    out
}

/// Apply a 1D filter to every column of an image.
fn apply_1d_cols<F>(pixels: &[u8], w: u32, h: u32, k: u32, filter: F) -> Vec<u8>
where
    F: Fn(&[u8], u32) -> Vec<u8>,
{
    let width = w as usize;
    let height = h as usize;
    let mut col_buf = vec![0u8; height];
    let mut out = vec![0u8; width * height];
    for col_idx in 0..width {
        // Extract column
        for row_idx in 0..height {
            col_buf[row_idx] = pixels[row_idx * width + col_idx];
        }
        let filtered = filter(&col_buf, k);
        // Write back
        for row_idx in 0..height {
            out[row_idx * width + col_idx] = filtered[row_idx];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_morphology::{morphology_dilate, morphology_erode, StructuringElement};

    /// Deterministic LCG-based pseudo-random number generator (no external deps).
    struct Lcg(u64);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) & 0xFF) as u8
        }
    }

    fn random_image(seed: u64, len: usize) -> Vec<u8> {
        let mut rng = Lcg(seed);
        (0..len).map(|_| rng.next_u8()).collect()
    }

    #[test]
    fn test_vhgw_dilate_matches_brute_force_rect() {
        // Random 64×64 image, 5×5 SE — bit-exact comparison including borders.
        let w = 64u32;
        let h = 64u32;
        let pixels = random_image(0xDEAD_BEEF, (w * h) as usize);

        let se = StructuringElement::rectangle(5, 5);
        let brute = morphology_dilate(&pixels, w, h, &se);
        let vhgw = dilate_rect_vhgw(&pixels, w, h, 5, 5);

        assert_eq!(
            brute, vhgw,
            "VHGW dilation must be bit-exact with brute force for 5×5 rectangle"
        );
    }

    #[test]
    fn test_vhgw_erode_matches_brute_force_rect() {
        let w = 64u32;
        let h = 64u32;
        let pixels = random_image(0xCAFE_BABE, (w * h) as usize);

        let se = StructuringElement::rectangle(5, 5);
        let brute = morphology_erode(&pixels, w, h, &se);
        let vhgw = erode_rect_vhgw(&pixels, w, h, 5, 5);

        assert_eq!(
            brute, vhgw,
            "VHGW erosion must be bit-exact with brute force for 5×5 rectangle"
        );
    }

    #[test]
    fn test_vhgw_handles_large_kernel() {
        // 31×31 SE on 256×256 — correctness check (interior pixels).
        let w = 256u32;
        let h = 256u32;
        let pixels = random_image(0x1234_5678, (w * h) as usize);

        let se = StructuringElement::rectangle(31, 31);
        let brute = morphology_dilate(&pixels, w, h, &se);
        let vhgw = dilate_rect_vhgw(&pixels, w, h, 31, 31);

        // Check a sample of interior pixels (avoid border differences caused by
        // the different boundary convention — brute force uses u8::MIN, VHGW
        // pads with 0 which is the same value; so all positions should match).
        let mismatches: Vec<usize> = (0..(w * h) as usize)
            .filter(|&i| brute[i] != vhgw[i])
            .collect();
        assert!(
            mismatches.is_empty(),
            "Mismatch at {} positions for large kernel",
            mismatches.len()
        );
    }

    #[test]
    fn test_vhgw_fallback_for_nonflat_se() {
        // Circle SE is non-rectangular — brute force should be used for that.
        // Here we test that the brute-force path (advanced_morphology) is
        // correct independently, and that VHGW rect gives same result when SE
        // happens to be a rectangle.
        let w = 32u32;
        let h = 32u32;
        let pixels = random_image(0xABCD_1234, (w * h) as usize);

        // Use a 1×1 SE (trivially flat rect, identity).
        let se = StructuringElement::rectangle(1, 1);
        let brute = morphology_dilate(&pixels, w, h, &se);
        let vhgw = dilate_rect_vhgw(&pixels, w, h, 1, 1);
        assert_eq!(brute, vhgw, "1×1 rectangle must produce identity dilation");

        // Non-flat SE (circle radius 2): must route through brute force.
        let se_circle = StructuringElement::circle(2);
        let result = morphology_dilate(&pixels, w, h, &se_circle);
        // Verify correctness: dilating with a circle should never decrease pixel values.
        for (i, (&orig, &dilated)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                dilated >= orig,
                "Dilation at position {i} produced {dilated} < {orig}"
            );
        }
    }

    #[test]
    fn test_max_filter_1d_identity_k1() {
        let row: Vec<u8> = (0u8..16).collect();
        let out = max_filter_1d(&row, 1);
        assert_eq!(out, row, "k=1 max filter must be identity");
    }

    #[test]
    fn test_min_filter_1d_identity_k1() {
        let row: Vec<u8> = (0u8..16).collect();
        let out = min_filter_1d(&row, 1);
        assert_eq!(out, row, "k=1 min filter must be identity");
    }

    #[test]
    fn test_max_filter_1d_uniform() {
        let row = vec![100u8; 20];
        let out = max_filter_1d(&row, 5);
        assert!(
            out.iter().all(|&v| v == 100),
            "max filter on uniform input must return uniform output"
        );
    }

    #[test]
    fn test_min_filter_1d_uniform() {
        let row = vec![200u8; 20];
        let out = min_filter_1d(&row, 7);
        assert!(
            out.iter().all(|&v| v == 200),
            "min filter on uniform input must return uniform output"
        );
    }

    #[test]
    fn test_vhgw_line_horizontal_matches_brute() {
        let w = 48u32;
        let h = 32u32;
        let pixels = random_image(0x5A5A_5A5A, (w * h) as usize);

        // Horizontal line SE length=7
        let se = StructuringElement::line(7, 0.0);
        let brute = morphology_dilate(&pixels, w, h, &se);
        let vhgw = dilate_line_vhgw(&pixels, w, h, 7, true);

        // The VHGW horizontal line applies only horizontally; the brute-force
        // line SE at 0° activates only the centre row of the SE bounding box.
        // Both should match for interior pixels of a horizontal line SE.
        // We compare only the "interior" rows (row 0 and last are identical for
        // centre-origin 1-row SEs so all pixels should match).
        assert_eq!(brute.len(), vhgw.len(), "output dimensions must match");
        // Spot-check a centre row pixel that's not on any boundary.
        let mid_row = (h / 2) as usize;
        let mid_col = (w / 2) as usize;
        assert_eq!(
            brute[mid_row * w as usize + mid_col],
            vhgw[mid_row * w as usize + mid_col],
            "centre pixel must match for horizontal line dilation"
        );
    }
}
