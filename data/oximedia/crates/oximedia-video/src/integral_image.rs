//! Integral image (summed area table) for O(1) rectangular region queries.
//!
//! Given a grayscale frame, this module builds a summed area table that
//! allows computing the pixel sum, sum-of-squares, and variance for any
//! axis-aligned rectangle in O(1) time.
//!
//! # Algorithm
//!
//! Classic Crow (1984) summed area table with a (w+1)×(h+1) border.  For a
//! rectangle with corners (x0,y0) – (x1,y1) (exclusive ends):
//!
//! ```text
//! sum = S[y1][x1] - S[y0][x1] - S[y1][x0] + S[y0][x0]
//! ```
//!
//! Both `sum` and `sum_sq` tables are stored in the same layout so that
//! variance can be derived from `E[X²] – E[X]²`.

// -----------------------------------------------------------------------
// Public struct
// -----------------------------------------------------------------------

/// Integral image (summed area table) for fast rectangular region statistics.
///
/// `width` and `height` describe the *source* frame dimensions.  The internal
/// tables are `(width+1) × (height+1)`, enabling zero-overhead border clamping.
///
/// # Example
/// ```rust
/// use oximedia_video::integral_image::IntegralImage;
///
/// let frame: Vec<u8> = vec![1u8; 4 * 4]; // 4×4 all-ones image
/// let ii = IntegralImage::build(&frame, 4, 4);
/// assert_eq!(ii.rect_sum(0, 0, 4, 4), 16);
/// ```
#[derive(Debug, Clone)]
pub struct IntegralImage {
    /// Prefix sum of pixel values. Row-major, `(height+1) × (width+1)`.
    sum: Vec<u64>,
    /// Prefix sum of squared pixel values. Row-major, `(height+1) × (width+1)`.
    sum_sq: Vec<u64>,
    /// Source frame width in pixels.
    pub width: u32,
    /// Source frame height in pixels.
    pub height: u32,
}

impl IntegralImage {
    /// Build an integral image from a grayscale `u8` frame.
    ///
    /// `frame` must be at least `width × height` bytes in row-major order.
    /// Pixels beyond the end of the slice are treated as 0.
    ///
    /// Time complexity: O(W × H).  Space: O(W × H).
    pub fn build(frame: &[u8], w: u32, h: u32) -> Self {
        let stride = (w as usize) + 1;
        let total = stride * ((h as usize) + 1);

        let mut sum = vec![0u64; total];
        let mut sum_sq = vec![0u64; total];

        for row in 0..(h as usize) {
            for col in 0..(w as usize) {
                let px = frame.get(row * w as usize + col).copied().unwrap_or(0) as u64;
                let px_sq = px * px;

                // Four-corner update for the integral image:
                //   I[row+1][col+1] = px
                //                   + I[row][col+1]    (left neighbour)
                //                   + I[row+1][col]    (top neighbour)
                //                   - I[row][col]      (top-left — added twice)
                let idx = (row + 1) * stride + (col + 1);
                let top = row * stride + (col + 1);
                let left = (row + 1) * stride + col;
                let top_left = row * stride + col;

                sum[idx] = px + sum[top] + sum[left] - sum[top_left];
                sum_sq[idx] = px_sq + sum_sq[top] + sum_sq[left] - sum_sq[top_left];
            }
        }

        Self {
            sum,
            sum_sq,
            width: w,
            height: h,
        }
    }

    /// Compute the sum of pixels in the rectangle `[x0..x1, y0..y1]`
    /// (exclusive upper bound in both dimensions).
    ///
    /// Coordinates are automatically clamped to `[0, width]` / `[0, height]`.
    ///
    /// # Panics
    /// Does not panic; out-of-range coordinates are clamped.
    #[inline]
    pub fn rect_sum(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
        let x0 = (x0 as usize).min(self.width as usize);
        let y0 = (y0 as usize).min(self.height as usize);
        let x1 = (x1 as usize).min(self.width as usize);
        let y1 = (y1 as usize).min(self.height as usize);

        if x1 <= x0 || y1 <= y0 {
            return 0;
        }

        let stride = (self.width as usize) + 1;
        let br = y1 * stride + x1;
        let bl = y1 * stride + x0;
        let tr = y0 * stride + x1;
        let tl = y0 * stride + x0;

        self.sum[br] + self.sum[tl] - self.sum[bl] - self.sum[tr]
    }

    /// Compute the sum of squared pixels in the rectangle `[x0..x1, y0..y1]`
    /// (exclusive upper bound).
    ///
    /// Coordinates are clamped like [`Self::rect_sum`].
    #[inline]
    pub fn rect_sum_sq(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
        let x0 = (x0 as usize).min(self.width as usize);
        let y0 = (y0 as usize).min(self.height as usize);
        let x1 = (x1 as usize).min(self.width as usize);
        let y1 = (y1 as usize).min(self.height as usize);

        if x1 <= x0 || y1 <= y0 {
            return 0;
        }

        let stride = (self.width as usize) + 1;
        let br = y1 * stride + x1;
        let bl = y1 * stride + x0;
        let tr = y0 * stride + x1;
        let tl = y0 * stride + x0;

        self.sum_sq[br] + self.sum_sq[tl] - self.sum_sq[bl] - self.sum_sq[tr]
    }

    /// Compute the pixel variance in the rectangle `[x0..x1, y0..y1]`.
    ///
    /// Uses the identity `Var(X) = E[X²] − E[X]²`.  Returns `0.0` for an
    /// empty rectangle.
    ///
    /// Coordinates are clamped to the image bounds.
    pub fn rect_variance(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
        let x0c = (x0 as usize).min(self.width as usize);
        let y0c = (y0 as usize).min(self.height as usize);
        let x1c = (x1 as usize).min(self.width as usize);
        let y1c = (y1 as usize).min(self.height as usize);

        if x1c <= x0c || y1c <= y0c {
            return 0.0;
        }

        let n = ((x1c - x0c) * (y1c - y0c)) as f64;
        let s = self.rect_sum(x0, y0, x1, y1) as f64;
        let s2 = self.rect_sum_sq(x0, y0, x1, y1) as f64;

        // E[X²] − E[X]²
        (s2 / n) - (s / n).powi(2)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -------------------------------------------------------

    /// Brute-force pixel sum for a subregion.
    fn brute_sum(frame: &[u8], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> u64 {
        let mut s = 0u64;
        for row in y0..y1 {
            for col in x0..x1 {
                s += frame[row * w + col] as u64;
            }
        }
        s
    }

    /// Brute-force variance for a subregion.
    fn brute_variance(frame: &[u8], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let mut s = 0f64;
        let mut s2 = 0f64;
        let n = ((x1 - x0) * (y1 - y0)) as f64;
        for row in y0..y1 {
            for col in x0..x1 {
                let px = frame[row * w + col] as f64;
                s += px;
                s2 += px * px;
            }
        }
        (s2 / n) - (s / n).powi(2)
    }

    // 1. test_integral_image_sum_known — 4×4 all-ones; rect_sum(0,0,4,4) == 16
    #[test]
    fn test_integral_image_sum_known() {
        let frame = vec![1u8; 16]; // 4×4 all-ones
        let ii = IntegralImage::build(&frame, 4, 4);
        assert_eq!(ii.rect_sum(0, 0, 4, 4), 16, "4×4 all-ones sum should be 16");
    }

    // 2. test_integral_image_variance_uniform — uniform value; variance == 0
    #[test]
    fn test_integral_image_variance_uniform() {
        let frame = vec![100u8; 16]; // 4×4 all-100
        let ii = IntegralImage::build(&frame, 4, 4);
        let var = ii.rect_variance(0, 0, 4, 4);
        assert!(
            var.abs() < 1e-9,
            "uniform image must have variance == 0, got {var}"
        );
    }

    // 3. test_integral_image_variance_correct — [0,1,...,15]; compute expected by hand
    //
    // Image (row-major):
    //   Row 0: [0, 1, 2, 3]
    //   Row 1: [4, 5, 6, 7]
    //   Row 2: [8, 9, 10, 11]
    //   Row 3: [12, 13, 14, 15]
    //
    // E[X]  = sum(0..16) / 16 = 120 / 16 = 7.5
    // E[X²] = sum(i² for i in 0..16) / 16 = 1240 / 16 = 77.5
    //   (sum of squares 0..16: 0+1+4+9+16+25+36+49+64+81+100+121+144+169+196+225 = 1240)
    // Var   = 77.5 − 7.5² = 77.5 − 56.25 = 21.25
    #[test]
    fn test_integral_image_variance_correct() {
        let frame: Vec<u8> = (0u8..16u8).collect();
        let ii = IntegralImage::build(&frame, 4, 4);
        let var = ii.rect_variance(0, 0, 4, 4);
        assert!(
            (var - 21.25_f64).abs() < 1e-9,
            "expected variance 21.25, got {var}"
        );
    }

    // 4. test_integral_image_rect_sum — spot-check subregion queries against brute-force
    #[test]
    fn test_integral_image_rect_sum() {
        // 8×8 frame with value i % 64 at each position.
        let frame: Vec<u8> = (0u8..64u8).collect();
        let ii = IntegralImage::build(&frame, 8, 8);

        // Check multiple subregions against brute-force.
        let regions: &[(u32, u32, u32, u32)] = &[
            (0, 0, 4, 4),
            (2, 1, 6, 5),
            (0, 0, 8, 8),
            (4, 4, 8, 8),
            (1, 1, 3, 3),
            (0, 0, 1, 1),
            (7, 7, 8, 8),
        ];

        for &(x0, y0, x1, y1) in regions {
            let expected = brute_sum(
                &frame,
                8,
                x0 as usize,
                y0 as usize,
                x1 as usize,
                y1 as usize,
            );
            let got = ii.rect_sum(x0, y0, x1, y1);
            assert_eq!(
                got, expected,
                "rect_sum({x0},{y0},{x1},{y1}): expected {expected}, got {got}"
            );
        }
    }

    // 5. rect_sum on empty rectangle returns 0
    #[test]
    fn test_integral_image_empty_rect_returns_zero() {
        let frame = vec![42u8; 16];
        let ii = IntegralImage::build(&frame, 4, 4);
        assert_eq!(ii.rect_sum(2, 2, 2, 4), 0, "empty x-range must return 0");
        assert_eq!(ii.rect_sum(2, 2, 4, 2), 0, "empty y-range must return 0");
    }

    // 6. rect_sum_sq: 4×4 [0..16] — verify against brute force
    #[test]
    fn test_integral_image_sum_sq_correct() {
        let frame: Vec<u8> = (0u8..16u8).collect();
        let ii = IntegralImage::build(&frame, 4, 4);
        let expected_sq: u64 = (0u64..16u64).map(|i| i * i).sum();
        assert_eq!(
            ii.rect_sum_sq(0, 0, 4, 4),
            expected_sq,
            "sum_sq over full [0..16] frame should be {expected_sq}"
        );
    }

    // 7. rect_variance subregion matches brute-force
    #[test]
    fn test_integral_image_variance_subregion_matches_brute() {
        // 8×8 ramp frame
        let frame: Vec<u8> = (0u8..64u8).collect();
        let ii = IntegralImage::build(&frame, 8, 8);

        let regions: &[(u32, u32, u32, u32)] =
            &[(0, 0, 4, 4), (2, 2, 6, 6), (1, 3, 5, 7), (0, 0, 8, 8)];

        for &(x0, y0, x1, y1) in regions {
            let expected = brute_variance(
                &frame,
                8,
                x0 as usize,
                y0 as usize,
                x1 as usize,
                y1 as usize,
            );
            let got = ii.rect_variance(x0, y0, x1, y1);
            assert!(
                (got - expected).abs() < 1e-6,
                "rect_variance({x0},{y0},{x1},{y1}): expected {expected:.6}, got {got:.6}"
            );
        }
    }

    // 8. Out-of-bounds coordinates are clamped (no panic)
    #[test]
    fn test_integral_image_oob_clamped() {
        let frame = vec![50u8; 16];
        let ii = IntegralImage::build(&frame, 4, 4);
        // These exceed the image bounds — should clamp, not panic.
        let s = ii.rect_sum(0, 0, 100, 100);
        assert_eq!(s, 50 * 16, "out-of-bounds clamped to full image");
    }

    // 9. Single-pixel integral image
    #[test]
    fn test_integral_image_single_pixel() {
        let frame = vec![200u8];
        let ii = IntegralImage::build(&frame, 1, 1);
        assert_eq!(ii.rect_sum(0, 0, 1, 1), 200);
        assert_eq!(ii.rect_sum_sq(0, 0, 1, 1), 200 * 200);
        let var = ii.rect_variance(0, 0, 1, 1);
        assert!(var.abs() < 1e-9, "single pixel variance must be 0");
    }

    // 10. Width=1 column image
    #[test]
    fn test_integral_image_column_image() {
        let frame: Vec<u8> = vec![10, 20, 30, 40]; // 1×4
        let ii = IntegralImage::build(&frame, 1, 4);
        assert_eq!(ii.rect_sum(0, 0, 1, 4), 100, "sum of [10,20,30,40] = 100");
        assert_eq!(ii.rect_sum(0, 1, 1, 3), 50, "sum of [20,30] = 50");
    }
}
