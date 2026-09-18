//! Optical flow computation for frame-to-frame alignment.
//!
//! Provides both sparse (block-matching) and dense (Farneback polynomial
//! expansion) optical flow algorithms suitable for estimating camera motion
//! between consecutive video frames.

use crate::{AlignError, AlignResult};

/// A single optical flow vector at one block location.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub struct FlowVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
    /// Normalised confidence score in [0, 1].
    pub confidence: f32,
}

impl FlowVector {
    /// Create a new flow vector.
    #[must_use]
    pub fn new(dx: f32, dy: f32, confidence: f32) -> Self {
        Self { dx, dy, confidence }
    }

    /// Euclidean magnitude of the vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Angle of the vector in radians (atan2 convention).
    #[must_use]
    pub fn angle_radians(&self) -> f32 {
        self.dy.atan2(self.dx)
    }
}

/// A dense optical flow field covering a frame on a block grid.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FlowField {
    /// Flat storage, row-major over block rows × block columns.
    pub vectors: Vec<FlowVector>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Block size in pixels (each cell in the grid covers this many pixels).
    pub block_size: u32,
}

impl FlowField {
    /// Create a new, zeroed flow field.
    #[must_use]
    pub fn new(width: u32, height: u32, block_size: u32) -> Self {
        let cols = cols(width, block_size);
        let rows = rows(height, block_size);
        Self {
            vectors: vec![FlowVector::new(0.0, 0.0, 0.0); (cols * rows) as usize],
            width,
            height,
            block_size,
        }
    }

    /// Set the flow vector at block position `(x, y)`.
    ///
    /// `x` is the block column, `y` is the block row.
    /// Does nothing if the indices are out of range.
    pub fn set(&mut self, x: u32, y: u32, flow: FlowVector) {
        let c = cols(self.width, self.block_size);
        let r = rows(self.height, self.block_size);
        if x < c && y < r {
            self.vectors[(y * c + x) as usize] = flow;
        }
    }

    /// Get the flow vector at block position `(x, y)`.
    ///
    /// Returns `None` if the indices are out of range.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<&FlowVector> {
        let c = cols(self.width, self.block_size);
        let r = rows(self.height, self.block_size);
        if x < c && y < r {
            Some(&self.vectors[(y * c + x) as usize])
        } else {
            None
        }
    }

    /// Number of block columns.
    #[must_use]
    pub fn block_cols(&self) -> u32 {
        cols(self.width, self.block_size)
    }

    /// Number of block rows.
    #[must_use]
    pub fn block_rows(&self) -> u32 {
        rows(self.height, self.block_size)
    }

    /// Average magnitude of all flow vectors weighted by confidence.
    ///
    /// Returns 0.0 when the field is empty or all confidences are zero.
    #[must_use]
    pub fn avg_magnitude(&self) -> f32 {
        let mut weighted_sum = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for v in &self.vectors {
            weighted_sum += v.magnitude() * v.confidence;
            weight_total += v.confidence;
        }

        if weight_total == 0.0 {
            return 0.0;
        }
        weighted_sum / weight_total
    }

    /// Dominant (average) direction as `(dx, dy)`, confidence-weighted.
    ///
    /// Returns `(0.0, 0.0)` when the field is empty or all confidences are zero.
    #[must_use]
    pub fn dominant_direction(&self) -> (f32, f32) {
        let mut sum_dx = 0.0_f32;
        let mut sum_dy = 0.0_f32;
        let mut weight_total = 0.0_f32;

        for v in &self.vectors {
            sum_dx += v.dx * v.confidence;
            sum_dy += v.dy * v.confidence;
            weight_total += v.confidence;
        }

        if weight_total == 0.0 {
            return (0.0, 0.0);
        }
        (sum_dx / weight_total, sum_dy / weight_total)
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────────

fn cols(width: u32, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    width.div_ceil(block_size)
}

fn rows(height: u32, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    height.div_ceil(block_size)
}

/// Compute sum of squared differences between two equal-length byte slices.
#[must_use]
pub fn sum_squared_diff(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u64
        })
        .sum()
}

/// Compute a dense optical flow field using block matching (exhaustive search).
///
/// # Arguments
///
/// * `prev` – Previous frame pixels (grayscale, row-major, length = `width * height`).
/// * `curr` – Current frame pixels (same layout).
/// * `width` / `height` – Frame dimensions.
/// * `block_size` – Size of the matching block in pixels.
/// * `search_range` – Maximum displacement to search in each direction (pixels).
///
/// # Returns
///
/// A [`FlowField`] with one vector per block.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn block_match_flow(
    prev: &[u8],
    curr: &[u8],
    width: u32,
    height: u32,
    block_size: u32,
    search_range: i32,
) -> FlowField {
    let mut field = FlowField::new(width, height, block_size);

    let bsize = block_size as i32;
    let w = width as i32;
    let h = height as i32;

    for by in 0..field.block_rows() {
        for bx in 0..field.block_cols() {
            let px0 = (bx * block_size) as i32;
            let py0 = (by * block_size) as i32;

            let mut best_ssd = u64::MAX;
            let mut best_dx = 0_i32;
            let mut best_dy = 0_i32;

            // Exhaustive search over the search window
            for dy in -search_range..=search_range {
                for dx in -search_range..=search_range {
                    let cx0 = px0 + dx;
                    let cy0 = py0 + dy;

                    // Skip if block goes out of bounds
                    if cx0 < 0 || cy0 < 0 || cx0 + bsize > w || cy0 + bsize > h {
                        continue;
                    }

                    // Accumulate SSD
                    let mut ssd = 0_u64;
                    for row in 0..bsize {
                        let prev_row_start = (py0 + row) * w + px0;
                        let curr_row_start = (cy0 + row) * w + cx0;

                        let p_row =
                            &prev[prev_row_start as usize..(prev_row_start + bsize) as usize];
                        let c_row =
                            &curr[curr_row_start as usize..(curr_row_start + bsize) as usize];

                        ssd += sum_squared_diff(p_row, c_row);
                    }

                    if ssd < best_ssd
                        || (ssd == best_ssd
                            && (dx.unsigned_abs() + dy.unsigned_abs())
                                < (best_dx.unsigned_abs() + best_dy.unsigned_abs()))
                    {
                        best_ssd = ssd;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            // Compute confidence: lower SSD → higher confidence.
            // Normalise against a maximum SSD of 255² per pixel.
            let max_ssd = 255_u64 * 255 * (bsize * bsize) as u64;
            let confidence = if max_ssd == 0 {
                0.0
            } else {
                1.0 - (best_ssd as f32 / max_ssd as f32).min(1.0)
            };

            field.set(
                bx,
                by,
                FlowVector::new(best_dx as f32, best_dy as f32, confidence),
            );
        }
    }

    field
}

/// Compute dense optical flow using the Farneback polynomial expansion method.
///
/// Each pixel in `prev` and `curr` is a grayscale `f32` value (any range;
/// the algorithm is invariant to absolute intensity). The two slices must
/// have the same length `width * height`.
///
/// Returns a flat, row-major `Vec<(f32, f32)>` where each element is the
/// `(dx, dy)` displacement at the corresponding pixel location.
///
/// The implementation delegates to [`crate::farneback_flow::compute_farneback_flow`]
/// with default configuration (3 pyramid levels, 5 iterations, poly_n = 3).
///
/// # Errors
///
/// Returns an error if `prev.len() != curr.len()`, the dimensions do not
/// match `width * height`, or the image is smaller than 8 × 8 pixels.
pub fn compute_dense_flow(
    prev: &[f32],
    curr: &[f32],
    width: u32,
    height: u32,
) -> AlignResult<Vec<(f32, f32)>> {
    let w = width as usize;
    let h = height as usize;

    if prev.len() != w * h || curr.len() != w * h {
        return Err(AlignError::InvalidConfig(
            "dense flow: slice length does not match width * height".to_string(),
        ));
    }

    // Convert f32 → u8 for the Farneback implementation (which expects u8 frames).
    // We scale to [0, 255] based on the combined min/max of both slices.
    let min_val = prev
        .iter()
        .chain(curr.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_val = prev
        .iter()
        .chain(curr.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let to_u8 = |v: f32| -> u8 {
        if range < 1e-8 {
            128u8
        } else {
            (((v - min_val) / range) * 255.0).round().clamp(0.0, 255.0) as u8
        }
    };

    let prev_u8: Vec<u8> = prev.iter().copied().map(to_u8).collect();
    let curr_u8: Vec<u8> = curr.iter().copied().map(to_u8).collect();

    let config = crate::farneback_flow::FarnebackConfig::default();
    let field = crate::farneback_flow::compute_farneback_flow(&prev_u8, &curr_u8, w, h, &config)?;

    Ok(field.dx.into_iter().zip(field.dy).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FlowVector ────────────────────────────────────────────────────────────

    #[test]
    fn test_flow_vector_magnitude_zero() {
        let v = FlowVector::new(0.0, 0.0, 1.0);
        assert_eq!(v.magnitude(), 0.0);
    }

    #[test]
    fn test_flow_vector_magnitude_pythagorean() {
        let v = FlowVector::new(3.0, 4.0, 1.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_angle() {
        let v = FlowVector::new(1.0, 0.0, 1.0);
        assert!(v.angle_radians().abs() < 1e-5); // pointing right → 0 rad

        let v2 = FlowVector::new(0.0, 1.0, 1.0);
        assert!((v2.angle_radians() - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    // ── FlowField ─────────────────────────────────────────────────────────────

    #[test]
    fn test_flow_field_dimensions() {
        let f = FlowField::new(64, 48, 8);
        assert_eq!(f.block_cols(), 8);
        assert_eq!(f.block_rows(), 6);
        assert_eq!(f.vectors.len(), 48);
    }

    #[test]
    fn test_flow_field_set_get() {
        let mut f = FlowField::new(32, 32, 8);
        f.set(1, 2, FlowVector::new(3.0, -1.0, 0.8));
        let v = f.get(1, 2).expect("valid position");
        assert_eq!(v.dx, 3.0);
        assert_eq!(v.dy, -1.0);
        assert_eq!(v.confidence, 0.8);
    }

    #[test]
    fn test_flow_field_get_out_of_bounds() {
        let f = FlowField::new(32, 32, 8);
        assert!(f.get(100, 100).is_none());
    }

    #[test]
    fn test_flow_field_avg_magnitude_all_zero() {
        let f = FlowField::new(32, 32, 8);
        assert_eq!(f.avg_magnitude(), 0.0);
    }

    #[test]
    fn test_flow_field_avg_magnitude_single_vector() {
        let mut f = FlowField::new(8, 8, 8);
        f.set(0, 0, FlowVector::new(3.0, 4.0, 1.0)); // magnitude = 5
        assert!((f.avg_magnitude() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_flow_field_dominant_direction_zero_confidence() {
        let f = FlowField::new(16, 16, 8);
        assert_eq!(f.dominant_direction(), (0.0, 0.0));
    }

    #[test]
    fn test_flow_field_dominant_direction() {
        let mut f = FlowField::new(16, 8, 8);
        // Two blocks: equal confidence, opposite dy but same dx
        f.set(0, 0, FlowVector::new(2.0, 1.0, 1.0));
        f.set(1, 0, FlowVector::new(2.0, -1.0, 1.0));
        let (ddx, ddy) = f.dominant_direction();
        assert!((ddx - 2.0).abs() < 1e-4);
        assert!(ddy.abs() < 1e-4); // cancels out
    }

    // ── sum_squared_diff ──────────────────────────────────────────────────────

    #[test]
    fn test_ssd_identical() {
        let a = [10_u8, 20, 30];
        assert_eq!(sum_squared_diff(&a, &a), 0);
    }

    #[test]
    fn test_ssd_known_value() {
        let a = [0_u8, 0, 0];
        let b = [3_u8, 4, 0];
        // 9 + 16 + 0 = 25
        assert_eq!(sum_squared_diff(&a, &b), 25);
    }

    // ── block_match_flow ──────────────────────────────────────────────────────

    #[test]
    fn test_block_match_identical_frames() {
        let frame = vec![100_u8; 64 * 48];
        let field = block_match_flow(&frame, &frame, 64, 48, 8, 4);
        // All blocks should have zero displacement
        for v in &field.vectors {
            assert_eq!(v.dx, 0.0);
            assert_eq!(v.dy, 0.0);
        }
    }

    #[test]
    fn test_block_match_returns_correct_field_size() {
        let prev = vec![0_u8; 32 * 32];
        let curr = vec![0_u8; 32 * 32];
        let field = block_match_flow(&prev, &curr, 32, 32, 8, 2);
        assert_eq!(field.block_cols(), 4);
        assert_eq!(field.block_rows(), 4);
    }

    // -- compute_dense_flow ---------------------------------------------------

    #[test]
    fn test_compute_dense_flow_identical_frames() {
        let w = 32u32;
        let h = 32u32;
        let img: Vec<f32> = (0..w * h).map(|i| (i % 50) as f32).collect();

        let flow = compute_dense_flow(&img, &img, w, h).expect("should succeed");
        assert_eq!(flow.len(), (w * h) as usize);

        // Identical frames → near-zero flow (Farneback may not be exactly zero
        // at every pixel due to the polynomial solver, but the mean should be small).
        let avg_mag: f32 = flow
            .iter()
            .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
            .sum::<f32>()
            / flow.len() as f32;

        assert!(
            avg_mag < 1.0,
            "avg magnitude on identical frames: {avg_mag}"
        );
    }

    #[test]
    fn test_compute_dense_flow_length_mismatch() {
        let a = vec![0.0_f32; 64 * 64];
        let b = vec![0.0_f32; 32 * 32];
        let result = compute_dense_flow(&a, &b, 64, 64);
        assert!(result.is_err(), "mismatched lengths should error");
    }

    #[test]
    fn test_compute_dense_flow_too_small() {
        let a = vec![0.0_f32; 4 * 4];
        let b = vec![0.0_f32; 4 * 4];
        let result = compute_dense_flow(&a, &b, 4, 4);
        assert!(result.is_err(), "images < 8x8 should error");
    }

    #[test]
    fn test_compute_dense_flow_non_zero_motion() {
        // Create a stripe pattern and shift it by 2 pixels.
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let mut prev = vec![0.0_f32; n];
        let mut curr = vec![0.0_f32; n];

        for y in 0..h as usize {
            for x in 0..w as usize {
                prev[y * w as usize + x] = if (x / 8) % 2 == 0 { 200.0 } else { 50.0 };
                let sx = (x + 2).min(w as usize - 1);
                curr[y * w as usize + sx] = if (x / 8) % 2 == 0 { 200.0 } else { 50.0 };
            }
        }

        let flow = compute_dense_flow(&prev, &curr, w, h).expect("should succeed");
        let max_mag = flow
            .iter()
            .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
            .fold(0.0_f32, f32::max);

        assert!(
            max_mag > 0.0,
            "shifted stripe pattern should produce non-zero flow"
        );
    }

    #[test]
    fn test_compute_dense_flow_constant_frame() {
        // All-constant frames → no gradient → flow should be near-zero.
        let w = 16u32;
        let h = 16u32;
        let img = vec![1.0_f32; (w * h) as usize];
        let flow = compute_dense_flow(&img, &img, w, h).expect("should succeed");
        for (dx, dy) in &flow {
            assert!(dx.abs() < 1.0 && dy.abs() < 1.0, "dx={dx}, dy={dy}");
        }
    }
}
