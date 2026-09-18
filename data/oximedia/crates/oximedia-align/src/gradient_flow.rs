//! Gradient-based optical flow for video alignment.
//!
//! Provides a simplified Lucas-Kanade style optical flow computation operating
//! on blocks of grayscale pixels represented as `f32` slices.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A 2D flow vector with single-precision components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
}

impl FlowVector {
    /// Create a new flow vector.
    #[must_use]
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Euclidean magnitude of the vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Angle of the vector in radians (atan2 convention, range −π..π).
    #[must_use]
    pub fn angle_rad(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Return `true` when both components are exactly zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.dx == 0.0 && self.dy == 0.0
    }
}

impl Default for FlowVector {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// A dense optical flow field over a regular block grid.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Row-major vector storage (`block_rows` × `block_cols`).
    pub vectors: Vec<FlowVector>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
}

impl FlowField {
    /// Create a new zeroed flow field for a frame of the given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize, block_size: usize) -> Self {
        let cols = width.div_ceil(block_size.max(1));
        let rows = height.div_ceil(block_size.max(1));
        Self {
            vectors: vec![FlowVector::default(); cols * rows],
            width,
            height,
        }
    }

    /// Number of block columns stored in this field.
    #[must_use]
    pub fn block_cols(&self, block_size: usize) -> usize {
        self.width.div_ceil(block_size.max(1))
    }

    /// Number of block rows stored in this field.
    #[must_use]
    pub fn block_rows(&self, block_size: usize) -> usize {
        self.height.div_ceil(block_size.max(1))
    }

    /// Return a reference to the flow vector at block column `x`, block row `y`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> &FlowVector {
        let cols = self.width.div_ceil(1); // always non-zero width
        &self.vectors[y * cols + x]
    }

    /// Arithmetic mean of the magnitudes of all flow vectors.
    ///
    /// Returns `0.0` if the field is empty.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(FlowVector::magnitude).sum();
        sum / self.vectors.len() as f32
    }

    /// Return a single [`FlowVector`] representing the arithmetic mean of all
    /// vectors (the dominant translation direction).
    #[must_use]
    pub fn dominant_direction(&self) -> FlowVector {
        if self.vectors.is_empty() {
            return FlowVector::default();
        }
        let n = self.vectors.len() as f32;
        let sum_dx: f32 = self.vectors.iter().map(|v| v.dx).sum();
        let sum_dy: f32 = self.vectors.iter().map(|v| v.dy).sum();
        FlowVector::new(sum_dx / n, sum_dy / n)
    }
}

/// Compute a simplified Lucas-Kanade style flow vector for a single block.
///
/// The function estimates the spatiotemporal gradient using finite differences
/// and solves the brightness-constancy equation:
///
/// > `Ix * vx + Iy * vy = -It`
///
/// when both spatial gradients are non-zero; otherwise it returns a zero
/// vector.
///
/// # Arguments
///
/// * `prev_block` – Pixel values (f32) for the block in the previous frame.
/// * `curr_block` – Pixel values (f32) for the block in the current frame.
/// * `block_size` – Width (= height) of the square block.
#[must_use]
pub fn lucas_kanade_block(prev_block: &[f32], curr_block: &[f32], block_size: usize) -> FlowVector {
    if prev_block.is_empty() || curr_block.is_empty() || block_size == 0 {
        return FlowVector::default();
    }

    let n = (block_size * block_size) as f32;

    // Accumulate normal-equation components
    let mut sum_ix2 = 0.0_f32;
    let mut sum_iy2 = 0.0_f32;
    let mut sum_ix_iy = 0.0_f32;
    let mut sum_ix_it = 0.0_f32;
    let mut sum_iy_it = 0.0_f32;

    let len = prev_block.len().min(curr_block.len());
    for i in 0..len {
        let it = curr_block[i] - prev_block[i];
        // Approximate spatial gradients using central differences when possible
        let ix = if i + 1 < len {
            curr_block[i + 1] - curr_block[i]
        } else {
            0.0
        };
        let iy = if i + block_size < len {
            curr_block[i + block_size] - curr_block[i]
        } else {
            0.0
        };
        sum_ix2 += ix * ix;
        sum_iy2 += iy * iy;
        sum_ix_iy += ix * iy;
        sum_ix_it += ix * it;
        sum_iy_it += iy * it;
    }

    // Normalise by number of pixels
    let a = sum_ix2 / n;
    let b = sum_ix_iy / n;
    let c = sum_iy2 / n;
    let d = -sum_ix_it / n;
    let e = -sum_iy_it / n;

    // Solve 2×2 system [a b; b c] [vx; vy] = [d; e]
    let det = a * c - b * b;
    if det.abs() < 1e-8 {
        return FlowVector::default();
    }

    let vx = (d * c - e * b) / det;
    let vy = (a * e - b * d) / det;
    FlowVector::new(vx, vy)
}

/// Compute a dense flow field for an entire frame using [`lucas_kanade_block`].
///
/// # Arguments
///
/// * `prev_frame` – Grayscale pixel values of the previous frame (row-major).
/// * `curr_frame` – Grayscale pixel values of the current frame (row-major).
/// * `width` / `height` – Frame dimensions in pixels.
/// * `block_size` – Block size in pixels (blocks do not overlap).
#[must_use]
pub fn compute_flow_field(
    prev_frame: &[f32],
    curr_frame: &[f32],
    width: usize,
    height: usize,
    block_size: usize,
) -> FlowField {
    let bs = block_size.max(1);
    let block_cols = width.div_ceil(bs);
    let block_rows = height.div_ceil(bs);
    let mut vectors = Vec::with_capacity(block_cols * block_rows);

    for by in 0..block_rows {
        for bx in 0..block_cols {
            let x0 = bx * bs;
            let y0 = by * bs;

            // Extract block pixels
            let mut prev_block = Vec::with_capacity(bs * bs);
            let mut curr_block = Vec::with_capacity(bs * bs);

            for row in 0..bs {
                let y = y0 + row;
                if y >= height {
                    break;
                }
                for col in 0..bs {
                    let x = x0 + col;
                    if x >= width {
                        break;
                    }
                    let idx = y * width + x;
                    if idx < prev_frame.len() {
                        prev_block.push(prev_frame[idx]);
                    }
                    if idx < curr_frame.len() {
                        curr_block.push(curr_frame[idx]);
                    }
                }
            }

            let fv = lucas_kanade_block(&prev_block, &curr_block, bs);
            vectors.push(fv);
        }
    }

    FlowField {
        vectors,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_flow_vector_magnitude_zero() {
        let v = FlowVector::new(0.0, 0.0);
        assert_eq!(v.magnitude(), 0.0);
    }

    #[test]
    fn test_flow_vector_magnitude_345() {
        let v = FlowVector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_angle_right() {
        let v = FlowVector::new(1.0, 0.0);
        assert!(v.angle_rad().abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_angle_up() {
        let v = FlowVector::new(0.0, 1.0);
        assert!((v.angle_rad() - PI / 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_vector_is_zero_true() {
        let v = FlowVector::new(0.0, 0.0);
        assert!(v.is_zero());
    }

    #[test]
    fn test_flow_vector_is_zero_false() {
        let v = FlowVector::new(0.0, 0.001);
        assert!(!v.is_zero());
    }

    #[test]
    fn test_flow_field_average_magnitude_empty_vectors() {
        // A field with explicit zero vectors should return 0.0 average magnitude
        let field = FlowField {
            vectors: vec![FlowVector::default(); 4],
            width: 8,
            height: 8,
        };
        assert_eq!(field.average_magnitude(), 0.0);
    }

    #[test]
    fn test_flow_field_average_magnitude_uniform() {
        let v = FlowVector::new(3.0, 4.0); // magnitude 5
        let field = FlowField {
            vectors: vec![v; 4],
            width: 8,
            height: 8,
        };
        assert!((field.average_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_dominant_direction_uniform() {
        let field = FlowField {
            vectors: vec![FlowVector::new(2.0, -1.0); 6],
            width: 12,
            height: 8,
        };
        let d = field.dominant_direction();
        assert!((d.dx - 2.0).abs() < 1e-5);
        assert!((d.dy + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_field_dominant_direction_cancels() {
        // Opposite dx vectors should cancel out
        let field = FlowField {
            vectors: vec![FlowVector::new(1.0, 0.0), FlowVector::new(-1.0, 0.0)],
            width: 4,
            height: 4,
        };
        let d = field.dominant_direction();
        assert!(d.dx.abs() < 1e-5);
    }

    #[test]
    fn test_lucas_kanade_block_empty_returns_zero() {
        let fv = lucas_kanade_block(&[], &[], 8);
        assert!(fv.is_zero());
    }

    #[test]
    fn test_lucas_kanade_block_constant_frames_returns_zero() {
        let prev: Vec<f32> = vec![128.0; 64];
        let curr = prev.clone();
        let fv = lucas_kanade_block(&prev, &curr, 8);
        // No temporal gradient → zero flow
        assert!(fv.is_zero());
    }

    #[test]
    fn test_compute_flow_field_dimensions() {
        let prev = vec![0.0_f32; 64 * 48];
        let curr = prev.clone();
        let field = compute_flow_field(&prev, &curr, 64, 48, 8);
        // Should have 8*6 = 48 blocks
        assert_eq!(field.vectors.len(), 48);
    }

    #[test]
    fn test_compute_flow_field_constant_frames() {
        let frame = vec![100.0_f32; 16 * 16];
        let field = compute_flow_field(&frame, &frame, 16, 16, 4);
        for v in &field.vectors {
            assert!(v.is_zero());
        }
    }
}
