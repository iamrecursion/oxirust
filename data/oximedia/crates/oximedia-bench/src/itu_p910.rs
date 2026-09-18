//! ITU-T P.910 video quality metrics.
//!
//! Implements the three core subjective video quality predictors defined in
//! ITU-T P.910 Annex A:
//!
//! * **Spatial Information (SI)** — standard deviation of the Sobel magnitude
//!   image of a frame's luma plane.
//! * **Temporal Information (TI)** — standard deviation of the per-pixel
//!   absolute difference between consecutive luma planes.
//! * **Motion Complexity** — mean 16 × 16 block SAD (Sum of Absolute
//!   Differences) versus the previous frame.

/// Aggregated ITU-T P.910 metrics for a complete sequence.
#[derive(Debug, Clone)]
pub struct P910Metrics {
    /// Spatial Information — maximum over all frames.
    pub si_max: f64,
    /// Temporal Information — maximum over all consecutive frame-pairs.
    pub ti_max: f64,
    /// Motion complexity — mean block-SAD averaged over all frame-pairs.
    pub motion_mean: f64,
}

/// Stateful accumulator that accepts luma frames one at a time and tracks the
/// running SI max, TI max, and mean motion complexity.
pub struct P910Analyzer {
    prev_luma: Option<Vec<u8>>,
    width: usize,
    height: usize,
    si_max: f64,
    ti_max: f64,
    motion_sum: f64,
    motion_frame_count: usize,
}

impl P910Analyzer {
    /// Create a new analyzer for frames of the given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            prev_luma: None,
            width,
            height,
            si_max: 0.0,
            ti_max: 0.0,
            motion_sum: 0.0,
            motion_frame_count: 0,
        }
    }

    /// Feed one frame's luma plane (`width × height` bytes) to the analyzer.
    ///
    /// Updates `si_max` with the spatial information of this frame.
    /// Updates `ti_max` and `motion_mean` (against the previous frame) when a
    /// previous frame is available.
    pub fn feed_frame(&mut self, luma: &[u8]) {
        // --- Spatial Information (per-frame) ---
        let si = spatial_information(luma, self.width, self.height);
        if si > self.si_max {
            self.si_max = si;
        }

        // --- Temporal Information and Motion Complexity (frame-pair) ---
        if let Some(ref prev) = self.prev_luma {
            let ti = temporal_information(luma, prev);
            if ti > self.ti_max {
                self.ti_max = ti;
            }

            let mc = motion_complexity(luma, prev, self.width, self.height);
            self.motion_sum += mc;
            self.motion_frame_count += 1;
        }

        self.prev_luma = Some(luma.to_vec());
    }

    /// Return the metrics accumulated so far.
    ///
    /// Calling this on an analyzer that has received no frames returns all
    /// zeros.
    #[must_use]
    pub fn metrics(&self) -> P910Metrics {
        let motion_mean = if self.motion_frame_count == 0 {
            0.0
        } else {
            self.motion_sum / self.motion_frame_count as f64
        };
        P910Metrics {
            si_max: self.si_max,
            ti_max: self.ti_max,
            motion_mean,
        }
    }

    /// Reset the analyzer to its initial state, ready for a new sequence.
    pub fn reset(&mut self) {
        self.prev_luma = None;
        self.si_max = 0.0;
        self.ti_max = 0.0;
        self.motion_sum = 0.0;
        self.motion_frame_count = 0;
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the ITU-T P.910 Spatial Information (SI) for a single luma frame.
///
/// SI = std_dev over the Sobel magnitude image:
///   `G = sqrt(Gx² + Gy²)` per inner pixel (1-pixel border excluded).
///
/// Sobel kernels:
/// ```text
///  Gx = [-1  0  1]   Gy = [ 1  2  1]
///       [-2  0  2]        [ 0  0  0]
///       [-1  0  1]        [-1 -2 -1]
/// ```
fn spatial_information(luma: &[u8], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }

    let pixel = |row: usize, col: usize| -> f64 { luma[row * width + col] as f64 };

    let inner_count = (width - 2) * (height - 2);
    let mut magnitudes = Vec::with_capacity(inner_count);

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let gx = -pixel(row - 1, col - 1) - 2.0 * pixel(row, col - 1) - pixel(row + 1, col - 1)
                + pixel(row - 1, col + 1)
                + 2.0 * pixel(row, col + 1)
                + pixel(row + 1, col + 1);

            let gy = pixel(row - 1, col - 1) + 2.0 * pixel(row - 1, col) + pixel(row - 1, col + 1)
                - pixel(row + 1, col - 1)
                - 2.0 * pixel(row + 1, col)
                - pixel(row + 1, col + 1);

            magnitudes.push((gx * gx + gy * gy).sqrt());
        }
    }

    std_dev(&magnitudes)
}

/// Compute ITU-T P.910 Temporal Information (TI) for a consecutive frame pair.
///
/// TI = std_dev of the per-pixel absolute difference image.
fn temporal_information(curr: &[u8], prev: &[u8]) -> f64 {
    let n = curr.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean: f64 = curr
        .iter()
        .zip(prev.iter())
        .map(|(&c, &p)| (c as f64 - p as f64).abs())
        .sum::<f64>()
        / n;

    let variance: f64 = curr
        .iter()
        .zip(prev.iter())
        .map(|(&c, &p)| {
            let d = (c as f64 - p as f64).abs() - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    variance.sqrt()
}

/// Compute motion complexity as the mean 16 × 16 block SAD (Sum of Absolute
/// Differences) between `curr` and `prev`.
///
/// Only full 16 × 16 blocks that fit within the frame are considered.
fn motion_complexity(curr: &[u8], prev: &[u8], width: usize, height: usize) -> f64 {
    const BLOCK_W: usize = 16;
    const BLOCK_H: usize = 16;

    if width < BLOCK_W || height < BLOCK_H {
        return 0.0;
    }

    let mut total_sad: u64 = 0;
    let mut block_count: u64 = 0;

    let mut by = 0;
    while by + BLOCK_H <= height {
        let mut bx = 0;
        while bx + BLOCK_W <= width {
            let sad: u64 = (0..BLOCK_H)
                .flat_map(|dy| {
                    let row = (by + dy) * width;
                    (0..BLOCK_W).map(move |dx| {
                        let idx = row + bx + dx;
                        (curr[idx] as i64 - prev[idx] as i64).unsigned_abs()
                    })
                })
                .sum();
            total_sad += sad;
            block_count += 1;
            bx += BLOCK_W;
        }
        by += BLOCK_H;
    }

    if block_count == 0 {
        0.0
    } else {
        total_sad as f64 / block_count as f64
    }
}

/// Compute the population standard deviation of a slice of `f64` values.
fn std_dev(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    variance.sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_zero_for_solid_frame() {
        let w = 128;
        let h = 96;
        let luma = vec![128u8; w * h];
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&luma);
        let si = a.metrics().si_max;
        assert!(si < 1.0, "SI for solid frame = {si}");
    }

    #[test]
    fn si_nonzero_for_edge_frame() {
        let w = 128;
        let h = 96;
        // Vertical edge at column w/2: left half = 0, right half = 255.
        let luma: Vec<u8> = (0..w * h)
            .map(|i| if (i % w) < w / 2 { 0 } else { 255 })
            .collect();
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&luma);
        assert!(
            a.metrics().si_max > 50.0,
            "SI for edge frame = {}",
            a.metrics().si_max
        );
    }

    #[test]
    fn ti_zero_for_identical_frames() {
        let w = 64;
        let h = 48;
        let luma = vec![100u8; w * h];
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&luma);
        a.feed_frame(&luma);
        assert!(a.metrics().ti_max < 1.0);
    }

    #[test]
    fn motion_zero_for_static_frames() {
        let w = 64;
        let h = 48;
        let luma = vec![50u8; w * h];
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&luma);
        a.feed_frame(&luma);
        assert!(a.metrics().motion_mean < 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let w = 32;
        let h = 32;
        let luma: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&luma);
        a.feed_frame(&luma);

        a.reset();
        let m = a.metrics();
        assert_eq!(m.si_max, 0.0);
        assert_eq!(m.ti_max, 0.0);
        assert_eq!(m.motion_mean, 0.0);
    }

    #[test]
    fn motion_nonzero_for_random_frames() {
        let w = 64;
        let h = 48;
        // First frame: all zeros; second frame: all 255 → huge block SAD.
        let frame_a = vec![0u8; w * h];
        let frame_b = vec![255u8; w * h];
        let mut a = P910Analyzer::new(w, h);
        a.feed_frame(&frame_a);
        a.feed_frame(&frame_b);
        assert!(
            a.metrics().motion_mean > 100.0,
            "motion_mean = {}",
            a.metrics().motion_mean
        );
    }
}
