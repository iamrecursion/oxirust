//! Color matching across camera angles.

use super::{ColorMatrix, ColorStats};
use crate::{AngleId, Result};

/// Color matcher
#[derive(Debug)]
pub struct ColorMatcher {
    /// Reference angle for color matching
    reference_angle: AngleId,
    /// Color statistics for each angle
    stats: Vec<ColorStats>,
    /// Color correction matrices
    corrections: Vec<ColorMatrix>,
}

impl ColorMatcher {
    /// Create a new color matcher
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId) -> Self {
        Self {
            reference_angle,
            stats: (0..angle_count).map(ColorStats::new).collect(),
            corrections: vec![ColorMatrix::identity(); angle_count],
        }
    }

    /// Set reference angle
    pub fn set_reference_angle(&mut self, angle: AngleId) {
        self.reference_angle = angle;
    }

    /// Get reference angle
    #[must_use]
    pub fn reference_angle(&self) -> AngleId {
        self.reference_angle
    }

    /// Update color statistics for an angle
    pub fn update_stats(&mut self, stats: ColorStats) {
        if stats.angle < self.stats.len() {
            self.stats[stats.angle] = stats;
        }
    }

    /// Calculate color correction for all angles
    ///
    /// # Errors
    ///
    /// Returns an error if calculation fails
    pub fn calculate_corrections(&mut self) -> Result<()> {
        if self.reference_angle >= self.stats.len() {
            return Err(crate::MultiCamError::AngleNotFound(self.reference_angle));
        }

        let reference = &self.stats[self.reference_angle];

        for (angle, stats) in self.stats.iter().enumerate() {
            if angle == self.reference_angle {
                self.corrections[angle] = ColorMatrix::identity();
                continue;
            }

            // Calculate color correction matrix
            let correction = self.calculate_correction_matrix(stats, reference);
            self.corrections[angle] = correction;
        }

        Ok(())
    }

    /// Calculate correction matrix from source to target statistics
    fn calculate_correction_matrix(&self, source: &ColorStats, target: &ColorStats) -> ColorMatrix {
        // Simple scaling correction
        let mut matrix = [[0.0; 3]; 3];

        for i in 0..3 {
            if source.mean_rgb[i] > 0.0 {
                matrix[i][i] = target.mean_rgb[i] / source.mean_rgb[i];
            } else {
                matrix[i][i] = 1.0;
            }
        }

        ColorMatrix { matrix }
    }

    /// Get correction matrix for angle
    #[must_use]
    pub fn get_correction(&self, angle: AngleId) -> Option<&ColorMatrix> {
        self.corrections.get(angle)
    }

    /// Apply color correction to RGB values
    #[must_use]
    pub fn apply_correction(&self, angle: AngleId, rgb: [f32; 3]) -> [f32; 3] {
        if let Some(matrix) = self.get_correction(angle) {
            let corrected = matrix.apply(rgb);
            [
                corrected[0].clamp(0.0, 1.0),
                corrected[1].clamp(0.0, 1.0),
                corrected[2].clamp(0.0, 1.0),
            ]
        } else {
            rgb
        }
    }

    /// Get color statistics for angle
    #[must_use]
    pub fn get_stats(&self, angle: AngleId) -> Option<&ColorStats> {
        self.stats.get(angle)
    }

    /// Calculate average color statistics across all angles
    #[must_use]
    pub fn average_stats(&self) -> ColorStats {
        let mut avg = ColorStats::new(0);
        // Reset to zero so we can accumulate without the default values
        avg.mean_rgb = [0.0, 0.0, 0.0];
        avg.std_rgb = [0.0, 0.0, 0.0];
        avg.temperature = 0.0;
        avg.tint = 0.0;

        let count = self.stats.len() as f32;

        for stats in &self.stats {
            for i in 0..3 {
                avg.mean_rgb[i] += stats.mean_rgb[i] / count;
                avg.std_rgb[i] += stats.std_rgb[i] / count;
            }
            avg.temperature += stats.temperature / count;
            avg.tint += stats.tint / count;
        }

        avg
    }

    /// Check color consistency across angles
    #[must_use]
    pub fn check_consistency(&self, threshold: f32) -> bool {
        if self.reference_angle >= self.stats.len() {
            return false;
        }

        let reference = &self.stats[self.reference_angle];

        for stats in &self.stats {
            if stats.angle != self.reference_angle {
                let distance = stats.distance_to(reference);
                if distance > threshold {
                    return false;
                }
            }
        }

        true
    }

    /// Get angles that need color correction
    #[must_use]
    pub fn angles_needing_correction(&self, threshold: f32) -> Vec<AngleId> {
        if self.reference_angle >= self.stats.len() {
            return Vec::new();
        }

        let reference = &self.stats[self.reference_angle];
        let mut angles = Vec::new();

        for stats in &self.stats {
            if stats.angle != self.reference_angle {
                let distance = stats.distance_to(reference);
                if distance > threshold {
                    angles.push(stats.angle);
                }
            }
        }

        angles
    }

    /// Reset all corrections
    pub fn reset_corrections(&mut self) {
        for correction in &mut self.corrections {
            *correction = ColorMatrix::identity();
        }
    }
}

// ── Tile-based color matching ─────────────────────────────────────────────────

/// Sample a `tile_n × tile_n` evenly-spaced grid of tiles from an RGB pixel
/// buffer and return the per-channel mean and standard deviation.
///
/// * `pixels` – Interleaved RGB byte buffer (length == `width * height * 3`).
/// * `width`, `height` – Frame dimensions in pixels.
/// * `tile_n` – Number of tiles per axis.  Clamped to `[1, min(width, height)]`.
///
/// Returns `([mean_r, mean_g, mean_b], [std_r, std_g, std_b])`.
///
/// Returns zeros for all channels when `pixels` is empty or `width`/`height`
/// are zero.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_tiled_stats(
    pixels: &[u8],
    width: u32,
    height: u32,
    tile_n: usize,
) -> ([f32; 3], [f32; 3]) {
    let w = width as usize;
    let h = height as usize;

    // Guard against degenerate inputs.
    if pixels.is_empty() || w == 0 || h == 0 || tile_n == 0 {
        return ([0.0; 3], [0.0; 3]);
    }

    // Clamp tile_n so we always have at least 1 pixel per tile.
    let effective_n = tile_n.min(w).min(h).max(1);

    // Expected minimum buffer length: w × h × 3 channels.
    let expected = w * h * 3;
    if pixels.len() < expected {
        return ([0.0; 3], [0.0; 3]);
    }

    // Tile side lengths (integer division, at least 1 pixel wide/tall).
    let tile_w = (w / effective_n).max(1);
    let tile_h = (h / effective_n).max(1);

    // Half-tile offsets so that sample points sit in the *centre* of each tile.
    let half_w = tile_w / 2;
    let half_h = tile_h / 2;

    let mut sums = [0.0f64; 3];
    let mut sq_sums = [0.0f64; 3];
    let mut count = 0usize;

    for ty in 0..effective_n {
        let cy = (ty * tile_h + half_h).min(h - 1);
        for tx in 0..effective_n {
            let cx = (tx * tile_w + half_w).min(w - 1);

            // Sample a small patch (up to 3×3) around the tile centre to
            // reduce single-pixel noise without scanning the whole tile.
            let x0 = cx.saturating_sub(1);
            let x1 = (cx + 2).min(w);
            let y0 = cy.saturating_sub(1);
            let y1 = (cy + 2).min(h);

            for py in y0..y1 {
                for px in x0..x1 {
                    let base = (py * w + px) * 3;
                    // Bounds check — should never fire given the earlier guard,
                    // but be defensive.
                    if base + 2 >= pixels.len() {
                        continue;
                    }
                    for ch in 0..3usize {
                        let v = pixels[base + ch] as f64 / 255.0;
                        sums[ch] += v;
                        sq_sums[ch] += v * v;
                    }
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return ([0.0; 3], [0.0; 3]);
    }

    let n = count as f64;
    let mut mean = [0.0f32; 3];
    let mut std = [0.0f32; 3];
    for ch in 0..3usize {
        let m = sums[ch] / n;
        let variance = (sq_sums[ch] / n) - m * m;
        mean[ch] = m as f32;
        std[ch] = variance.max(0.0).sqrt() as f32;
    }

    (mean, std)
}

/// Apply a mean-and-standard-deviation color transfer from `src_stats` to
/// `reference_stats` to every pixel in `pixels` (in place).
///
/// The transfer formula per channel:
///
/// ```text
/// output = (input − src_mean) × (ref_std / src_std) + ref_mean
/// ```
///
/// When `src_std[ch]` is zero the channel is shifted by mean difference only.
#[allow(clippy::cast_precision_loss)]
pub fn apply_mean_std_transfer(
    pixels: &mut [u8],
    src_mean: [f32; 3],
    src_std: [f32; 3],
    ref_mean: [f32; 3],
    ref_std: [f32; 3],
) {
    // Precompute per-channel scale and shift to avoid per-pixel division.
    let mut scale = [1.0f32; 3];
    let mut shift = [0.0f32; 3];
    for ch in 0..3usize {
        scale[ch] = if src_std[ch] > f32::EPSILON {
            ref_std[ch] / src_std[ch]
        } else {
            1.0
        };
        shift[ch] = ref_mean[ch] - src_mean[ch] * scale[ch];
    }

    for chunk in pixels.chunks_exact_mut(3) {
        for (ch, byte) in chunk.iter_mut().enumerate() {
            let v = *byte as f32 / 255.0;
            let corrected = (v * scale[ch] + shift[ch]).clamp(0.0, 1.0);
            *byte = (corrected * 255.0 + 0.5) as u8;
        }
    }
}

/// Color-match `src` to `reference` using a tile-grid sample instead of every
/// pixel.
///
/// Only a `tile_n × tile_n` evenly-spaced grid of sample patches is used when
/// computing the color statistics, which keeps the cost sub-linear in the
/// number of pixels.  The resulting mean-std transfer is then applied to every
/// pixel in `src`.
///
/// # Arguments
///
/// * `src` – Mutable interleaved RGB pixel buffer (`width × height × 3` bytes).
/// * `reference` – Reference RGB pixel buffer (same dimensions).
/// * `width`, `height` – Frame dimensions in pixels.
/// * `tile_n` – Number of tiles per axis (clamped to `[1, min(width, height)]`).
///
/// # Returns
///
/// The modified `src` pixels with the color transfer applied.
#[must_use]
pub fn match_color_tiled(
    src: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    tile_n: usize,
) -> Vec<u8> {
    let (ref_mean, ref_std) = compute_tiled_stats(reference, width, height, tile_n);
    let (src_mean, src_std) = compute_tiled_stats(src, width, height, tile_n);

    let mut output = src.to_vec();
    apply_mean_std_transfer(&mut output, src_mean, src_std, ref_mean, ref_std);
    output
}

/// Color transfer method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTransferMethod {
    /// Mean matching
    Mean,
    /// Mean and standard deviation matching
    MeanStd,
    /// Histogram matching
    Histogram,
    /// 3D LUT-based transfer
    Lut3D,
}

/// Advanced color matcher with multiple methods
#[derive(Debug)]
pub struct AdvancedColorMatcher {
    /// Base matcher
    matcher: ColorMatcher,
    /// Transfer method
    method: ColorTransferMethod,
}

impl AdvancedColorMatcher {
    /// Create a new advanced color matcher
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId, method: ColorTransferMethod) -> Self {
        Self {
            matcher: ColorMatcher::new(angle_count, reference_angle),
            method,
        }
    }

    /// Set transfer method
    pub fn set_method(&mut self, method: ColorTransferMethod) {
        self.method = method;
    }

    /// Get transfer method
    #[must_use]
    pub fn method(&self) -> ColorTransferMethod {
        self.method
    }

    /// Get base matcher
    #[must_use]
    pub fn matcher(&self) -> &ColorMatcher {
        &self.matcher
    }

    /// Get mutable base matcher
    pub fn matcher_mut(&mut self) -> &mut ColorMatcher {
        &mut self.matcher
    }

    /// Apply color transfer using selected method
    #[must_use]
    pub fn transfer_color(&self, angle: AngleId, rgb: [f32; 3]) -> [f32; 3] {
        match self.method {
            ColorTransferMethod::Mean | ColorTransferMethod::MeanStd => {
                self.matcher.apply_correction(angle, rgb)
            }
            ColorTransferMethod::Histogram => {
                // Placeholder for histogram matching
                self.matcher.apply_correction(angle, rgb)
            }
            ColorTransferMethod::Lut3D => {
                // Placeholder for 3D LUT
                self.matcher.apply_correction(angle, rgb)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_matcher_creation() {
        let matcher = ColorMatcher::new(3, 0);
        assert_eq!(matcher.reference_angle(), 0);
        assert_eq!(matcher.stats.len(), 3);
    }

    #[test]
    fn test_update_stats() {
        let mut matcher = ColorMatcher::new(3, 0);
        let mut stats = ColorStats::new(1);
        stats.mean_rgb = [0.8, 0.7, 0.6];

        matcher.update_stats(stats);
        assert_eq!(
            matcher
                .get_stats(1)
                .expect("multicam test operation should succeed")
                .mean_rgb,
            [0.8, 0.7, 0.6]
        );
    }

    #[test]
    fn test_calculate_corrections() {
        let mut matcher = ColorMatcher::new(2, 0);
        let result = matcher.calculate_corrections();
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_correction() {
        let matcher = ColorMatcher::new(2, 0);
        let rgb = [1.0, 0.5, 0.25];
        let corrected = matcher.apply_correction(0, rgb);
        assert_eq!(corrected, rgb); // Identity for reference angle
    }

    #[test]
    fn test_average_stats() {
        let mut matcher = ColorMatcher::new(2, 0);

        let mut stats1 = ColorStats::new(0);
        stats1.mean_rgb = [0.4, 0.4, 0.4];
        matcher.update_stats(stats1);

        let mut stats2 = ColorStats::new(1);
        stats2.mean_rgb = [0.6, 0.6, 0.6];
        matcher.update_stats(stats2);

        let avg = matcher.average_stats();
        assert!((avg.mean_rgb[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_check_consistency() {
        let matcher = ColorMatcher::new(3, 0);
        assert!(matcher.check_consistency(1.0)); // Should be consistent with large threshold
    }

    #[test]
    fn test_angles_needing_correction() {
        let mut matcher = ColorMatcher::new(3, 0);

        let mut stats = ColorStats::new(1);
        stats.mean_rgb = [1.0, 1.0, 1.0]; // Very different from default
        matcher.update_stats(stats);

        let angles = matcher.angles_needing_correction(0.1);
        assert!(!angles.is_empty());
    }

    #[test]
    fn test_advanced_matcher() {
        let matcher = AdvancedColorMatcher::new(3, 0, ColorTransferMethod::Mean);
        assert_eq!(matcher.method(), ColorTransferMethod::Mean);
    }

    #[test]
    fn test_transfer_color() {
        let matcher = AdvancedColorMatcher::new(2, 0, ColorTransferMethod::Mean);
        let rgb = [0.8, 0.6, 0.4];
        let transferred = matcher.transfer_color(0, rgb);
        assert!(transferred[0] <= 1.0);
        assert!(transferred[1] <= 1.0);
        assert!(transferred[2] <= 1.0);
    }

    // ── Tile-based color matching tests ──────────────────────────────────────

    /// Helper: build a synthetic gradient frame (width × height × 3 RGB bytes).
    fn make_gradient(width: u32, height: u32, r_bias: u8, g_bias: u8, b_bias: u8) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0u8; w * h * 3];
        for row in 0..h {
            for col in 0..w {
                let base = (row * w + col) * 3;
                buf[base] = ((col * 255 / w.max(1)) as u8).saturating_add(r_bias);
                buf[base + 1] = ((row * 255 / h.max(1)) as u8).saturating_add(g_bias);
                buf[base + 2] = (((col + row) * 127 / (w + h).max(1)) as u8).saturating_add(b_bias);
            }
        }
        buf
    }

    /// Tiled match at tile_n=8 should produce a result that is close to what
    /// a full-frame mean-std transfer would produce.  We measure the mean
    /// absolute error of the output means rather than pixel-exact equality
    /// because the two sampling strategies differ slightly.
    #[test]
    fn test_tile_match_approx_full_match() {
        let (w, h) = (64u32, 64u32);
        let src = make_gradient(w, h, 10, 20, 30);
        let reference = make_gradient(w, h, 80, 60, 40);

        // Full-frame transfer (tile_n = min(w, h)).
        let full_result = match_color_tiled(&src, &reference, w, h, 64);
        // Tile-sampled transfer.
        let tiled_result = match_color_tiled(&src, &reference, w, h, 8);

        assert_eq!(full_result.len(), tiled_result.len());

        // Compute mean channel values for both results.
        let mean_for = |buf: &[u8]| -> [f32; 3] {
            let n = (buf.len() / 3) as f64;
            let mut sums = [0.0f64; 3];
            for chunk in buf.chunks_exact(3) {
                for (ch, &v) in chunk.iter().enumerate() {
                    sums[ch] += v as f64;
                }
            }
            [
                (sums[0] / n) as f32,
                (sums[1] / n) as f32,
                (sums[2] / n) as f32,
            ]
        };

        let full_means = mean_for(&full_result);
        let tiled_means = mean_for(&tiled_result);

        // Allow up to 15 units (out of 255) mean difference per channel.
        for ch in 0..3usize {
            let delta = (full_means[ch] - tiled_means[ch]).abs();
            assert!(
                delta < 15.0,
                "channel {ch}: tiled mean {:.1} differs from full mean {:.1} by {:.1}",
                tiled_means[ch],
                full_means[ch],
                delta
            );
        }
    }

    /// tile_n=1 (single center tile) must not panic and must return a buffer of
    /// the correct length.
    #[test]
    fn test_tile_match_single_tile() {
        let (w, h) = (32u32, 32u32);
        let src = make_gradient(w, h, 0, 0, 0);
        let reference = make_gradient(w, h, 50, 50, 50);
        let result = match_color_tiled(&src, &reference, w, h, 1);
        assert_eq!(result.len(), src.len());
    }

    /// tile_n larger than both width and height should degrade gracefully (clamp
    /// to frame dimensions) and not panic.
    #[test]
    fn test_tile_match_larger_than_frame() {
        let (w, h) = (8u32, 8u32);
        let src = make_gradient(w, h, 5, 10, 15);
        let reference = make_gradient(w, h, 30, 40, 50);
        // tile_n = 9999 >> 8 — should clamp internally and still work.
        let result = match_color_tiled(&src, &reference, w, h, 9_999);
        assert_eq!(result.len(), src.len(), "output length should match input");
        // All values must be valid u8 bytes — the length check above already
        // guarantees the result is well-formed RGB; no further assertion needed
        // because u8 is by definition in [0, 255].
        let _ = result;
    }
}
