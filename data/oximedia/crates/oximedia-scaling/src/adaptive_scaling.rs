//! Content-aware adaptive scaling using seam carving.
//!
//! Implements seam carving for retargeting, with support for protected content regions
//! and pillarbox detection/removal.

/// A content region with an importance weight for seam-carving protection.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ContentRegion {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
    /// Importance weight (1.0 = normal, higher = more protected).
    pub importance: f32,
}

/// Seam carver for content-aware width reduction.
pub struct SeamCarver;

impl SeamCarver {
    /// Compute pixel energy using gradient magnitude (Sobel-like).
    ///
    /// Returns a flattened row-major energy map the same size as the image.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute_energy(img: &[f32], w: u32, h: u32) -> Vec<f32> {
        let w = w as usize;
        let h = h as usize;
        let mut energy = vec![0.0f32; w * h];

        for y in 0..h {
            for x in 0..w {
                let left = if x > 0 {
                    img[y * w + x - 1]
                } else {
                    img[y * w + x]
                };
                let right = if x + 1 < w {
                    img[y * w + x + 1]
                } else {
                    img[y * w + x]
                };
                let up = if y > 0 {
                    img[(y - 1) * w + x]
                } else {
                    img[y * w + x]
                };
                let down = if y + 1 < h {
                    img[(y + 1) * w + x]
                } else {
                    img[y * w + x]
                };

                let dx = right - left;
                let dy = down - up;
                energy[y * w + x] = (dx * dx + dy * dy).sqrt();
            }
        }
        energy
    }

    /// Find the minimum-energy vertical seam using dynamic programming.
    ///
    /// Returns the column index for each row of the seam.
    #[must_use]
    #[allow(dead_code)]
    pub fn find_vertical_seam(energy: &[f32], w: u32, h: u32) -> Vec<u32> {
        let w = w as usize;
        let h = h as usize;
        let mut dp = vec![0.0f32; w * h];
        let mut path = vec![0usize; w * h];

        // Initialize first row
        for x in 0..w {
            dp[x] = energy[x];
        }

        // Fill DP table
        for y in 1..h {
            for x in 0..w {
                let left = if x > 0 {
                    dp[(y - 1) * w + x - 1]
                } else {
                    f32::INFINITY
                };
                let center = dp[(y - 1) * w + x];
                let right = if x + 1 < w {
                    dp[(y - 1) * w + x + 1]
                } else {
                    f32::INFINITY
                };

                let (min_val, min_x) = if left <= center && left <= right {
                    (left, x.saturating_sub(1))
                } else if center <= right {
                    (center, x)
                } else {
                    (right, (x + 1).min(w - 1))
                };

                dp[y * w + x] = energy[y * w + x] + min_val;
                path[y * w + x] = min_x;
            }
        }

        // Trace back seam
        let mut seam = vec![0u32; h];
        let last_row_start = (h - 1) * w;
        let mut min_x = 0usize;
        let mut min_val = f32::INFINITY;
        for x in 0..w {
            if dp[last_row_start + x] < min_val {
                min_val = dp[last_row_start + x];
                min_x = x;
            }
        }
        seam[h - 1] = min_x as u32;

        for y in (0..h - 1).rev() {
            min_x = path[(y + 1) * w + min_x];
            seam[y] = min_x as u32;
        }

        seam
    }

    /// Remove a vertical seam from an image.
    ///
    /// Returns an image with width reduced by 1.
    #[must_use]
    #[allow(dead_code)]
    pub fn remove_seam(img: &[f32], w: u32, h: u32, seam: &[u32]) -> Vec<f32> {
        let w = w as usize;
        let h = h as usize;
        let new_w = w - 1;
        let mut dst = Vec::with_capacity(new_w * h);

        for y in 0..h {
            let seam_x = seam[y] as usize;
            for x in 0..w {
                if x != seam_x {
                    dst.push(img[y * w + x]);
                }
            }
        }
        dst
    }

    /// Reduce image width to `target_w` by iteratively removing minimum-energy seams.
    #[must_use]
    #[allow(dead_code)]
    pub fn carve_to_width(src: &[f32], src_w: u32, src_h: u32, target_w: u32) -> Vec<f32> {
        if target_w >= src_w {
            return src.to_vec();
        }
        let n_seams = (src_w - target_w) as usize;
        let mut img = src.to_vec();
        let mut cur_w = src_w;

        for _ in 0..n_seams {
            let energy = Self::compute_energy(&img, cur_w, src_h);
            let seam = Self::find_vertical_seam(&energy, cur_w, src_h);
            img = Self::remove_seam(&img, cur_w, src_h, &seam);
            cur_w -= 1;
        }
        img
    }
}

/// Content-aware scaler that combines seam carving with region protection.
pub struct AdaptiveScaler;

impl AdaptiveScaler {
    /// Scale image from `(src_w, src_h)` to `(dst_w, dst_h)` using seam carving.
    ///
    /// Protected regions have their energy boosted to prevent seam removal through them.
    /// Currently supports width reduction only; height is handled by bilinear scaling.
    #[must_use]
    #[allow(dead_code)]
    pub fn scale(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        protected_regions: &[ContentRegion],
    ) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        // Boost energy in protected regions
        let mut energy_boost = vec![1.0f32; (src_w * src_h) as usize];
        for region in protected_regions {
            let rx = region.x as usize;
            let ry = region.y as usize;
            let rw = region.w as usize;
            let rh = region.h as usize;
            let sw = src_w as usize;
            let sh = src_h as usize;
            for y in ry..((ry + rh).min(sh)) {
                for x in rx..((rx + rw).min(sw)) {
                    energy_boost[y * sw + x] = region.importance * 1000.0;
                }
            }
        }

        // Apply boost to a modified source for seam finding
        let boosted: Vec<f32> = src
            .iter()
            .zip(energy_boost.iter())
            .map(|(&v, &b)| v * b.min(1.0) + (b - 1.0).max(0.0))
            .collect();

        // Width reduction via seam carving
        let width_carved = if dst_w < src_w {
            // Use boosted image for energy computation
            let mut img = src.to_vec();
            let mut cur_w = src_w;
            let n_seams = (src_w - dst_w) as usize;

            for _ in 0..n_seams {
                let base_energy = SeamCarver::compute_energy(&img, cur_w, src_h);
                let boost_energy = SeamCarver::compute_energy(&boosted, cur_w, src_h);
                // Combine: use max to protect boosted regions
                let combined: Vec<f32> = base_energy
                    .iter()
                    .zip(boost_energy.iter())
                    .map(|(&b, &e)| b + e)
                    .collect();
                let seam = SeamCarver::find_vertical_seam(&combined, cur_w, src_h);
                img = SeamCarver::remove_seam(&img, cur_w, src_h, &seam);
                cur_w -= 1;
            }
            (img, dst_w)
        } else {
            (src.to_vec(), src_w)
        };

        let (img, cur_w) = width_carved;

        // Height reduction by simple row sampling
        if dst_h < src_h {
            let sw = cur_w as usize;
            let sh = src_h as usize;
            let dh = dst_h as usize;
            let mut out = Vec::with_capacity(sw * dh);
            for dy in 0..dh {
                let sy = dy * sh / dh;
                out.extend_from_slice(&img[sy * sw..(sy * sw + sw)]);
            }
            out
        } else {
            img
        }
    }
}

/// Pillarbox detector and remover.
pub struct PillarboxRemover;

impl PillarboxRemover {
    /// Detect left and right pillarbox bar widths in a frame.
    ///
    /// Returns `(left_bar_px, right_bar_px)`.
    /// A bar is detected if columns are near-constant (low variance) and dark (<0.1).
    #[must_use]
    #[allow(dead_code)]
    pub fn detect_bars(frame: &[f32], w: u32, h: u32) -> (u32, u32) {
        let w = w as usize;
        let h = h as usize;
        if w == 0 || h == 0 {
            return (0, 0);
        }

        let bar_threshold = 0.1f32;
        let variance_threshold = 0.005f32;

        // Detect left bar
        let mut left_bar = 0u32;
        'left: for x in 0..w {
            let col: Vec<f32> = (0..h).map(|y| frame[y * w + x]).collect();
            let mean = col.iter().sum::<f32>() / h as f32;
            let variance = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            if mean < bar_threshold && variance < variance_threshold {
                left_bar = x as u32 + 1;
            } else {
                break 'left;
            }
        }

        // Detect right bar
        let mut right_bar = 0u32;
        'right: for x in (0..w).rev() {
            let col: Vec<f32> = (0..h).map(|y| frame[y * w + x]).collect();
            let mean = col.iter().sum::<f32>() / h as f32;
            let variance = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            if mean < bar_threshold && variance < variance_threshold {
                right_bar = (w - x) as u32;
            } else {
                break 'right;
            }
        }

        (left_bar, right_bar)
    }

    /// Remove pillarbox bars and return the cropped frame.
    #[must_use]
    #[allow(dead_code)]
    pub fn remove(frame: &[f32], w: u32, h: u32, left: u32, right: u32) -> Vec<f32> {
        let w = w as usize;
        let h = h as usize;
        let left = left as usize;
        let right = right as usize;

        if left + right >= w {
            return Vec::new();
        }
        let new_w = w - left - right;
        let mut out = Vec::with_capacity(new_w * h);

        for y in 0..h {
            out.extend_from_slice(&frame[y * w + left..y * w + left + new_w]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient(w: usize, h: usize) -> Vec<f32> {
        (0..w * h).map(|i| (i % w) as f32 / w as f32).collect()
    }

    #[test]
    fn test_compute_energy_uniform() {
        let src = vec![0.5f32; 16];
        let energy = SeamCarver::compute_energy(&src, 4, 4);
        for &e in &energy {
            assert!(e.abs() < 1e-5);
        }
    }

    #[test]
    fn test_compute_energy_gradient() {
        let src = make_gradient(4, 4);
        let energy = SeamCarver::compute_energy(&src, 4, 4);
        assert_eq!(energy.len(), 16);
        // Interior pixels should have positive energy
        assert!(energy[1 * 4 + 1] > 0.0);
    }

    #[test]
    fn test_find_vertical_seam_length() {
        let src = make_gradient(4, 4);
        let energy = SeamCarver::compute_energy(&src, 4, 4);
        let seam = SeamCarver::find_vertical_seam(&energy, 4, 4);
        assert_eq!(seam.len(), 4);
    }

    #[test]
    fn test_find_vertical_seam_valid_columns() {
        let src = make_gradient(4, 4);
        let energy = SeamCarver::compute_energy(&src, 4, 4);
        let seam = SeamCarver::find_vertical_seam(&energy, 4, 4);
        for &col in &seam {
            assert!(col < 4);
        }
    }

    #[test]
    fn test_remove_seam_reduces_width() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let seam = vec![1u32; 4]; // remove column 1 from each row
        let dst = SeamCarver::remove_seam(&src, 4, 4, &seam);
        assert_eq!(dst.len(), 12); // 3*4
    }

    #[test]
    fn test_carve_to_width() {
        let src = make_gradient(6, 4);
        let dst = SeamCarver::carve_to_width(&src, 6, 4, 4);
        assert_eq!(dst.len(), 16); // 4*4
    }

    #[test]
    fn test_carve_to_width_no_op_when_wider() {
        let src = make_gradient(4, 4);
        let dst = SeamCarver::carve_to_width(&src, 4, 4, 6);
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_adaptive_scaler_empty() {
        let dst = AdaptiveScaler::scale(&[], 0, 0, 4, 4, &[]);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_adaptive_scaler_no_reduction() {
        let src = make_gradient(4, 4);
        let dst = AdaptiveScaler::scale(&src, 4, 4, 4, 4, &[]);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn test_pillarbox_detect_no_bars() {
        let src = vec![0.5f32; 16]; // 4x4 gray, no bars
        let (l, r) = PillarboxRemover::detect_bars(&src, 4, 4);
        assert_eq!(l, 0);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_pillarbox_detect_left_bar() {
        let mut src = vec![0.5f32; 16]; // 4x4
                                        // Make leftmost column black
        for y in 0..4 {
            src[y * 4] = 0.0;
        }
        let (l, _r) = PillarboxRemover::detect_bars(&src, 4, 4);
        assert_eq!(l, 1);
    }

    #[test]
    fn test_pillarbox_remove() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let dst = PillarboxRemover::remove(&src, 4, 4, 1, 1);
        assert_eq!(dst.len(), 8); // 2*4
    }
}
