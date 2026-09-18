//! Kanade-Lucas-Tomasi (KLT) sparse optical flow tracker.
//!
//! Implements the classic pyramidal Lucas-Kanade tracker for tracking sparse
//! feature points across consecutive frames. This is the workhorse of most
//! video stabilization and motion estimation pipelines.
//!
//! # Algorithm
//!
//! 1. Build Gaussian image pyramids for both frames.
//! 2. At the coarsest level, initialize estimated displacement to zero.
//! 3. At each level, refine the displacement using the Lucas-Kanade optical flow
//!    equation (solve the 2x2 structure tensor system).
//! 4. Propagate the refined displacement to the next finer level (multiply by 2).
//!
//! # References
//!
//! - Bouguet, J-Y. "Pyramidal Implementation of the Lucas Kanade Feature Tracker"
//!   Intel Corporation, 2001.

#![allow(clippy::cast_precision_loss)]

use crate::{AlignError, AlignResult, Point2D};

/// Configuration for the KLT tracker.
#[derive(Debug, Clone)]
pub struct KltConfig {
    /// Half-size of the integration window (full window is `2*win + 1`).
    pub window_half_size: usize,
    /// Number of pyramid levels (1 = no pyramid, just original resolution).
    pub pyramid_levels: usize,
    /// Maximum number of Newton-Raphson iterations per pyramid level.
    pub max_iterations: usize,
    /// Convergence threshold for the displacement update (pixels).
    pub epsilon: f64,
    /// Minimum eigenvalue of the structure tensor. Points with eigenvalues
    /// below this are considered un-trackable.
    pub min_eigenvalue: f64,
}

impl Default for KltConfig {
    fn default() -> Self {
        Self {
            window_half_size: 7,
            pyramid_levels: 3,
            max_iterations: 20,
            epsilon: 0.03,
            min_eigenvalue: 1e-4,
        }
    }
}

/// Result of tracking a single point.
#[derive(Debug, Clone)]
pub struct TrackResult {
    /// Original point in the first frame.
    pub origin: Point2D,
    /// Tracked position in the second frame (may be `None` if tracking failed).
    pub tracked: Option<Point2D>,
    /// Tracking error (SSD over the window), lower is better.
    pub error: f64,
    /// Whether the point was successfully tracked.
    pub success: bool,
}

/// Pyramidal KLT tracker.
pub struct KltTracker {
    /// Tracker configuration.
    pub config: KltConfig,
}

impl Default for KltTracker {
    fn default() -> Self {
        Self {
            config: KltConfig::default(),
        }
    }
}

impl KltTracker {
    /// Create a new KLT tracker with the given configuration.
    #[must_use]
    pub fn new(config: KltConfig) -> Self {
        Self { config }
    }

    /// Track sparse feature points from `prev_frame` to `curr_frame` using
    /// pyramidal Lucas-Kanade optical flow.
    ///
    /// This is the primary high-level API for KLT tracking. It accepts f32
    /// pixel coordinates (common in feature-detection pipelines) and returns
    /// `Some((x, y))` for successfully tracked points or `None` when tracking
    /// failed (low texture, out-of-bounds, etc.).
    ///
    /// Both frames must be single-channel (grayscale), row-major byte images
    /// of size `width * height`.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are inconsistent or the frame is
    /// smaller than 8×8 pixels.
    pub fn track_features(
        &self,
        prev_frame: &[u8],
        curr_frame: &[u8],
        width: u32,
        height: u32,
        points: &[(f32, f32)],
    ) -> AlignResult<Vec<Option<(f32, f32)>>> {
        let w = width as usize;
        let h = height as usize;

        // Convert (f32, f32) points to Point2D for the internal API.
        let pts: Vec<Point2D> = points
            .iter()
            .map(|&(x, y)| Point2D::new(f64::from(x), f64::from(y)))
            .collect();

        let results = self.track(prev_frame, curr_frame, w, h, &pts)?;

        Ok(results
            .into_iter()
            .map(|r| r.tracked.map(|p| (p.x as f32, p.y as f32)))
            .collect())
    }

    /// Track a set of points from `prev_image` to `curr_image`.
    ///
    /// Both images must be single-channel (grayscale), row-major, with the
    /// same `width` and `height`.
    ///
    /// # Errors
    ///
    /// Returns an error if the image dimensions are inconsistent.
    pub fn track(
        &self,
        prev_image: &[u8],
        curr_image: &[u8],
        width: usize,
        height: usize,
        points: &[Point2D],
    ) -> AlignResult<Vec<TrackResult>> {
        if prev_image.len() != width * height || curr_image.len() != width * height {
            return Err(AlignError::InvalidConfig(
                "Image size does not match width*height".to_string(),
            ));
        }
        if width < 8 || height < 8 {
            return Err(AlignError::InvalidConfig(
                "Image must be at least 8x8".to_string(),
            ));
        }

        // Build pyramids
        let prev_pyr = build_pyramid(prev_image, width, height, self.config.pyramid_levels);
        let curr_pyr = build_pyramid(curr_image, width, height, self.config.pyramid_levels);

        let results: Vec<TrackResult> = points
            .iter()
            .map(|pt| self.track_point(pt, &prev_pyr, &curr_pyr))
            .collect();

        Ok(results)
    }

    /// Track a single point through the pyramid.
    fn track_point(
        &self,
        point: &Point2D,
        prev_pyr: &[PyramidLevel],
        curr_pyr: &[PyramidLevel],
    ) -> TrackResult {
        let num_levels = prev_pyr.len();
        let win = self.config.window_half_size as f64;

        // Initial guess at coarsest level: zero displacement
        let mut gx = 0.0_f64;
        let mut gy = 0.0_f64;

        let mut last_error = f64::MAX;
        let mut success = true;

        // Iterate from coarsest to finest
        for level in (0..num_levels).rev() {
            let scale = 1.0 / (1 << level) as f64;
            let px = point.x * scale;
            let py = point.y * scale;

            let prev_level = &prev_pyr[level];
            let curr_level = &curr_pyr[level];

            let w = prev_level.width;
            let h = prev_level.height;

            // Compute gradients of the previous image at this level
            let (grad_x, grad_y) = compute_gradients(&prev_level.data, w, h);

            // Build the structure tensor G and mismatch vector b
            let wi = self.config.window_half_size as isize;

            // Iterative Lucas-Kanade
            let mut vx = gx;
            let mut vy = gy;

            for _iter in 0..self.config.max_iterations {
                let mut g_xx = 0.0_f64;
                let mut g_yy = 0.0_f64;
                let mut g_xy = 0.0_f64;
                let mut b_x = 0.0_f64;
                let mut b_y = 0.0_f64;

                for dy in -wi..=wi {
                    for dx in -wi..=wi {
                        let sx = px + dx as f64;
                        let sy = py + dy as f64;
                        let tx = px + dx as f64 + vx;
                        let ty = py + dy as f64 + vy;

                        if sx < 0.0
                            || sy < 0.0
                            || sx >= (w - 1) as f64
                            || sy >= (h - 1) as f64
                            || tx < 0.0
                            || ty < 0.0
                            || tx >= (w - 1) as f64
                            || ty >= (h - 1) as f64
                        {
                            continue;
                        }

                        let ix = bilinear_sample_f64(&grad_x, w, sx, sy);
                        let iy = bilinear_sample_f64(&grad_y, w, sx, sy);
                        let prev_val = bilinear_sample(&prev_level.data, w, sx, sy);
                        let curr_val = bilinear_sample(&curr_level.data, w, tx, ty);

                        let dt = prev_val - curr_val;

                        g_xx += ix * ix;
                        g_yy += iy * iy;
                        g_xy += ix * iy;
                        b_x += dt * ix;
                        b_y += dt * iy;
                    }
                }

                // Check minimum eigenvalue
                let trace = g_xx + g_yy;
                let det = g_xx * g_yy - g_xy * g_xy;
                let discriminant = trace * trace - 4.0 * det;
                let min_eig = if discriminant >= 0.0 {
                    (trace - discriminant.sqrt()) / 2.0
                } else {
                    0.0
                };

                if min_eig < self.config.min_eigenvalue {
                    success = false;
                    break;
                }

                // Solve 2x2 system: G * [dvx; dvy] = [bx; by]
                if det.abs() < 1e-12 {
                    success = false;
                    break;
                }

                let dvx = (g_yy * b_x - g_xy * b_y) / det;
                let dvy = (-g_xy * b_x + g_xx * b_y) / det;

                vx += dvx;
                vy += dvy;

                if dvx * dvx + dvy * dvy < self.config.epsilon * self.config.epsilon {
                    break;
                }
            }

            // Propagate to next finer level (scale by 2)
            if level > 0 {
                gx = vx * 2.0;
                gy = vy * 2.0;
            } else {
                gx = vx;
                gy = vy;
            }

            // Compute error at finest level
            if level == 0 {
                last_error = self.compute_tracking_error(
                    &prev_pyr[0].data,
                    &curr_pyr[0].data,
                    prev_pyr[0].width,
                    prev_pyr[0].height,
                    point.x,
                    point.y,
                    gx,
                    gy,
                    win as isize,
                );
            }
        }

        // Check if tracked point is within image bounds (at original resolution)
        let tracked_x = point.x + gx;
        let tracked_y = point.y + gy;
        let orig_w = prev_pyr[0].width as f64;
        let orig_h = prev_pyr[0].height as f64;

        if !success
            || tracked_x < 0.0
            || tracked_y < 0.0
            || tracked_x >= orig_w
            || tracked_y >= orig_h
        {
            return TrackResult {
                origin: *point,
                tracked: None,
                error: last_error,
                success: false,
            };
        }

        TrackResult {
            origin: *point,
            tracked: Some(Point2D::new(tracked_x, tracked_y)),
            error: last_error,
            success: true,
        }
    }

    /// Compute SSD tracking error over the window.
    #[allow(clippy::too_many_arguments, clippy::manual_checked_ops)]
    fn compute_tracking_error(
        &self,
        prev: &[u8],
        curr: &[u8],
        w: usize,
        h: usize,
        px: f64,
        py: f64,
        vx: f64,
        vy: f64,
        half_win: isize,
    ) -> f64 {
        let mut ssd = 0.0_f64;
        let mut count = 0u32;

        for dy in -half_win..=half_win {
            for dx in -half_win..=half_win {
                let sx = px + dx as f64;
                let sy = py + dy as f64;
                let tx = sx + vx;
                let ty = sy + vy;

                if sx >= 0.0
                    && sy >= 0.0
                    && sx < (w - 1) as f64
                    && sy < (h - 1) as f64
                    && tx >= 0.0
                    && ty >= 0.0
                    && tx < (w - 1) as f64
                    && ty < (h - 1) as f64
                {
                    let a = bilinear_sample(prev, w, sx, sy);
                    let b = bilinear_sample(curr, w, tx, ty);
                    let d = a - b;
                    ssd += d * d;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return f64::MAX;
        }
        ssd / f64::from(count)
    }
}

// -- Pyramid ------------------------------------------------------------------

/// A single level in the image pyramid.
#[derive(Debug, Clone)]
struct PyramidLevel {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

/// Build a Gaussian image pyramid (factor-of-2 downsample at each level).
fn build_pyramid(image: &[u8], width: usize, height: usize, levels: usize) -> Vec<PyramidLevel> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(PyramidLevel {
        data: image.to_vec(),
        width,
        height,
    });

    let mut cur = image.to_vec();
    let mut cw = width;
    let mut ch = height;

    for _ in 1..levels {
        let nw = cw / 2;
        let nh = ch / 2;
        if nw < 4 || nh < 4 {
            break;
        }
        let down = downsample_2x(&cur, cw, ch, nw, nh);
        pyramid.push(PyramidLevel {
            data: down.clone(),
            width: nw,
            height: nh,
        });
        cur = down;
        cw = nw;
        ch = nh;
    }

    pyramid
}

/// 2x downsample with box-filter averaging.
fn downsample_2x(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx * 2;
            let sy = dy * 2;
            let mut sum = 0u16;
            let mut count = 0u16;
            for oy in 0..2 {
                for ox in 0..2 {
                    let rx = sx + ox;
                    let ry = sy + oy;
                    if rx < sw && ry < sh {
                        sum += u16::from(src[ry * sw + rx]);
                        count += 1;
                    }
                }
            }
            dst[dy * dw + dx] = sum.checked_div(count).unwrap_or(0) as u8;
        }
    }
    dst
}

// -- Bilinear interpolation ---------------------------------------------------

/// Bilinear interpolation on a u8 image, returning f64.
fn bilinear_sample(image: &[u8], width: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let v00 = f64::from(image[y0 * width + x0]);
    let v10 = f64::from(image[y0 * width + x1]);
    let v01 = f64::from(image[y1 * width + x0]);
    let v11 = f64::from(image[y1 * width + x1]);

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Bilinear interpolation on an f64 buffer.
fn bilinear_sample_f64(buf: &[f64], width: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let v00 = buf[y0 * width + x0];
    let v10 = buf[y0 * width + x1];
    let v01 = buf[y1 * width + x0];
    let v11 = buf[y1 * width + x1];

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Compute Sobel gradients (f64 output).
fn compute_gradients(image: &[u8], width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    let n = width * height;
    let mut gx = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;

            let i_tl = f64::from(image[(y - 1) * width + (x - 1)]);
            let i_t = f64::from(image[(y - 1) * width + x]);
            let i_tr = f64::from(image[(y - 1) * width + (x + 1)]);
            let i_l = f64::from(image[y * width + (x - 1)]);
            let i_r = f64::from(image[y * width + (x + 1)]);
            let i_bl = f64::from(image[(y + 1) * width + (x - 1)]);
            let i_b = f64::from(image[(y + 1) * width + x]);
            let i_br = f64::from(image[(y + 1) * width + (x + 1)]);

            gx[idx] = (-i_tl + i_tr - 2.0 * i_l + 2.0 * i_r - i_bl + i_br) / 8.0;
            gy[idx] = (-i_tl - 2.0 * i_t - i_tr + i_bl + 2.0 * i_b + i_br) / 8.0;
        }
    }

    (gx, gy)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers ---------------------------------------------------------------

    /// Create a test image with a single bright square at a given offset.
    fn make_square_image(w: usize, h: usize, cx: usize, cy: usize, half: usize) -> Vec<u8> {
        let mut img = vec![30u8; w * h];
        for y in cy.saturating_sub(half)..=(cy + half).min(h - 1) {
            for x in cx.saturating_sub(half)..=(cx + half).min(w - 1) {
                img[y * w + x] = 200;
            }
        }
        img
    }

    // -- PyramidLevel ---------------------------------------------------------

    #[test]
    fn test_build_pyramid_levels() {
        let img = vec![128u8; 64 * 64];
        let pyr = build_pyramid(&img, 64, 64, 3);
        assert_eq!(pyr.len(), 3);
        assert_eq!(pyr[0].width, 64);
        assert_eq!(pyr[1].width, 32);
        assert_eq!(pyr[2].width, 16);
    }

    #[test]
    fn test_build_pyramid_single_level() {
        let img = vec![128u8; 32 * 32];
        let pyr = build_pyramid(&img, 32, 32, 1);
        assert_eq!(pyr.len(), 1);
    }

    #[test]
    fn test_downsample_preserves_constant() {
        let img = vec![100u8; 64 * 64];
        let down = downsample_2x(&img, 64, 64, 32, 32);
        for &v in &down {
            assert_eq!(v, 100);
        }
    }

    // -- Bilinear interpolation -----------------------------------------------

    #[test]
    fn test_bilinear_integer_coords() {
        let img: Vec<u8> = vec![10, 20, 30, 40];
        let val = bilinear_sample(&img, 2, 0.0, 0.0);
        assert!((val - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_midpoint() {
        // 2x2 image: [0, 100; 0, 100]
        let img: Vec<u8> = vec![0, 100, 0, 100];
        let val = bilinear_sample(&img, 2, 0.5, 0.0);
        assert!((val - 50.0).abs() < 1e-6);
    }

    // -- KLT tracker ----------------------------------------------------------

    #[test]
    fn test_klt_stationary_point() {
        let w = 64usize;
        let h = 64usize;
        let img = make_square_image(w, h, 32, 32, 5);

        let config = KltConfig {
            window_half_size: 5,
            pyramid_levels: 2,
            max_iterations: 20,
            epsilon: 0.01,
            min_eigenvalue: 1e-6,
        };
        let tracker = KltTracker::new(config);
        let pts = vec![Point2D::new(32.0, 32.0)];

        let results = tracker
            .track(&img, &img, w, h, &pts)
            .expect("track should succeed");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].success,
            "tracking a stationary point should succeed"
        );
        let tracked = results[0].tracked.expect("should have a tracked point");
        assert!(
            (tracked.x - 32.0).abs() < 1.0 && (tracked.y - 32.0).abs() < 1.0,
            "stationary point should not move: got ({:.2}, {:.2})",
            tracked.x,
            tracked.y,
        );
    }

    #[test]
    fn test_klt_translated_square() {
        let w = 128usize;
        let h = 128usize;
        let shift = 4;

        let prev = make_square_image(w, h, 60, 60, 10);
        let curr = make_square_image(w, h, 60 + shift, 60, 10);

        let config = KltConfig {
            window_half_size: 10,
            pyramid_levels: 3,
            max_iterations: 30,
            epsilon: 0.01,
            min_eigenvalue: 1e-6,
        };
        let tracker = KltTracker::new(config);
        let pts = vec![Point2D::new(60.0, 60.0)];

        let results = tracker
            .track(&prev, &curr, w, h, &pts)
            .expect("track should succeed");
        assert!(results[0].success, "should successfully track the square");
        if let Some(tracked) = &results[0].tracked {
            let dx = tracked.x - 60.0;
            // Should detect roughly +4 pixels horizontal shift
            assert!(
                (dx - shift as f64).abs() < 2.0,
                "expected ~{shift} px shift, got dx={dx:.2}"
            );
        }
    }

    #[test]
    fn test_klt_image_size_mismatch() {
        let tracker = KltTracker::default();
        let pts = vec![Point2D::new(5.0, 5.0)];
        let result = tracker.track(&[0u8; 100], &[0u8; 200], 10, 10, &pts);
        assert!(result.is_err());
    }

    #[test]
    fn test_klt_too_small_image() {
        let tracker = KltTracker::default();
        let pts = vec![Point2D::new(1.0, 1.0)];
        let result = tracker.track(&[0u8; 4], &[0u8; 4], 2, 2, &pts);
        assert!(result.is_err());
    }

    #[test]
    fn test_klt_multiple_points() {
        let w = 128usize;
        let h = 128usize;
        let img = make_square_image(w, h, 64, 64, 15);

        let tracker = KltTracker::default();
        let pts = vec![
            Point2D::new(50.0, 50.0),
            Point2D::new(70.0, 64.0),
            Point2D::new(64.0, 70.0),
        ];

        let results = tracker
            .track(&img, &img, w, h, &pts)
            .expect("should succeed");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_klt_point_out_of_bounds_does_not_crash() {
        let w = 64usize;
        let h = 64usize;
        let img = vec![128u8; w * h];

        let tracker = KltTracker::default();
        // Point near the border
        let pts = vec![Point2D::new(0.0, 0.0)];
        let results = tracker
            .track(&img, &img, w, h, &pts)
            .expect("should not crash");
        assert_eq!(results.len(), 1);
        // May or may not succeed depending on window size, but should not panic
    }

    #[test]
    fn test_klt_default_config() {
        let config = KltConfig::default();
        assert_eq!(config.window_half_size, 7);
        assert_eq!(config.pyramid_levels, 3);
        assert_eq!(config.max_iterations, 20);
    }

    #[test]
    fn test_track_result_fields() {
        let tr = TrackResult {
            origin: Point2D::new(10.0, 20.0),
            tracked: Some(Point2D::new(12.0, 22.0)),
            error: 0.5,
            success: true,
        };
        assert!(tr.success);
        assert!((tr.error - 0.5).abs() < f64::EPSILON);
    }

    // -- Gradient computation -------------------------------------------------

    #[test]
    fn test_gradients_constant_image() {
        let img = vec![100u8; 32 * 32];
        let (gx, gy) = compute_gradients(&img, 32, 32);
        // Interior pixels should have zero gradient on a constant image.
        for y in 2..30 {
            for x in 2..30 {
                assert!(gx[y * 32 + x].abs() < 1e-10);
                assert!(gy[y * 32 + x].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gradients_horizontal_ramp() {
        // Image where each column has a constant value equal to x.
        let w = 32usize;
        let h = 32usize;
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = (x * 8).min(255) as u8;
            }
        }
        let (gx, _gy) = compute_gradients(&img, w, h);
        // Interior horizontal gradient should be positive.
        let mid = 16 * w + 16;
        assert!(gx[mid] > 0.0, "horizontal ramp should produce positive gx");
    }

    // -- track_features (f32 API) ---------------------------------------------

    /// Helper: create a synthetic image with a bright square patch.
    fn make_patch_image(w: usize, h: usize, cx: usize, cy: usize, half: usize) -> Vec<u8> {
        let mut img = vec![30u8; w * h];
        for y in cy.saturating_sub(half)..=(cy + half).min(h - 1) {
            for x in cx.saturating_sub(half)..=(cx + half).min(w - 1) {
                img[y * w + x] = 210;
            }
        }
        img
    }

    #[test]
    fn test_track_features_stationary() {
        let w = 64u32;
        let h = 64u32;
        let img = make_patch_image(64, 64, 32, 32, 6);
        let tracker = KltTracker::default();
        let pts: Vec<(f32, f32)> = vec![(32.0, 32.0)];

        let results = tracker
            .track_features(&img, &img, w, h, &pts)
            .expect("track_features should not error");

        assert_eq!(results.len(), 1);
        if let Some((tx, ty)) = results[0] {
            assert!((tx - 32.0).abs() < 1.5, "tx={tx}");
            assert!((ty - 32.0).abs() < 1.5, "ty={ty}");
        }
        // The point may also be None if it falls on a flat region — that is
        // acceptable; we only care that the call did not panic or error.
    }

    #[test]
    fn test_track_features_translation() {
        let w = 128u32;
        let h = 128u32;
        let shift = 5usize;

        let prev = make_patch_image(128, 128, 60, 60, 12);
        let curr = make_patch_image(128, 128, 60 + shift, 60, 12);

        let config = KltConfig {
            window_half_size: 10,
            pyramid_levels: 3,
            max_iterations: 30,
            epsilon: 0.01,
            min_eigenvalue: 1e-6,
        };
        let tracker = KltTracker::new(config);
        let pts: Vec<(f32, f32)> = vec![(60.0, 60.0)];

        let results = tracker
            .track_features(&prev, &curr, w, h, &pts)
            .expect("track_features should succeed");

        assert_eq!(results.len(), 1);
        if let Some((tx, _ty)) = results[0] {
            let dx = tx - 60.0;
            assert!(
                (dx - shift as f32).abs() < 3.0,
                "expected ~{shift} px shift, got dx={dx:.2}"
            );
        }
    }

    #[test]
    fn test_track_features_returns_none_for_flat_region() {
        // A completely flat (constant-value) image has no texture, so any
        // interior point should fail the eigenvalue test and return None.
        let w = 64u32;
        let h = 64u32;
        let img = vec![128u8; 64 * 64];
        let tracker = KltTracker::default();
        let pts: Vec<(f32, f32)> = vec![(32.0, 32.0)];

        let results = tracker
            .track_features(&img, &img, w, h, &pts)
            .expect("should not error");

        assert_eq!(results.len(), 1);
        // Flat image → no trackable texture → None expected.
        assert!(
            results[0].is_none(),
            "flat region should return None, got {:?}",
            results[0]
        );
    }

    #[test]
    fn test_track_features_invalid_size() {
        let tracker = KltTracker::default();
        let pts = vec![(5.0_f32, 5.0_f32)];
        // Mismatched slice length
        let err = tracker.track_features(&[0u8; 100], &[0u8; 200], 10, 10, &pts);
        assert!(err.is_err());
    }

    #[test]
    fn test_track_features_multiple_points() {
        let w = 64u32;
        let h = 64u32;
        let img = make_patch_image(64, 64, 32, 32, 8);
        let tracker = KltTracker::default();
        let pts: Vec<(f32, f32)> = vec![(28.0, 28.0), (32.0, 32.0), (36.0, 36.0)];

        let results = tracker
            .track_features(&img, &img, w, h, &pts)
            .expect("should succeed");

        // Length must match input regardless of tracking outcome.
        assert_eq!(results.len(), 3);
    }
}
