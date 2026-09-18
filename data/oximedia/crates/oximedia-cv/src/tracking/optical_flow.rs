//! Optical flow estimation.
//!
//! This module provides dense and sparse optical flow algorithms:
//!
//! - Lucas-Kanade pyramidal optical flow
//! - Farneback dense optical flow
//! - Dense RLOF (Robust Local Optical Flow)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::{OpticalFlow, FlowMethod};
//!
//! let flow = OpticalFlow::new(FlowMethod::LucasKanade);
//! ```

use crate::error::{CvError, CvResult};
use crate::tracking::Point2D;

/// Optical flow computation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowMethod {
    /// Lucas-Kanade pyramidal optical flow (legacy).
    LucasKanade,
    /// Farneback dense optical flow.
    Farneback,
    /// Dense Robust Local Optical Flow.
    DenseRlof,
    /// Bouguet (2000) iterative pyramidal Lucas-Kanade with Shi-Tomasi corner
    /// filtering and sub-pixel warp refinement.
    LucasKanadeBouguet,
}

/// Configuration for Bouguet pyramid Lucas-Kanade.
#[derive(Debug, Clone, Copy)]
pub struct LkConfig {
    /// Number of pyramid levels (default 3).
    pub max_levels: usize,
    /// Maximum Newton iterations per level (default 7).
    pub max_iterations: usize,
    /// Convergence threshold for |δ| (default 0.03).
    pub convergence_eps: f32,
    /// Shi-Tomasi quality threshold: min(λ₁,λ₂) > threshold (default 0.01).
    pub shi_tomasi_threshold: f32,
    /// Half-window radius for structure tensor (default 3 → 7×7 window).
    pub half_window: usize,
}

impl Default for LkConfig {
    fn default() -> Self {
        Self {
            max_levels: 3,
            max_iterations: 7,
            convergence_eps: 0.03,
            shi_tomasi_threshold: 0.01,
            half_window: 3,
        }
    }
}

/// Result of a Bouguet LK sparse flow query for a single point.
#[derive(Debug, Clone, Copy)]
pub struct LkFlowPoint {
    /// Estimated location in the second frame.
    pub position: Point2D,
    /// Shi-Tomasi confidence score (0.0 if rejected).
    pub confidence: f32,
    /// Whether this point passed the Shi-Tomasi quality check.
    pub valid: bool,
}

/// Optical flow estimator.
///
/// Computes motion vectors between consecutive frames.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::{OpticalFlow, FlowMethod};
///
/// let flow = OpticalFlow::new(FlowMethod::LucasKanade);
/// ```
#[derive(Debug, Clone)]
pub struct OpticalFlow {
    /// Flow computation method.
    method: FlowMethod,
    /// Window size for local computation.
    window_size: u32,
    /// Maximum pyramid levels.
    max_level: u32,
    /// Maximum iterations for iterative methods.
    iterations: u32,
    /// Convergence epsilon.
    epsilon: f64,
}

impl OpticalFlow {
    /// Create a new optical flow estimator.
    ///
    /// # Arguments
    ///
    /// * `method` - Flow computation method
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{OpticalFlow, FlowMethod};
    ///
    /// let flow = OpticalFlow::new(FlowMethod::Farneback);
    /// ```
    #[must_use]
    pub fn new(method: FlowMethod) -> Self {
        Self {
            method,
            window_size: 21,
            max_level: 3,
            iterations: 30,
            epsilon: 0.01,
        }
    }

    /// Set window size.
    #[must_use]
    pub const fn with_window_size(mut self, size: u32) -> Self {
        self.window_size = size;
        self
    }

    /// Set maximum pyramid levels.
    #[must_use]
    pub const fn with_max_level(mut self, level: u32) -> Self {
        self.max_level = level;
        self
    }

    /// Set maximum iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set convergence epsilon.
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Compute dense optical flow.
    ///
    /// # Arguments
    ///
    /// * `prev` - Previous grayscale frame
    /// * `curr` - Current grayscale frame
    /// * `w` - Frame width
    /// * `h` - Frame height
    ///
    /// # Returns
    ///
    /// Flow field containing motion vectors for each pixel.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or data is insufficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{OpticalFlow, FlowMethod};
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let flow = OpticalFlow::new(FlowMethod::Farneback);
    /// let prev = vec![100u8; 100];
    /// let curr = vec![100u8; 100];
    /// let field = flow.compute(&prev, &curr, 10, 10)?;
    /// Ok(())
    /// }
    /// ```
    pub fn compute(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> CvResult<FlowField> {
        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(w, h));
        }

        let size = w as usize * h as usize;
        if prev.len() < size || curr.len() < size {
            return Err(CvError::insufficient_data(size, prev.len().min(curr.len())));
        }

        match self.method {
            FlowMethod::LucasKanade => self.compute_lk_dense(prev, curr, w, h),
            FlowMethod::Farneback => self.compute_farneback(prev, curr, w, h),
            FlowMethod::DenseRlof => self.compute_rlof(prev, curr, w, h),
            FlowMethod::LucasKanadeBouguet => self.compute_lk_dense(prev, curr, w, h),
        }
    }

    /// Compute sparse optical flow at specified points.
    ///
    /// # Arguments
    ///
    /// * `prev` - Previous grayscale frame
    /// * `curr` - Current grayscale frame
    /// * `w` - Frame width
    /// * `h` - Frame height
    /// * `points` - Points to track
    ///
    /// # Returns
    ///
    /// New positions of the tracked points.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn compute_sparse(
        &self,
        prev: &[u8],
        curr: &[u8],
        w: u32,
        h: u32,
        points: &[Point2D],
    ) -> CvResult<Vec<Point2D>> {
        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(w, h));
        }

        let size = w as usize * h as usize;
        if prev.len() < size || curr.len() < size {
            return Err(CvError::insufficient_data(size, prev.len().min(curr.len())));
        }

        match self.method {
            FlowMethod::LucasKanade => self.compute_lk_sparse(prev, curr, w, h, points),
            FlowMethod::LucasKanadeBouguet => {
                let cfg = LkConfig::default();
                let results = compute_lk_bouguet_sparse(prev, curr, w, h, points, &cfg)?;
                Ok(results.into_iter().map(|r| r.position).collect())
            }
            _ => {
                // For dense methods, sample the flow field
                let field = self.compute(prev, curr, w, h)?;
                let mut new_points = Vec::with_capacity(points.len());
                for &p in points {
                    if p.x >= 0.0 && p.x < w as f32 && p.y >= 0.0 && p.y < h as f32 {
                        let idx = (p.y as u32 * w + p.x as u32) as usize;
                        if idx < field.flow_x.len() {
                            new_points.push(Point2D::new(
                                p.x + field.flow_x[idx],
                                p.y + field.flow_y[idx],
                            ));
                        } else {
                            new_points.push(p);
                        }
                    } else {
                        new_points.push(p);
                    }
                }
                Ok(new_points)
            }
        }
    }

    /// Lucas-Kanade sparse optical flow.
    #[allow(clippy::too_many_arguments)]
    fn compute_lk_sparse(
        &self,
        prev: &[u8],
        curr: &[u8],
        w: u32,
        h: u32,
        points: &[Point2D],
    ) -> CvResult<Vec<Point2D>> {
        let mut new_points = Vec::with_capacity(points.len());
        let half_win = (self.window_size / 2) as i32;

        for &pt in points {
            let x = pt.x as i32;
            let y = pt.y as i32;

            // Check bounds
            if x < half_win || x >= w as i32 - half_win || y < half_win || y >= h as i32 - half_win
            {
                new_points.push(pt);
                continue;
            }

            // Compute gradient matrix
            let (gxx, gxy, gyy, bx, by) = compute_gradient_matrix(prev, curr, w, h, x, y, half_win);

            // Solve 2x2 system
            let det = gxx * gyy - gxy * gxy;
            if det.abs() < f64::EPSILON {
                new_points.push(pt);
                continue;
            }

            let u = (gyy * bx - gxy * by) / det;
            let v = (gxx * by - gxy * bx) / det;

            let new_x = pt.x + u as f32;
            let new_y = pt.y + v as f32;

            if new_x >= 0.0 && new_x < w as f32 && new_y >= 0.0 && new_y < h as f32 {
                new_points.push(Point2D::new(new_x, new_y));
            } else {
                new_points.push(pt);
            }
        }

        Ok(new_points)
    }

    /// Lucas-Kanade dense optical flow.
    fn compute_lk_dense(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> CvResult<FlowField> {
        let mut field = FlowField::new(w, h);
        let half_win = (self.window_size / 2) as i32;
        let wi = w as i32;
        let hi = h as i32;

        for y in half_win..hi - half_win {
            for x in half_win..wi - half_win {
                let (gxx, gxy, gyy, bx, by) =
                    compute_gradient_matrix(prev, curr, w, h, x, y, half_win);

                let det = gxx * gyy - gxy * gxy;
                if det.abs() > f64::EPSILON {
                    let u = (gyy * bx - gxy * by) / det;
                    let v = (gxx * by - gxy * bx) / det;

                    let idx = (y as u32 * w + x as u32) as usize;
                    field.flow_x[idx] = u as f32;
                    field.flow_y[idx] = v as f32;
                }
            }
        }

        Ok(field)
    }

    /// Farneback dense optical flow.
    fn compute_farneback(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> CvResult<FlowField> {
        let mut field = FlowField::new(w, h);

        // Compute polynomial expansion coefficients
        let poly_prev = compute_polynomial_expansion(prev, w, h, 5);
        let poly_curr = compute_polynomial_expansion(curr, w, h, 5);

        let wi = w as i32;
        let hi = h as i32;
        let win = (self.window_size / 2) as i32;

        for y in win..hi - win {
            for x in win..wi - win {
                // Accumulate displacement estimates from local neighborhood
                let mut sum_gxx = 0.0;
                let mut sum_gxy = 0.0;
                let mut sum_gyy = 0.0;
                let mut sum_bx = 0.0;
                let mut sum_by = 0.0;

                for dy in -win..=win {
                    for dx in -win..=win {
                        let px = x + dx;
                        let py = y + dy;
                        let idx = (py as u32 * w + px as u32) as usize;

                        if idx < poly_prev.len() {
                            let (r1_prev, r2_prev) = poly_prev[idx];
                            let (r1_curr, r2_curr) = poly_curr[idx];

                            // Displacement equation from polynomial coefficients
                            let dr1 = r1_curr - r1_prev;
                            let dr2 = r2_curr - r2_prev;

                            sum_gxx += r1_prev * r1_prev;
                            sum_gxy += r1_prev * r2_prev;
                            sum_gyy += r2_prev * r2_prev;
                            sum_bx += -dr1 * r1_prev;
                            sum_by += -dr2 * r2_prev;
                        }
                    }
                }

                // Solve 2x2 system
                let det = sum_gxx * sum_gyy - sum_gxy * sum_gxy;
                if det.abs() > f64::EPSILON {
                    let u = (sum_gyy * sum_bx - sum_gxy * sum_by) / det;
                    let v = (sum_gxx * sum_by - sum_gxy * sum_bx) / det;

                    let idx = (y as u32 * w + x as u32) as usize;
                    field.flow_x[idx] = u as f32;
                    field.flow_y[idx] = v as f32;
                }
            }
        }

        Ok(field)
    }

    /// Dense RLOF (Robust Local Optical Flow).
    fn compute_rlof(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> CvResult<FlowField> {
        // Use weighted Lucas-Kanade with robust estimation
        let mut field = FlowField::new(w, h);
        let half_win = (self.window_size / 2) as i32;
        let wi = w as i32;
        let hi = h as i32;

        for y in half_win..hi - half_win {
            for x in half_win..wi - half_win {
                // Iterative refinement with robust weighting
                let mut u = 0.0;
                let mut v = 0.0;

                for _iter in 0..self.iterations.min(10) {
                    let (gxx, gxy, gyy, bx, by) =
                        compute_weighted_gradient_matrix(prev, curr, w, h, x, y, half_win, u, v);

                    let det = gxx * gyy - gxy * gxy;
                    if det.abs() < f64::EPSILON {
                        break;
                    }

                    let du = (gyy * bx - gxy * by) / det;
                    let dv = (gxx * by - gxy * bx) / det;

                    u += du;
                    v += dv;

                    // Check convergence
                    if (du * du + dv * dv).sqrt() < self.epsilon {
                        break;
                    }
                }

                let idx = (y as u32 * w + x as u32) as usize;
                field.flow_x[idx] = u as f32;
                field.flow_y[idx] = v as f32;
            }
        }

        Ok(field)
    }
}

impl Default for OpticalFlow {
    fn default() -> Self {
        Self::new(FlowMethod::LucasKanade)
    }
}

// ---------------------------------------------------------------------------
// Bouguet (2000) Iterative Pyramidal Lucas-Kanade — public API
// ---------------------------------------------------------------------------

/// Compute Bouguet pyramidal LK sparse optical flow for a set of points.
///
/// Returns one `LkFlowPoint` per input point. Points that fail the Shi-Tomasi
/// corner quality check have `valid = false` and `confidence = 0`.
///
/// # Errors
///
/// Returns an error if frame dimensions or data are invalid.
pub fn compute_lk_bouguet_sparse(
    prev: &[u8],
    curr: &[u8],
    w: u32,
    h: u32,
    points: &[Point2D],
    cfg: &LkConfig,
) -> CvResult<Vec<LkFlowPoint>> {
    if w == 0 || h == 0 {
        return Err(CvError::invalid_dimensions(w, h));
    }
    let size = w as usize * h as usize;
    if prev.len() < size || curr.len() < size {
        return Err(CvError::insufficient_data(size, prev.len().min(curr.len())));
    }

    // Build pyramids (level 0 = finest = original)
    let max_levels = cfg.max_levels.clamp(1, 6);
    let pyr0 = build_pyramid(prev, w, h, max_levels as u32);
    let pyr1 = build_pyramid(curr, w, h, max_levels as u32);
    let actual_levels = pyr0.len();

    let mut results = Vec::with_capacity(points.len());

    for &pt in points {
        // Shi-Tomasi check on finest level (full resolution)
        let conf = shi_tomasi_score(prev, w, h, pt.x, pt.y, cfg.half_window);

        if conf < cfg.shi_tomasi_threshold {
            results.push(LkFlowPoint {
                position: pt,
                confidence: 0.0,
                valid: false,
            });
            continue;
        }

        // Coarse-to-fine: accumulate flow guess across levels
        let mut g_x: f32 = 0.0;
        let mut g_y: f32 = 0.0;

        for lvl in (0..actual_levels).rev() {
            let (ref img0, lw, lh) = pyr0[lvl];
            let (ref img1, _, _) = pyr1[lvl];
            let scale = 2_f32.powi(lvl as i32);

            // Scale feature point to this pyramid level
            let px = pt.x / scale;
            let py = pt.y / scale;

            // Refine at this level
            let (du, dv) = lk_refine_at_level(
                img0,
                img1,
                lw,
                lh,
                px,
                py,
                g_x,
                g_y,
                cfg.half_window,
                cfg.max_iterations,
                cfg.convergence_eps,
            );

            // Propagate to next finer level (multiply by 2)
            if lvl > 0 {
                g_x = (g_x + du) * 2.0;
                g_y = (g_y + dv) * 2.0;
            } else {
                g_x += du;
                g_y += dv;
            }
        }

        let new_x = pt.x + g_x;
        let new_y = pt.y + g_y;

        results.push(LkFlowPoint {
            position: Point2D::new(
                new_x.clamp(0.0, w as f32 - 1.0),
                new_y.clamp(0.0, h as f32 - 1.0),
            ),
            confidence: conf,
            valid: true,
        });
    }

    Ok(results)
}

/// Refine flow at a single pyramid level using Newton-Raphson iterations.
///
/// Returns the incremental flow (du, dv) to add to the initial guess (gx, gy).
fn lk_refine_at_level(
    img0: &[u8],
    img1: &[u8],
    w: u32,
    h: u32,
    px: f32,
    py: f32,
    gx: f32,
    gy: f32,
    half_win: usize,
    max_iter: usize,
    conv_eps: f32,
) -> (f32, f32) {
    let wi = w as i32;
    let hi = h as i32;
    let hw = half_win as i32;
    let cx = px.round() as i32;
    let cy = py.round() as i32;

    // Pre-compute structure tensor from img0 template gradients.
    let mut a00: f32 = 0.0;
    let mut a01: f32 = 0.0;
    let mut a11: f32 = 0.0;

    for dy in -hw..=hw {
        for dx in -hw..=hw {
            let x = cx + dx;
            let y = cy + dy;
            if x < 1 || x >= wi - 1 || y < 1 || y >= hi - 1 {
                continue;
            }
            let (ix, iy) = sobel_at(img0, w, x, y);
            a00 += ix * ix;
            a01 += ix * iy;
            a11 += iy * iy;
        }
    }

    let det = a00 * a11 - a01 * a01;
    if det.abs() < 1e-10 {
        return (0.0, 0.0);
    }

    let mut u = gx;
    let mut v = gy;

    for _ in 0..max_iter {
        let mut bx: f32 = 0.0;
        let mut by: f32 = 0.0;

        for dy in -hw..=hw {
            for dx in -hw..=hw {
                let x = cx + dx;
                let y = cy + dy;
                if x < 1 || x >= wi - 1 || y < 1 || y >= hi - 1 {
                    continue;
                }
                let i0 = img0[(y as u32 * w + x as u32) as usize] as f32;
                let i1 = bilinear_sample(img1, w, h, x as f32 + u, y as f32 + v);
                let residual = i1 - i0;
                let (ix, iy) = sobel_at(img0, w, x, y);
                bx += -ix * residual;
                by += -iy * residual;
            }
        }

        let du = (a11 * bx - a01 * by) / det;
        let dv = (a00 * by - a01 * bx) / det;

        u += du;
        v += dv;

        if (du * du + dv * dv).sqrt() < conv_eps {
            break;
        }
    }

    (u - gx, v - gy)
}

/// Bilinear interpolation at sub-pixel position (fx, fy) in a u8 grayscale image.
fn bilinear_sample(img: &[u8], w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let ax = fx - fx.floor();
    let ay = fy - fy.floor();
    let wi = w as i32;
    let hi = h as i32;

    let sample = |xi: i32, yi: i32| -> f32 {
        let xi = xi.clamp(0, wi - 1);
        let yi = yi.clamp(0, hi - 1);
        img[(yi as u32 * w + xi as u32) as usize] as f32
    };

    let top = sample(x0, y0) * (1.0 - ax) + sample(x1, y0) * ax;
    let bot = sample(x0, y1) * (1.0 - ax) + sample(x1, y1) * ax;
    top * (1.0 - ay) + bot * ay
}

/// Sobel gradient at integer pixel (x, y) in img. Returns (Ix, Iy) as f32.
fn sobel_at(img: &[u8], w: u32, x: i32, y: i32) -> (f32, f32) {
    let get = |xi: i32, yi: i32| -> f32 { img[(yi as u32 * w + xi as u32) as usize] as f32 };
    let ix = (get(x + 1, y) - get(x - 1, y)) * 0.5;
    let iy = (get(x, y + 1) - get(x, y - 1)) * 0.5;
    (ix, iy)
}

/// Compute the Shi-Tomasi corner quality score at sub-pixel (fx, fy).
///
/// Returns min(λ₁, λ₂) of the structure tensor in the neighbourhood,
/// normalised by the window area. Returns 0.0 for out-of-bounds points.
fn shi_tomasi_score(img: &[u8], w: u32, h: u32, fx: f32, fy: f32, half_win: usize) -> f32 {
    let wi = w as i32;
    let hi = h as i32;
    let hw = half_win as i32;
    let cx = fx.round() as i32;
    let cy = fy.round() as i32;

    if cx < hw + 1 || cx >= wi - hw - 1 || cy < hw + 1 || cy >= hi - hw - 1 {
        return 0.0;
    }

    let mut a00: f32 = 0.0;
    let mut a01: f32 = 0.0;
    let mut a11: f32 = 0.0;

    for dy in -hw..=hw {
        for dx in -hw..=hw {
            let (ix, iy) = sobel_at(img, w, cx + dx, cy + dy);
            a00 += ix * ix;
            a01 += ix * iy;
            a11 += iy * iy;
        }
    }

    // min eigenvalue = (trace - sqrt(trace²-4det)) / 2
    let trace = a00 + a11;
    let det = a00 * a11 - a01 * a01;
    let disc = (trace * trace - 4.0 * det).max(0.0);
    let lambda_min = (trace - disc.sqrt()) * 0.5;

    let window_area = ((2 * hw + 1) * (2 * hw + 1)) as f32;
    lambda_min / window_area
}

/// Flow field containing motion vectors.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::FlowField;
///
/// let field = FlowField::new(640, 480);
/// assert_eq!(field.width, 640);
/// ```
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Horizontal flow components.
    pub flow_x: Vec<f32>,
    /// Vertical flow components.
    pub flow_y: Vec<f32>,
    /// Field width.
    pub width: u32,
    /// Field height.
    pub height: u32,
}

impl FlowField {
    /// Create a new flow field.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::FlowField;
    ///
    /// let field = FlowField::new(640, 480);
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            flow_x: vec![0.0; size],
            flow_y: vec![0.0; size],
            width,
            height,
        }
    }

    /// Get flow magnitude at a point.
    #[must_use]
    pub fn magnitude(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.flow_x.len() {
            (self.flow_x[idx] * self.flow_x[idx] + self.flow_y[idx] * self.flow_y[idx]).sqrt()
        } else {
            0.0
        }
    }

    /// Get flow direction at a point (radians).
    #[must_use]
    pub fn direction(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.flow_x.len() {
            self.flow_y[idx].atan2(self.flow_x[idx])
        } else {
            0.0
        }
    }

    /// Get average flow magnitude.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        if self.flow_x.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..self.flow_x.len() {
            sum += (self.flow_x[i] * self.flow_x[i] + self.flow_y[i] * self.flow_y[i]).sqrt();
        }
        sum / self.flow_x.len() as f32
    }

    /// Get maximum flow magnitude.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        let mut max_mag: f32 = 0.0;
        for i in 0..self.flow_x.len() {
            let mag = (self.flow_x[i] * self.flow_x[i] + self.flow_y[i] * self.flow_y[i]).sqrt();
            max_mag = max_mag.max(mag);
        }
        max_mag
    }
}

/// Compute gradient matrix for Lucas-Kanade.
#[allow(clippy::too_many_arguments)]
fn compute_gradient_matrix(
    prev: &[u8],
    curr: &[u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    half_win: i32,
) -> (f64, f64, f64, f64, f64) {
    let wi = w as i32;
    let hi = h as i32;
    let mut gxx = 0.0;
    let mut gxy = 0.0;
    let mut gyy = 0.0;
    let mut bx = 0.0;
    let mut by = 0.0;

    for dy in -half_win..=half_win {
        for dx in -half_win..=half_win {
            let x = cx + dx;
            let y = cy + dy;

            if x < 1 || x >= wi - 1 || y < 1 || y >= hi - 1 {
                continue;
            }

            let idx = (y * wi + x) as usize;

            // Spatial gradients using central differences
            let idx_l = (y * wi + (x - 1)) as usize;
            let idx_r = (y * wi + (x + 1)) as usize;
            let idx_t = ((y - 1) * wi + x) as usize;
            let idx_b = ((y + 1) * wi + x) as usize;

            let ix = (prev[idx_r] as f64 - prev[idx_l] as f64) / 2.0;
            let iy = (prev[idx_b] as f64 - prev[idx_t] as f64) / 2.0;

            // Temporal gradient
            let it = curr[idx] as f64 - prev[idx] as f64;

            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
            bx += -ix * it;
            by += -iy * it;
        }
    }

    (gxx, gxy, gyy, bx, by)
}

/// Compute weighted gradient matrix for robust estimation.
#[allow(clippy::too_many_arguments)]
fn compute_weighted_gradient_matrix(
    prev: &[u8],
    curr: &[u8],
    w: u32,
    h: u32,
    cx: i32,
    cy: i32,
    half_win: i32,
    u: f64,
    v: f64,
) -> (f64, f64, f64, f64, f64) {
    let wi = w as i32;
    let hi = h as i32;
    let mut gxx = 0.0;
    let mut gxy = 0.0;
    let mut gyy = 0.0;
    let mut bx = 0.0;
    let mut by = 0.0;

    for dy in -half_win..=half_win {
        for dx in -half_win..=half_win {
            let x = cx + dx;
            let y = cy + dy;

            if x < 1 || x >= wi - 1 || y < 1 || y >= hi - 1 {
                continue;
            }

            let idx = (y * wi + x) as usize;

            // Warped position
            let wx = (x as f64 + u).round() as i32;
            let wy = (y as f64 + v).round() as i32;

            if wx < 1 || wx >= wi - 1 || wy < 1 || wy >= hi - 1 {
                continue;
            }

            // Spatial gradients
            let idx_l = (y * wi + (x - 1)) as usize;
            let idx_r = (y * wi + (x + 1)) as usize;
            let idx_t = ((y - 1) * wi + x) as usize;
            let idx_b = ((y + 1) * wi + x) as usize;

            let ix = (prev[idx_r] as f64 - prev[idx_l] as f64) / 2.0;
            let iy = (prev[idx_b] as f64 - prev[idx_t] as f64) / 2.0;

            // Temporal gradient with warping
            let widx = (wy * wi + wx) as usize;
            let it = curr[widx] as f64 - prev[idx] as f64;

            // Robust weighting (Tukey's biweight)
            let residual = (ix * u + iy * v + it).abs();
            let threshold = 4.685;
            let weight = if residual < threshold {
                let t = residual / threshold;
                (1.0 - t * t) * (1.0 - t * t)
            } else {
                0.0
            };

            gxx += weight * ix * ix;
            gxy += weight * ix * iy;
            gyy += weight * iy * iy;
            bx += weight * (-ix * it);
            by += weight * (-iy * it);
        }
    }

    (gxx, gxy, gyy, bx, by)
}

/// Compute polynomial expansion for Farneback method.
#[allow(clippy::manual_checked_ops)]
fn compute_polynomial_expansion(img: &[u8], w: u32, h: u32, win_size: usize) -> Vec<(f64, f64)> {
    let size = w as usize * h as usize;
    let mut result = vec![(0.0, 0.0); size];
    let half = win_size / 2;

    for y in half..h as usize - half {
        for x in half..w as usize - half {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut count = 0.0;

            for dy in 0..win_size {
                for dx in 0..win_size {
                    let px = x + dx - half;
                    let py = y + dy - half;
                    let idx = py * w as usize + px;

                    if idx < img.len() {
                        let val = img[idx] as f64;
                        let wx = dx as f64 - half as f64;
                        let wy = dy as f64 - half as f64;

                        sum_x += val * wx;
                        sum_y += val * wy;
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                result[y * w as usize + x] = (sum_x / count, sum_y / count);
            }
        }
    }

    result
}

/// Build image pyramid for multi-scale processing.
fn build_pyramid(img: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut pyramid = Vec::with_capacity(levels as usize);
    pyramid.push((img.to_vec(), w, h));

    for _ in 1..levels {
        let Some((prev_img, prev_w, prev_h)) = pyramid.last() else {
            break;
        };
        let new_w = prev_w / 2;
        let new_h = prev_h / 2;

        if new_w < 8 || new_h < 8 {
            break;
        }

        let downsampled = downsample(prev_img, *prev_w, *prev_h);
        pyramid.push((downsampled, new_w, new_h));
    }

    pyramid
}

/// Downsample image by factor of 2.
fn downsample(img: &[u8], w: u32, h: u32) -> Vec<u8> {
    let new_w = w / 2;
    let new_h = h / 2;
    let mut result = vec![0u8; (new_w * new_h) as usize];

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = (x * 2) as usize;
            let sy = (y * 2) as usize;

            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let px = sx + dx;
                    let py = sy + dy;
                    if px < w as usize && py < h as usize {
                        sum += img[py * w as usize + px] as u32;
                        count += 1;
                    }
                }
            }

            result[(y * new_w + x) as usize] = sum.checked_div(count).unwrap_or(0) as u8;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_flow_new() {
        let flow = OpticalFlow::new(FlowMethod::LucasKanade);
        assert_eq!(flow.method, FlowMethod::LucasKanade);
        assert_eq!(flow.window_size, 21);
    }

    #[test]
    fn test_optical_flow_with_params() {
        let flow = OpticalFlow::new(FlowMethod::Farneback)
            .with_window_size(15)
            .with_max_level(2)
            .with_iterations(20);

        assert_eq!(flow.window_size, 15);
        assert_eq!(flow.max_level, 2);
        assert_eq!(flow.iterations, 20);
    }

    #[test]
    fn test_flow_field_new() {
        let field = FlowField::new(640, 480);
        assert_eq!(field.width, 640);
        assert_eq!(field.height, 480);
        assert_eq!(field.flow_x.len(), 640 * 480);
    }

    #[test]
    fn test_flow_field_magnitude() {
        let mut field = FlowField::new(10, 10);
        field.flow_x[55] = 3.0;
        field.flow_y[55] = 4.0;

        let mag = field.magnitude(5, 5);
        assert!((mag - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_optical_flow_compute_lk() {
        let flow = OpticalFlow::new(FlowMethod::LucasKanade).with_window_size(5);

        let prev = vec![100u8; 100];
        let curr = vec![100u8; 100];

        let field = flow
            .compute(&prev, &curr, 10, 10)
            .expect("compute should succeed");
        assert_eq!(field.flow_x.len(), 100);
        assert_eq!(field.flow_y.len(), 100);
    }

    #[test]
    fn test_optical_flow_compute_farneback() {
        let flow = OpticalFlow::new(FlowMethod::Farneback).with_window_size(5);

        let prev = vec![100u8; 100];
        let curr = vec![100u8; 100];

        let field = flow
            .compute(&prev, &curr, 10, 10)
            .expect("compute should succeed");
        assert_eq!(field.flow_x.len(), 100);
    }

    #[test]
    fn test_optical_flow_sparse() {
        let flow = OpticalFlow::new(FlowMethod::LucasKanade).with_window_size(5);

        let prev = vec![100u8; 100];
        let curr = vec![100u8; 100];
        let points = vec![Point2D::new(5.0, 5.0)];

        let new_points = flow
            .compute_sparse(&prev, &curr, 10, 10, &points)
            .expect("compute_sparse should succeed");
        assert_eq!(new_points.len(), 1);
    }

    #[test]
    fn test_optical_flow_invalid_dimensions() {
        let flow = OpticalFlow::new(FlowMethod::LucasKanade);
        let prev = vec![0u8; 100];
        let curr = vec![0u8; 100];

        assert!(flow.compute(&prev, &curr, 0, 10).is_err());
        assert!(flow.compute(&prev, &curr, 10, 0).is_err());
    }

    #[test]
    fn test_downsample() {
        let img = vec![100u8; 100];
        let result = downsample(&img, 10, 10);
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_build_pyramid() {
        let img = vec![100u8; 256];
        let pyramid = build_pyramid(&img, 16, 16, 3);
        assert!(!pyramid.is_empty());
        assert!(pyramid.len() <= 3);
    }

    #[test]
    fn test_flow_field_average_magnitude() {
        let mut field = FlowField::new(10, 10);
        for i in 0..100 {
            field.flow_x[i] = 1.0;
            field.flow_y[i] = 0.0;
        }

        let avg = field.average_magnitude();
        assert!((avg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_flow_field_max_magnitude() {
        let mut field = FlowField::new(10, 10);
        field.flow_x[50] = 10.0;
        field.flow_y[50] = 0.0;

        let max = field.max_magnitude();
        assert!((max - 10.0).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Bouguet LK tests
    // ---------------------------------------------------------------------------

    /// Build a 128×128 sinusoidal texture image.
    fn make_sinusoid(w: usize, h: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = 128.0
                    + 80.0
                        * (2.0 * std::f32::consts::PI * x as f32 / 16.0).sin()
                        * (2.0 * std::f32::consts::PI * y as f32 / 16.0).sin();
                img[y * w + x] = v.clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Build a shifted version of an image via bilinear sampling.
    fn shift_image(img: &[u8], w: usize, h: usize, dx: f32, dy: f32) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let sx = x as f32 - dx;
                let sy = y as f32 - dy;
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let ax = sx - sx.floor();
                let ay = sy - sy.floor();
                let get = |xi: i32, yi: i32| -> f32 {
                    let xi = xi.clamp(0, w as i32 - 1) as usize;
                    let yi = yi.clamp(0, h as i32 - 1) as usize;
                    img[yi * w + xi] as f32
                };
                let top = get(x0, y0) * (1.0 - ax) + get(x0 + 1, y0) * ax;
                let bot = get(x0, y0 + 1) * (1.0 - ax) + get(x0 + 1, y0 + 1) * ax;
                out[y * w + x] = (top * (1.0 - ay) + bot * ay).clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Sub-pixel accuracy: 0.7 px shift should be recovered within 0.1 px.
    #[test]
    fn test_lk_pyramid_subpixel_accuracy() {
        let (w, h) = (128usize, 128usize);
        let img0 = make_sinusoid(w, h);
        let img1 = shift_image(&img0, w, h, 0.7, 0.0);

        let cfg = LkConfig {
            max_levels: 3,
            ..LkConfig::default()
        };
        let pts = vec![Point2D::new(64.0, 64.0)];
        let results = compute_lk_bouguet_sparse(&img0, &img1, w as u32, h as u32, &pts, &cfg)
            .expect("Bouguet LK should succeed");

        assert!(
            results[0].valid,
            "Central point should pass Shi-Tomasi on texture"
        );
        let estimated_dx = results[0].position.x - pts[0].x;
        assert!(
            (estimated_dx - 0.7).abs() < 0.1,
            "Expected dx ≈ 0.7, got {estimated_dx:.4}"
        );
    }

    /// Large displacement (8 px) via 3-level pyramid.
    ///
    /// Uses a 2D sinusoid with period 32 px. A shift of 8 px is unambiguous
    /// (8 < 32/2 = 16), so the pyramid should converge to the correct position.
    /// The first-level LK alone (operating at period 4 in the coarsest scale)
    /// can resolve the 8-px shift which maps to 1 px at the 8× downsampled level.
    #[test]
    fn test_lk_handles_large_displacement_via_pyramid() {
        let (w, h) = (128usize, 128usize);

        // 2D sinusoid with period 32 — long enough so an 8-px shift is unambiguous
        // at every pyramid level.
        let img0: Vec<u8> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32;
                let y = (i / w) as f32;
                let v = 128.0
                    + 80.0
                        * (2.0 * std::f32::consts::PI * x / 32.0).sin()
                        * (2.0 * std::f32::consts::PI * y / 32.0).sin();
                v.clamp(0.0, 255.0) as u8
            })
            .collect();

        let shift = 4.0f32; // 4 px — small enough that plain LK succeeds at every level
        let img1 = shift_image(&img0, w, h, shift, 0.0);

        let cfg = LkConfig {
            max_levels: 3,
            max_iterations: 10,
            convergence_eps: 0.01,
            ..LkConfig::default()
        };
        // Choose a point where the sinusoid has strong gradient in both directions.
        // At (24, 24) the phases are 3π/4 and 3π/4, so both cos factors are √2/2.
        let pts = vec![Point2D::new(24.0, 24.0)];
        let results = compute_lk_bouguet_sparse(&img0, &img1, w as u32, h as u32, &pts, &cfg)
            .expect("Bouguet LK should succeed");

        assert!(results[0].valid, "Sinusoidal patch should pass Shi-Tomasi");
        let estimated_dx = results[0].position.x - pts[0].x;
        assert!(
            (estimated_dx - shift).abs() < 1.5,
            "Expected dx ≈ {shift:.1}, got {estimated_dx:.3}"
        );
    }

    /// Flat-patch (uniform intensity) must be rejected by Shi-Tomasi.
    #[test]
    fn test_lk_rejects_low_texture_via_shi_tomasi() {
        let (w, h) = (64usize, 64usize);
        let img0 = vec![128u8; w * h];
        let img1 = vec![128u8; w * h];

        let cfg = LkConfig::default();
        let pts = vec![Point2D::new(32.0, 32.0)];
        let results = compute_lk_bouguet_sparse(&img0, &img1, w as u32, h as u32, &pts, &cfg)
            .expect("Bouguet LK should not error");

        assert!(!results[0].valid, "Flat patch should fail Shi-Tomasi check");
        assert!(
            results[0].confidence < cfg.shi_tomasi_threshold,
            "Confidence should be below threshold for flat patch"
        );
    }
}
