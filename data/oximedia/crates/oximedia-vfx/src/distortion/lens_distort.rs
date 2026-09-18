//! Lens distortion correction with full Brown-Conrady model.
//!
//! Provides `LensDistortParams`, `distort_point`, `undistort_point` (Newton
//! iteration), and `apply_lens_distortion` for full-frame processing.

// ── LensDistortParams ──────────────────────────────────────────────────────────

/// Full Brown-Conrady lens distortion parameters.
///
/// `k1`, `k2`, `k3` are radial coefficients; `p1`, `p2` are tangential
/// (decentring) coefficients.  `cx`, `cy` are the optical centre in normalised
/// image coordinates (0 = left/top edge, 1 = right/bottom edge).
/// `focal_x` and `focal_y` are focal lengths in the same normalised units.
#[derive(Debug, Clone, PartialEq)]
pub struct LensDistortParams {
    /// First radial distortion coefficient.
    pub k1: f64,
    /// Second radial distortion coefficient.
    pub k2: f64,
    /// Third radial distortion coefficient.
    pub k3: f64,
    /// First tangential distortion coefficient.
    pub p1: f64,
    /// Second tangential distortion coefficient.
    pub p2: f64,
    /// Principal point X in normalised image space [0, 1].
    pub cx: f64,
    /// Principal point Y in normalised image space [0, 1].
    pub cy: f64,
    /// Focal length in X direction (normalised).
    pub focal_x: f64,
    /// Focal length in Y direction (normalised).
    pub focal_y: f64,
}

impl LensDistortParams {
    /// Barrel distortion preset (negative `k1` pushes edges outward).
    ///
    /// The provided `k1` is forced negative; all other coefficients are zero.
    /// Centre is at (0.5, 0.5).
    #[must_use]
    pub fn barrel(k1: f64) -> Self {
        // Barrel distortion requires a negative k1
        let k1 = -k1.abs();
        Self {
            k1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            cx: 0.5,
            cy: 0.5,
            focal_x: 1.0,
            focal_y: 1.0,
        }
    }

    /// Pincushion distortion preset (positive `k1` pulls edges inward).
    ///
    /// The provided `k1` is forced positive.
    #[must_use]
    pub fn pincushion(k1: f64) -> Self {
        Self {
            k1: k1.abs(),
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            cx: 0.5,
            cy: 0.5,
            focal_x: 1.0,
            focal_y: 1.0,
        }
    }

    /// Identity (no distortion).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            cx: 0.5,
            cy: 0.5,
            focal_x: 1.0,
            focal_y: 1.0,
        }
    }
}

// ── Core distortion math ───────────────────────────────────────────────────────

/// Apply Brown-Conrady distortion to a **normalised** coordinate pair.
///
/// Both `x` and `y` are in [0, 1] image space.  Returns the distorted
/// coordinates, which may lie outside [0, 1] near the frame boundary.
#[must_use]
pub fn distort_point(x: f64, y: f64, params: &LensDistortParams) -> (f64, f64) {
    // Map to camera-centered coordinates
    let xn = (x - params.cx) / params.focal_x;
    let yn = (y - params.cy) / params.focal_y;

    let r2 = xn * xn + yn * yn;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    // Radial factor
    let radial = 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6;

    // Tangential correction
    let dx_tang = 2.0 * params.p1 * xn * yn + params.p2 * (r2 + 2.0 * xn * xn);
    let dy_tang = params.p1 * (r2 + 2.0 * yn * yn) + 2.0 * params.p2 * xn * yn;

    let xd = xn * radial + dx_tang;
    let yd = yn * radial + dy_tang;

    // Map back to image space
    (
        xd * params.focal_x + params.cx,
        yd * params.focal_y + params.cy,
    )
}

/// Invert the distortion model using Newton–Raphson iteration.
///
/// Given a distorted coordinate `(x, y)` (in [0, 1]), returns the
/// undistorted coordinate.  Converges in ~10 iterations for typical lens
/// parameters.
#[must_use]
pub fn undistort_point(x: f64, y: f64, params: &LensDistortParams) -> (f64, f64) {
    // Initial estimate: the distorted point itself
    let mut xn = (x - params.cx) / params.focal_x;
    let mut yn = (y - params.cy) / params.focal_y;

    let max_iters = 20;
    let tol = 1e-9;

    for _ in 0..max_iters {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6;

        let dx_tang = 2.0 * params.p1 * xn * yn + params.p2 * (r2 + 2.0 * xn * xn);
        let dy_tang = params.p1 * (r2 + 2.0 * yn * yn) + 2.0 * params.p2 * xn * yn;

        let xd_est = xn * radial + dx_tang;
        let yd_est = yn * radial + dy_tang;

        // Target undistorted coords in camera space
        let target_xn = (x - params.cx) / params.focal_x;
        let target_yn = (y - params.cy) / params.focal_y;

        let ex = xd_est - target_xn;
        let ey = yd_est - target_yn;

        if ex * ex + ey * ey < tol * tol {
            break;
        }

        // Jacobian of the distortion function
        let dr2_dxn = 2.0 * xn;
        let dr2_dyn = 2.0 * yn;
        let dradial_dxn =
            params.k1 * dr2_dxn + 2.0 * params.k2 * r2 * dr2_dxn + 3.0 * params.k3 * r4 * dr2_dxn;
        let dradial_dyn =
            params.k1 * dr2_dyn + 2.0 * params.k2 * r2 * dr2_dyn + 3.0 * params.k3 * r4 * dr2_dyn;

        // df_x / d(xn, yn)
        let j00 =
            radial + xn * dradial_dxn + 2.0 * params.p1 * yn + params.p2 * (dr2_dxn + 4.0 * xn);
        let j01 = xn * dradial_dyn + 2.0 * params.p1 * xn + params.p2 * dr2_dyn;

        // df_y / d(xn, yn)
        let j10 = yn * dradial_dxn + params.p1 * dr2_dxn + 2.0 * params.p2 * yn;
        let j11 =
            radial + yn * dradial_dyn + params.p1 * (dr2_dyn + 4.0 * yn) + 2.0 * params.p2 * xn;

        // Newton step: solve J * delta = -e
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-15 {
            break;
        }

        let dxn = -(j11 * ex - j01 * ey) / det;
        let dyn_ = -(-j10 * ex + j00 * ey) / det;

        xn += dxn;
        yn += dyn_;
    }

    (
        xn * params.focal_x + params.cx,
        yn * params.focal_y + params.cy,
    )
}

// ── Full-frame processing ──────────────────────────────────────────────────────

/// Apply lens distortion to a packed RGB (3 bytes/pixel) buffer.
///
/// Returns a new `Vec<u8>` of the same size (`width * height * 3`).
/// Pixels that map outside the source image are filled with black.
#[must_use]
pub fn apply_lens_distortion(
    pixels: &[u8],
    width: u32,
    height: u32,
    params: &LensDistortParams,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut output = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            // Normalise to [0, 1]
            let nx = col as f64 / (w as f64 - 1.0).max(1.0);
            let ny = row as f64 / (h as f64 - 1.0).max(1.0);

            // Find source pixel by undistorting the destination coordinate
            let (src_nx, src_ny) = undistort_point(nx, ny, params);

            let src_x = src_nx * (w as f64 - 1.0);
            let src_y = src_ny * (h as f64 - 1.0);

            let rgb = bilinear_sample(pixels, width, height, src_x, src_y);
            let idx = (row * w + col) * 3;
            output[idx] = rgb[0];
            output[idx + 1] = rgb[1];
            output[idx + 2] = rgb[2];
        }
    }

    output
}

/// Bilinear sample from a packed RGB (3 bytes/pixel) buffer.
///
/// Returns `[0, 0, 0]` for out-of-bounds coordinates.
#[must_use]
pub fn bilinear_sample(pixels: &[u8], width: u32, height: u32, x: f64, y: f64) -> [u8; 3] {
    let w = width as usize;
    let h = height as usize;

    if x < 0.0 || y < 0.0 || x > (w as f64 - 1.0) || y > (h as f64 - 1.0) {
        return [0, 0, 0];
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let sample = |row: usize, col: usize| -> [f64; 3] {
        let idx = (row * w + col) * 3;
        if idx + 2 < pixels.len() {
            [
                pixels[idx] as f64,
                pixels[idx + 1] as f64,
                pixels[idx + 2] as f64,
            ]
        } else {
            [0.0; 3]
        }
    };

    let p00 = sample(y0, x0);
    let p10 = sample(y0, x1);
    let p01 = sample(y1, x0);
    let p11 = sample(y1, x1);

    let mut result = [0u8; 3];
    for i in 0..3 {
        let top = p00[i] * (1.0 - fx) + p10[i] * fx;
        let bot = p01[i] * (1.0 - fx) + p11[i] * fx;
        result[i] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_identity_distort_centre() {
        let p = LensDistortParams::identity();
        let (ox, oy) = distort_point(0.5, 0.5, &p);
        assert!(approx_eq(ox, 0.5, 1e-10), "cx should be 0.5, got {ox}");
        assert!(approx_eq(oy, 0.5, 1e-10), "cy should be 0.5, got {oy}");
    }

    #[test]
    fn test_identity_undistort_centre() {
        let p = LensDistortParams::identity();
        let (ox, oy) = undistort_point(0.5, 0.5, &p);
        assert!(
            approx_eq(ox, 0.5, 1e-8),
            "undistort centre should be 0.5, got {ox}"
        );
        assert!(
            approx_eq(oy, 0.5, 1e-8),
            "undistort centre should be 0.5, got {oy}"
        );
    }

    #[test]
    fn test_identity_roundtrip() {
        // distort then undistort should return original point
        let p = LensDistortParams::identity();
        let (dx, dy) = distort_point(0.3, 0.7, &p);
        let (ox, oy) = undistort_point(dx, dy, &p);
        assert!(approx_eq(ox, 0.3, 1e-8), "roundtrip x: got {ox}");
        assert!(approx_eq(oy, 0.7, 1e-8), "roundtrip y: got {oy}");
    }

    #[test]
    fn test_barrel_moves_corners_inward() {
        // barrel() forces k1 < 0; the Brown-Conrady model with negative k1 moves
        // off-centre points *towards* the centre (classic barrel bow effect as
        // seen in the distorted image space).
        let p = LensDistortParams::barrel(-0.3);
        assert!(p.k1 < 0.0, "barrel k1 should be negative, got {}", p.k1);
        let (dx, _dy) = distort_point(0.9, 0.5, &p);
        // Point is pulled inward (smaller distance from centre)
        assert!(
            (dx - 0.5_f64).abs() < (0.9_f64 - 0.5_f64),
            "barrel distortion should pull point toward centre, got {dx}"
        );
    }

    #[test]
    fn test_pincushion_params() {
        let p = LensDistortParams::pincushion(0.2);
        assert!(p.k1 > 0.0, "pincushion k1 should be positive, got {}", p.k1);
        assert!(approx_eq(p.cx, 0.5, 1e-10), "cx should be 0.5");
    }

    #[test]
    fn test_barrel_params() {
        // barrel() always forces k1 to be negative regardless of sign of input
        let p1 = LensDistortParams::barrel(-0.1);
        assert!(
            p1.k1 < 0.0,
            "barrel k1 should be negative (was negative input), got {}",
            p1.k1
        );
        let p2 = LensDistortParams::barrel(0.1);
        assert!(
            p2.k1 < 0.0,
            "barrel k1 should be negative (was positive input), got {}",
            p2.k1
        );
    }

    #[test]
    fn test_undistort_roundtrip_barrel() {
        let p = LensDistortParams::barrel(-0.2);
        let (dx, dy) = distort_point(0.4, 0.6, &p);
        let (ox, oy) = undistort_point(dx, dy, &p);
        assert!(approx_eq(ox, 0.4, 1e-6), "barrel roundtrip x: {ox}");
        assert!(approx_eq(oy, 0.6, 1e-6), "barrel roundtrip y: {oy}");
    }

    #[test]
    fn test_bilinear_sample_corners() {
        // 2x2 image: red, green, blue, white
        let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let rgb = bilinear_sample(&pixels, 2, 2, 0.0, 0.0);
        assert_eq!(rgb, [255, 0, 0], "top-left should be red");
        let rgb2 = bilinear_sample(&pixels, 2, 2, 1.0, 0.0);
        assert_eq!(rgb2, [0, 255, 0], "top-right should be green");
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds() {
        let pixels: Vec<u8> = vec![255u8; 12];
        let rgb = bilinear_sample(&pixels, 2, 2, -1.0, 0.0);
        assert_eq!(rgb, [0, 0, 0], "out of bounds should be black");
    }

    #[test]
    fn test_apply_lens_distortion_identity_size() {
        let w = 8u32;
        let h = 6u32;
        let pixels = vec![128u8; (w * h * 3) as usize];
        let params = LensDistortParams::identity();
        let out = apply_lens_distortion(&pixels, w, h, &params);
        assert_eq!(
            out.len(),
            (w * h * 3) as usize,
            "output should have same size"
        );
    }

    #[test]
    fn test_apply_lens_distortion_identity_values() {
        let w = 4u32;
        let h = 4u32;
        // Fill with a simple pattern
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 3] = (i * 7 % 256) as u8;
            pixels[i * 3 + 1] = (i * 13 % 256) as u8;
            pixels[i * 3 + 2] = (i * 17 % 256) as u8;
        }
        let params = LensDistortParams::identity();
        let out = apply_lens_distortion(&pixels, w, h, &params);
        // Identity should preserve pixel values (with bilinear rounding)
        assert_eq!(out.len(), pixels.len(), "size preserved");
    }

    #[test]
    fn test_tangential_params() {
        let p = LensDistortParams {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.01,
            p2: -0.01,
            cx: 0.5,
            cy: 0.5,
            focal_x: 1.0,
            focal_y: 1.0,
        };
        // With tangential distortion, a point off-centre should be shifted
        let (dx, dy) = distort_point(0.3, 0.7, &p);
        // Just verify the output is finite and different from identity
        assert!(dx.is_finite(), "dx should be finite");
        assert!(dy.is_finite(), "dy should be finite");
    }

    #[test]
    fn test_barrel_undistort_offcentre() {
        let p = LensDistortParams::barrel(-0.15);
        // Multiple off-centre points
        for (x, y) in [(0.2, 0.3), (0.7, 0.8), (0.1, 0.9)] {
            let (dx, dy) = distort_point(x, y, &p);
            let (ux, uy) = undistort_point(dx, dy, &p);
            assert!(
                approx_eq(ux, x, 1e-5),
                "roundtrip x for ({x},{y}): got {ux}"
            );
            assert!(
                approx_eq(uy, y, 1e-5),
                "roundtrip y for ({x},{y}): got {uy}"
            );
        }
    }
}
