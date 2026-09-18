//! Planar tracking for perspective-corrected tracking.

use super::point::{PointTracker, TrackPoint, TrackingConfig};
use crate::{Frame, VfxError, VfxResult};
use serde::{Deserialize, Serialize};

/// A corner of a planar surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Corner {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl From<TrackPoint> for Corner {
    fn from(point: TrackPoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }
}

/// Planar tracking data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanarData {
    /// Top-left corner.
    pub top_left: Corner,
    /// Top-right corner.
    pub top_right: Corner,
    /// Bottom-right corner.
    pub bottom_right: Corner,
    /// Bottom-left corner.
    pub bottom_left: Corner,
    /// Average confidence.
    pub confidence: f32,
    /// Whether all corners tracked successfully.
    pub success: bool,
}

impl PlanarData {
    /// Calculate 3x3 homography matrix mapping points from `reference`'s
    /// corner quad into `self`'s corner quad.
    ///
    /// Used by [`PlanarTracker::warp_to_reference`] to map each output
    /// (reference-space) pixel to the position it currently occupies in the
    /// tracked (`self`) frame, so the homography returned here must satisfy
    /// `H * reference_corner_i ≈ self_corner_i` for each of the four
    /// corners (up to homogeneous scale).
    #[must_use]
    pub fn calculate_homography(&self, reference: &Self) -> [[f32; 3]; 3] {
        let src = [
            (reference.top_left.x, reference.top_left.y),
            (reference.top_right.x, reference.top_right.y),
            (reference.bottom_right.x, reference.bottom_right.y),
            (reference.bottom_left.x, reference.bottom_left.y),
        ];

        let dst = [
            (self.top_left.x, self.top_left.y),
            (self.top_right.x, self.top_right.y),
            (self.bottom_right.x, self.bottom_right.y),
            (self.bottom_left.x, self.bottom_left.y),
        ];

        solve_homography_dlt(src, dst)
    }

    /// Get center point of planar surface.
    #[must_use]
    pub fn center(&self) -> Corner {
        Corner {
            x: (self.top_left.x + self.top_right.x + self.bottom_right.x + self.bottom_left.x)
                / 4.0,
            y: (self.top_left.y + self.top_right.y + self.bottom_right.y + self.bottom_left.y)
                / 4.0,
        }
    }
}

/// Solves the 8-parameter planar homography (`h[2][2]` fixed to 1) that
/// maps each `src[i]` point to the corresponding `dst[i]` point, using the
/// standard Direct Linear Transform (DLT) for four exact point
/// correspondences (e.g. Hartley & Zisserman, "Multiple View Geometry",
/// §4.1, specialised to the four-point case used throughout planar/corner
/// tracking and perspective warping, cf. `getPerspectiveTransform`).
///
/// For a homography `H = [[h0,h1,h2],[h3,h4,h5],[h6,h7,1]]` and a
/// correspondence `(x,y) -> (x',y')`:
///
/// ```text
/// x' = (h0*x + h1*y + h2) / (h6*x + h7*y + 1)
/// y' = (h3*x + h4*y + h5) / (h6*x + h7*y + 1)
/// ```
///
/// Clearing denominators turns each correspondence into two linear
/// equations in the 8 unknowns `h0..h7`:
///
/// ```text
/// h0*x + h1*y + h2                 - h6*x*x' - h7*y*x' = x'
///                 h3*x + h4*y + h5 - h6*x*y' - h7*y*y' = y'
/// ```
///
/// Four correspondences give exactly 8 equations for 8 unknowns, solved
/// here via Gaussian elimination with partial pivoting (`f64` internally
/// for numerical stability, since the trailing `h6`/`h7` terms are
/// typically orders of magnitude smaller than `h0..h5`).
///
/// Returns the identity matrix if the system is singular (degenerate,
/// e.g. collinear or coincident correspondences) rather than dividing by
/// (near-)zero.
fn solve_homography_dlt(src: [(f32, f32); 4], dst: [(f32, f32); 4]) -> [[f32; 3]; 3] {
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Augmented 8x9 matrix: 8 equations (2 per correspondence) in 8
    // unknowns [h0..h7], last column is the right-hand side.
    let mut m = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (f64::from(src[i].0), f64::from(src[i].1));
        let (xp, yp) = (f64::from(dst[i].0), f64::from(dst[i].1));

        m[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp, xp];
        m[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp, yp];
    }

    // Gaussian elimination with partial pivoting (Gauss-Jordan: reduce to
    // the identity on the left, solution appears in the last column).
    for col in 0..8 {
        let mut pivot_row = col;
        let mut pivot_val = m[col][col].abs();
        for row in (col + 1)..8 {
            if m[row][col].abs() > pivot_val {
                pivot_val = m[row][col].abs();
                pivot_row = row;
            }
        }

        if pivot_val < 1e-9 {
            // Degenerate correspondences (e.g. collinear corners) — no
            // unique homography exists; fail safe to identity rather than
            // dividing by (near-)zero.
            return IDENTITY;
        }

        m.swap(col, pivot_row);

        let scale = m[col][col];
        for v in &mut m[col] {
            *v /= scale;
        }

        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            if factor != 0.0 {
                for k in 0..9 {
                    m[row][k] -= factor * m[col][k];
                }
            }
        }
    }

    [
        [m[0][8] as f32, m[1][8] as f32, m[2][8] as f32],
        [m[3][8] as f32, m[4][8] as f32, m[5][8] as f32],
        [m[6][8] as f32, m[7][8] as f32, 1.0],
    ]
}

/// Four-point planar tracker for perspective tracking.
#[derive(Debug, Clone)]
pub struct PlanarTracker {
    tracker_tl: PointTracker,
    tracker_tr: PointTracker,
    tracker_br: PointTracker,
    tracker_bl: PointTracker,
    reference_data: Option<PlanarData>,
}

impl PlanarTracker {
    /// Create a new planar tracker.
    #[must_use]
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            tracker_tl: PointTracker::new(config),
            tracker_tr: PointTracker::new(config),
            tracker_br: PointTracker::new(config),
            tracker_bl: PointTracker::new(config),
            reference_data: None,
        }
    }

    /// Initialize with four corner points.
    pub fn initialize(&mut self, frame: &Frame, corners: [Corner; 4]) -> VfxResult<()> {
        let [tl, tr, br, bl] = corners;

        self.tracker_tl.initialize(
            frame,
            TrackPoint {
                x: tl.x,
                y: tl.y,
                confidence: 1.0,
            },
        )?;
        self.tracker_tr.initialize(
            frame,
            TrackPoint {
                x: tr.x,
                y: tr.y,
                confidence: 1.0,
            },
        )?;
        self.tracker_br.initialize(
            frame,
            TrackPoint {
                x: br.x,
                y: br.y,
                confidence: 1.0,
            },
        )?;
        self.tracker_bl.initialize(
            frame,
            TrackPoint {
                x: bl.x,
                y: bl.y,
                confidence: 1.0,
            },
        )?;

        self.reference_data = Some(PlanarData {
            top_left: tl,
            top_right: tr,
            bottom_right: br,
            bottom_left: bl,
            confidence: 1.0,
            success: true,
        });

        Ok(())
    }

    /// Track planar surface in new frame.
    pub fn track(&self, frame: &Frame) -> VfxResult<PlanarData> {
        let result_tl = self.tracker_tl.track(frame)?;
        let result_tr = self.tracker_tr.track(frame)?;
        let result_br = self.tracker_br.track(frame)?;
        let result_bl = self.tracker_bl.track(frame)?;

        let confidence = (result_tl.point.confidence
            + result_tr.point.confidence
            + result_br.point.confidence
            + result_bl.point.confidence)
            / 4.0;

        let success =
            result_tl.success && result_tr.success && result_br.success && result_bl.success;

        Ok(PlanarData {
            top_left: result_tl.point.into(),
            top_right: result_tr.point.into(),
            bottom_right: result_br.point.into(),
            bottom_left: result_bl.point.into(),
            confidence,
            success,
        })
    }

    /// Warp frame using planar tracking data.
    pub fn warp_to_reference(
        &self,
        input: &Frame,
        tracking: &PlanarData,
        output: &mut Frame,
    ) -> VfxResult<()> {
        let reference = self
            .reference_data
            .as_ref()
            .ok_or_else(|| VfxError::ProcessingError("Tracker not initialized".to_string()))?;

        let homography = tracking.calculate_homography(reference);

        // Apply homography warp
        for y in 0..output.height {
            for x in 0..output.width {
                let (src_x, src_y) = self.apply_homography(&homography, x as f32, y as f32);

                let pixel = if src_x >= 0.0
                    && src_x < input.width as f32
                    && src_y >= 0.0
                    && src_y < input.height as f32
                {
                    self.bilinear_sample(input, src_x, src_y)
                } else {
                    [0, 0, 0, 0]
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn apply_homography(&self, h: &[[f32; 3]; 3], x: f32, y: f32) -> (f32, f32) {
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        if w.abs() < 1e-6 {
            return (x, y);
        }
        let x_out = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
        let y_out = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
        (x_out, y_out)
    }

    fn bilinear_sample(&self, frame: &Frame, x: f32, y: f32) -> [u8; 4] {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(frame.width - 1);
        let y1 = (y0 + 1).min(frame.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

        let mut result = [0u8; 4];
        for i in 0..4 {
            let v0 = f32::from(p00[i]) * (1.0 - fx) + f32::from(p10[i]) * fx;
            let v1 = f32::from(p01[i]) * (1.0 - fx) + f32::from(p11[i]) * fx;
            result[i] = (v0 * (1.0 - fy) + v1 * fy) as u8;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planar_data_center() {
        let data = PlanarData {
            top_left: Corner { x: 0.0, y: 0.0 },
            top_right: Corner { x: 100.0, y: 0.0 },
            bottom_right: Corner { x: 100.0, y: 100.0 },
            bottom_left: Corner { x: 0.0, y: 100.0 },
            confidence: 1.0,
            success: true,
        };

        let center = data.center();
        assert_eq!(center.x, 50.0);
        assert_eq!(center.y, 50.0);
    }

    #[test]
    fn test_planar_tracker_initialization() -> VfxResult<()> {
        let frame = Frame::new(200, 200)?;
        let mut tracker = PlanarTracker::new(TrackingConfig::default());

        let corners = [
            Corner { x: 50.0, y: 50.0 },
            Corner { x: 150.0, y: 50.0 },
            Corner { x: 150.0, y: 150.0 },
            Corner { x: 50.0, y: 150.0 },
        ];

        tracker.initialize(&frame, corners)?;
        Ok(())
    }

    fn quad(tl: (f32, f32), tr: (f32, f32), br: (f32, f32), bl: (f32, f32)) -> PlanarData {
        PlanarData {
            top_left: Corner { x: tl.0, y: tl.1 },
            top_right: Corner { x: tr.0, y: tr.1 },
            bottom_right: Corner { x: br.0, y: br.1 },
            bottom_left: Corner { x: bl.0, y: bl.1 },
            confidence: 1.0,
            success: true,
        }
    }

    #[test]
    fn test_calculate_homography_identity_for_matching_quads() {
        let square = quad((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0));
        let h = square.calculate_homography(&square);

        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (h[row][col] - expected).abs() < 1e-3,
                    "identity check h[{row}][{col}] = {}, expected {expected}",
                    h[row][col]
                );
            }
        }
    }

    #[test]
    fn test_calculate_homography_pure_translation_matches_expected_matrix() {
        // reference (self=tracking is a pure +10/+20 translation of it).
        let reference = quad((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0));
        let tracking = quad((10.0, 20.0), (110.0, 20.0), (110.0, 120.0), (10.0, 120.0));

        let h = tracking.calculate_homography(&reference);

        let expected = [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]];
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (h[row][col] - expected[row][col]).abs() < 1e-3,
                    "h[{row}][{col}] = {}, expected {}",
                    h[row][col],
                    expected[row][col]
                );
            }
        }
    }

    #[test]
    fn test_calculate_homography_nonidentity_moves_test_point() {
        let reference = quad((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0));
        // Tracking quad is the reference quad scaled 1x and shifted by (+50,+50).
        let tracking = quad((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0));

        let h = tracking.calculate_homography(&reference);

        // Apply H to the reference quad's center (50, 50); it must land on
        // the tracking quad's center (100, 100), not stay put.
        let (x, y) = (50.0_f32, 50.0_f32);
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        let px = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
        let py = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;

        assert!(
            (px - 100.0).abs() < 1e-2 && (py - 100.0).abs() < 1e-2,
            "expected mapped center ≈ (100,100), got ({px},{py})"
        );
        assert!(
            (px - x).abs() > 1.0 || (py - y).abs() > 1.0,
            "homography must not be identity: point must actually move"
        );
    }

    #[test]
    fn test_calculate_homography_recovers_known_perspective_transform() {
        // A genuine (non-affine) perspective transform: h6/h7 nonzero.
        let h_true: [[f32; 3]; 3] = [[1.2, 0.1, 5.0], [0.05, 1.1, 8.0], [0.0008, 0.0005, 1.0]];
        let project = |x: f32, y: f32| -> (f32, f32) {
            let w = h_true[2][0] * x + h_true[2][1] * y + h_true[2][2];
            (
                (h_true[0][0] * x + h_true[0][1] * y + h_true[0][2]) / w,
                (h_true[1][0] * x + h_true[1][1] * y + h_true[1][2]) / w,
            )
        };

        let ref_corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let tracked: Vec<(f32, f32)> = ref_corners.iter().map(|&(x, y)| project(x, y)).collect();

        let reference = quad(
            ref_corners[0],
            ref_corners[1],
            ref_corners[2],
            ref_corners[3],
        );
        let tracking = quad(tracked[0], tracked[1], tracked[2], tracked[3]);

        let h = tracking.calculate_homography(&reference);

        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (h[row][col] - h_true[row][col]).abs() < 1e-2,
                    "h[{row}][{col}] = {}, expected {} (recovered homography must match the \
                     ground truth used to generate the correspondences)",
                    h[row][col],
                    h_true[row][col]
                );
            }
        }
    }

    #[test]
    fn test_calculate_homography_degenerate_source_falls_back_to_identity() {
        // Degenerate correspondence: all four "reference" corners are the
        // same point, so at most 2 independent constraints are available
        // instead of 8 — the 8x8 DLT system is provably rank-deficient
        // (each pair of equations collapses to a duplicate). No unique
        // homography exists; must fail safe to identity rather than
        // panicking or dividing by (near-)zero.
        let reference = quad((5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0));
        let tracking = quad((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0));

        let h = tracking.calculate_homography(&reference);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (h[row][col] - expected).abs() < 1e-3,
                    "degenerate correspondences must fail safe to identity, got h[{row}][{col}] \
                     = {}",
                    h[row][col]
                );
            }
        }
    }
}
