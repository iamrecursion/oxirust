//! Color gamut operations for `OxiMedia`.
//!
//! Provides gamut boundary definitions for standard color spaces, gamut mapping,
//! and gamut coverage analysis using chromaticity coordinates.

#![allow(dead_code)]

/// Gamut boundary defined by three RGB primaries and a white point in xy chromaticity.
///
/// All coordinates are in the CIE 1931 xy chromaticity space.
#[derive(Debug, Clone)]
pub struct GamutBoundary {
    /// Primary chromaticities as `[[x, y]; 3]` for R, G, B.
    pub primaries: [[f32; 2]; 3],
    /// White point chromaticity `[x, y]`.
    pub white_point: [f32; 2],
}

impl GamutBoundary {
    /// ITU-R BT.709 / sRGB gamut (D65 white point).
    #[must_use]
    pub fn rec709() -> Self {
        Self {
            primaries: [
                [0.6400, 0.3300], // R
                [0.3000, 0.6000], // G
                [0.1500, 0.0600], // B
            ],
            white_point: [0.3127, 0.3290],
        }
    }

    /// ITU-R BT.2020 gamut (D65 white point).
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            primaries: [
                [0.7080, 0.2920], // R
                [0.1700, 0.7970], // G
                [0.1310, 0.0460], // B
            ],
            white_point: [0.3127, 0.3290],
        }
    }

    /// DCI-P3 gamut (DCI white point).
    #[must_use]
    pub fn dci_p3() -> Self {
        Self {
            primaries: [
                [0.6800, 0.3200], // R
                [0.2650, 0.6900], // G
                [0.1500, 0.0600], // B
            ],
            white_point: [0.3140, 0.3510],
        }
    }
}

/// Tests whether a chromaticity point (x, y) lies inside a gamut boundary triangle.
///
/// Uses barycentric coordinate method for point-in-triangle testing.
#[must_use]
pub fn is_in_gamut_xy(x: f32, y: f32, boundary: &GamutBoundary) -> bool {
    let [r, g, b] = boundary.primaries;
    point_in_triangle(x, y, r[0], r[1], g[0], g[1], b[0], b[1])
}

/// Returns `true` if point `(px, py)` is inside the triangle `(ax,ay)-(bx,by)-(cx,cy)`.
fn point_in_triangle(
    px: f32,
    py: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    cx: f32,
    cy: f32,
) -> bool {
    let sign = |p1x: f32, p1y: f32, p2x: f32, p2y: f32, p3x: f32, p3y: f32| -> f32 {
        (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    };
    let d1 = sign(px, py, ax, ay, bx, by);
    let d2 = sign(px, py, bx, by, cx, cy);
    let d3 = sign(px, py, cx, cy, ax, ay);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Maps chromaticity coordinates from one gamut to another.
#[derive(Debug, Clone)]
pub struct GamutMapper {
    /// Source gamut boundary.
    pub src: GamutBoundary,
    /// Destination gamut boundary.
    pub dst: GamutBoundary,
}

impl GamutMapper {
    /// Returns `true` if the point `(x, y)` is outside the destination gamut.
    #[must_use]
    pub fn is_out_of_gamut(&self, x: f32, y: f32) -> bool {
        !is_in_gamut_xy(x, y, &self.dst)
    }

    /// Clips an out-of-gamut chromaticity point to the nearest point on the
    /// destination gamut boundary.
    ///
    /// If the point is already in-gamut, it is returned unchanged. Otherwise the
    /// point is linearly interpolated from the white point toward the primaries
    /// until it lies on the boundary.
    #[must_use]
    pub fn clip_to_dst(&self, x: f32, y: f32) -> (f32, f32) {
        if !self.is_out_of_gamut(x, y) {
            return (x, y);
        }
        // Move the point toward the white point until it is inside the gamut.
        let [wp_x, wp_y] = self.dst.white_point;
        let steps = 64u32;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let cx = x + t * (wp_x - x);
            let cy = y + t * (wp_y - y);
            if is_in_gamut_xy(cx, cy, &self.dst) {
                return (cx, cy);
            }
        }
        (wp_x, wp_y)
    }
}

/// Computes the fraction of test pixels that fall within a given gamut boundary.
pub struct GamutCoverage;

impl GamutCoverage {
    /// Computes the coverage ratio: proportion of `test_pixels` inside `boundary`.
    ///
    /// # Returns
    ///
    /// A value in `[0.0, 1.0]`. Returns `0.0` if `test_pixels` is empty.
    #[must_use]
    pub fn compute(test_pixels: &[[f32; 2]], boundary: &GamutBoundary) -> f32 {
        if test_pixels.is_empty() {
            return 0.0;
        }
        let inside = test_pixels
            .iter()
            .filter(|p| is_in_gamut_xy(p[0], p[1], boundary))
            .count();
        inside as f32 / test_pixels.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec709_white_point() {
        let g = GamutBoundary::rec709();
        // D65 white point
        assert!((g.white_point[0] - 0.3127).abs() < 1e-4);
        assert!((g.white_point[1] - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn test_rec2020_wider_than_rec709() {
        // BT.2020 red x-chromaticity (0.708) > BT.709 red x-chromaticity (0.640)
        let g709 = GamutBoundary::rec709();
        let g2020 = GamutBoundary::rec2020();
        assert!(g2020.primaries[0][0] > g709.primaries[0][0]);
    }

    #[test]
    fn test_dci_p3_primaries() {
        let p3 = GamutBoundary::dci_p3();
        assert!((p3.primaries[0][0] - 0.6800).abs() < 1e-4); // R_x
        assert!((p3.primaries[1][1] - 0.6900).abs() < 1e-4); // G_y
    }

    #[test]
    fn test_is_in_gamut_white_point_inside() {
        // D65 white point should be inside Rec.709
        let g = GamutBoundary::rec709();
        assert!(is_in_gamut_xy(0.3127, 0.3290, &g));
    }

    #[test]
    fn test_is_in_gamut_primary_on_boundary() {
        let g = GamutBoundary::rec709();
        // Red primary itself should be on/inside the boundary
        assert!(is_in_gamut_xy(0.6400, 0.3300, &g));
    }

    #[test]
    fn test_is_not_in_gamut_extreme() {
        let g = GamutBoundary::rec709();
        // Far outside any reasonable gamut
        assert!(!is_in_gamut_xy(0.9, 0.9, &g));
    }

    #[test]
    fn test_gamut_mapper_in_gamut_unchanged() {
        let mapper = GamutMapper {
            src: GamutBoundary::rec2020(),
            dst: GamutBoundary::rec709(),
        };
        // White point is inside Rec.709
        let (cx, cy) = mapper.clip_to_dst(0.3127, 0.3290);
        assert!((cx - 0.3127).abs() < 1e-4);
        assert!((cy - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn test_gamut_mapper_out_of_gamut_clipped() {
        let mapper = GamutMapper {
            src: GamutBoundary::rec2020(),
            dst: GamutBoundary::rec709(),
        };
        // BT.2020 red primary (0.708, 0.292) is outside Rec.709
        assert!(mapper.is_out_of_gamut(0.708, 0.292));
        let (cx, cy) = mapper.clip_to_dst(0.708, 0.292);
        // Clipped point should now be inside dst
        assert!(is_in_gamut_xy(cx, cy, &mapper.dst));
    }

    #[test]
    fn test_gamut_mapper_is_out_of_gamut() {
        let mapper = GamutMapper {
            src: GamutBoundary::rec2020(),
            dst: GamutBoundary::rec709(),
        };
        assert!(mapper.is_out_of_gamut(0.708, 0.292));
        assert!(!mapper.is_out_of_gamut(0.3127, 0.3290));
    }

    #[test]
    fn test_gamut_coverage_empty() {
        let g = GamutBoundary::rec709();
        assert_eq!(GamutCoverage::compute(&[], &g), 0.0);
    }

    #[test]
    fn test_gamut_coverage_all_inside() {
        let g = GamutBoundary::rec709();
        // White point should always be inside
        let pixels = vec![[0.3127_f32, 0.3290_f32]; 10];
        let cov = GamutCoverage::compute(&pixels, &g);
        assert!((cov - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gamut_coverage_partial() {
        let g = GamutBoundary::rec709();
        let pixels = vec![
            [0.3127, 0.3290], // inside
            [0.9, 0.9],       // outside
        ];
        let cov = GamutCoverage::compute(&pixels, &g);
        assert!((cov - 0.5).abs() < 1e-5);
    }
}
