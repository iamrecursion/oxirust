//! Linear-triangle geometry shared by the fluid and the mesh-motion solvers.

/// Area, shape-function gradients and size of a linear triangle.
#[derive(Debug, Clone, Copy)]
pub(super) struct Triangle {
    /// Signed twice the area (positive for counter-clockwise node order).
    pub(super) twice_signed_area: f64,
    /// Area |A|.
    pub(super) area: f64,
    /// Constant gradients ∇λ_a of the three linear shape functions.
    pub(super) grad: [[f64; 2]; 3],
    /// Length of the longest edge.
    pub(super) longest_edge: f64,
}

impl Triangle {
    /// Geometry of the triangle with the given corner coordinates, or `None`
    /// if the triangle is degenerate (zero or non-finite area).
    pub(super) fn new(p: &[[f64; 2]; 3]) -> Option<Self> {
        let twice =
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        if !twice.is_finite() || twice == 0.0 {
            return None;
        }
        let grad = [
            [(p[1][1] - p[2][1]) / twice, (p[2][0] - p[1][0]) / twice],
            [(p[2][1] - p[0][1]) / twice, (p[0][0] - p[2][0]) / twice],
            [(p[0][1] - p[1][1]) / twice, (p[1][0] - p[0][0]) / twice],
        ];
        let edge = |a: usize, b: usize| (p[b][0] - p[a][0]).hypot(p[b][1] - p[a][1]);
        let longest_edge = edge(0, 1).max(edge(1, 2)).max(edge(2, 0));
        Some(Self {
            twice_signed_area: twice,
            area: 0.5 * twice.abs(),
            grad,
            longest_edge,
        })
    }
}

/// Corner coordinates of element `element` in `positions`, or `None` if a node
/// index is out of range.
pub(super) fn element_corners(
    positions: &[[f64; 2]],
    element: &[usize; 3],
) -> Option<[[f64; 2]; 3]> {
    Some([
        *positions.get(element[0])?,
        *positions.get(element[1])?,
        *positions.get(element[2])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_reproduce_linear_functions_and_sum_to_zero() {
        let p = [[0.2, 0.1], [1.3, 0.4], [0.5, 1.1]];
        let tri = Triangle::new(&p).expect("non-degenerate");
        assert!(tri.twice_signed_area > 0.0);
        assert!((tri.area - 0.5 * tri.twice_signed_area).abs() < 1e-15);
        // Σ_a ∇λ_a = 0 and Σ_a x_a ⊗ ∇λ_a = I.
        for d in 0..2 {
            let sum: f64 = tri.grad.iter().map(|g| g[d]).sum();
            assert!(sum.abs() < 1e-14, "gradient sum {sum}");
            for c in 0..2 {
                let moment: f64 = p
                    .iter()
                    .zip(tri.grad.iter())
                    .map(|(corner, grad)| corner[c] * grad[d])
                    .sum();
                let expected = if c == d { 1.0 } else { 0.0 };
                assert!(
                    (moment - expected).abs() < 1e-14,
                    "moment[{c}][{d}] = {moment}"
                );
            }
        }
    }

    #[test]
    fn clockwise_order_flips_the_sign_but_not_the_area() {
        let ccw = Triangle::new(&[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]).expect("ccw");
        let cw = Triangle::new(&[[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]]).expect("cw");
        assert!(ccw.twice_signed_area > 0.0 && cw.twice_signed_area < 0.0);
        assert!((ccw.area - 1.0).abs() < 1e-15 && (cw.area - 1.0).abs() < 1e-15);
        assert!((ccw.longest_edge - 5.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn degenerate_triangles_are_rejected() {
        assert!(Triangle::new(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_none());
        assert!(Triangle::new(&[[0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]]).is_none());
        assert!(element_corners(&[[0.0, 0.0]], &[0, 0, 1]).is_none());
    }
}
