//! Two-dimensional affine transformation matrices.
//!
//! [`Matrix`] is the algebraic core that every SVG coordinate-system feature
//! is built from: the `transform` attribute microsyntax
//! ([`crate::values::transform`]), the `viewBox`-to-viewport algorithm, and
//! gradient/pattern/marker/clipPath coordinate resolution all reduce to
//! composing and applying [`Matrix`] values.
//!
//! # Convention
//!
//! A matrix is stored as the six components `a, b, c, d, e, f` of the SVG
//! `matrix(a b c d e f)` transform function, and maps a point using SVG's own
//! **column-vector** convention:
//!
//! ```text
//! (x, y) -> (a*x + c*y + e,  b*x + d*y + f)
//! ```
//!
//! Written as the 3x3 homogeneous matrix SVG's own specification uses:
//!
//! ```text
//! | a  c  e |   | x |   | a*x + c*y + e |
//! | b  d  f | * | y | = | b*x + d*y + f |
//! | 0  0  1 |   | 1 |   |       1       |
//! ```
//!
//! # Composition order
//!
//! [`Matrix::multiply`] implements ordinary matrix multiplication: for
//! `self.multiply(other)`, applying the result to a point gives the same
//! answer as applying `other` first and `self` second —
//! `self.multiply(other).apply(x, y) == self.apply(other.apply(x, y).0, other.apply(x, y).1)`
//! (see the worked example on [`Matrix::multiply`]). This is exactly the
//! convention [`crate::values::transform::TransformList::to_matrix`] relies
//! on to compose a whole `transform` attribute, and the reason
//! [`Matrix::pre_concat`]/[`Matrix::post_concat`] exist: `multiply` alone
//! reads the same whether the caller means "concatenate `other` before me"
//! or "concatenate `other` after me", so this module spells both out under
//! distinct names rather than leaving callers to work out `self.multiply(x)`
//! vs `x.multiply(self)` from first principles every time.

use std::fmt;

/// A 2D affine transformation matrix, stored as the six components of the
/// SVG `matrix(a b c d e f)` transform function.
///
/// See the [module documentation](self) for the exact mapping this applies
/// to points and the composition convention [`Matrix::multiply`] follows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    /// The `a` component (x-scale / x-axis x-component).
    pub a: f64,
    /// The `b` component (x-axis y-component, i.e. the y-shear from x).
    pub b: f64,
    /// The `c` component (y-axis x-component, i.e. the x-shear from y).
    pub c: f64,
    /// The `d` component (y-scale / y-axis y-component).
    pub d: f64,
    /// The `e` component (x translation).
    pub e: f64,
    /// The `f` component (y translation).
    pub f: f64,
}

impl Matrix {
    /// Builds a matrix directly from its six components, in the same order
    /// as the SVG `matrix(a, b, c, d, e, f)` function.
    #[must_use]
    pub const fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// The identity matrix: leaves every point unchanged.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    }

    /// The matrix equivalent to SVG's `translate(tx, ty)` transform function.
    #[must_use]
    pub const fn translate(tx: f64, ty: f64) -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0, tx, ty)
    }

    /// The matrix equivalent to SVG's `scale(sx, sy)` transform function.
    #[must_use]
    pub const fn scale(sx: f64, sy: f64) -> Self {
        Self::new(sx, 0.0, 0.0, sy, 0.0, 0.0)
    }

    /// The matrix equivalent to SVG's `rotate(angle_deg)` transform function:
    /// a rotation about the origin.
    ///
    /// `angle_deg` is in degrees, matching the SVG transform-list grammar
    /// (bare numbers are degrees; there is no unit suffix as there is in the
    /// CSS `transform` property grammar).
    #[must_use]
    pub fn rotate(angle_deg: f64) -> Self {
        let (sin, cos) = angle_deg.to_radians().sin_cos();
        Self::new(cos, sin, -sin, cos, 0.0, 0.0)
    }

    /// The matrix equivalent to SVG's three-argument `rotate(angle_deg, cx,
    /// cy)` transform function: a rotation about the point `(cx, cy)`.
    ///
    /// Defined, per the SVG specification, as
    /// `translate(cx, cy) · rotate(angle_deg) · translate(-cx, -cy)`: move
    /// the center to the origin, rotate, then move back.
    #[must_use]
    pub fn rotate_about(angle_deg: f64, cx: f64, cy: f64) -> Self {
        Self::translate(cx, cy)
            .multiply(&Self::rotate(angle_deg))
            .multiply(&Self::translate(-cx, -cy))
    }

    /// The matrix equivalent to SVG's `skewX(angle_deg)` transform function:
    /// shears the x coordinate by `y * tan(angle_deg)`.
    #[must_use]
    pub fn skew_x(angle_deg: f64) -> Self {
        Self::new(1.0, 0.0, angle_deg.to_radians().tan(), 1.0, 0.0, 0.0)
    }

    /// The matrix equivalent to SVG's `skewY(angle_deg)` transform function:
    /// shears the y coordinate by `x * tan(angle_deg)`.
    #[must_use]
    pub fn skew_y(angle_deg: f64) -> Self {
        Self::new(1.0, angle_deg.to_radians().tan(), 0.0, 1.0, 0.0, 0.0)
    }

    /// Applies this matrix to the point `(x, y)`, returning the mapped
    /// point `(a*x + c*y + e, b*x + d*y + f)`.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    /// Composes two matrices as ordinary matrix multiplication, `self *
    /// other`.
    ///
    /// The result applies `other` to a point first and `self` to the
    /// outcome second:
    ///
    /// ```
    /// use oxixml_svg::geometry::matrix::Matrix;
    ///
    /// let a = Matrix::translate(10.0, 0.0);
    /// let b = Matrix::rotate(90.0);
    /// let composed = a.multiply(&b);
    ///
    /// // composed.apply(p) == a.apply(b.apply(p)): b (rotate) acts first,
    /// // a (translate) acts second, on the rotated point.
    /// let (x, y) = composed.apply(1.0, 0.0);
    /// assert!((x - 10.0).abs() < 1e-9);
    /// assert!((y - 1.0).abs() < 1e-9);
    /// ```
    ///
    /// This is exactly the convention an SVG `transform="A B C"` attribute
    /// composes under: the resulting matrix is `M(A) · M(B) · M(C)`, i.e.
    /// `M(A).multiply(&M(B)).multiply(&M(C))` folded left-to-right over the
    /// transform list, even though `C` (the last-written function) is the
    /// one that acts on a point first. See
    /// [`crate::values::transform::TransformList::to_matrix`].
    #[must_use]
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        Matrix::new(
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f,
        )
    }

    /// Concatenates `other` onto `self` so that `other` takes effect first
    /// (closer to the point being transformed).
    ///
    /// Equivalent to `self.multiply(other)`, i.e. `other` is a
    /// "pre-transform" relative to `self`. Use this when building up a
    /// composed matrix by appending transforms that should run before
    /// whatever is already accumulated — exactly the fold
    /// [`crate::values::transform::TransformList::to_matrix`] performs.
    #[must_use]
    pub fn pre_concat(&self, other: &Matrix) -> Matrix {
        self.multiply(other)
    }

    /// Concatenates `other` onto `self` so that `other` takes effect last
    /// (after `self`, farthest from the point being transformed).
    ///
    /// Equivalent to `other.multiply(self)`, i.e. `other` is a
    /// "post-transform" relative to `self`.
    ///
    /// `pre_concat` and `post_concat` generally disagree — matrix
    /// multiplication does not commute — which is exactly why both are
    /// named explicitly instead of leaving callers to guess an order from a
    /// single generic `multiply`/`then`:
    ///
    /// ```
    /// use oxixml_svg::geometry::matrix::Matrix;
    ///
    /// let a = Matrix::translate(10.0, 0.0);
    /// let b = Matrix::scale(2.0, 2.0);
    ///
    /// // pre_concat: b (scale) acts first, a (translate) acts second.
    /// assert_eq!(a.pre_concat(&b).apply(1.0, 1.0), (12.0, 2.0));
    /// // post_concat: a (translate) acts first, b (scale) acts second.
    /// assert_eq!(a.post_concat(&b).apply(1.0, 1.0), (22.0, 2.0));
    /// ```
    #[must_use]
    pub fn post_concat(&self, other: &Matrix) -> Matrix {
        other.multiply(self)
    }

    /// The determinant `a*d - b*c` of the matrix's linear (non-translation)
    /// part.
    ///
    /// A determinant of `0` means the matrix collapses the plane onto a
    /// line or a point and therefore has no inverse; see [`Matrix::invert`].
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    /// The inverse matrix, or `None` when this matrix is not invertible
    /// (its [`determinant`](Matrix::determinant) is zero or non-finite, or
    /// the computed inverse itself would contain a non-finite component).
    #[must_use]
    pub fn invert(&self) -> Option<Matrix> {
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let inv_det = 1.0 / det;
        let inverted = Matrix::new(
            self.d * inv_det,
            -self.b * inv_det,
            -self.c * inv_det,
            self.a * inv_det,
            (self.c * self.f - self.d * self.e) * inv_det,
            (self.b * self.e - self.a * self.f) * inv_det,
        );
        let finite = inverted.a.is_finite()
            && inverted.b.is_finite()
            && inverted.c.is_finite()
            && inverted.d.is_finite()
            && inverted.e.is_finite()
            && inverted.f.is_finite();
        if finite { Some(inverted) } else { None }
    }

    /// True when this matrix is exactly the identity matrix (bit-for-bit
    /// equal to [`Matrix::identity`]).
    ///
    /// This is an exact comparison, not a tolerance-based one: a matrix
    /// built from a computation that is only *numerically* close to the
    /// identity (for example `rotate(360.0)`, which accumulates
    /// floating-point error) will not satisfy it. Use
    /// [`Matrix::approx_eq`] against [`Matrix::identity()`] for a
    /// tolerance-based check.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::identity()
    }

    /// True when every component of `self` and `other` differs by at most
    /// `epsilon`. Intended for numerically-computed matrices in tests,
    /// where exact `==` is too strict.
    #[must_use]
    pub fn approx_eq(&self, other: &Matrix, epsilon: f64) -> bool {
        (self.a - other.a).abs() <= epsilon
            && (self.b - other.b).abs() <= epsilon
            && (self.c - other.c).abs() <= epsilon
            && (self.d - other.d).abs() <= epsilon
            && (self.e - other.e).abs() <= epsilon
            && (self.f - other.f).abs() <= epsilon
    }
}

impl Default for Matrix {
    /// The identity matrix; see [`Matrix::identity`].
    fn default() -> Self {
        Self::identity()
    }
}

impl fmt::Display for Matrix {
    /// Formats the matrix using the SVG transform-list `matrix(...)`
    /// function syntax, e.g. `matrix(1, 0, 0, 1, 5, 7)`.
    ///
    /// `f64`'s own `Display` already formats whole numbers without a
    /// trailing `.0` and non-integers in their shortest round-tripping
    /// form, so components are interpolated directly. A component that is
    /// non-finite (possible only if this matrix was built from an
    /// attribute value whose numbers overflowed `f64`, or from arithmetic
    /// that produced infinity/NaN) formats as Rust's `inf`/`-inf`/`NaN`,
    /// which is not valid SVG and will not round-trip through
    /// [`crate::values::transform`]'s parser; this crate does not clamp
    /// such values since doing so would silently discard information.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix({}, {}, {}, {}, {}, {})",
            self.a, self.b, self.c, self.d, self.e, self.f
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: (f64, f64), expected: (f64, f64)) {
        assert!(
            (actual.0 - expected.0).abs() < 1e-9 && (actual.1 - expected.1).abs() < 1e-9,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn identity_leaves_points_unchanged() {
        let m = Matrix::identity();
        assert!(m.is_identity());
        assert_eq!(m.apply(3.0, 4.0), (3.0, 4.0));
        assert_eq!(Matrix::default(), m);
    }

    #[test]
    fn translate_offsets_points() {
        let m = Matrix::translate(5.0, -3.0);
        assert_eq!(m.apply(1.0, 1.0), (6.0, -2.0));
        assert!(!m.is_identity());
    }

    #[test]
    fn scale_scales_each_axis_independently() {
        let m = Matrix::scale(2.0, 3.0);
        assert_eq!(m.apply(1.0, 1.0), (2.0, 3.0));
        assert_eq!(m.apply(0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn rotate_90_degrees_maps_x_axis_onto_y_axis() {
        let m = Matrix::rotate(90.0);
        approx(m.apply(1.0, 0.0), (0.0, 1.0));
    }

    #[test]
    fn rotate_about_orbits_the_given_center() {
        // Orbiting (2, 0) by 90 degrees about the center (1, 0): the point
        // starts one unit to the +x side of the center and should end up
        // one unit to the "rotate(90) of +x", i.e. +y, side of the center.
        let m = Matrix::rotate_about(90.0, 1.0, 0.0);
        approx(m.apply(2.0, 0.0), (1.0, 1.0));
        // The center itself never moves.
        approx(m.apply(1.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn skew_x_shifts_x_by_y_times_tan_angle() {
        let m = Matrix::skew_x(45.0);
        approx(m.apply(0.0, 1.0), (1.0, 1.0));
        approx(m.apply(0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn skew_y_shifts_y_by_x_times_tan_angle() {
        let m = Matrix::skew_y(45.0);
        approx(m.apply(1.0, 0.0), (1.0, 1.0));
        approx(m.apply(0.0, 0.0), (0.0, 0.0));
    }

    /// The worked example from the SVG coordinate-systems specification
    /// text this crate's spec catalog transcribes: `translate(10,0)
    /// rotate(90)` applied to `(1,0)` must yield `(10,1)`.
    #[test]
    fn multiply_matches_the_spec_worked_example() {
        let composed = Matrix::translate(10.0, 0.0).multiply(&Matrix::rotate(90.0));
        approx(composed.apply(1.0, 0.0), (10.0, 1.0));
    }

    /// The mission's own numeric lock: `translate(10,0)` then `scale(2)`
    /// must map `(1,1)` to `(12,2)`. Uses only translation and scaling (no
    /// trigonometry), so the result is exact, not merely approximate.
    #[test]
    fn multiply_composes_translate_then_scale_exactly() {
        let composed = Matrix::translate(10.0, 0.0).multiply(&Matrix::scale(2.0, 2.0));
        assert_eq!(composed.apply(1.0, 1.0), (12.0, 2.0));
    }

    #[test]
    fn pre_concat_and_post_concat_apply_other_on_opposite_sides() {
        let a = Matrix::translate(10.0, 0.0);
        let b = Matrix::scale(2.0, 2.0);

        // pre_concat: `b` acts first (inner), `a` acts second (outer).
        assert_eq!(a.pre_concat(&b), a.multiply(&b));
        assert_eq!(a.pre_concat(&b).apply(1.0, 1.0), (12.0, 2.0));

        // post_concat: `a` acts first (inner), `b` acts second (outer).
        assert_eq!(a.post_concat(&b), b.multiply(&a));
        assert_eq!(a.post_concat(&b).apply(1.0, 1.0), (22.0, 2.0));
    }

    #[test]
    fn multiply_is_associative_for_a_three_step_chain() {
        let t1 = Matrix::translate(1.0, 2.0);
        let t2 = Matrix::rotate(30.0);
        let t3 = Matrix::scale(2.0, 0.5);

        let left_fold = t1.multiply(&t2).multiply(&t3);
        let right_fold = t1.multiply(&t2.multiply(&t3));
        assert!(left_fold.approx_eq(&right_fold, 1e-9));
    }

    #[test]
    fn determinant_of_identity_and_scale() {
        assert_eq!(Matrix::identity().determinant(), 1.0);
        assert_eq!(Matrix::scale(2.0, 3.0).determinant(), 6.0);
        assert_eq!(Matrix::scale(0.0, 5.0).determinant(), 0.0);
    }

    #[test]
    fn invert_undoes_translate_scale_and_rotate() {
        for m in [
            Matrix::translate(5.0, -7.0),
            Matrix::scale(2.0, 4.0),
            Matrix::rotate(37.0),
            Matrix::rotate_about(60.0, 3.0, -2.0),
            Matrix::skew_x(20.0),
        ] {
            let inv = m.invert().expect("these matrices are all invertible");
            let round_trip = m.multiply(&inv);
            assert!(
                round_trip.approx_eq(&Matrix::identity(), 1e-9),
                "m * inverse(m) should be the identity, got {round_trip:?}"
            );
        }
    }

    #[test]
    fn invert_returns_none_for_singular_matrices() {
        // Rows (1,2) and (2,4) are proportional: determinant is zero.
        let singular = Matrix::new(1.0, 2.0, 2.0, 4.0, 0.0, 0.0);
        assert_eq!(singular.determinant(), 0.0);
        assert_eq!(singular.invert(), None);

        assert_eq!(Matrix::scale(0.0, 1.0).invert(), None);
    }

    #[test]
    fn approx_eq_respects_the_given_epsilon() {
        let a = Matrix::identity();
        let b = Matrix::new(1.0 + 1e-7, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!(a.approx_eq(&b, 1e-6));
        assert!(!a.approx_eq(&b, 1e-9));
    }

    #[test]
    fn display_uses_the_matrix_function_syntax() {
        let m = Matrix::new(1.0, 0.0, 0.0, 1.0, 5.0, 7.5);
        assert_eq!(m.to_string(), "matrix(1, 0, 0, 1, 5, 7.5)");
    }

    #[test]
    fn display_omits_trailing_zero_for_whole_numbers() {
        // Rust's f64 Display already formats whole numbers without a
        // trailing ".0"; this test pins that assumption since the writer
        // relies on it for "minimal" number formatting.
        assert_eq!(1.0_f64.to_string(), "1");
        assert_eq!(Matrix::identity().to_string(), "matrix(1, 0, 0, 1, 0, 0)");
    }
}
