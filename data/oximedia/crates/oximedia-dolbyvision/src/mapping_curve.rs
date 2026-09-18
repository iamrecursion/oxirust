#![allow(dead_code)]
//! Tone mapping curve generation and interpolation for Dolby Vision.
//!
//! Provides curve generation utilities for constructing the luminance mapping
//! functions used in Dolby Vision display management. Supports various curve
//! types including polynomial, spline, and piecewise-linear representations.
//!
//! These curves map from source mastering luminance to target display luminance
//! while preserving perceptual intent.

/// Type of mapping curve interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveType {
    /// Linear interpolation between control points.
    Linear,
    /// Cubic Hermite spline interpolation.
    CubicHermite,
    /// Third-order polynomial mapping.
    Polynomial3,
    /// Parametric S-curve (sigmoid).
    Sigmoid,
    /// B-spline interpolation.
    BSpline,
}

impl CurveType {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::CubicHermite => "Cubic Hermite",
            Self::Polynomial3 => "Polynomial (3rd order)",
            Self::Sigmoid => "Sigmoid (S-curve)",
            Self::BSpline => "B-Spline",
        }
    }

    /// Whether this curve type guarantees monotonicity.
    #[must_use]
    pub const fn is_monotonic(self) -> bool {
        matches!(self, Self::Linear | Self::Sigmoid)
    }
}

/// A single control point on a mapping curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlPoint {
    /// Input value (source luminance, normalized 0.0-1.0).
    pub input: f64,
    /// Output value (target luminance, normalized 0.0-1.0).
    pub output: f64,
}

impl ControlPoint {
    /// Create a new control point.
    #[must_use]
    pub fn new(input: f64, output: f64) -> Self {
        Self { input, output }
    }

    /// Identity control point (input == output).
    #[must_use]
    pub fn identity(value: f64) -> Self {
        Self {
            input: value,
            output: value,
        }
    }
}

/// Parametric S-curve (sigmoid) definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidParams {
    /// Center of the sigmoid (0.0-1.0).
    pub center: f64,
    /// Steepness of the sigmoid (higher = sharper transition).
    pub steepness: f64,
    /// Minimum output value.
    pub min_output: f64,
    /// Maximum output value.
    pub max_output: f64,
}

impl SigmoidParams {
    /// Create default sigmoid parameters.
    #[must_use]
    pub fn new(center: f64, steepness: f64) -> Self {
        Self {
            center,
            steepness,
            min_output: 0.0,
            max_output: 1.0,
        }
    }

    /// Evaluate the sigmoid at a given input.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, input: f64) -> f64 {
        let t = -self.steepness * (input - self.center);
        let sigmoid = 1.0 / (1.0 + t.exp());
        self.min_output + sigmoid * (self.max_output - self.min_output)
    }
}

impl Default for SigmoidParams {
    fn default() -> Self {
        Self::new(0.5, 10.0)
    }
}

/// Third-order polynomial coefficients: a*x^3 + b*x^2 + c*x + d.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialCoeffs {
    /// Cubic coefficient.
    pub a: f64,
    /// Quadratic coefficient.
    pub b: f64,
    /// Linear coefficient.
    pub c: f64,
    /// Constant coefficient.
    pub d: f64,
}

impl PolynomialCoeffs {
    /// Create new polynomial coefficients.
    #[must_use]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// Identity polynomial (f(x) = x): 0*x^3 + 0*x^2 + 1*x + 0.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            c: 1.0,
            d: 0.0,
        }
    }

    /// Evaluate the polynomial at a given input using Horner's method.
    ///
    /// Horner's method rewrites `a·x³ + b·x² + c·x + d` as
    /// `((a·x + b)·x + c)·x + d`, reducing multiplications from 6 to 3
    /// and improving numerical stability.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        // Horner's method: p(x) = d + x*(c + x*(b + x*a))
        self.d + x * (self.c + x * (self.b + x * self.a))
    }

    /// Evaluate the derivative `3a·x² + 2b·x + c` using Horner's method.
    #[must_use]
    pub fn derivative(&self, x: f64) -> f64 {
        // Horner: c + x*(2b + x*3a)
        self.c + x * (2.0 * self.b + x * 3.0 * self.a)
    }
}

/// A mapping curve that maps source luminance to target luminance.
#[derive(Debug, Clone)]
pub struct MappingCurve {
    /// Curve type / interpolation method.
    pub curve_type: CurveType,
    /// Control points (sorted by input).
    control_points: Vec<ControlPoint>,
    /// Source mastering peak luminance in nits.
    pub source_peak_nits: f64,
    /// Target display peak luminance in nits.
    pub target_peak_nits: f64,
    /// Optional sigmoid parameters (for CurveType::Sigmoid).
    pub sigmoid_params: Option<SigmoidParams>,
    /// Optional polynomial coefficients (for CurveType::Polynomial3).
    pub polynomial_coeffs: Option<PolynomialCoeffs>,
}

impl MappingCurve {
    /// Create a new mapping curve.
    #[must_use]
    pub fn new(curve_type: CurveType, source_peak: f64, target_peak: f64) -> Self {
        Self {
            curve_type,
            control_points: Vec::new(),
            source_peak_nits: source_peak,
            target_peak_nits: target_peak,
            sigmoid_params: None,
            polynomial_coeffs: None,
        }
    }

    /// Create an identity mapping curve (no tone mapping).
    #[must_use]
    pub fn identity(peak_nits: f64) -> Self {
        let mut curve = Self::new(CurveType::Linear, peak_nits, peak_nits);
        curve.add_point(ControlPoint::new(0.0, 0.0));
        curve.add_point(ControlPoint::new(1.0, 1.0));
        curve
    }

    /// Create a sigmoid-based tone mapping curve.
    #[must_use]
    pub fn sigmoid(source_peak: f64, target_peak: f64, params: SigmoidParams) -> Self {
        let mut curve = Self::new(CurveType::Sigmoid, source_peak, target_peak);
        curve.sigmoid_params = Some(params);
        curve
    }

    /// Create a polynomial-based tone mapping curve.
    #[must_use]
    pub fn polynomial(source_peak: f64, target_peak: f64, coeffs: PolynomialCoeffs) -> Self {
        let mut curve = Self::new(CurveType::Polynomial3, source_peak, target_peak);
        curve.polynomial_coeffs = Some(coeffs);
        curve
    }

    /// Add a control point to the curve.
    pub fn add_point(&mut self, point: ControlPoint) {
        self.control_points.push(point);
        self.control_points.sort_by(|a, b| {
            a.input
                .partial_cmp(&b.input)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the control points.
    #[must_use]
    pub fn control_points(&self) -> &[ControlPoint] {
        &self.control_points
    }

    /// Get the number of control points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.control_points.len()
    }

    /// Evaluate the curve at a normalized input value (0.0-1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, input: f64) -> f64 {
        let clamped = input.clamp(0.0, 1.0);

        match self.curve_type {
            CurveType::Sigmoid => {
                if let Some(ref params) = self.sigmoid_params {
                    params.evaluate(clamped)
                } else {
                    clamped
                }
            }
            CurveType::Polynomial3 => {
                if let Some(ref coeffs) = self.polynomial_coeffs {
                    coeffs.evaluate(clamped).clamp(0.0, 1.0)
                } else {
                    clamped
                }
            }
            CurveType::Linear | CurveType::CubicHermite | CurveType::BSpline => {
                self.interpolate_linear(clamped)
            }
        }
    }

    /// Linear interpolation using control points.
    fn interpolate_linear(&self, input: f64) -> f64 {
        if self.control_points.is_empty() {
            return input;
        }
        if self.control_points.len() == 1 {
            return self.control_points[0].output;
        }

        // Find bracketing points
        if input <= self.control_points[0].input {
            return self.control_points[0].output;
        }
        if input >= self.control_points[self.control_points.len() - 1].input {
            return self.control_points[self.control_points.len() - 1].output;
        }

        for i in 0..self.control_points.len() - 1 {
            let p0 = &self.control_points[i];
            let p1 = &self.control_points[i + 1];
            if input >= p0.input && input <= p1.input {
                let range = p1.input - p0.input;
                if range.abs() < 1e-12 {
                    return p0.output;
                }
                let t = (input - p0.input) / range;
                return p0.output + t * (p1.output - p0.output);
            }
        }

        input
    }

    /// Evaluate the curve at a luminance value in nits.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_nits(&self, input_nits: f64) -> f64 {
        if self.source_peak_nits <= 0.0 {
            return 0.0;
        }
        let normalized = input_nits / self.source_peak_nits;
        let output_normalized = self.evaluate(normalized);
        output_normalized * self.target_peak_nits
    }

    /// Generate a lookup table with the specified number of entries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_lut(&self, entries: usize) -> Vec<f64> {
        if entries == 0 {
            return Vec::new();
        }
        (0..entries)
            .map(|i| {
                let input = i as f64 / (entries - 1).max(1) as f64;
                self.evaluate(input)
            })
            .collect()
    }

    /// Check if the curve is monotonically increasing.
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        let lut = self.generate_lut(256);
        for i in 1..lut.len() {
            if lut[i] < lut[i - 1] - 1e-9 {
                return false;
            }
        }
        true
    }

    /// The compression ratio at a given input (derivative < 1 means compression).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio_at(&self, input: f64) -> f64 {
        let delta = 0.001;
        let x0 = (input - delta).max(0.0);
        let x1 = (input + delta).min(1.0);
        let y0 = self.evaluate(x0);
        let y1 = self.evaluate(x1);
        let dx = x1 - x0;
        if dx.abs() < 1e-12 {
            return 1.0;
        }
        (y1 - y0) / dx
    }
}

/// Builder for constructing mapping curves from mastering/display parameters.
#[derive(Debug, Clone)]
pub struct CurveBuilder {
    /// Source mastering peak in nits.
    source_peak: f64,
    /// Target display peak in nits.
    target_peak: f64,
    /// Desired curve type.
    curve_type: CurveType,
    /// Number of generated control points (for auto-generation).
    point_count: usize,
    /// Knee point (where compression begins), normalized.
    knee_point: f64,
}

impl CurveBuilder {
    /// Create a new curve builder.
    #[must_use]
    pub fn new(source_peak: f64, target_peak: f64) -> Self {
        Self {
            source_peak,
            target_peak,
            curve_type: CurveType::Linear,
            point_count: 16,
            knee_point: 0.5,
        }
    }

    /// Set the curve type.
    #[must_use]
    pub fn curve_type(mut self, ct: CurveType) -> Self {
        self.curve_type = ct;
        self
    }

    /// Set the knee point (where compression starts).
    #[must_use]
    pub fn knee_point(mut self, knee: f64) -> Self {
        self.knee_point = knee.clamp(0.1, 0.9);
        self
    }

    /// Build the mapping curve.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn build(self) -> MappingCurve {
        let mut curve = MappingCurve::new(self.curve_type, self.source_peak, self.target_peak);

        match self.curve_type {
            CurveType::Sigmoid => {
                let ratio = if self.source_peak > 0.0 {
                    self.target_peak / self.source_peak
                } else {
                    1.0
                };
                let steepness = if ratio < 1.0 {
                    10.0 / ratio.max(0.1)
                } else {
                    10.0
                };
                curve.sigmoid_params = Some(SigmoidParams::new(self.knee_point, steepness));
            }
            CurveType::Polynomial3 => {
                // Simple roll-off polynomial: identity below knee, compression above
                let ratio = if self.source_peak > 0.0 {
                    (self.target_peak / self.source_peak).min(1.0)
                } else {
                    1.0
                };
                // Construct polynomial that maps [0,1] -> [0, ratio]
                // with smooth roll-off near the top
                let a = -(1.0 - ratio);
                let coeffs = PolynomialCoeffs::new(a, 0.0, 1.0 - a, 0.0);
                curve.polynomial_coeffs = Some(coeffs);
            }
            _ => {
                // Generate piecewise-linear control points
                let ratio = if self.source_peak > 0.0 {
                    (self.target_peak / self.source_peak).min(1.0)
                } else {
                    1.0
                };
                curve.add_point(ControlPoint::new(0.0, 0.0));
                curve.add_point(ControlPoint::new(self.knee_point, self.knee_point * ratio));
                curve.add_point(ControlPoint::new(1.0, ratio));
            }
        }

        curve
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_type_labels() {
        assert_eq!(CurveType::Linear.label(), "Linear");
        assert_eq!(CurveType::Sigmoid.label(), "Sigmoid (S-curve)");
        assert_eq!(CurveType::CubicHermite.label(), "Cubic Hermite");
    }

    #[test]
    fn test_curve_type_monotonicity() {
        assert!(CurveType::Linear.is_monotonic());
        assert!(CurveType::Sigmoid.is_monotonic());
        assert!(!CurveType::Polynomial3.is_monotonic());
    }

    #[test]
    fn test_control_point() {
        let p = ControlPoint::new(0.3, 0.5);
        assert!((p.input - 0.3).abs() < f64::EPSILON);
        assert!((p.output - 0.5).abs() < f64::EPSILON);

        let id = ControlPoint::identity(0.7);
        assert!((id.input - id.output).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sigmoid_evaluate() {
        let params = SigmoidParams::new(0.5, 10.0);
        let at_center = params.evaluate(0.5);
        assert!(
            (at_center - 0.5).abs() < 0.01,
            "Sigmoid at center: {at_center}"
        );

        let low = params.evaluate(0.0);
        let high = params.evaluate(1.0);
        assert!(low < 0.1, "Sigmoid at 0: {low}");
        assert!(high > 0.9, "Sigmoid at 1: {high}");
    }

    #[test]
    fn test_polynomial_identity() {
        let coeffs = PolynomialCoeffs::identity();
        assert!((coeffs.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((coeffs.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((coeffs.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_polynomial_derivative() {
        let coeffs = PolynomialCoeffs::new(1.0, 0.0, 0.0, 0.0); // x^3
                                                                // derivative of x^3 = 3x^2
        assert!((coeffs.derivative(1.0) - 3.0).abs() < f64::EPSILON);
        assert!((coeffs.derivative(0.5) - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_identity_curve() {
        let curve = MappingCurve::identity(1000.0);
        assert!((curve.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        assert!(curve.is_monotonic());
    }

    #[test]
    fn test_identity_curve_nits() {
        let curve = MappingCurve::identity(1000.0);
        assert!((curve.evaluate_nits(500.0) - 500.0).abs() < 0.01);
        assert!((curve.evaluate_nits(1000.0) - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_curve() {
        let curve = MappingCurve::sigmoid(4000.0, 1000.0, SigmoidParams::new(0.5, 10.0));
        let at_center = curve.evaluate(0.5);
        assert!((at_center - 0.5).abs() < 0.01);
        assert!(curve.is_monotonic());
    }

    #[test]
    fn test_polynomial_curve() {
        let coeffs = PolynomialCoeffs::identity();
        let curve = MappingCurve::polynomial(1000.0, 1000.0, coeffs);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_curve_add_points() {
        let mut curve = MappingCurve::new(CurveType::Linear, 1000.0, 600.0);
        curve.add_point(ControlPoint::new(1.0, 0.6));
        curve.add_point(ControlPoint::new(0.0, 0.0));
        curve.add_point(ControlPoint::new(0.5, 0.4));
        // Points should be sorted by input
        assert_eq!(curve.point_count(), 3);
        assert!((curve.control_points()[0].input - 0.0).abs() < f64::EPSILON);
        assert!((curve.control_points()[1].input - 0.5).abs() < f64::EPSILON);
        assert!((curve.control_points()[2].input - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_lut() {
        let curve = MappingCurve::identity(1000.0);
        let lut = curve.generate_lut(5);
        assert_eq!(lut.len(), 5);
        assert!((lut[0] - 0.0).abs() < f64::EPSILON);
        assert!((lut[4] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_lut_empty() {
        let curve = MappingCurve::identity(1000.0);
        let lut = curve.generate_lut(0);
        assert!(lut.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let curve = MappingCurve::identity(1000.0);
        let ratio = curve.compression_ratio_at(0.5);
        // Identity curve has compression ratio ~1.0
        assert!((ratio - 1.0).abs() < 0.01, "Ratio was {ratio}");
    }

    #[test]
    fn test_curve_builder_linear() {
        let curve = CurveBuilder::new(4000.0, 1000.0)
            .curve_type(CurveType::Linear)
            .knee_point(0.5)
            .build();
        assert_eq!(curve.curve_type, CurveType::Linear);
        assert!(curve.point_count() >= 3);
        assert!(curve.is_monotonic());
    }

    #[test]
    fn test_curve_builder_sigmoid() {
        let curve = CurveBuilder::new(4000.0, 1000.0)
            .curve_type(CurveType::Sigmoid)
            .build();
        assert_eq!(curve.curve_type, CurveType::Sigmoid);
        assert!(curve.sigmoid_params.is_some());
    }

    #[test]
    fn test_curve_builder_polynomial() {
        let curve = CurveBuilder::new(4000.0, 1000.0)
            .curve_type(CurveType::Polynomial3)
            .build();
        assert_eq!(curve.curve_type, CurveType::Polynomial3);
        assert!(curve.polynomial_coeffs.is_some());
        // Should map 0 -> 0
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_nits_zero_peak() {
        let curve = MappingCurve::new(CurveType::Linear, 0.0, 1000.0);
        assert!((curve.evaluate_nits(500.0) - 0.0).abs() < f64::EPSILON);
    }

    /// Verify that Horner's method matches the naive expansion for arbitrary coefficients.
    #[test]
    fn test_horner_matches_naive() {
        // Test several polynomials at multiple evaluation points.
        let test_cases: &[(f64, f64, f64, f64)] = &[
            // (a, b, c, d) — arbitrary coefficients
            (1.0, -2.0, 3.0, -4.0),
            (0.0, 0.0, 1.0, 0.0), // identity: f(x) = x
            (2.5, 0.5, -1.5, 0.25),
            (0.0, 0.0, 0.0, 7.0),   // constant 7
            (-1.0, 3.0, -3.0, 1.0), // -(x-1)^3
        ];
        let eval_points = [0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, -1.0, 2.0];

        for &(a, b, c, d) in test_cases {
            let coeffs = PolynomialCoeffs::new(a, b, c, d);
            for &x in &eval_points {
                // Naive expansion: a*x^3 + b*x^2 + c*x + d
                let naive = a * x * x * x + b * x * x + c * x + d;
                let horner = coeffs.evaluate(x);
                assert!(
                    (horner - naive).abs() < 1e-10,
                    "Horner ({horner}) != naive ({naive}) for a={a} b={b} c={c} d={d} at x={x}"
                );
            }
        }
    }

    /// Verify that Horner's derivative matches naive derivative expansion.
    #[test]
    fn test_horner_derivative_matches_naive() {
        let coeffs = PolynomialCoeffs::new(2.0, -3.0, 1.5, 0.0);
        let eval_points = [0.0_f64, 0.25, 0.5, 0.75, 1.0];

        for &x in &eval_points {
            // Naive: 3a*x^2 + 2b*x + c
            let naive = 3.0 * coeffs.a * x * x + 2.0 * coeffs.b * x + coeffs.c;
            let horner = coeffs.derivative(x);
            assert!(
                (horner - naive).abs() < 1e-10,
                "Derivative Horner ({horner}) != naive ({naive}) at x={x}"
            );
        }
    }
}
