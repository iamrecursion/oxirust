//! Long-form timing drift correction.
//!
//! Provides tools to model and correct gradual clock drift between recording
//! devices over extended recording sessions:
//!
//! - [`DriftMeasurement`] – an observed drift sample at a given time.
//! - [`LinearDriftEstimator`] – least-squares linear drift model.
//! - [`DriftModel`] – choice of correction model.
//! - [`DriftCorrector`] – applies the fitted model to compute per-timestamp
//!   corrections.
//! - [`DriftQuality`] – evaluates the quality of a fitted model.

#![allow(dead_code)]

/// A single drift observation: the measured offset between two clocks at
/// `time_ms` milliseconds into the recording.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftMeasurement {
    /// Elapsed time in milliseconds since the start of the recording.
    pub time_ms: u64,
    /// Measured drift (in milliseconds) at `time_ms`.  Positive means the
    /// secondary device is ahead of the reference.
    pub drift_ms: f64,
}

impl DriftMeasurement {
    /// Create a new drift measurement.
    #[must_use]
    pub fn new(time_ms: u64, drift_ms: f64) -> Self {
        Self { time_ms, drift_ms }
    }
}

/// Selects which mathematical model to use for drift correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftModel {
    /// First-order (linear) drift: `drift = slope * t + intercept`.
    Linear,
    /// Second-order polynomial: `drift = a*t² + b*t + c`.
    Polynomial,
    /// Piecewise-linear: drift is linearly interpolated between measurement
    /// points.
    PiecewiseLinear,
}

/// Fits a linear drift model via ordinary least squares.
pub struct LinearDriftEstimator;

impl LinearDriftEstimator {
    /// Fit a linear model to the provided measurements.
    ///
    /// Returns `(slope_ms_per_sec, intercept_ms)` where slope is the rate of
    /// drift in milliseconds per second, and intercept is the drift at `t=0`.
    ///
    /// Falls back to `(0.0, 0.0)` when fewer than 2 measurements are provided.
    #[must_use]
    pub fn fit(measurements: &[DriftMeasurement]) -> (f64, f64) {
        let n = measurements.len();
        if n < 2 {
            return (0.0, 0.0);
        }

        // Convert time to seconds for numerically stable regression.
        let xs: Vec<f64> = measurements
            .iter()
            .map(|m| m.time_ms as f64 / 1000.0)
            .collect();
        let ys: Vec<f64> = measurements.iter().map(|m| m.drift_ms).collect();

        let n_f = n as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = xs.iter().map(|x| x * x).sum();

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            // All time values are identical → only an intercept can be estimated.
            let intercept = sum_y / n_f;
            return (0.0, intercept);
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        (slope, intercept)
    }
}

/// Applies a fitted drift model to compute per-timestamp corrections.
#[derive(Debug, Clone)]
pub struct DriftCorrector {
    /// The model variant in use.
    pub model: DriftModel,
    /// Model coefficients (interpretation depends on `model`):
    /// - `Linear`: `[slope_ms_per_sec, intercept_ms]`
    /// - `Polynomial`: `[a, b, c]` for `a*t²+b*t+c` (t in seconds)
    /// - `PiecewiseLinear`: interleaved `[t0_s, d0, t1_s, d1, …]`
    pub coefficients: Vec<f64>,
}

impl DriftCorrector {
    /// Create a new corrector with explicit model and coefficients.
    #[must_use]
    pub fn new(model: DriftModel, coefficients: Vec<f64>) -> Self {
        Self {
            model,
            coefficients,
        }
    }

    /// Fit a corrector from observations using the specified model.
    #[must_use]
    pub fn from_measurements(measurements: &[DriftMeasurement], model: DriftModel) -> Self {
        match model {
            DriftModel::Linear => {
                let (slope, intercept) = LinearDriftEstimator::fit(measurements);
                Self::new(model, vec![slope, intercept])
            }
            DriftModel::Polynomial => {
                // Simple quadratic fit via normal equations (3×3 system).
                let coeffs = fit_quadratic(measurements);
                Self::new(model, coeffs)
            }
            DriftModel::PiecewiseLinear => {
                // Store measurement pairs directly.
                let mut coeffs = Vec::with_capacity(measurements.len() * 2);
                for m in measurements {
                    coeffs.push(m.time_ms as f64 / 1000.0);
                    coeffs.push(m.drift_ms);
                }
                Self::new(model, coeffs)
            }
        }
    }

    /// Compute the drift correction (in milliseconds, rounded to integer) at
    /// `time_ms` milliseconds.
    ///
    /// The correction to *apply* to the secondary device's timestamp is the
    /// negative of the predicted drift: `corrected = original - correction`.
    #[must_use]
    pub fn correct(&self, time_ms: u64) -> i64 {
        let t_s = time_ms as f64 / 1000.0;
        let drift = match self.model {
            DriftModel::Linear => {
                let slope = self.coefficients.first().copied().unwrap_or(0.0);
                let intercept = self.coefficients.get(1).copied().unwrap_or(0.0);
                slope * t_s + intercept
            }
            DriftModel::Polynomial => {
                let a = self.coefficients.first().copied().unwrap_or(0.0);
                let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                let c = self.coefficients.get(2).copied().unwrap_or(0.0);
                a * t_s * t_s + b * t_s + c
            }
            DriftModel::PiecewiseLinear => piecewise_linear_eval(&self.coefficients, t_s),
        };
        drift.round() as i64
    }
}

/// Quality metrics for a fitted drift model.
#[derive(Debug, Clone, Copy)]
pub struct DriftQuality {
    /// Root-mean-square of the residuals (in ms).
    pub rms_error_ms: f64,
    /// Maximum absolute residual (in ms).
    pub max_error_ms: f64,
    /// Coefficient of determination R².
    pub r_squared: f64,
}

impl DriftQuality {
    /// Evaluate the quality of `model` against the provided measurements.
    #[must_use]
    pub fn evaluate(model: &DriftCorrector, measurements: &[DriftMeasurement]) -> Self {
        if measurements.is_empty() {
            return Self {
                rms_error_ms: 0.0,
                max_error_ms: 0.0,
                r_squared: 1.0,
            };
        }

        let n = measurements.len() as f64;
        let mean_drift = measurements.iter().map(|m| m.drift_ms).sum::<f64>() / n;

        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        let mut max_err = 0.0f64;

        for m in measurements {
            let predicted = model.correct(m.time_ms) as f64;
            let residual = m.drift_ms - predicted;
            ss_res += residual * residual;
            ss_tot += (m.drift_ms - mean_drift) * (m.drift_ms - mean_drift);
            max_err = max_err.max(residual.abs());
        }

        let rms = (ss_res / n).sqrt();
        let r2 = if ss_tot < 1e-12 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Self {
            rms_error_ms: rms,
            max_error_ms: max_err,
            r_squared: r2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a quadratic `y = a*t² + b*t + c` to the measurements using normal
/// equations.  Falls back to linear when fewer than 3 points.
fn fit_quadratic(measurements: &[DriftMeasurement]) -> Vec<f64> {
    let n = measurements.len();
    if n < 3 {
        let (slope, intercept) = LinearDriftEstimator::fit(measurements);
        return vec![0.0, slope, intercept];
    }

    // Build sums for the 3×3 normal equations: X'X β = X'y
    //   X = [t², t, 1],  y = drift
    let xs: Vec<f64> = measurements
        .iter()
        .map(|m| m.time_ms as f64 / 1000.0)
        .collect();
    let ys: Vec<f64> = measurements.iter().map(|m| m.drift_ms).collect();

    let n_f = n as f64;
    let s1: f64 = xs.iter().sum();
    let s2: f64 = xs.iter().map(|x| x * x).sum();
    let s3: f64 = xs.iter().map(|x| x * x * x).sum();
    let s4: f64 = xs.iter().map(|x| x * x * x * x).sum();
    let t0: f64 = ys.iter().sum();
    let t1: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let t2: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * x * y).sum();

    // 3×3 system: [[s4, s3, s2], [s3, s2, s1], [s2, s1, n]] [a,b,c]' = [t2,t1,t0]
    let mat = [[s4, s3, s2], [s3, s2, s1], [s2, s1, n_f]];
    let rhs = [t2, t1, t0];

    if let Some([a, b, c]) = solve_3x3(&mat, &rhs) {
        vec![a, b, c]
    } else {
        // Singular matrix – fall back to linear.
        let (slope, intercept) = LinearDriftEstimator::fit(measurements);
        vec![0.0, slope, intercept]
    }
}

/// Solve a 3×3 linear system via Cramer's rule.  Returns `None` if singular.
fn solve_3x3(m: &[[f64; 3]; 3], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let mut result = [0.0f64; 3];
    for k in 0..3 {
        let mut mat_k = *m;
        for i in 0..3 {
            mat_k[i][k] = rhs[i];
        }
        let det_k = mat_k[0][0] * (mat_k[1][1] * mat_k[2][2] - mat_k[1][2] * mat_k[2][1])
            - mat_k[0][1] * (mat_k[1][0] * mat_k[2][2] - mat_k[1][2] * mat_k[2][0])
            + mat_k[0][2] * (mat_k[1][0] * mat_k[2][1] - mat_k[1][1] * mat_k[2][0]);
        result[k] = det_k / det;
    }
    Some(result)
}

/// Evaluate a piecewise-linear function stored as interleaved `[t, v, …]` pairs
/// at time `t_s`.
fn piecewise_linear_eval(coeffs: &[f64], t_s: f64) -> f64 {
    if coeffs.len() < 2 {
        return 0.0;
    }

    // Pairs: (t0, d0), (t1, d1), …
    let pairs: Vec<(f64, f64)> = coeffs.chunks(2).map(|c| (c[0], c[1])).collect();

    if t_s <= pairs[0].0 {
        return pairs[0].1;
    }
    let last = pairs[pairs.len() - 1];
    if t_s >= last.0 {
        return last.1;
    }

    for i in 0..pairs.len() - 1 {
        let (t0, d0) = pairs[i];
        let (t1, d1) = pairs[i + 1];
        if t_s >= t0 && t_s <= t1 {
            let alpha = (t_s - t0) / (t1 - t0);
            return d0 + alpha * (d1 - d0);
        }
    }
    0.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DriftMeasurement ─────────────────────────────────────────────────────

    #[test]
    fn test_measurement_creation() {
        let m = DriftMeasurement::new(5000, 1.5);
        assert_eq!(m.time_ms, 5000);
        assert!((m.drift_ms - 1.5).abs() < f64::EPSILON);
    }

    // ── LinearDriftEstimator ─────────────────────────────────────────────────

    #[test]
    fn test_linear_fit_insufficient_data() {
        let (slope, intercept) = LinearDriftEstimator::fit(&[]);
        assert_eq!(slope, 0.0);
        assert_eq!(intercept, 0.0);

        let (s, i) = LinearDriftEstimator::fit(&[DriftMeasurement::new(0, 1.0)]);
        assert_eq!(s, 0.0);
        assert_eq!(i, 0.0);
    }

    #[test]
    fn test_linear_fit_zero_drift() {
        let measurements: Vec<DriftMeasurement> = (0..5)
            .map(|i| DriftMeasurement::new(i * 1000, 0.0))
            .collect();
        let (slope, intercept) = LinearDriftEstimator::fit(&measurements);
        assert!(slope.abs() < 1e-9);
        assert!(intercept.abs() < 1e-9);
    }

    #[test]
    fn test_linear_fit_perfect_linear() {
        // drift = 2 * t_s + 0.5  (t in seconds)
        let measurements: Vec<DriftMeasurement> = (0..5)
            .map(|i| {
                let t_s = i as f64;
                DriftMeasurement::new((t_s * 1000.0) as u64, 2.0 * t_s + 0.5)
            })
            .collect();
        let (slope, intercept) = LinearDriftEstimator::fit(&measurements);
        assert!((slope - 2.0).abs() < 1e-6, "slope: {slope}");
        assert!((intercept - 0.5).abs() < 1e-6, "intercept: {intercept}");
    }

    // ── DriftCorrector (linear) ───────────────────────────────────────────────

    #[test]
    fn test_corrector_linear_zero() {
        let corrector = DriftCorrector::new(DriftModel::Linear, vec![0.0, 0.0]);
        assert_eq!(corrector.correct(0), 0);
        assert_eq!(corrector.correct(60_000), 0);
    }

    #[test]
    fn test_corrector_linear_constant_drift() {
        // slope = 0, intercept = 10 ms
        let corrector = DriftCorrector::new(DriftModel::Linear, vec![0.0, 10.0]);
        assert_eq!(corrector.correct(0), 10);
        assert_eq!(corrector.correct(30_000), 10);
    }

    #[test]
    fn test_corrector_from_measurements_linear() {
        let measurements = vec![
            DriftMeasurement::new(0, 0.0),
            DriftMeasurement::new(1_000, 1.0),
            DriftMeasurement::new(2_000, 2.0),
        ];
        let corrector = DriftCorrector::from_measurements(&measurements, DriftModel::Linear);
        // slope ≈ 1 ms/s, intercept ≈ 0
        let correction_at_3s = corrector.correct(3_000);
        assert!((correction_at_3s - 3).abs() <= 1, "got {correction_at_3s}");
    }

    // ── DriftCorrector (piecewise linear) ────────────────────────────────────

    #[test]
    fn test_corrector_piecewise_clamping() {
        let measurements = vec![
            DriftMeasurement::new(1_000, 5.0),
            DriftMeasurement::new(3_000, 15.0),
        ];
        let corrector =
            DriftCorrector::from_measurements(&measurements, DriftModel::PiecewiseLinear);
        // Before first point → clamp to first value.
        assert_eq!(corrector.correct(0), 5);
        // After last point → clamp to last value.
        assert_eq!(corrector.correct(10_000), 15);
        // At mid-point (2 s) → ~10.
        let mid = corrector.correct(2_000);
        assert!(
            (mid - 10).abs() <= 1,
            "mid correction should be ~10, got {mid}"
        );
    }

    // ── DriftQuality ─────────────────────────────────────────────────────────

    #[test]
    fn test_quality_perfect_fit() {
        let measurements = vec![
            DriftMeasurement::new(0, 0.0),
            DriftMeasurement::new(1_000, 1.0),
            DriftMeasurement::new(2_000, 2.0),
        ];
        let corrector = DriftCorrector::from_measurements(&measurements, DriftModel::Linear);
        let quality = DriftQuality::evaluate(&corrector, &measurements);
        assert!(quality.rms_error_ms < 0.5, "rms: {}", quality.rms_error_ms);
        assert!(quality.r_squared > 0.99, "r²: {}", quality.r_squared);
    }

    #[test]
    fn test_quality_empty_measurements() {
        let corrector = DriftCorrector::new(DriftModel::Linear, vec![0.0, 0.0]);
        let quality = DriftQuality::evaluate(&corrector, &[]);
        assert_eq!(quality.rms_error_ms, 0.0);
        assert!((quality.r_squared - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_fields_exist() {
        let corrector = DriftCorrector::new(DriftModel::Linear, vec![0.0, 5.0]);
        let measurements = vec![
            DriftMeasurement::new(0, 5.0),
            DriftMeasurement::new(1000, 5.0),
        ];
        let q = DriftQuality::evaluate(&corrector, &measurements);
        // Both measurements predict 5 ms perfectly.
        assert!(q.rms_error_ms < 0.1);
        assert!(q.max_error_ms < 0.1);
    }

    // ── DriftCorrector (polynomial) ───────────────────────────────────────────

    #[test]
    fn test_corrector_polynomial_from_measurements() {
        let measurements: Vec<DriftMeasurement> = (0..5)
            .map(|i| DriftMeasurement::new(i * 1000, (i * i) as f64))
            .collect();
        let corrector = DriftCorrector::from_measurements(&measurements, DriftModel::Polynomial);
        // At t=3 s, drift should be ≈ 9 ms.
        let c = corrector.correct(3_000);
        assert!((c - 9).abs() <= 2, "polynomial correction at 3s: {c}");
    }
}
