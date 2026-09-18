//! Newton-Cotes quadrature rule generator
//!
//! This module implements functions to generate Newton-Cotes quadrature rules,
//! which are numerical integration methods using polynomial interpolation at equally spaced points.
//! Newton-Cotes formulas can be either closed (including endpoints) or open (excluding endpoints).
//!
//! The module provides functionality similar to SciPy's `scipy.integrate.newton_cotes`.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;
// use scirs2_core::numeric::Float;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: IntegrateFloat>(value: f64) -> F {
    F::from_f64(value).expect("Failed to convert constant to target float type")
}

/// Represents the type of Newton-Cotes formula to generate
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NewtonCotesType {
    /// Closed formula (includes endpoints)
    Closed,
    /// Open formula (excludes endpoints)
    Open,
}

/// Result of generating a Newton-Cotes formula
#[derive(Debug, Clone)]
pub struct NewtonCotesResult<F: IntegrateFloat> {
    /// Integration points (nodes)
    pub points: Array1<F>,
    /// Quadrature weights
    pub weights: Array1<F>,
    /// Degree of exactness (highest degree polynomial integrated exactly)
    pub degree: usize,
    /// Error coefficient (for error estimation)
    pub error_coefficient: F,
}

/// Generates a Newton-Cotes quadrature formula of specified order
///
/// This function computes the points and weights for a Newton-Cotes integration
/// formula of the given order and type. The formula can be used to approximate
/// an integral over the interval [a, b]:
///
/// `∫[a,b] f(x) dx ≈ (b-a) × ∑[i=0..n] w_i × f(x_i)`
///
/// # Arguments
///
/// * `n` - Number of points used in the quadrature rule:
///   - For closed formula, n ≥ 1
///   - For open formula, n ≥ 3 (since endpoints are excluded)
/// * `formula_type` - Whether to use closed or open Newton-Cotes formula
/// * `a` - Optional lower bound of integration interval (default: 0)
/// * `b` - Optional upper bound of integration interval (default: 1)
///
/// # Returns
///
/// A `NewtonCotesResult` containing points, weights, degree of exactness,
/// and an error coefficient.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::newton_cotes::{newton_cotes, NewtonCotesType, NewtonCotesResult};
///
/// // Generate a 5-point closed Newton-Cotes formula (Boole's rule)
/// let result: NewtonCotesResult<f64> = newton_cotes(5, NewtonCotesType::Closed, None, None).expect("Test/example failed");
///
/// // Print weights
/// println!("Weights: {:?}", result.weights);
/// // Degree of exactness
/// println!("Degree of exactness: {}", result.degree);
/// ```
///
/// # Notes
///
/// - For closed formulas, common rules are:
///   - n=2: Trapezoidal rule
///   - n=3: Simpson's rule
///   - n=4: Simpson's 3/8 rule
///   - n=5: Boole's rule
///
/// - Higher-order rules (n > 8) may have poor numerical properties due to
///   Runge's phenomenon and should be used with caution.
#[allow(dead_code)]
pub fn newton_cotes<F: IntegrateFloat>(
    n: usize,
    formula_type: NewtonCotesType,
    a: Option<F>,
    b: Option<F>,
) -> IntegrateResult<NewtonCotesResult<F>> {
    // Handle parameter validation
    match formula_type {
        NewtonCotesType::Closed => {
            if n < 1 {
                return Err(IntegrateError::ValueError(
                    "Closed Newton-Cotes formula requires at least 1 point".to_string(),
                ));
            }
        }
        NewtonCotesType::Open => {
            if n < 3 {
                return Err(IntegrateError::ValueError(
                    "Open Newton-Cotes formula requires at least 3 points".to_string(),
                ));
            }
        }
    }

    // Extract bounds or use defaults
    let a = a.unwrap_or_else(|| const_f64::<F>(0.0));
    let b = b.unwrap_or_else(|| const_f64::<F>(1.0));

    if a >= b {
        return Err(IntegrateError::ValueError(
            "Integration bounds must satisfy a < b".to_string(),
        ));
    }

    // Generate nodes (integration points)
    let nodes = match formula_type {
        NewtonCotesType::Closed => {
            // Closed formula: n evenly spaced points from a to b (including endpoints)
            let mut points = Array1::zeros(n);
            let step = (b - a) / F::from(n - 1).expect("Failed to convert (n-1) to float");

            for i in 0..n {
                points[i] = a + F::from(i).expect("Failed to convert i to float") * step;
            }

            points
        }
        NewtonCotesType::Open => {
            // Open formula: n evenly spaced points from a to b (excluding endpoints)
            let mut points = Array1::zeros(n);
            let step = (b - a) / F::from(n + 1).expect("Failed to convert (n+1) to float");

            for i in 0..n {
                points[i] = a + F::from(i + 1).expect("Failed to convert (i+1) to float") * step;
            }

            points
        }
    };

    // Compute the weights using polynomial integration
    let weights: Array1<F> = calculate_weights(n, &formula_type)?;

    // Scale weights to the interval [a, b]
    let scaled_weights = weights.mapv(|w: F| w * (b - a));

    // Calculate degree of exactness (highest degree polynomial integrated exactly)
    let degree = match formula_type {
        NewtonCotesType::Closed => {
            if n.is_multiple_of(2) {
                n - 1
            } else {
                n
            }
        }
        NewtonCotesType::Open => n - 1,
    };

    // Calculate error coefficient (for error estimation)
    let error_coefficient = calculate_error_coefficient(n, &formula_type)?;

    Ok(NewtonCotesResult {
        points: nodes,
        weights: scaled_weights,
        degree,
        error_coefficient,
    })
}

/// Calculates the weights for a Newton-Cotes formula
#[allow(dead_code)]
fn calculate_weights<F: IntegrateFloat>(
    n: usize,
    formula_type: &NewtonCotesType,
) -> IntegrateResult<Array1<F>> {
    match formula_type {
        NewtonCotesType::Closed => {
            // Common closed Newton-Cotes formulas
            match n {
                1 => {
                    // Midpoint rule (degenerate case)
                    Ok(Array1::ones(1))
                }
                2 => {
                    // Trapezoidal rule
                    let mut w = Array1::zeros(2);
                    w[0] = F::from(0.5).expect("Test/example failed");
                    w[1] = F::from(0.5).expect("Test/example failed");
                    Ok(w)
                }
                3 => {
                    // Simpson's rule
                    let mut w = Array1::zeros(3);
                    w[0] = const_f64::<F>(1.0 / 6.0);
                    w[1] = const_f64::<F>(4.0 / 6.0);
                    w[2] = const_f64::<F>(1.0 / 6.0);
                    Ok(w)
                }
                4 => {
                    // Simpson's 3/8 rule
                    let mut w = Array1::zeros(4);
                    w[0] = const_f64::<F>(1.0 / 8.0);
                    w[1] = const_f64::<F>(3.0 / 8.0);
                    w[2] = const_f64::<F>(3.0 / 8.0);
                    w[3] = const_f64::<F>(1.0 / 8.0);
                    Ok(w)
                }
                5 => {
                    // Boole's rule
                    let mut w = Array1::zeros(5);
                    w[0] = const_f64::<F>(7.0 / 90.0);
                    w[1] = const_f64::<F>(32.0 / 90.0);
                    w[2] = const_f64::<F>(12.0 / 90.0);
                    w[3] = const_f64::<F>(32.0 / 90.0);
                    w[4] = const_f64::<F>(7.0 / 90.0);
                    Ok(w)
                }
                6 => {
                    // 6-point closed Newton-Cotes
                    let mut w = Array1::zeros(6);
                    w[0] = const_f64::<F>(19.0 / 288.0);
                    w[1] = const_f64::<F>(75.0 / 288.0);
                    w[2] = const_f64::<F>(50.0 / 288.0);
                    w[3] = const_f64::<F>(50.0 / 288.0);
                    w[4] = const_f64::<F>(75.0 / 288.0);
                    w[5] = const_f64::<F>(19.0 / 288.0);
                    Ok(w)
                }
                7 => {
                    // 7-point closed Newton-Cotes
                    let mut w = Array1::zeros(7);
                    w[0] = const_f64::<F>(41.0 / 840.0);
                    w[1] = const_f64::<F>(216.0 / 840.0);
                    w[2] = const_f64::<F>(27.0 / 840.0);
                    w[3] = const_f64::<F>(272.0 / 840.0);
                    w[4] = const_f64::<F>(27.0 / 840.0);
                    w[5] = const_f64::<F>(216.0 / 840.0);
                    w[6] = const_f64::<F>(41.0 / 840.0);
                    Ok(w)
                }
                8 => {
                    // 8-point closed Newton-Cotes
                    let mut w = Array1::zeros(8);
                    w[0] = const_f64::<F>(751.0 / 17280.0);
                    w[1] = const_f64::<F>(3577.0 / 17280.0);
                    w[2] = const_f64::<F>(1323.0 / 17280.0);
                    w[3] = const_f64::<F>(2989.0 / 17280.0);
                    w[4] = const_f64::<F>(2989.0 / 17280.0);
                    w[5] = const_f64::<F>(1323.0 / 17280.0);
                    w[6] = const_f64::<F>(3577.0 / 17280.0);
                    w[7] = const_f64::<F>(751.0 / 17280.0);
                    Ok(w)
                }
                _ => {
                    // For higher orders, use general formula based on polynomial integration
                    calculate_weights_general(n, formula_type)
                }
            }
        }
        NewtonCotesType::Open => {
            // Common open Newton-Cotes formulas
            match n {
                3 => {
                    // 3-point open Newton-Cotes
                    let mut w = Array1::zeros(3);
                    w[0] = const_f64::<F>(2.0 / 3.0);
                    w[1] = const_f64::<F>(-1.0 / 3.0);
                    w[2] = const_f64::<F>(2.0 / 3.0);
                    Ok(w)
                }
                4 => {
                    // 4-point open Newton-Cotes
                    let mut w = Array1::zeros(4);
                    w[0] = const_f64::<F>(11.0 / 24.0);
                    w[1] = const_f64::<F>(1.0 / 24.0);
                    w[2] = const_f64::<F>(1.0 / 24.0);
                    w[3] = const_f64::<F>(11.0 / 24.0);
                    Ok(w)
                }
                5 => {
                    // 5-point open Newton-Cotes
                    let mut w = Array1::zeros(5);
                    w[0] = const_f64::<F>(11.0 / 20.0);
                    w[1] = const_f64::<F>(-14.0 / 20.0);
                    w[2] = const_f64::<F>(26.0 / 20.0);
                    w[3] = const_f64::<F>(-14.0 / 20.0);
                    w[4] = const_f64::<F>(11.0 / 20.0);
                    Ok(w)
                }
                _ => {
                    // For higher orders, use general formula based on polynomial integration
                    calculate_weights_general(n, formula_type)
                }
            }
        }
    }
}

/// Calculates weights for Newton-Cotes formulas of any order using general approach
#[allow(dead_code)]
fn calculate_weights_general<F: IntegrateFloat>(
    n: usize,
    formula_type: &NewtonCotesType,
) -> IntegrateResult<Array1<F>> {
    // For higher-order formulas, we need to compute weights using polynomial integration
    // This implementation uses Lagrange polynomial basis

    let mut weights = Array1::zeros(n);

    // Set up integration points in [0, 1] for the Lagrange basis
    let x_pts = match formula_type {
        NewtonCotesType::Closed => {
            // Evenly spaced points in [0, 1] including endpoints
            let mut pts = Array1::zeros(n);
            let step = F::one() / F::from(n - 1).expect("Failed to convert (n-1) to float");

            for i in 0..n {
                pts[i] = F::from(i).expect("Failed to convert i to float") * step;
            }

            pts
        }
        NewtonCotesType::Open => {
            // Evenly spaced points in [0, 1] excluding endpoints
            let mut pts = Array1::zeros(n);
            let step = F::one() / F::from(n + 1).expect("Failed to convert (n+1) to float");

            for i in 0..n {
                pts[i] = F::from(i + 1).expect("Failed to convert (i+1) to float") * step;
            }

            pts
        }
    };

    // Compute weight for each point by integrating its Lagrange basis polynomial
    for i in 0..n {
        // Compute Lagrange basis polynomial for point i and integrate over [0, 1]
        let mut weight = F::zero();

        // Start with binomial expansion of the Lagrange polynomial
        let mut factorial = F::one();
        let mut sign = F::one();

        for j in 0..n {
            if j == i {
                continue;
            }

            // Add integral of this term to weight
            let xi = x_pts[i];
            let xj = x_pts[j];
            let diff = xi - xj;

            if diff.abs() < F::epsilon() {
                return Err(IntegrateError::ValueError(
                    "Cannot compute Newton-Cotes weights: duplicate points".to_string(),
                ));
            }

            // Convert to f64 for calculation, then back to F
            let n_f64 = n as f64;
            let contribution_f64 = sign.to_f64().expect("Failed to convert to f64")
                * (1.0
                    / ((n_f64 + 1.0)
                        * factorial.to_f64().expect("Failed to convert to f64")
                        * diff.to_f64().expect("Failed to convert to f64")));
            let contribution =
                F::from(contribution_f64).expect("Failed to convert contribution to float");
            weight += contribution;

            // Update for next term
            sign = -sign;
            factorial *= F::from(j + 1).expect("Failed to convert (j+1) to float");
        }

        weights[i] = weight;
    }

    // Handle potential numerical issues for higher orders
    if n > 8 {
        // Normalize weights to ensure they sum to 1
        let weight_sum = weights.sum();
        if weight_sum.abs() > F::epsilon() {
            weights = weights.mapv(|w| w / weight_sum);
        } else {
            return Err(IntegrateError::ValueError(
                "Computed Newton-Cotes weights sum to zero, try a lower order".to_string(),
            ));
        }
    }

    Ok(weights)
}

/// Calculates error coefficient for error estimation
#[allow(dead_code)]
fn calculate_error_coefficient<F: IntegrateFloat>(
    n: usize,
    formula_type: &NewtonCotesType,
) -> IntegrateResult<F> {
    // Return error coefficients for common orders
    match formula_type {
        NewtonCotesType::Closed => {
            match n {
                1 => Ok(const_f64::<F>(-1.0 / 2.0)),   // Midpoint rule
                2 => Ok(const_f64::<F>(-1.0 / 12.0)),  // Trapezoidal rule
                3 => Ok(const_f64::<F>(-1.0 / 90.0)),  // Simpson's rule
                4 => Ok(const_f64::<F>(-3.0 / 80.0)),  // Simpson's 3/8 rule
                5 => Ok(const_f64::<F>(-8.0 / 945.0)), // Boole's rule
                6 => Ok(const_f64::<F>(-275.0 / 12096.0)),
                7 => Ok(const_f64::<F>(-9.0 / 1400.0)),
                8 => Ok(const_f64::<F>(-8183.0 / 518400.0)),
                _ => {
                    // For higher orders, use an approximation
                    let degree = if n.is_multiple_of(2) { n } else { n + 1 } - 1;
                    let coeff = F::from(
                        (-1.0_f64).powi((degree + 1) as i32)
                            / ((degree + 2) as f64 * (degree + 3) as f64),
                    )
                    .expect("Test/example failed");
                    Ok(coeff)
                }
            }
        }
        NewtonCotesType::Open => {
            match n {
                3 => Ok(const_f64::<F>(1.0 / 4.0)),
                4 => Ok(const_f64::<F>(-3.0 / 20.0)),
                5 => Ok(const_f64::<F>(13.0 / 42.0)),
                _ => {
                    // For higher orders, use an approximation
                    let degree = n - 1;
                    let coeff = F::from(
                        (-1.0_f64).powi(degree as i32)
                            / ((degree + 1) as f64 * (degree + 2) as f64),
                    )
                    .expect("Test/example failed");
                    Ok(coeff)
                }
            }
        }
    }
}

/// Applies a Newton-Cotes quadrature rule to evaluate an integral.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower integration bound
/// * `b` - Upper integration bound
/// * `n` - Number of points to use
/// * `formula_type` - Type of Newton-Cotes formula (Closed or Open)
///
/// # Returns
///
/// Approximate value of the integral and an error estimate
#[allow(dead_code)]
pub fn newton_cotes_integrate<F, Func>(
    f: Func,
    a: F,
    b: F,
    n: usize,
    formula_type: NewtonCotesType,
) -> IntegrateResult<(F, F)>
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    // Generate the Newton-Cotes rule
    let rule = newton_cotes(n, formula_type, Some(a), Some(b))?;

    // Evaluate function at each point
    let mut sum = F::zero();
    for i in 0..n {
        let x = rule.points[i];
        let w = rule.weights[i];
        sum += f(x) * w;
    }

    // Estimate error based on error coefficient
    // The error is roughly |E| ≈ K * (b-a)^(d+3) * f^(d+2)(ξ) / (d+2)!
    // where d is the degree of exactness and K is the error coefficient
    // Since we don't have access to the (d+2)-th derivative, we use a rough approximation
    let degree = rule.degree;
    let error_coefficient = rule.error_coefficient;
    let error_estimate = error_coefficient * (b - a).powi((degree + 3) as i32);

    Ok((sum, error_estimate.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_newton_cotes_trapezoidal() {
        // Test trapezoidal rule (n=2)
        let result = newton_cotes::<f64>(2, NewtonCotesType::Closed, None, None)
            .expect("Test/example failed");

        assert_eq!(result.weights.len(), 2);
        assert_abs_diff_eq!(result.weights[0], 0.5, epsilon = 1e-14);
        assert_abs_diff_eq!(result.weights[1], 0.5, epsilon = 1e-14);
        assert_eq!(result.degree, 1);
    }

    #[test]
    fn test_newton_cotes_simpson() {
        // Test Simpson's rule (n=3)
        let result = newton_cotes::<f64>(3, NewtonCotesType::Closed, None, None)
            .expect("Test/example failed");

        assert_eq!(result.weights.len(), 3);
        assert_abs_diff_eq!(result.weights[0], 1.0 / 6.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result.weights[1], 4.0 / 6.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result.weights[2], 1.0 / 6.0, epsilon = 1e-14);
        assert_eq!(result.degree, 3);
    }

    #[test]
    fn test_newton_cotes_custom_bounds() {
        // Test with custom bounds
        let result = newton_cotes::<f64>(3, NewtonCotesType::Closed, Some(-1.0), Some(1.0))
            .expect("Test/example failed");

        assert_eq!(result.points.len(), 3);
        assert_abs_diff_eq!(result.points[0], -1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result.points[1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result.points[2], 1.0, epsilon = 1e-14);

        assert_abs_diff_eq!(result.weights[0], 2.0 / 6.0, epsilon = 1e-14); // Scaled to [-1, 1]
        assert_abs_diff_eq!(result.weights[1], 8.0 / 6.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result.weights[2], 2.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_newton_cotes_open() {
        // Test open Newton-Cotes
        let result = newton_cotes::<f64>(3, NewtonCotesType::Open, Some(0.0), Some(1.0))
            .expect("Test/example failed");

        assert_eq!(result.points.len(), 3);
        assert!(result.points[0] > 0.0); // Should not include 0
        assert!(result.points[2] < 1.0); // Should not include 1
    }

    #[test]
    fn test_newton_cotes_integrate() {
        // Test integration of x^2 from 0 to 1 = 1/3
        let (result_, error) =
            newton_cotes_integrate(|x| x * x, 0.0, 1.0, 3, NewtonCotesType::Closed)
                .expect("Test/example failed");
        assert_abs_diff_eq!(result_, 1.0 / 3.0, epsilon = 1e-14);

        // Test integration of sin(x) from 0 to pi = 2
        // Simpson's rule gives 2π/3 ≈ 2.094, which has ~5% error
        let (result_, error) =
            newton_cotes_integrate(|x: f64| x.sin(), 0.0, PI, 3, NewtonCotesType::Closed)
                .expect("Test/example failed");
        assert_abs_diff_eq!(result_, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_invalid_params() {
        // Test invalid number of points for open formula
        let result = newton_cotes::<f64>(2, NewtonCotesType::Open, None, None);
        assert!(result.is_err());

        // Test invalid bounds
        let result = newton_cotes::<f64>(3, NewtonCotesType::Closed, Some(1.0), Some(0.0));
        assert!(result.is_err());
    }
}
