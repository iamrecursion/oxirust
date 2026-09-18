//! Connection formula generator for special functions.
//!
//! Connection formulas express one basis of solutions of a differential
//! equation in terms of another, enabling analytic continuation and
//! uniform evaluation across different parameter regimes.
//!
//! This module provides:
//! - A generic `ConnectionFormula` type with a connection matrix
//! - Factory functions for standard connections (Bessel, hypergeometric,
//!   Legendre, Kummer)
//! - A catalogue of all known formula names for each function family
//!
//! # Mathematical background
//!
//! If `f_A` and `f_B` are two bases of solutions to the same ODE, the
//! connection formula is a matrix `C` (possibly depending on parameters)
//! such that `f_A = C * f_B` (as column vectors).
//!
//! # References
//! - Abramowitz & Stegun, chapters 9, 13, 15
//! - Olver, "Asymptotics and Special Functions"
//! - DLMF, <https://dlmf.nist.gov/>

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Errors arising from connection formula operations.
#[derive(Debug, thiserror::Error)]
pub enum ConnectionError {
    /// A singular or degenerate connection (e.g., sin(nu*pi) = 0).
    #[error("Singular connection: {0}")]
    Singular(String),
    /// Parameter out of valid domain for this formula.
    #[error("Domain mismatch: {0}")]
    DomainMismatch(String),
    /// The coefficient array has wrong length for the connection matrix.
    #[error("Dimension mismatch: matrix is {rows}x{cols}, coefficients have length {given}")]
    DimensionMismatch {
        rows: usize,
        cols: usize,
        given: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionFormula type
// ─────────────────────────────────────────────────────────────────────────────

/// A connection formula: expresses basis A in terms of basis B.
///
/// If `b = (b_1, ..., b_n)^T` are the coefficients in basis B and
/// `a = (a_1, ..., a_n)^T` in basis A, then `a = matrix * b`.
#[derive(Debug, Clone)]
pub struct ConnectionFormula {
    /// Human-readable label for the source basis.
    pub from_basis: String,
    /// Human-readable label for the target basis.
    pub to_basis: String,
    /// Connection matrix stored in row-major order.
    pub matrix: Vec<Vec<Complex64>>,
    /// Parameter interval `(lo, hi)` for which the formula is valid.
    pub valid_range: Option<(f64, f64)>,
}

impl ConnectionFormula {
    /// Apply the connection formula: compute `a = matrix * b`.
    ///
    /// Returns an error when `basis_b_coeffs` has a different length than the
    /// number of columns in `matrix`.
    pub fn apply(&self, basis_b_coeffs: &[Complex64]) -> Result<Vec<Complex64>, ConnectionError> {
        let rows = self.matrix.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        let cols = self.matrix[0].len();
        if basis_b_coeffs.len() != cols {
            return Err(ConnectionError::DimensionMismatch {
                rows,
                cols,
                given: basis_b_coeffs.len(),
            });
        }
        let mut result = vec![Complex64::new(0.0, 0.0); rows];
        for (i, row) in self.matrix.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                result[i] += entry * basis_b_coeffs[j];
            }
        }
        Ok(result)
    }

    /// Check whether a given parameter value falls within the valid range.
    pub fn is_valid_at(&self, param: f64) -> bool {
        match self.valid_range {
            None => true,
            Some((lo, hi)) => param >= lo && param <= hi,
        }
    }

    /// Return the number of rows (output dimension) of the matrix.
    pub fn rows(&self) -> usize {
        self.matrix.len()
    }

    /// Return the number of columns (input dimension) of the matrix.
    pub fn cols(&self) -> usize {
        self.matrix.first().map(|r| r.len()).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a 2×2 complex matrix from four f64 entries
// ─────────────────────────────────────────────────────────────────────────────

fn mat2x2(c00: f64, c01: f64, c10: f64, c11: f64) -> Vec<Vec<Complex64>> {
    vec![
        vec![Complex64::new(c00, 0.0), Complex64::new(c01, 0.0)],
        vec![Complex64::new(c10, 0.0), Complex64::new(c11, 0.0)],
    ]
}

fn mat2x2_c(c00: Complex64, c01: Complex64, c10: Complex64, c11: Complex64) -> Vec<Vec<Complex64>> {
    vec![vec![c00, c01], vec![c10, c11]]
}

// ─────────────────────────────────────────────────────────────────────────────
// Bessel connection formulas
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the connection formula expressing `(J_nu, J_{-nu})` in terms of
/// `(Y_nu, J_nu)` — equivalently, express `Y_nu` as a linear combination of
/// `J_nu` and `J_{-nu}` for non-integer `nu`.
///
/// The standard relation is:
/// `Y_nu(z) = (cos(nu*pi) * J_nu(z) - J_{-nu}(z)) / sin(nu*pi)`
///
/// Inverted, the connection matrix C such that
/// `[J_nu, J_{-nu}]^T = C * [J_nu, Y_nu]^T` is:
/// ```text
/// C = [ 1               0           ]
///     [ cos(nu*pi)  -sin(nu*pi)     ]
/// ```
///
/// Returns an error when `nu` is an integer (formula degenerates).
pub fn bessel_j_to_y_connection(nu: f64) -> Result<ConnectionFormula, ConnectionError> {
    let sin_nu_pi = (nu * PI).sin();
    if sin_nu_pi.abs() < 1e-12 {
        return Err(ConnectionError::Singular(format!(
            "nu = {} is an integer; J-Y Bessel connection is degenerate",
            nu
        )));
    }
    let cos_nu_pi = (nu * PI).cos();
    // Matrix C: [J_nu, J_{-nu}] = C * [J_nu, Y_nu]
    //   C[0,0]=1, C[0,1]=0
    //   C[1,0]=cos(nu*pi), C[1,1]=-sin(nu*pi)
    let matrix = mat2x2(1.0, 0.0, cos_nu_pi, -sin_nu_pi);
    Ok(ConnectionFormula {
        from_basis: format!("(J_{nu}, J_{{-{nu}}})"),
        to_basis: format!("(J_{nu}, Y_{nu})"),
        matrix,
        valid_range: None, // valid for all non-integer nu
    })
}

/// Generate the connection formula expressing the Hankel functions
/// `(H^(1)_nu, H^(2)_nu)` in terms of `(J_nu, Y_nu)`.
///
/// ```text
/// H^(1)_nu = J_nu + i Y_nu
/// H^(2)_nu = J_nu - i Y_nu
/// ```
///
/// Matrix: `[H^(1), H^(2)]^T = C * [J_nu, Y_nu]^T`
/// ```text
/// C = [ 1   i  ]
///     [ 1  -i  ]
/// ```
pub fn bessel_j_to_hankel_connection() -> ConnectionFormula {
    let i = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let matrix = mat2x2_c(one, i, one, -i);
    ConnectionFormula {
        from_basis: "(H^(1)_nu, H^(2)_nu)".to_string(),
        to_basis: "(J_nu, Y_nu)".to_string(),
        matrix,
        valid_range: None,
    }
}

/// Generate the connection formula expressing `(I_nu, I_{-nu})` in terms
/// of `(J_nu, J_{-nu})` via the imaginary-argument substitution.
///
/// `I_nu(z) = e^{-i nu pi/2} J_nu(i z)`  →  on real axis: `I_nu(x) ~ J_nu(ix)`.
///
/// The connection matrix C such that
/// `[I_nu, I_{-nu}]^T = C * [J_nu, J_{-nu}]^T` (entry-wise phase factors):
/// ```text
/// C = [ e^{-i nu pi/2}    0              ]
///     [ 0                  e^{i nu pi/2} ]
/// ```
pub fn bessel_j_to_modified_connection(nu: f64) -> ConnectionFormula {
    let phase_neg = Complex64::new(0.0, -nu * PI / 2.0).exp();
    let phase_pos = Complex64::new(0.0, nu * PI / 2.0).exp();
    let zero = Complex64::new(0.0, 0.0);
    let matrix = mat2x2_c(phase_neg, zero, zero, phase_pos);
    ConnectionFormula {
        from_basis: "(I_nu, I_{-nu})".to_string(),
        to_basis: "(J_nu, J_{-nu})".to_string(),
        matrix,
        valid_range: None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypergeometric connection formulas (Gauss, z=0 -> z=1)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Gauss connection formula for `_2F_1(a,b;c;z)` around `z=1`.
///
/// Near `z=0`, the two linearly independent solutions are:
///   - `f_1 = 2F1(a,b;c;z)`
///   - `f_2 = z^{1-c} * 2F1(a-c+1, b-c+1; 2-c; z)`
///
/// Near `z=1`, using the substitution `1-z = w`, the two solutions are:
///   - `g_1 = 2F1(a,b;a+b-c+1; 1-z)`
///   - `g_2 = (1-z)^{c-a-b} * 2F1(c-a, c-b; c-a-b+1; 1-z)`
///
/// The Gauss connection formula is:
/// ```text
/// f_1 = [Gamma(c) Gamma(c-a-b)] / [Gamma(c-a) Gamma(c-b)] * g_1
///       + [Gamma(c) Gamma(a+b-c)] / [Gamma(a) Gamma(b)] * g_2
/// ```
///
/// Returns an error when `c - a - b` or `a + b - c` is a non-positive integer
/// (degenerate Gamma poles).
pub fn hypergeometric_z0_to_z1_connection(
    a: f64,
    b: f64,
    c: f64,
) -> Result<ConnectionFormula, ConnectionError> {
    // Check non-degeneracy: c, c-a, c-b, a+b-c must not be non-positive integers.
    for (name, val) in [
        ("c", c),
        ("c-a", c - a),
        ("c-b", c - b),
        ("a+b-c", a + b - c),
    ] {
        if val <= 0.0 && (val - val.round()).abs() < 1e-10 {
            return Err(ConnectionError::Singular(format!(
                "{} = {} is a non-positive integer; Gamma function has a pole",
                name, val
            )));
        }
    }

    let gamma_c = gamma_f64(c);
    let gamma_c_minus_a_minus_b = gamma_f64(c - a - b);
    let gamma_c_minus_a = gamma_f64(c - a);
    let gamma_c_minus_b = gamma_f64(c - b);
    let gamma_a_plus_b_minus_c = gamma_f64(a + b - c);
    let gamma_a = gamma_f64(a);
    let gamma_b = gamma_f64(b);

    // Connection coefficients.
    let c11 = (gamma_c * gamma_c_minus_a_minus_b) / (gamma_c_minus_a * gamma_c_minus_b);
    let c12 = (gamma_c * gamma_a_plus_b_minus_c) / (gamma_a * gamma_b);

    // f_1 = c11 * g_1 + c12 * g_2
    // f_2 is the second solution; we also need the connection for it.
    // Using Kummer's 24 relations, the second row is:
    // Gamma(2-c)/Gamma(1-a) / Gamma(1-b) * g1 + Gamma(2-c)/Gamma(a-c+1)/Gamma(b-c+1) * g2
    let gamma_2_minus_c = gamma_f64(2.0 - c);
    let gamma_1_minus_a = gamma_f64(1.0 - a);
    let gamma_1_minus_b = gamma_f64(1.0 - b);
    let gamma_a_minus_c_plus_1 = gamma_f64(a - c + 1.0);
    let gamma_b_minus_c_plus_1 = gamma_f64(b - c + 1.0);
    let c21 = gamma_2_minus_c / (gamma_1_minus_a * gamma_1_minus_b);
    let c22 = gamma_2_minus_c / (gamma_a_minus_c_plus_1 * gamma_b_minus_c_plus_1);

    let matrix = mat2x2(c11, c12, c21, c22);

    Ok(ConnectionFormula {
        from_basis: format!("2F1(a={a},b={b};c={c};z) near z=0"),
        to_basis: format!("2F1 near z=1 (Gauss)"),
        matrix,
        valid_range: Some((-1.0, 1.0)), // valid for |z| < 1 and |1-z| < 1
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Legendre P/Q connection
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the connection formula relating `Q_n(x)` to `P_n(x)`.
///
/// The Legendre function of the second kind satisfies:
/// `Q_n(x) = P_n(x) * Q_0(x) - W_{n-1}(x)`
/// where `W_{n-1}` is a polynomial of degree `n-1`.
///
/// For the purpose of a 2×2 connection matrix at the leading polynomial
/// level, the standard connection between `P_n` (first kind) and `Q_n`
/// (second kind) is encoded as the identity
/// `[P_n, Q_n]^T = C * [P_n, Q_n]^T` with `C = I_2` (trivial: they are
/// already a basis).
///
/// More usefully, the *Wronskian connection* gives
/// `W[P_n, Q_n](x) = 1/(1 - x^2)` for `|x| < 1`.
/// The corresponding "change of basis to Wronskian normalisation" matrix is
/// encoded as:
/// ```text
/// C = [ 1   0 ]
///     [ 0   1 ]   with det(C) = 1  and off-diagonal = Wronskian factor
/// ```
///
/// For practical use in connection formula applications, the formula returned
/// here represents the expansion of the *Wronskian identity*:
/// `[P_n * dQ_n/dx - Q_n * dP_n/dx] = -1/(x^2-1)`.
pub fn legendre_pq_connection(n: u32) -> ConnectionFormula {
    // The Wronskian W[P_n, Q_n] = -1/(x^2-1).
    // Encode the two-element basis {P_n, Q_n} and the trivial identity
    // connection (P_n, Q_n) -> (P_n, Q_n), with the Wronskian noted in
    // the metadata.
    // For n=0: P_0 = 1, Q_0 = (1/2) ln((1+x)/(1-x)).
    // For n=1: P_1 = x, Q_1 = (x/2) ln((1+x)/(1-x)) - 1.
    // The connection matrix from {P_n, Q_n} to itself is I_2; the actual
    // content is the identity plus the Wronskian normalisation.
    let _ = n; // n affects only the specific polynomial, not the 2x2 structure.
    let matrix = mat2x2(1.0, 0.0, 0.0, 1.0); // identity: trivial connection
    ConnectionFormula {
        from_basis: format!("(P_{n}, Q_{n}) [Legendre basis]"),
        to_basis: format!("(P_{n}, Q_{n}) [Wronskian normalised]"),
        matrix,
        valid_range: Some((-1.0, 1.0)), // valid for |x| < 1 (cut plane)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kummer transformation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Kummer transformation connection formula.
///
/// The Kummer transformation for the confluent hypergeometric function
/// (Kummer's function M, also written `_1F_1`):
///
/// `M(a; b; z) = e^z * M(b-a; b; -z)`
///
/// This is a scalar identity, encoded as a 1×1 connection matrix.
/// The "matrix" is `[e^z]` but since we cannot fix `z` at formula-generation
/// time, we record the rule as a symbolic description with a unit matrix
/// (the exponential factor is to be applied at evaluation time).
///
/// For a 2×2 version relating the two standard Kummer solutions
/// `M(a;b;z)` and `U(a;b;z)`, the Kummer identity gives:
/// ```text
/// M(a;b;z) = (Gamma(b)/Gamma(b-a)) U(a;b;z) + (Gamma(b)/Gamma(a)) z^{1-b} U(b-a; 2-b; z)
/// ```
/// (valid for Re(b) > 0, b not a non-positive integer).
pub fn kummer_connection(a: f64, b: f64) -> Result<ConnectionFormula, ConnectionError> {
    // Validate that Gamma(b), Gamma(a), Gamma(b-a) are finite.
    for (name, val) in [("b", b), ("a", a), ("b-a", b - a)] {
        if val <= 0.0 && (val - val.round()).abs() < 1e-10 {
            return Err(ConnectionError::Singular(format!(
                "{} = {} is a non-positive integer; Kummer connection is degenerate",
                name, val
            )));
        }
    }
    let gamma_b = gamma_f64(b);
    let gamma_b_minus_a = gamma_f64(b - a);
    let gamma_a = gamma_f64(a);
    // Connection: [M(a;b;z), z^{1-b}*M(b-a;2-b;z)] = C * [U(a;b;z), z^{1-b}*U(b-a;2-b;z)]
    let c00 = gamma_b / gamma_b_minus_a;
    let c01 = gamma_b / gamma_a;
    // The second row uses the standard connection for U:
    // U(a;b;z) = (pi/sin(pi*b)) * [M(a;b;z)/Gamma(b-a+1)Gamma(a) - z^{1-b}M(b-a;2-b;z)/Gamma(b)]
    // Simplified 2nd row using the Wronskian W[M, U] = -Gamma(b)/Gamma(a) * z^{-b} * e^z:
    let sin_pi_b = (PI * b).sin();
    if sin_pi_b.abs() < 1e-12 {
        return Err(ConnectionError::Singular(format!(
            "sin(pi*b) = 0 for b = {}; Kummer connection is degenerate",
            b
        )));
    }
    let c10 = PI / (sin_pi_b * gamma_b_minus_a * gamma_a);
    let c11 = -PI / (sin_pi_b * gamma_b);
    let matrix = mat2x2(c00, c01, c10, c11);
    Ok(ConnectionFormula {
        from_basis: format!("(M({a};{b};z), z^{{1-{b}}}*M({}-{a};{};z))", b, 2.0 - b),
        to_basis: format!("(U({a};{b};z), z^{{1-{b}}}*U({}-{a};{};z))", b, 2.0 - b),
        matrix,
        valid_range: Some((f64::NEG_INFINITY, f64::INFINITY)),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Catalogue
// ─────────────────────────────────────────────────────────────────────────────

/// List all known connection formulas for a named function family.
///
/// Returns a `Vec<String>` of human-readable formula names.
pub fn list_connections(function_family: &str) -> Vec<String> {
    match function_family {
        "bessel" => vec![
            "J_nu -> Y_nu (standard, non-integer nu)".to_string(),
            "J_nu -> H_nu^(1) and H_nu^(2) (Hankel first and second kind)".to_string(),
            "J_nu -> I_nu (modified Bessel, imaginary argument)".to_string(),
        ],
        "legendre" => vec![
            "P_n -> Q_n (second kind, Wronskian identity)".to_string(),
            "P_n^m -> P_n^{-m} (associated Legendre, sign convention)".to_string(),
        ],
        "hypergeometric" => vec![
            "2F1(a,b;c;z) -> 2F1(a,b;a+b-c+1;1-z) (Gauss z=0 to z=1)".to_string(),
            "2F1(z) -> z^{-a} 2F1(a, a-c+1; a-b+1; 1/z) (Pfaff-Euler)".to_string(),
            "2F1(z) -> (1-z)^{-a} 2F1(a, c-b; c; z/(z-1)) (Kummer transformation)".to_string(),
        ],
        "kummer" => vec![
            "M(a;b;z) = e^z M(b-a;b;-z) (Kummer symmetry)".to_string(),
            "M(a;b;z) -> (U(a;b;z), z^{1-b}*U(b-a;2-b;z)) (Tricomi U)".to_string(),
        ],
        "airy" => vec![
            "Ai(z) -> Bi(z) (second solution)".to_string(),
            "Ai(z) -> Ai(z e^{2pi i/3}) (asymptotic sector rotation)".to_string(),
        ],
        "parabolic" => vec![
            "D_nu(z) -> D_{-nu-1}(iz) (rotation by pi/2)".to_string(),
            "D_nu(z) -> D_nu(-z) (reflection symmetry)".to_string(),
        ],
        _ => vec![],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma function helper (forward to scirs2-special gamma module)
// ─────────────────────────────────────────────────────────────────────────────

/// Gamma function evaluated at a real argument via the crate's gamma module.
fn gamma_f64(x: f64) -> f64 {
    crate::gamma::gamma(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_connections_bessel() {
        let connections = list_connections("bessel");
        assert_eq!(connections.len(), 3, "expected 3 Bessel connections");
        assert!(connections[0].contains("J_nu -> Y_nu"));
    }

    #[test]
    fn test_list_connections_hypergeometric() {
        let connections = list_connections("hypergeometric");
        assert_eq!(
            connections.len(),
            3,
            "expected 3 hypergeometric connections"
        );
    }

    #[test]
    fn test_list_connections_kummer() {
        let connections = list_connections("kummer");
        assert!(!connections.is_empty());
        assert!(connections.iter().any(|s| s.contains("Kummer")));
    }

    #[test]
    fn test_list_connections_unknown() {
        let connections = list_connections("nonexistent_family");
        assert!(connections.is_empty());
    }

    #[test]
    fn test_bessel_j_to_y_connection_identity() {
        // For nu = 0.5, construct the connection matrix and verify its
        // determinant equals -sin(nu*pi) (from the 2x2 structure).
        let nu = 0.5_f64;
        let cf = bessel_j_to_y_connection(nu).expect("non-integer nu should succeed");
        assert_eq!(cf.matrix.len(), 2);
        // C[0,0] = 1, C[0,1] = 0.
        assert!((cf.matrix[0][0].re - 1.0).abs() < 1e-12);
        assert!(cf.matrix[0][1].re.abs() < 1e-12);
        // C[1,0] = cos(0.5 pi) = 0, C[1,1] = -sin(0.5 pi) = -1.
        let cos_val = (nu * PI).cos();
        let sin_val = (nu * PI).sin();
        assert!((cf.matrix[1][0].re - cos_val).abs() < 1e-12);
        assert!((cf.matrix[1][1].re + sin_val).abs() < 1e-12);
    }

    #[test]
    fn test_bessel_j_to_y_integer_nu_error() {
        // Integer nu should return a Singular error.
        assert!(bessel_j_to_y_connection(2.0).is_err());
        assert!(bessel_j_to_y_connection(0.0).is_err());
    }

    #[test]
    fn test_bessel_j_to_hankel_connection() {
        let cf = bessel_j_to_hankel_connection();
        // H^(1) = J + iY: row 0 = [1, i]
        assert!((cf.matrix[0][0].re - 1.0).abs() < 1e-12);
        assert!((cf.matrix[0][1].im - 1.0).abs() < 1e-12);
        // H^(2) = J - iY: row 1 = [1, -i]
        assert!((cf.matrix[1][0].re - 1.0).abs() < 1e-12);
        assert!((cf.matrix[1][1].im + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hypergeometric_connection() {
        // a=0.5, b=0.5, c=1.5: Gauss connection should succeed.
        // a+b-c = 0.5+0.5-1.5 = -0.5, c-a = 1.0, c-b = 1.0, c-a-b = 0.5 — all nonzero.
        let cf = hypergeometric_z0_to_z1_connection(0.5, 0.5, 1.5).expect("regular parameters");
        assert_eq!(cf.matrix.len(), 2);
        // Check that the matrix entries are finite.
        for row in &cf.matrix {
            for entry in row {
                assert!(entry.re.is_finite(), "matrix entry should be finite");
            }
        }
    }

    #[test]
    fn test_hypergeometric_connection_degenerate() {
        // c = 0 is a non-positive integer -> should fail.
        let err = hypergeometric_z0_to_z1_connection(1.0, 1.0, 0.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_legendre_pq_connection() {
        let cf = legendre_pq_connection(3);
        assert_eq!(cf.matrix.len(), 2);
        // Identity matrix.
        assert!((cf.matrix[0][0].re - 1.0).abs() < 1e-12);
        assert!(cf.matrix[0][1].re.abs() < 1e-12);
        assert!(cf.matrix[1][0].re.abs() < 1e-12);
        assert!((cf.matrix[1][1].re - 1.0).abs() < 1e-12);
        // Valid range should be (-1, 1).
        assert!(cf.is_valid_at(0.0));
        assert!(!cf.is_valid_at(2.0));
    }

    #[test]
    fn test_kummer_connection() {
        // a=1.5, b=2.5: standard Kummer connection (non-integer b avoids sin(pi*b)=0).
        let cf = kummer_connection(1.5, 2.5).expect("valid a,b");
        assert_eq!(cf.matrix.len(), 2);
        for row in &cf.matrix {
            for entry in row {
                assert!(entry.re.is_finite(), "Kummer matrix entry should be finite");
            }
        }
    }

    #[test]
    fn test_kummer_connection_degenerate() {
        // b = -1 is a non-positive integer -> should fail.
        let err = kummer_connection(1.0, -1.0);
        assert!(err.is_err());
    }

    #[test]
    fn test_apply_identity_matrix() {
        let cf = legendre_pq_connection(2);
        let coeffs = vec![Complex64::new(3.0, 0.0), Complex64::new(-2.0, 0.0)];
        let result = cf.apply(&coeffs).expect("application should succeed");
        assert_eq!(result.len(), 2);
        // Identity matrix: result should equal coeffs.
        assert!((result[0].re - 3.0).abs() < 1e-12);
        assert!((result[1].re + 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_dimension_mismatch() {
        let cf = legendre_pq_connection(2); // 2x2 matrix
        let short_coeffs = vec![Complex64::new(1.0, 0.0)]; // only 1 element
        assert!(cf.apply(&short_coeffs).is_err());
    }

    #[test]
    fn test_bessel_modified_connection_phases() {
        let nu = 1.0_f64;
        let cf = bessel_j_to_modified_connection(nu);
        // C[0,0] = e^{-i nu pi/2} = e^{-i pi/2} = -i
        let expected_phase_neg = Complex64::new(0.0, -nu * PI / 2.0).exp();
        let expected_phase_pos = Complex64::new(0.0, nu * PI / 2.0).exp();
        assert!((cf.matrix[0][0] - expected_phase_neg).norm() < 1e-12);
        assert!((cf.matrix[1][1] - expected_phase_pos).norm() < 1e-12);
        assert!(cf.matrix[0][1].norm() < 1e-12);
        assert!(cf.matrix[1][0].norm() < 1e-12);
    }
}
