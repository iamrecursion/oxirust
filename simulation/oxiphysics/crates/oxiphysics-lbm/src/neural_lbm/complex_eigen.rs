// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Dense complex linear algebra for the reduced-order models of
//! [`neural_lbm`](super).
//!
//! The dynamic mode decomposition needs the eigendecomposition of a small
//! *non-symmetric* real matrix, whose eigenvalues and eigenvectors are in general
//! complex, and a complex least-squares solve for the mode amplitudes.  Both are
//! provided here without external dependencies:
//!
//! * [`eigen_decomposition`] reduces the matrix to upper Hessenberg form with
//!   Householder reflections, runs the single-shift QR iteration (Wilkinson
//!   shifts, exceptional shifts against stagnation, deflation) to the complex
//!   Schur form `A = Z T Z^H`, and obtains the eigenvectors of the triangular
//!   factor `T` by back-substitution.  Cyclic permutation matrices (pure
//!   translations on a periodic grid) and defective matrices are handled; the
//!   iteration reports failure through [`Error::ConvergenceFailure`] instead of
//!   looping forever.
//! * [`least_squares`] solves the (possibly rank-deficient) complex least-squares
//!   problem by modified Gram-Schmidt with re-orthogonalisation, dropping columns
//!   that are numerically dependent on the preceding ones.

use crate::error::{Error, Result};
use oxiphysics_core::complex::Complex;
use std::ops::{Index, IndexMut};

/// QR sweeps granted to a single eigenvalue before the iteration gives up.
const MAX_SWEEPS_PER_EIGENVALUE: usize = 120;

/// Dense square complex matrix stored row-major.
#[derive(Debug, Clone)]
pub(super) struct ComplexMatrix {
    n: usize,
    data: Vec<Complex>,
}

impl ComplexMatrix {
    /// The `n × n` zero matrix.
    pub(super) fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![Complex::zero(); n * n],
        }
    }

    /// The `n × n` identity matrix.
    pub(super) fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m[(i, i)] = Complex::one();
        }
        m
    }

    /// Builds a matrix from a real row-major array of `n * n` entries.
    pub(super) fn from_real_row_major(n: usize, entries: &[f64]) -> Result<Self> {
        if entries.len() != n * n {
            return Err(Error::ComponentCountMismatch {
                expected: n * n,
                found: entries.len(),
                field: "complex matrix entries".to_string(),
            });
        }
        let data = entries.iter().map(|&x| Complex::new(x, 0.0)).collect();
        Ok(Self { n, data })
    }

    /// Dimension `n` of the `n × n` matrix.
    pub(super) fn dim(&self) -> usize {
        self.n
    }

    /// Largest `|re| + |im|` over all entries.
    fn max_abs1(&self) -> f64 {
        self.data.iter().map(|&z| abs1(z)).fold(0.0, f64::max)
    }
}

impl Index<(usize, usize)> for ComplexMatrix {
    type Output = Complex;

    fn index(&self, (row, col): (usize, usize)) -> &Complex {
        &self.data[row * self.n + col]
    }
}

impl IndexMut<(usize, usize)> for ComplexMatrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Complex {
        &mut self.data[row * self.n + col]
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/// Modulus without intermediate overflow.
fn modulus(z: Complex) -> f64 {
    z.re.hypot(z.im)
}

/// `|re| + |im|`, the cheap norm used in deflation tests.
fn abs1(z: Complex) -> f64 {
    z.re.abs() + z.im.abs()
}

/// Multiplication by a real factor.
fn scale(z: Complex, factor: f64) -> Complex {
    Complex::new(z.re * factor, z.im * factor)
}

/// Complex division by Smith's algorithm (no spurious overflow).
fn divide(a: Complex, b: Complex) -> Complex {
    if b.re.abs() >= b.im.abs() {
        let ratio = b.im / b.re;
        let denominator = b.re + b.im * ratio;
        Complex::new(
            (a.re + a.im * ratio) / denominator,
            (a.im - a.re * ratio) / denominator,
        )
    } else {
        let ratio = b.re / b.im;
        let denominator = b.re * ratio + b.im;
        Complex::new(
            (a.re * ratio + a.im) / denominator,
            (a.im * ratio - a.re) / denominator,
        )
    }
}

/// Principal square root.
fn sqrt(z: Complex) -> Complex {
    let m = modulus(z);
    if m == 0.0 {
        return Complex::zero();
    }
    let t = ((m + z.re.abs()) * 0.5).sqrt();
    if z.re >= 0.0 {
        Complex::new(t, z.im / (2.0 * t))
    } else {
        Complex::new(z.im.abs() / (2.0 * t), t.copysign(z.im))
    }
}

/// Euclidean norm of a complex vector.
fn vector_norm(v: &[Complex]) -> f64 {
    let largest = v.iter().map(|&z| modulus(z)).fold(0.0, f64::max);
    if largest == 0.0 || !largest.is_finite() {
        return largest;
    }
    let sum: f64 = v
        .iter()
        .map(|&z| {
            let t = modulus(z) / largest;
            t * t
        })
        .sum();
    largest * sum.sqrt()
}

/// Hermitian inner product `sum conj(a_i) b_i`.
fn inner(a: &[Complex], b: &[Complex]) -> Complex {
    a.iter()
        .zip(b.iter())
        .fold(Complex::zero(), |acc, (&x, &y)| acc + x.conj() * y)
}

// ---------------------------------------------------------------------------
// Hessenberg reduction
// ---------------------------------------------------------------------------

/// Reduces `h` to upper Hessenberg form by Householder similarity transforms
/// `h <- P h P`, accumulating them into `z <- z P`.
fn reduce_to_hessenberg(h: &mut ComplexMatrix, z: &mut ComplexMatrix) {
    let n = h.n;
    if n < 3 {
        return;
    }
    for k in 0..n - 2 {
        let column: Vec<Complex> = (k + 1..n).map(|i| h[(i, k)]).collect();
        let norm = vector_norm(&column);
        if norm == 0.0 {
            continue;
        }
        // v = x + e^{i arg(x_0)} |x| e_1 maps x to -e^{i arg(x_0)} |x| e_1.
        let lead = column[0];
        let lead_modulus = modulus(lead);
        let phase = if lead_modulus == 0.0 {
            Complex::one()
        } else {
            scale(lead, 1.0 / lead_modulus)
        };
        let mut v = column;
        v[0] = v[0] + scale(phase, norm);
        let v_norm_sq: f64 = v.iter().map(|c| c.norm_sq()).sum();
        if v_norm_sq == 0.0 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // h <- P h on the columns that still carry entries below the subdiagonal.
        for j in k..n {
            let mut s = Complex::zero();
            for (offset, &vi) in v.iter().enumerate() {
                s = s + vi.conj() * h[(k + 1 + offset, j)];
            }
            let s = scale(s, beta);
            for (offset, &vi) in v.iter().enumerate() {
                h[(k + 1 + offset, j)] = h[(k + 1 + offset, j)] - vi * s;
            }
        }
        // h <- h P and z <- z P on all rows.
        for matrix in [&mut *h, &mut *z] {
            for i in 0..n {
                let mut s = Complex::zero();
                for (offset, &vi) in v.iter().enumerate() {
                    s = s + matrix[(i, k + 1 + offset)] * vi;
                }
                let s = scale(s, beta);
                for (offset, &vi) in v.iter().enumerate() {
                    matrix[(i, k + 1 + offset)] = matrix[(i, k + 1 + offset)] - s * vi.conj();
                }
            }
        }
        for i in k + 2..n {
            h[(i, k)] = Complex::zero();
        }
    }
}

// ---------------------------------------------------------------------------
// Shifted QR iteration
// ---------------------------------------------------------------------------

/// Eigenvalue of the trailing 2x2 block `h[hi-1..=hi]` closest to `h[hi][hi]`.
fn wilkinson_shift(h: &ComplexMatrix, hi: usize) -> Complex {
    let a = h[(hi - 1, hi - 1)];
    let b = h[(hi - 1, hi)];
    let c = h[(hi, hi - 1)];
    let d = h[(hi, hi)];
    let p = scale(a - d, 0.5);
    let disc = sqrt(p * p + b * c);
    // The two shifts are d + p +- disc; the one closer to d is d - bc/q with q
    // the root of larger modulus among p +- disc (avoids cancellation).
    let q = if (p.conj() * disc).re >= 0.0 {
        p + disc
    } else {
        p - disc
    };
    if modulus(q) == 0.0 {
        d
    } else {
        d - divide(b * c, q)
    }
}

/// One explicitly shifted QR sweep on the unreduced block `l..=hi` of the
/// Hessenberg matrix `h`, applied as a similarity transform to the whole matrix
/// and accumulated into the Schur vectors `z`.
fn qr_sweep(h: &mut ComplexMatrix, z: &mut ComplexMatrix, l: usize, hi: usize, shift: Complex) {
    let n = h.n;
    for i in l..=hi {
        h[(i, i)] = h[(i, i)] - shift;
    }
    // Triangularise with Givens rotations G_k = [[conj(c), conj(s)], [-s, c]].
    let mut rotations: Vec<(Complex, Complex)> = Vec::with_capacity(hi - l);
    for k in l..hi {
        let a = h[(k, k)];
        let b = h[(k + 1, k)];
        let r = a.re.hypot(a.im).hypot(b.re.hypot(b.im));
        let (c, s) = if r == 0.0 {
            (Complex::one(), Complex::zero())
        } else {
            (scale(a, 1.0 / r), scale(b, 1.0 / r))
        };
        rotations.push((c, s));
        for j in k..n {
            let x = h[(k, j)];
            let y = h[(k + 1, j)];
            h[(k, j)] = c.conj() * x + s.conj() * y;
            h[(k + 1, j)] = y * c - x * s;
        }
    }
    // Multiply by the adjoint rotations from the right: columns (k, k+1) of the
    // rows above and on the subdiagonal, and of the Schur vectors.
    for (offset, &(c, s)) in rotations.iter().enumerate() {
        let k = l + offset;
        for i in 0..=k + 1 {
            let x = h[(i, k)];
            let y = h[(i, k + 1)];
            h[(i, k)] = x * c + y * s;
            h[(i, k + 1)] = y * c.conj() - x * s.conj();
        }
        for i in 0..n {
            let x = z[(i, k)];
            let y = z[(i, k + 1)];
            z[(i, k)] = x * c + y * s;
            z[(i, k + 1)] = y * c.conj() - x * s.conj();
        }
    }
    for i in l..=hi {
        h[(i, i)] = h[(i, i)] + shift;
    }
}

/// Iterates the Hessenberg matrix `h` to upper triangular (complex Schur) form,
/// accumulating the unitary transformations into `z`.
fn schur_iteration(h: &mut ComplexMatrix, z: &mut ComplexMatrix) -> Result<()> {
    let n = h.n;
    if n < 2 {
        return Ok(());
    }
    let matrix_scale = h.max_abs1();
    let mut hi = n - 1;
    let mut sweeps = 0_usize;
    while hi > 0 {
        // Find the start `l` of the unreduced block that ends at `hi`.
        let mut l = hi;
        while l > 0 {
            let sub = abs1(h[(l, l - 1)]);
            let mut local = abs1(h[(l - 1, l - 1)]) + abs1(h[(l, l)]);
            if local == 0.0 {
                local = matrix_scale;
            }
            if sub <= f64::EPSILON * local {
                h[(l, l - 1)] = Complex::zero();
                break;
            }
            l -= 1;
        }
        if l == hi {
            hi -= 1;
            sweeps = 0;
            continue;
        }
        sweeps += 1;
        if sweeps > MAX_SWEEPS_PER_EIGENVALUE {
            return Err(Error::ConvergenceFailure {
                residual: abs1(h[(hi, hi - 1)]),
                iterations: sweeps,
                tolerance: f64::EPSILON * matrix_scale,
            });
        }
        let shift = if sweeps.is_multiple_of(10) {
            // Exceptional shifts break the symmetry of matrices (cyclic
            // permutations, for instance) on which the Wilkinson shift stalls.
            if (sweeps / 10) % 2 == 1 {
                h[(l + 1, l + 1)] + Complex::new(0.75 * modulus(h[(l + 1, l)]), 0.0)
            } else {
                h[(hi, hi)] + Complex::new(0.75 * modulus(h[(hi, hi - 1)]), 0.0)
            }
        } else {
            wilkinson_shift(h, hi)
        };
        qr_sweep(h, z, l, hi, shift);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Eigenvectors of the triangular Schur factor
// ---------------------------------------------------------------------------

/// Eigenvectors of `A = Z T Z^H` from the upper triangular `T`, as columns
/// `vectors[k]`, normalised to unit length with their largest component real.
fn schur_eigenvectors(t: &ComplexMatrix, z: &ComplexMatrix) -> Vec<Vec<Complex>> {
    let n = t.n;
    let floor = (f64::EPSILON * t.max_abs1()).max(f64::MIN_POSITIVE);
    let mut vectors = Vec::with_capacity(n);
    for k in 0..n {
        let lambda = t[(k, k)];
        // Back-substitution of (T - lambda I) x = 0 with x_k = 1, x_j = 0 for j > k.
        let mut x = vec![Complex::zero(); n];
        x[k] = Complex::one();
        for i in (0..k).rev() {
            let mut sum = Complex::zero();
            for (j, &xj) in x.iter().enumerate().take(k + 1).skip(i + 1) {
                sum = sum + t[(i, j)] * xj;
            }
            let mut denominator = t[(i, i)] - lambda;
            if abs1(denominator) < floor {
                // (Nearly) repeated eigenvalue: perturb the pivot as LAPACK does.
                denominator = Complex::new(floor, 0.0);
            }
            x[i] = -divide(sum, denominator);
            let size = modulus(x[i]);
            if size > 1.0e100 {
                for entry in x.iter_mut().take(k + 1) {
                    *entry = scale(*entry, 1.0e-100);
                }
            }
        }
        // v = Z x
        let mut v: Vec<Complex> = (0..n)
            .map(|row| (0..=k).fold(Complex::zero(), |acc, col| acc + z[(row, col)] * x[col]))
            .collect();
        let norm = vector_norm(&v);
        if norm > 0.0 && norm.is_finite() {
            for entry in v.iter_mut() {
                *entry = scale(*entry, 1.0 / norm);
            }
        }
        // Phase convention: the component of largest modulus is real and positive.
        let mut pivot = Complex::zero();
        for &entry in &v {
            if modulus(entry) > modulus(pivot) {
                pivot = entry;
            }
        }
        let size = modulus(pivot);
        if size > 0.0 {
            let rotation = scale(pivot.conj(), 1.0 / size);
            for entry in v.iter_mut() {
                *entry = *entry * rotation;
            }
        }
        vectors.push(v);
    }
    vectors
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Eigendecomposition `A v_k = lambda_k v_k` of a general square matrix.
#[derive(Debug, Clone)]
pub(super) struct EigenDecomposition {
    /// Eigenvalues ordered by decreasing modulus (ties: decreasing real part,
    /// then decreasing imaginary part).
    pub(super) values: Vec<Complex>,
    /// Unit-norm eigenvectors; `vectors[k]` belongs to `values[k]`.
    pub(super) vectors: Vec<Vec<Complex>>,
}

/// Computes all eigenvalues and eigenvectors of the square matrix `a`.
///
/// # Errors
/// [`Error::NonFiniteField`] if `a` contains NaN or infinity and
/// [`Error::ConvergenceFailure`] if the QR iteration does not converge.
pub(super) fn eigen_decomposition(a: &ComplexMatrix) -> Result<EigenDecomposition> {
    let n = a.dim();
    for (index, entry) in a.data.iter().enumerate() {
        for value in [entry.re, entry.im] {
            if !value.is_finite() {
                return Err(Error::NonFiniteField {
                    field: "matrix entry".to_string(),
                    index,
                    value,
                });
            }
        }
    }
    let mut t = a.clone();
    let mut z = ComplexMatrix::identity(n);
    reduce_to_hessenberg(&mut t, &mut z);
    schur_iteration(&mut t, &mut z)?;
    let eigenvectors = schur_eigenvectors(&t, &z);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&p, &q| {
        let (vp, vq) = (t[(p, p)], t[(q, q)]);
        modulus(vq)
            .total_cmp(&modulus(vp))
            .then(vq.re.total_cmp(&vp.re))
            .then(vq.im.total_cmp(&vp.im))
    });
    let values = order.iter().map(|&k| t[(k, k)]).collect();
    let vectors = order.iter().map(|&k| eigenvectors[k].clone()).collect();
    Ok(EigenDecomposition { values, vectors })
}

/// Least-squares solution `b` of `sum_j b_j columns[j] ≈ rhs` for complex data.
///
/// Columns are orthogonalised in the given order (modified Gram-Schmidt with one
/// re-orthogonalisation pass).  A column whose component orthogonal to the
/// preceding ones is below `relative_tolerance` times its own norm is treated as
/// linearly dependent and receives the coefficient zero, which yields the basic
/// least-squares solution of a rank-deficient problem.  Zero columns also get
/// the coefficient zero.
pub(super) fn least_squares(
    columns: &[Vec<Complex>],
    rhs: &[Complex],
    relative_tolerance: f64,
) -> Vec<Complex> {
    let mut basis: Vec<Vec<Complex>> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    // r_columns[p][i] = <q_i, column kept[p]> for i <= p (upper triangular R).
    let mut r_columns: Vec<Vec<Complex>> = Vec::new();
    for (index, column) in columns.iter().enumerate() {
        let original = vector_norm(column);
        if original == 0.0 || !original.is_finite() {
            continue;
        }
        let mut v = column.clone();
        let mut coefficients = vec![Complex::zero(); basis.len()];
        for _pass in 0..2 {
            for (slot, q) in basis.iter().enumerate() {
                let c = inner(q, &v);
                coefficients[slot] = coefficients[slot] + c;
                for (vi, &qi) in v.iter_mut().zip(q.iter()) {
                    *vi = *vi - c * qi;
                }
            }
        }
        let remainder = vector_norm(&v);
        if remainder <= relative_tolerance * original {
            continue;
        }
        for entry in v.iter_mut() {
            *entry = scale(*entry, 1.0 / remainder);
        }
        coefficients.push(Complex::new(remainder, 0.0));
        basis.push(v);
        kept.push(index);
        r_columns.push(coefficients);
    }
    let projections: Vec<Complex> = basis.iter().map(|q| inner(q, rhs)).collect();
    let m = basis.len();
    let mut solution = vec![Complex::zero(); m];
    for p in (0..m).rev() {
        let mut acc = projections[p];
        for p2 in p + 1..m {
            acc = acc - r_columns[p2][p] * solution[p2];
        }
        solution[p] = divide(acc, r_columns[p][p]);
    }
    let mut out = vec![Complex::zero(); columns.len()];
    for (p, &index) in kept.iter().enumerate() {
        out[index] = solution[p];
    }
    out
}

#[cfg(test)]
mod tests;
