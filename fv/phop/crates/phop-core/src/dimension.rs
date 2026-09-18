//! Buckingham-π dimensional reduction (a discovery *prior* for phop).
//!
//! In the pure EML grammar every internal node is `exp(x) − ln(y)`, so **every** argument to a
//! node must be dimensionless. An in-loss dimensional penalty on single-variable leaves would
//! therefore be degenerate (it could only forbid using a dimensioned variable at all). The
//! productive, EML-appropriate way to bring dimensional analysis into the search — the same way
//! AI-Feynman uses it — is as a **preprocessing reduction**: combine the dimensioned inputs into
//! the maximal set of independent *dimensionless* groups (the Buckingham-π theorem) and let phop
//! discover a law over those. The number of groups is `n_features − rank(D)`, where `D` is the
//! dimension matrix.
//!
//! Each feature carries a [`Dimension`] — integer exponents over the seven SI base units
//! `[s, m, kg, A, K, mol, cd]`. [`pi_groups`] returns an integer basis of the nullspace of `D`
//! (each basis vector `e` gives a dimensionless monomial `∏ xᵢ^{eᵢ}`); [`DataSet::to_dimensionless`]
//! materializes those monomials as the new feature columns.

use crate::dataset::DataSet;
use crate::error::{PhopError, Result};
use scirs2_core::ndarray::Array2;

/// Integer exponents over the seven SI base units `[s, m, kg, A, K, mol, cd]`.
pub type Dimension = [i32; 7];

/// The dimensionless dimension (all exponents zero).
pub const DIMENSIONLESS: Dimension = [0; 7];
/// Time `[s]`.
pub const TIME: Dimension = [1, 0, 0, 0, 0, 0, 0];
/// Length `[m]`.
pub const LENGTH: Dimension = [0, 1, 0, 0, 0, 0, 0];
/// Mass `[kg]`.
pub const MASS: Dimension = [0, 0, 1, 0, 0, 0, 0];

/// An exact rational (kept normalized: `den > 0`, `gcd(|num|, den) = 1`).
#[derive(Clone, Copy)]
struct Frac {
    num: i64,
    den: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

impl Frac {
    fn new(num: i64, den: i64) -> Frac {
        debug_assert!(den != 0);
        let s = if den < 0 { -1 } else { 1 };
        let (num, den) = (num * s, den * s);
        let g = gcd(num, den);
        Frac {
            num: num / g,
            den: den / g,
        }
    }
    fn int(n: i64) -> Frac {
        Frac { num: n, den: 1 }
    }
    fn zero() -> Frac {
        Frac { num: 0, den: 1 }
    }
    fn is_zero(self) -> bool {
        self.num == 0
    }
    fn sub(self, o: Frac) -> Frac {
        Frac::new(self.num * o.den - o.num * self.den, self.den * o.den)
    }
    fn mul(self, o: Frac) -> Frac {
        Frac::new(self.num * o.num, self.den * o.den)
    }
    fn div(self, o: Frac) -> Frac {
        Frac::new(self.num * o.den, self.den * o.num)
    }
}

/// Integer basis of the nullspace of the dimension matrix of `dims`.
///
/// Each returned vector `e` (length `dims.len()`) satisfies `∑ eᵢ · dim(xᵢ) = 0`, i.e. the monomial
/// `∏ xᵢ^{eᵢ}` is dimensionless. The basis has `n − rank` vectors; for fully-dimensionally-independent
/// inputs it is empty. Returned vectors are reduced (gcd 1) with a leading-positive sign convention.
#[must_use]
pub fn pi_groups(dims: &[Dimension]) -> Vec<Vec<i32>> {
    let n = dims.len();
    if n == 0 {
        return Vec::new();
    }
    // Dimension matrix D: 7 rows (base units) × n cols (features), as exact rationals.
    let mut m: Vec<Vec<Frac>> = (0..7)
        .map(|r| (0..n).map(|c| Frac::int(i64::from(dims[c][r]))).collect())
        .collect();

    // Reduced row echelon form; record the pivot column of each row.
    let rows = 7;
    let mut pivot_col = vec![usize::MAX; rows];
    let mut r = 0usize;
    for c in 0..n {
        if r >= rows {
            break;
        }
        // Find a pivot in column c at or below row r.
        let Some(p) = (r..rows).find(|&i| !m[i][c].is_zero()) else {
            continue;
        };
        m.swap(r, p);
        // Scale pivot row so the pivot is 1.
        let pivot = m[r][c];
        for cell in m[r].iter_mut() {
            *cell = cell.div(pivot);
        }
        // Eliminate column c from all other rows (clone the pivot row to avoid aliasing).
        let prow = m[r].clone();
        for (i, mrow) in m.iter_mut().enumerate() {
            if i != r && !mrow[c].is_zero() {
                let factor = mrow[c];
                for (cell, &pj) in mrow.iter_mut().zip(prow.iter()) {
                    *cell = cell.sub(factor.mul(pj));
                }
            }
        }
        pivot_col[r] = c;
        r += 1;
    }

    let pivots: Vec<usize> = pivot_col
        .iter()
        .copied()
        .filter(|&c| c != usize::MAX)
        .collect();
    let is_pivot = |c: usize| pivots.contains(&c);

    // One nullspace basis vector per free column.
    let mut basis = Vec::new();
    for free in (0..n).filter(|&c| !is_pivot(c)) {
        let mut vec_q = vec![Frac::zero(); n];
        vec_q[free] = Frac::int(1);
        // For each pivot row with pivot column `pc`: x_pc = -m[row][free].
        for (row, &pc) in pivot_col.iter().enumerate() {
            if pc != usize::MAX {
                vec_q[pc] = Frac::zero().sub(m[row][free]);
            }
        }
        // Clear denominators → integers, reduce by gcd, fix sign (first nonzero positive).
        let lcm = vec_q
            .iter()
            .fold(1i64, |acc, f| acc / gcd(acc, f.den) * f.den);
        let mut ints: Vec<i32> = vec_q
            .iter()
            .map(|f| (f.num * (lcm / f.den)) as i32)
            .collect();
        let g = ints
            .iter()
            .fold(0i64, |acc, &v| gcd(acc, i64::from(v)))
            .max(1) as i32;
        for v in &mut ints {
            *v /= g;
        }
        if let Some(&first) = ints.iter().find(|&&v| v != 0) {
            if first < 0 {
                for v in &mut ints {
                    *v = -*v;
                }
            }
        }
        basis.push(ints);
    }
    basis
}

impl DataSet {
    /// Reduce the features to their dimensionless Buckingham-π groups, given each feature's
    /// [`Dimension`]. The new feature columns are the monomials `∏ xᵢ^{eᵢ}` for each π-group; the
    /// target is left unchanged. The π-group exponent vectors are returned alongside.
    ///
    /// # Errors
    /// Returns [`PhopError::ShapeMismatch`] if `feature_dims.len() != n_vars`, or
    /// [`PhopError::NotConverged`] if the inputs are dimensionally independent (no π-groups exist).
    pub fn to_dimensionless(&self, feature_dims: &[Dimension]) -> Result<(DataSet, Vec<Vec<i32>>)> {
        let nv = self.n_vars();
        if feature_dims.len() != nv {
            return Err(PhopError::ShapeMismatch(format!(
                "expected {nv} feature dimensions, got {}",
                feature_dims.len()
            )));
        }
        let groups = pi_groups(feature_dims);
        if groups.is_empty() {
            return Err(PhopError::NotConverged(
                "no dimensionless π-groups: the features are dimensionally independent".to_string(),
            ));
        }
        let n_rows = self.len();
        let n_pi = groups.len();
        let mut data = Vec::with_capacity(n_rows * n_pi);
        for row in 0..n_rows {
            for g in &groups {
                let mut v = 1.0_f64;
                for (j, &e) in g.iter().enumerate() {
                    if e != 0 {
                        v *= self.x[[row, j]].powi(e);
                    }
                }
                data.push(v);
            }
        }
        let x = Array2::from_shape_vec((n_rows, n_pi), data)
            .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
        let feature_names = (0..n_pi).map(|k| format!("pi{k}")).collect();
        let ds = DataSet {
            x,
            y: self.y.clone(),
            feature_names,
            target_name: self.target_name.clone(),
        };
        Ok((ds, groups))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn pi_group_for_velocity_time_length() {
        // x0 = length [m], x1 = time [s], x2 = velocity [m/s]. The dimensionless group is
        // x2 · x1 / x0  (velocity·time/length), i.e. exponents (x0:-1, x1:+1, x2:+1).
        let dims = [
            LENGTH,                 // x0
            TIME,                   // x1
            [-1, 1, 0, 0, 0, 0, 0], // x2 = m·s^-1
        ];
        let groups = pi_groups(&dims);
        assert_eq!(
            groups.len(),
            1,
            "expected exactly one π-group, got {groups:?}"
        );
        // The group must be proportional to (-1, +1, +1) (sign-normalized to leading positive).
        let g = &groups[0];
        // leading-positive convention → first nonzero is +1 (x0 coeff = -1 would flip): so expect
        // either (-1,1,1) or its negation normalized; check the *ratios*.
        let nonzero: Vec<i32> = g.clone();
        // x0:x1:x2 should be -1:1:1 up to sign.
        assert!(
            nonzero == vec![-1, 1, 1] || nonzero == vec![1, -1, -1],
            "unexpected π-group {nonzero:?}"
        );
    }

    #[test]
    fn to_dimensionless_yields_constant_group() {
        // Build data where x2 = x0 / x1 exactly, so the group x2·x1/x0 = 1 for every row.
        let n = 12;
        let mut xd = Vec::new();
        for i in 0..n {
            let x0 = 1.0 + i as f64; // length
            let x1 = 2.0 + i as f64; // time
            let x2 = x0 / x1; // velocity
            xd.extend_from_slice(&[x0, x1, x2]);
        }
        let x = Array2::from_shape_vec((n, 3), xd).unwrap();
        let y = Array1::from(vec![0.0; n]);
        let ds = DataSet::from_arrays(x, y).unwrap();

        let dims = [LENGTH, TIME, [-1, 1, 0, 0, 0, 0, 0]];
        let (red, groups) = ds.to_dimensionless(&dims).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(red.n_vars(), 1);
        // Every π value should equal 1 (x2·x1/x0 = 1), regardless of the basis sign.
        for r in 0..n {
            let v = red.x[[r, 0]];
            assert!((v - 1.0).abs() < 1e-9, "π not constant: {v}");
        }
    }

    #[test]
    fn independent_dims_have_no_group() {
        // length and time are dimensionally independent → no dimensionless group.
        assert!(pi_groups(&[LENGTH, TIME]).is_empty());
    }
}
