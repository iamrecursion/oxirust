#![allow(dead_code, unused_imports)]
pub mod error {
    #[derive(Debug)]
    pub enum Error {
        General(String),
        InvalidInput(String),
        DofOutOfRange { dof: usize, n: usize },
        SingularSystem,
        NotConverged { solver: String, iterations: usize, residual: f64 },
    }
    pub type Result<T> = std::result::Result<T, Error>;
}
pub mod sparse {
    use std::collections::BTreeMap;
    pub struct CsrMatrix {
        pub row_ptr: Vec<usize>,
        pub col_indices: Vec<usize>,
        pub values: Vec<f64>,
        pub nrows: usize,
        pub ncols: usize,
    }
    impl CsrMatrix {
        pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
            let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); nrows];
            for &(r, c, v) in triplets {
                assert!(r < nrows && c < ncols);
                *rows[r].entry(c).or_insert(0.0) += v;
            }
            let mut row_ptr = vec![0];
            let mut col_indices = vec![];
            let mut values = vec![];
            for r in rows {
                for (c, v) in r {
                    col_indices.push(c);
                    values.push(v);
                }
                row_ptr.push(col_indices.len());
            }
            Self { row_ptr, col_indices, values, nrows, ncols }
        }
        pub fn get(&self, row: usize, col: usize) -> f64 {
            for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                if self.col_indices[idx] == col {
                    return self.values[idx];
                }
            }
            0.0
        }
        pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
            let mut y = vec![0.0; self.nrows];
            for row in 0..self.nrows {
                let mut s = 0.0;
                for idx in self.row_ptr[row]..self.row_ptr[row + 1] {
                    s += self.values[idx] * x[self.col_indices[idx]];
                }
                y[row] = s;
            }
            y
        }
    }
}
pub mod solvers {
    use crate::sparse::CsrMatrix;
    pub struct SolverStats {
        pub iterations: usize,
        pub residual: f64,
        pub converged: bool,
    }
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    pub fn pcg_solve(a: &CsrMatrix, b: &[f64], precond: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, SolverStats) {
        let n = b.len();
        let mut x = vec![0.0; n];
        let ax = a.mul_vec(&x);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();
        let mut z: Vec<f64> = r.iter().zip(precond.iter()).map(|(ri, mi)| ri * mi).collect();
        let mut p = z.clone();
        let mut rz_old = dot(&r, &z);
        let b_norm = dot(b, b).sqrt();
        let mut iters = 0;
        let mut converged = false;
        if b_norm < 1e-60 {
            let residual = dot(&r, &r).sqrt();
            return (x, SolverStats { iterations: 0, residual, converged: true });
        }
        for _ in 0..max_iter {
            iters += 1;
            let ap = a.mul_vec(&p);
            let p_ap = dot(&p, &ap);
            if p_ap.abs() < 1e-60 {
                break;
            }
            let alpha = rz_old / p_ap;
            for ((x_i, r_i), (&p_i, &ap_i)) in x.iter_mut().zip(r.iter_mut()).zip(p.iter().zip(ap.iter())) {
                *x_i += alpha * p_i;
                *r_i -= alpha * ap_i;
            }
            let r_norm = dot(&r, &r).sqrt();
            if r_norm / b_norm < tol {
                converged = true;
                break;
            }
            for ((z_i, &r_i), &pre_i) in z.iter_mut().zip(r.iter()).zip(precond.iter()) {
                *z_i = r_i * pre_i;
            }
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz_old;
            for (p_i, &z_i) in p.iter_mut().zip(z.iter()) {
                *p_i = z_i + beta * *p_i;
            }
            rz_old = rz_new;
        }
        let residual = dot(&r, &r).sqrt();
        (x, SolverStats { iterations: iters, residual, converged })
    }
}
pub mod homogenization {
    #[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/homogenization/matrix_utils.rs"]
    pub mod matrix_utils;
    #[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/homogenization/bounds.rs"]
    pub mod bounds;
    #[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/homogenization/advanced.rs"]
    pub mod advanced;
    #[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/homogenization/extended.rs"]
    pub mod extended;
    #[cfg(test)]
    #[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/homogenization/extended_tests.rs"]
    mod extended_tests;
}
fn main() {}
