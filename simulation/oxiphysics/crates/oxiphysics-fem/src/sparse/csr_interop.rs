// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Lossless conversions between the CSR matrix types of the workspace.
//!
//! Three compressed-sparse-row containers carry the same data under different field
//! names:
//!
//! | type | pointer array | column array |
//! |---|---|---|
//! | [`crate::sparse::CsrMatrix`] | `row_ptr` | `col_indices` |
//! | [`crate::parallel_solver::CsrMatrix`] | `row_offsets` | `col_indices` |
//! | [`oxiphysics_core::sparse::CsrMatrix`] (the canonical interchange type) | `row_ptr` | `col_idx` |
//!
//! All three store `nrows`, `ncols`, the row pointer array (length `nrows + 1`), the
//! column index array and the value array in the same layout, so `From` impls convert in
//! every direction.  Converting by value **moves the three arrays** (no copy), so the
//! output of the colored parallel assembler
//! ([`crate::parallel_solver::ParallelAssembler`]) can be handed to
//! [`crate::assembly::conjugate_gradient`], to
//! [`ParallelPcgSolver`](crate::parallel_solver::ParallelPcgSolver) or to the solvers of
//! `oxiphysics-core` at no cost; converting a reference clones the arrays.
//!
//! The conversions do not inspect the data: whatever invariants the source holds
//! (sorted columns, no duplicates, ...) are carried over unchanged.

use crate::parallel_solver::CsrMatrix as ParCsr;
use oxiphysics_core::sparse::CsrMatrix as CoreCsr;

use super::types::CsrMatrix as FemCsr;

/// Copy a core matrix (which does not implement `Clone`) array by array.
fn clone_core(matrix: &CoreCsr) -> CoreCsr {
    CoreCsr {
        nrows: matrix.nrows,
        ncols: matrix.ncols,
        row_ptr: matrix.row_ptr.clone(),
        col_idx: matrix.col_idx.clone(),
        values: matrix.values.clone(),
    }
}

// ── crate::sparse::CsrMatrix ↔ crate::parallel_solver::CsrMatrix ─────────────

impl From<FemCsr> for ParCsr {
    fn from(matrix: FemCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_offsets: matrix.row_ptr,
            col_indices: matrix.col_indices,
            values: matrix.values,
        }
    }
}

impl From<&FemCsr> for ParCsr {
    fn from(matrix: &FemCsr) -> Self {
        Self::from(matrix.clone())
    }
}

impl From<ParCsr> for FemCsr {
    fn from(matrix: ParCsr) -> Self {
        Self {
            row_ptr: matrix.row_offsets,
            col_indices: matrix.col_indices,
            values: matrix.values,
            nrows: matrix.nrows,
            ncols: matrix.ncols,
        }
    }
}

impl From<&ParCsr> for FemCsr {
    fn from(matrix: &ParCsr) -> Self {
        Self::from(matrix.clone())
    }
}

// ── oxiphysics_core::sparse::CsrMatrix ↔ crate::sparse::CsrMatrix ────────────

impl From<CoreCsr> for FemCsr {
    fn from(matrix: CoreCsr) -> Self {
        Self {
            row_ptr: matrix.row_ptr,
            col_indices: matrix.col_idx,
            values: matrix.values,
            nrows: matrix.nrows,
            ncols: matrix.ncols,
        }
    }
}

impl From<&CoreCsr> for FemCsr {
    fn from(matrix: &CoreCsr) -> Self {
        Self::from(clone_core(matrix))
    }
}

impl From<FemCsr> for CoreCsr {
    fn from(matrix: FemCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_ptr,
            col_idx: matrix.col_indices,
            values: matrix.values,
        }
    }
}

impl From<&FemCsr> for CoreCsr {
    fn from(matrix: &FemCsr) -> Self {
        Self::from(matrix.clone())
    }
}

// ── oxiphysics_core::sparse::CsrMatrix ↔ crate::parallel_solver::CsrMatrix ───

impl From<CoreCsr> for ParCsr {
    fn from(matrix: CoreCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_offsets: matrix.row_ptr,
            col_indices: matrix.col_idx,
            values: matrix.values,
        }
    }
}

impl From<&CoreCsr> for ParCsr {
    fn from(matrix: &CoreCsr) -> Self {
        Self::from(clone_core(matrix))
    }
}

impl From<ParCsr> for CoreCsr {
    fn from(matrix: ParCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_offsets,
            col_idx: matrix.col_indices,
            values: matrix.values,
        }
    }
}

impl From<&ParCsr> for CoreCsr {
    fn from(matrix: &ParCsr) -> Self {
        Self::from(matrix.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::conjugate_gradient;
    use crate::parallel_solver::{AssemblyTask, ParallelAssembler, ParallelPcgSolver};
    use oxiphysics_core::sparse::cg_solve;

    /// 3 x 4 matrix `[[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]]` in the layout of the
    /// parallel-solver type.
    fn sample() -> ParCsr {
        ParCsr {
            nrows: 3,
            ncols: 4,
            row_offsets: vec![0, 2, 2, 4],
            col_indices: vec![0, 2, 1, 3],
            values: vec![1.0, 2.0, 3.0, 4.0],
        }
    }

    fn assert_same(a: &ParCsr, b: &ParCsr) {
        assert_eq!(a.nrows, b.nrows);
        assert_eq!(a.ncols, b.ncols);
        assert_eq!(a.row_offsets, b.row_offsets);
        assert_eq!(a.col_indices, b.col_indices);
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn every_conversion_chain_returns_the_original_arrays() {
        let original = sample();

        let fem = FemCsr::from(original.clone());
        assert_eq!(fem.row_ptr, vec![0, 2, 2, 4]);
        assert_eq!(fem.col_indices, vec![0, 2, 1, 3]);
        assert_eq!((fem.nrows, fem.ncols), (3, 4));
        assert_eq!(fem.get(2, 3), 4.0);
        assert_eq!(fem.get(1, 1), 0.0);
        assert_same(&original, &ParCsr::from(fem.clone()));

        let core = CoreCsr::from(original.clone());
        assert_eq!(core.row_ptr, vec![0, 2, 2, 4]);
        assert_eq!(core.col_idx, vec![0, 2, 1, 3]);
        assert_eq!((core.nrows, core.ncols), (3, 4));
        assert_eq!(core.get(0, 2), 2.0);
        assert_same(&original, &ParCsr::from(core));

        let core_from_fem = CoreCsr::from(fem.clone());
        let fem_again = FemCsr::from(core_from_fem);
        assert_eq!(fem_again.row_ptr, fem.row_ptr);
        assert_eq!(fem_again.col_indices, fem.col_indices);
        assert_eq!(fem_again.values, fem.values);
        assert_same(&original, &ParCsr::from(fem_again));
    }

    #[test]
    fn converting_a_reference_clones_and_leaves_the_source_intact() {
        let original = sample();

        let fem = FemCsr::from(&original);
        let core = CoreCsr::from(&original);
        assert_same(&original, &sample());
        assert_same(&original, &ParCsr::from(&fem));
        assert_same(&original, &ParCsr::from(&core));

        let fem_from_core = FemCsr::from(&core);
        let core_from_fem = CoreCsr::from(&fem);
        assert_eq!(fem_from_core.col_indices, fem.col_indices);
        assert_eq!(core_from_fem.col_idx, core.col_idx);
        assert_eq!(core_from_fem.values, core.values);
        assert_eq!(core.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn by_value_conversion_moves_the_arrays() {
        let original = sample();
        let values_ptr = original.values.as_ptr();
        let columns_ptr = original.col_indices.as_ptr();
        let fem = FemCsr::from(original);
        assert_eq!(fem.values.as_ptr(), values_ptr);
        assert_eq!(fem.col_indices.as_ptr(), columns_ptr);
    }

    /// Stiffness tasks of a bar of `n` unit springs clamped at node 0 and loaded at
    /// node `n`: DOF `i` is node `i + 1`.  `K` is `tridiag(-1, 2, -1)` with a last
    /// diagonal entry of 1 and the exact solution of `K u = e_last` is `u_i = i + 1`.
    fn bar_tasks(n: usize) -> Vec<AssemblyTask> {
        let mut tasks = vec![AssemblyTask::new(vec![0], vec![1.0])];
        for e in 1..n {
            tasks.push(AssemblyTask::new(
                vec![e - 1, e],
                vec![1.0, -1.0, -1.0, 1.0],
            ));
        }
        tasks
    }

    #[test]
    fn assembled_matrix_feeds_every_solver_of_the_workspace() {
        let n = 40;
        let assembled = ParallelAssembler::new(n).assemble(&bar_tasks(n));
        let mut load = vec![0.0; n];
        load[n - 1] = 1.0;
        let exact: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let assert_exact = |label: &str, x: &[f64]| {
            for (i, (xi, ei)) in x.iter().zip(&exact).enumerate() {
                assert!((xi - ei).abs() < 1e-6, "{label}: u[{i}] = {xi}, exact {ei}");
            }
        };

        // The rayon PCG consumes the assembler output as it is.
        let mut x = vec![0.0; n];
        let stats = ParallelPcgSolver::new(1000, 1e-12).solve(&assembled, &load, &mut x);
        assert!(stats.converged, "{stats:?}");
        assert_exact("ParallelPcgSolver", &x);

        // `assembly::conjugate_gradient` takes the sparse-module type: a by-value
        // conversion hands the arrays over.
        let fem: FemCsr = assembled.clone().into();
        for i in 0..n {
            let expected = if i + 1 == n { 1.0 } else { 2.0 };
            assert_eq!(fem.get(i, i), expected, "diagonal {i}");
            if i + 1 < n {
                assert_eq!(fem.get(i, i + 1), -1.0);
                assert_eq!(fem.get(i + 1, i), -1.0);
            }
        }
        let (x, iterations, residual) = conjugate_gradient(&fem, &load, 1e-12, 1000);
        assert!(residual < 1e-12, "residual {residual} after {iterations}");
        assert_exact("conjugate_gradient", &x);

        // The canonical core type works with the core solvers.
        let core: CoreCsr = (&assembled).into();
        let x0 = vec![0.0; n];
        let (x, _, residual) = cg_solve(&core, &load, &x0, 1e-12, 1000);
        assert!(residual < 1e-12, "residual {residual}");
        assert_exact("cg_solve", &x);
    }
}
