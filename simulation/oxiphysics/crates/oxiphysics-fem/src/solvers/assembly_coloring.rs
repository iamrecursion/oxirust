// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Element graph coloring for race-free, reproducible parallel FEM assembly.
//!
//! Elements sharing a DOF cannot be scattered into the global matrix at the same time
//! without a data race.  Graph coloring assigns each element a color such that no two
//! elements of the same color share any DOF.  The colors are then processed one after
//! another and the elements within one color in parallel, without any synchronization:
//! an entry of the global matrix is touched by at most one element per color.
//!
//! Because the colors always run in the same order, every entry is summed in the same
//! order whatever the number of threads, so the assembled matrix is bitwise
//! reproducible.  (Floating-point addition is not associative; a lock- or CAS-based
//! scatter that adds in thread-arrival order would round differently from run to run.)
//!
//! The assembly itself ([`try_assemble_colored_csr`]) takes no lock.  The sparsity
//! pattern is built once with sorted, unique column indices per row, and every element
//! locates its entries by binary search in the row it scatters into.

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::parallel_solver::{AssemblyTask, CsrMatrix};

/// Result of coloring the element graph.
#[derive(Debug, Clone)]
pub struct ElementColoringResult {
    /// `colors[elem_idx]` — color (0-based) assigned to this element.
    pub colors: Vec<usize>,
    /// Total number of colors used.
    pub n_colors: usize,
    /// `buckets[color]` — list of element indices with that color, in ascending order.
    ///
    /// The elements of a bucket share no DOF, with one exception: when more than
    /// `MAX_COLORS` (64) colors would be needed, every element that does not fit goes
    /// into the last bucket.  That *serial fallback bucket* may hold elements that share
    /// DOFs; the assemblers of this module detect such a bucket and scatter it
    /// sequentially in element order.
    pub buckets: Vec<Vec<usize>>,
}

/// Maximum number of graph colors before switching to the serial fallback bucket.
const MAX_COLORS: usize = 64;

/// Assign colors to elements such that no two elements of the same color share a DOF.
///
/// Uses a greedy graph coloring algorithm in a single pass over the elements, in
/// ascending order:
/// 1. Every DOF remembers the set of colors of the elements that use it.
/// 2. The colors forbidden for an element are those remembered at its DOFs, i.e. the
///    colors of the adjacent elements (those sharing a DOF) colored so far.
/// 3. Assign the smallest unused color.
/// 4. If more than `MAX_COLORS` would be needed, the remaining elements go into the last
///    bucket, which then is a serial fallback bucket (see
///    [`ElementColoringResult::buckets`]).
///
/// The result is deterministic: it depends only on the element order and the DOF lists.
///
/// # Arguments
/// * `n_elements` — total number of elements.
/// * `element_dofs` — for each element, the list of global DOF indices it uses.
pub fn color_elements(n_elements: usize, element_dofs: &[Vec<usize>]) -> ElementColoringResult {
    color_by(n_elements, |e| element_dofs[e].as_slice())
}

/// Color the elements of an assembly task list by the DOFs of the tasks.
///
/// Equivalent to [`color_elements`] with `element_dofs[e] = tasks[e].global_dofs`, but
/// without copying the DOF lists.
pub fn color_tasks(tasks: &[AssemblyTask]) -> ElementColoringResult {
    color_by(tasks.len(), |e| tasks[e].global_dofs.as_slice())
}

/// Greedy coloring of `n_elements` elements whose DOFs are given by `dofs_of`.
///
/// Every DOF keeps a 64-bit mask of the colors already used by the elements that touch
/// it.  The colors forbidden for an element are the union of the masks of its DOFs, which
/// is exactly the set of colors of its already-colored neighbours, so the work is linear
/// in the total number of element DOFs (scanning the neighbour lists instead would be
/// quadratic in the number of elements that share a DOF).
fn color_by<'a, F>(n_elements: usize, dofs_of: F) -> ElementColoringResult
where
    F: Fn(usize) -> &'a [usize],
{
    if n_elements == 0 {
        return ElementColoringResult {
            colors: Vec::new(),
            n_colors: 0,
            buckets: Vec::new(),
        };
    }

    let n_dofs = (0..n_elements)
        .flat_map(|e| dofs_of(e).iter().copied())
        .max()
        .map_or(0, |max_dof| max_dof + 1);
    // Bit `c` of `used[d]` is set when an element of color `c` uses DOF `d`.
    let mut used = vec![0_u64; n_dofs];
    let mut colors = vec![0_usize; n_elements];
    let mut n_colors_used = 0_usize;

    for (e, color_slot) in colors.iter_mut().enumerate() {
        let dofs = dofs_of(e);
        let forbidden = dofs.iter().fold(0_u64, |mask, &d| mask | used[d]);

        // Smallest color that no neighbour uses (64 when all of them are taken).
        let mut chosen_color = forbidden.trailing_ones() as usize;
        if chosen_color >= MAX_COLORS {
            // Serial fallback bucket: the last color, shared with neighbours.
            chosen_color = MAX_COLORS - 1;
        }

        let bit = 1_u64 << chosen_color;
        for &d in dofs {
            used[d] |= bit;
        }
        *color_slot = chosen_color;
        n_colors_used = n_colors_used.max(chosen_color + 1);
    }

    // Build buckets (ascending element order inside every bucket)
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_colors_used];
    for (e, &c) in colors.iter().enumerate() {
        buckets[c].push(e);
    }

    ElementColoringResult {
        colors,
        n_colors: n_colors_used,
        buckets,
    }
}

/// Check that every DOF index addresses a system with `ndofs` degrees of freedom.
///
/// # Errors
/// [`Error::DofOutOfRange`] for the first offending index.
pub(crate) fn check_dofs(ndofs: usize, dofs: &[usize]) -> Result<()> {
    match dofs.iter().find(|&&dof| dof >= ndofs) {
        Some(&dof) => Err(Error::DofOutOfRange { dof, n: ndofs }),
        None => Ok(()),
    }
}

/// Check the DOF indices and the local matrix size of every assembly task.
///
/// # Errors
/// [`Error::DofOutOfRange`] for a DOF index `>= ndofs`, [`Error::InvalidInput`] for a
/// local matrix `ke` that does not have `ndof * ndof` entries.
pub(crate) fn validate_tasks(ndofs: usize, tasks: &[AssemblyTask]) -> Result<()> {
    for (e, task) in tasks.iter().enumerate() {
        let ndof = task.ndof();
        if ndof.checked_mul(ndof) != Some(task.ke.len()) {
            return Err(Error::InvalidInput(format!(
                "element {e} has {ndof} DOFs, so its local matrix needs {ndof} x {ndof} entries, \
                 but {} were given",
                task.ke.len()
            )));
        }
        check_dofs(ndofs, &task.global_dofs)?;
    }
    Ok(())
}

/// Check that `coloring` lists every element `0..n_tasks` exactly once.
fn validate_coloring(n_tasks: usize, coloring: &ElementColoringResult) -> Result<()> {
    let mut seen = vec![false; n_tasks];
    for &e in coloring.buckets.iter().flatten() {
        match seen.get_mut(e) {
            None => {
                return Err(Error::InvalidInput(format!(
                    "the coloring refers to element {e}, but there are only {n_tasks} elements"
                )));
            }
            Some(flag) if *flag => {
                return Err(Error::InvalidInput(format!(
                    "element {e} appears more than once in the coloring"
                )));
            }
            Some(flag) => *flag = true,
        }
    }
    if let Some(missing) = seen.iter().position(|&listed| !listed) {
        return Err(Error::InvalidInput(format!(
            "element {missing} is not part of the coloring, its stiffness would be lost"
        )));
    }
    Ok(())
}

/// Rows whose raw column list reaches this length are compacted (sorted and
/// deduplicated) while the pattern is built, which keeps the peak memory proportional to
/// the number of distinct entries rather than to the sum of the squared element sizes.
const COMPACT_MIN_LEN: usize = 64;

/// Sparsity pattern of the assembled matrix as CSR `(row_offsets, col_indices)` with
/// sorted, unique column indices in every row.  All DOFs must be `< ndofs`.
fn build_pattern(ndofs: usize, tasks: &[AssemblyTask]) -> (Vec<usize>, Vec<usize>) {
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); ndofs];
    let mut settled = vec![0usize; ndofs];
    for task in tasks {
        for &row in &task.global_dofs {
            let cols = &mut rows[row];
            cols.extend_from_slice(&task.global_dofs);
            if cols.len() >= 2 * settled[row].max(COMPACT_MIN_LEN) {
                cols.sort_unstable();
                cols.dedup();
                settled[row] = cols.len();
            }
        }
    }
    rows.par_iter_mut().for_each(|cols| {
        cols.sort_unstable();
        cols.dedup();
    });

    let mut row_offsets = Vec::with_capacity(ndofs + 1);
    row_offsets.push(0);
    let mut nnz = 0usize;
    for cols in &rows {
        nnz += cols.len();
        row_offsets.push(nnz);
    }
    let mut col_indices = Vec::with_capacity(nnz);
    for cols in &rows {
        col_indices.extend_from_slice(cols);
    }
    (row_offsets, col_indices)
}

/// For every bucket, whether two different elements of it share a DOF.
///
/// Two elements that write the same matrix entry necessarily share the DOF of its row
/// (and both write the diagonal entry of that DOF), so comparing DOFs is exact.
fn conflicting_buckets(
    ndofs: usize,
    tasks: &[AssemblyTask],
    coloring: &ElementColoringResult,
) -> Vec<bool> {
    // (bucket, element) that claimed a DOF last.
    let mut claimed = vec![(usize::MAX, usize::MAX); ndofs];
    coloring
        .buckets
        .iter()
        .enumerate()
        .map(|(bucket_index, bucket)| {
            for &e in bucket {
                for &dof in &tasks[e].global_dofs {
                    let (claimed_bucket, claimed_by) = claimed[dof];
                    if claimed_bucket == bucket_index && claimed_by != e {
                        return true;
                    }
                    claimed[dof] = (bucket_index, e);
                }
            }
            false
        })
        .collect()
}

/// `slot += value` for an `f64` kept as bits in an [`AtomicU64`].
///
/// This is a plain load followed by a store, not an atomic read-modify-write: it is only
/// correct while no other thread touches `slot` concurrently, which the callers guarantee
/// by scattering only conflict-free buckets in parallel.
fn add_to_slot(slot: &AtomicU64, value: f64) {
    let sum = f64::from_bits(slot.load(Ordering::Relaxed)) + value;
    slot.store(sum.to_bits(), Ordering::Relaxed);
}

/// Scatter the local matrix of one element into the CSR `values`.
fn scatter_element(
    task: &AssemblyTask,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[AtomicU64],
) -> Result<()> {
    let ndof = task.ndof();
    for (local_row, &row) in task.global_dofs.iter().enumerate() {
        let first = row_offsets[row];
        let row_cols = &col_indices[first..row_offsets[row + 1]];
        let ke_row = &task.ke[local_row * ndof..(local_row + 1) * ndof];
        for (&col, &ke_value) in task.global_dofs.iter().zip(ke_row) {
            let offset = row_cols.binary_search(&col).map_err(|_| {
                Error::General(format!(
                    "assembly: entry ({row}, {col}) is missing from the sparsity pattern"
                ))
            })?;
            add_to_slot(&values[first + offset], ke_value);
        }
    }
    Ok(())
}

/// Assemble a global CSR stiffness matrix using element graph coloring.
///
/// The sparsity pattern is the union of the element connectivities; the column indices of
/// every row are sorted and unique.  The colors are then scattered one after another and
/// the elements of one color in parallel.  Elements of the same color share no DOF, hence
/// no matrix entry, so no lock is needed, and the entries are summed color by color in a
/// fixed order: **the result is bitwise identical for every thread count and every run.**
///
/// A bucket whose elements do share a DOF — the serial fallback bucket of
/// [`color_elements`] or an invalid hand-made coloring — is detected and scattered
/// sequentially in element order, so the result is correct (and reproducible) for any
/// coloring that lists every element once.
///
/// # Arguments
/// * `ndofs` — total number of global DOFs.
/// * `tasks` — per-element stiffness contributions.
/// * `coloring` — result of [`color_elements`] or [`color_tasks`]; only its `buckets`
///   are read.
///
/// # Errors
/// * [`Error::DofOutOfRange`] if a DOF index is not `< ndofs`.
/// * [`Error::InvalidInput`] if a local matrix is not `ndof x ndof`, or if `coloring`
///   does not list every element of `tasks` exactly once.
pub fn try_assemble_colored_csr(
    ndofs: usize,
    tasks: &[AssemblyTask],
    coloring: &ElementColoringResult,
) -> Result<CsrMatrix> {
    validate_tasks(ndofs, tasks)?;
    validate_coloring(tasks.len(), coloring)?;

    // ── Step 1: sparsity pattern ─────────────────────────────────────────────
    let (row_offsets, col_indices) = build_pattern(ndofs, tasks);

    // ── Step 2: scatter color by color ───────────────────────────────────────
    // Values live in atomics only so that the parallel scatter needs no `unsafe`; within
    // a conflict-free bucket no two threads touch the same entry, and consecutive
    // buckets are separated by the join of the parallel loop.
    let values: Vec<AtomicU64> = (0..col_indices.len())
        .map(|_| AtomicU64::new(0.0_f64.to_bits()))
        .collect();
    let needs_serial = conflicting_buckets(ndofs, tasks, coloring);

    for (bucket, &serial) in coloring.buckets.iter().zip(&needs_serial) {
        if serial {
            for &e in bucket {
                scatter_element(&tasks[e], &row_offsets, &col_indices, &values)?;
            }
        } else {
            bucket.par_iter().try_for_each(|&e| {
                scatter_element(&tasks[e], &row_offsets, &col_indices, &values)
            })?;
        }
    }

    let values: Vec<f64> = values
        .into_iter()
        .map(|slot| f64::from_bits(slot.into_inner()))
        .collect();

    Ok(CsrMatrix {
        nrows: ndofs,
        ncols: ndofs,
        row_offsets,
        col_indices,
        values,
    })
}

/// Assemble a global CSR stiffness matrix using element graph coloring.
///
/// See [`try_assemble_colored_csr`] for the algorithm and the reproducibility guarantee.
///
/// # Panics
///
/// Panics for the inputs that [`try_assemble_colored_csr`] rejects with an error (a DOF
/// index `>= ndofs`, a local matrix of the wrong size, or a coloring that does not list
/// every element once).  Use [`try_assemble_colored_csr`] to handle them.
pub fn assemble_colored_csr(
    ndofs: usize,
    tasks: &[AssemblyTask],
    coloring: &ElementColoringResult,
) -> CsrMatrix {
    match try_assemble_colored_csr(ndofs, tasks, coloring) {
        Ok(matrix) => matrix,
        Err(error) => panic!("assemble_colored_csr: {error}"),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_solver::ParallelAssembler;

    fn make_test_tasks() -> (Vec<AssemblyTask>, Vec<Vec<usize>>) {
        // 4 elements in a 1D bar: DOFs [0,1], [1,2], [2,3], [3,4]
        // Element stiffness: [[1,-1],[-1,1]]
        let dofs = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
        let ke = vec![1.0, -1.0, -1.0, 1.0];
        let tasks: Vec<AssemblyTask> = dofs
            .iter()
            .map(|d| AssemblyTask::new(d.clone(), ke.clone()))
            .collect();
        (tasks, dofs)
    }

    #[test]
    fn test_coloring_valid() {
        let (tasks, dofs) = make_test_tasks();
        let coloring = color_elements(tasks.len(), &dofs);

        // Verify: no two elements of the same color share a DOF
        for (color, bucket) in coloring.buckets.iter().enumerate() {
            for &e1 in bucket {
                for &e2 in bucket {
                    if e1 == e2 {
                        continue;
                    }
                    let dofs1: std::collections::HashSet<usize> =
                        dofs[e1].iter().copied().collect();
                    let dofs2: std::collections::HashSet<usize> =
                        dofs[e2].iter().copied().collect();
                    let overlap: std::collections::HashSet<usize> =
                        dofs1.intersection(&dofs2).copied().collect();
                    assert!(
                        overlap.is_empty(),
                        "Color {color}: elements {e1} and {e2} share DOFs {overlap:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn colored_assembly_matches_the_closed_form_bar_stiffness() {
        let (tasks, dofs) = make_test_tasks();
        let ndofs = 5;

        // The stiffness of the bar is known in closed form: tridiagonal with the diagonal
        // (1, 2, 2, 2, 1) and off-diagonal entries -1.
        let mut expected = vec![vec![0.0; ndofs]; ndofs];
        for (i, row) in expected.iter_mut().enumerate() {
            row[i] = if i == 0 || i == ndofs - 1 { 1.0 } else { 2.0 };
            if i + 1 < ndofs {
                row[i + 1] = -1.0;
            }
            if i > 0 {
                row[i - 1] = -1.0;
            }
        }

        // `ParallelAssembler::assemble` delegates to the colored path: check both entry
        // points against the closed form, not against each other.
        let asm = ParallelAssembler::new(ndofs);
        let mat_default = asm.assemble(&tasks);
        let coloring = color_elements(tasks.len(), &dofs);
        let mat_colored = assemble_colored_csr(ndofs, &tasks, &coloring);

        for mat in [&mat_default, &mat_colored] {
            assert_eq!(mat.nrows, ndofs);
            assert_eq!(mat.ncols, ndofs);
            assert_eq!(mat.nnz(), 3 * ndofs - 2);
            for (i, row) in expected.iter().enumerate() {
                let first = mat.row_offsets[i];
                let cols = &mat.col_indices[first..mat.row_offsets[i + 1]];
                for (j, &want) in row.iter().enumerate() {
                    let got = cols
                        .binary_search(&j)
                        .map_or(0.0, |k| mat.values[first + k]);
                    assert!(
                        (got - want).abs() < 1e-14,
                        "K[{i}][{j}] = {got}, expected {want}"
                    );
                }
            }
        }
        assert_eq!(mat_default.row_offsets, mat_colored.row_offsets);
        assert_eq!(mat_default.col_indices, mat_colored.col_indices);
    }

    #[test]
    fn test_coloring_small_mesh() {
        // 4 elements sharing nodes:
        //   elem 0: dofs [0,1]
        //   elem 1: dofs [1,2]
        //   elem 2: dofs [2,3]
        //   elem 3: dofs [0,3]  (wraps around)
        let dofs = vec![vec![0usize, 1], vec![1, 2], vec![2, 3], vec![0, 3]];
        let coloring = color_elements(4, &dofs);

        assert_eq!(coloring.colors.len(), 4);
        assert!(coloring.n_colors >= 2, "Expected at least 2 colors");

        // Elements 0 and 2 don't share DOFs → can be same color
        // Elements 1 and 3 don't share DOFs → can be same color
        // Verify no two same-color elements share a DOF
        for bucket in &coloring.buckets {
            for &e1 in bucket {
                for &e2 in bucket {
                    if e1 == e2 {
                        continue;
                    }
                    let dofs1: std::collections::HashSet<usize> =
                        dofs[e1].iter().copied().collect();
                    let dofs2: std::collections::HashSet<usize> =
                        dofs[e2].iter().copied().collect();
                    let overlap: Vec<usize> = dofs1.intersection(&dofs2).copied().collect();
                    assert!(
                        overlap.is_empty(),
                        "Same-color elements {e1} and {e2} share DOFs {overlap:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_coloring_empty() {
        let result = color_elements(0, &[]);
        assert_eq!(result.n_colors, 0);
        assert_eq!(result.buckets.len(), 0);
    }
}

#[cfg(test)]
mod regression_tests;
