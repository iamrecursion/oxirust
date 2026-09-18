// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests of the colored parallel assembly.
//!
//! The element matrices of these tests hold values spread over nine decades that have no
//! short binary expansion, so the rounded sum of an entry depends on the order of the
//! additions.  A scatter that adds in thread-arrival order (the former mutex-per-entry
//! implementation) therefore fails the bitwise comparisons between thread counts; with
//! exactly representable values such as `±1` these tests would pass vacuously.

use super::*;
use crate::error::Error;
use crate::parallel_solver::ParallelAssembler;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a rayon pool can be built")
}

/// Deterministic, order-sensitive stiffness value.
fn awkward(seed: usize) -> f64 {
    let fraction = (seed as f64 * 0.618_033_988_749_894_9).fract();
    10.0_f64.powi((seed % 9) as i32 - 4) * (1.0 + fraction) / 3.0
}

/// `nx * ny` bilinear quadrilaterals with two DOFs per node, awkward local matrices.
fn grid_tasks(nx: usize, ny: usize) -> (Vec<AssemblyTask>, usize) {
    let node = |i: usize, j: usize| j * (nx + 1) + i;
    let mut tasks = Vec::new();
    for cj in 0..ny {
        for ci in 0..nx {
            let nodes = [
                node(ci, cj),
                node(ci + 1, cj),
                node(ci + 1, cj + 1),
                node(ci, cj + 1),
            ];
            let dofs: Vec<usize> = nodes.iter().flat_map(|&n| [2 * n, 2 * n + 1]).collect();
            let e = tasks.len();
            let ke: Vec<f64> = (0..64).map(|k| awkward(e * 64 + k)).collect();
            tasks.push(AssemblyTask::new(dofs, ke));
        }
    }
    (tasks, 2 * (nx + 1) * (ny + 1))
}

/// Dense matrix and touched-entry mask built by a plain sequential loop over the
/// elements, either in ascending or in descending element order.
fn dense_reference(
    ndofs: usize,
    tasks: &[AssemblyTask],
    ascending: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
    let mut dense = vec![vec![0.0_f64; ndofs]; ndofs];
    let mut touched = vec![vec![false; ndofs]; ndofs];
    let order: Vec<usize> = if ascending {
        (0..tasks.len()).collect()
    } else {
        (0..tasks.len()).rev().collect()
    };
    for e in order {
        let task = &tasks[e];
        let ndof = task.ndof();
        for (li, &row) in task.global_dofs.iter().enumerate() {
            for (lj, &col) in task.global_dofs.iter().enumerate() {
                dense[row][col] += task.ke[li * ndof + lj];
                touched[row][col] = true;
            }
        }
    }
    (dense, touched)
}

fn entry(matrix: &CsrMatrix, row: usize, col: usize) -> Option<f64> {
    let first = matrix.row_offsets[row];
    let cols = &matrix.col_indices[first..matrix.row_offsets[row + 1]];
    cols.binary_search(&col)
        .ok()
        .map(|k| matrix.values[first + k])
}

#[test]
fn the_test_data_really_depend_on_the_summation_order() {
    // Guard against a vacuous determinism test: adding the elements in ascending and in
    // descending order must round differently somewhere.
    let (tasks, ndofs) = grid_tasks(12, 12);
    let (forward, _) = dense_reference(ndofs, &tasks, true);
    let (backward, _) = dense_reference(ndofs, &tasks, false);
    let differing = forward
        .iter()
        .flatten()
        .zip(backward.iter().flatten())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(differing > 0, "the element matrices are order-insensitive");
}

#[test]
fn assembled_values_are_bitwise_identical_for_every_thread_count() {
    let (tasks, ndofs) = grid_tasks(40, 40);
    let assembler = ParallelAssembler::new(ndofs);
    let reference = pool(1).install(|| assembler.assemble(&tasks));
    assert!(reference.nnz() > 10_000);

    for threads in [2, 3, 8] {
        let pool = pool(threads);
        for repeat in 0..4 {
            let matrix = pool.install(|| assembler.assemble(&tasks));
            assert_eq!(matrix.row_offsets, reference.row_offsets);
            assert_eq!(matrix.col_indices, reference.col_indices);
            assert_eq!(
                bits(&matrix.values),
                bits(&reference.values),
                "{threads} threads, repeat {repeat}"
            );
        }
    }
}

#[test]
fn assembly_matches_an_independent_dense_reference() {
    let (tasks, ndofs) = grid_tasks(20, 20);
    let matrix = ParallelAssembler::new(ndofs).assemble(&tasks);
    let (dense, touched) = dense_reference(ndofs, &tasks, true);

    // Pattern: sorted unique columns, exactly the touched entries.
    assert_eq!(matrix.nrows, ndofs);
    assert_eq!(matrix.ncols, ndofs);
    let mut stored = 0;
    for (row, touched_row) in touched.iter().enumerate() {
        let cols = &matrix.col_indices[matrix.row_offsets[row]..matrix.row_offsets[row + 1]];
        assert!(cols.windows(2).all(|w| w[0] < w[1]), "row {row} not sorted");
        for (col, &is_touched) in touched_row.iter().enumerate() {
            assert_eq!(cols.contains(&col), is_touched, "pattern at ({row}, {col})");
        }
        stored += cols.len();
    }
    assert_eq!(stored, matrix.nnz());

    // Values: every term is positive, so the order of addition changes only the last bits.
    for (row, dense_row) in dense.iter().enumerate() {
        for (col, &want) in dense_row.iter().enumerate() {
            if let Some(got) = entry(&matrix, row, col) {
                assert!(
                    (got - want).abs() <= 1e-13 * want,
                    "K[{row}][{col}] = {got}, expected {want}"
                );
            }
        }
    }
}

#[test]
fn more_than_sixty_four_colors_fall_back_to_a_serial_bucket() {
    // 70 elements [0, i + 1] all share DOF 0, so they need 70 colors.
    let tasks: Vec<AssemblyTask> = (0..70)
        .map(|i| {
            let ke: Vec<f64> = (0..4).map(|k| awkward(4 * i + k)).collect();
            AssemblyTask::new(vec![0, i + 1], ke)
        })
        .collect();
    let ndofs = 71;
    let coloring = color_tasks(&tasks);
    assert_eq!(coloring.n_colors, 64);
    assert_eq!(coloring.buckets[63], (63..70).collect::<Vec<_>>());

    let assembler = ParallelAssembler::new(ndofs);
    let reference = pool(1).install(|| assembler.assemble(&tasks));
    let wide = pool(8).install(|| assembler.assemble(&tasks));
    assert_eq!(bits(&reference.values), bits(&wide.values));

    // The fallback bucket is scattered serially in element order, and the other buckets
    // hold one element each: K[0][0] is exactly the sequential sum over the elements.
    let sequential = tasks.iter().fold(0.0_f64, |sum, t| sum + t.ke[0]);
    let k00 = entry(&reference, 0, 0).expect("K[0][0] is stored");
    assert_eq!(k00.to_bits(), sequential.to_bits());
    for (i, task) in tasks.iter().enumerate() {
        assert_eq!(entry(&reference, 0, i + 1), Some(task.ke[1]));
        assert_eq!(entry(&reference, i + 1, 0), Some(task.ke[2]));
        assert_eq!(entry(&reference, i + 1, i + 1), Some(task.ke[3]));
    }
}

#[test]
fn a_coloring_whose_buckets_share_dofs_is_still_assembled_correctly() {
    let (tasks, ndofs) = grid_tasks(10, 10);
    // Everything in one bucket: maximally wrong as a coloring.
    let one_bucket = ElementColoringResult {
        colors: vec![0; tasks.len()],
        n_colors: 1,
        buckets: vec![(0..tasks.len()).collect()],
    };
    for threads in [1, 8] {
        let matrix = pool(threads)
            .install(|| try_assemble_colored_csr(ndofs, &tasks, &one_bucket))
            .expect("assembly succeeds");
        // The conflicting bucket is scattered sequentially in element order, which is
        // exactly the summation order of the dense reference.
        let (dense, touched) = dense_reference(ndofs, &tasks, true);
        for (row, touched_row) in touched.iter().enumerate() {
            for (col, &is_touched) in touched_row.iter().enumerate() {
                if is_touched {
                    let got = entry(&matrix, row, col).expect("entry is stored");
                    assert_eq!(got.to_bits(), dense[row][col].to_bits(), "({row}, {col})");
                }
            }
        }
    }
}

#[test]
fn a_coloring_that_does_not_match_the_tasks_cannot_corrupt_the_result() {
    // `element_dofs` claims that the elements are disjoint although they overlap.
    let (tasks, ndofs) = grid_tasks(8, 8);
    let disjoint_claim: Vec<Vec<usize>> = (0..tasks.len()).map(|e| vec![e]).collect();
    let assembler = ParallelAssembler::new(ndofs.max(tasks.len()));
    let matrix = assembler
        .try_assemble_colored(&tasks, &disjoint_claim)
        .expect("assembly succeeds");
    let (dense, touched) = dense_reference(assembler.ndofs, &tasks, true);
    for (row, touched_row) in touched.iter().enumerate() {
        for (col, &is_touched) in touched_row.iter().enumerate() {
            if is_touched {
                let got = entry(&matrix, row, col).expect("entry is stored");
                assert_eq!(got.to_bits(), dense[row][col].to_bits(), "({row}, {col})");
            }
        }
    }
}

#[test]
fn assemble_and_assemble_colored_agree_bit_for_bit() {
    let (tasks, ndofs) = grid_tasks(16, 16);
    let dofs: Vec<Vec<usize>> = tasks.iter().map(|t| t.global_dofs.clone()).collect();
    let assembler = ParallelAssembler::new(ndofs);
    let default = assembler.assemble(&tasks);
    let colored = assembler.assemble_colored(&tasks, &dofs);
    assert_eq!(default.row_offsets, colored.row_offsets);
    assert_eq!(default.col_indices, colored.col_indices);
    assert_eq!(bits(&default.values), bits(&colored.values));

    let coloring = color_elements(tasks.len(), &dofs);
    let by_tasks = color_tasks(&tasks);
    assert_eq!(coloring.colors, by_tasks.colors);
    assert_eq!(coloring.buckets, by_tasks.buckets);
}

#[test]
fn greedy_coloring_of_a_quad_grid_is_valid_and_small() {
    let (tasks, _) = grid_tasks(30, 30);
    let coloring = color_tasks(&tasks);
    // Four elements around an interior node pairwise share DOFs; an element has at most
    // eight neighbours, so the greedy coloring needs between 4 and 9 colors.
    assert!(
        (4..=9).contains(&coloring.n_colors),
        "{}",
        coloring.n_colors
    );
    let mut listed = 0;
    for bucket in &coloring.buckets {
        let mut claimed = std::collections::HashSet::new();
        for &e in bucket {
            for &dof in &tasks[e].global_dofs {
                assert!(
                    claimed.insert(dof),
                    "two elements of a bucket share DOF {dof}"
                );
            }
            listed += 1;
        }
    }
    assert_eq!(listed, tasks.len());
    for (e, &color) in coloring.colors.iter().enumerate() {
        assert!(coloring.buckets[color].contains(&e));
    }
}

#[test]
fn invalid_input_is_reported_not_swallowed() {
    let ke = vec![1.0, -1.0, -1.0, 1.0];
    let assembler = ParallelAssembler::new(5);

    let out_of_range = vec![AssemblyTask::new(vec![4, 5], ke.clone())];
    assert!(matches!(
        assembler.try_assemble(&out_of_range),
        Err(Error::DofOutOfRange { dof: 5, n: 5 })
    ));

    let wrong_size = vec![AssemblyTask {
        global_dofs: vec![0, 1],
        ke: vec![1.0, 2.0, 3.0],
    }];
    assert!(matches!(
        assembler.try_assemble(&wrong_size),
        Err(Error::InvalidInput(_))
    ));

    let tasks = vec![
        AssemblyTask::new(vec![0, 1], ke.clone()),
        AssemblyTask::new(vec![1, 2], ke.clone()),
        AssemblyTask::new(vec![2, 3], ke),
    ];
    let coloring = color_tasks(&tasks);
    let check = |buckets: Vec<Vec<usize>>| {
        let n_colors = buckets.len();
        let broken = ElementColoringResult {
            colors: Vec::new(),
            n_colors,
            buckets,
        };
        try_assemble_colored_csr(5, &tasks, &broken)
    };
    assert!(matches!(
        check(vec![vec![0, 2]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        check(vec![vec![0, 1, 2], vec![2]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        check(vec![vec![0, 1, 2, 3]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(try_assemble_colored_csr(5, &tasks, &coloring).is_ok());

    // Mismatched element DOF lists are rejected up front.
    assert!(matches!(
        assembler.try_assemble_colored(&tasks, &[vec![0, 1]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        assembler.try_assemble_colored(&tasks, &[vec![0], vec![1], vec![9]]),
        Err(Error::DofOutOfRange { dof: 9, n: 5 })
    ));
}

#[test]
#[should_panic(expected = "ParallelAssembler::assemble")]
fn the_infallible_entry_point_panics_on_a_dof_outside_the_system() {
    let tasks = vec![AssemblyTask::new(vec![0, 7], vec![1.0, 0.0, 0.0, 1.0])];
    let _ = ParallelAssembler::new(3).assemble(&tasks);
}

#[test]
fn empty_input_gives_an_empty_matrix() {
    let matrix = ParallelAssembler::new(4).assemble(&[]);
    assert_eq!(matrix.nrows, 4);
    assert_eq!(matrix.row_offsets, vec![0; 5]);
    assert!(matrix.col_indices.is_empty());
    assert!(matrix.values.is_empty());

    let none = ParallelAssembler::new(0).assemble(&[]);
    assert_eq!(none.row_offsets, vec![0]);
}

#[test]
fn a_dof_listed_twice_in_one_element_is_summed_in_local_order() {
    // K[0][0] receives all four local entries, in the order ke[0], ke[1], ke[2], ke[3].
    let ke = vec![awkward(1), awkward(2), awkward(3), awkward(4)];
    let task = AssemblyTask::new(vec![0, 0], ke.clone());
    let matrix = ParallelAssembler::new(1).assemble(&[task]);
    let expected = ke.iter().fold(0.0_f64, |sum, &v| sum + v);
    assert_eq!(matrix.values.len(), 1);
    assert_eq!(matrix.values[0].to_bits(), expected.to_bits());
}

/// Per-element load vectors on the grid of [`grid_tasks`].
fn grid_loads(tasks: &[AssemblyTask]) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let dofs: Vec<Vec<usize>> = tasks.iter().map(|t| t.global_dofs.clone()).collect();
    let forces: Vec<Vec<f64>> = dofs
        .iter()
        .enumerate()
        .map(|(e, d)| (0..d.len()).map(|k| awkward(1000 + 8 * e + k)).collect())
        .collect();
    (dofs, forces)
}

#[test]
fn rhs_equals_the_sequential_element_loop_for_every_thread_count() {
    let (tasks, ndofs) = grid_tasks(30, 30);
    let (dofs, forces) = grid_loads(&tasks);
    let assembler = ParallelAssembler::new(ndofs);

    let mut sequential = vec![0.0_f64; ndofs];
    for (d, f) in dofs.iter().zip(&forces) {
        for (&dof, &force) in d.iter().zip(f) {
            sequential[dof] += force;
        }
    }
    for threads in [1, 2, 8] {
        let rhs = pool(threads).install(|| assembler.assemble_rhs(&dofs, &forces));
        assert_eq!(bits(&rhs), bits(&sequential), "{threads} threads");
    }
}

#[test]
fn rhs_reports_inconsistent_input() {
    let assembler = ParallelAssembler::new(3);
    assert!(matches!(
        assembler.try_assemble_rhs(&[vec![0, 1]], &[]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        assembler.try_assemble_rhs(&[vec![0, 1]], &[vec![1.0]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        assembler.try_assemble_rhs(&[vec![0, 3]], &[vec![1.0, 2.0]]),
        Err(Error::DofOutOfRange { dof: 3, n: 3 })
    ));
    let rhs = assembler
        .try_assemble_rhs(&[vec![0, 1], vec![1, 2]], &[vec![1.0, 2.0], vec![3.0, 4.0]])
        .expect("consistent input");
    assert_eq!(rhs, vec![1.0, 5.0, 4.0]);
}
