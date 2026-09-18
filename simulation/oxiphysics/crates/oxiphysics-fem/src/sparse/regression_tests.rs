//! Regression tests for the hash-free triplet compression behind
//! [`CsrMatrix::from_triplets`], [`CscMatrix::from_triplets`] and
//! [`BlockCsrMatrix3::from_block_triplets`].
//!
//! Every global assembly funnels through these constructors (about 144
//! triplets per tetrahedron).  They used to accumulate in a `HashMap` keyed by
//! `(row, col)` and sort the result; the counting-sort rewrite must produce the
//! identical matrix, including the order in which duplicates are summed.

use super::*;
use crate::assembly::CooMatrix;
use std::collections::BTreeMap;

/// Deterministic generator (64-bit LCG, high bits).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn value(&mut self) -> f64 {
        (self.next() % 2001) as f64 / 100.0 - 10.0
    }
}

/// Unordered triplets with many duplicate positions and several empty rows.
fn scattered_triplets(
    seed: u64,
    count: usize,
    nrows: usize,
    ncols: usize,
) -> Vec<(usize, usize, f64)> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|_| {
            // Rows divisible by 7 receive no entry at all.
            let row = (rng.below(nrows / 2) * 2 + 1) % nrows;
            let row = if row.is_multiple_of(7) {
                (row + 1) % nrows
            } else {
                row
            };
            (row, rng.below(ncols), rng.value())
        })
        .collect()
}

/// Reference: ordered map accumulating in input order.
fn reference_map(triplets: &[(usize, usize, f64)]) -> BTreeMap<(usize, usize), f64> {
    let mut map = BTreeMap::new();
    for &(r, c, v) in triplets {
        *map.entry((r, c)).or_insert(0.0) += v;
    }
    map
}

fn assert_csr_matches(csr: &CsrMatrix, reference: &BTreeMap<(usize, usize), f64>) {
    assert_eq!(csr.row_ptr.len(), csr.nrows + 1);
    assert_eq!(csr.row_ptr[0], 0);
    assert_eq!(csr.row_ptr[csr.nrows], csr.nnz());
    assert_eq!(csr.col_indices.len(), csr.values.len());
    assert_eq!(csr.nnz(), reference.len());
    let mut expected = reference.iter();
    for row in 0..csr.nrows {
        assert!(csr.row_ptr[row] <= csr.row_ptr[row + 1], "row_ptr monotone");
        let cols = &csr.col_indices[csr.row_ptr[row]..csr.row_ptr[row + 1]];
        assert!(
            cols.windows(2).all(|w| w[0] < w[1]),
            "columns of row {row} must be strictly increasing"
        );
        for idx in csr.row_ptr[row]..csr.row_ptr[row + 1] {
            let (&(r, c), &v) = expected.next().expect("more stored entries than reference");
            assert_eq!((r, c), (row, csr.col_indices[idx]));
            assert_eq!(csr.values[idx], v, "entry ({r},{c})");
        }
    }
    assert!(expected.next().is_none());
}

#[test]
fn csr_from_unsorted_triplets_matches_the_ordered_map_reference() {
    let (nrows, ncols) = (60, 45);
    for (seed, count) in [(1, 0), (2, 1), (3, 25), (4, 400), (5, 5000)] {
        let triplets = scattered_triplets(seed, count, nrows, ncols);
        let csr = CsrMatrix::from_triplets(nrows, ncols, &triplets);
        assert_csr_matches(&csr, &reference_map(&triplets));
        if count >= 400 {
            // Rows 7, 14, ... are empty by construction.
            let empty = (0..nrows)
                .filter(|&r| csr.row_ptr[r] == csr.row_ptr[r + 1])
                .count();
            assert!(empty >= 8, "expected empty rows, got {empty}");
        }
    }
}

#[test]
fn csc_from_unsorted_triplets_matches_the_ordered_map_reference() {
    let (nrows, ncols) = (50, 70);
    for (seed, count) in [(11, 0), (12, 3), (13, 300), (14, 6000)] {
        let triplets = scattered_triplets(seed, count, nrows, ncols);
        let csc = CscMatrix::from_triplets(nrows, ncols, &triplets);
        let reference = reference_map(&triplets);
        assert_eq!(csc.col_ptr.len(), ncols + 1);
        assert_eq!(csc.col_ptr[ncols], csc.nnz());
        assert_eq!(csc.nnz(), reference.len());
        // Walk the columns: rows strictly increasing, values equal to the map.
        let mut seen = 0;
        for col in 0..ncols {
            let rows = &csc.row_indices[csc.col_ptr[col]..csc.col_ptr[col + 1]];
            assert!(rows.windows(2).all(|w| w[0] < w[1]), "column {col}");
            for (offset, &row) in rows.iter().enumerate() {
                let stored = csc.values[csc.col_ptr[col] + offset];
                assert_eq!(Some(&stored), reference.get(&(row, col)), "({row},{col})");
                seen += 1;
            }
        }
        assert_eq!(seen, reference.len());
    }
}

#[test]
fn degenerate_dimensions_and_empty_input_are_handled() {
    let empty = CsrMatrix::from_triplets(0, 0, &[]);
    assert_eq!(empty.row_ptr, vec![0]);
    assert_eq!(empty.nnz(), 0);
    let no_rows = CsrMatrix::from_triplets(0, 5, &[]);
    assert_eq!(no_rows.row_ptr, vec![0]);
    let no_cols = CsrMatrix::from_triplets(4, 0, &[]);
    assert_eq!(no_cols.row_ptr, vec![0; 5]);
    let csc = CscMatrix::from_triplets(0, 3, &[]);
    assert_eq!(csc.col_ptr, vec![0; 4]);
    let csc = CscMatrix::from_triplets(3, 0, &[]);
    assert_eq!(csc.col_ptr, vec![0]);
    let blocks = BlockCsrMatrix3::from_block_triplets(0, 0, &[]);
    assert_eq!(blocks.row_ptr, vec![0]);
    // A single entry in the last row and column.
    let single = CsrMatrix::from_triplets(3, 3, &[(2, 2, 4.5)]);
    assert_eq!(single.row_ptr, vec![0, 0, 0, 1]);
    assert_eq!(single.get(2, 2), 4.5);
}

#[test]
fn duplicates_are_summed_in_input_order_and_zeros_stay_stored() {
    // 1e16 + 1 rounds back to 1e16 in f64, so the two orders differ: the sum
    // follows the order of appearance, exactly like the old hash accumulation.
    let first = CsrMatrix::from_triplets(1, 1, &[(0, 0, 1e16), (0, 0, 1.0), (0, 0, -1e16)]);
    assert_eq!(first.get(0, 0), 0.0);
    let second = CsrMatrix::from_triplets(1, 1, &[(0, 0, 1e16), (0, 0, -1e16), (0, 0, 1.0)]);
    assert_eq!(second.get(0, 0), 1.0);
    let csc = CscMatrix::from_triplets(1, 1, &[(0, 0, 1e16), (0, 0, -1e16), (0, 0, 1.0)]);
    assert_eq!(csc.get(0, 0), 1.0);
    // Cancelling duplicates keep an explicit zero (nnz counts it).
    let cancelled = CsrMatrix::from_triplets(2, 2, &[(0, 1, 2.0), (0, 1, -2.0), (1, 0, 3.0)]);
    assert_eq!(cancelled.nnz(), 2);
    assert_eq!(cancelled.get(0, 1), 0.0);
    // Interleaved duplicates in different rows stay attached to their own entry.
    let mixed = CsrMatrix::from_triplets(
        2,
        3,
        &[
            (1, 2, 1.0),
            (0, 0, 2.0),
            (1, 2, 3.0),
            (0, 0, 4.0),
            (1, 0, 5.0),
        ],
    );
    assert_eq!(mixed.row_ptr, vec![0, 1, 3]);
    assert_eq!(mixed.col_indices, vec![0, 0, 2]);
    assert_eq!(mixed.values, vec![6.0, 5.0, 4.0]);
}

#[test]
fn already_sorted_and_reverse_sorted_rows_give_the_same_matrix() {
    let n: usize = 40;
    let ascending: Vec<(usize, usize, f64)> = (0..n)
        .flat_map(|r| {
            (0..n)
                .filter(move |c| (r + c).is_multiple_of(3))
                .map(move |c| (r, c, (r * n + c) as f64))
        })
        .collect();
    let mut descending = ascending.clone();
    descending.reverse();
    let a = CsrMatrix::from_triplets(n, n, &ascending);
    let b = CsrMatrix::from_triplets(n, n, &descending);
    assert_eq!(a.row_ptr, b.row_ptr);
    assert_eq!(a.col_indices, b.col_indices);
    assert_eq!(a.values, b.values);
}

#[test]
fn conversions_round_trip_through_the_new_constructors() {
    let (nrows, ncols) = (35, 28);
    let triplets = scattered_triplets(77, 900, nrows, ncols);
    let csr = CsrMatrix::from_triplets(nrows, ncols, &triplets);
    let back = csr.transpose().transpose();
    assert_eq!(csr.row_ptr, back.row_ptr);
    assert_eq!(csr.col_indices, back.col_indices);
    assert_eq!(csr.values, back.values);
    let via_csc = csr.to_csc().to_csr();
    assert_eq!(csr.row_ptr, via_csc.row_ptr);
    assert_eq!(csr.col_indices, via_csc.col_indices);
    assert_eq!(csr.values, via_csc.values);
    // The transpose indexes the same values.
    let transposed = csr.transpose();
    for &(r, c, _) in &triplets {
        assert_eq!(transposed.get(c, r), csr.get(r, c));
    }
}

#[test]
fn coo_matrix_converts_without_touching_its_entries() {
    let mut coo = CooMatrix::new(6, 6);
    let mut rng = Lcg(5);
    for _ in 0..500 {
        coo.add(rng.below(6), rng.below(6), rng.value());
    }
    let before = coo.entries.clone();
    let csr = coo.to_csr();
    assert_eq!(coo.entries, before, "to_csr must not modify the triplets");
    assert_csr_matches(&csr, &reference_map(&before));
    // Converting twice gives the same matrix.
    let again = coo.to_csr();
    assert_eq!(csr.values, again.values);
}

#[test]
fn block_triplets_sum_duplicate_blocks_elementwise() {
    let block = |s: f64| -> [f64; 9] { std::array::from_fn(|k| s * (k as f64 + 1.0)) };
    let triplets = vec![
        (1, 1, block(1.0)),
        (0, 2, block(2.0)),
        (1, 1, block(10.0)),
        (0, 0, block(3.0)),
        (0, 2, block(-2.0)),
    ];
    let m = BlockCsrMatrix3::from_block_triplets(2, 3, &triplets);
    // Row 0 holds blocks at columns 0 and 2 (the cancelled one stays stored as an
    // explicit zero block), row 1 the summed block at column 1.
    assert_eq!(m.row_ptr, vec![0, 2, 3]);
    assert_eq!(m.col_indices, vec![0, 2, 1]);
    assert_eq!(m.n_blocks(), 3);
    assert_eq!(m.get_block(0, 0), block(3.0));
    assert_eq!(m.get_block(0, 2), [0.0; 9], "cancelling blocks sum to zero");
    assert_eq!(m.get_block(1, 1), block(11.0));
    assert_eq!(m.get_block(1, 2), [0.0; 9]);
    // The scalar expansion agrees with the scalar constructor.
    let mut scalar = Vec::new();
    for &(br, bc, blk) in &triplets {
        for i in 0..3 {
            for j in 0..3 {
                scalar.push((3 * br + i, 3 * bc + j, blk[3 * i + j]));
            }
        }
    }
    let expected = CsrMatrix::from_triplets(6, 9, &scalar);
    let expanded = m.to_scalar_csr();
    for r in 0..6 {
        for c in 0..9 {
            assert_eq!(expanded.get(r, c), expected.get(r, c), "({r},{c})");
        }
    }
}

#[test]
#[should_panic(expected = "row index 3 out of bounds for 3 rows")]
fn csr_rejects_a_row_index_out_of_range() {
    let _ = CsrMatrix::from_triplets(3, 3, &[(0, 0, 1.0), (3, 0, 1.0)]);
}

#[test]
#[should_panic(expected = "col index 4 out of bounds for 4 cols")]
fn csr_rejects_a_column_index_out_of_range() {
    let _ = CsrMatrix::from_triplets(3, 4, &[(0, 4, 1.0)]);
}

#[test]
#[should_panic(expected = "row index 2 out of bounds for 2 rows")]
fn csc_rejects_a_row_index_out_of_range() {
    let _ = CscMatrix::from_triplets(2, 5, &[(2, 1, 1.0)]);
}

#[test]
#[should_panic(expected = "col index 5 out of bounds for 5 cols")]
fn csc_rejects_a_column_index_out_of_range() {
    let _ = CscMatrix::from_triplets(2, 5, &[(0, 5, 1.0)]);
}

#[test]
#[should_panic(expected = "block col index 3 out of bounds for 3 block cols")]
fn block_triplets_reject_an_out_of_range_block_column() {
    let _ = BlockCsrMatrix3::from_block_triplets(2, 3, &[(0, 3, [0.0; 9])]);
}
