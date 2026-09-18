//! Regression tests for [`rank_of_matrix`], which eliminates over the prime
//! field `GF(2^61 - 1)` instead of the overflowing division-free integer
//! elimination.

use super::test_support::{complex_from_simplices, tetrahedron_surface, torus_surface};
use super::{
    CellularHomology, CwCell, CwComplex, RANK_MODULUS, entry_residue, inv_mod, mul_mod,
    rank_of_matrix, smith_normal_form,
};

/// xorshift64 generator for deterministic pseudo-random matrices.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform integer in `0..bound`.
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Reference rank by fraction-free (Bareiss) elimination over `i128`, exact as
/// long as no minor leaves the `i128` range (true for the small matrices below).
fn bareiss_rank(matrix: &[Vec<i32>]) -> usize {
    let mut a: Vec<Vec<i128>> = matrix
        .iter()
        .map(|row| row.iter().map(|&x| i128::from(x)).collect())
        .collect();
    let nrows = a.len();
    let ncols = a.iter().map(Vec::len).max().unwrap_or(0);
    let mut rank = 0;
    let mut previous_pivot: i128 = 1;
    for col in 0..ncols {
        if rank == nrows {
            break;
        }
        let Some(pivot_row) = (rank..nrows).find(|&r| a[r][col] != 0) else {
            continue;
        };
        a.swap(rank, pivot_row);
        let pivot = a[rank][col];
        for i in (rank + 1)..nrows {
            for j in (col + 1)..ncols {
                a[i][j] = (a[i][j] * pivot - a[i][col] * a[rank][j]) / previous_pivot;
            }
            a[i][col] = 0;
        }
        previous_pivot = pivot;
        rank += 1;
    }
    rank
}

/// The 10 x 10 matrix of the audit: `x0 = 42`, `x = (1103515245 x + 12345) mod
/// 2^31`, entry `((x >> 16) % 3) - 1`, row-major.  The division-free `i32`
/// elimination reached 154-bit entries on it (a panic in debug builds, a wrong
/// rank in release builds).
fn lcg_matrix_10x10() -> Vec<Vec<i32>> {
    let mut x: u64 = 42;
    (0..10)
        .map(|_| {
            (0..10)
                .map(|_| {
                    x = (1_103_515_245 * x + 12_345) % (1 << 31);
                    ((x >> 16) % 3) as i32 - 1
                })
                .collect()
        })
        .collect()
}

#[test]
fn audit_matrix_has_full_rank_without_overflow() {
    let matrix = lcg_matrix_10x10();
    assert_eq!(matrix[0], vec![0, 1, 1, -1, 1, 0, -1, 0, 0, 0]);
    assert_eq!(rank_of_matrix(&matrix), 10);
    assert_eq!(bareiss_rank(&matrix), 10);
}

#[test]
fn rank_matches_exact_reference_on_random_small_matrices() {
    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    let mut deficient = 0;
    for case in 0..400 {
        let nrows = 1 + rng.below(10) as usize;
        let ncols = 1 + rng.below(10) as usize;
        let mut matrix: Vec<Vec<i32>> = (0..nrows)
            .map(|_| (0..ncols).map(|_| rng.below(3) as i32 - 1).collect())
            .collect();
        // Make many of them rank-deficient: zero rows and rows that are
        // combinations of earlier ones.
        for r in 1..nrows {
            match rng.below(5) {
                0 => matrix[r] = vec![0; ncols],
                1 => {
                    let (a, b) = (rng.below(r as u64) as usize, rng.below(r as u64) as usize);
                    matrix[r] = (0..ncols).map(|c| matrix[a][c] - matrix[b][c]).collect();
                }
                _ => {}
            }
        }
        let want = bareiss_rank(&matrix);
        if want < nrows.min(ncols) {
            deficient += 1;
        }
        assert_eq!(rank_of_matrix(&matrix), want, "case {case}: {matrix:?}");
    }
    assert!(
        deficient > 50,
        "the sample must contain rank-deficient matrices"
    );
}

#[test]
fn entries_near_the_i32_limits_do_not_overflow() {
    // The old elimination multiplied entries in i32 and panicked (debug) or
    // wrapped (release) on products like i32::MAX * i32::MAX.
    let full = vec![vec![i32::MAX, 1], vec![1, i32::MAX]];
    assert_eq!(rank_of_matrix(&full), 2);
    let extreme = vec![vec![i32::MIN, i32::MAX], vec![i32::MAX, i32::MIN]];
    assert_eq!(rank_of_matrix(&extreme), 2);
    // Rows 3 = 1 - 2 keeps the rank at 2 although every entry is huge.
    let r1 = vec![1 << 30, 5, -7];
    let r2 = vec![1 << 30, 3, 1 << 30];
    let r3: Vec<i32> = r1.iter().zip(r2.iter()).map(|(a, b)| a - b).collect();
    assert_eq!(rank_of_matrix(&[r1, r2, r3]), 2);
    // Proportional rows with large entries.
    let half = vec![1 << 29, 3, -4];
    let doubled: Vec<i32> = half.iter().map(|x| 2 * x).collect();
    assert_eq!(rank_of_matrix(&[half, doubled]), 1);
    // A 30 x 30 matrix of large entries: a diagonal one is trivially full rank,
    // and the same with the last row equal to the first drops the rank by one.
    let mut diagonal: Vec<Vec<i32>> = (0..30)
        .map(|i| {
            let mut row = vec![0; 30];
            row[i] = i32::MAX - i as i32;
            row
        })
        .collect();
    assert_eq!(rank_of_matrix(&diagonal), 30);
    diagonal[29] = diagonal[0].clone();
    assert_eq!(rank_of_matrix(&diagonal), 29);
}

#[test]
fn empty_ragged_and_zero_matrices() {
    assert_eq!(rank_of_matrix(&[]), 0);
    assert_eq!(rank_of_matrix(&[vec![]]), 0);
    assert_eq!(rank_of_matrix(&[vec![], vec![]]), 0);
    assert_eq!(rank_of_matrix(&[vec![0, 0, 0], vec![0, 0, 0]]), 0);
    // Short rows are padded with zeros: [[1, 2], [1, 0]] has determinant -2.
    assert_eq!(rank_of_matrix(&[vec![1, 2], vec![1]]), 2);
    assert_eq!(rank_of_matrix(&[vec![0], vec![0, 0, 5]]), 1);
    // More rows than columns and vice versa.
    assert_eq!(rank_of_matrix(&[vec![1, 0], vec![0, 1], vec![1, 1]]), 2);
    assert_eq!(rank_of_matrix(&[vec![1, 2, 3, 4], vec![2, 4, 6, 8]]), 1);
}

#[test]
fn field_arithmetic_matches_wide_integer_arithmetic() {
    let p = u128::from(RANK_MODULUS);
    let mut rng = Xorshift(0xDEAD_BEEF_0BAD_F00D);
    let edge = [
        0,
        1,
        2,
        RANK_MODULUS - 2,
        RANK_MODULUS - 1,
        1 << 60,
        (1 << 60) + 1,
    ];
    let mut values: Vec<u64> = edge.to_vec();
    values.extend((0..200).map(|_| rng.next_u64() % RANK_MODULUS));
    for &a in &values {
        for &b in values.iter().take(40) {
            let want = (u128::from(a) * u128::from(b) % p) as u64;
            assert_eq!(mul_mod(a, b), want, "{a} * {b}");
        }
        if a != 0 {
            assert_eq!(mul_mod(a, inv_mod(a)), 1, "inverse of {a}");
        }
    }
    // Signed entries: negative values map to p - |x|.
    assert_eq!(entry_residue(0), 0);
    assert_eq!(entry_residue(7), 7);
    assert_eq!(entry_residue(-1), RANK_MODULUS - 1);
    assert_eq!(entry_residue(i32::MAX), u64::from(i32::MAX.unsigned_abs()));
    assert_eq!(entry_residue(i32::MIN), RANK_MODULUS - (1 << 31));
}

#[test]
fn ranks_of_the_boundary_matrices_of_a_tetrahedron_surface() {
    let sphere = complex_from_simplices(&tetrahedron_surface());
    // ∂_1 : 6 edges -> 4 vertices has rank 3, ∂_2 : 4 triangles -> 6 edges rank 3.
    assert_eq!(rank_of_matrix(&sphere.boundary_matrix(1)), 3);
    assert_eq!(rank_of_matrix(&sphere.boundary_matrix(2)), 3);
    let homology = CellularHomology::compute(&sphere);
    assert_eq!(homology.betti, vec![1, 0, 1]);
}

#[test]
fn homology_of_the_standard_models_uses_the_modular_rank() {
    let torus = CwComplex::standard_torus();
    assert_eq!(CellularHomology::compute(&torus).betti, vec![1, 2, 1]);
    let ball = CwComplex::standard_tetrahedron();
    assert_eq!(CellularHomology::compute(&ball).betti, vec![1, 0, 0, 0]);
    // The projective plane: ∂_2 = [2], no free part in H_1, torsion Z/2.
    let mut plane = CwComplex::new();
    plane.add_cell(CwCell::vertex(0, [0.0; 3]));
    plane.add_cell(CwCell {
        id: 0,
        dim: 1,
        label: "a".into(),
        coords: None,
        boundary: vec![0, 0],
        boundary_signs: vec![1, -1],
    });
    plane.add_cell(CwCell::face(0, vec![0, 0], vec![1, 1]));
    assert_eq!(plane.boundary_matrix(2), vec![vec![2]]);
    assert_eq!(rank_of_matrix(&plane.boundary_matrix(2)), 1);
    let homology = CellularHomology::compute(&plane);
    assert_eq!(homology.betti, vec![1, 0, 0]);
    assert_eq!(smith_normal_form(&plane.boundary_matrix(2)), vec![2]);
}

#[test]
fn ranks_of_a_triangulated_torus_give_its_betti_numbers() {
    // 20 x 20 grid: 400 vertices, 1200 edges, 800 triangles.
    let torus = complex_from_simplices(&torus_surface(20));
    assert_eq!(torus.count(0), 400);
    assert_eq!(torus.count(1), 1200);
    assert_eq!(torus.count(2), 800);
    let rank_1 = rank_of_matrix(&torus.boundary_matrix(1));
    let rank_2 = rank_of_matrix(&torus.boundary_matrix(2));
    // Connected: rank ∂_1 = V - 1; one fundamental class: rank ∂_2 = F - 1.
    assert_eq!(rank_1, 399);
    assert_eq!(rank_2, 799);
    let betti_1 = 1200 - rank_1 - rank_2;
    assert_eq!(betti_1, 2);
}
