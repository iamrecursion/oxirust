//! Regression tests for the periodic linked-cell list, the variable-`h`
//! neighbour search and the octree search: duplicate-free periodic queries in
//! boxes only one or two cells wide, symmetric gather–scatter neighbour
//! relations, and a build that terminates for coincident, clustered and
//! non-finite positions.

use super::types::{AdaptiveNeighborSearch, OctreeNeighborSearch, OctreeNode, PbcLinkedCellList};

/// Deterministic pseudo-random number in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

/// `n` pseudo-random points in the axis-aligned box `[0, extent]`.
fn random_points(n: usize, extent: [f64; 3], seed: u64) -> Vec<[f64; 3]> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            [
                extent[0] * lcg_unit(&mut state),
                extent[1] * lcg_unit(&mut state),
                extent[2] * lcg_unit(&mut state),
            ]
        })
        .collect()
}

/// Brute-force neighbour lists under the minimum-image convention.
fn brute_force_periodic(positions: &[[f64; 3]], box_size: [f64; 3], r_cut: f64) -> Vec<Vec<usize>> {
    (0..positions.len())
        .map(|i| {
            (0..positions.len())
                .filter(|&j| {
                    let mut d2 = 0.0;
                    for k in 0..3 {
                        let mut d = positions[i][k] - positions[j][k];
                        d -= (d / box_size[k]).round() * box_size[k];
                        d2 += d * d;
                    }
                    j != i && d2 <= r_cut * r_cut
                })
                .collect()
        })
        .collect()
}

/// `true` when `list` is strictly increasing, i.e. sorted without duplicates.
fn is_strictly_increasing(list: &[usize]) -> bool {
    list.windows(2).all(|w| w[0] < w[1])
}

// ---------------------------------------------------------------------------
// PbcLinkedCellList
// ---------------------------------------------------------------------------

#[test]
fn pbc_query_has_no_duplicates_in_a_thin_slab() {
    // 1 × 1 × 0.2 box with cell size 0.3: three cells along x and y but only
    // one along z, where all three offsets used to hit the same cell.
    let box_size = [1.0, 1.0, 0.2];
    let r_cut = 0.3;
    let positions = random_points(10, box_size, 11);
    let list = PbcLinkedCellList::build(&positions, r_cut, box_size);
    assert_eq!(list.n_cells, [3, 3, 1]);
    let expected = brute_force_periodic(&positions, box_size, r_cut);
    let mut total_pairs = 0;
    for (i, want) in expected.iter().enumerate() {
        let got = list.query(i, r_cut);
        assert!(
            is_strictly_increasing(&got),
            "particle {i}: duplicate or unsorted neighbours {got:?}"
        );
        assert_eq!(&got, want, "particle {i}");
        total_pairs += got.len();
    }
    assert!(
        total_pairs > 0,
        "the slab test must contain neighbour pairs"
    );
}

#[test]
fn pbc_query_has_no_duplicates_when_an_axis_has_two_cells() {
    // 0.7 / 0.3 = 2 cells per axis: the offsets -1 and +1 wrap onto the same
    // cell, so every particle of the other cell used to be listed twice.
    let box_size = [0.7, 0.7, 0.7];
    let r_cut = 0.3;
    let positions = random_points(40, box_size, 12);
    let list = PbcLinkedCellList::build(&positions, r_cut, box_size);
    assert_eq!(list.n_cells, [2, 2, 2]);
    let expected = brute_force_periodic(&positions, box_size, r_cut);
    for (i, want) in expected.iter().enumerate() {
        let got = list.query(i, r_cut);
        assert!(
            is_strictly_increasing(&got),
            "particle {i}: duplicate or unsorted neighbours {got:?}"
        );
        assert_eq!(&got, want, "particle {i}");
    }
}

#[test]
fn pbc_query_mixed_one_two_and_many_cells_per_axis() {
    // 0.25 / 0.8 / 1.7 with cell size 0.3 give 1 / 2 / 5 cells.
    let box_size = [0.25, 0.8, 1.7];
    let r_cut = 0.3;
    let positions = random_points(60, box_size, 13);
    let nls = PbcLinkedCellList::all_neighbors(&positions, r_cut, box_size);
    let expected = brute_force_periodic(&positions, box_size, r_cut);
    assert_eq!(nls, expected);
    for list in &nls {
        assert!(is_strictly_increasing(list));
    }
}

#[test]
fn pbc_query_single_cell_box_lists_every_particle_once() {
    // The whole box is one cell and every minimum-image distance (at most
    // sqrt(3) * 0.125) is below the cut-off: all other particles, once each.
    let box_size = [0.25, 0.25, 0.25];
    let positions = random_points(12, box_size, 14);
    let nls = PbcLinkedCellList::all_neighbors(&positions, 0.3, box_size);
    for (i, list) in nls.iter().enumerate() {
        let everyone_else: Vec<usize> = (0..positions.len()).filter(|&j| j != i).collect();
        assert_eq!(list, &everyone_else, "particle {i}");
    }
}

#[test]
fn pbc_query_matches_brute_force_in_a_regular_box_with_unwrapped_positions() {
    // Positions outside [0, L) are wrapped into the box before binning.
    let box_size = [2.0, 1.5, 1.0];
    let r_cut = 0.35;
    let mut positions = random_points(120, box_size, 15);
    for (n, p) in positions.iter_mut().enumerate() {
        if n % 3 == 0 {
            p[0] += 2.0 * box_size[0];
        }
        if n % 4 == 0 {
            p[1] -= box_size[1];
        }
        if n % 5 == 0 {
            p[2] -= 3.0 * box_size[2];
        }
    }
    let nls = PbcLinkedCellList::all_neighbors(&positions, r_cut, box_size);
    assert_eq!(nls, brute_force_periodic(&positions, box_size, r_cut));
}

#[test]
fn pbc_query_honours_a_cutoff_larger_than_the_cell_size() {
    // Cells of 0.1 but a query cut-off of 0.25: the window must span three
    // cells on each side, not one.
    let box_size = [1.0, 1.0, 1.0];
    let positions = random_points(150, box_size, 16);
    let list = PbcLinkedCellList::build(&positions, 0.1, box_size);
    assert_eq!(list.n_cells, [10, 10, 10]);
    let r_cut = 0.25;
    let expected = brute_force_periodic(&positions, box_size, r_cut);
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(&list.query(i, r_cut), want, "particle {i}");
    }
    // A cut-off that spans several box lengths terminates and lists everyone.
    let everyone = list.query(0, 100.0);
    assert_eq!(everyone, (1..positions.len()).collect::<Vec<usize>>());
}

// ---------------------------------------------------------------------------
// AdaptiveNeighborSearch
// ---------------------------------------------------------------------------

#[test]
fn adaptive_search_finds_neighbors_covered_only_by_the_larger_h() {
    // Particle 1 reaches particle 0 (0.45 <= 0.5) but particle 0 (h = 0.1)
    // does not reach particle 1: the pair still belongs to both lists.
    let positions = vec![[0.05, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let h = vec![0.1, 0.5];
    let search = AdaptiveNeighborSearch::build(&positions, &h);
    assert_eq!(search.query(0), vec![1]);
    assert_eq!(search.query(1), vec![0]);
}

#[test]
fn adaptive_search_is_symmetric_and_matches_brute_force() {
    let n = 200;
    let positions = random_points(n, [1.0, 1.0, 1.0], 21);
    let mut state = 22_u64;
    let h: Vec<f64> = (0..n).map(|_| 0.03 + 0.12 * lcg_unit(&mut state)).collect();
    let search = AdaptiveNeighborSearch::build(&positions, &h);
    let lists = search.all_neighbors();
    let mut pairs = 0;
    for i in 0..n {
        let expected: Vec<usize> = (0..n)
            .filter(|&j| {
                let d2: f64 = (0..3)
                    .map(|k| (positions[i][k] - positions[j][k]).powi(2))
                    .sum();
                let h_max = h[i].max(h[j]);
                j != i && d2 <= h_max * h_max
            })
            .collect();
        assert_eq!(lists[i], expected, "neighbours of particle {i}");
        for &j in &lists[i] {
            assert!(
                lists[j].contains(&i),
                "{j} is a neighbour of {i} but not vice versa"
            );
            pairs += 1;
        }
    }
    assert!(pairs > n, "the test needs a non-trivial neighbour graph");
}

#[test]
fn adaptive_search_with_a_strongly_varying_h_loses_no_pair() {
    // One particle with a huge support among many tiny ones: it must see all
    // particles within its h, and each of those must see it back.
    let mut positions = random_points(80, [1.0, 1.0, 1.0], 23);
    positions.push([0.5, 0.5, 0.5]);
    let mut h = vec![0.02; 80];
    h.push(0.4);
    let search = AdaptiveNeighborSearch::build(&positions, &h);
    let big = search.query(80);
    let expected: Vec<usize> = (0..80)
        .filter(|&j| {
            let d2: f64 = (0..3).map(|k| (positions[j][k] - 0.5).powi(2)).sum();
            d2 <= 0.4 * 0.4
        })
        .collect();
    assert!(expected.len() > 5);
    assert_eq!(big, expected);
    for &j in &expected {
        assert!(
            search.query(j).contains(&80),
            "particle {j} must see the big one"
        );
    }
}

// ---------------------------------------------------------------------------
// OctreeNeighborSearch
// ---------------------------------------------------------------------------

/// Maximum depth, maximum leaf population and total number of stored indices
/// of the subtree below `node` (`depth` is the depth of `node` itself).
fn tree_stats(node: &OctreeNode, depth: usize) -> (usize, usize, usize) {
    match node {
        OctreeNode::Leaf { indices, .. } => (depth, indices.len(), indices.len()),
        OctreeNode::Internal { children, .. } => {
            children
                .iter()
                .fold((depth, 0, 0), |(max_depth, max_leaf, total), child| {
                    let (d, l, t) = tree_stats(child, depth + 1);
                    (max_depth.max(d), max_leaf.max(l), total + t)
                })
        }
    }
}

fn brute_force_within(positions: &[[f64; 3]], h: f64) -> Vec<Vec<usize>> {
    (0..positions.len())
        .map(|i| {
            (0..positions.len())
                .filter(|&j| {
                    let d2: f64 = (0..3)
                        .map(|k| (positions[i][k] - positions[j][k]).powi(2))
                        .sum();
                    j != i && d2 <= h * h
                })
                .collect()
        })
        .collect()
}

#[test]
fn octree_build_terminates_for_coincident_particles() {
    let positions = vec![[0.0, 0.0, 0.0]; 20];
    let octree = OctreeNeighborSearch::build(&positions, 0.1);
    let mut found = octree.query([0.0, 0.0, 0.0]);
    found.sort_unstable();
    assert_eq!(found, (0..20).collect::<Vec<usize>>());
    let all = octree.all_neighbors();
    for (i, list) in all.iter().enumerate() {
        let others: Vec<usize> = (0..20).filter(|&j| j != i).collect();
        assert_eq!(list, &others);
    }
    // A query away from the stack finds nothing.
    assert!(octree.query([1.0, 1.0, 1.0]).is_empty());
}

#[test]
fn octree_with_coincident_groups_and_scattered_particles_matches_brute_force() {
    let mut positions = random_points(100, [1.0, 1.0, 1.0], 31);
    positions.extend(std::iter::repeat_n([0.5, 0.5, 0.5], 30));
    positions.extend(std::iter::repeat_n([0.2, 0.9, 0.1], 12));
    let h = 0.12;
    let octree = OctreeNeighborSearch::build(&positions, h);
    assert_eq!(octree.all_neighbors(), brute_force_within(&positions, h));
}

#[test]
fn octree_separates_a_dense_cluster_instead_of_building_one_giant_leaf() {
    // 300 distinct points in a 1e-6 cube plus a far outlier: the first levels
    // put the whole cluster into one octant, yet it must still be subdivided
    // into leaves of at most eight particles.
    let mut positions: Vec<[f64; 3]> = random_points(300, [1.0e-6, 1.0e-6, 1.0e-6], 32);
    positions.push([1.0, 1.0, 1.0]);
    let h = 2.0e-7;
    let octree = OctreeNeighborSearch::build(&positions, h);
    let root = octree.root.as_deref().expect("non-empty tree");
    let (depth, max_leaf, total) = tree_stats(root, 0);
    assert_eq!(
        total,
        positions.len(),
        "every particle is stored exactly once"
    );
    assert!(max_leaf <= 8, "leaf with {max_leaf} particles");
    assert!(depth <= 48, "depth {depth}");
    assert_eq!(octree.all_neighbors(), brute_force_within(&positions, h));
}

#[test]
fn octree_terminates_for_particles_closer_than_the_resolution() {
    // Twelve distinct positions one ulp apart cannot be separated by any box
    // the tree can represent; they must share a leaf and still be found.
    let base = 1.0_f64;
    let positions: Vec<[f64; 3]> = (0..12)
        .map(|k| [base + k as f64 * f64::EPSILON, 2.0, 3.0])
        .collect();
    let octree = OctreeNeighborSearch::build(&positions, 0.1);
    let root = octree.root.as_deref().expect("non-empty tree");
    let (depth, _, total) = tree_stats(root, 0);
    assert!(depth <= 48);
    assert_eq!(total, 12);
    let mut found = octree.query([base, 2.0, 3.0]);
    found.sort_unstable();
    assert_eq!(found, (0..12).collect::<Vec<usize>>());
}

#[test]
fn octree_leaves_non_finite_positions_out_of_the_tree() {
    let mut positions = random_points(60, [1.0, 1.0, 1.0], 33);
    positions.extend(std::iter::repeat_n([0.4, 0.4, 0.4], 20));
    let finite_count = positions.len();
    positions.insert(0, [f64::NAN, 0.5, 0.5]);
    positions.insert(7, [f64::INFINITY, 0.0, 0.0]);
    positions.insert(20, [0.5, f64::NEG_INFINITY, 0.5]);
    positions.push([0.5, 0.5, f64::NAN]);
    let h = 0.15;
    let octree = OctreeNeighborSearch::build(&positions, h);
    let root = octree.root.as_deref().expect("finite particles present");
    let (_, _, total) = tree_stats(root, 0);
    assert_eq!(total, finite_count);
    let lists = octree.all_neighbors();
    let expected = brute_force_within(&positions, h);
    assert_eq!(lists.len(), positions.len());
    assert_eq!(lists, expected);
    for (i, p) in positions.iter().enumerate() {
        if p.iter().any(|c| !c.is_finite()) {
            assert!(
                lists[i].is_empty(),
                "non-finite particle {i} has neighbours"
            );
        }
        for &j in &lists[i] {
            assert!(positions[j].iter().all(|c| c.is_finite()));
        }
    }
}

#[test]
fn octree_of_only_non_finite_positions_is_empty_but_keeps_the_indices() {
    let positions = vec![[f64::NAN, 0.0, 0.0]; 12];
    let octree = OctreeNeighborSearch::build(&positions, 0.1);
    assert!(octree.root.is_none());
    let lists = octree.all_neighbors();
    assert_eq!(lists.len(), 12);
    assert!(lists.iter().all(Vec::is_empty));
    assert!(octree.query([0.0, 0.0, 0.0]).is_empty());
}

#[test]
fn octree_build_survives_degenerate_radii() {
    let positions = random_points(40, [1.0, 1.0, 1.0], 34);
    // Infinite radius: everything is a neighbour of everything.
    let infinite = OctreeNeighborSearch::build(&positions, f64::INFINITY);
    let mut found = infinite.query([0.5, 0.5, 0.5]);
    found.sort_unstable();
    assert_eq!(found, (0..40).collect::<Vec<usize>>());
    // NaN radius: nothing is within it.
    let nan = OctreeNeighborSearch::build(&positions, f64::NAN);
    assert!(nan.query([0.5, 0.5, 0.5]).is_empty());
    // Zero radius: only exactly coincident points qualify.
    let zero = OctreeNeighborSearch::build(&positions, 0.0);
    assert_eq!(zero.query(positions[3]), vec![3]);
}
