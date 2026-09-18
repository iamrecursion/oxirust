//! Fixed-radius neighbour candidates over an arbitrary particle cloud.
//!
//! [`GpuCellList`] bins positions on a dense grid anchored at the origin.  The
//! SPH gather kernels need a cell list for clouds that live anywhere in space and
//! may be sparse compared with their interaction radius; [`NeighborGrid`] wraps a
//! [`GpuCellList`] for that purpose:
//!
//! * the cloud is shifted to its bounding-box minimum, so negative and distant
//!   coordinates are binned correctly;
//! * the cells start at the interaction radius and are enlarged until the grid
//!   holds at most a bounded number of cells, so a sparse cloud with a small
//!   radius never needs an astronomically large dense grid (a larger cell only
//!   adds candidates, never drops any);
//! * candidates are returned in ascending index order, so a sum over them adds
//!   its terms in the same order as a loop over all particles.

use super::functions::compute_bounding_box;
use super::types::GpuCellList;

/// The grid holds at most this many cells per particle ...
const MAX_CELLS_PER_PARTICLE: usize = 8;
/// ... but always at least this many cells in total.
const MIN_CELL_BUDGET: usize = 64;
/// Factor by which the cell size grows while the grid is over budget.
const CELL_GROWTH: f64 = 1.5;
/// Relative margin added to the query radius: positions are binned after the
/// shift to the bounding-box minimum, which rounds each coordinate, so a pair
/// exactly at the radius could otherwise be lost.
const RADIUS_MARGIN: f64 = 1.0e-6;

/// A cell grid for fixed-radius neighbour queries over a particle cloud.
///
/// The grid stores a shifted copy of the positions it was built from and answers
/// [`candidates`](Self::candidates) for the particles of that cloud.  It is a
/// snapshot: after the particles move it must be rebuilt.
pub struct NeighborGrid {
    /// Positions shifted by the bounding-box minimum.
    shifted: Vec<[f64; 3]>,
    /// Bounding-box minimum of the original positions (zero for a degenerate cloud).
    origin: [f64; 3],
    cells: GpuCellList,
}

impl NeighborGrid {
    /// Build a grid for `positions` whose cells are at least `radius` wide.
    ///
    /// A `radius` that is not a positive finite number makes the cells as large
    /// as the cloud (queries then test every particle).  A cloud with a
    /// non-finite coordinate is binned into a single cell for the same reason.
    pub fn build(positions: &[[f64; 3]], radius: f64) -> Self {
        let (min, max) = compute_bounding_box(positions);
        let finite_box = (0..3).all(|axis| min[axis].is_finite() && max[axis].is_finite());
        if !finite_box {
            let cells = GpuCellList::build(positions, [1, 1, 1], 1.0, [1.0; 3]);
            return Self {
                shifted: positions.to_vec(),
                origin: [0.0; 3],
                cells,
            };
        }
        let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let widest = extent.iter().copied().fold(0.0, f64::max);
        let mut cell_size = if radius.is_finite() && radius > 0.0 {
            radius
        } else if widest > 0.0 {
            widest
        } else {
            1.0
        };
        let budget = positions
            .len()
            .saturating_mul(MAX_CELLS_PER_PARTICLE)
            .max(MIN_CELL_BUDGET);
        let dims = loop {
            let dims = extent.map(|length| ((length / cell_size).ceil() as usize).max(1));
            let total = dims
                .iter()
                .try_fold(1usize, |acc, &count| acc.checked_mul(count));
            if total.is_some_and(|cells| cells <= budget) || !cell_size.is_finite() {
                break dims;
            }
            cell_size *= CELL_GROWTH;
        };
        let box_lengths = dims.map(|count| count as f64 * cell_size);
        let shifted: Vec<[f64; 3]> = positions
            .iter()
            .map(|p| [p[0] - min[0], p[1] - min[1], p[2] - min[2]])
            .collect();
        let cells = GpuCellList::build(&shifted, dims, cell_size, box_lengths);
        Self {
            shifted,
            origin: min,
            cells,
        }
    }

    /// Number of particles the grid was built from.
    pub fn len(&self) -> usize {
        self.shifted.len()
    }

    /// `true` if the grid was built from no particles.
    pub fn is_empty(&self) -> bool {
        self.shifted.is_empty()
    }

    /// Total number of cells of the underlying grid.
    pub fn cell_count(&self) -> usize {
        self.cells.total_cells()
    }

    /// Candidate neighbours of particle `index`: every particle (the particle
    /// itself included) within `radius` of it, in ascending index order.
    ///
    /// The result may also hold particles up to a relative `1e-6` beyond
    /// `radius`; callers apply their own exact cutoff on the original
    /// coordinates.  An `index` outside the cloud yields no candidates.
    pub fn candidates(&self, index: usize, radius: f64) -> Vec<usize> {
        match self.shifted.get(index) {
            Some(&query) => self.candidates_of_shifted(query, radius),
            None => Vec::new(),
        }
    }

    /// Candidate neighbours of an arbitrary `point` given in the coordinates the
    /// grid was built from (see [`candidates`](Self::candidates)).
    pub fn candidates_near(&self, point: [f64; 3], radius: f64) -> Vec<usize> {
        let shifted = [
            point[0] - self.origin[0],
            point[1] - self.origin[1],
            point[2] - self.origin[2],
        ];
        self.candidates_of_shifted(shifted, radius)
    }

    fn candidates_of_shifted(&self, query: [f64; 3], radius: f64) -> Vec<usize> {
        let padded = radius * (1.0 + RADIUS_MARGIN);
        let mut found = self.cells.neighbors_in_radius(&self.shifted, query, padded);
        found.sort_unstable();
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform samples in `[0, 1)` (splitmix64).
    struct Samples(u64);

    impl Samples {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn cloud(count: usize, seed: u64, lo: f64, hi: f64) -> Vec<[f64; 3]> {
        let mut samples = Samples(seed);
        (0..count)
            .map(|_| {
                [
                    lo + (hi - lo) * samples.next_f64(),
                    lo + (hi - lo) * samples.next_f64(),
                    lo + (hi - lo) * samples.next_f64(),
                ]
            })
            .collect()
    }

    fn brute_force(positions: &[[f64; 3]], index: usize, radius: f64) -> Vec<usize> {
        let p = positions[index];
        (0..positions.len())
            .filter(|&j| {
                let q = positions[j];
                let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= radius * radius
            })
            .collect()
    }

    /// The exact neighbour set must be contained in the candidates.
    fn assert_candidates_cover_brute_force(positions: &[[f64; 3]], radius: f64) {
        let grid = NeighborGrid::build(positions, radius);
        assert_eq!(grid.len(), positions.len());
        for i in 0..positions.len() {
            let exact = brute_force(positions, i, radius);
            let candidates = grid.candidates(i, radius);
            assert!(
                candidates.windows(2).all(|w| w[0] < w[1]),
                "candidates of {i} are not strictly ascending"
            );
            for j in &exact {
                assert!(
                    candidates.binary_search(j).is_ok(),
                    "particle {i} lost neighbour {j}"
                );
            }
            assert!(
                candidates.len() <= exact.len() + 2,
                "{i}: far too many candidates"
            );
        }
    }

    #[test]
    fn candidates_cover_the_brute_force_neighbours_of_a_unit_cube_cloud() {
        assert_candidates_cover_brute_force(&cloud(600, 1, 0.0, 1.0), 0.12);
    }

    #[test]
    fn candidates_cover_the_neighbours_of_a_cloud_at_negative_coordinates() {
        assert_candidates_cover_brute_force(&cloud(500, 2, -7.5, -6.5), 0.15);
    }

    #[test]
    fn candidates_cover_the_neighbours_of_a_cloud_straddling_the_origin() {
        assert_candidates_cover_brute_force(&cloud(500, 3, -0.5, 0.5), 0.1);
    }

    #[test]
    fn candidates_cover_the_neighbours_when_the_radius_exceeds_the_cell_budget() {
        // A radius much larger than the cloud collapses the grid to one cell.
        assert_candidates_cover_brute_force(&cloud(200, 4, 0.0, 1.0), 5.0);
    }

    #[test]
    fn sparse_cloud_with_a_tiny_radius_keeps_the_grid_bounded() {
        // 1000 particles over a 1e6-wide box with radius 1e-3 would need 1e27
        // cells at the nominal cell size.
        let positions = cloud(1000, 5, 0.0, 1.0e6);
        let grid = NeighborGrid::build(&positions, 1.0e-3);
        assert!(
            grid.cell_count() <= 8 * 1000,
            "grid has {} cells",
            grid.cell_count()
        );
        for i in 0..positions.len() {
            let candidates = grid.candidates(i, 1.0e-3);
            assert!(candidates.contains(&i), "a particle is its own candidate");
        }
    }

    #[test]
    fn duplicate_and_coincident_particles_are_found() {
        let positions = vec![[1.0, 2.0, 3.0]; 5];
        let grid = NeighborGrid::build(&positions, 0.5);
        assert_eq!(grid.candidates(2, 0.5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_and_single_particle_clouds_are_handled() {
        let empty = NeighborGrid::build(&[], 1.0);
        assert!(empty.is_empty());
        assert!(empty.candidates(0, 1.0).is_empty());
        let single = NeighborGrid::build(&[[4.0, -2.0, 9.0]], 1.0);
        assert_eq!(single.candidates(0, 1.0), vec![0]);
        assert!(single.candidates(1, 1.0).is_empty());
    }

    #[test]
    fn invalid_radius_degrades_to_testing_every_particle() {
        let positions = cloud(50, 6, 0.0, 1.0);
        for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let grid = NeighborGrid::build(&positions, radius);
            assert!(
                grid.cell_count() <= 64usize.max(8 * positions.len()),
                "radius {radius}"
            );
        }
        // With a usable query radius on such a grid every neighbour is still found.
        let grid = NeighborGrid::build(&positions, 0.0);
        for i in 0..positions.len() {
            for j in brute_force(&positions, i, 0.3) {
                assert!(grid.candidates(i, 0.3).contains(&j));
            }
        }
    }

    #[test]
    fn non_finite_positions_neither_panic_nor_hide_finite_neighbours() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 1.0, 1.0],
        ];
        let grid = NeighborGrid::build(&positions, 0.2);
        assert_eq!(grid.candidates(0, 0.2), vec![0, 1]);
        let nan_first = vec![[f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let grid = NeighborGrid::build(&nan_first, 0.2);
        assert_eq!(grid.candidates(1, 0.2), vec![1, 2]);
    }

    #[test]
    fn candidates_near_matches_candidates_of_a_particle() {
        let positions = cloud(300, 7, -2.0, 2.0);
        let grid = NeighborGrid::build(&positions, 0.4);
        for i in [0, 17, 150, 299] {
            assert_eq!(
                grid.candidates_near(positions[i], 0.4),
                grid.candidates(i, 0.4)
            );
        }
    }

    #[test]
    fn extreme_extents_and_radii_neither_hang_nor_panic() {
        let positions = vec![[-1.0e308, 0.0, 0.0], [1.0e308, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let grid = NeighborGrid::build(&positions, 1.0);
        assert!(grid.cell_count() <= 64);
        assert!(grid.candidates(2, 1.0).contains(&2));
        // A subnormal radius over a unit cloud needs ~1e300 cells at the nominal
        // size.
        let unit_cloud = cloud(100, 8, 0.0, 1.0);
        let grid = NeighborGrid::build(&unit_cloud, 1.0e-300);
        assert!(grid.cell_count() <= 64usize.max(8 * unit_cloud.len()));
        assert!(grid.candidates(0, 1.0e-300).contains(&0));
    }

    #[test]
    fn particles_on_the_upper_boundary_are_binned_with_their_neighbours() {
        // The cloud's largest coordinate is an exact multiple of the radius.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let grid = NeighborGrid::build(&positions, 0.5);
        assert_eq!(grid.candidates(2, 0.5), vec![1, 2]);
        assert_eq!(grid.candidates(3, 0.5), vec![3]);
    }
}
