// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU-side molecular dynamics (MD) solver — CPU mock backend.
//!
//! Implements a Lennard-Jones MD solver pipeline using plain Rust loops as
//! a CPU fallback. Periodic boundary conditions (minimum image convention)
//! are applied; a box edge that is not positive disables the wrapping along
//! that axis. Pairs within the cutoff are found with a periodic cell list (see
//! [`compute_forces_gpu`]), so force and energy evaluation scale linearly with
//! the atom count at constant density. The API mirrors a GPU kernel dispatch
//! for easy substitution.

// ── Data structures ──────────────────────────────────────────────────────────

/// A single MD atom stored in the GPU buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMdAtom {
    /// Atom position \[x, y, z\] in Angstroms.
    pub pos: [f32; 3],
    /// Atom velocity \[vx, vy, vz\] in Å/ps.
    pub vel: [f32; 3],
    /// Current force on atom \[fx, fy, fz\] in kJ/(mol·Å).
    pub force: [f32; 3],
    /// Atom mass in atomic mass units (amu).
    pub mass: f32,
    /// Partial charge in elementary charge units.
    pub charge: f32,
}

/// Simulation parameters for the GPU MD solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMdParams {
    /// Lennard-Jones well depth ε (kJ/mol).
    pub epsilon: f32,
    /// Lennard-Jones radius σ (Å).
    pub sigma: f32,
    /// Interaction cutoff distance (Å).
    pub cutoff: f32,
    /// Periodic simulation box dimensions \[Lx, Ly, Lz\] in Å.
    pub box_size: [f32; 3],
    /// Number of atoms.
    pub n_atoms: usize,
}

/// GPU-side buffer holding the full MD system state.
#[derive(Debug, Clone)]
pub struct GpuMdBuffer {
    /// Per-atom data.
    pub atoms: Vec<GpuMdAtom>,
    /// Current simulation step counter.
    pub step: usize,
}

impl GpuMdBuffer {
    /// Allocate a new buffer for `n` atoms, all at origin with zero velocity.
    pub fn new(n: usize) -> Self {
        Self {
            atoms: vec![
                GpuMdAtom {
                    pos: [0.0; 3],
                    vel: [0.0; 3],
                    force: [0.0; 3],
                    mass: 1.0,
                    charge: 0.0,
                };
                n
            ],
            step: 0,
        }
    }

    /// Number of atoms in this buffer.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

// ── LJ kernel functions ───────────────────────────────────────────────────────

/// Lennard-Jones 12-6 radial force `F(r) = -dU/dr` (positive = repulsive).
///
/// Returns `F(r) = 4ε [ 12σ¹²/r¹³ − 6σ⁶/r⁷ ]`, the component along the unit
/// vector pointing *away* from the partner atom: positive (pushing the pair
/// apart) for `r < 2^(1/6) σ`, zero at the potential minimum and negative
/// (attractive) beyond it.
/// Returns zero for `r >= cutoff` or `r < 1e-10`.
pub fn lj_force_gpu(r: f32, params: &GpuMdParams) -> f32 {
    if r >= params.cutoff || r < 1e-10 {
        return 0.0;
    }
    let sr = params.sigma / r;
    let sr6 = sr * sr * sr * sr * sr * sr;
    let sr12 = sr6 * sr6;
    // Return the force F = -dU/dr = 4ε(12σ¹²/r¹³ - 6σ⁶/r⁷)
    4.0 * params.epsilon * (12.0 * sr12 - 6.0 * sr6) / r
}

/// Lennard-Jones 12-6 pair potential.
///
/// Returns `U(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]`.
/// Returns zero for `r >= cutoff`.
pub fn lj_potential_gpu(r: f32, params: &GpuMdParams) -> f32 {
    if r >= params.cutoff || r < 1e-10 {
        return 0.0;
    }
    let sr = params.sigma / r;
    let sr6 = sr * sr * sr * sr * sr * sr;
    let sr12 = sr6 * sr6;
    4.0 * params.epsilon * (sr12 - sr6)
}

// ── Periodic boundary conditions ──────────────────────────────────────────────

/// Minimum-image distance between two atoms under periodic boundary conditions.
///
/// Applies the minimum image convention along each box axis.
pub fn pbc_distance_gpu(a: [f32; 3], b: [f32; 3], box_size: [f32; 3]) -> f32 {
    let mut r2 = 0.0_f32;
    for k in 0..3 {
        let mut d = a[k] - b[k];
        let l = box_size[k];
        if l > 0.0 {
            d -= (d / l).round() * l;
        }
        r2 += d * d;
    }
    r2.sqrt()
}

/// Minimum-image displacement vector from atom `b` to atom `a`.
fn pbc_displacement_gpu(a: [f32; 3], b: [f32; 3], box_size: [f32; 3]) -> [f32; 3] {
    let mut disp = [0.0_f32; 3];
    for k in 0..3 {
        let mut d = a[k] - b[k];
        let l = box_size[k];
        if l > 0.0 {
            d -= (d / l).round() * l;
        }
        disp[k] = d;
    }
    disp
}

// ── Neighbour search (periodic cell list) ─────────────────────────────────────

/// Pair separations below this (Å) are treated as coincident atoms and exert
/// no force.
const MIN_PAIR_DISTANCE: f32 = 1e-10;

/// A cell list needs at least this many cells along every axis.  With fewer
/// than three, the `-1` and `+1` neighbours of a cell coincide modulo the grid
/// size and a pair of cells would be visited twice.
const MIN_CELLS_PER_AXIS: usize = 3;

/// The 13 neighbour-cell offsets of the half shell: exactly one of every pair
/// of opposite offsets `(d, -d)` of the 26-cell neighbourhood.  Visiting only
/// the half shell from every cell touches each unordered pair of adjacent
/// cells once.
const HALF_SHELL: [[isize; 3]; 13] = [
    [1, 0, 0],
    [-1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [-1, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [-1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Spatial binning of atoms into a periodic grid of cells.
///
/// Every cell is at least `cutoff` wide (plus a few ulps of rounding guard) on
/// every axis, so two atoms closer than the cutoff under the minimum-image
/// convention lie in the same cell or in adjacent cells, the neighbour index
/// wrapping modulo the grid size.
#[derive(Debug)]
struct PeriodicCellGrid {
    /// Number of cells along x, y and z (each at least
    /// [`MIN_CELLS_PER_AXIS`]).
    dims: [usize; 3],
    /// `cell_start[c]..cell_start[c + 1]` is the slice of `order` that holds
    /// the atoms of cell `c`.
    cell_start: Vec<usize>,
    /// Atom indices grouped by cell (counting sort: ascending within a cell).
    order: Vec<usize>,
}

impl PeriodicCellGrid {
    /// Choose the number of cells per axis, or `None` when a cell list does not
    /// apply and the caller must fall back to the all-pairs loop.
    ///
    /// A grid is used only when the cutoff is positive and finite, every box
    /// edge is positive and finite (a non-positive edge means "not periodic
    /// along this axis", which a wrapped grid cannot represent) and the box
    /// holds at least [`MIN_CELLS_PER_AXIS`] cells of cutoff width per axis.
    /// The cell count is capped at `max(27, 2 * n_atoms)` by coarsening the
    /// grid — larger cells only add candidate pairs, never lose any — so a
    /// sparse system in a huge box does not allocate a huge grid.
    fn choose_dims(n_atoms: usize, params: &GpuMdParams) -> Option<[usize; 3]> {
        let cutoff = params.cutoff;
        if !(cutoff.is_finite() && cutoff > 0.0) {
            return None;
        }
        let mut dims = [0_usize; 3];
        for (dim, &edge) in dims.iter_mut().zip(params.box_size.iter()) {
            if !(edge.is_finite() && edge > 0.0) {
                return None;
            }
            // Rounding guard: wrapping a coordinate into the box and the
            // minimum-image arithmetic are accurate to a few ulps of the box
            // edge; widening the cell by that much keeps every pair closer than
            // the cutoff inside the 27-cell neighbourhood.
            let min_width = cutoff + 8.0 * f32::EPSILON * edge;
            let cells = (edge / min_width).floor();
            if cells.is_nan() || cells < MIN_CELLS_PER_AXIS as f32 {
                return None;
            }
            // Saturating float-to-integer conversion: an astronomically fine
            // grid is coarsened below.
            *dim = cells as usize;
        }
        let budget = n_atoms.saturating_mul(2).max(MIN_CELLS_PER_AXIS.pow(3));
        loop {
            let total = dims
                .iter()
                .try_fold(1_usize, |acc, &cells| acc.checked_mul(cells));
            if total.is_some_and(|cells| cells <= budget) {
                return Some(dims);
            }
            let product: f64 = dims.iter().map(|&cells| cells as f64).product();
            // Shrink every axis by the same factor (never below the minimum).
            // At least one axis above the minimum shrinks per pass, and once
            // all axes sit at the minimum the product (27) is within budget.
            let shrink = (product / budget as f64).cbrt().max(1.000_001);
            for cells in dims.iter_mut() {
                *cells = ((*cells as f64 / shrink).floor() as usize).max(MIN_CELLS_PER_AXIS);
            }
        }
    }

    /// Bin `positions` into a grid with `dims` cells per axis.
    fn build(positions: &[[f32; 3]], dims: [usize; 3], box_size: [f32; 3]) -> Self {
        let n_cells = dims[0] * dims[1] * dims[2];
        let cell_of: Vec<usize> = positions
            .iter()
            .map(|pos| {
                let cx = Self::axis_cell(pos[0], box_size[0], dims[0]);
                let cy = Self::axis_cell(pos[1], box_size[1], dims[1]);
                let cz = Self::axis_cell(pos[2], box_size[2], dims[2]);
                (cz * dims[1] + cy) * dims[0] + cx
            })
            .collect();

        // Counting sort: histogram, exclusive prefix sum, stable scatter.
        let mut cell_start = vec![0_usize; n_cells + 1];
        for &cell in &cell_of {
            cell_start[cell + 1] += 1;
        }
        let mut running = 0_usize;
        for start in cell_start.iter_mut() {
            running += *start;
            *start = running;
        }
        let mut cursor = cell_start.clone();
        let mut order = vec![0_usize; positions.len()];
        for (atom, &cell) in cell_of.iter().enumerate() {
            order[cursor[cell]] = atom;
            cursor[cell] += 1;
        }
        Self {
            dims,
            cell_start,
            order,
        }
    }

    /// Cell coordinate of the wrapped coordinate `x` along one axis of length
    /// `edge` divided into `cells` cells.
    fn axis_cell(x: f32, edge: f32, cells: usize) -> usize {
        // `rem_euclid` can return exactly `edge` for a tiny negative `x`, and a
        // non-finite `x` gives NaN (which converts to cell 0).
        let wrapped = x.rem_euclid(edge);
        let cell = (wrapped / edge * cells as f32) as usize;
        cell.min(cells - 1)
    }

    /// Atoms of the cell with the given linear index.
    fn atoms_of(&self, cell: usize) -> &[usize] {
        &self.order[self.cell_start[cell]..self.cell_start[cell + 1]]
    }

    /// Call `visit(i, j)` once for every unordered pair of atoms that share a
    /// cell or sit in adjacent cells (periodic wrap included).
    fn for_each_candidate_pair(&self, mut visit: impl FnMut(usize, usize)) {
        let [nx, ny, nz] = self.dims;
        for cz in 0..nz {
            for cy in 0..ny {
                for cx in 0..nx {
                    let atoms = self.atoms_of((cz * ny + cy) * nx + cx);
                    for (slot, &i) in atoms.iter().enumerate() {
                        for &j in atoms.iter().skip(slot + 1) {
                            visit(i, j);
                        }
                    }
                    for offset in &HALF_SHELL {
                        let ox = Self::wrap_step(cx, offset[0], nx);
                        let oy = Self::wrap_step(cy, offset[1], ny);
                        let oz = Self::wrap_step(cz, offset[2], nz);
                        let neighbours = self.atoms_of((oz * ny + oy) * nx + ox);
                        for &i in atoms {
                            for &j in neighbours {
                                visit(i, j);
                            }
                        }
                    }
                }
            }
        }
    }

    /// `(cell + delta) mod cells` for a signed step `delta` of at most one cell.
    fn wrap_step(cell: usize, delta: isize, cells: usize) -> usize {
        (cell as isize + delta).rem_euclid(cells as isize) as usize
    }
}

/// Call `visit(i, j, disp, r)` exactly once for every unordered pair of atoms
/// whose minimum-image distance `r` lies in `[1e-10, cutoff)`.
///
/// `disp` is the minimum-image displacement from atom `j` to atom `i`.  The
/// candidate pairs come from a periodic cell list when one applies (see
/// [`PeriodicCellGrid::choose_dims`]) and from the all-pairs double loop
/// otherwise.  The cutoff test compares `r²` with `cutoff²`, so only pairs
/// that are actually in range pay for a square root.
fn for_each_pair_in_cutoff(
    positions: &[[f32; 3]],
    params: &GpuMdParams,
    mut visit: impl FnMut(usize, usize, [f32; 3], f32),
) {
    let n = positions.len();
    if n < 2 {
        return;
    }
    let cutoff_sq = params.cutoff * params.cutoff;
    let mut test_pair = |i: usize, j: usize| {
        let disp = pbc_displacement_gpu(positions[i], positions[j], params.box_size);
        let r2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
        if r2 >= cutoff_sq {
            return;
        }
        let r = r2.sqrt();
        if r < MIN_PAIR_DISTANCE {
            return;
        }
        visit(i, j, disp, r);
    };
    match PeriodicCellGrid::choose_dims(n, params) {
        Some(dims) => {
            PeriodicCellGrid::build(positions, dims, params.box_size)
                .for_each_candidate_pair(&mut test_pair);
        }
        None => {
            for i in 0..n {
                for j in (i + 1)..n {
                    test_pair(i, j);
                }
            }
        }
    }
}

// ── Force computation ─────────────────────────────────────────────────────────

/// Compute the Lennard-Jones force on every atom and store it in `atom.force`.
///
/// Clears previous forces before accumulating new ones.  Pairs beyond
/// `params.cutoff` (minimum-image distance) do not interact.  Neighbours are
/// found with a periodic cell list, so the cost grows linearly with the number
/// of atoms at fixed density; a box smaller than three cutoffs along some axis,
/// or a box edge that is not positive, uses the all-pairs loop instead.
pub fn compute_forces_gpu(buf: &mut GpuMdBuffer, params: &GpuMdParams) {
    for atom in buf.atoms.iter_mut() {
        atom.force = [0.0; 3];
    }
    let positions: Vec<[f32; 3]> = buf.atoms.iter().map(|atom| atom.pos).collect();
    for_each_pair_in_cutoff(&positions, params, |i, j, disp, r| {
        // `lj_force_gpu` returns F(r) = -dU/dr (positive = repulsive) and
        // `disp = r_i - r_j` points from j to i, so the force on atom i is
        // F_i = F(r) * disp / r; Newton's third law gives F_j = -F_i.
        let scale = lj_force_gpu(r, params) / r;
        for (axis, &component) in disp.iter().enumerate() {
            buf.atoms[i].force[axis] += scale * component;
            buf.atoms[j].force[axis] -= scale * component;
        }
    });
}

// ── Integration ───────────────────────────────────────────────────────────────

/// Velocity-Verlet integration step.
///
/// Updates positions and velocities using the current forces.
/// `v += (F/m) * dt`, `x += v * dt`.
/// Increments the step counter.
pub fn verlet_integrate_gpu(buf: &mut GpuMdBuffer, dt: f32) {
    for atom in buf.atoms.iter_mut() {
        let inv_mass = if atom.mass > 1e-10 {
            1.0 / atom.mass
        } else {
            0.0
        };
        atom.vel[0] += atom.force[0] * inv_mass * dt;
        atom.vel[1] += atom.force[1] * inv_mass * dt;
        atom.vel[2] += atom.force[2] * inv_mass * dt;
        atom.pos[0] += atom.vel[0] * dt;
        atom.pos[1] += atom.vel[1] * dt;
        atom.pos[2] += atom.vel[2] * dt;
    }
    buf.step += 1;
}

// ── Thermodynamic observables ─────────────────────────────────────────────────

/// Compute total kinetic energy: `KE = Σ 0.5 * m_i * |v_i|²`.
pub fn kinetic_energy_gpu(buf: &GpuMdBuffer) -> f32 {
    buf.atoms
        .iter()
        .map(|a| 0.5 * a.mass * (a.vel[0] * a.vel[0] + a.vel[1] * a.vel[1] + a.vel[2] * a.vel[2]))
        .sum()
}

/// Compute total LJ potential energy.
///
/// Sums `lj_potential_gpu` over every pair within `params.cutoff`, visiting
/// exactly the pairs [`compute_forces_gpu`] does (same periodic cell list, same
/// all-pairs fallback), so the energy is consistent with the forces.
pub fn potential_energy_gpu(buf: &GpuMdBuffer, params: &GpuMdParams) -> f32 {
    let positions: Vec<[f32; 3]> = buf.atoms.iter().map(|atom| atom.pos).collect();
    let mut pe = 0.0_f32;
    for_each_pair_in_cutoff(&positions, params, |_, _, _, r| {
        pe += lj_potential_gpu(r, params);
    });
    pe
}

/// Estimate instantaneous temperature from kinetic energy.
///
/// Uses the equipartition theorem: `T = 2 * KE / (3 * N * kB)`.
/// Boltzmann constant `kB = 8.314e-3` kJ/(mol·K).
pub fn temperature_gpu(buf: &GpuMdBuffer) -> f32 {
    let n = buf.atoms.len();
    if n == 0 {
        return 0.0;
    }
    let kb = 8.314e-3_f32; // kJ/(mol·K)
    let ke = kinetic_energy_gpu(buf);
    2.0 * ke / (3.0 * n as f32 * kb)
}

// ── Thermostat ────────────────────────────────────────────────────────────────

/// Rescale all atom velocities to achieve `target_temp` (velocity scaling).
///
/// No-op when the current temperature is zero.
pub fn rescale_velocities_gpu(buf: &mut GpuMdBuffer, target_temp: f32) {
    let t_curr = temperature_gpu(buf);
    if t_curr < 1e-10 || target_temp < 0.0 {
        return;
    }
    let scale = (target_temp / t_curr).sqrt();
    for atom in buf.atoms.iter_mut() {
        atom.vel[0] *= scale;
        atom.vel[1] *= scale;
        atom.vel[2] *= scale;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> GpuMdParams {
        GpuMdParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 3.5,
            box_size: [10.0; 3],
            n_atoms: 4,
        }
    }

    fn make_buf_grid(n: usize) -> GpuMdBuffer {
        let mut buf = GpuMdBuffer::new(n);
        for i in 0..n {
            buf.atoms[i].pos = [i as f32 * 1.5, 0.0, 0.0];
            buf.atoms[i].mass = 1.0;
        }
        buf
    }

    #[test]
    fn test_gpu_md_atom_fields() {
        let a = GpuMdAtom {
            pos: [1.0, 2.0, 3.0],
            vel: [0.1, 0.2, 0.3],
            force: [0.0; 3],
            mass: 12.0,
            charge: -0.5,
        };
        assert_eq!(a.mass, 12.0);
        assert_eq!(a.charge, -0.5);
    }

    #[test]
    fn test_gpu_md_params_fields() {
        let p = default_params();
        assert_eq!(p.n_atoms, 4);
        assert!(p.cutoff > p.sigma);
    }

    #[test]
    fn test_gpu_md_buffer_new() {
        let buf = GpuMdBuffer::new(5);
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert_eq!(buf.step, 0);
    }

    #[test]
    fn test_gpu_md_buffer_empty() {
        let buf = GpuMdBuffer::new(0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_lj_potential_minimum() {
        let params = default_params();
        // LJ minimum at r = 2^(1/6) * sigma
        let r_min = 2.0_f32.powf(1.0 / 6.0) * params.sigma;
        let u = lj_potential_gpu(r_min, &params);
        // At minimum: U = -epsilon
        assert!((u - (-params.epsilon)).abs() < 0.01);
    }

    #[test]
    fn test_lj_potential_zero_beyond_cutoff() {
        let params = default_params();
        assert_eq!(lj_potential_gpu(params.cutoff + 0.1, &params), 0.0);
    }

    #[test]
    fn test_lj_potential_zero_near_zero_r() {
        let params = default_params();
        assert_eq!(lj_potential_gpu(0.0, &params), 0.0);
    }

    #[test]
    fn test_lj_force_repulsive_close() {
        let params = default_params();
        // For r < 2^(1/6)*sigma the pair repels: F = -dU/dr > 0.
        let f = lj_force_gpu(0.8, &params);
        assert!(f > 0.0);
    }

    #[test]
    fn test_lj_force_attractive_far() {
        let params = default_params();
        // Beyond 2^(1/6)*sigma ~ 1.122 sigma the pair attracts: F = -dU/dr < 0.
        let f = lj_force_gpu(1.2, &params);
        assert!(f < 0.0);
    }

    #[test]
    fn test_lj_force_vanishes_at_potential_minimum() {
        let params = default_params();
        let r_min = 2.0_f32.powf(1.0 / 6.0) * params.sigma;
        assert!(lj_force_gpu(r_min, &params).abs() < 1e-3);
    }

    #[test]
    fn test_lj_force_zero_beyond_cutoff() {
        let params = default_params();
        assert_eq!(lj_force_gpu(params.cutoff + 1.0, &params), 0.0);
    }

    #[test]
    fn test_pbc_distance_no_wrap() {
        let params = default_params();
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let d = pbc_distance_gpu(a, b, params.box_size);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pbc_distance_wrap() {
        let box_size = [10.0_f32; 3];
        let a = [0.5, 0.0, 0.0];
        let b = [9.5, 0.0, 0.0];
        // Minimum image: 1.0, not 9.0
        let d = pbc_distance_gpu(a, b, box_size);
        assert!((d - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pbc_distance_self() {
        let box_size = [10.0_f32; 3];
        let a = [3.0, 4.0, 5.0];
        let d = pbc_distance_gpu(a, a, box_size);
        assert!(d < 1e-5);
    }

    #[test]
    fn test_compute_forces_gpu_newton3() {
        let mut buf = GpuMdBuffer::new(2);
        buf.atoms[0].pos = [0.0; 3];
        buf.atoms[1].pos = [1.2, 0.0, 0.0];
        buf.atoms[0].mass = 1.0;
        buf.atoms[1].mass = 1.0;
        let params = default_params();
        compute_forces_gpu(&mut buf, &params);
        // Newton's third law
        assert!((buf.atoms[0].force[0] + buf.atoms[1].force[0]).abs() < 1e-5);
    }

    /// Two atoms on the x axis at `x0` and `x1` in the default box.
    fn pair_buf(x0: f32, x1: f32) -> GpuMdBuffer {
        let mut buf = GpuMdBuffer::new(2);
        buf.atoms[0].pos = [x0, 0.0, 0.0];
        buf.atoms[1].pos = [x1, 0.0, 0.0];
        buf
    }

    /// Regression: the pair force used to have the wrong sign, so repulsion
    /// pulled the atoms together.  At 0.9 sigma the pair must push apart.
    #[test]
    fn test_pair_force_repulsive_pushes_atoms_apart() {
        let params = default_params();
        let mut buf = pair_buf(0.0, 0.9);
        compute_forces_gpu(&mut buf, &params);
        let expected = lj_force_gpu(0.9, &params);
        assert!(expected > 0.0);
        assert!(
            buf.atoms[0].force[0] < 0.0,
            "atom 0 must be pushed toward -x, got {}",
            buf.atoms[0].force[0]
        );
        assert!(
            buf.atoms[1].force[0] > 0.0,
            "atom 1 must be pushed toward +x, got {}",
            buf.atoms[1].force[0]
        );
        assert!((buf.atoms[1].force[0] - expected).abs() <= 1e-4 * expected);
        assert!((buf.atoms[0].force[0] + expected).abs() <= 1e-4 * expected);
        assert!(buf.atoms[0].force[1].abs() < 1e-6 && buf.atoms[0].force[2].abs() < 1e-6);
    }

    /// Beyond the potential minimum the same pair attracts.
    #[test]
    fn test_pair_force_attractive_pulls_atoms_together() {
        let params = default_params();
        let mut buf = pair_buf(0.0, 1.5);
        compute_forces_gpu(&mut buf, &params);
        assert!(
            buf.atoms[0].force[0] > 0.0,
            "atom 0 must be pulled toward +x, got {}",
            buf.atoms[0].force[0]
        );
        assert!(
            buf.atoms[1].force[0] < 0.0,
            "atom 1 must be pulled toward -x, got {}",
            buf.atoms[1].force[0]
        );
        let expected = lj_force_gpu(1.5, &params);
        assert!((buf.atoms[1].force[0] - expected).abs() <= 1e-4 * expected.abs());
    }

    /// The minimum-image path must orient the force across the periodic
    /// boundary: atoms at x = 0 and x = 9.1 (box 10) are 0.9 apart through the
    /// boundary, so atom 0 is pushed toward +x and atom 1 toward -x.
    #[test]
    fn test_pair_force_orientation_across_periodic_boundary() {
        let params = default_params();
        let mut buf = pair_buf(0.0, 9.1);
        compute_forces_gpu(&mut buf, &params);
        assert!(buf.atoms[0].force[0] > 0.0, "{:?}", buf.atoms[0].force);
        assert!(buf.atoms[1].force[0] < 0.0, "{:?}", buf.atoms[1].force);
        let expected = lj_force_gpu(0.9, &params);
        assert!((buf.atoms[0].force[0] - expected).abs() <= 1e-3 * expected);
    }

    /// Central finite difference of the pair potential energy for atom `atom`
    /// along `axis` (step `h`), i.e. `-dU/dx`.
    fn fd_force(buf: &GpuMdBuffer, params: &GpuMdParams, atom: usize, axis: usize, h: f32) -> f32 {
        let mut plus = buf.clone();
        plus.atoms[atom].pos[axis] += h;
        let mut minus = buf.clone();
        minus.atoms[atom].pos[axis] -= h;
        -(potential_energy_gpu(&plus, params) - potential_energy_gpu(&minus, params)) / (2.0 * h)
    }

    /// The accumulated forces must equal `-grad U` of `potential_energy_gpu`.
    #[test]
    fn test_pair_force_matches_negative_potential_gradient() {
        let params = default_params();
        for &r in &[1.15_f32, 1.3, 1.5, 2.2] {
            let mut buf = pair_buf(0.0, r);
            compute_forces_gpu(&mut buf, &params);
            let fd = fd_force(&buf, &params, 0, 0, 1e-3);
            assert!(
                (fd - buf.atoms[0].force[0]).abs() < 1e-3,
                "r={r}: finite difference {fd} vs analytic {}",
                buf.atoms[0].force[0]
            );
        }
    }

    /// Three atoms in a generic (non-axis-aligned) arrangement: every force
    /// component equals the negative potential gradient.
    #[test]
    fn test_three_atom_forces_match_negative_potential_gradient() {
        let params = default_params();
        let mut buf = GpuMdBuffer::new(3);
        buf.atoms[0].pos = [1.0, 1.0, 1.0];
        buf.atoms[1].pos = [2.2, 1.1, 0.9];
        buf.atoms[2].pos = [1.5, 2.3, 1.4];
        compute_forces_gpu(&mut buf, &params);
        for atom in 0..3 {
            for axis in 0..3 {
                let fd = fd_force(&buf, &params, atom, axis, 1e-3);
                let analytic = buf.atoms[atom].force[axis];
                assert!(
                    (fd - analytic).abs() < 2e-3,
                    "atom {atom} axis {axis}: finite difference {fd} vs analytic {analytic}"
                );
            }
        }
    }

    /// Dynamics: a compressed pair released from rest separates, a stretched
    /// pair (inside the cutoff) approaches.
    #[test]
    fn test_pair_dynamics_repel_then_attract() {
        let params = default_params();

        let mut close = pair_buf(0.0, 0.9);
        for _ in 0..10 {
            compute_forces_gpu(&mut close, &params);
            verlet_integrate_gpu(&mut close, 1e-3);
        }
        let sep_close = close.atoms[1].pos[0] - close.atoms[0].pos[0];
        assert!(
            sep_close > 0.9,
            "compressed pair must separate: {sep_close}"
        );

        let mut far = pair_buf(0.0, 1.5);
        for _ in 0..10 {
            compute_forces_gpu(&mut far, &params);
            verlet_integrate_gpu(&mut far, 1e-3);
        }
        let sep_far = far.atoms[1].pos[0] - far.atoms[0].pos[0];
        assert!(sep_far < 1.5, "stretched pair must approach: {sep_far}");
    }

    #[test]
    fn test_compute_forces_gpu_zero_beyond_cutoff() {
        let mut buf = GpuMdBuffer::new(2);
        buf.atoms[0].pos = [0.0; 3];
        buf.atoms[1].pos = [5.0, 0.0, 0.0]; // beyond cutoff=3.5
        let params = default_params();
        compute_forces_gpu(&mut buf, &params);
        assert!(buf.atoms[0].force[0].abs() < 1e-8);
    }

    #[test]
    fn test_verlet_integrate_gpu_position() {
        let mut buf = GpuMdBuffer::new(1);
        buf.atoms[0].vel = [1.0, 0.0, 0.0];
        buf.atoms[0].force = [0.0; 3];
        verlet_integrate_gpu(&mut buf, 0.1);
        assert!((buf.atoms[0].pos[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_verlet_integrate_gpu_step_counter() {
        let mut buf = GpuMdBuffer::new(1);
        verlet_integrate_gpu(&mut buf, 0.01);
        assert_eq!(buf.step, 1);
    }

    #[test]
    fn test_kinetic_energy_gpu_zero_vel() {
        let buf = make_buf_grid(4);
        let ke = kinetic_energy_gpu(&buf);
        assert!(ke.abs() < 1e-8);
    }

    #[test]
    fn test_kinetic_energy_gpu_nonzero() {
        let mut buf = GpuMdBuffer::new(2);
        buf.atoms[0].vel = [1.0, 0.0, 0.0];
        buf.atoms[0].mass = 2.0;
        buf.atoms[1].vel = [0.0, 0.0, 0.0];
        buf.atoms[1].mass = 1.0;
        let ke = kinetic_energy_gpu(&buf);
        // 0.5 * 2 * 1^2 = 1.0
        assert!((ke - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_potential_energy_gpu_empty() {
        let buf = GpuMdBuffer::new(0);
        let params = default_params();
        assert_eq!(potential_energy_gpu(&buf, &params), 0.0);
    }

    #[test]
    fn test_potential_energy_gpu_single() {
        let buf = GpuMdBuffer::new(1);
        let params = default_params();
        // Only one atom -> no pairs -> PE = 0
        assert_eq!(potential_energy_gpu(&buf, &params), 0.0);
    }

    #[test]
    fn test_temperature_gpu_zero_vel() {
        let buf = make_buf_grid(4);
        let t = temperature_gpu(&buf);
        assert!(t < 1e-6);
    }

    #[test]
    fn test_temperature_gpu_empty() {
        let buf = GpuMdBuffer::new(0);
        assert_eq!(temperature_gpu(&buf), 0.0);
    }

    #[test]
    fn test_temperature_gpu_nonzero() {
        let mut buf = GpuMdBuffer::new(3);
        for a in buf.atoms.iter_mut() {
            a.vel = [1.0, 1.0, 1.0];
            a.mass = 1.0;
        }
        let t = temperature_gpu(&buf);
        assert!(t > 0.0);
    }

    #[test]
    fn test_rescale_velocities_gpu() {
        let mut buf = GpuMdBuffer::new(4);
        for a in buf.atoms.iter_mut() {
            a.vel = [1.0, 0.5, 0.2];
            a.mass = 1.0;
        }
        let target = 300.0;
        rescale_velocities_gpu(&mut buf, target);
        let t_after = temperature_gpu(&buf);
        assert!((t_after - target).abs() < 1.0);
    }

    #[test]
    fn test_rescale_velocities_gpu_zero_vel_noop() {
        let mut buf = GpuMdBuffer::new(2);
        // All velocities zero -> temperature is zero -> no rescaling
        rescale_velocities_gpu(&mut buf, 300.0);
        for a in &buf.atoms {
            assert!(a.vel[0].abs() < 1e-8);
        }
    }

    #[test]
    fn test_buf_clone() {
        let buf = make_buf_grid(3);
        let buf2 = buf.clone();
        assert_eq!(buf2.len(), 3);
    }

    #[test]
    fn test_compute_forces_accumulate_many() {
        let mut buf = make_buf_grid(4);
        let params = default_params();
        compute_forces_gpu(&mut buf, &params);
        // Total force should be ~zero (Newton's 3rd law globally)
        let fx_total: f32 = buf.atoms.iter().map(|a| a.force[0]).sum();
        assert!(fx_total.abs() < 1e-4);
    }

    #[test]
    fn test_lj_potential_positive_repulsive() {
        let params = default_params();
        // Very short r -> strongly repulsive -> U > 0
        let u = lj_potential_gpu(0.5, &params);
        assert!(u > 0.0);
    }

    #[test]
    fn test_verlet_integrate_gpu_velocity_from_force() {
        let mut buf = GpuMdBuffer::new(1);
        buf.atoms[0].force = [2.0, 0.0, 0.0];
        buf.atoms[0].mass = 1.0;
        verlet_integrate_gpu(&mut buf, 0.5);
        // v = 0 + (2/1)*0.5 = 1.0
        assert!((buf.atoms[0].vel[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pbc_distance_3d_wrap() {
        let box_size = [5.0_f32; 3];
        let a = [0.1, 0.1, 0.1];
        let b = [4.9, 4.9, 4.9];
        let d = pbc_distance_gpu(a, b, box_size);
        // Minimum image: delta = (-0.2, -0.2, -0.2) => r = 0.2*sqrt(3)
        let expected = 0.2 * 3.0_f32.sqrt();
        assert!((d - expected).abs() < 1e-4);
    }

    #[test]
    fn test_total_energy_two_atoms() {
        let mut buf = GpuMdBuffer::new(2);
        buf.atoms[0].pos = [0.0; 3];
        buf.atoms[0].mass = 1.0;
        buf.atoms[1].pos = [1.1, 0.0, 0.0]; // near LJ minimum
        buf.atoms[1].mass = 1.0;
        let params = default_params();
        let pe = potential_energy_gpu(&buf, &params);
        // Should be negative near minimum
        assert!(pe < 0.0);
    }

    // ── Periodic cell list ────────────────────────────────────────────────

    /// Small deterministic generator (SplitMix64) for the cell-list tests.
    struct TestRng(u64);

    impl TestRng {
        /// Uniform value in `[0, 1)`.
        fn next_unit(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32) / (1_u64 << 24) as f32
        }
    }

    /// `n_atoms` atoms taken from a `side`³ simple-cubic lattice of the given
    /// `spacing`, each displaced by up to `jitter` along every axis, so no two
    /// atoms sit at a tiny separation (which would give astronomic forces).
    fn lattice_buffer(
        side: usize,
        n_atoms: usize,
        spacing: f32,
        jitter: f32,
        seed: u64,
    ) -> GpuMdBuffer {
        let mut rng = TestRng(seed);
        let mut buf = GpuMdBuffer::new(n_atoms);
        for (index, atom) in buf.atoms.iter_mut().enumerate() {
            let ix = index % side;
            let iy = (index / side) % side;
            let iz = index / (side * side);
            let mut coordinate = |cell: usize| {
                (cell as f32 + 0.5) * spacing + (rng.next_unit() * 2.0 - 1.0) * jitter
            };
            atom.pos = [coordinate(ix), coordinate(iy), coordinate(iz)];
        }
        buf
    }

    /// LJ parameters with a cubic box of the given edge.
    fn cubic_params(edge: f32, cutoff: f32, n_atoms: usize) -> GpuMdParams {
        GpuMdParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff,
            box_size: [edge; 3],
            n_atoms,
        }
    }

    fn positions_of(buf: &GpuMdBuffer) -> Vec<[f32; 3]> {
        buf.atoms.iter().map(|atom| atom.pos).collect()
    }

    /// Independent all-pairs reference: every `(i, j)`, `i < j`, with
    /// minimum-image distance in `[1e-10, cutoff)`.
    fn reference_pairs(positions: &[[f32; 3]], params: &GpuMdParams) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (i, &pi) in positions.iter().enumerate() {
            for (j, &pj) in positions.iter().enumerate().skip(i + 1) {
                let r = pbc_distance_gpu(pi, pj, params.box_size);
                if (1e-10..params.cutoff).contains(&r) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Independent all-pairs reference forces (the pre-cell-list algorithm).
    fn reference_forces(positions: &[[f32; 3]], params: &GpuMdParams) -> Vec<[f32; 3]> {
        let mut forces = vec![[0.0_f32; 3]; positions.len()];
        for (i, j) in reference_pairs(positions, params) {
            let disp = pbc_displacement_gpu(positions[i], positions[j], params.box_size);
            let r = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
            let scale = lj_force_gpu(r, params) / r;
            for (axis, &component) in disp.iter().enumerate() {
                forces[i][axis] += scale * component;
                forces[j][axis] -= scale * component;
            }
        }
        forces
    }

    fn reference_energy(positions: &[[f32; 3]], params: &GpuMdParams) -> f32 {
        reference_pairs(positions, params)
            .into_iter()
            .map(|(i, j)| {
                let r = pbc_distance_gpu(positions[i], positions[j], params.box_size);
                lj_potential_gpu(r, params)
            })
            .sum()
    }

    /// Largest `|F|` component of a force set, floored at one.
    fn force_scale(forces: &[[f32; 3]]) -> f32 {
        forces
            .iter()
            .flat_map(|force| force.iter())
            .fold(1.0_f32, |acc, &component| acc.max(component.abs()))
    }

    /// Assert that the cell grid with `dims` cells per axis visits every pair
    /// of `positions` that is in range under the minimum-image convention, and
    /// visits no pair twice.  Returns the number of candidate pairs.
    fn assert_grid_covers_all_pairs(
        positions: &[[f32; 3]],
        params: &GpuMdParams,
        dims: [usize; 3],
    ) -> usize {
        let grid = PeriodicCellGrid::build(positions, dims, params.box_size);
        let mut candidates = Vec::new();
        grid.for_each_candidate_pair(|i, j| candidates.push((i.min(j), i.max(j))));
        let visited = candidates.len();
        candidates.sort_unstable();
        candidates.dedup();
        assert_eq!(candidates.len(), visited, "a cell pair was visited twice");
        let expected = reference_pairs(positions, params);
        assert!(!expected.is_empty());
        for pair in &expected {
            assert!(
                candidates.binary_search(pair).is_ok(),
                "in-range pair {pair:?} is missing from the cell-list candidates (dims {dims:?})"
            );
        }
        visited
    }

    #[test]
    fn test_cell_grid_dims_selection() {
        let params = |box_size: [f32; 3], cutoff: f32| GpuMdParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff,
            box_size,
            n_atoms: 500,
        };
        // Default test box: 10 / 3.5 gives two cells per axis -> all-pairs.
        assert_eq!(PeriodicCellGrid::choose_dims(500, &default_params()), None);
        // 7.6 / 2.5 = 3.04 cells -> exactly the minimum of three.
        assert_eq!(
            PeriodicCellGrid::choose_dims(500, &params([7.6; 3], 2.5)),
            Some([3; 3])
        );
        assert_eq!(
            PeriodicCellGrid::choose_dims(500, &params([11.0; 3], 2.5)),
            Some([4; 3])
        );
        // Anisotropic box: the long axis gets more cells.
        assert_eq!(
            PeriodicCellGrid::choose_dims(1000, &params([100.0, 10.0, 10.0], 2.5)),
            Some([39, 3, 3])
        );
        // One short axis is enough to fall back.
        assert_eq!(
            PeriodicCellGrid::choose_dims(500, &params([30.0, 30.0, 6.0], 2.5)),
            None
        );
    }

    #[test]
    fn test_cell_grid_rejects_unusable_box_or_cutoff() {
        let params = |box_size: [f32; 3], cutoff: f32| GpuMdParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff,
            box_size,
            n_atoms: 500,
        };
        for edge in [0.0, -10.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                PeriodicCellGrid::choose_dims(500, &params([20.0, edge, 20.0], 2.5)),
                None,
                "box edge {edge}"
            );
        }
        for cutoff in [0.0, -2.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                PeriodicCellGrid::choose_dims(500, &params([20.0; 3], cutoff)),
                None,
                "cutoff {cutoff}"
            );
        }
    }

    #[test]
    fn test_cell_grid_cell_count_is_capped() {
        let params = |edge: f32, cutoff: f32| cubic_params(edge, cutoff, 0);
        // Few atoms in a huge box: the grid collapses to the minimum.
        assert_eq!(
            PeriodicCellGrid::choose_dims(10, &params(1000.0, 1.0)),
            Some([3; 3])
        );
        // About a million cells per axis (10^18 in total) is coarsened to the
        // minimum as well; the product is formed with checked arithmetic.
        assert_eq!(
            PeriodicCellGrid::choose_dims(10, &params(1.0e30, 1.0e-3)),
            Some([3; 3])
        );
        // More atoms: the budget is 2n cells, shared evenly between the axes.
        let dims = PeriodicCellGrid::choose_dims(1000, &params(1000.0, 1.0))
            .expect("a grid applies to a huge box");
        let total: usize = dims.iter().product();
        assert!(total <= 2000, "{dims:?} exceeds the cell budget");
        assert!(total >= 1000, "{dims:?} is needlessly coarse");
        assert!(dims.iter().all(|&cells| cells >= MIN_CELLS_PER_AXIS));
        assert_eq!(dims[0], dims[1]);
        assert_eq!(dims[1], dims[2]);
    }

    #[test]
    fn test_cell_grid_covers_every_in_range_pair() {
        // 500 atoms on a jittered lattice in a box of edge 10 (spacing 1.25).
        let buf = lattice_buffer(8, 500, 1.25, 0.15, 11);
        let positions = positions_of(&buf);
        let n_pairs = positions.len() * (positions.len() - 1) / 2;
        // cutoff 2.5 -> 3 cells (every cell is a neighbour of every other:
        // all pairs are candidates, each exactly once); 2.4 -> 4; 1.9 -> 5.
        for (cutoff, expected_cells) in [(2.5_f32, 3_usize), (2.4, 4), (1.9, 5)] {
            let params = cubic_params(10.0, cutoff, 500);
            let dims = PeriodicCellGrid::choose_dims(500, &params)
                .expect("the cell list must apply to this configuration");
            assert_eq!(dims, [expected_cells; 3], "cutoff {cutoff}");
            let visited = assert_grid_covers_all_pairs(&positions, &params, dims);
            if expected_cells == 3 {
                assert_eq!(visited, n_pairs);
            } else {
                assert!(
                    visited < n_pairs,
                    "cutoff {cutoff}: the grid prunes nothing"
                );
            }
        }
    }

    #[test]
    fn test_cell_grid_pairs_across_every_periodic_face() {
        // Atoms hugging opposite faces, edges and corners of the box: each pair
        // is 1.2 apart through the boundary and must still be found.
        let edge = 11.0_f32;
        let params = cubic_params(edge, 2.5, 0);
        let mut positions = Vec::new();
        for axis in 0..3 {
            let mut low = [5.5_f32; 3];
            low[axis] = 0.4;
            let mut high = [5.5_f32; 3];
            high[axis] = edge - 0.8;
            positions.push(low);
            positions.push(high);
        }
        positions.push([0.3, 0.3, 0.3]);
        positions.push([edge - 0.4, edge - 0.5, edge - 0.6]);
        // Four cells per axis (cell width 2.75): the low and high faces of the
        // box are cells 0 and 3, neighbours only through the periodic wrap.
        let visited = assert_grid_covers_all_pairs(&positions, &params, [4; 3]);
        assert!(visited > 0);
    }

    #[test]
    fn test_cell_grid_atoms_on_or_outside_the_box_are_binned_safely() {
        let edge = 10.0_f32;
        let params = cubic_params(edge, 2.4, 0);
        // `rem_euclid` of a tiny negative coordinate is exactly `edge`.
        let mut positions = vec![
            [-1.0e-9, 5.0, 5.0],
            [1.1, 5.0, 5.0],
            [edge, 5.0, 2.0],
            [0.0, 5.0, 3.0],
            [edge - 0.3, -1.0e-9, edge],
            [0.5, 0.2, 0.0],
            // Unwrapped atoms, several box lengths away.
            [3.0 * edge + 1.0, 5.0, 5.0],
            [-4.0 * edge + 2.0, 5.0, 5.0],
        ];
        // Plus a deterministic cloud displaced by whole box lengths.
        let mut rng = TestRng(5);
        for _ in 0..200 {
            let mut pos = [0.0_f32; 3];
            for coordinate in pos.iter_mut() {
                let wraps = (rng.next_unit() * 7.0).floor() - 3.0;
                *coordinate = rng.next_unit() * edge + wraps * edge;
            }
            positions.push(pos);
        }
        let dims = PeriodicCellGrid::choose_dims(positions.len(), &params)
            .expect("the cell list must apply to this configuration");
        assert_eq!(dims, [4; 3]);
        assert_grid_covers_all_pairs(&positions, &params, dims);
    }

    #[test]
    fn test_cell_list_forces_match_all_pairs_reference() {
        let buf = lattice_buffer(8, 500, 1.25, 0.15, 21);
        let positions = positions_of(&buf);
        for cutoff in [2.5_f32, 2.4, 1.9] {
            let params = cubic_params(10.0, cutoff, 500);
            assert!(PeriodicCellGrid::choose_dims(500, &params).is_some());
            let mut fast = buf.clone();
            compute_forces_gpu(&mut fast, &params);
            let reference = reference_forces(&positions, &params);
            let tolerance = 1e-5 * force_scale(&reference);
            for (atom, expected) in fast.atoms.iter().zip(reference.iter()) {
                for (axis, &want) in expected.iter().enumerate() {
                    assert!(
                        (atom.force[axis] - want).abs() <= tolerance,
                        "cutoff {cutoff}: force {:?} vs reference {expected:?}",
                        atom.force
                    );
                }
            }
        }
    }

    #[test]
    fn test_cell_list_forces_match_reference_for_unwrapped_positions() {
        // Atoms that drifted out of the box (the integrator does not wrap).
        let mut buf = lattice_buffer(8, 500, 1.25, 0.15, 33);
        let mut rng = TestRng(77);
        for atom in buf.atoms.iter_mut() {
            for coordinate in atom.pos.iter_mut() {
                *coordinate += ((rng.next_unit() * 5.0).floor() - 2.0) * 10.0;
            }
        }
        let positions = positions_of(&buf);
        let params = cubic_params(10.0, 2.4, 500);
        let mut fast = buf.clone();
        compute_forces_gpu(&mut fast, &params);
        let reference = reference_forces(&positions, &params);
        // Coordinates shifted by whole boxes carry f32 rounding of that
        // magnitude in the minimum-image arithmetic of both paths.
        let tolerance = 1e-4 * force_scale(&reference);
        for (atom, expected) in fast.atoms.iter().zip(reference.iter()) {
            for (axis, &want) in expected.iter().enumerate() {
                assert!((atom.force[axis] - want).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn test_cell_list_potential_energy_matches_reference() {
        let buf = lattice_buffer(8, 500, 1.25, 0.15, 44);
        let positions = positions_of(&buf);
        for cutoff in [2.5_f32, 2.4, 1.9] {
            let params = cubic_params(10.0, cutoff, 500);
            let fast = potential_energy_gpu(&buf, &params);
            let reference = reference_energy(&positions, &params);
            assert!(reference < 0.0, "a lattice solid is bound: {reference}");
            assert!(
                (fast - reference).abs() <= 1e-4 * reference.abs(),
                "cutoff {cutoff}: energy {fast} vs reference {reference}"
            );
        }
    }

    #[test]
    fn test_cell_list_total_force_vanishes() {
        let buf = lattice_buffer(8, 500, 1.25, 0.15, 55);
        let params = cubic_params(10.0, 2.4, 500);
        let mut fast = buf.clone();
        compute_forces_gpu(&mut fast, &params);
        let mut total = [0.0_f32; 3];
        let mut largest = 0.0_f32;
        for atom in &fast.atoms {
            for (sum, &component) in total.iter_mut().zip(atom.force.iter()) {
                *sum += component;
                largest = largest.max(component.abs());
            }
        }
        assert!(largest > 0.1, "the system must actually be interacting");
        for sum in total {
            assert!(
                sum.abs() < 1e-3,
                "net force {total:?} violates Newton's 3rd law"
            );
        }
    }

    #[test]
    fn test_all_pairs_fallback_for_non_periodic_axis() {
        // A non-positive edge means "not periodic along this axis": no
        // wrapping there, and a wrapped grid cannot be used.
        let buf = lattice_buffer(4, 60, 1.3, 0.1, 66);
        let mut params = cubic_params(5.2, 2.0, 60);
        params.box_size[1] = 0.0;
        assert!(PeriodicCellGrid::choose_dims(60, &params).is_none());
        let positions = positions_of(&buf);
        let mut fast = buf.clone();
        compute_forces_gpu(&mut fast, &params);
        let reference = reference_forces(&positions, &params);
        let tolerance = 1e-5 * force_scale(&reference);
        for (atom, expected) in fast.atoms.iter().zip(reference.iter()) {
            for (axis, &want) in expected.iter().enumerate() {
                assert!((atom.force[axis] - want).abs() <= tolerance);
            }
        }
        let energy = potential_energy_gpu(&buf, &params);
        let expected_energy = reference_energy(&positions, &params);
        assert!((energy - expected_energy).abs() <= 1e-4 * expected_energy.abs());
    }

    #[test]
    fn test_cell_list_handles_non_finite_positions_without_panicking() {
        let mut buf = lattice_buffer(8, 100, 1.25, 0.1, 88);
        buf.atoms[3].pos = [f32::NAN, 1.0, 1.0];
        buf.atoms[4].pos = [f32::INFINITY, 1.0, 1.0];
        buf.atoms[5].pos = [1.0, f32::NEG_INFINITY, 1.0];
        let params = cubic_params(10.0, 2.4, 100);
        compute_forces_gpu(&mut buf, &params);
        let _ = potential_energy_gpu(&buf, &params);
        assert_eq!(buf.atoms.len(), 100);
    }

    #[test]
    fn test_cell_list_dynamics_conserve_momentum() {
        // A few velocity steps of a 125-atom lattice at the cell-list sizes.
        let mut buf = lattice_buffer(5, 125, 2.0, 0.1, 99);
        let params = cubic_params(10.0, 2.4, 125);
        assert!(PeriodicCellGrid::choose_dims(125, &params).is_some());
        for _ in 0..5 {
            compute_forces_gpu(&mut buf, &params);
            verlet_integrate_gpu(&mut buf, 1e-3);
        }
        let mut momentum = [0.0_f32; 3];
        for atom in &buf.atoms {
            for (sum, &v) in momentum.iter_mut().zip(atom.vel.iter()) {
                *sum += atom.mass * v;
            }
        }
        for component in momentum {
            assert!(component.abs() < 1e-3, "momentum drift {momentum:?}");
        }
        assert_eq!(buf.step, 5);
    }
}
