// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Implicit / semi-implicit SPH viscosity solver (Takahashi et al. 2015).
//!
//! The explicit viscous update `v* = vⁿ + Δt·(ν∇²v)` is conditionally stable,
//! limited to `Δt < h²/(2·d·ν)` (d = spatial dimension).  For viscous-dominated
//! flows this bound is far smaller than the advective CFL limit, so explicit
//! integration wastes work.  The implicit update instead solves the linear
//! system
//!
//! ```text
//! (I − Δt·L) vⁿ⁺¹ = vⁿ
//! ```
//!
//! where `L` is the SPH viscous Laplacian **including the viscosity**: `L v` is
//! the viscous acceleration `ν∇²v`, in units of `m/s²` per `m/s`, i.e. `1/s`.
//! There is no further scalar `ν` in the system; the per-particle dynamic
//! viscosity `μ_i = ρ_i·ν_i` carried by [`ViscosityParticles`] sets the strength.
//!
//! # Discrete operator (Morris / Brookshaw)
//!
//! The Morris et al. (1997) symmetric second-derivative estimator gives, for
//! particle `i` with neighbour `j` (with `r_ij = x_i − x_j`,
//! `∇W_ij = ∇_i W(r_ij, h)`):
//!
//! ```text
//! (L v)_i = Σ_j c_ij (v_j − v_i),
//! c_ij = (m_j / ρ_j) · (μ_i + μ_j) / ρ_i · (−(r_ij · ∇W_ij)) / (|r_ij|² + ε h²).
//! ```
//!
//! For `μ_i = μ_j = ρν` this is `c_ij = 2ν (m_j/ρ_j) g_ij` with
//! `g_ij = −(r_ij·∇W_ij)/(|r_ij|²+εh²)`, the Brookshaw estimate of `ν∇²v`.  The
//! kernel gradient `∇W_ij = (dW/dr) r_ij / |r_ij|` points opposite to `r_ij` (the
//! kernel decreases outward, `dW/dr < 0`), so `r_ij · ∇W_ij < 0` and the leading
//! `−` makes `c_ij ≥ 0`.
//!
//! # Symmetric positive-definite form
//!
//! `c_ij` itself is *not* symmetric when the masses differ
//! (`c_ij ≠ c_ji`), so `I − Δt·L` is not symmetric and conjugate gradients is not
//! applicable to it.  The pair weight
//!
//! ```text
//! w_ij = m_i c_ij = m_i m_j (μ_i + μ_j) / (ρ_i ρ_j) · g_ij = w_ji
//! ```
//!
//! is symmetric, however.  Scaling row `i` of the system by `m_i` (which does not
//! change its solution) gives
//!
//! ```text
//! (M + Δt·K) vⁿ⁺¹ = M vⁿ,     M = diag(m_i),
//! K_ii = Σ_j w_ij,   K_ij = −w_ij,
//! ```
//!
//! i.e. `A_ii = m_i + Δt Σ_j w_ij` and `A_ij = −Δt w_ij`: a symmetric,
//! positive-definite M-matrix with `A_ii = m_i + Σ_{j≠i} |A_ij|`.  The three
//! velocity components are decoupled and share the same matrix `A`, so we
//! assemble `A` once and run three CG solves with right-hand sides `m_i vⁿ_x`,
//! `m_i vⁿ_y`, `m_i vⁿ_z`.  Because `K` has zero row sums and is symmetric the
//! update conserves the linear momentum `Σ m_i v_i` exactly (up to the CG
//! tolerance), and it never increases the kinetic energy `Σ ½ m_i |v_i|²`.
//!
//! The pair weights are symmetrised (`w_ij ← (w_ij + w_ji)/2`), which removes
//! round-off asymmetry; neighbour lists must therefore be symmetric (`j` listed
//! for `i` exactly when `i` is listed for `j`), otherwise
//! [`ViscosityError::AsymmetricNeighbors`] is returned.
//!
//! The linear algebra (`CsrMatrix`, `pcg_solve`) is reused from the ISPH
//! module — self-contained, pure Rust, no external sparse-solver dependency.

use crate::isph::{CsrMatrix, pcg_solve};

/// Small dimensionless regulariser for the Morris denominator `|r_ij|² + ε h²`.
///
/// Matches the value used across the explicit viscosity discretisations and
/// prevents the singularity as `r_ij → 0`.
const MORRIS_EPSILON: f64 = 0.01;

/// One neighbour interaction for the implicit viscosity stencil.
///
/// Precomputing `r_ij` and `∇W_ij` keeps the assembly independent of any
/// particular kernel or neighbour-search implementation: callers fill these in
/// however they already compute SPH interactions.  The neighbour lists must be
/// symmetric: if `neighbors[i]` holds `(j, r_ij, ∇_iW_ij)` then `neighbors[j]`
/// must hold `(i, −r_ij, −∇_iW_ij)`.
#[derive(Debug, Clone, Copy)]
pub struct NeighborEntry {
    /// Index `j` of the neighbouring particle.
    pub j: usize,
    /// Displacement `r_ij = x_i − x_j`.
    pub r_ij: [f64; 3],
    /// Kernel gradient `∇W_ij = ∇_i W(r_ij, h)` (points opposite to `r_ij`).
    pub grad_w_ij: [f64; 3],
}

impl NeighborEntry {
    /// Convenience constructor.
    pub fn new(j: usize, r_ij: [f64; 3], grad_w_ij: [f64; 3]) -> Self {
        Self { j, r_ij, grad_w_ij }
    }
}

/// Per-particle state consumed by [`solve_implicit_viscosity`].
///
/// Structure-of-references rather than owned data: the caller keeps its own
/// SoA / AoS storage and lends read-only slices for the duration of the solve.
#[derive(Debug, Clone, Copy)]
pub struct ViscosityParticles<'a> {
    /// Current velocities `[vx, vy, vz]` per particle (the system right-hand
    /// side).
    pub velocities: &'a [[f64; 3]],
    /// Per-particle density `ρ_i` (kg/m³).
    pub densities: &'a [f64],
    /// Per-particle mass `m_i` (kg).
    pub masses: &'a [f64],
    /// Per-particle **dynamic** viscosity `μ_i = ρ_i · ν_i` (Pa·s).  This is the
    /// only place the viscosity enters the solve.
    pub viscosities: &'a [f64],
}

/// Errors that can arise during an implicit viscosity solve.
#[derive(Debug, thiserror::Error)]
pub enum ViscosityError {
    /// CG failed to reach the requested tolerance within the iteration budget.
    #[error("implicit viscosity CG did not converge: {iters} iters, residual {residual:.3e}")]
    SolverNotConverged {
        /// Iterations performed before giving up.
        iters: usize,
        /// Final relative residual `‖b − Ax‖ / ‖b‖`.
        residual: f64,
    },
    /// The particle list was empty.
    #[error("empty particle list")]
    EmptyParticleList,
    /// Slice lengths (`velocities`, `densities`, `masses`, `viscosities`,
    /// `neighbors`) did not all match.
    #[error("inconsistent particle slice lengths")]
    InconsistentLengths,
    /// `dt` or `h` was not finite and strictly positive, a density or mass was
    /// not finite and strictly positive, a viscosity was negative or not
    /// finite, or a neighbour entry had a kernel gradient pointing along
    /// `r_ij` (`r_ij · ∇W_ij > 0`, a kernel that increases outward).
    #[error(
        "dt and h must be finite and > 0, densities and masses finite and > 0, viscosities finite \
         and >= 0, and kernel gradients must point against r_ij"
    )]
    InvalidConfig,
    /// The neighbour lists are not symmetric: particle `i` lists `j` as a
    /// neighbour (with a non-zero coupling) but `j` does not list `i`.
    #[error("neighbour lists are not symmetric: particle {i} lists {j} but {j} does not list {i}")]
    AsymmetricNeighbors {
        /// Particle whose list contains the unmatched entry.
        i: usize,
        /// Neighbour that lacks the reciprocal entry.
        j: usize,
    },
}

/// Relative residual tolerance and iteration cap for the implicit solve.
#[derive(Debug, Clone, Copy)]
pub struct ViscositySolveOptions {
    /// Relative residual tolerance for each component CG solve.
    pub tol: f64,
    /// Maximum CG iterations per component.
    pub max_iters: usize,
}

impl Default for ViscositySolveOptions {
    fn default() -> Self {
        Self {
            tol: 1.0e-8,
            max_iters: 2000,
        }
    }
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the Morris coupling `c_ij ≥ 0` (units `1/s`) for one neighbour entry.
///
/// The coupling already contains the viscosity through `μ_i + μ_j`: it is the
/// weight with which `v_j − v_i` contributes to the viscous acceleration `ν∇²v`
/// of particle `i`.  Returns `0` for a coincident pair (`|r_ij|² ≈ 0`) where the
/// stencil is not defined.  `h` enters only through the `ε h²` regulariser.
fn morris_coupling(
    entry: &NeighborEntry,
    mu_i: f64,
    mu_j: f64,
    rho_i: f64,
    rho_j: f64,
    m_j: f64,
    h: f64,
) -> f64 {
    let r2 = dot3(entry.r_ij, entry.r_ij);
    if r2 < 1.0e-24 || rho_i <= 0.0 || rho_j <= 0.0 {
        return 0.0;
    }
    let denom = r2 + MORRIS_EPSILON * h * h;
    // r_ij · ∇W_ij ≤ 0, so the leading `-` yields c_ij ≥ 0.
    let geom = -dot3(entry.r_ij, entry.grad_w_ij) / denom;
    (m_j / rho_j) * (mu_i + mu_j) / rho_i * geom
}

/// The assembled symmetric system `A = M̂ + Δt·K` in row form.
///
/// Masses are normalised by the largest mass (`m̂_i = m_i / max_k m_k`), a global
/// scaling of the whole system that leaves its solution unchanged but keeps the
/// entries of `A` well inside the floating-point range for any mass unit.
struct ViscositySystem {
    /// Off-diagonal entries `A_ij = −Δt w_ij` of every row, sorted by column.
    off_diagonals: Vec<Vec<(usize, f64)>>,
    /// Diagonal `A_ii = m̂_i + Δt Σ_j w_ij`.
    diagonal: Vec<f64>,
    /// Normalised masses `m̂_i`: the row scaling and the weight of the
    /// right-hand side `m̂_i vⁿ_i`.
    scaled_masses: Vec<f64>,
}

impl ViscositySystem {
    /// Convert to the CSR matrix consumed by `pcg_solve`.
    fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_rows(self.diagonal.len(), &self.off_diagonals, &self.diagonal)
    }
}

/// Check the inputs of [`solve_implicit_viscosity`] and return the largest mass.
fn validate_inputs(
    particles: &ViscosityParticles<'_>,
    neighbors: &[Vec<NeighborEntry>],
    dt: f64,
    h: f64,
) -> Result<f64, ViscosityError> {
    let n = particles.velocities.len();
    if n == 0 {
        return Err(ViscosityError::EmptyParticleList);
    }
    if particles.densities.len() != n
        || particles.masses.len() != n
        || particles.viscosities.len() != n
        || neighbors.len() != n
    {
        return Err(ViscosityError::InconsistentLengths);
    }
    let positive_finite = |x: f64| x.is_finite() && x > 0.0;
    if !positive_finite(dt) || !positive_finite(h) {
        return Err(ViscosityError::InvalidConfig);
    }
    if !particles.densities.iter().all(|&rho| positive_finite(rho))
        || !particles.masses.iter().all(|&m| positive_finite(m))
        || !particles
            .viscosities
            .iter()
            .all(|&mu| mu.is_finite() && mu >= 0.0)
    {
        return Err(ViscosityError::InvalidConfig);
    }
    Ok(particles.masses.iter().copied().fold(0.0_f64, f64::max))
}

/// Symmetrised pair weights `w_ij = m̂_i c_ij` of every row, sorted by column.
///
/// Duplicate neighbour entries for the same `j` are accumulated into a single
/// weight.  Each weight is replaced by the mean of `w_ij` and `w_ji`, so the
/// result is exactly symmetric; a pair whose reciprocal entry is missing is
/// reported as [`ViscosityError::AsymmetricNeighbors`].
fn symmetric_pair_weights(
    particles: &ViscosityParticles<'_>,
    neighbors: &[Vec<NeighborEntry>],
    scaled_masses: &[f64],
    h: f64,
) -> Result<Vec<Vec<(usize, f64)>>, ViscosityError> {
    let n = particles.velocities.len();
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let mu_i = particles.viscosities[i];
        let rho_i = particles.densities[i];
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(neighbors[i].len());
        for entry in &neighbors[i] {
            let j = entry.j;
            if j == i || j >= n {
                continue;
            }
            let c = morris_coupling(
                entry,
                mu_i,
                particles.viscosities[j],
                rho_i,
                particles.densities[j],
                particles.masses[j],
                h,
            );
            if c == 0.0 {
                continue;
            }
            // A physical kernel decreases outward (`r_ij · ∇W_ij < 0`).  A
            // negative coupling would make the system indefinite and CG
            // meaningless, so reject it instead of returning garbage.
            if c < 0.0 {
                return Err(ViscosityError::InvalidConfig);
            }
            row.push((j, scaled_masses[i] * c));
        }
        // Accumulate duplicate neighbours into one weight per column.
        row.sort_by_key(|&(j, _)| j);
        row.dedup_by(|later, kept| {
            if later.0 == kept.0 {
                kept.1 += later.1;
                true
            } else {
                false
            }
        });
        rows.push(row);
    }
    let mut symmetric: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for (i, row) in rows.iter().enumerate() {
        let mut sym_row = Vec::with_capacity(row.len());
        for &(j, w_ij) in row {
            let w_ji = rows[j]
                .binary_search_by_key(&i, |&(col, _)| col)
                .map(|pos| rows[j][pos].1)
                .map_err(|_| ViscosityError::AsymmetricNeighbors { i, j })?;
            sym_row.push((j, 0.5 * (w_ij + w_ji)));
        }
        symmetric.push(sym_row);
    }
    Ok(symmetric)
}

/// Assemble the implicit-viscosity system `A = M̂ + Δt·K` (see the module
/// documentation): off-diagonal `A_ij = −Δt·w_ij`, diagonal
/// `A_ii = m̂_i + Δt·Σ_j w_ij`.
fn assemble_system(
    particles: &ViscosityParticles<'_>,
    neighbors: &[Vec<NeighborEntry>],
    dt: f64,
    h: f64,
) -> Result<ViscositySystem, ViscosityError> {
    let max_mass = validate_inputs(particles, neighbors, dt, h)?;
    let scaled_masses: Vec<f64> = particles.masses.iter().map(|&m| m / max_mass).collect();
    let weights = symmetric_pair_weights(particles, neighbors, &scaled_masses, h)?;
    let mut diagonal = scaled_masses.clone();
    let mut off_diagonals: Vec<Vec<(usize, f64)>> = Vec::with_capacity(weights.len());
    for (i, row) in weights.iter().enumerate() {
        let mut off_row = Vec::with_capacity(row.len());
        for &(j, w) in row {
            diagonal[i] += dt * w; // A_ii = m̂_i + Δt Σ_j w_ij
            off_row.push((j, -dt * w)); // A_ij = −Δt w_ij
        }
        off_diagonals.push(off_row);
    }
    Ok(ViscositySystem {
        off_diagonals,
        diagonal,
        scaled_masses,
    })
}

/// Implicit viscosity update via Jacobi-preconditioned conjugate gradient.
///
/// Solves `(I − Δt·L) vⁿ⁺¹ = vⁿ` for the new velocity field, where `L` is the
/// Morris symmetric SPH viscous Laplacian (see the module documentation).  `L`
/// already contains the viscosity, so the strength of the damping is set solely
/// by `particles.viscosities`: a mode of wave number `k` decays at about
/// `ν k²` with `ν = μ/ρ`.  The system is solved in its symmetric form
/// `(M + Δt·K) vⁿ⁺¹ = M vⁿ`; the three velocity components share the assembled
/// SPD matrix and are solved independently.  The update conserves `Σ m_i v_i`
/// and never increases the kinetic energy.
///
/// # Arguments
/// - `particles`: current velocities, densities, masses and dynamic
///   viscosities (`μ_i = ρ_i·ν_i`).
/// - `neighbors`: `neighbors[i]` lists every interaction `(j, r_ij, ∇W_ij)` of
///   particle `i`; the lists must be symmetric.
/// - `dt`: time step `Δt` (s).
/// - `h`: smoothing length (only used in the `ε h²` regulariser).
/// - `options`: CG tolerance and iteration cap.
///
/// # Errors
/// - [`ViscosityError::EmptyParticleList`] if there are no particles.
/// - [`ViscosityError::InconsistentLengths`] if the slices disagree in length.
/// - [`ViscosityError::InvalidConfig`] if `dt` or `h` is not finite and
///   positive, a density or mass is not finite and positive, a viscosity is
///   negative or not finite, or a kernel gradient points along `r_ij`.
/// - [`ViscosityError::AsymmetricNeighbors`] if a neighbour entry has no
///   reciprocal entry in the other particle's list.
/// - [`ViscosityError::SolverNotConverged`] if any component CG exceeds the
///   iteration budget without reaching `options.tol`.
///
/// # Returns
/// `Ok(velocities)` with the updated `[vx, vy, vz]` for every particle, indexed
/// to match `particles.velocities`.
pub fn solve_implicit_viscosity(
    particles: &ViscosityParticles<'_>,
    neighbors: &[Vec<NeighborEntry>],
    dt: f64,
    h: f64,
    options: ViscositySolveOptions,
) -> Result<Vec<[f64; 3]>, ViscosityError> {
    let system = assemble_system(particles, neighbors, dt, h)?;
    let a = system.to_csr();

    let n = particles.velocities.len();
    let mut result = vec![[0.0_f64; 3]; n];
    let mut rhs = vec![0.0_f64; n];
    let mut sol = vec![0.0_f64; n];
    // The x/y/z velocity components are decoupled and share the SPD matrix `a`,
    // so solve them one at a time, scattering each solution back into `result`.
    for comp in 0..3 {
        for (i, (r, s)) in rhs.iter_mut().zip(sol.iter_mut()).enumerate() {
            *r = system.scaled_masses[i] * particles.velocities[i][comp];
            *s = particles.velocities[i][comp]; // warm start
        }
        let (iters, residual) = pcg_solve(&a, &rhs, &mut sol, options.tol, options.max_iters);
        // A zero right-hand side legitimately returns residual 0; only a residual
        // above tolerance (or a non-finite one) is a genuine non-convergence.
        if !residual.is_finite() || residual > options.tol {
            return Err(ViscosityError::SolverNotConverged { iters, residual });
        }
        for (out, &si) in result.iter_mut().zip(sol.iter()) {
            out[comp] = si;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{CubicSplineKernel, grad};

    /// Build a neighbour list for a set of positions using the cubic-spline
    /// kernel and support radius `2h`.  Shared by several tests.
    fn build_neighbors(positions: &[[f64; 3]], h: f64) -> Vec<Vec<NeighborEntry>> {
        let kernel = CubicSplineKernel;
        let support = 2.0 * h;
        let n = positions.len();
        let mut neighbors = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_ij = [
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                    positions[i][2] - positions[j][2],
                ];
                let r = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]).sqrt();
                if r >= support || r < 1.0e-12 {
                    continue;
                }
                let gw = grad(&kernel, r_ij, h);
                neighbors[i].push(NeighborEntry::new(j, r_ij, gw));
            }
        }
        neighbors
    }

    /// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
    fn lcg_unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
    }

    /// [`solve_implicit_viscosity`] with the default options.
    fn solve_default(
        particles: &ViscosityParticles<'_>,
        neighbors: &[Vec<NeighborEntry>],
        dt: f64,
        h: f64,
    ) -> Result<Vec<[f64; 3]>, ViscosityError> {
        solve_implicit_viscosity(
            particles,
            neighbors,
            dt,
            h,
            ViscositySolveOptions::default(),
        )
    }

    /// Owned particle data for tests.
    struct Fixture {
        velocities: Vec<[f64; 3]>,
        densities: Vec<f64>,
        masses: Vec<f64>,
        viscosities: Vec<f64>,
        neighbors: Vec<Vec<NeighborEntry>>,
        h: f64,
    }

    impl Fixture {
        fn particles(&self) -> ViscosityParticles<'_> {
            ViscosityParticles {
                velocities: &self.velocities,
                densities: &self.densities,
                masses: &self.masses,
                viscosities: &self.viscosities,
            }
        }

        fn solve(&self, dt: f64) -> Result<Vec<[f64; 3]>, ViscosityError> {
            let options = ViscositySolveOptions {
                tol: 1.0e-12,
                max_iters: 5000,
            };
            solve_implicit_viscosity(&self.particles(), &self.neighbors, dt, self.h, options)
        }
    }

    /// Five fully connected particles (a cross) with different masses,
    /// densities and viscosities and pseudo-random velocities.
    fn unequal_cross() -> Fixture {
        let h = 0.1;
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, -0.08, 0.0],
        ];
        let densities = vec![900.0, 1100.0, 1000.0, 1200.0, 800.0];
        let masses = vec![1.0, 2.0, 0.5, 3.0, 1.5];
        let kinematic = [0.01, 0.02, 0.005, 0.03, 0.015];
        let viscosities: Vec<f64> = densities
            .iter()
            .zip(kinematic.iter())
            .map(|(rho, nu)| rho * nu)
            .collect();
        let mut state = 0xdead_beef_u64;
        let velocities: Vec<[f64; 3]> = (0..positions.len())
            .map(|_| {
                [
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                    lcg_unit(&mut state),
                ]
            })
            .collect();
        let neighbors = build_neighbors(&positions, h);
        Fixture {
            velocities,
            densities,
            masses,
            viscosities,
            neighbors,
            h,
        }
    }

    #[test]
    fn coupling_is_non_negative() {
        // r_ij · ∇W_ij < 0 ⇒ c_ij ≥ 0 for any positive masses/densities.
        let kernel = CubicSplineKernel;
        let h = 0.1;
        let r_ij = [0.08_f64, 0.0, 0.0];
        let gw = grad(&kernel, r_ij, h);
        let entry = NeighborEntry::new(1, r_ij, gw);
        let c = morris_coupling(&entry, 1.0, 1.0, 1000.0, 1000.0, 0.001, h);
        assert!(c >= 0.0, "Morris coupling must be non-negative, got {c}");
        assert!(c > 0.0, "coupling should be strictly positive for r < 2h");
    }

    #[test]
    fn coupling_zero_for_coincident() {
        let entry = NeighborEntry::new(1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let c = morris_coupling(&entry, 1.0, 1.0, 1000.0, 1000.0, 0.001, 0.1);
        assert_eq!(c, 0.0, "coincident particles contribute nothing");
    }

    #[test]
    fn coupling_already_carries_the_kinematic_viscosity() {
        // c_ij = 2 ν (m_j/ρ_j) g_ij for μ = ρ ν: linear in ν, with units 1/s.
        let kernel = CubicSplineKernel;
        let h = 0.1;
        let r_ij = [0.06_f64, 0.0, 0.0];
        let entry = NeighborEntry::new(1, r_ij, grad(&kernel, r_ij, h));
        let rho = 1000.0;
        let m = 0.002;
        let c_low = morris_coupling(&entry, rho * 0.01, rho * 0.01, rho, rho, m, h);
        let c_high = morris_coupling(&entry, rho * 0.04, rho * 0.04, rho, rho, m, h);
        assert!((c_high / c_low - 4.0).abs() < 1e-12);
        let g = -dot3(r_ij, entry.grad_w_ij) / (dot3(r_ij, r_ij) + MORRIS_EPSILON * h * h);
        let expected = 2.0 * 0.01 * (m / rho) * g;
        assert!(
            (c_low - expected).abs() < 1e-12 * expected,
            "{c_low} vs {expected}"
        );
    }

    #[test]
    fn empty_particle_list_errors() {
        let particles = ViscosityParticles {
            velocities: &[],
            densities: &[],
            masses: &[],
            viscosities: &[],
        };
        let neighbors: Vec<Vec<NeighborEntry>> = Vec::new();
        let err = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.01,
            0.1,
            ViscositySolveOptions::default(),
        );
        assert!(matches!(err, Err(ViscosityError::EmptyParticleList)));
    }

    #[test]
    fn inconsistent_lengths_error() {
        let velocities = vec![[0.0; 3]; 2];
        let densities = vec![1000.0; 2];
        let masses = vec![0.001; 2];
        let viscosities = vec![1.0]; // wrong length
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let neighbors = vec![Vec::new(); 2];
        let err = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.01,
            0.1,
            ViscositySolveOptions::default(),
        );
        assert!(matches!(err, Err(ViscosityError::InconsistentLengths)));
    }

    #[test]
    fn invalid_config_errors() {
        let velocities = vec![[0.0; 3]; 2];
        let densities = vec![1000.0; 2];
        let masses = vec![0.001; 2];
        let viscosities = vec![1.0; 2];
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let neighbors = vec![Vec::new(); 2];
        let solve =
            |p: &ViscosityParticles<'_>, dt: f64, h: f64| solve_default(p, &neighbors, dt, h);
        // dt <= 0, dt not finite
        assert!(matches!(
            solve(&particles, 0.0, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
        assert!(matches!(
            solve(&particles, -1.0, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
        assert!(matches!(
            solve(&particles, f64::NAN, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
        // h <= 0
        assert!(matches!(
            solve(&particles, 0.01, 0.0),
            Err(ViscosityError::InvalidConfig)
        ));
        // negative viscosity
        let negative_mu = vec![1.0, -1.0];
        let bad = ViscosityParticles {
            viscosities: &negative_mu,
            ..particles
        };
        assert!(matches!(
            solve(&bad, 0.01, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
        // non-positive mass
        let zero_mass = vec![0.001, 0.0];
        let bad = ViscosityParticles {
            masses: &zero_mass,
            ..particles
        };
        assert!(matches!(
            solve(&bad, 0.01, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
        // non-positive density
        let zero_density = vec![1000.0, 0.0];
        let bad = ViscosityParticles {
            densities: &zero_density,
            ..particles
        };
        assert!(matches!(
            solve(&bad, 0.01, 0.1),
            Err(ViscosityError::InvalidConfig)
        ));
    }

    #[test]
    fn kernel_gradient_pointing_along_r_ij_is_rejected() {
        // A kernel that increases outward would give a negative coupling and an
        // indefinite system; it must not be handed to CG.
        let velocities = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let densities = vec![1000.0; 2];
        let masses = vec![1.0; 2];
        let viscosities = vec![1000.0; 2];
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let neighbors = vec![
            vec![NeighborEntry::new(1, [0.05, 0.0, 0.0], [5.0, 0.0, 0.0])],
            vec![NeighborEntry::new(0, [-0.05, 0.0, 0.0], [-5.0, 0.0, 0.0])],
        ];
        let err = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.01,
            0.1,
            ViscositySolveOptions::default(),
        );
        assert!(matches!(err, Err(ViscosityError::InvalidConfig)), "{err:?}");
    }

    #[test]
    fn isolated_particle_velocity_unchanged() {
        // No neighbours ⇒ A = M̂ ⇒ vⁿ⁺¹ = vⁿ.
        let velocities = vec![[1.0, -2.0, 3.0]];
        let densities = vec![1000.0];
        let masses = vec![0.001];
        let viscosities = vec![1.0];
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let neighbors = vec![Vec::new()];
        let out = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.01,
            0.1,
            ViscositySolveOptions::default(),
        )
        .expect("solve should succeed");
        for k in 0..3 {
            assert!(
                (out[0][k] - velocities[0][k]).abs() < 1e-12,
                "isolated particle velocity must be unchanged"
            );
        }
    }

    #[test]
    fn two_particles_velocities_relax_together() {
        // Two equal particles with opposite velocities: viscous diffusion must
        // pull them toward their mean (here zero), conserving total momentum.
        let h = 0.1;
        let positions = vec![[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]];
        let velocities = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let densities = vec![1000.0, 1000.0];
        let masses = vec![1.0, 1.0];
        let viscosities = vec![1000.0, 1000.0]; // μ = ρ·ν, ν = 1
        let neighbors = build_neighbors(&positions, h);
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let out = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.5,
            h,
            ViscositySolveOptions::default(),
        )
        .expect("solve should succeed");
        // Magnitudes shrink toward the mean.
        assert!(out[0][0] < velocities[0][0], "particle 0 vx must decrease");
        assert!(out[1][0] > velocities[1][0], "particle 1 vx must increase");
        // Equal-mass momentum is conserved by the symmetric operator.
        let p_before = masses[0] * velocities[0][0] + masses[1] * velocities[1][0];
        let p_after = masses[0] * out[0][0] + masses[1] * out[1][0];
        assert!(
            (p_before - p_after).abs() < 1e-9,
            "momentum drift {}",
            (p_before - p_after).abs()
        );
    }

    #[test]
    fn unscaled_operator_is_asymmetric_for_unequal_masses() {
        // This is the defect the symmetric form removes: `c_ij != c_ji` when the
        // masses differ, although `m_i c_ij == m_j c_ji`.
        let fixture = unequal_cross();
        let particles = fixture.particles();
        let coupling = |i: usize, j: usize| -> f64 {
            fixture.neighbors[i]
                .iter()
                .find(|entry| entry.j == j)
                .map(|entry| {
                    morris_coupling(
                        entry,
                        particles.viscosities[i],
                        particles.viscosities[j],
                        particles.densities[i],
                        particles.densities[j],
                        particles.masses[j],
                        fixture.h,
                    )
                })
                .unwrap_or(0.0)
        };
        let c_01 = coupling(0, 1);
        let c_10 = coupling(1, 0);
        assert!(c_01 > 0.0 && c_10 > 0.0);
        assert!(
            (c_01 - c_10).abs() > 0.1 * c_01.max(c_10),
            "{c_01} vs {c_10}"
        );
        let w_01 = particles.masses[0] * c_01;
        let w_10 = particles.masses[1] * c_10;
        assert!((w_01 - w_10).abs() < 1e-12 * w_01, "{w_01} vs {w_10}");
    }

    #[test]
    fn system_matrix_is_exactly_symmetric_for_unequal_masses() {
        let fixture = unequal_cross();
        let dt = 0.05;
        let system = assemble_system(&fixture.particles(), &fixture.neighbors, dt, fixture.h)
            .expect("assembly should succeed");
        let n = fixture.masses.len();
        let mut dense = vec![vec![0.0_f64; n]; n];
        for (i, row) in dense.iter_mut().enumerate() {
            row[i] = system.diagonal[i];
            for &(j, a_ij) in &system.off_diagonals[i] {
                row[j] += a_ij;
            }
        }
        let mut stored_pairs = 0;
        for (i, row) in dense.iter().enumerate() {
            let mut off_sum = 0.0;
            for (j, &a_ij) in row.iter().enumerate() {
                assert_eq!(a_ij, dense[j][i], "A not symmetric at ({i},{j})");
                if i != j {
                    assert!(a_ij <= 0.0, "off-diagonal must be non-positive");
                    off_sum += a_ij.abs();
                    if a_ij < 0.0 {
                        stored_pairs += 1;
                    }
                }
            }
            // M-matrix with A_ii = m̂_i + Σ_j |A_ij|.
            assert!(
                (row[i] - system.scaled_masses[i] - off_sum).abs() < 1e-12 * row[i],
                "row {i}: diag {} vs m + sum|off| {}",
                row[i],
                system.scaled_masses[i] + off_sum
            );
        }
        assert_eq!(stored_pairs, n * (n - 1), "the cross is fully connected");
        // The normalised masses are m_i / max m.
        assert!((system.scaled_masses[3] - 1.0).abs() < 1e-15);
        assert!((system.scaled_masses[2] - 0.5 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn masses_one_and_two_give_a_symmetric_system_and_relax_to_the_momentum_mean() {
        // The configuration of the audit finding: two particles, m = [1, 2].
        let h = 0.1;
        let positions = vec![[0.0, 0.0, 0.0], [0.07, 0.0, 0.0]];
        let fixture = Fixture {
            velocities: vec![[2.0, 0.0, 0.0], [-1.0, 0.5, 0.0]],
            densities: vec![1000.0, 1000.0],
            masses: vec![1.0, 2.0],
            viscosities: vec![10.0, 10.0], // ν = 0.01
            neighbors: build_neighbors(&positions, h),
            h,
        };
        let dt = 0.1;
        let system = assemble_system(&fixture.particles(), &fixture.neighbors, dt, h)
            .expect("assembly should succeed");
        let a_01 = system.off_diagonals[0]
            .iter()
            .find(|&&(j, _)| j == 1)
            .map(|&(_, a)| a)
            .expect("entry (0, 1)");
        let a_10 = system.off_diagonals[1]
            .iter()
            .find(|&&(j, _)| j == 0)
            .map(|&(_, a)| a)
            .expect("entry (1, 0)");
        assert!(a_01 < 0.0);
        assert_eq!(a_01, a_10, "CSR entries (0,1) and (1,0) must be identical");
        // The momentum is conserved exactly by one step ...
        let out = fixture.solve(dt).expect("solve should succeed");
        let momentum =
            |v: &[[f64; 3]], k: usize| fixture.masses[0] * v[0][k] + fixture.masses[1] * v[1][k];
        for k in 0..3 {
            assert!(
                (momentum(&out, k) - momentum(&fixture.velocities, k)).abs() < 1e-10,
                "component {k}"
            );
        }
        // ... and a very long step relaxes both particles to the mass-weighted
        // mean velocity (0, 1/3, 0): the deviation shrinks like 1 / (dt λ).
        let options = ViscositySolveOptions {
            tol: 1.0e-10,
            max_iters: 100,
        };
        let relaxed =
            solve_implicit_viscosity(&fixture.particles(), &fixture.neighbors, 1.0e5, h, options)
                .expect("stiff solve should succeed");
        for v in &relaxed {
            assert!(v[0].abs() < 1e-3, "{relaxed:?}");
            assert!((v[1] - 1.0 / 3.0).abs() < 1e-3, "{relaxed:?}");
            assert!(v[2].abs() < 1e-3, "{relaxed:?}");
        }
    }

    #[test]
    fn unequal_masses_conserve_momentum_and_do_not_add_energy() {
        let fixture = unequal_cross();
        let out = fixture.solve(0.05).expect("solve should succeed");
        let momentum = |v: &[[f64; 3]]| -> [f64; 3] {
            let mut p = [0.0_f64; 3];
            for (vi, &m) in v.iter().zip(fixture.masses.iter()) {
                for (pk, vk) in p.iter_mut().zip(vi.iter()) {
                    *pk += m * vk;
                }
            }
            p
        };
        let energy = |v: &[[f64; 3]]| -> f64 {
            v.iter()
                .zip(fixture.masses.iter())
                .map(|(vi, &m)| 0.5 * m * dot3(*vi, *vi))
                .sum()
        };
        let p_before = momentum(&fixture.velocities);
        let p_after = momentum(&out);
        for k in 0..3 {
            assert!(
                (p_before[k] - p_after[k]).abs() < 1e-9,
                "momentum component {k} drifted: {} -> {}",
                p_before[k],
                p_after[k]
            );
        }
        let e_before = energy(&fixture.velocities);
        let e_after = energy(&out);
        assert!(
            e_after < e_before,
            "viscosity must dissipate: {e_before} -> {e_after}"
        );
    }

    #[test]
    fn solution_satisfies_the_unscaled_implicit_equation() {
        // Row scaling and symmetrisation must not change the solution of
        // (I − Δt L) vⁿ⁺¹ = vⁿ with the physical (asymmetric) c_ij.
        let fixture = unequal_cross();
        let dt = 0.05;
        let out = fixture.solve(dt).expect("solve should succeed");
        let particles = fixture.particles();
        for i in 0..fixture.masses.len() {
            for (k, (&v_ik, &out_ik)) in fixture.velocities[i].iter().zip(out[i].iter()).enumerate()
            {
                let mut laplacian = 0.0;
                for entry in &fixture.neighbors[i] {
                    let j = entry.j;
                    let c = morris_coupling(
                        entry,
                        particles.viscosities[i],
                        particles.viscosities[j],
                        particles.densities[i],
                        particles.densities[j],
                        particles.masses[j],
                        fixture.h,
                    );
                    laplacian += c * (out[j][k] - out_ik);
                }
                let residual = out_ik - dt * laplacian - v_ik;
                assert!(
                    residual.abs() < 1e-8,
                    "particle {i} component {k}: residual {residual}"
                );
            }
        }
    }

    #[test]
    fn one_sided_neighbor_list_is_rejected() {
        let velocities = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let densities = vec![1000.0; 2];
        let masses = vec![1.0; 2];
        let viscosities = vec![1000.0; 2];
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let kernel = CubicSplineKernel;
        let r_01 = [-0.05, 0.0, 0.0];
        let neighbors = vec![
            vec![NeighborEntry::new(1, r_01, grad(&kernel, r_01, 0.1))],
            Vec::new(),
        ];
        let err = solve_implicit_viscosity(
            &particles,
            &neighbors,
            0.01,
            0.1,
            ViscositySolveOptions::default(),
        );
        assert!(
            matches!(err, Err(ViscosityError::AsymmetricNeighbors { i: 0, j: 1 })),
            "{err:?}"
        );
    }

    #[test]
    fn duplicate_entries_accumulate_into_one_coupling() {
        // Listing a pair twice must equal one entry with the summed gradient.
        let kernel = CubicSplineKernel;
        let h = 0.1;
        let r_01 = [-0.06, 0.0, 0.0];
        let g = grad(&kernel, r_01, h);
        let g_neg = [-g[0], -g[1], -g[2]];
        let r_10 = [0.06, 0.0, 0.0];
        let doubled_g = [2.0 * g[0], 2.0 * g[1], 2.0 * g[2]];
        let doubled_g_neg = [-doubled_g[0], -doubled_g[1], -doubled_g[2]];
        let velocities = vec![[1.0, 0.5, 0.0], [-1.0, 0.25, 0.5]];
        let densities = vec![1000.0, 1200.0];
        let masses = vec![0.4, 0.9];
        let viscosities = vec![10.0, 14.0];
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let twice = vec![
            vec![
                NeighborEntry::new(1, r_01, g),
                NeighborEntry::new(1, r_01, g),
            ],
            vec![
                NeighborEntry::new(0, r_10, g_neg),
                NeighborEntry::new(0, r_10, g_neg),
            ],
        ];
        let once = vec![
            vec![NeighborEntry::new(1, r_01, doubled_g)],
            vec![NeighborEntry::new(0, r_10, doubled_g_neg)],
        ];
        let options = ViscositySolveOptions {
            tol: 1.0e-12,
            max_iters: 1000,
        };
        let a = solve_implicit_viscosity(&particles, &twice, 0.1, h, options).expect("twice");
        let b = solve_implicit_viscosity(&particles, &once, 0.1, h, options).expect("once");
        for i in 0..2 {
            for k in 0..3 {
                assert!((a[i][k] - b[i][k]).abs() < 1e-10, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn result_does_not_depend_on_the_mass_unit() {
        // Scaling mass, density and dynamic viscosity by the same factor leaves
        // the physics (volumes m/ρ and ν = μ/ρ) unchanged.  The assembled
        // entries must stay representable even for absurdly small masses.
        let reference = unequal_cross();
        let scale = 1.0e-35;
        let scaled = Fixture {
            velocities: reference.velocities.clone(),
            densities: reference.densities.iter().map(|x| x * scale).collect(),
            masses: reference.masses.iter().map(|x| x * scale).collect(),
            viscosities: reference.viscosities.iter().map(|x| x * scale).collect(),
            neighbors: reference.neighbors.clone(),
            h: reference.h,
        };
        let a = reference.solve(0.05).expect("reference solve");
        let b = scaled.solve(0.05).expect("scaled solve");
        for i in 0..a.len() {
            for k in 0..3 {
                assert!((a[i][k] - b[i][k]).abs() < 1e-9, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn exhausted_iteration_budget_is_reported() {
        let fixture = unequal_cross();
        let options = ViscositySolveOptions {
            tol: 1.0e-14,
            max_iters: 1,
        };
        let err = solve_implicit_viscosity(
            &fixture.particles(),
            &fixture.neighbors,
            5.0,
            fixture.h,
            options,
        );
        assert!(
            matches!(
                err,
                Err(ViscosityError::SolverNotConverged { iters: 1, .. })
            ),
            "{err:?}"
        );
    }
}
