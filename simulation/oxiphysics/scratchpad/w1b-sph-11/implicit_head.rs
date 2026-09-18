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
//! The linear algebra (`CsrMatrix`, [`pcg_solve`]) is reused from the ISPH
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
    /// Convert to the CSR matrix consumed by [`pcg_solve`].
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
