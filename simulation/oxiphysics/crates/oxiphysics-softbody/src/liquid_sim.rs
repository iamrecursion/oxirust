// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Position-Based Fluid (PBF) simulation for liquid dynamics.
//!
//! Implements the Macklin & Müller (2013) position-based fluid method:
//!
//! - Poly6 density kernel and Spiky pressure gradient kernel
//! - Density constraint with XPBD-style lambda multipliers
//! - XSPH viscosity for coherent motion
//! - Surface particle detection via density threshold
//! - Wall and periodic boundary conditions
//! - Spatial-hash neighbour search ([`NeighbourLists`]): O(n) per search, one
//!   search per solver stage instead of one per solver iteration
//!
//! All quantities use SI units.

use oxiphysics_geometry::spatial_hash::SpatialHashPos3D;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Small vector helpers (no nalgebra dependency)
// ---------------------------------------------------------------------------

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn len3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = len3(v);
    if l < 1e-300 {
        [0.0; 3]
    } else {
        scale3(v, 1.0 / l)
    }
}

// ---------------------------------------------------------------------------
// SPH kernels
// ---------------------------------------------------------------------------

/// Müller poly6 smoothing kernel W(r, h).
///
/// W(r, h) = (315 / (64π h⁹)) · (h² - r²)³  for r ≤ h, else 0.
pub fn poly6_kernel(r: f64, h: f64) -> f64 {
    if r < 0.0 || r > h {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = r * r;
    let diff = h2 - r2;
    (315.0 / (64.0 * PI * h.powi(9))) * diff.powi(3)
}

/// Gradient of the Debrun Spiky kernel ∇W(r⃗, h).
///
/// ∇W(r⃗, h) = −(45 / (π h⁶)) · (h − |r⃗|)² · r̂
///
/// Used for pressure / density constraint gradients because it is non-zero
/// at r = 0 (unlike poly6).
pub fn spiky_gradient(r: [f64; 3], h: f64) -> [f64; 3] {
    let mag = len3(r);
    if mag <= 0.0 || mag > h {
        return [0.0; 3];
    }
    let coeff = -(45.0 / (PI * h.powi(6))) * (h - mag).powi(2);
    let dir = norm3(r);
    scale3(dir, coeff)
}

/// Compute the XPBD constraint multiplier λᵢ for particle i.
///
/// λᵢ = −C(ρᵢ) / (Σⱼ |∇ⱼC|² + ε)
///
/// where C(ρᵢ) = ρᵢ/ρ₀ − 1 is the density constraint residual and
/// `grad_sum` = Σⱼ |∇ⱼC|² is the sum of squared gradient magnitudes.
/// A small relaxation parameter ε = 1e-6 is added for stability.
pub fn compute_lambda(density: f64, rest_density: f64, grad_sum: f64) -> f64 {
    let constraint = density / rest_density.max(1e-300) - 1.0;
    -constraint / (grad_sum + 1e-6)
}

// ---------------------------------------------------------------------------
// Boundary condition
// ---------------------------------------------------------------------------

/// Boundary condition type for the simulation domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Reflective wall boundaries defined by axis-aligned box `[min, max]`.
    Walls {
        /// Minimum corner of the bounding box (x, y, z).
        min: [f64; 3],
        /// Maximum corner of the bounding box (x, y, z).
        max: [f64; 3],
    },
    /// Periodic (wrap-around) boundaries with given cell extents.
    Periodic {
        /// Cell size along each axis.
        cell: [f64; 3],
    },
}

impl BoundaryCondition {
    /// Apply boundary conditions to a position and velocity, modifying them
    /// in-place.
    pub fn apply(&self, position: &mut [f64; 3], velocity: &mut [f64; 3]) {
        match self {
            BoundaryCondition::Walls { min, max } => {
                for k in 0..3 {
                    if position[k] < min[k] {
                        position[k] = min[k];
                        if velocity[k] < 0.0 {
                            velocity[k] = -velocity[k] * 0.5;
                        }
                    }
                    if position[k] > max[k] {
                        position[k] = max[k];
                        if velocity[k] > 0.0 {
                            velocity[k] = -velocity[k] * 0.5;
                        }
                    }
                }
            }
            BoundaryCondition::Periodic { cell } => {
                for k in 0..3 {
                    let c = cell[k];
                    if c > 0.0 {
                        position[k] = position[k].rem_euclid(c);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LiquidParticle
// ---------------------------------------------------------------------------

/// A single position-based fluid particle.
#[derive(Clone, Debug)]
pub struct LiquidParticle {
    /// World-space position (m).
    pub position: [f64; 3],
    /// Velocity (m/s).
    pub velocity: [f64; 3],
    /// Predicted position used during constraint solving (m).
    pub predicted: [f64; 3],
    /// Local density estimate ρᵢ (kg/m³).
    pub density: f64,
    /// Local pressure p (Pa).
    pub pressure: f64,
    /// XPBD constraint multiplier λᵢ.
    pub lambda: f64,
    /// Whether this particle is on the free surface.
    pub is_surface: bool,
    /// Outward-pointing surface normal (unit vector).
    pub surface_normal: [f64; 3],
}

impl LiquidParticle {
    /// Create a particle at `position` with zero velocity.
    pub fn new(position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            predicted: position,
            density: 0.0,
            pressure: 0.0,
            lambda: 0.0,
            is_surface: false,
            surface_normal: [0.0, 1.0, 0.0],
        }
    }

    /// Create a particle at `position` with initial `velocity`.
    pub fn with_velocity(position: [f64; 3], velocity: [f64; 3]) -> Self {
        let mut p = Self::new(position);
        p.velocity = velocity;
        p
    }
}

// ---------------------------------------------------------------------------
// DensityConstraint
// ---------------------------------------------------------------------------

/// Computes density from neighbour particles and solves position corrections.
pub struct DensityConstraint {
    /// Rest (target) density ρ₀ (kg/m³).
    pub rest_density: f64,
    /// Smoothing kernel radius h (m).
    pub smoothing_h: f64,
}

impl DensityConstraint {
    /// Create a new density constraint with given rest density and smoothing radius.
    pub fn new(rest_density: f64, smoothing_h: f64) -> Self {
        Self {
            rest_density,
            smoothing_h,
        }
    }

    /// Estimate density at particle `i` from `neighbours` (indices into `particles`).
    pub fn compute_density(
        &self,
        i: usize,
        particles: &[LiquidParticle],
        neighbours: &[usize],
    ) -> f64 {
        let xi = particles[i].predicted;
        let mut rho = 0.0;
        // Include self-contribution (r = 0).
        rho += poly6_kernel(0.0, self.smoothing_h);
        for &j in neighbours {
            if j == i {
                continue;
            }
            let xj = particles[j].predicted;
            let r = len3(sub3(xi, xj));
            rho += poly6_kernel(r, self.smoothing_h);
        }
        rho
    }

    /// Compute the position correction Δxᵢ for particle `i`.
    ///
    /// Returns the position delta `[f64; 3]` to add to `particles[i].predicted`.
    pub fn position_correction(
        &self,
        i: usize,
        particles: &[LiquidParticle],
        neighbours: &[usize],
    ) -> [f64; 3] {
        let xi = particles[i].predicted;
        let lambda_i = particles[i].lambda;
        let mut delta = [0.0_f64; 3];
        for &j in neighbours {
            if j == i {
                continue;
            }
            let xj = particles[j].predicted;
            let r_vec = sub3(xi, xj);
            let grad = spiky_gradient(r_vec, self.smoothing_h);
            let lambda_j = particles[j].lambda;
            let coeff = (lambda_i + lambda_j) / self.rest_density.max(1e-300);
            delta = add3(delta, scale3(grad, coeff));
        }
        delta
    }
}

// ---------------------------------------------------------------------------
// ViscosityForce (XSPH)
// ---------------------------------------------------------------------------

/// XSPH viscosity: smooths velocity field for coherent fluid motion.
///
/// v̄ᵢ = vᵢ + c · Σⱼ (vⱼ − vᵢ) W(|xᵢ − xⱼ|, h)
pub struct ViscosityForce {
    /// XSPH coefficient c (typically 0.01 – 0.1).
    pub coefficient: f64,
    /// Smoothing kernel radius h (m).
    pub smoothing_h: f64,
}

impl ViscosityForce {
    /// Create an XSPH viscosity with given coefficient and smoothing radius.
    pub fn new(coefficient: f64, smoothing_h: f64) -> Self {
        Self {
            coefficient,
            smoothing_h,
        }
    }

    /// Apply XSPH velocity correction to particle `i`, returning the corrected velocity.
    pub fn apply(&self, i: usize, particles: &[LiquidParticle], neighbours: &[usize]) -> [f64; 3] {
        let xi = particles[i].position;
        let vi = particles[i].velocity;
        let mut correction = [0.0_f64; 3];
        for &j in neighbours {
            if j == i {
                continue;
            }
            let xj = particles[j].position;
            let vj = particles[j].velocity;
            let r = len3(sub3(xi, xj));
            let w = poly6_kernel(r, self.smoothing_h);
            let dv = sub3(vj, vi);
            correction = add3(correction, scale3(dv, w));
        }
        add3(vi, scale3(correction, self.coefficient))
    }
}

// ---------------------------------------------------------------------------
// SurfaceDetection
// ---------------------------------------------------------------------------

/// Detects surface particles by comparing their density to a threshold fraction
/// of the rest density.
pub struct SurfaceDetection {
    /// Fraction of rest density below which a particle is considered a surface
    /// particle (e.g. 0.7 means ρ < 0.7 ρ₀ → surface).
    pub threshold_fraction: f64,
}

impl SurfaceDetection {
    /// Create a surface detector with the given density threshold fraction.
    pub fn new(threshold_fraction: f64) -> Self {
        Self { threshold_fraction }
    }

    /// Return `true` if the given particle density indicates a surface particle.
    pub fn is_surface(&self, density: f64, rest_density: f64) -> bool {
        density < self.threshold_fraction * rest_density
    }

    /// Compute the outward surface normal for particle `i` using colour field gradient.
    ///
    /// n̂ = −Σⱼ ∇W(rᵢⱼ, h) / |Σⱼ ∇W(rᵢⱼ, h)|
    pub fn compute_normal(
        &self,
        i: usize,
        particles: &[LiquidParticle],
        neighbours: &[usize],
        h: f64,
    ) -> [f64; 3] {
        let xi = particles[i].position;
        let mut grad_sum = [0.0_f64; 3];
        for &j in neighbours {
            if j == i {
                continue;
            }
            let xj = particles[j].position;
            let r_vec = sub3(xi, xj);
            let g = spiky_gradient(r_vec, h);
            grad_sum = add3(grad_sum, g);
        }
        // Outward normal is negative of the inward colour gradient.
        let neg = scale3(grad_sum, -1.0);
        norm3(neg)
    }
}

// ---------------------------------------------------------------------------
// NeighbourLists
// ---------------------------------------------------------------------------

/// Largest distance from the origin, in cells of the neighbour-search grid, at
/// which a particle is still binned. It is far below the `i32` range of the
/// grid's cell indices, so `index + 1` can never overflow, even for a
/// simulation that has diverged.
const MAX_GRID_CELL_INDEX: f64 = 1.0e9;

/// The grid cells are this factor wider than the search radius, so a pair
/// closer than the radius never ends up two cells apart, however the divisions
/// that locate the cells round.
const GRID_CELL_MARGIN: f64 = 1.0 + 1.0e-9;

/// Neighbour lists of a particle cloud in compressed sparse row (CSR) form.
///
/// The neighbours of particle `i` are `indices[offsets[i]..offsets[i + 1]]`:
/// every *other* particle whose distance to `i` is at most the search radius,
/// in ascending order. Because the order is fixed, sums over a list do not
/// depend on how the lists were found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NeighbourLists {
    /// `offsets[i]..offsets[i + 1]` is the range of particle `i` in `indices`;
    /// one entry more than there are particles.
    offsets: Vec<usize>,
    /// The concatenated neighbour lists.
    indices: Vec<usize>,
}

impl NeighbourLists {
    /// Find, for every position, the other positions within `radius`.
    ///
    /// The search uses a uniform grid with cell size `radius`
    /// ([`SpatialHashPos3D`]), so it costs O(n) for a fluid of roughly uniform
    /// density instead of the O(n²) of testing every pair. A pair is a
    /// neighbour pair when its squared distance is at most `radius * radius`.
    ///
    /// A position that is not finite, or farther than `1e9` cells from the
    /// origin, has no neighbours and is nobody's neighbour. When `radius` is
    /// not a positive finite number nobody has neighbours.
    pub fn build(positions: &[[f64; 3]], radius: f64) -> Self {
        let count = positions.len();
        let usable_radius = radius > 0.0 && radius.is_finite();
        if !usable_radius {
            return Self {
                offsets: vec![0; count + 1],
                indices: Vec::new(),
            };
        }
        // NaN fails the comparison, so a non-finite coordinate is not binned.
        let binnable = |p: &[f64; 3]| p.iter().all(|c| (c / radius).abs() < MAX_GRID_CELL_INDEX);
        let mut grid: SpatialHashPos3D<()> = SpatialHashPos3D::new(radius * GRID_CELL_MARGIN);
        for (i, p) in positions.iter().enumerate() {
            if binnable(p) {
                grid.insert(i, *p, ());
            }
        }
        let mut offsets = Vec::with_capacity(count + 1);
        let mut indices: Vec<usize> = Vec::new();
        offsets.push(0);
        for (i, p) in positions.iter().enumerate() {
            if binnable(p) {
                let start = indices.len();
                indices.extend(
                    grid.query_radius(*p, radius)
                        .into_iter()
                        .map(|(j, _)| j)
                        .filter(|&j| j != i),
                );
                indices[start..].sort_unstable();
            }
            offsets.push(indices.len());
        }
        Self { offsets, indices }
    }

    /// Number of particles the lists were built for.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// `true` if the lists were built for no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The neighbours of particle `i`, in ascending order and without `i`
    /// itself. Empty for an index out of range.
    pub fn neighbours(&self, i: usize) -> &[usize] {
        match (self.offsets.get(i), self.offsets.get(i + 1)) {
            (Some(&start), Some(&end)) => self.indices.get(start..end).unwrap_or(&[]),
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// LiquidSimulation
// ---------------------------------------------------------------------------

/// Position-based fluid simulation (Macklin & Müller 2013).
pub struct LiquidSimulation {
    /// Simulated fluid particles.
    pub particles: Vec<LiquidParticle>,
    /// Rest density ρ₀ (kg/m³).
    pub rest_density: f64,
    /// Smoothing kernel radius h (m).
    pub smoothing_h: f64,
    /// XSPH viscosity coefficient c.
    pub viscosity_coeff: f64,
    /// Gravity acceleration vector (m/s²).
    pub gravity: [f64; 3],
    /// Boundary conditions.
    pub boundary: BoundaryCondition,
    /// Number of constraint solver iterations per step.
    pub solver_iterations: usize,
}

impl LiquidSimulation {
    /// Construct a new liquid simulation.
    pub fn new(
        rest_density: f64,
        smoothing_h: f64,
        viscosity_coeff: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        Self {
            particles: Vec::new(),
            rest_density,
            smoothing_h,
            viscosity_coeff,
            gravity: [0.0, -9.81, 0.0],
            boundary,
            solver_iterations: 3,
        }
    }

    /// Add a particle to the simulation.
    pub fn add_particle(&mut self, particle: LiquidParticle) {
        self.particles.push(particle);
    }

    /// Neighbour lists of the particles at the positions picked by `position`.
    fn neighbour_lists(&self, position: impl Fn(&LiquidParticle) -> [f64; 3]) -> NeighbourLists {
        let positions: Vec<[f64; 3]> = self.particles.iter().map(position).collect();
        NeighbourLists::build(&positions, self.smoothing_h)
    }

    /// Advance the simulation by time step `dt` (s).
    ///
    /// The neighbour lists are found with a spatial hash, once per stage as in
    /// Macklin and Müller (2013): from the predicted positions for the density
    /// solver (shared by all of its iterations), from the updated positions for
    /// the XSPH viscosity, and from the positions after the boundary conditions
    /// for the surface normals.
    pub fn step(&mut self, dt: f64) {
        let n = self.particles.len();
        // 1. Apply gravity and compute predicted positions.
        for p in &mut self.particles {
            p.velocity = add3(p.velocity, scale3(self.gravity, dt));
            p.predicted = add3(p.position, scale3(p.velocity, dt));
        }

        // 2. Solver iterations.
        let density_constraint = DensityConstraint::new(self.rest_density, self.smoothing_h);
        let neighbours = self.neighbour_lists(|p| p.predicted);
        let mut densities = vec![0.0_f64; n];
        let mut lambdas = vec![0.0_f64; n];
        let mut deltas = vec![[0.0_f64; 3]; n];
        for _ in 0..self.solver_iterations {
            // Compute density and lambda for all particles.
            for i in 0..n {
                let rho = density_constraint.compute_density(
                    i,
                    &self.particles,
                    neighbours.neighbours(i),
                );
                densities[i] = rho;
                // Compute gradient sum for lambda denominator.
                let xi = self.particles[i].predicted;
                let mut grad_sq_sum = 0.0;
                for &j in neighbours.neighbours(i) {
                    let xj = self.particles[j].predicted;
                    let r_vec = sub3(xi, xj);
                    let g = spiky_gradient(r_vec, self.smoothing_h);
                    let g_scaled = scale3(g, 1.0 / self.rest_density);
                    grad_sq_sum += dot3(g_scaled, g_scaled);
                }
                lambdas[i] = compute_lambda(rho, self.rest_density, grad_sq_sum);
            }

            // Write back densities and lambdas.
            for (p, (&rho, &lambda)) in self
                .particles
                .iter_mut()
                .zip(densities.iter().zip(lambdas.iter()))
            {
                p.density = rho;
                p.lambda = lambda;
            }

            // Compute and apply position corrections.
            for (i, delta) in deltas.iter_mut().enumerate() {
                *delta = density_constraint.position_correction(
                    i,
                    &self.particles,
                    neighbours.neighbours(i),
                );
            }
            for (p, delta) in self.particles.iter_mut().zip(deltas.iter()) {
                p.predicted = add3(p.predicted, *delta);
            }
        }

        // 3. Update velocity and position.
        for p in &mut self.particles {
            p.velocity = scale3(sub3(p.predicted, p.position), 1.0 / dt);
            p.position = p.predicted;
        }

        // 4. XSPH viscosity.
        let viscosity = ViscosityForce::new(self.viscosity_coeff, self.smoothing_h);
        let neighbours = self.neighbour_lists(|p| p.position);
        let new_velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| viscosity.apply(i, &self.particles, neighbours.neighbours(i)))
            .collect();
        for (p, nv) in self.particles.iter_mut().zip(new_velocities.iter()) {
            p.velocity = *nv;
        }

        // 5. Apply boundary conditions.
        for p in &mut self.particles {
            self.boundary.apply(&mut p.position, &mut p.velocity);
        }

        // 6. Surface detection.
        let surface_detect = SurfaceDetection::new(0.7);
        let neighbours = self.neighbour_lists(|p| p.position);
        let is_surface: Vec<bool> = (0..n)
            .map(|i| surface_detect.is_surface(self.particles[i].density, self.rest_density))
            .collect();
        let normals: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                surface_detect.compute_normal(
                    i,
                    &self.particles,
                    neighbours.neighbours(i),
                    self.smoothing_h,
                )
            })
            .collect();
        for i in 0..n {
            self.particles[i].is_surface = is_surface[i];
            self.particles[i].surface_normal = normals[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // 1. poly6_kernel is zero beyond h.
    #[test]
    fn test_poly6_zero_beyond_h() {
        assert!(poly6_kernel(0.5, 0.4).abs() < EPS);
        assert!(poly6_kernel(1.0, 0.4).abs() < EPS);
    }

    // 2. poly6_kernel is positive inside h.
    #[test]
    fn test_poly6_positive_inside_h() {
        let h = 0.1;
        assert!(poly6_kernel(0.0, h) > 0.0);
        assert!(poly6_kernel(0.05, h) > 0.0);
    }

    // 3. poly6_kernel at r = 0 is the maximum.
    #[test]
    fn test_poly6_max_at_zero() {
        let h = 0.1;
        let w0 = poly6_kernel(0.0, h);
        let w1 = poly6_kernel(0.05, h);
        assert!(w0 > w1, "poly6 should be maximum at r=0");
    }

    // 4. poly6_kernel decreases monotonically with r.
    #[test]
    fn test_poly6_monotone() {
        let h = 0.1;
        let vals: Vec<f64> = (0..10).map(|i| poly6_kernel(i as f64 * 0.01, h)).collect();
        for w in vals.windows(2) {
            assert!(
                w[1] <= w[0] + EPS,
                "poly6 not monotone: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    // 5. poly6_kernel is exactly zero at r = h.
    #[test]
    fn test_poly6_zero_at_h() {
        let h = 0.1;
        assert!(poly6_kernel(h, h).abs() < EPS);
    }

    // 6. spiky_gradient is zero beyond h.
    #[test]
    fn test_spiky_gradient_zero_beyond_h() {
        let r = [0.5, 0.0, 0.0];
        let h = 0.1;
        let g = spiky_gradient(r, h);
        assert!(len3(g).abs() < EPS);
    }

    // 7. spiky_gradient direction is along r (pointing inward, negative).
    #[test]
    fn test_spiky_gradient_direction() {
        let r = [0.05, 0.0, 0.0];
        let h = 0.1;
        let g = spiky_gradient(r, h);
        // Gradient should be negative (pointing toward j, against r).
        assert!(g[0] < 0.0, "Spiky gradient should point inward: {:?}", g);
        assert!(g[1].abs() < EPS);
        assert!(g[2].abs() < EPS);
    }

    // 8. spiky_gradient magnitude increases as r → 0.
    #[test]
    fn test_spiky_gradient_increases_toward_center() {
        let h = 0.1;
        let g1 = len3(spiky_gradient([0.08, 0.0, 0.0], h));
        let g2 = len3(spiky_gradient([0.02, 0.0, 0.0], h));
        assert!(g2 > g1, "Spiky gradient should be larger closer to center");
    }

    // 9. compute_lambda returns zero when density = rest_density and grad_sum > 0.
    #[test]
    fn test_compute_lambda_zero_constraint() {
        let rho0 = 1000.0;
        let lam = compute_lambda(rho0, rho0, 1.0);
        assert!(lam.abs() < EPS, "lambda should be 0 at rest density: {lam}");
    }

    // 10. compute_lambda is negative when density > rest_density (compression).
    #[test]
    fn test_compute_lambda_compressed() {
        let lam = compute_lambda(1100.0, 1000.0, 1.0);
        assert!(
            lam < 0.0,
            "lambda should be negative for compressed fluid: {lam}"
        );
    }

    // 11. compute_lambda is positive when density < rest_density (tension).
    #[test]
    fn test_compute_lambda_tension() {
        let lam = compute_lambda(900.0, 1000.0, 1.0);
        assert!(
            lam > 0.0,
            "lambda should be positive for rarefied fluid: {lam}"
        );
    }

    // 12. compute_lambda scales inversely with grad_sum.
    #[test]
    fn test_compute_lambda_scales_with_grad_sum() {
        let rho = 1100.0;
        let rho0 = 1000.0;
        let l1 = compute_lambda(rho, rho0, 1.0);
        let l2 = compute_lambda(rho, rho0, 2.0);
        // Both should be negative; larger grad_sum → smaller magnitude.
        assert!(
            l1.abs() > l2.abs(),
            "Higher grad_sum should reduce |lambda|"
        );
    }

    // 13. LiquidParticle::new initialises velocity to zero.
    #[test]
    fn test_liquid_particle_zero_velocity() {
        let p = LiquidParticle::new([1.0, 2.0, 3.0]);
        assert!(len3(p.velocity) < EPS);
    }

    // 14. LiquidParticle::with_velocity stores velocity.
    #[test]
    fn test_liquid_particle_with_velocity() {
        let vel = [1.0, -2.0, 0.5];
        let p = LiquidParticle::with_velocity([0.0; 3], vel);
        for (&vk, &vk_expected) in p.velocity.iter().zip(vel.iter()) {
            assert!((vk - vk_expected).abs() < EPS);
        }
    }

    // 15. DensityConstraint::compute_density includes self contribution.
    #[test]
    fn test_density_constraint_self_contribution() {
        let dc = DensityConstraint::new(1000.0, 0.1);
        let particles = vec![LiquidParticle::new([0.0; 3])];
        let rho = dc.compute_density(0, &particles, &[]);
        // W(0, h) > 0.
        assert!(rho > 0.0, "Density should include self-contribution");
    }

    // 16. DensityConstraint: density increases with more neighbours.
    #[test]
    fn test_density_constraint_increases_with_neighbours() {
        let h = 0.1;
        let dc = DensityConstraint::new(1000.0, h);
        let mut particles = vec![
            LiquidParticle::new([0.0, 0.0, 0.0]),
            LiquidParticle::new([0.01, 0.0, 0.0]),
            LiquidParticle::new([0.02, 0.0, 0.0]),
        ];
        // Set predicted positions.
        for p in &mut particles {
            p.predicted = p.position;
        }
        let rho_alone = dc.compute_density(0, &particles, &[]);
        let rho_with_neighbors = dc.compute_density(0, &particles, &[1, 2]);
        assert!(
            rho_with_neighbors > rho_alone,
            "More neighbours → higher density"
        );
    }

    // 17. BoundaryCondition::Walls clamps positions.
    #[test]
    fn test_walls_boundary_clamps_position() {
        let bc = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut pos = [-0.5, 1.5, 0.5];
        let mut vel = [-1.0, 1.0, 0.0];
        bc.apply(&mut pos, &mut vel);
        assert!(pos[0] >= 0.0);
        assert!(pos[1] <= 1.0);
    }

    // 18. BoundaryCondition::Walls reverses velocity on collision.
    #[test]
    fn test_walls_boundary_reverses_velocity() {
        let bc = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut pos = [-0.1, 0.5, 0.5];
        let mut vel = [-2.0, 0.0, 0.0];
        bc.apply(&mut pos, &mut vel);
        assert!(vel[0] > 0.0, "Velocity should reverse on wall hit");
    }

    // 19. BoundaryCondition::Periodic wraps positions.
    #[test]
    fn test_periodic_boundary_wraps() {
        let bc = BoundaryCondition::Periodic { cell: [1.0; 3] };
        let mut pos = [1.5, -0.3, 2.1];
        let mut vel = [0.0; 3];
        bc.apply(&mut pos, &mut vel);
        assert!(pos[0] >= 0.0 && pos[0] < 1.0, "Periodic x wrap: {}", pos[0]);
        assert!(pos[1] >= 0.0 && pos[1] < 1.0, "Periodic y wrap: {}", pos[1]);
        assert!(pos[2] >= 0.0 && pos[2] < 1.0, "Periodic z wrap: {}", pos[2]);
    }

    // 20. ViscosityForce returns original velocity when no neighbours.
    #[test]
    fn test_viscosity_no_neighbours() {
        let visc = ViscosityForce::new(0.1, 0.1);
        let particles = vec![LiquidParticle::with_velocity([0.0; 3], [1.0, 2.0, 3.0])];
        let v = visc.apply(0, &particles, &[]);
        for (&vk, &pk) in v.iter().zip(particles[0].velocity.iter()) {
            assert!((vk - pk).abs() < EPS);
        }
    }

    // 21. ViscosityForce blends toward neighbour velocity.
    #[test]
    fn test_viscosity_blends_velocity() {
        let h = 0.05;
        let visc = ViscosityForce::new(0.5, h);
        let mut particles = vec![
            LiquidParticle::with_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            LiquidParticle::with_velocity([0.01, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ];
        for p in &mut particles {
            p.predicted = p.position;
        }
        let v = visc.apply(0, &particles, &[1]);
        assert!(
            v[0] > 0.0,
            "Viscosity should blend toward neighbour velocity"
        );
    }

    // 22. SurfaceDetection::is_surface true when density < threshold * rho0.
    #[test]
    fn test_surface_detection_is_surface() {
        let sd = SurfaceDetection::new(0.7);
        assert!(sd.is_surface(600.0, 1000.0)); // 600 < 700
        assert!(!sd.is_surface(800.0, 1000.0)); // 800 > 700
    }

    // 23. SurfaceDetection::compute_normal returns unit vector.
    #[test]
    fn test_surface_normal_unit_length() {
        let sd = SurfaceDetection::new(0.7);
        let particles = vec![
            LiquidParticle::new([0.0, 0.0, 0.0]),
            LiquidParticle::new([0.05, 0.0, 0.0]),
        ];
        let n = sd.compute_normal(0, &particles, &[1], 0.1);
        let mag = len3(n);
        // Either unit vector or zero (degenerate).
        assert!(
            mag < EPS || (mag - 1.0).abs() < 1e-8,
            "Normal magnitude: {mag}"
        );
    }

    // 24. LiquidSimulation step does not panic on single particle.
    #[test]
    fn test_single_particle_step_no_panic() {
        let bc = BoundaryCondition::Walls {
            min: [-5.0; 3],
            max: [5.0; 3],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.1, 0.01, bc);
        sim.add_particle(LiquidParticle::new([0.0, 1.0, 0.0]));
        sim.step(0.01);
        // Particle should have fallen due to gravity.
        assert!(sim.particles[0].position[1] < 1.0);
    }

    // 25. Particle falls under gravity.
    #[test]
    fn test_gravity_moves_particle_downward() {
        let bc = BoundaryCondition::Walls {
            min: [-100.0; 3],
            max: [100.0; 3],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.1, 0.0, bc);
        sim.add_particle(LiquidParticle::new([0.0, 10.0, 0.0]));
        for _ in 0..60 {
            sim.step(1.0 / 60.0);
        }
        assert!(
            sim.particles[0].position[1] < 10.0,
            "Particle should fall: y={}",
            sim.particles[0].position[1]
        );
    }

    // 26. Wall boundary prevents falling below floor.
    #[test]
    fn test_wall_boundary_floor() {
        let bc = BoundaryCondition::Walls {
            min: [0.0, 0.0, 0.0],
            max: [10.0, 10.0, 10.0],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.1, 0.0, bc);
        sim.add_particle(LiquidParticle::new([5.0, 0.5, 5.0]));
        for _ in 0..200 {
            sim.step(1.0 / 60.0);
        }
        assert!(
            sim.particles[0].position[1] >= 0.0,
            "Particle should not fall below floor: y={}",
            sim.particles[0].position[1]
        );
    }

    // 27. poly6_kernel is symmetric: W(r, h) = W(-r, h) (r is scalar).
    #[test]
    fn test_poly6_symmetric() {
        let h = 0.1;
        let w1 = poly6_kernel(0.03, h);
        let w2 = poly6_kernel(0.03, h);
        assert!((w1 - w2).abs() < EPS);
    }

    // 28. compute_lambda is invariant to grad_sum = 0 (uses epsilon guard).
    #[test]
    fn test_compute_lambda_epsilon_guard() {
        let lam = compute_lambda(1100.0, 1000.0, 0.0);
        // Should not be infinite/NaN.
        assert!(lam.is_finite(), "lambda should be finite: {lam}");
    }

    // 29. LiquidSimulation add_particle increases count.
    #[test]
    fn test_add_particle_increases_count() {
        let bc = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.1, 0.01, bc);
        assert_eq!(sim.particles.len(), 0);
        sim.add_particle(LiquidParticle::new([0.5; 3]));
        assert_eq!(sim.particles.len(), 1);
        sim.add_particle(LiquidParticle::new([0.6; 3]));
        assert_eq!(sim.particles.len(), 2);
    }

    // 30. Multiple particles step together without panic.
    #[test]
    fn test_multi_particle_step() {
        let bc = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [5.0; 3],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.3, 0.01, bc);
        for i in 0..5 {
            sim.add_particle(LiquidParticle::new([i as f64 * 0.2, 2.0, 0.5]));
        }
        for _ in 0..10 {
            sim.step(0.01);
        }
        // All particles should be within bounds.
        for p in &sim.particles {
            assert!(p.position[1] >= 0.0);
        }
    }

    // 31. spiky_gradient is zero for r = 0.
    #[test]
    fn test_spiky_gradient_zero_at_origin() {
        let g = spiky_gradient([0.0, 0.0, 0.0], 0.1);
        assert!(
            len3(g) < EPS,
            "Spiky gradient at origin should be zero: {:?}",
            g
        );
    }

    // 32. Periodic boundary wraps near-boundary particle correctly.
    #[test]
    fn test_periodic_boundary_near_cell_edge() {
        let bc = BoundaryCondition::Periodic {
            cell: [2.0, 2.0, 2.0],
        };
        let mut pos = [1.99, 0.5, 0.5];
        let mut vel = [5.0, 0.0, 0.0];
        bc.apply(&mut pos, &mut vel);
        assert!(pos[0] < 2.0 && pos[0] >= 0.0);
    }

    // 33. ViscosityForce coefficient = 0 → no change.
    #[test]
    fn test_viscosity_zero_coefficient() {
        let h = 0.1;
        let visc = ViscosityForce::new(0.0, h);
        let particles = vec![
            LiquidParticle::with_velocity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            LiquidParticle::with_velocity([0.05, 0.0, 0.0], [4.0, 5.0, 6.0]),
        ];
        let v = visc.apply(0, &particles, &[1]);
        for (&vk, &pk) in v.iter().zip(particles[0].velocity.iter()) {
            assert!(
                (vk - pk).abs() < EPS,
                "Zero coefficient viscosity should not change velocity"
            );
        }
    }

    // 34. Surface detection threshold = 1.0 → all particles are surface.
    #[test]
    fn test_surface_detection_all_surface() {
        let sd = SurfaceDetection::new(1.0);
        // Any density < 1.0 * rho0 is surface.
        assert!(sd.is_surface(999.0, 1000.0));
        assert!(!sd.is_surface(1001.0, 1000.0));
    }

    // -----------------------------------------------------------------------
    // Neighbour search: spatial hash + CSR neighbour lists
    //
    // `step` used to find the neighbours with an all-pairs scan for every
    // particle in every solver iteration, and with two more all-pairs scans for
    // the XSPH viscosity and the surface normals.
    // -----------------------------------------------------------------------

    /// SplitMix64: a small deterministic generator of floats in `[0, 1)`.
    struct SplitMix(u64);

    impl SplitMix {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1_u64 << 53) as f64
        }

        /// A point of the cube `[low, high)^3`.
        fn point(&mut self, low: f64, high: f64) -> [f64; 3] {
            let mut next = || low + (high - low) * self.next_f64();
            [next(), next(), next()]
        }
    }

    /// Reference neighbour search: for every position, the ascending indices
    /// of the other positions whose squared distance is at most `radius²`.
    fn brute_force_neighbours(positions: &[[f64; 3]], radius: f64) -> Vec<Vec<usize>> {
        positions
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                positions
                    .iter()
                    .enumerate()
                    .filter(|&(j, xj)| {
                        let d = sub3(*xi, *xj);
                        j != i && dot3(d, d) <= radius * radius
                    })
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect()
    }

    fn random_positions(seed: u64, count: usize, low: f64, high: f64) -> Vec<[f64; 3]> {
        let mut rng = SplitMix(seed);
        (0..count).map(|_| rng.point(low, high)).collect()
    }

    // 35. The hashed neighbour sets are the brute-force ones, and symmetric.
    #[test]
    fn test_neighbour_lists_equal_brute_force_on_random_particles() {
        let positions = random_positions(1, 500, 0.0, 1.0);
        let h = 0.15;
        let lists = NeighbourLists::build(&positions, h);
        let brute = brute_force_neighbours(&positions, h);
        assert_eq!(lists.len(), 500);
        assert!(!lists.is_empty());
        let mut entries = 0;
        for (i, expected) in brute.iter().enumerate() {
            assert_eq!(lists.neighbours(i), expected.as_slice(), "particle {i}");
            entries += expected.len();
            for &j in expected {
                assert!(
                    lists.neighbours(j).contains(&i),
                    "{j} is a neighbour of {i} but not the other way round"
                );
            }
        }
        assert!(
            entries > 1000,
            "the test needs real neighbourhoods: {entries}"
        );
    }

    // 36. Densities from the hashed lists equal those from the all-pairs scan.
    #[test]
    fn test_density_from_hashed_neighbours_equals_brute_force_density() {
        let positions = random_positions(2, 500, 0.0, 1.0);
        let h = 0.15;
        let particles: Vec<LiquidParticle> =
            positions.iter().map(|&x| LiquidParticle::new(x)).collect();
        let constraint = DensityConstraint::new(1000.0, h);
        let lists = NeighbourLists::build(&positions, h);
        for (i, xi) in positions.iter().enumerate() {
            // The all-pairs scan listed every particle within h, i included.
            let all_pairs: Vec<usize> = (0..positions.len())
                .filter(|&j| len3(sub3(*xi, positions[j])) <= h)
                .collect();
            let expected = constraint.compute_density(i, &particles, &all_pairs);
            let actual = constraint.compute_density(i, &particles, lists.neighbours(i));
            assert!(
                (actual - expected).abs() <= 1e-12 * expected,
                "particle {i}: density {actual} vs {expected}"
            );
            // The self contribution is a lower bound (to rounding: the two
            // evaluations of the kernel may differ in the last bit).
            assert!(
                actual >= poly6_kernel(0.0, h) * (1.0 - 1e-12),
                "particle {i}"
            );
        }
    }

    // 37. On a lattice every particle has an analytically known neighbourhood.
    #[test]
    fn test_neighbour_lists_of_a_lattice_have_the_analytic_count() {
        const SIDE: i64 = 9;
        let spacing = 0.05;
        // Lattice vectors of squared length 1..=6 (6.25 = 2.5² is the squared
        // support radius in lattice units, so no pair sits on the boundary).
        let index = |x: i64, y: i64, z: i64| ((x * SIDE + y) * SIDE + z) as usize;
        let mut positions = vec![[0.0; 3]; (SIDE * SIDE * SIDE) as usize];
        for x in 0..SIDE {
            for y in 0..SIDE {
                for z in 0..SIDE {
                    positions[index(x, y, z)] =
                        [x as f64 * spacing, y as f64 * spacing, z as f64 * spacing];
                }
            }
        }
        let lists = NeighbourLists::build(&positions, 2.5 * spacing);
        let mut interior = 0;
        for x in 0..SIDE {
            for y in 0..SIDE {
                for z in 0..SIDE {
                    let mut expected = Vec::new();
                    for dx in -2..=2_i64 {
                        for dy in -2..=2_i64 {
                            for dz in -2..=2_i64 {
                                let inside = (0..SIDE).contains(&(x + dx))
                                    && (0..SIDE).contains(&(y + dy))
                                    && (0..SIDE).contains(&(z + dz));
                                let squared = dx * dx + dy * dy + dz * dz;
                                if inside && squared > 0 && squared <= 6 {
                                    expected.push(index(x + dx, y + dy, z + dz));
                                }
                            }
                        }
                    }
                    expected.sort_unstable();
                    assert_eq!(
                        lists.neighbours(index(x, y, z)),
                        expected.as_slice(),
                        "lattice site ({x}, {y}, {z})"
                    );
                    if (2..SIDE - 2).contains(&x)
                        && (2..SIDE - 2).contains(&y)
                        && (2..SIDE - 2).contains(&z)
                    {
                        // 6 + 12 + 8 + 6 + 24 + 24 integer vectors of squared
                        // length 1, 2, 3, 4, 5, 6.
                        assert_eq!(expected.len(), 80);
                        interior += 1;
                    }
                }
            }
        }
        assert_eq!(interior, 5 * 5 * 5);
    }

    // 38. Degenerate input never panics and never invents neighbours.
    #[test]
    fn test_neighbour_lists_degenerate_inputs() {
        let empty = NeighbourLists::build(&[], 0.1);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.neighbours(0).is_empty());
        assert!(NeighbourLists::default().is_empty());

        // Without a positive finite radius nobody has neighbours.
        let close = [[0.0; 3], [0.01, 0.0, 0.0]];
        for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let lists = NeighbourLists::build(&close, radius);
            assert_eq!(lists.len(), 2, "radius {radius}");
            assert!(lists.neighbours(0).is_empty(), "radius {radius}");
            assert!(lists.neighbours(1).is_empty(), "radius {radius}");
        }

        // Non-finite and astronomically distant particles are isolated; the
        // others are not affected by them.
        let positions = [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [1e300, 1e300, 1e300],
            [1e300, 1e300, 1e300],
            [0.0, 0.07, 0.0],
        ];
        let lists = NeighbourLists::build(&positions, 0.1);
        assert_eq!(lists.neighbours(0), &[1, 6]);
        assert_eq!(lists.neighbours(1), &[0, 6]);
        assert_eq!(lists.neighbours(6), &[0, 1]);
        for isolated in 2..6 {
            assert!(lists.neighbours(isolated).is_empty(), "particle {isolated}");
        }
        assert!(lists.neighbours(positions.len()).is_empty());

        // Coincident particles are neighbours, and so is a pair exactly h apart.
        let twins = NeighbourLists::build(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], 0.1);
        assert_eq!(twins.neighbours(0), &[1]);
        assert_eq!(twins.neighbours(1), &[0]);
        let on_the_boundary = NeighbourLists::build(&[[0.0; 3], [0.1, 0.0, 0.0]], 0.1);
        assert_eq!(on_the_boundary.neighbours(0), &[1]);

        // Cells on both sides of the origin.
        let across = NeighbourLists::build(&[[-0.04; 3], [0.04; 3], [0.5; 3]], 0.2);
        assert_eq!(across.neighbours(0), &[1]);
        assert_eq!(across.neighbours(1), &[0]);
        assert!(across.neighbours(2).is_empty());
    }

    /// The pre-change `step`, with the all-pairs scan standing in for the
    /// spatial hash: neighbours from the predicted positions once for all
    /// solver iterations, from the new positions for XSPH, and from the
    /// positions after the boundary conditions for the surface normals.
    fn reference_step(sim: &mut LiquidSimulation, dt: f64) {
        let n = sim.particles.len();
        let h = sim.smoothing_h;
        for p in &mut sim.particles {
            p.velocity = add3(p.velocity, scale3(sim.gravity, dt));
            p.predicted = add3(p.position, scale3(p.velocity, dt));
        }
        let constraint = DensityConstraint::new(sim.rest_density, h);
        let predicted: Vec<[f64; 3]> = sim.particles.iter().map(|p| p.predicted).collect();
        let lists = brute_force_neighbours(&predicted, h);
        for _ in 0..sim.solver_iterations {
            let mut densities = vec![0.0; n];
            let mut lambdas = vec![0.0; n];
            for i in 0..n {
                let rho = constraint.compute_density(i, &sim.particles, &lists[i]);
                densities[i] = rho;
                let mut grad_sq_sum = 0.0;
                for &j in &lists[i] {
                    let r_vec = sub3(sim.particles[i].predicted, sim.particles[j].predicted);
                    let g = scale3(spiky_gradient(r_vec, h), 1.0 / sim.rest_density);
                    grad_sq_sum += dot3(g, g);
                }
                lambdas[i] = compute_lambda(rho, sim.rest_density, grad_sq_sum);
            }
            for i in 0..n {
                sim.particles[i].density = densities[i];
                sim.particles[i].lambda = lambdas[i];
            }
            let deltas: Vec<[f64; 3]> = (0..n)
                .map(|i| constraint.position_correction(i, &sim.particles, &lists[i]))
                .collect();
            for (p, delta) in sim.particles.iter_mut().zip(deltas.iter()) {
                p.predicted = add3(p.predicted, *delta);
            }
        }
        for p in &mut sim.particles {
            p.velocity = scale3(sub3(p.predicted, p.position), 1.0 / dt);
            p.position = p.predicted;
        }
        let viscosity = ViscosityForce::new(sim.viscosity_coeff, h);
        let positions: Vec<[f64; 3]> = sim.particles.iter().map(|p| p.position).collect();
        let lists = brute_force_neighbours(&positions, h);
        let velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| viscosity.apply(i, &sim.particles, &lists[i]))
            .collect();
        for (p, v) in sim.particles.iter_mut().zip(velocities.iter()) {
            p.velocity = *v;
        }
        for p in &mut sim.particles {
            sim.boundary.apply(&mut p.position, &mut p.velocity);
        }
        let detector = SurfaceDetection::new(0.7);
        let positions: Vec<[f64; 3]> = sim.particles.iter().map(|p| p.position).collect();
        let lists = brute_force_neighbours(&positions, h);
        for (i, list) in lists.iter().enumerate() {
            let is_surface = detector.is_surface(sim.particles[i].density, sim.rest_density);
            let normal = detector.compute_normal(i, &sim.particles, list, h);
            sim.particles[i].is_surface = is_surface;
            sim.particles[i].surface_normal = normal;
        }
    }

    fn random_fluid(seed: u64, count: usize) -> LiquidSimulation {
        let boundary = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [0.6; 3],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.2, 0.05, boundary);
        let mut rng = SplitMix(seed);
        for _ in 0..count {
            let position = rng.point(0.0, 0.6);
            let velocity = rng.point(-1.0, 1.0);
            sim.add_particle(LiquidParticle::with_velocity(position, velocity));
        }
        sim
    }

    /// `|a - b| <= tolerance * max(1, |b|)`.
    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * b.abs().max(1.0)
    }

    /// Assert that two simulations hold the same state, to rounding.
    fn assert_same_state(hashed: &LiquidSimulation, reference: &LiquidSimulation, context: &str) {
        for (i, (a, b)) in hashed
            .particles
            .iter()
            .zip(reference.particles.iter())
            .enumerate()
        {
            for (what, x, y) in [
                ("position", a.position, b.position),
                ("velocity", a.velocity, b.velocity),
                ("normal", a.surface_normal, b.surface_normal),
            ] {
                for (axis, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                    assert!(
                        close(*u, *v, 1e-9),
                        "{context}, particle {i}: {what}[{axis}] {u} vs {v}"
                    );
                }
            }
            assert!(
                close(a.density, b.density, 1e-9),
                "{context}, particle {i}: density {} vs {}",
                a.density,
                b.density
            );
            assert!(
                close(a.lambda, b.lambda, 1e-9),
                "{context}, particle {i}: lambda {} vs {}",
                a.lambda,
                b.lambda
            );
            assert_eq!(a.is_surface, b.is_surface, "{context}, particle {i}");
        }
    }

    // 39. `step` follows the documented stages with the hashed lists: the
    // solver iterations share the lists of the predicted positions, and the
    // XSPH viscosity and the surface normals use fresh lists (the particles
    // move a lot during the iterations and are clamped by the walls).
    #[test]
    fn test_step_equals_the_brute_force_reference() {
        for seed in 0..8 {
            let count = 60 + 15 * seed as usize;
            let mut hashed = random_fluid(seed, count);
            let mut reference = random_fluid(seed, count);
            let positions: Vec<[f64; 3]> = hashed.particles.iter().map(|p| p.position).collect();
            let entries: usize = brute_force_neighbours(&positions, hashed.smoothing_h)
                .iter()
                .map(Vec::len)
                .sum();
            assert!(
                entries > 2 * count,
                "seed {seed}: the particles must interact"
            );
            for step in 0..2 {
                hashed.step(1.0 / 120.0);
                reference_step(&mut reference, 1.0 / 120.0);
                assert_same_state(&hashed, &reference, &format!("seed {seed}, step {step}"));
            }
        }
    }

    // 39b. Degenerate simulations: no particles, no solver iterations.
    #[test]
    fn test_step_without_particles_or_iterations() {
        let boundary = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut empty = LiquidSimulation::new(1000.0, 0.1, 0.01, boundary);
        empty.step(0.01);
        assert!(empty.particles.is_empty());

        let mut idle = LiquidSimulation::new(1000.0, 0.1, 0.01, boundary);
        idle.solver_iterations = 0;
        idle.add_particle(LiquidParticle::new([0.5, 0.5, 0.5]));
        idle.add_particle(LiquidParticle::new([0.52, 0.5, 0.5]));
        idle.step(0.01);
        // Gravity alone moves the particles: no density correction ran.
        for p in &idle.particles {
            assert!(p.position[1] < 0.5, "y = {}", p.position[1]);
            assert_eq!(p.lambda, 0.0);
        }
        assert!((idle.particles[1].position[0] - 0.52).abs() < 0.01);
    }

    // 40. A few thousand particles step without trouble (this is where the
    // all-pairs scans, repeated iterations + 2 times, used to dominate).
    #[test]
    fn test_step_with_thousands_of_particles_stays_finite_and_in_bounds() {
        let boundary = BoundaryCondition::Walls {
            min: [0.0; 3],
            max: [1.0, 1.0, 0.5],
        };
        let mut sim = LiquidSimulation::new(1000.0, 0.1, 0.01, boundary);
        let spacing = 0.05;
        for x in 0..20 {
            for y in 0..20 {
                for z in 0..10 {
                    let position = [
                        (x as f64 + 0.5) * spacing,
                        (y as f64 + 0.5) * spacing,
                        (z as f64 + 0.5) * spacing,
                    ];
                    sim.add_particle(LiquidParticle::new(position));
                }
            }
        }
        assert_eq!(sim.particles.len(), 4000);
        for _ in 0..2 {
            sim.step(1.0 / 120.0);
        }
        for (i, p) in sim.particles.iter().enumerate() {
            assert!(
                p.position
                    .iter()
                    .chain(p.velocity.iter())
                    .all(|c| c.is_finite()),
                "particle {i}: {:?} {:?}",
                p.position,
                p.velocity
            );
            assert!(p.position[0] >= 0.0 && p.position[0] <= 1.0, "particle {i}");
            assert!(p.position[1] >= 0.0 && p.position[1] <= 1.0, "particle {i}");
            assert!(p.position[2] >= 0.0 && p.position[2] <= 0.5, "particle {i}");
            assert!(p.density > 0.0, "particle {i}");
        }
    }
}
