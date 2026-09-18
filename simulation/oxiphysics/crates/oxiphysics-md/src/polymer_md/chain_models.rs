// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Chain models: PolymerChain, WormlikeChain, FreelyJointedChain, FENEBond,
//! WLCSampler, KratkyPorodChain.

use super::{dist3, dot3, fene_force, mag3, norm3, sub3, wlc_extension};

/// Bead-spring polymer chain with Lennard-Jones and FENE bonds.
///
/// Models a linear polymer using connected Lennard-Jones beads
/// and FENE (Finitely Extensible Nonlinear Elastic) springs.  Adjacent beads
/// interact only through the FENE bond; the Lennard-Jones interaction acts
/// between all non-bonded pairs (`|i - j| > 1`).  The dynamics are integrated with
/// velocity Verlet ([`PolymerChain::step`]).
#[derive(Debug, Clone)]
pub struct PolymerChain {
    /// Number of beads.
    pub n_beads: usize,
    /// Bead positions \[x, y, z\].
    pub positions: Vec<[f64; 3]>,
    /// Bead velocities.
    pub velocities: Vec<[f64; 3]>,
    /// Bead masses.
    pub masses: Vec<f64>,
    /// FENE maximum extension R0.
    pub fene_r0: f64,
    /// FENE spring constant.
    pub fene_k: f64,
    /// LJ epsilon.
    pub lj_eps: f64,
    /// LJ sigma.
    pub lj_sigma: f64,
    /// Persistence length in bond units.
    pub persistence_length: f64,
    /// Temperature kT.
    pub kt: f64,
    /// Time step.
    pub dt: f64,
}

impl PolymerChain {
    /// Create a new linear polymer chain.
    pub fn new(
        n_beads: usize,
        fene_r0: f64,
        fene_k: f64,
        lj_eps: f64,
        lj_sigma: f64,
        persistence_length: f64,
        kt: f64,
        dt: f64,
    ) -> Self {
        // Initialize stretched chain along x-axis
        let positions: Vec<[f64; 3]> = (0..n_beads)
            .map(|i| [i as f64 * lj_sigma, 0.0, 0.0])
            .collect();
        let velocities = vec![[0.0f64; 3]; n_beads];
        let masses = vec![1.0f64; n_beads];
        Self {
            n_beads,
            positions,
            velocities,
            masses,
            fene_r0,
            fene_k,
            lj_eps,
            lj_sigma,
            persistence_length,
            kt,
            dt,
        }
    }

    /// FENE bond force exerted on bead `i + 1` by its bond to bead `i`.
    ///
    /// Bead `i` feels the equal and opposite force.  For every bond length
    /// `0 < r < R0` the force points from bead `i + 1` towards bead `i`
    /// (attractive).  Returns zero if `i + 1` is not a bead of the chain.
    pub fn fene_force_pair(&self, i: usize) -> [f64; 3] {
        if i + 1 >= self.n_beads {
            return [0.0; 3];
        }
        let r = sub3(self.positions[i + 1], self.positions[i]);
        let rmag = mag3(r);
        let f = fene_force(rmag, self.fene_k, self.fene_r0);
        let rhat = norm3(r);
        [f * rhat[0], f * rhat[1], f * rhat[2]]
    }

    /// Lennard-Jones force on bead `i` due to bead `j`.
    ///
    /// Repulsive (pointing from `j` towards `i`) closer than `2^(1/6) σ`,
    /// attractive beyond; `lj_force_pair(j, i)` is its exact negative.
    pub fn lj_force_pair(&self, i: usize, j: usize) -> [f64; 3] {
        if i == j {
            return [0.0; 3];
        }
        let r = sub3(self.positions[j], self.positions[i]);
        let rmag = mag3(r).max(1e-10);
        let eps = self.lj_eps;
        let sig = self.lj_sigma;
        let s6 = (sig / rmag).powi(6);
        // Radial force −dU/dr: positive when the pair repels.
        let fmag = 24.0 * eps / rmag * (2.0 * s6 * s6 - s6);
        let rhat = norm3(r);
        // `rhat` points from i to j, so a repulsive pair pushes bead i along −rhat.
        [-fmag * rhat[0], -fmag * rhat[1], -fmag * rhat[2]]
    }

    /// Total force on bead i: the FENE bonds to beads `i - 1` and `i + 1` plus the
    /// Lennard-Jones force from every non-bonded bead (`|i - j| > 1`).
    pub fn total_force(&self, i: usize) -> [f64; 3] {
        let mut f = [0.0f64; 3];
        // FENE bond to i-1: `fene_force_pair(i - 1)` acts on bead i.
        if i > 0 {
            let fb = self.fene_force_pair(i - 1);
            f[0] += fb[0];
            f[1] += fb[1];
            f[2] += fb[2];
        }
        // FENE bond to i+1: the bond force on bead i is the reaction to the force
        // `fene_force_pair(i)` on bead i + 1.
        if i + 1 < self.n_beads {
            let fb = self.fene_force_pair(i);
            f[0] -= fb[0];
            f[1] -= fb[1];
            f[2] -= fb[2];
        }
        // LJ with all non-bonded beads
        for j in 0..self.n_beads {
            if j.abs_diff(i) > 1 {
                let flj = self.lj_force_pair(i, j);
                f[0] += flj[0];
                f[1] += flj[1];
                f[2] += flj[2];
            }
        }
        f
    }

    /// Forces on all beads at the current positions.
    pub fn forces(&self) -> Vec<[f64; 3]> {
        (0..self.n_beads).map(|i| self.total_force(i)).collect()
    }

    /// Velocity Verlet step: half kick with F(t), drift, force update at t + dt,
    /// second half kick.
    pub fn step(&mut self) {
        let dt = self.dt;
        let forces = self.forces();
        for (i, &m) in self.masses.iter().enumerate().take(self.n_beads) {
            for (d, &f) in forces[i].iter().enumerate() {
                self.velocities[i][d] += 0.5 * dt * f / m;
                self.positions[i][d] += dt * self.velocities[i][d];
            }
        }
        let forces_new = self.forces();
        for (i, &m) in self.masses.iter().enumerate().take(self.n_beads) {
            for (d, &f) in forces_new[i].iter().enumerate() {
                self.velocities[i][d] += 0.5 * dt * f / m;
            }
        }
    }

    /// Kinetic energy ½ Σ m v².
    pub fn kinetic_energy(&self) -> f64 {
        self.masses
            .iter()
            .zip(self.velocities.iter())
            .take(self.n_beads)
            .map(|(&m, v)| 0.5 * m * dot3(*v, *v))
            .sum()
    }

    /// Potential energy whose negative gradient is [`total_force`](Self::total_force):
    /// the FENE bond energies `-½ K R0² ln(1 - (r/R0)²)` (infinite once a bond reaches
    /// `R0`) plus the Lennard-Jones energies `4ε[(σ/r)¹² - (σ/r)⁶]` of all non-bonded
    /// pairs.
    pub fn potential_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.n_beads.saturating_sub(1) {
            let r = dist3(self.positions[i + 1], self.positions[i]);
            let x = r / self.fene_r0;
            energy += if x >= 1.0 {
                f64::INFINITY
            } else {
                -0.5 * self.fene_k * self.fene_r0 * self.fene_r0 * (1.0 - x * x).ln()
            };
        }
        for i in 0..self.n_beads {
            for j in (i + 2)..self.n_beads {
                let r = dist3(self.positions[i], self.positions[j]).max(1e-10);
                let s6 = (self.lj_sigma / r).powi(6);
                energy += 4.0 * self.lj_eps * (s6 * s6 - s6);
            }
        }
        energy
    }

    /// Total (kinetic + potential) energy, conserved by [`step`](Self::step) up to the
    /// O(dt²) fluctuation of the velocity Verlet integrator.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Compute end-to-end distance.
    pub fn end_to_end(&self) -> f64 {
        if self.n_beads < 2 {
            return 0.0;
        }
        dist3(self.positions[0], self.positions[self.n_beads - 1])
    }

    /// Compute radius of gyration.
    pub fn radius_of_gyration(&self) -> f64 {
        let n = self.n_beads as f64;
        let com: [f64; 3] = self.positions.iter().fold([0.0; 3], |acc, p| {
            [acc[0] + p[0] / n, acc[1] + p[1] / n, acc[2] + p[2] / n]
        });
        let rg2: f64 = self
            .positions
            .iter()
            .map(|&p| {
                let d = sub3(p, com);
                dot3(d, d)
            })
            .sum::<f64>()
            / n;
        rg2.sqrt()
    }

    /// Mean square end-to-end distance (ideal chain: <R^2> = N*b^2).
    pub fn ideal_r2(&self) -> f64 {
        self.n_beads as f64 * self.lj_sigma * self.lj_sigma
    }
}

/// Worm-like chain (WLC) force extension model.
///
/// Implements F = kT/p * (1/(4(1-r/L)²) - 1/4 + r/L)
#[derive(Debug, Clone)]
pub struct WormlikeChain {
    /// Contour length L in nm.
    pub contour_length: f64,
    /// Persistence length p in nm.
    pub persistence_length: f64,
    /// Thermal energy kT in pN·nm.
    pub kt: f64,
}

impl WormlikeChain {
    /// Create a new WLC model.
    pub fn new(contour_length: f64, persistence_length: f64, kt: f64) -> Self {
        Self {
            contour_length,
            persistence_length,
            kt,
        }
    }

    /// Compute WLC force at extension r (in nm).
    pub fn force(&self, r: f64) -> f64 {
        wlc_extension(r, self.contour_length, self.persistence_length, self.kt)
    }

    /// Compute WLC elastic energy by integration.
    pub fn energy(&self, r: f64) -> f64 {
        let n_steps = 100usize;
        let dr = r / n_steps as f64;
        (0..n_steps).map(|i| self.force(i as f64 * dr) * dr).sum()
    }

    /// Relative extension r/L.
    pub fn relative_extension(&self, r: f64) -> f64 {
        (r / self.contour_length).clamp(0.0, 0.9999)
    }

    /// Stiffness dF/dr at extension r.
    pub fn stiffness(&self, r: f64) -> f64 {
        let x = self.relative_extension(r);
        let p = self.persistence_length;
        let l = self.contour_length;
        self.kt / p * (0.5 / ((1.0 - x) * (1.0 - x) * (1.0 - x)) + 1.0 / l) * 1.0 / l
    }
}

// ─── WLC extended model ───────────────────────────────────────────────────────

/// Worm-like chain Monte Carlo sampler.
///
/// Generates WLC conformations using discrete bond angle model
/// with bending rigidity κ = kT * persistence_length.
#[derive(Debug, Clone)]
pub struct WLCSampler {
    /// Number of segments.
    pub n_segments: usize,
    /// Segment length b.
    pub b: f64,
    /// Persistence length l_p.
    pub persistence_length: f64,
    /// Thermal energy kT.
    pub kt: f64,
}

impl WLCSampler {
    /// Create a new WLC sampler.
    pub fn new(n_segments: usize, b: f64, persistence_length: f64, kt: f64) -> Self {
        WLCSampler {
            n_segments,
            b,
            persistence_length,
            kt,
        }
    }

    /// Bending stiffness κ = kT * l_p / b.
    pub fn bending_stiffness(&self) -> f64 {
        self.kt * self.persistence_length / self.b
    }

    /// Mean cosine of bond angle: <cos θ> = exp(-b / l_p).
    pub fn mean_cos_angle(&self) -> f64 {
        (-self.b / self.persistence_length).exp()
    }

    /// End-to-end distance squared in WLC limit.
    ///
    /// `R²` = 2 * l_p * L * \[1 - (l_p/L)(1 - exp(-L/l_p))\].
    pub fn mean_r2(&self) -> f64 {
        let l = self.n_segments as f64 * self.b;
        let lp = self.persistence_length;
        2.0 * lp * l * (1.0 - (lp / l) * (1.0 - (-l / lp).exp()))
    }

    /// Compute tangent-tangent correlation <t(0)·t(s)> = exp(-s/l_p).
    pub fn tangent_correlation(&self, s: f64) -> f64 {
        (-s / self.persistence_length).exp()
    }

    /// Persistence length from given bond correlation data.
    ///
    /// Fits l_p = -b / ln(<cos θ>).
    pub fn fit_persistence_length(b: f64, mean_cos_theta: f64) -> f64 {
        if mean_cos_theta <= 0.0 || mean_cos_theta >= 1.0 {
            return f64::INFINITY;
        }
        -b / mean_cos_theta.ln()
    }

    /// Radius of gyration for WLC.
    ///
    /// Rg² = (l_p * L / 3)\[1 - (3 l_p / L)(1 - (l_p/L)(1 - exp(-L/l_p)))\].
    pub fn radius_of_gyration_sq(&self) -> f64 {
        let l = self.n_segments as f64 * self.b;
        let lp = self.persistence_length;
        let ratio = lp / l;
        (lp * l / 3.0)
            * (1.0 - 3.0 * ratio + 6.0 * ratio * ratio
                - 6.0 * ratio * ratio * ratio * (1.0 - (-1.0 / ratio).exp()))
    }
}

// ─── Freely-jointed chain ─────────────────────────────────────────────────────

/// Freely-jointed chain (FJC) model with Kuhn segments.
///
/// Each segment has length b (Kuhn length) and is independently oriented.
/// Stretching is described by the Langevin function.
#[derive(Debug, Clone)]
pub struct FreelyJointedChain {
    /// Number of Kuhn segments N.
    pub n_segments: usize,
    /// Kuhn segment length b.
    pub b: f64,
    /// Thermal energy kT.
    pub kt: f64,
}

impl FreelyJointedChain {
    /// Create a new FJC model.
    pub fn new(n_segments: usize, b: f64, kt: f64) -> Self {
        FreelyJointedChain { n_segments, b, kt }
    }

    /// Contour length L = N * b.
    pub fn contour_length(&self) -> f64 {
        self.n_segments as f64 * self.b
    }

    /// Mean-square end-to-end distance `R²` = N * b².
    pub fn mean_r2(&self) -> f64 {
        self.n_segments as f64 * self.b * self.b
    }

    /// Root-mean-square end-to-end distance.
    pub fn rms_end_to_end(&self) -> f64 {
        self.mean_r2().sqrt()
    }

    /// Langevin function L(x) = coth(x) - 1/x.
    pub fn langevin(x: f64) -> f64 {
        if x.abs() < 1e-6 {
            return x / 3.0;
        }
        1.0 / x.tanh() - 1.0 / x
    }

    /// Inverse Langevin via Padé approximation: L⁻¹(y) ≈ y(3 - y²) / (1 - y²).
    pub fn inverse_langevin(y: f64) -> f64 {
        let y = y.clamp(-0.9999, 0.9999);
        y * (3.0 - y * y) / (1.0 - y * y)
    }

    /// Extension `r` = N * b * L(f * b / kT) under force f.
    pub fn mean_extension(&self, force: f64) -> f64 {
        let x = force * self.b / self.kt;
        self.contour_length() * Self::langevin(x)
    }

    /// Force for given extension r using inverse Langevin.
    pub fn force_extension(&self, r: f64) -> f64 {
        let y = r / self.contour_length();
        let y_clamped = y.clamp(-0.9999, 0.9999);
        self.kt / self.b * Self::inverse_langevin(y_clamped)
    }

    /// Entropic spring constant k_eff = 3 kT / (N b²) (Gaussian limit).
    pub fn entropic_spring_constant(&self) -> f64 {
        3.0 * self.kt / (self.n_segments as f64 * self.b * self.b)
    }

    /// Flory exponent ν = 0.5 (ideal chain, no excluded volume).
    pub fn flory_exponent() -> f64 {
        0.5
    }
}

// ─── FENE bond extended model ─────────────────────────────────────────────────

/// FENE (Finitely Extensible Nonlinear Elastic) bond extended model.
///
/// U(r) = -K R₀²/2 * ln(1 - (r/R₀)²)
#[derive(Debug, Clone)]
pub struct FENEBond {
    /// Spring constant K.
    pub k: f64,
    /// Maximum extension R₀.
    pub r0: f64,
    /// Equilibrium bond length r_eq (for reference).
    pub r_eq: f64,
}

impl FENEBond {
    /// Create a new FENE bond.
    pub fn new(k: f64, r0: f64, r_eq: f64) -> Self {
        FENEBond { k, r0, r_eq }
    }

    /// FENE potential energy U(r).
    pub fn potential(&self, r: f64) -> f64 {
        let x = r / self.r0;
        if x >= 1.0 {
            return f64::INFINITY;
        }
        -0.5 * self.k * self.r0 * self.r0 * (1.0 - x * x).ln()
    }

    /// FENE force magnitude F(r) = K*r / (1 - (r/R₀)²).
    pub fn force_magnitude(&self, r: f64) -> f64 {
        let x = r / self.r0;
        if x >= 1.0 {
            return f64::INFINITY;
        }
        self.k * r / (1.0 - x * x)
    }

    /// Maximum force at r → R₀.
    pub fn max_force_approx(&self, epsilon: f64) -> f64 {
        self.force_magnitude(self.r0 * (1.0 - epsilon))
    }

    /// Taylor expansion of potential for small r: U ≈ (1/2) K r².
    pub fn harmonic_approx_stiffness(&self) -> f64 {
        self.k
    }

    /// Combined FENE + WCA (truncated LJ) potential (Kremer-Grest model).
    ///
    /// Returns total potential for separation r with LJ parameters eps, sig.
    pub fn fene_wca_potential(&self, r: f64, eps: f64, sig: f64) -> f64 {
        let r_cut = 2.0_f64.powf(1.0 / 6.0) * sig;
        let u_fene = if r < self.r0 {
            self.potential(r)
        } else {
            f64::INFINITY
        };
        let u_wca = if r < r_cut {
            let s6 = (sig / r).powi(6);
            4.0 * eps * (s6 * s6 - s6) + eps
        } else {
            0.0
        };
        u_fene + u_wca
    }
}

// ─── Kratky-Porod worm-like chain (extended) ─────────────────────────────────

/// Kratky-Porod worm-like chain model with exact partition function.
///
/// The Kratky-Porod chain is the continuous limit of a chain with bending
/// rigidity κ = kT * l_p. Used for semi-flexible polymers (DNA, actin, etc.)
///
/// References:
/// - Kratky, O. & Porod, G. (1949). *Rec. Trav. Chim.* 68, 1106.
/// - Marko, J.F. & Siggia, E.D. (1995). *Macromolecules* 28, 8759.
#[derive(Debug, Clone)]
pub struct KratkyPorodChain {
    /// Contour length L.
    pub contour_length: f64,
    /// Persistence length l_p.
    pub lp: f64,
    /// Thermal energy kT.
    pub kt: f64,
    /// Number of segments for discretization.
    pub n_seg: usize,
}

impl KratkyPorodChain {
    /// Create a new Kratky-Porod chain.
    pub fn new(contour_length: f64, lp: f64, kt: f64, n_seg: usize) -> Self {
        Self {
            contour_length,
            lp,
            kt,
            n_seg,
        }
    }

    /// Bending stiffness κ = kT * l_p.
    pub fn bending_stiffness(&self) -> f64 {
        self.kt * self.lp
    }

    /// Reduced length u = L / (2 l_p).
    pub fn reduced_length(&self) -> f64 {
        self.contour_length / (2.0 * self.lp)
    }

    /// Mean-square end-to-end distance.
    ///
    /// `R²` = 2 l_p L \[1 - (l_p/L)(1 - exp(-L/l_p))\]
    pub fn mean_r2(&self) -> f64 {
        let l = self.contour_length;
        let lp = self.lp;
        2.0 * lp * l * (1.0 - (lp / l) * (1.0 - (-l / lp).exp()))
    }

    /// Radius of gyration (Rg²).
    ///
    /// Rg² = l_p L / 3 * \[1 - (3l_p/L)(1 - (3l_p²/L²)(1 - exp(-L/l_p)))\]
    pub fn radius_of_gyration_sq(&self) -> f64 {
        let l = self.contour_length;
        let lp = self.lp;
        let u = l / lp;
        lp * l / 3.0 * (1.0 - 3.0 / u + 6.0 / (u * u) - 6.0 / (u * u * u) * (1.0 - (-u).exp()))
    }

    /// WLC force extension (Marko-Siggia interpolation formula).
    ///
    /// F = kT/l_p * \[1/(4(1-x)²) - 1/4 + x\]  where x = r/L.
    pub fn force_extension(&self, r: f64) -> f64 {
        let x = (r / self.contour_length).clamp(0.0, 0.9999);
        self.kt / self.lp * (0.25 / ((1.0 - x) * (1.0 - x)) - 0.25 + x)
    }

    /// Extension at given force (numerical inversion using bisection).
    pub fn extension_at_force(&self, force: f64) -> f64 {
        let l = self.contour_length;
        let mut lo = 0.0f64;
        let mut hi = 0.9999 * l;
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if self.force_extension(mid) < force {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Tangent-tangent correlation: <t(s)·t(0)> = exp(-s/l_p).
    pub fn tangent_correlation(&self, s: f64) -> f64 {
        (-s / self.lp.max(1e-30)).exp()
    }

    /// Persistence ratio l_p / L.
    pub fn persistence_ratio(&self) -> f64 {
        self.lp / self.contour_length.max(1e-30)
    }

    /// Classify regime: rod (l_p >> L), semi-flexible (l_p ~ L), or coil (l_p << L).
    pub fn regime(&self) -> &'static str {
        let ratio = self.persistence_ratio();
        if ratio > 10.0 {
            "rod"
        } else if ratio > 0.1 {
            "semi-flexible"
        } else {
            "coil"
        }
    }

    /// Effective spring constant (entropic) at small extension.
    ///
    /// k_eff = 3 kT / (2 l_p L) (WLC linear response).
    pub fn spring_constant_linear(&self) -> f64 {
        3.0 * self.kt / (2.0 * self.lp * self.contour_length)
    }
}

#[cfg(test)]
mod chain_force_tests {
    use super::*;

    /// Deterministic pseudo-random numbers in [-1, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Kremer-Grest-like parameters (R0 = 1.5 σ, K = 30 ε/σ²) with a straight chain
    /// of spacing σ whose beads are displaced by up to `amplitude` in every
    /// direction and given small random velocities.
    fn perturbed_chain(n: usize, seed: u64, amplitude: f64, speed: f64, dt: f64) -> PolymerChain {
        let mut chain = PolymerChain::new(n, 1.5, 30.0, 1.0, 1.0, 5.0, 1.0, dt);
        let mut state = seed;
        for i in 0..n {
            for d in 0..3 {
                chain.positions[i][d] += amplitude * lcg(&mut state);
                chain.velocities[i][d] = speed * lcg(&mut state);
            }
        }
        chain
    }

    fn two_beads(distance: f64) -> PolymerChain {
        let mut chain = PolymerChain::new(2, 1.5, 30.0, 1.0, 1.0, 5.0, 1.0, 1e-3);
        chain.positions[1] = [distance, 0.0, 0.0];
        chain
    }

    #[test]
    fn total_forces_sum_to_zero() {
        // Newton's third law for the FENE bonds and the LJ pairs together.
        for seed in 0..5u64 {
            let chain = perturbed_chain(9, 10 + seed, 0.2, 0.0, 1e-3);
            let mut net = [0.0f64; 3];
            let mut scale = 0.0f64;
            for i in 0..chain.n_beads {
                let f = chain.total_force(i);
                for d in 0..3 {
                    net[d] += f[d];
                }
                scale = scale.max(mag3(f));
            }
            assert!(scale > 1.0, "forces should be non-trivial: {scale}");
            assert!(
                mag3(net) < 1e-9 * scale,
                "net force {net:?} (largest bead force {scale})"
            );
        }
    }

    #[test]
    fn fene_bond_attracts_the_beads() {
        let chain = two_beads(1.2);
        let expected = -fene_force(1.2, 30.0, 1.5); // magnitude of the attraction
        assert!(expected > 0.0);
        let f0 = chain.total_force(0);
        let f1 = chain.total_force(1);
        // Bead 0 is pulled towards +x (bead 1), bead 1 towards -x.
        assert!((f0[0] - expected).abs() < 1e-9 * expected, "f0 = {f0:?}");
        assert!((f1[0] + expected).abs() < 1e-9 * expected, "f1 = {f1:?}");
        assert!(f0[1].abs() < 1e-12 && f0[2].abs() < 1e-12);
        // `fene_force_pair(0)` is the force on bead 1 (reaction on bead 0 is its negative).
        let on_bead_1 = chain.fene_force_pair(0);
        assert!((on_bead_1[0] - f1[0]).abs() < 1e-12);
        assert_eq!(chain.fene_force_pair(1), [0.0; 3]);
    }

    #[test]
    fn stretched_bond_contracts_when_integrated() {
        let mut chain = two_beads(1.4);
        let mut previous = chain.end_to_end();
        for _ in 0..50 {
            chain.step();
            let now = chain.end_to_end();
            assert!(now < previous, "bond must shrink: {previous} -> {now}");
            previous = now;
        }
        assert!(chain.end_to_end() < 1.4);
    }

    #[test]
    fn stretched_chain_contracts() {
        // A straight chain stretched to 1.35 σ per bond (R0 = 1.5 σ) is pulled together.
        let mut chain = PolymerChain::new(6, 1.5, 30.0, 1.0, 1.0, 5.0, 1.0, 5e-4);
        for (i, p) in chain.positions.iter_mut().enumerate() {
            *p = [1.35 * i as f64, 0.0, 0.0];
        }
        let initial = chain.end_to_end();
        let f_first = chain.total_force(0);
        let f_last = chain.total_force(5);
        assert!(f_first[0] > 0.0, "first bead pulled inwards: {f_first:?}");
        assert!(f_last[0] < 0.0, "last bead pulled inwards: {f_last:?}");
        for _ in 0..100 {
            chain.step();
        }
        let after = chain.end_to_end();
        assert!(after < initial, "chain must contract: {initial} -> {after}");
        assert!(after.is_finite());
        // No bond may exceed the finite extensibility R0.
        for i in 0..5 {
            assert!(dist3(chain.positions[i], chain.positions[i + 1]) < 1.5);
        }
    }

    #[test]
    fn lj_force_is_repulsive_inside_the_minimum_and_attractive_outside() {
        let mut chain = PolymerChain::new(3, 1.5, 30.0, 1.0, 1.0, 5.0, 1.0, 1e-3);
        let r_min = 2.0f64.powf(1.0 / 6.0);
        // Beads 0 and 2 are not bonded; bead 1 is irrelevant for `lj_force_pair`.
        chain.positions[2] = [0.9 * r_min, 0.0, 0.0];
        let f = chain.lj_force_pair(0, 2);
        assert!(
            f[0] < 0.0,
            "repulsion pushes bead 0 away from bead 2: {f:?}"
        );
        let expected = 24.0
            * (2.0 * (1.0 / (0.9 * r_min)).powi(12) - (1.0 / (0.9 * r_min)).powi(6))
            / (0.9 * r_min);
        assert!((f[0] + expected).abs() < 1e-9 * expected);

        chain.positions[2] = [1.6 * r_min, 0.0, 0.0];
        let f = chain.lj_force_pair(0, 2);
        assert!(f[0] > 0.0, "attraction pulls bead 0 towards bead 2: {f:?}");

        chain.positions[2] = [r_min, 0.0, 0.0];
        assert!(chain.lj_force_pair(0, 2)[0].abs() < 1e-9);
    }

    #[test]
    fn lj_pair_forces_obey_newtons_third_law() {
        let chain = perturbed_chain(6, 77, 0.3, 0.0, 1e-3);
        for i in 0..6 {
            for j in 0..6 {
                let fij = chain.lj_force_pair(i, j);
                let fji = chain.lj_force_pair(j, i);
                for d in 0..3 {
                    assert!((fij[d] + fji[d]).abs() < 1e-12 * (1.0 + fij[d].abs()));
                }
            }
        }
    }

    #[test]
    fn forces_are_the_negative_gradient_of_the_potential_energy() {
        let chain = perturbed_chain(7, 5, 0.25, 0.0, 1e-3);
        let h = 1e-6;
        for i in 0..chain.n_beads {
            let analytic = chain.total_force(i);
            for (d, &force) in analytic.iter().enumerate() {
                let mut plus = chain.clone();
                plus.positions[i][d] += h;
                let mut minus = chain.clone();
                minus.positions[i][d] -= h;
                let numeric = -(plus.potential_energy() - minus.potential_energy()) / (2.0 * h);
                assert!(
                    (numeric - force).abs() < 1e-5 * (1.0 + force.abs()),
                    "bead {i} component {d}: -dU/dx = {numeric}, F = {force}"
                );
            }
        }
    }

    #[test]
    fn potential_energy_matches_the_fene_closed_form() {
        let chain = two_beads(1.2);
        let bond = FENEBond::new(30.0, 1.5, 0.0);
        let expected = bond.potential(1.2);
        assert!(expected > 0.0);
        assert!((chain.potential_energy() - expected).abs() < 1e-12);
        // At or beyond R0 the finite-extensibility energy is infinite.
        assert!(two_beads(1.5).potential_energy().is_infinite());
    }

    #[test]
    fn step_is_velocity_verlet() {
        let mut chain = perturbed_chain(5, 3, 0.15, 0.3, 2e-3);
        let reference = chain.clone();
        chain.step();

        let dt = reference.dt;
        let f0 = reference.forces();
        let mut x = reference.positions.clone();
        let mut v = reference.velocities.clone();
        for i in 0..5 {
            for d in 0..3 {
                v[i][d] += 0.5 * dt * f0[i][d];
                x[i][d] += dt * v[i][d];
            }
        }
        let mut moved = reference.clone();
        moved.positions = x.clone();
        let f1 = moved.forces();
        for i in 0..5 {
            for d in 0..3 {
                v[i][d] += 0.5 * dt * f1[i][d];
                assert!((chain.positions[i][d] - x[i][d]).abs() < 1e-14);
                assert!((chain.velocities[i][d] - v[i][d]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn dynamics_are_time_reversible() {
        // Velocity Verlet is exactly time reversible; symplectic Euler is not.
        let mut chain = perturbed_chain(6, 21, 0.12, 0.4, 1e-3);
        let start = chain.positions.clone();
        for _ in 0..200 {
            chain.step();
        }
        assert!(
            chain
                .positions
                .iter()
                .zip(start.iter())
                .any(|(a, b)| dist3(*a, *b) > 1e-3),
            "the chain should have moved"
        );
        for v in chain.velocities.iter_mut() {
            *v = [-v[0], -v[1], -v[2]];
        }
        for _ in 0..200 {
            chain.step();
        }
        for (a, b) in chain.positions.iter().zip(start.iter()) {
            assert!(dist3(*a, *b) < 1e-9, "not reversible: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn total_energy_is_conserved_over_1000_steps() {
        // Velocity Verlet keeps the relative energy fluctuation of these violently
        // relaxing chains below ~1e-4 at dt = 5e-4; a first-order symplectic-Euler
        // update drifts by 0.5-1 % here, so the 1e-3 bound separates the schemes.
        for seed in 0..3u64 {
            let mut chain = perturbed_chain(8, 40 + seed, 0.12, 0.3, 5e-4);
            let e0 = chain.total_energy();
            assert!(e0.is_finite());
            let mut worst = 0.0f64;
            for _ in 0..1000 {
                chain.step();
                worst = worst.max((chain.total_energy() - e0).abs());
            }
            assert!(
                worst < 1e-3 * e0.abs().max(1.0),
                "seed {seed}: energy drifted by {worst} from {e0}"
            );
        }
    }
}
