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
        assert!((c_low - expected).abs() < 1e-12 * expected, "{c_low} vs {expected}");
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
        let solve = |p: &ViscosityParticles<'_>, dt: f64, h: f64| {
            solve_implicit_viscosity(p, &neighbors, dt, h, ViscositySolveOptions::default())
        };
        // dt <= 0, dt not finite
        assert!(matches!(solve(&particles, 0.0, 0.1), Err(ViscosityError::InvalidConfig)));
        assert!(matches!(solve(&particles, -1.0, 0.1), Err(ViscosityError::InvalidConfig)));
        assert!(matches!(solve(&particles, f64::NAN, 0.1), Err(ViscosityError::InvalidConfig)));
        // h <= 0
        assert!(matches!(solve(&particles, 0.01, 0.0), Err(ViscosityError::InvalidConfig)));
        // negative viscosity
        let negative_mu = vec![1.0, -1.0];
        let bad = ViscosityParticles {
            viscosities: &negative_mu,
            ..particles
        };
        assert!(matches!(solve(&bad, 0.01, 0.1), Err(ViscosityError::InvalidConfig)));
        // non-positive mass
        let zero_mass = vec![0.001, 0.0];
        let bad = ViscosityParticles {
            masses: &zero_mass,
            ..particles
        };
        assert!(matches!(solve(&bad, 0.01, 0.1), Err(ViscosityError::InvalidConfig)));
        // non-positive density
        let zero_density = vec![1000.0, 0.0];
        let bad = ViscosityParticles {
            densities: &zero_density,
            ..particles
        };
        assert!(matches!(solve(&bad, 0.01, 0.1), Err(ViscosityError::InvalidConfig)));
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
        assert!((c_01 - c_10).abs() > 0.1 * c_01.max(c_10), "{c_01} vs {c_10}");
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
        for i in 0..n {
            dense[i][i] = system.diagonal[i];
            for &(j, a_ij) in &system.off_diagonals[i] {
                dense[i][j] += a_ij;
            }
        }
        let mut stored_pairs = 0;
        for i in 0..n {
            let mut off_sum = 0.0;
            for j in 0..n {
                assert_eq!(dense[i][j], dense[j][i], "A not symmetric at ({i},{j})");
                if i != j {
                    assert!(dense[i][j] <= 0.0, "off-diagonal must be non-positive");
                    off_sum += dense[i][j].abs();
                    if dense[i][j] < 0.0 {
                        stored_pairs += 1;
                    }
                }
            }
            // M-matrix with A_ii = m̂_i + Σ_j |A_ij|.
            assert!(
                (dense[i][i] - system.scaled_masses[i] - off_sum).abs() < 1e-12 * dense[i][i],
                "row {i}: diag {} vs m + sum|off| {}",
                dense[i][i],
                system.scaled_masses[i] + off_sum
            );
        }
        assert_eq!(stored_pairs, n * (n - 1), "the cross is fully connected");
        // The normalised masses are m_i / max m.
        assert!((system.scaled_masses[3] - 1.0).abs() < 1e-15);
        assert!((system.scaled_masses[2] - 0.5 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn unequal_masses_conserve_momentum_and_do_not_add_energy() {
        let fixture = unequal_cross();
        let out = fixture.solve(0.05).expect("solve should succeed");
        let momentum = |v: &[[f64; 3]]| -> [f64; 3] {
            let mut p = [0.0_f64; 3];
            for (vi, &m) in v.iter().zip(fixture.masses.iter()) {
                for k in 0..3 {
                    p[k] += m * vi[k];
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
        assert!(e_after < e_before, "viscosity must dissipate: {e_before} -> {e_after}");
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
            for k in 0..3 {
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
                    laplacian += c * (out[j][k] - out[i][k]);
                }
                let residual = out[i][k] - dt * laplacian - fixture.velocities[i][k];
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
            vec![NeighborEntry::new(1, r_01, g), NeighborEntry::new(1, r_01, g)],
            vec![NeighborEntry::new(0, r_10, g_neg), NeighborEntry::new(0, r_10, g_neg)],
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
            matches!(err, Err(ViscosityError::SolverNotConverged { iters: 1, .. })),
            "{err:?}"
        );
    }
}
