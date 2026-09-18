// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Moving solid walls for [`D3q19Simulation`] (Ladd's momentum correction of the on-node
//! bounce-back).
//!
//! [`D3q19Simulation::apply_bounce_back`] reflects every population that reached a solid node,
//! which realises a wall at rest. A wall that moves tangentially (the lid of a driven cavity, the
//! upper plate of a Couette channel) must in addition hand its momentum to the populations that
//! leave it for the fluid; [`D3q19Simulation::apply_moving_wall_correction`] adds that term.

use super::{CS2, D3Q19_EX, D3Q19_EY, D3Q19_EZ, D3Q19_WEIGHTS, D3q19Simulation, N_DIRS};

/// Coordinate `coordinate + step` of a periodic axis with `extent` nodes.
#[inline]
fn wrapped(coordinate: usize, step: i32, extent: usize) -> usize {
    (coordinate as i64 + i64::from(step)).rem_euclid(extent as i64) as usize
}

impl D3q19Simulation {
    /// Add the momentum a moving wall hands to the populations leaving its solid nodes (Ladd 1994).
    ///
    /// After [`Self::apply_bounce_back`] every solid node holds the reflected populations. For a
    /// wall moving with the velocity `u_w` the reflected population of every direction `i` whose
    /// destination node `x + e_i` is a **fluid** node must be raised by
    ///
    /// `Δf_i = 2 w_i ρ₀ (e_i · u_w) / cs²`   (`cs² = 1/3`),
    ///
    /// so that the fluid sees a no-slip wall with the velocity `u_w` half-way between the wall
    /// node and the fluid node. Destinations are found like in [`Self::stream`] (periodic
    /// wrap-around). Links into other solid nodes (along the wall, or through the periodic wrap
    /// into another wall) are left alone: those populations never reach the fluid, and
    /// correcting them would only pile up momentum inside the wall.
    ///
    /// * `wall_nodes` — flat node indices (`x + y·nx + z·nx·ny`) of the moving wall, each listed
    ///   once. Indices that are out of range or not flagged in [`Self::solid`] are ignored.
    /// * `wall_velocity` — wall velocity `u_w` in lattice units (keep `|u_w| ≪ cs`).
    /// * `reference_density` — the density `ρ₀` of the fluid at the wall (1 in lattice units for
    ///   an incompressible flow). It is deliberately a parameter: the populations of a solid
    ///   node are bookkeeping, their sum has no physical meaning, and using it as the wall density
    ///   feeds the correction back on itself and makes a driven cavity diverge.
    ///
    /// Call it **once per step**, right after [`Self::apply_bounce_back`] (or [`Self::step`],
    /// [`Self::step_mrt`], [`Self::step_trt`], which include it) and before the next collision.
    /// The wall nodes must be flagged solid so that the collision skips them.
    ///
    /// A flat wall moving *tangentially* conserves mass (the corrections of opposite
    /// links cancel) and injects the momentum `ρ₀ u_w / 3` per wall node and step. A wall velocity
    /// with a component normal to the wall injects mass as well.
    ///
    /// Returns the number of links from the wall nodes into fluid nodes.
    pub fn apply_moving_wall_correction(
        &mut self,
        wall_nodes: &[usize],
        wall_velocity: [f64; 3],
        reference_density: f64,
    ) -> usize {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let node_count = nx * ny * nz;
        let mut fluid_links = 0;
        for &node in wall_nodes {
            if node >= node_count || !self.solid[node] {
                continue;
            }
            let x = node % nx;
            let y = (node / nx) % ny;
            let z = node / (nx * ny);
            for i in 1..N_DIRS {
                let destination = self.idx(
                    wrapped(x, D3Q19_EX[i], nx),
                    wrapped(y, D3Q19_EY[i], ny),
                    wrapped(z, D3Q19_EZ[i], nz),
                );
                if self.solid[destination] {
                    continue;
                }
                fluid_links += 1;
                let e_dot_u = f64::from(D3Q19_EX[i]) * wall_velocity[0]
                    + f64::from(D3Q19_EY[i]) * wall_velocity[1]
                    + f64::from(D3Q19_EZ[i]) * wall_velocity[2];
                self.f[node * N_DIRS + i] +=
                    2.0 * D3Q19_WEIGHTS[i] * reference_density * e_dot_u / CS2;
            }
        }
        fluid_links
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat node index.
    fn node(sim: &D3q19Simulation, coordinates: [usize; 3]) -> usize {
        sim.idx(coordinates[0], coordinates[1], coordinates[2])
    }

    /// Nodes of the layer `coordinate[axis] == layer`.
    fn layer(sim: &D3q19Simulation, axis: usize, layer: usize) -> Vec<usize> {
        let mut nodes = Vec::new();
        for z in 0..sim.nz {
            for y in 0..sim.ny {
                for x in 0..sim.nx {
                    if [x, y, z][axis] == layer {
                        nodes.push(sim.idx(x, y, z));
                    }
                }
            }
        }
        nodes
    }

    /// Plane Couette flow between a resting wall at `coordinate[wall_axis] = 0` and a wall at the
    /// largest index moving with `u_wall` along `flow_axis`. Returns the largest deviation of the
    /// steady velocity profile from the linear profile `u_w (s - 1/2) / H`, in units of `u_w`
    /// (the no-slip planes sit half-way between the wall nodes and the first fluid nodes, so
    /// `H = n - 2`).
    fn couette_error(wall_axis: usize, flow_axis: usize) -> f64 {
        assert_ne!(wall_axis, flow_axis);
        let mut dims = [3usize, 3, 3];
        dims[wall_axis] = 22;
        let u_wall = 0.05;
        let tau = 0.8;
        let mut sim = D3q19Simulation::new(dims[0], dims[1], dims[2], tau);
        let bottom = layer(&sim, wall_axis, 0);
        let top = layer(&sim, wall_axis, dims[wall_axis] - 1);
        for &k in bottom.iter().chain(top.iter()) {
            sim.solid[k] = true;
        }
        let mut velocity = [0.0; 3];
        velocity[flow_axis] = u_wall;
        for _ in 0..6000 {
            sim.step();
            sim.apply_moving_wall_correction(&top, velocity, 1.0);
        }
        let gap = (dims[wall_axis] - 2) as f64;
        let mut worst = 0.0_f64;
        for s in 1..dims[wall_axis] - 1 {
            let mut coordinates = [1usize; 3];
            coordinates[wall_axis] = s;
            let (rho, u) = sim.macroscopic(node(&sim, coordinates));
            assert!((rho - 1.0).abs() < 1e-9, "density {rho} at layer {s}");
            let exact = u_wall * (s as f64 - 0.5) / gap;
            worst = worst.max((u[flow_axis] - exact).abs() / u_wall);
            for (axis, component) in u.iter().enumerate() {
                if axis != flow_axis {
                    assert!(
                        component.abs() < 1e-9,
                        "cross flow {component} along {axis}"
                    );
                }
            }
        }
        worst
    }

    #[test]
    fn couette_flow_between_a_resting_and_a_moving_wall_is_linear() {
        // wall normal y, flow along x: the profile is exact to the solver tolerance.
        let error = couette_error(1, 0);
        assert!(error < 1e-5, "profile error {error:.3e} u_w");
    }

    #[test]
    fn the_moving_wall_correction_works_for_every_wall_orientation() {
        for (wall_axis, flow_axis) in [(1, 2), (0, 1), (0, 2), (2, 0), (2, 1)] {
            let error = couette_error(wall_axis, flow_axis);
            assert!(
                error < 1e-5,
                "wall normal {wall_axis}, flow along {flow_axis}: profile error {error:.3e} u_w"
            );
        }
    }

    /// 5x6x5 box with a resting wall row at `y = 0` and a moving one at `y = 5`.
    fn lid_box() -> (D3q19Simulation, Vec<usize>) {
        let mut sim = D3q19Simulation::new(5, 6, 5, 1.0);
        let bottom = layer(&sim, 1, 0);
        let top = layer(&sim, 1, 5);
        for &k in bottom.iter().chain(top.iter()) {
            sim.solid[k] = true;
        }
        (sim, top)
    }

    #[test]
    fn only_links_into_fluid_nodes_are_corrected() {
        let (mut sim, top) = lid_box();
        let reference = sim.f.clone();
        let u_wall = [0.04, 0.0, 0.02];
        let rho0 = 1.2;
        let links = sim.apply_moving_wall_correction(&top, u_wall, rho0);
        // Five links per node leave the lid downwards into the fluid; the links along the row and
        // the upward ones (periodic wrap into the resting wall row) end in solid nodes.
        assert_eq!(links, 5 * top.len());
        for (k, (&after, &before)) in sim.f.iter().zip(reference.iter()).enumerate() {
            let (cell, i) = (k / N_DIRS, k % N_DIRS);
            let expected = if top.contains(&cell) && D3Q19_EY[i] == -1 {
                let e_dot_u =
                    f64::from(D3Q19_EX[i]) * u_wall[0] + f64::from(D3Q19_EZ[i]) * u_wall[2];
                6.0 * D3Q19_WEIGHTS[i] * rho0 * e_dot_u
            } else {
                0.0
            };
            assert!(
                (after - before - expected).abs() < 1e-15,
                "node {cell}, direction {i}: change {} vs {expected}",
                after - before
            );
        }
    }

    #[test]
    fn tangential_wall_motion_injects_rho_u_over_three_of_momentum_and_no_mass() {
        let (mut sim, _) = lid_box();
        let single = [node(&sim, [2, 5, 2])];
        let reference = sim.f.clone();
        let u_wall = [0.03, 0.0, -0.06];
        let rho0 = 0.9;
        sim.apply_moving_wall_correction(&single, u_wall, rho0);
        let mut mass = 0.0;
        let mut momentum = [0.0; 3];
        for i in 0..N_DIRS {
            let change = sim.f[single[0] * N_DIRS + i] - reference[single[0] * N_DIRS + i];
            mass += change;
            momentum[0] += change * f64::from(D3Q19_EX[i]);
            momentum[1] += change * f64::from(D3Q19_EY[i]);
            momentum[2] += change * f64::from(D3Q19_EZ[i]);
        }
        assert!(mass.abs() < 1e-15, "mass change {mass}");
        assert!(
            (momentum[0] - rho0 * u_wall[0] / 3.0).abs() < 1e-15,
            "{momentum:?}"
        );
        assert!(momentum[1].abs() < 1e-15, "{momentum:?}");
        assert!(
            (momentum[2] - rho0 * u_wall[2] / 3.0).abs() < 1e-15,
            "{momentum:?}"
        );
        // Nothing else was touched.
        let touched = sim
            .f
            .iter()
            .zip(reference.iter())
            .filter(|(after, before)| after != before)
            .count();
        assert!(touched <= 5);
    }

    #[test]
    fn a_wall_moving_along_its_normal_injects_mass() {
        let (mut sim, _) = lid_box();
        let single = [node(&sim, [2, 5, 2])];
        let reference = sim.f.clone();
        sim.apply_moving_wall_correction(&single, [0.0, -0.05, 0.0], 1.0);
        let mass: f64 = (0..N_DIRS)
            .map(|i| sim.f[single[0] * N_DIRS + i] - reference[single[0] * N_DIRS + i])
            .sum();
        // The wall moves towards the fluid (-y): the five links into the fluid gain
        // 6 w_i (e_i . u_w) = 0.3 * 0.05 * w_i / (1/18) summed over (0,-1,0), (+-1,-1,0), (0,-1,+-1).
        let expected = 6.0 * 0.05 * (1.0 / 18.0 + 4.0 / 36.0);
        assert!(
            (mass - expected).abs() < 1e-15,
            "mass change {mass} vs {expected}"
        );
    }

    #[test]
    fn zero_velocity_fluid_and_out_of_range_nodes_leave_the_populations_alone() {
        let (mut sim, top) = lid_box();
        let reference = sim.f.clone();
        // Zero wall velocity: nothing changes, the fluid-facing links are still counted.
        assert_eq!(
            sim.apply_moving_wall_correction(&top, [0.0; 3], 1.0),
            5 * top.len()
        );
        assert_eq!(sim.f, reference);
        // A fluid node and indices past the end of the grid are ignored.
        let fluid = node(&sim, [2, 2, 2]);
        let past_the_end = sim.nx * sim.ny * sim.nz;
        assert_eq!(
            sim.apply_moving_wall_correction(
                &[fluid, past_the_end, usize::MAX],
                [0.1, 0.0, 0.0],
                1.0
            ),
            0
        );
        assert_eq!(sim.f, reference);
    }

    #[test]
    fn a_resting_fluid_between_walls_moving_alike_stays_at_the_wall_velocity() {
        // Both plates move with the same velocity: the fluid is in rigid translation, which is a
        // steady solution of the lattice scheme, so the correction must hold it exactly.
        let u_wall = 0.04;
        let mut sim = D3q19Simulation::new(3, 12, 3, 0.9);
        let mut walls = layer(&sim, 1, 0);
        walls.extend(layer(&sim, 1, 11));
        for &k in &walls {
            sim.solid[k] = true;
        }
        sim.init_with_velocity(1.0, |_, _, _| [u_wall, 0.0, 0.0]);
        for _ in 0..400 {
            sim.step();
            sim.apply_moving_wall_correction(&walls, [u_wall, 0.0, 0.0], 1.0);
        }
        for y in 1..11 {
            let (rho, u) = sim.macroscopic(node(&sim, [1, y, 1]));
            assert!((rho - 1.0).abs() < 1e-3, "density {rho}");
            assert!(
                (u[0] - u_wall).abs() < 1e-3 * u_wall,
                "u = {u:?} at row {y}"
            );
        }
    }
}
