//! F4180: the pressure projection removes (instead of doubling) the divergence.

use super::*;

/// Central-difference divergence `D v` at an interior node (the operator that
/// `pressure_projection` uses).
fn central_divergence(grid: &MpmGrid, ix: usize, iy: usize, iz: usize) -> f64 {
    let v = |x: usize, y: usize, z: usize| grid.nodes[grid.idx(x, y, z)].velocity;
    ((v(ix + 1, iy, iz)[0] - v(ix - 1, iy, iz)[0])
        + (v(ix, iy + 1, iz)[1] - v(ix, iy - 1, iz)[1])
        + (v(ix, iy, iz + 1)[2] - v(ix, iy, iz - 1)[2]))
        * 0.5
        / grid.h
}

/// Largest `|D v|` over the nodes at least `margin` layers away from the grid faces.
fn max_divergence(grid: &MpmGrid, margin: usize) -> f64 {
    let [nx, ny, nz] = grid.dims;
    let mut worst = 0.0f64;
    for ix in margin..nx - margin {
        for iy in margin..ny - margin {
            for iz in margin..nz - margin {
                worst = worst.max(central_divergence(grid, ix, iy, iz).abs());
            }
        }
    }
    worst
}

/// 16³ grid on the unit cube, unit mass everywhere, `v = (sin(2 pi x), 0, 0)`.
fn sinusoidal_grid() -> MpmGrid {
    let mut grid = MpmGrid::new([16, 16, 16], 1.0 / 16.0, [0.0; 3]);
    let [nx, ny, nz] = grid.dims;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = grid.idx(ix, iy, iz);
                let x = grid.node_pos(ix, iy, iz)[0];
                grid.nodes[idx].mass = 1.0;
                grid.nodes[idx].velocity = [(2.0 * std::f64::consts::PI * x).sin(), 0.0, 0.0];
            }
        }
    }
    grid
}

#[test]
fn f4180_projection_reduces_the_interior_divergence_of_a_sinusoidal_field() {
    let mut grid = sinusoidal_grid();
    let before = max_divergence(&grid, 2);
    assert!(before > 5.0, "test field has divergence {before}");
    pressure_projection(&mut grid, 0.01, 1000.0, 200);
    let after = max_divergence(&grid, 2);
    // The inverted sign of the old right-hand side doubled the divergence.
    assert!(after < 0.1 * before, "divergence {before} -> {after}");
    // 200 Jacobi sweeps leave the smoothest (slowest) mode only partly converged.
    assert!(after < 0.01 * before, "divergence {before} -> {after}");

    // The same holds for every node whose velocity is corrected, including the
    // layer next to the free boundary.
    let mut grid = sinusoidal_grid();
    let before_all = max_divergence(&grid, 1);
    pressure_projection(&mut grid, 0.01, 1000.0, 200);
    let after_all = max_divergence(&grid, 1);
    assert!(
        after_all < 0.1 * before_all,
        "divergence {before_all} -> {after_all}"
    );
}

#[test]
fn f4180_converged_projection_is_exact_for_the_discrete_divergence() {
    let mut grid = sinusoidal_grid();
    let before = max_divergence(&grid, 1);
    pressure_projection(&mut grid, 0.01, 1000.0, 1500);
    // Every node whose velocity can be corrected ends up with D v' = 0.
    let after = max_divergence(&grid, 1);
    assert!(after < 1e-9 * before, "divergence {before} -> {after}");
    // The boundary layer is a free surface: its velocity is not modified.
    let [nx, ny, nz] = grid.dims;
    let reference = sinusoidal_grid();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let boundary =
                    ix == 0 || iy == 0 || iz == 0 || ix == nx - 1 || iy == ny - 1 || iz == nz - 1;
                if boundary {
                    let idx = grid.idx(ix, iy, iz);
                    assert_eq!(grid.nodes[idx].velocity, reference.nodes[idx].velocity);
                }
            }
        }
    }
}

#[test]
fn f4180_projection_leaves_a_divergence_free_field_unchanged() {
    let mut grid = MpmGrid::new([12, 12, 12], 0.1, [0.0; 3]);
    for node in &mut grid.nodes {
        node.mass = 2.0;
        node.velocity = [0.3, -0.1, 0.2];
    }
    pressure_projection(&mut grid, 0.01, 1000.0, 100);
    for node in &grid.nodes {
        assert!(vec_diff(node.velocity, [0.3, -0.1, 0.2]) < 1e-13);
    }
}

#[test]
fn f4180_projection_treats_massless_nodes_as_a_free_surface() {
    let mut grid = MpmGrid::new([16, 16, 16], 1.0 / 16.0, [0.0; 3]);
    let [nx, ny, nz] = grid.dims;
    let two_pi = 2.0 * std::f64::consts::PI;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = grid.idx(ix, iy, iz);
                let x = grid.node_pos(ix, iy, iz);
                grid.nodes[idx].velocity = [(two_pi * x[0]).sin(), (two_pi * x[1]).cos(), 0.0];
                // Fluid occupies the block of nodes 4..12 in every direction.
                let inside =
                    (4..12).contains(&ix) && (4..12).contains(&iy) && (4..12).contains(&iz);
                grid.nodes[idx].mass = if inside { 1.0 } else { 0.0 };
            }
        }
    }
    let reference: Vec<[f64; 3]> = grid.nodes.iter().map(|n| n.velocity).collect();
    let before = max_divergence(&grid, 4);
    pressure_projection(&mut grid, 0.01, 1000.0, 1500);
    // Empty space keeps its velocity exactly.
    for (idx, node) in grid.nodes.iter().enumerate() {
        if node.mass == 0.0 {
            assert_eq!(
                node.velocity, reference[idx],
                "node {idx} outside the fluid changed"
            );
        }
    }
    // Inside the fluid the divergence is removed everywhere the pressure can act.
    let mut worst = 0.0f64;
    for ix in 4..12 {
        for iy in 4..12 {
            for iz in 4..12 {
                worst = worst.max(central_divergence(&grid, ix, iy, iz).abs());
            }
        }
    }
    assert!(before > 1.0);
    assert!(
        worst < 1e-8 * before,
        "divergence inside the fluid: {worst}"
    );
}

#[test]
fn f4180_projection_ignores_degenerate_parameters() {
    let reference = sinusoidal_grid();
    for (dt, rho) in [
        (0.0, 1000.0),
        (-1.0, 1000.0),
        (f64::NAN, 1000.0),
        (0.01, 0.0),
        (0.01, f64::INFINITY),
    ] {
        let mut grid = sinusoidal_grid();
        pressure_projection(&mut grid, dt, rho, 50);
        for (node, original) in grid.nodes.iter().zip(&reference.nodes) {
            assert_eq!(node.velocity, original.velocity, "dt = {dt}, rho = {rho}");
        }
    }
    // No iterations means no pressure and no change.
    let mut grid = sinusoidal_grid();
    pressure_projection(&mut grid, 0.01, 1000.0, 0);
    for (node, original) in grid.nodes.iter().zip(&reference.nodes) {
        assert_eq!(node.velocity, original.velocity);
    }
    // Grids without an interior must not panic (the old loops underflowed).
    for dims in [[1, 1, 1], [2, 5, 5], [5, 2, 5], [5, 5, 2], [0, 4, 4]] {
        let mut grid = MpmGrid::new(dims, 0.1, [0.0; 3]);
        for node in &mut grid.nodes {
            node.mass = 1.0;
            node.velocity = [1.0, 2.0, 3.0];
        }
        pressure_projection(&mut grid, 0.01, 1000.0, 10);
        for node in &grid.nodes {
            assert_eq!(node.velocity, [1.0, 2.0, 3.0]);
        }
    }
    // The smallest grid with an interior node runs and leaves it finite.
    let mut grid = MpmGrid::new([3, 3, 3], 0.1, [0.0; 3]);
    for node in &mut grid.nodes {
        node.mass = 1.0;
    }
    let centre = grid.idx(1, 1, 1);
    grid.nodes[centre].velocity = [1.0, 0.0, 0.0];
    pressure_projection(&mut grid, 0.01, 1000.0, 10);
    assert!(
        grid.nodes
            .iter()
            .all(|n| n.velocity.iter().all(|c| c.is_finite()))
    );
}
