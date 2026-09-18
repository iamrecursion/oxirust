//! Tests of the finite-element heat conduction of [`ThermalHistoryFem`].

use super::*;
use crate::mesh::TetrahedralMesh;

const CONDUCTIVITY: f64 = 12.0;
const SPECIFIC_HEAT: f64 = 620.0;
const DENSITY: f64 = 8000.0;

/// Complementary error function for `x >= 0` from the Maclaurin series of `erf`,
/// accurate to far better than 1e-9 on the range used here (`x < 4`).
fn erfc(x: f64) -> f64 {
    if x >= 4.0 {
        return 0.0;
    }
    let mut term = x;
    let mut sum = x;
    for n in 1..80 {
        let k = n as f64;
        term *= -x * x / k;
        sum += term / (2.0 * k + 1.0);
    }
    1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
}

fn positions_of(mesh: &TetrahedralMesh) -> Vec<[f64; 3]> {
    mesh.nodes.iter().map(|p| [p.x, p.y, p.z]).collect()
}

/// A source that deposits no heat.
fn idle_source() -> GoldakSource {
    GoldakSource::new(0.0, 0.35, 1.5e-4, 1.0e-4, 1.0e-4, 2.0e-4)
}

fn model_on(mesh: &TetrahedralMesh, source: GoldakSource) -> ThermalHistoryFem {
    ThermalHistoryFem::with_mesh(
        positions_of(mesh),
        mesh.elements.clone(),
        source,
        1923.0,
        1723.0,
        CONDUCTIVITY,
        SPECIFIC_HEAT,
        DENSITY,
    )
    .expect("a valid beam mesh")
}

/// Nodes of `model` whose x coordinate equals `x`.
fn nodes_at_x(model: &ThermalHistoryFem, x: f64) -> Vec<usize> {
    model
        .positions
        .iter()
        .enumerate()
        .filter(|&(_, p)| (p[0] - x).abs() < 1e-12)
        .map(|(i, _)| i)
        .collect()
}

fn one_tetrahedron() -> (Vec<[f64; 3]>, Vec<[usize; 4]>) {
    (
        vec![
            [0.0, 0.0, 0.0],
            [1.0e-3, 0.0, 0.0],
            [0.0, 1.0e-3, 0.0],
            [0.0, 0.0, 1.0e-3],
        ],
        vec![[0, 1, 2, 3]],
    )
}

fn model_from(positions: Vec<[f64; 3]>, elements: Vec<[usize; 4]>) -> Result<ThermalHistoryFem> {
    ThermalHistoryFem::with_mesh(
        positions,
        elements,
        idle_source(),
        1923.0,
        1723.0,
        CONDUCTIVITY,
        SPECIFIC_HEAT,
        DENSITY,
    )
}

// ---- operators ----------------------------------------------------------

#[test]
fn lumped_capacity_and_exposed_area_match_the_geometry() {
    // A 2 mm x 1 mm x 1 mm block.
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 4, 2, 2);
    let model = model_on(&mesh, idle_source());
    let operators = model.operators.as_ref().expect("operators");
    let volume = 2.0e-9;
    let total_capacity: f64 = operators.capacity.iter().sum();
    assert!(
        (total_capacity - DENSITY * SPECIFIC_HEAT * volume).abs() < 1e-12 * total_capacity,
        "total capacity {total_capacity}"
    );
    let total_volume: f64 = operators.element_volume.iter().sum();
    assert!((total_volume - volume).abs() < 1e-12 * volume);
    // 2·(LW + LH + WH) of the box; the former model had no surface area at all.
    let surface = 2.0 * (2.0e-6 + 2.0e-6 + 1.0e-6);
    let total_area: f64 = operators.exposed_area.iter().sum();
    assert!(
        (total_area - surface).abs() < 1e-12 * surface,
        "exposed area {total_area}, expected {surface}"
    );
    // Interior nodes expose no surface.
    let interior = operators
        .exposed_area
        .iter()
        .filter(|&&area| area == 0.0)
        .count();
    assert!(interior >= 3, "the 4x2x2 block has interior nodes");
}

#[test]
fn conduction_matrix_is_symmetric_and_conserves_heat() {
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 4, 2, 2);
    let model = model_on(&mesh, idle_source());
    let operators = model.operators.as_ref().expect("operators");
    let k = &operators.conduction;
    let n = model.positions.len();
    let largest = k.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // Row sums vanish: a uniform temperature conducts no heat.
    let row_sums = k.mul_vec(&vec![1.0; n]);
    for (row, sum) in row_sums.iter().enumerate() {
        assert!(sum.abs() < 1e-12 * largest, "row {row} sums to {sum}");
    }
    // Symmetry and a positive diagonal.
    for row in 0..n {
        assert!(k.get(row, row) > 0.0);
        for col in 0..n {
            let difference = (k.get(row, col) - k.get(col, row)).abs();
            assert!(difference < 1e-12 * largest);
        }
    }
}

#[test]
fn conduction_matrix_passes_the_linear_patch_test() {
    // A linear temperature field has a constant gradient and no divergence: the
    // nodal conduction residual must vanish at every node without surface.
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 4, 2, 2);
    let model = model_on(&mesh, idle_source());
    let operators = model.operators.as_ref().expect("operators");
    let field: Vec<f64> = model
        .positions
        .iter()
        .map(|p| 300.0 + 1.0e5 * p[0] - 4.0e4 * p[1] + 2.5e4 * p[2])
        .collect();
    let residual = operators.conduction.mul_vec(&field);
    let scale = CONDUCTIVITY * 1.0e5 * 1.0e-6;
    let mut checked = 0;
    for (node, value) in residual.iter().enumerate() {
        if operators.exposed_area[node] == 0.0 {
            assert!(value.abs() < 1e-9 * scale, "node {node}: residual {value}");
            checked += 1;
        }
    }
    assert!(checked > 0);
    // The residual of the whole mesh sums to zero.
    let total: f64 = residual.iter().sum();
    assert!(total.abs() < 1e-9 * scale);
}

#[test]
fn single_tetrahedron_conduction_matches_the_analytic_element_matrix() {
    let (positions, elements) = one_tetrahedron();
    let model = model_from(positions, elements).expect("valid element");
    let operators = model.operators.as_ref().expect("operators");
    // Right-corner tetrahedron with legs L: V = L³/6; ∇λ₁ = (1,0,0)/L etc.
    let l = 1.0e-3;
    let volume = l * l * l / 6.0;
    let scale = CONDUCTIVITY * volume / (l * l);
    let k = &operators.conduction;
    assert!((k.get(1, 1) - scale).abs() < 1e-12 * scale);
    assert!((k.get(2, 2) - scale).abs() < 1e-12 * scale);
    assert!((k.get(3, 3) - scale).abs() < 1e-12 * scale);
    assert!((k.get(0, 0) - 3.0 * scale).abs() < 1e-12 * scale);
    assert!(k.get(1, 2).abs() < 1e-12 * scale);
    assert!((k.get(0, 1) + scale).abs() < 1e-12 * scale);
    assert!((operators.element_volume[0] - volume).abs() < 1e-24);
    // All four faces are free surface: total area of the right-corner tetrahedron.
    let area: f64 = operators.exposed_area.iter().sum();
    let expected = l * l * (1.5 + 0.5 * 3.0_f64.sqrt());
    assert!((area - expected).abs() < 1e-12 * expected, "area {area}");
}

#[test]
fn gershgorin_bound_dominates_the_true_spectral_radius() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 5.0e-4, 5.0e-4, 5, 2, 2);
    let model = model_on(&mesh, idle_source());
    let operators = model.operators.as_ref().expect("operators");
    let n = model.positions.len();
    // Power iteration on C⁻¹K from a start vector without special symmetry.
    let mut x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
    let mut quotient = 0.0;
    for _ in 0..400 {
        let kx = operators.conduction.mul_vec(&x);
        let numerator: f64 = x.iter().zip(&kx).map(|(a, b)| a * b).sum();
        let denominator: f64 = x
            .iter()
            .zip(&operators.capacity)
            .map(|(a, c)| c * a * a)
            .sum();
        quotient = numerator / denominator;
        x = kx
            .iter()
            .zip(&operators.capacity)
            .map(|(f, c)| f / c)
            .collect();
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        for value in &mut x {
            *value /= norm;
        }
    }
    let bound = 2.0 / model.stable_time_step(0.0).expect("mesh");
    assert!(
        quotient <= bound * (1.0 + 1e-12),
        "Rayleigh quotient {quotient} above the Gershgorin bound {bound}"
    );
    // ...and the bound is not vacuous.
    assert!(quotient > 0.3 * bound, "{quotient} vs {bound}");
}

// ---- stepping ------------------------------------------------------------

#[test]
fn heat_diffuses_into_a_bar_like_the_erfc_solution() {
    // A 100 K temperature step at one end of a long bar.  Before the fix every node
    // was integrated on its own and nothing left the heated end.
    let length = 6.0e-3;
    let width = 1.0e-4;
    let mesh = TetrahedralMesh::generate_beam(length, width, width, 60, 1, 1);
    let mut model = model_on(&mesh, idle_source());
    let step_size = 100.0;
    for node in nodes_at_x(&model, 0.0) {
        model
            .fix_temperature(node, T_AMBIENT + step_size)
            .expect("end node exists");
    }
    let alpha = model.thermal_diffusivity();
    let total_time = 0.5;
    for _ in 0..20 {
        model
            .step([0.0; 3], total_time / 20.0, 0.0)
            .expect("stable step");
    }
    let mut worst = 0.0_f64;
    for (p, &t) in model.positions.iter().zip(&model.temperatures) {
        let exact = T_AMBIENT + step_size * erfc(p[0] / (2.0 * (alpha * total_time).sqrt()));
        worst = worst.max((t - exact).abs());
    }
    assert!(
        worst < 0.03 * step_size,
        "worst deviation {worst} K from the erfc solution of a {step_size} K step"
    );
    // The solution is one-dimensional up to the lumping of the capacity: the nodes
    // of a cross-section agree to within a small fraction of the temperature step.
    for cell in [10, 20, 30] {
        let plane = nodes_at_x(&model, cell as f64 * length / 60.0);
        assert_eq!(plane.len(), 4);
        let temperatures: Vec<f64> = plane.iter().map(|&node| model.temperatures[node]).collect();
        let lowest = temperatures.iter().copied().fold(f64::INFINITY, f64::min);
        let highest = temperatures
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            highest - lowest < 0.005 * step_size,
            "cross-section {cell} spreads over {} K",
            highest - lowest
        );
    }
    // The far end has barely warmed up: erfc(3.4) is of order 1e-6.
    let last = nodes_at_x(&model, length)[0];
    assert!(model.temperatures[last] - T_AMBIENT < 0.1);
    // Heat has really entered the bar.
    let entered = model.stored_energy().expect("mesh");
    assert!(entered > 0.0);
}

#[test]
fn steady_conduction_between_two_fixed_ends_is_linear() {
    let length = 4.0e-3;
    let mesh = TetrahedralMesh::generate_beam(length, 2.0e-4, 2.0e-4, 20, 1, 1);
    let mut model = model_on(&mesh, idle_source());
    for node in nodes_at_x(&model, 0.0) {
        model.fix_temperature(node, 400.0).expect("node");
    }
    for node in nodes_at_x(&model, length) {
        model.fix_temperature(node, 300.0).expect("node");
    }
    for _ in 0..15 {
        model.step([0.0; 3], 1.0, 0.0).expect("stable step");
    }
    for (p, &t) in model.positions.iter().zip(&model.temperatures) {
        let exact = 400.0 - 100.0 * p[0] / length;
        assert!((t - exact).abs() < 1e-4, "x = {}: {t} vs {exact}", p[0]);
    }
}

#[test]
fn a_small_hot_block_cools_like_the_lumped_exponential() {
    // 1 mm cube, Biot number h·L/k ≈ 4e-3: the temperature stays uniform and the
    // block follows T − T∞ = ΔT₀·exp(−h·A·t / (ρ·c_p·V)).
    let side = 1.0e-3;
    let mesh = TetrahedralMesh::generate_beam(side, side, side, 2, 2, 2);
    let mut model = model_on(&mesh, idle_source());
    let initial = 400.0;
    model.temperatures = vec![initial; model.positions.len()];
    model.peak_temperatures = vec![initial; model.positions.len()];
    let h = 50.0;
    let time_constant = DENSITY * SPECIFIC_HEAT * side / (6.0 * h);
    let total_time = 10.0;
    for _ in 0..10 {
        model
            .step([0.0; 3], total_time / 10.0, h)
            .expect("stable step");
    }
    let operators = model.operators.as_ref().expect("operators");
    let total_capacity: f64 = operators.capacity.iter().sum();
    let mean = T_AMBIENT + model.stored_energy().expect("mesh") / total_capacity;
    let exact = T_AMBIENT + (initial - T_AMBIENT) * (-total_time / time_constant).exp();
    assert!(
        (mean - exact).abs() < 0.015 * (initial - T_AMBIENT),
        "mean temperature {mean} K, lumped solution {exact} K"
    );
    // Surface nodes are slightly cooler than the core but within a fraction of a
    // percent of the whole cooling.
    let spread = model
        .temperatures
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &t| {
            (lo.min(t), hi.max(t))
        });
    assert!(spread.1 - spread.0 < 0.01 * (initial - T_AMBIENT));
    // The heat leaves through the surface only: a zero coefficient keeps the energy.
    let mut adiabatic = model_on(&mesh, idle_source());
    adiabatic.temperatures = vec![initial; adiabatic.positions.len()];
    let before = adiabatic.stored_energy().expect("mesh");
    adiabatic.step([0.0; 3], 5.0, 0.0).expect("stable step");
    assert!((adiabatic.stored_energy().expect("mesh") - before).abs() < 1e-12 * before);
}

#[test]
fn a_broad_source_deposits_exactly_its_integrated_power() {
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 4, 2, 2);
    // Semi-axes of 5 cm make the source smooth over the millimetre-sized block; all
    // nodes lie ahead of the source, so only the front ellipsoid acts.
    let source = GoldakSource::new(1.0e5, 1.0, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2);
    let source_pos = [-5.0e-3, 5.0e-4, 5.0e-4];
    let mut model = model_on(&mesh, source.clone());
    let operators = model.operators.clone().expect("operators");
    let load = model.heat_load(&operators, source_pos);
    let total_load: f64 = load.iter().sum();

    // Independent reference: midpoint rule on a fine grid of the whole block.
    let (nx, ny, nz) = (80, 40, 40);
    let (dx, dy, dz) = (2.0e-3 / nx as f64, 1.0e-3 / ny as f64, 1.0e-3 / nz as f64);
    let mut expected = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let point = [
                    (i as f64 + 0.5) * dx,
                    (j as f64 + 0.5) * dy,
                    (k as f64 + 0.5) * dz,
                ];
                expected += source.heat_flux(point, source_pos) * dx * dy * dz;
            }
        }
    }
    assert!(
        (total_load - expected).abs() < 1e-6 * expected,
        "deposited {total_load} W, integrated {expected} W"
    );
    // The load follows the source: nodes closer to it receive more heat per volume.
    let near = load[nodes_at_x(&model, 0.0)[0]] / operators.capacity[nodes_at_x(&model, 0.0)[0]];
    let far =
        load[nodes_at_x(&model, 2.0e-3)[0]] / operators.capacity[nodes_at_x(&model, 2.0e-3)[0]];
    assert!(near > far);

    // A step without losses stores exactly the deposited heat.
    let dt = 0.05;
    model.step(source_pos, dt, 0.0).expect("stable step");
    let stored = model.stored_energy().expect("mesh");
    assert!(
        (stored - total_load * dt).abs() < 1e-9 * stored,
        "stored {stored} J, deposited {} J",
        total_load * dt
    );
    assert!(model.temperatures.iter().all(|&t| t > T_AMBIENT + 1.0));
}

#[test]
fn conduction_carries_source_heat_beyond_the_footprint() {
    // The Goldak footprint (semi-axis 1.5e-4 m) is negligible beyond four semi-axes,
    // yet nodes there must warm up: the heat is conducted to them.
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 20, 4, 4);
    let source = GoldakSource::new(50.0, 0.35, 1.5e-4, 1.0e-4, 1.0e-4, 1.0e-4);
    let mut model = model_on(&mesh, source);
    let centre_y = 5.0e-4;
    let centre_z = 5.0e-4;
    let source_pos = [5.0e-4, centre_y, centre_z];
    // The nodes on the axis of the block through the source.
    let line: Vec<usize> = model
        .positions
        .iter()
        .enumerate()
        .filter(|&(_, p)| (p[1] - centre_y).abs() < 1e-9 && (p[2] - centre_z).abs() < 1e-9)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(line.len(), 21);
    for _ in 0..10 {
        model.step(source_pos, 5.0e-3, 0.0).expect("stable step");
    }
    let rise_at = |x: f64| -> f64 {
        line.iter()
            .filter(|&&i| (model.positions[i][0] - x).abs() < 1e-9)
            .map(|&i| model.temperatures[i] - T_AMBIENT)
            .fold(0.0_f64, f64::max)
    };
    let near = rise_at(5.0e-4);
    let mid = rise_at(8.0e-4);
    let far = rise_at(1.1e-3);
    // Beyond four semi-axes of the source (x - x_source = 6e-4 m): conduction only.
    let q_there = model
        .source
        .heat_flux([1.1e-3, centre_y, centre_z], source_pos);
    assert!(q_there < 1e-12 * model.source.heat_flux(source_pos, source_pos));
    assert!(far > 1.0, "the far node rose by only {far} K");
    assert!(near > mid && mid > far, "{near} > {mid} > {far}");
    // Away from the heated region the bar is still cold.
    let tip = rise_at(2.0e-3);
    assert!(tip < far);
}

#[test]
fn conduction_conserves_the_energy_of_a_localised_source() {
    let mesh = TetrahedralMesh::generate_beam(2.0e-3, 1.0e-3, 1.0e-3, 8, 4, 4);
    let source = GoldakSource::new(200.0, 0.35, 3.0e-4, 2.0e-4, 2.0e-4, 3.0e-4);
    let source_pos = [8.0e-4, 5.0e-4, 1.0e-3];
    let mut model = model_on(&mesh, source);
    let operators = model.operators.clone().expect("operators");
    let deposited: f64 = model.heat_load(&operators, source_pos).iter().sum();
    assert!(deposited > 0.0);
    let dt = 2.0e-3;
    model.step(source_pos, dt, 0.0).expect("stable step");
    // Conduction only moves heat around: the stored energy is the deposited energy.
    let stored = model.stored_energy().expect("mesh");
    assert!(
        (stored - deposited * dt).abs() < 1e-9 * stored,
        "stored {stored} J, deposited {} J",
        deposited * dt
    );
}

#[test]
fn large_time_steps_are_divided_into_stable_sub_steps() {
    let mesh = TetrahedralMesh::generate_beam(3.0e-3, 1.0e-4, 1.0e-4, 30, 1, 1);
    let mut model = model_on(&mesh, idle_source());
    for node in nodes_at_x(&model, 0.0) {
        model.fix_temperature(node, 500.0).expect("node");
    }
    let limit = model.stable_time_step(0.0).expect("mesh");
    assert!(limit.is_finite() && limit > 0.0);
    // One huge step: 1000 times the stability limit of a single explicit step.
    model
        .step([0.0; 3], 1000.0 * limit, 0.0)
        .expect("sub-stepped");
    let mut previous = f64::INFINITY;
    for cell in 0..=30 {
        let node = nodes_at_x(&model, cell as f64 * 1.0e-4)[0];
        let t = model.temperatures[node];
        assert!(t.is_finite());
        assert!(
            (T_AMBIENT - 1e-9..=500.0 + 1e-9).contains(&t),
            "temperature {t} left the range of the data"
        );
        assert!(
            t <= previous + 1e-9,
            "profile must decrease from the hot end"
        );
        previous = t;
    }
    // Strictly more than a single unstable explicit step would have produced.
    let warmed = model
        .temperatures
        .iter()
        .filter(|&&t| t > T_AMBIENT + 1.0)
        .count();
    assert!(warmed > 8, "only {warmed} nodes warmed up");
}

#[test]
fn convection_enters_the_stability_limit() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-3, 1.0e-3, 2, 2, 2);
    let model = model_on(&mesh, idle_source());
    let dry = model.stable_time_step(0.0).expect("mesh");
    let cooled = model.stable_time_step(1.0e7).expect("mesh");
    assert!(cooled < dry, "{cooled} should be below {dry}");
}

#[test]
fn peak_temperatures_survive_cooling() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-3, 1.0e-3, 2, 2, 2);
    let mut model = model_on(&mesh, idle_source());
    model.temperatures = vec![900.0; model.positions.len()];
    model.peak_temperatures = vec![900.0; model.positions.len()];
    for _ in 0..5 {
        model.step([0.0; 3], 2.0, 2.0e3).expect("stable step");
    }
    assert!(model.temperatures.iter().all(|&t| t < 900.0));
    assert!(model.peak_temperatures.iter().all(|&peak| peak == 900.0));
    assert!(
        model
            .temperatures
            .iter()
            .zip(&model.peak_temperatures)
            .all(|(t, peak)| t <= peak)
    );
}

#[test]
fn peak_temperatures_record_excursions_inside_a_single_step() {
    // A hot fixed end heats its neighbours; the peak must be at least the final value
    // at every node, and the fixed node itself records its prescribed value.
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-4, 1.0e-4, 10, 1, 1);
    let mut model = model_on(&mesh, idle_source());
    for node in nodes_at_x(&model, 0.0) {
        model.fix_temperature(node, 700.0).expect("node");
    }
    model.step([0.0; 3], 0.05, 0.0).expect("stable step");
    for (t, peak) in model.temperatures.iter().zip(&model.peak_temperatures) {
        assert!(peak >= t);
    }
    for node in nodes_at_x(&model, 0.0) {
        assert_eq!(model.temperatures[node], 700.0);
        assert_eq!(model.peak_temperatures[node], 700.0);
    }
}

#[test]
fn nodes_outside_every_element_keep_their_temperature() {
    let (mut positions, elements) = one_tetrahedron();
    positions.push([5.0e-3, 5.0e-3, 5.0e-3]);
    let mut model = model_from(positions, elements).expect("valid");
    model.temperatures[4] = 777.0;
    model.step([0.0; 3], 1.0e-3, 10.0).expect("stable step");
    assert_eq!(model.temperatures[4], 777.0);
    assert!(model.stable_time_step(10.0).expect("mesh").is_finite());
}

#[test]
fn refreshing_the_operators_applies_a_new_material() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-3, 1.0e-3, 2, 2, 2);
    let mut model = model_on(&mesh, idle_source());
    let before = model.stable_time_step(0.0).expect("mesh");
    model.conductivity *= 2.0;
    model.refresh_operators().expect("valid material");
    let after = model.stable_time_step(0.0).expect("mesh");
    assert!((after - 0.5 * before).abs() < 1e-12 * before);
    // An invalid material is rejected and the previous operators are kept.
    model.conductivity = -1.0;
    assert!(matches!(
        model.refresh_operators(),
        Err(Error::InvalidInput(_))
    ));
    assert!((model.stable_time_step(0.0).expect("mesh") - after).abs() < 1e-18);
}

// ---- error handling ------------------------------------------------------

#[test]
fn a_model_without_a_mesh_cannot_be_stepped() {
    let mut model = ThermalHistoryFem::new(
        vec![[0.0; 3], [1.0e-3, 0.0, 0.0]],
        idle_source(),
        1923.0,
        1723.0,
        CONDUCTIVITY,
        SPECIFIC_HEAT,
        DENSITY,
    );
    assert!(model.stable_time_step(0.0).is_none());
    assert!(model.stored_energy().is_none());
    let result = model.step([0.0; 3], 1.0e-3, 10.0);
    assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");
    assert_eq!(model.temperatures, vec![T_AMBIENT; 2]);
}

#[test]
fn invalid_meshes_and_materials_are_rejected() {
    let (positions, elements) = one_tetrahedron();

    // An element that references a missing node.
    let result = model_from(positions.clone(), vec![[0, 1, 2, 9]]);
    assert!(matches!(result, Err(Error::InvalidInput(_))));

    // A flat element: the four corners are coplanar.
    let flat = vec![
        [0.0, 0.0, 0.0],
        [1.0e-3, 0.0, 0.0],
        [0.0, 1.0e-3, 0.0],
        [1.0e-3, 1.0e-3, 0.0],
    ];
    assert!(matches!(
        model_from(flat, vec![[0, 1, 2, 3]]),
        Err(Error::InvalidInput(_))
    ));

    // A non-finite position.
    let mut broken = positions.clone();
    broken[2][1] = f64::NAN;
    assert!(matches!(
        model_from(broken, elements.clone()),
        Err(Error::InvalidInput(_))
    ));

    // Three elements around one face.
    let mut stacked = positions.clone();
    stacked.push([1.0e-3, 1.0e-3, 1.0e-3]);
    stacked.push([-1.0e-3, -1.0e-3, -1.0e-3]);
    stacked.push([2.0e-3, 2.0e-3, 2.0e-3]);
    let result = model_from(
        stacked,
        vec![[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 2, 6]],
    );
    assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");

    // Non-positive or non-finite material data.
    for (k, c, rho) in [
        (0.0, SPECIFIC_HEAT, DENSITY),
        (CONDUCTIVITY, -1.0, DENSITY),
        (CONDUCTIVITY, SPECIFIC_HEAT, f64::NAN),
        (f64::INFINITY, SPECIFIC_HEAT, DENSITY),
    ] {
        let result = ThermalHistoryFem::with_mesh(
            positions.clone(),
            elements.clone(),
            idle_source(),
            1923.0,
            1723.0,
            k,
            c,
            rho,
        );
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "{k} {c} {rho}"
        );
    }
}

#[test]
fn invalid_step_arguments_are_rejected_and_leave_the_state_alone() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-3, 1.0e-3, 2, 2, 2);
    let mut model = model_on(
        &mesh,
        GoldakSource::new(100.0, 0.5, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4),
    );
    let before = model.temperatures.clone();
    let origin = [0.0; 3];
    for (dt, h) in [
        (0.0, 0.0),
        (-1.0, 0.0),
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0e-3, -1.0),
        (1.0e-3, f64::NAN),
    ] {
        let result = model.step(origin, dt, h);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "dt {dt}, h {h}"
        );
    }
    let result = model.step([f64::NAN, 0.0, 0.0], 1.0e-3, 0.0);
    assert!(matches!(result, Err(Error::InvalidInput(_))));
    assert_eq!(model.temperatures, before);

    // A Goldak source with a zero semi-axis has a non-finite load.
    model.source.a = 0.0;
    let result = model.step(origin, 1.0e-3, 0.0);
    assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");
    assert_eq!(model.temperatures, before);

    // A step that would need more than a million sub-steps.
    model.source.a = 1.0e-4;
    let limit = model.stable_time_step(0.0).expect("mesh");
    let result = model.step(origin, 1.0e8 * limit, 0.0);
    assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");
    assert_eq!(model.temperatures, before);

    // The state vectors must stay in step with the operators.
    model.temperatures.pop();
    assert!(matches!(
        model.step(origin, 1.0e-3, 0.0),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn fixing_a_temperature_validates_and_applies_immediately() {
    let mesh = TetrahedralMesh::generate_beam(1.0e-3, 1.0e-3, 1.0e-3, 2, 2, 2);
    let mut model = model_on(&mesh, idle_source());
    let n = model.positions.len();
    assert!(matches!(
        model.fix_temperature(n, 500.0),
        Err(Error::DofOutOfRange { .. })
    ));
    assert!(matches!(
        model.fix_temperature(0, f64::NAN),
        Err(Error::InvalidInput(_))
    ));
    model.fix_temperature(0, 650.0).expect("existing node");
    assert_eq!(model.temperatures[0], 650.0);
    assert_eq!(model.peak_temperatures[0], 650.0);
    // Cooling a held node below the peak keeps the peak.
    model.fix_temperature(0, 400.0).expect("existing node");
    assert_eq!(model.temperatures[0], 400.0);
    assert_eq!(model.peak_temperatures[0], 650.0);
}

#[test]
fn melt_pool_bookkeeping_is_unchanged_and_safe_for_empty_models() {
    let empty = ThermalHistoryFem::new(
        Vec::new(),
        idle_source(),
        1923.0,
        1723.0,
        12.0,
        620.0,
        8000.0,
    );
    assert_eq!(empty.mushy_zone_fraction(), 0.0);
    assert!(empty.melt_pool_nodes().is_empty());

    let mut model = ThermalHistoryFem::new(
        vec![[0.0; 3]; 4],
        idle_source(),
        1923.0,
        1723.0,
        12.0,
        620.0,
        8000.0,
    );
    model.temperatures = vec![300.0, 1800.0, 1950.0, 2100.0];
    assert_eq!(model.melt_pool_nodes(), vec![2, 3]);
    assert!((model.mushy_zone_fraction() - 0.75).abs() < 1e-15);
}
