//! Regression tests for the finite-difference forces of the Behler-Parrinello
//! neural-network potential and for the symmetric kernel-matrix build.
//!
//! The reference for every force is the gradient of
//! [`BpNeuralNetworkPotential::total_energy`] obtained by displacing one
//! coordinate of one atom at a time, which shares no code with the production
//! force routine (that one re-evaluates only the atomic energies that depend on
//! the moved atom).

use super::*;

/// A network with radial and angular inputs and a tanh hidden layer.
fn potential() -> BpNeuralNetworkPotential {
    let mut descriptor = BehlerParrinelloDescriptor::new(4.0);
    descriptor.add_g2(0.4, 0.0);
    descriptor.add_g2(1.0, 1.5);
    descriptor.add_g4(0.2, 2.0, 1.0);
    descriptor.add_g4(0.1, 1.0, -1.0);
    let hidden = NeuralNetworkLayer::new(
        vec![
            vec![0.7, -0.3, 0.2, 0.1],
            vec![-0.4, 0.5, 0.1, -0.2],
            vec![0.2, 0.2, -0.6, 0.3],
        ],
        vec![0.1, -0.05, 0.0],
        tanh_activation,
    );
    let output =
        NeuralNetworkLayer::new(vec![vec![0.5, -0.7, 0.9]], vec![0.0], identity_activation);
    BpNeuralNetworkPotential::new(vec![hidden, output], descriptor)
}

/// A distorted tetrahedron with every pair distance well inside the 4 Å cutoff.
fn tetrahedron() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [2.1, 0.1, -0.05],
        [1.0, 1.9, 0.12],
        [1.1, 0.7, 1.8],
    ]
}

/// Symmetric lists: every other atom is a neighbour.
fn full_lists(n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| (0..n).filter(|&j| j != i).collect())
        .collect()
}

/// Half lists: atom `i` lists only the atoms `j > i`.
fn half_lists(n: usize) -> Vec<Vec<usize>> {
    (0..n).map(|i| ((i + 1)..n).collect()).collect()
}

/// `-dE_total/dr` from the total energy alone, one displaced coordinate at a time.
fn reference_forces(
    pot: &BpNeuralNetworkPotential,
    positions: &[[f64; 3]],
    lists: &[Vec<usize>],
    h: f64,
) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0; 3]; positions.len()];
    for i in 0..positions.len() {
        for axis in 0..3 {
            let mut plus = positions.to_vec();
            plus[i][axis] += h;
            let mut minus = positions.to_vec();
            minus[i][axis] -= h;
            let e_plus = pot.total_energy(&plus, lists).unwrap();
            let e_minus = pot.total_energy(&minus, lists).unwrap();
            forces[i][axis] = -(e_plus - e_minus) / (2.0 * h);
        }
    }
    forces
}

fn max_abs_difference(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(fa, fb)| fa.iter().zip(fb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn forces_are_the_negative_gradient_of_the_total_energy() {
    let pot = potential();
    let positions = tetrahedron();
    let lists = full_lists(positions.len());
    let forces = pot
        .total_forces_by_finite_diff(&positions, &lists, 1.0e-5)
        .unwrap();
    let reference = reference_forces(&pot, &positions, &lists, 1.0e-4);
    let error = max_abs_difference(&forces, &reference);
    assert!(error < 1.0e-6, "max |F - (-dE/dr)| = {error:e}");
    let scale = forces.iter().flatten().fold(0.0_f64, |m, f| m.max(f.abs()));
    assert!(scale > 1.0e-3, "the test needs non-trivial forces: {scale}");
}

#[test]
fn half_neighbour_lists_use_the_reverse_dependencies() {
    let pot = potential();
    let positions = tetrahedron();
    let lists = half_lists(positions.len());
    let forces = pot
        .total_forces_by_finite_diff(&positions, &lists, 1.0e-5)
        .unwrap();
    let reference = reference_forces(&pot, &positions, &lists, 1.0e-4);
    let error = max_abs_difference(&forces, &reference);
    assert!(error < 1.0e-6, "max |F - (-dE/dr)| = {error:e}");
    // The last atom has an empty neighbour list, so its own atomic energy is a
    // constant, yet the atoms that list it feel it move: the force must not vanish.
    let last = forces[3];
    let magnitude = (last[0] * last[0] + last[1] * last[1] + last[2] * last[2]).sqrt();
    assert!(
        magnitude > 1.0e-3,
        "force on the unlisted atom vanished: {last:?}"
    );
    // The same configuration with full lists is a different model and a different force.
    let full = pot
        .total_forces_by_finite_diff(&positions, &full_lists(4), 1.0e-5)
        .unwrap();
    assert!(max_abs_difference(&forces, &full) > 1.0e-3);
}

#[test]
fn forces_sum_to_zero_and_exert_no_net_torque() {
    let pot = potential();
    let positions = tetrahedron();
    let lists = full_lists(positions.len());
    let forces = pot
        .total_forces_by_finite_diff(&positions, &lists, 1.0e-5)
        .unwrap();
    for axis in 0..3 {
        let total: f64 = forces.iter().map(|f| f[axis]).sum();
        assert!(
            total.abs() < 1.0e-7,
            "net force component {axis} = {total:e}"
        );
    }
    let mut torque = [0.0; 3];
    for (r, f) in positions.iter().zip(&forces) {
        torque[0] += r[1] * f[2] - r[2] * f[1];
        torque[1] += r[2] * f[0] - r[0] * f[2];
        torque[2] += r[0] * f[1] - r[1] * f[0];
    }
    for (axis, t) in torque.iter().enumerate() {
        assert!(t.abs() < 1.0e-6, "net torque component {axis} = {t:e}");
    }
}

#[test]
fn dimer_forces_are_equal_and_opposite_along_the_bond() {
    let pot = potential();
    let positions = [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]];
    let lists = full_lists(2);
    let forces = pot
        .total_forces_by_finite_diff(&positions, &lists, 1.0e-5)
        .unwrap();
    assert!(forces[0][0].abs() > 1.0e-3, "axial force {:?}", forces[0]);
    for axis in 0..3 {
        assert!(
            (forces[0][axis] + forces[1][axis]).abs() < 1.0e-9,
            "F_0 + F_1 along {axis}: {:?} {:?}",
            forces[0],
            forces[1]
        );
    }
    for force in &forces {
        assert!(
            force[1].abs() < 1.0e-9 && force[2].abs() < 1.0e-9,
            "{force:?}"
        );
    }
    // The old self term alone is only half of the pair force (the partner's atomic
    // energy depends on the distance as well).
    let self_term = pot
        .atomic_energy_gradient(&positions, 0, &lists[0], 1.0e-5)
        .unwrap();
    assert!(
        (-self_term[0] - forces[0][0]).abs() > 1.0e-3,
        "self term {self_term:?} vs force {:?}",
        forces[0]
    );
}

#[test]
fn atomic_energy_gradient_is_the_self_term_of_the_force() {
    let pot = potential();
    let positions = tetrahedron();
    let lists = full_lists(positions.len());
    let gradient = pot
        .atomic_energy_gradient(&positions, 0, &lists[0], 1.0e-5)
        .unwrap();
    // Independent central difference of the atomic energy of atom 0 alone.
    for axis in 0..3 {
        let h = 1.0e-4;
        let mut plus = positions.clone();
        plus[0][axis] += h;
        let mut minus = positions.clone();
        minus[0][axis] -= h;
        let e = |p: &[[f64; 3]]| pot.energy(&pot.descriptor.compute(p, 0, &lists[0]));
        let expected = (e(&plus) - e(&minus)) / (2.0 * h);
        assert!(
            (gradient[axis] - expected).abs() < 1.0e-7,
            "axis {axis}: {} vs {expected}",
            gradient[axis]
        );
    }
    // It is a gradient, not a force: only part of -dE_total/dr_0.
    let forces = pot
        .total_forces_by_finite_diff(&positions, &lists, 1.0e-5)
        .unwrap();
    let missing: f64 = (0..3)
        .map(|axis| (-gradient[axis] - forces[0][axis]).abs())
        .sum();
    assert!(missing > 1.0e-3, "self term equals the force: {missing:e}");
}

#[test]
fn total_energy_is_the_sum_of_atomic_energies_and_rigid_motion_invariant() {
    let pot = potential();
    let positions = tetrahedron();
    let lists = full_lists(positions.len());
    let total = pot.total_energy(&positions, &lists).unwrap();
    let summed: f64 = (0..positions.len())
        .map(|j| pot.energy(&pot.descriptor.compute(&positions, j, &lists[j])))
        .sum();
    assert!((total - summed).abs() < 1.0e-14, "{total} vs {summed}");
    let shifted: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| [p[0] + 3.0, p[1] - 2.0, p[2] + 5.0])
        .collect();
    let translated = pot.total_energy(&shifted, &lists).unwrap();
    assert!(
        (total - translated).abs() < 1.0e-12,
        "{total} vs {translated}"
    );
    let (sin, cos) = (PI / 6.0).sin_cos();
    let rotated: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1], p[2]])
        .collect();
    let turned = pot.total_energy(&rotated, &lists).unwrap();
    assert!((total - turned).abs() < 1.0e-12, "{total} vs {turned}");
}

#[test]
fn invalid_configurations_are_reported_instead_of_indexing_out_of_range() {
    let pot = potential();
    let positions = tetrahedron();
    // One list per atom is required.
    let short = full_lists(3);
    assert!(matches!(
        pot.total_forces_by_finite_diff(&positions, &short, 1.0e-5),
        Err(MdError::DimensionMismatch {
            expected: 4,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        pot.total_energy(&positions, &short),
        Err(MdError::DimensionMismatch {
            expected: 4,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        pot.total_energy(&positions, &[]),
        Err(MdError::DimensionMismatch {
            expected: 4,
            got: 0,
            ..
        })
    ));
    // Neighbour index out of range.
    let mut lists = full_lists(4);
    lists[2].push(9);
    assert!(matches!(
        pot.total_forces_by_finite_diff(&positions, &lists, 1.0e-5),
        Err(MdError::General(_))
    ));
    assert!(matches!(
        pot.atomic_energy_gradient(&positions, 0, &[1, 9], 1.0e-5),
        Err(MdError::General(_))
    ));
    assert!(matches!(
        pot.atomic_energy_gradient(&positions, 7, &[1], 1.0e-5),
        Err(MdError::General(_))
    ));
    // Non-finite coordinates and bad steps.
    let mut broken = tetrahedron();
    broken[1][2] = f64::NAN;
    let lists = full_lists(4);
    assert!(pot.total_energy(&broken, &lists).is_err());
    assert!(
        pot.total_forces_by_finite_diff(&broken, &lists, 1.0e-5)
            .is_err()
    );
    for dx in [0.0, -1.0e-5, f64::NAN, f64::INFINITY] {
        assert!(
            pot.total_forces_by_finite_diff(&positions, &lists, dx)
                .is_err(),
            "dx = {dx}"
        );
        assert!(
            pot.atomic_energy_gradient(&positions, 0, &lists[0], dx)
                .is_err(),
            "dx = {dx}"
        );
    }
    // An empty system has no forces.
    assert!(
        pot.total_forces_by_finite_diff(&[], &[], 1.0e-5)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn kernel_matrix_is_exactly_symmetric_and_matches_direct_evaluation() {
    let kernel = GapKernel::new(1.7, 0.9);
    let points = vec![
        vec![0.0, 0.5],
        vec![1.0, -0.5],
        vec![2.0, 1.5],
        vec![0.3, 0.3],
        vec![0.0, 0.5],
    ];
    let k = kernel.kernel_matrix(&points);
    assert_eq!(k.len(), points.len());
    for i in 0..points.len() {
        for j in 0..points.len() {
            let direct = kernel.evaluate(&points[i], &points[j]);
            assert_eq!(k[i][j].to_bits(), direct.to_bits(), "entry ({i}, {j})");
            assert_eq!(k[i][j].to_bits(), k[j][i].to_bits(), "symmetry ({i}, {j})");
        }
        assert!((k[i][i] - 1.7).abs() < 1.0e-15);
    }
    // Two identical descriptors give identical rows.
    assert_eq!(k[0], k[4]);
    assert!(kernel.kernel_matrix(&[]).is_empty());
}
