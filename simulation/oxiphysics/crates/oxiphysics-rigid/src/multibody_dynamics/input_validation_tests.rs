// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the argument validation of the dynamics queries and for
//! the error that replaces the old "all accelerations are zero" answer of a
//! singular mass matrix.
//!
//! Before the fix, `forward_dynamics`, `inverse_dynamics` and `jacobian` indexed
//! the caller's slices and the link table directly (a panic on a short slice or
//! an out-of-range link), and `solve_linear` returned the zero vector for a
//! singular system, so a link without inertia about its joint axis silently got
//! `q̈ = 0`.

use super::*;

const GRAVITY: [f64; 3] = [0.0, -9.81, 0.0];

/// A serial chain of `n` links built with [`Link::new`]: no inertia tensor, the
/// centre of mass `com` in every link and the joint axis `+z`.
fn point_mass_chain(n: usize, mass: f64, com: [f64; 3]) -> MultibodyChain {
    let links = (0..n).map(|_| Link::new(mass, com)).collect();
    let parent = (0..n)
        .map(|i| if i == 0 { None } else { Some(i - 1) })
        .collect();
    MultibodyChain::new(links, parent)
}

// ── short slices ────────────────────────────────────────────────────────

#[test]
fn forward_dynamics_rejects_torque_slices_of_the_wrong_length() {
    let chain = build_planar_arm(3, 1.0, 1.0);
    for (torques, got) in [(vec![], 0), (vec![0.0; 2], 2), (vec![0.0; 4], 4)] {
        assert_eq!(
            chain.forward_dynamics(&torques, GRAVITY),
            Err(MultibodyError::DimensionMismatch { expected: 3, got }),
            "{got} torques for 3 joints"
        );
    }
    assert!(chain.forward_dynamics(&[0.0; 3], GRAVITY).is_ok());
}

#[test]
fn inverse_dynamics_rejects_acceleration_slices_of_the_wrong_length() {
    let chain = build_planar_arm(3, 1.0, 1.0);
    for (q_ddot, got) in [(vec![], 0), (vec![1.0; 2], 2), (vec![1.0; 5], 5)] {
        assert_eq!(
            chain.inverse_dynamics(&q_ddot, GRAVITY),
            Err(MultibodyError::DimensionMismatch { expected: 3, got }),
            "{got} accelerations for 3 joints"
        );
    }
    assert!(chain.inverse_dynamics(&[1.0; 3], GRAVITY).is_ok());
}

#[test]
fn step_rk4_rejects_a_short_torque_slice_and_keeps_the_state() {
    let mut chain = build_planar_arm(2, 1.0, 1.0);
    chain.joint_angles = vec![0.3, -0.2];
    chain.joint_velocities = vec![0.1, 0.4];
    let before = chain.clone();
    assert_eq!(
        chain.step_rk4(&[1.0], GRAVITY, 1e-3),
        Err(MultibodyError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(chain.joint_angles, before.joint_angles);
    assert_eq!(chain.joint_velocities, before.joint_velocities);
}

// ── link indices ────────────────────────────────────────────────────────

#[test]
fn jacobian_rejects_a_link_index_past_the_end() {
    let chain = build_planar_arm(3, 1.0, 1.0);
    assert_eq!(chain.jacobian(3), Err(MultibodyError::LinkOutOfRange(3)));
    assert_eq!(
        chain.jacobian(usize::MAX),
        Err(MultibodyError::LinkOutOfRange(usize::MAX))
    );
    assert!(chain.jacobian(2).is_ok());
}

#[test]
fn an_empty_chain_has_no_links_to_query() {
    let chain = MultibodyChain::new(Vec::new(), Vec::new());
    assert_eq!(chain.n_dof(), 0);
    assert_eq!(chain.jacobian(0), Err(MultibodyError::LinkOutOfRange(0)));
    assert_eq!(chain.forward_dynamics(&[], GRAVITY), Ok(Vec::new()));
    assert_eq!(chain.inverse_dynamics(&[], GRAVITY), Ok(Vec::new()));
}

#[test]
fn jacobian_of_the_last_link_of_a_planar_arm_is_exact() {
    // At q = 0 the arm points along +y: joint 1 sits at (0, 1, 0), which is also
    // the reference point of link 1, so joint 1 does not move it and joint 0
    // swings it along -x (z × (0, 1, 0)).
    let chain = build_planar_arm(2, 1.0, 1.0);
    let jacobian = chain.jacobian(1).unwrap();
    assert_eq!(jacobian.len(), 2);
    assert_eq!(jacobian[0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0]);
    assert_eq!(jacobian[1], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
}

// ── singular mass matrix ────────────────────────────────────────────────

#[test]
fn a_link_with_its_mass_on_the_joint_axis_has_undetermined_accelerations() {
    // A point mass on the axis has no inertia about it: M = 0.
    let chain = point_mass_chain(1, 2.0, [0.0, 0.0, 0.5]);
    assert_eq!(chain.mass_matrix(), vec![vec![0.0]]);
    assert_eq!(
        chain.forward_dynamics(&[1.0], GRAVITY),
        Err(MultibodyError::SingularMassMatrix)
    );
}

#[test]
fn one_inertialess_joint_makes_the_whole_chain_singular() {
    // Link 0 is a normal rod, link 1 is a massless link: M = diag(m/3 + ..., 0).
    let rod = build_planar_arm(1, 1.0, 1.0).links.remove(0);
    let links = vec![rod, Link::new(0.0, [0.0; 3])];
    let chain = MultibodyChain::new(links, vec![None, Some(0)]);
    let mass = chain.mass_matrix();
    assert!(mass[0][0] > 0.0);
    assert_eq!(mass[1][1], 0.0);
    assert_eq!(
        chain.forward_dynamics(&[0.0, 0.0], GRAVITY),
        Err(MultibodyError::SingularMassMatrix)
    );
}

#[test]
fn step_rk4_propagates_the_singular_mass_matrix_and_keeps_the_state() {
    let mut chain = point_mass_chain(1, 2.0, [0.0, 0.0, 0.5]);
    chain.joint_angles = vec![0.7];
    chain.joint_velocities = vec![1.5];
    assert_eq!(
        chain.step_rk4(&[0.5], GRAVITY, 1e-3),
        Err(MultibodyError::SingularMassMatrix)
    );
    assert_eq!(chain.joint_angles, vec![0.7]);
    assert_eq!(chain.joint_velocities, vec![1.5]);
}

#[test]
fn a_chain_of_very_light_links_is_not_mistaken_for_a_singular_one() {
    // The pivot threshold is relative to the largest entry of M.  The mass
    // matrix of this rod is m/3 = 3.3e-21, far below the old absolute 1e-14
    // cut-off, but perfectly regular.
    let mass = 1e-20;
    let chain = build_planar_arm(1, 1.0, mass);
    let qdd = chain.forward_dynamics(&[mass], [0.0; 3]).unwrap();
    assert!(
        (qdd[0] - 3.0).abs() < 1e-9,
        "q'' = {} expected tau / (m/3) = 3",
        qdd[0]
    );
}

#[test]
fn a_non_finite_configuration_is_reported_as_singular() {
    let mut chain = build_planar_arm(2, 1.0, 1.0);
    chain.joint_angles[0] = f64::NAN;
    assert_eq!(
        chain.forward_dynamics(&[0.0, 0.0], GRAVITY),
        Err(MultibodyError::SingularMassMatrix)
    );
    chain.joint_angles[0] = 0.0;
    chain.joint_velocities[1] = f64::INFINITY;
    assert_eq!(
        chain.forward_dynamics(&[0.0, 0.0], GRAVITY),
        Err(MultibodyError::SingularMassMatrix)
    );
}

// ── inconsistent chain tables ───────────────────────────────────────────

#[test]
fn per_link_tables_that_fell_out_of_step_are_reported_not_indexed() {
    let mut short_angles = build_planar_arm(3, 1.0, 1.0);
    short_angles.joint_angles.truncate(2);
    let mut long_velocities = build_planar_arm(3, 1.0, 1.0);
    long_velocities.joint_velocities.push(0.0);
    let mut short_parents = build_planar_arm(3, 1.0, 1.0);
    short_parents.parent.pop();

    for (chain, got) in [(short_angles, 2), (long_velocities, 4), (short_parents, 2)] {
        let error = MultibodyError::DimensionMismatch { expected: 3, got };
        assert_eq!(chain.validate(), Err(error.clone()));
        assert_eq!(
            chain.forward_dynamics(&[0.0; 3], GRAVITY),
            Err(error.clone())
        );
        assert_eq!(
            chain.inverse_dynamics(&[0.0; 3], GRAVITY),
            Err(error.clone())
        );
        assert_eq!(chain.jacobian(0), Err(error));
    }
}

#[test]
fn a_parent_must_be_stored_before_its_child() {
    let bad_parents = [
        (0, Some(0)), // a root that is its own parent: an endless ancestor walk
        (1, Some(2)), // a parent stored after its child
        (2, Some(7)), // a parent that does not exist
    ];
    for (link, parent) in bad_parents {
        let mut chain = build_planar_arm(3, 1.0, 1.0);
        chain.parent[link] = parent;
        let expected = MultibodyError::InvalidParent {
            link,
            parent: parent.unwrap_or_default(),
        };
        assert_eq!(chain.validate(), Err(expected.clone()));
        assert_eq!(chain.forward_dynamics(&[0.0; 3], GRAVITY), Err(expected));
    }
    assert_eq!(build_planar_arm(3, 1.0, 1.0).validate(), Ok(()));
}

// ── solve_linear ────────────────────────────────────────────────────────

#[test]
fn solve_linear_reports_singular_systems() {
    // Zero matrix, rank-deficient matrix and a zero column.
    for a in [
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![vec![1.0, 2.0], vec![2.0, 4.0]],
        vec![vec![1.0, 0.0], vec![3.0, 0.0]],
    ] {
        assert_eq!(solve_linear(a.clone(), vec![1.0, 2.0]), None, "{a:?}");
    }
}

#[test]
fn solve_linear_tolerance_is_relative_to_the_matrix() {
    // Uniformly tiny but regular: x = (1, 2).
    let tiny = vec![vec![1e-20, 0.0], vec![0.0, 2e-20]];
    let x = solve_linear(tiny, vec![1e-20, 4e-20]).unwrap();
    assert!(
        (x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12,
        "{x:?}"
    );
    // A pivot far below the largest entry is round-off, not data.
    let ill = vec![vec![1.0e6, 0.0], vec![0.0, 1.0e-9]];
    assert_eq!(solve_linear(ill, vec![1.0, 1.0]), None);
    // A badly scaled but regular system is still solved.
    let scaled = vec![vec![1.0e6, 0.0], vec![0.0, 1.0e-6]];
    let x = solve_linear(scaled, vec![1.0e6, 1.0e-6]).unwrap();
    assert!(
        (x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12,
        "{x:?}"
    );
}

#[test]
fn solve_linear_rejects_non_finite_and_malformed_input() {
    let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_eq!(solve_linear(identity.clone(), vec![f64::NAN, 1.0]), None);
    assert_eq!(
        solve_linear(identity.clone(), vec![1.0, f64::INFINITY]),
        None
    );
    assert_eq!(
        solve_linear(vec![vec![f64::NAN, 0.0], vec![0.0, 1.0]], vec![1.0, 1.0]),
        None
    );
    assert_eq!(
        solve_linear(
            vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]],
            vec![1.0, 1.0]
        ),
        None
    );
    // Not square / not matching the right-hand side.
    assert_eq!(solve_linear(vec![vec![1.0, 0.0]], vec![1.0, 1.0]), None);
    assert_eq!(solve_linear(identity, vec![1.0, 1.0, 1.0]), None);
    // The empty system has the empty solution.
    assert_eq!(solve_linear(Vec::new(), Vec::new()), Some(Vec::new()));
}

#[test]
fn solve_linear_pivots_on_the_largest_entry() {
    // The first pivot is zero: only the row swap makes the system solvable.
    let a = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
    let x = solve_linear(a, vec![1.0, 8.0]).unwrap();
    assert!(
        (x[0] - 2.5).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12,
        "{x:?}"
    );
}

// ── error type ──────────────────────────────────────────────────────────

#[test]
fn the_error_messages_name_the_offending_values() {
    let text = |error: MultibodyError| error.to_string();
    assert_eq!(
        text(MultibodyError::DimensionMismatch {
            expected: 3,
            got: 2
        }),
        "expected 3 entries (one per joint), got 2"
    );
    assert_eq!(
        text(MultibodyError::LinkOutOfRange(7)),
        "link index 7 is out of range"
    );
    assert_eq!(
        text(MultibodyError::InvalidParent { link: 1, parent: 4 }),
        "link 1 has parent 4, which is not stored before it"
    );
    let singular = text(MultibodyError::SingularMassMatrix);
    assert!(singular.contains("undetermined") && singular.contains("singular"));
}
