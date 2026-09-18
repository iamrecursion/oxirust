// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the sparse EFG assembly and solution (F1234): the sparse path must give
//! the dense path's results, and never return an unconverged iterate.

use super::functions::Point2;
use super::*;

const KERNEL: KernelType = KernelType::QuarticSpline;

fn material() -> EfgMaterial {
    EfgMaterial {
        young: 1.0e3,
        poisson: 0.3,
        thickness: 0.5,
    }
}

fn cloud(n: usize) -> SupportDomain {
    SupportDomain::new(generate_regular_nodes(0.0, 0.0, 1.0, 1.0, n, n, 2.5), 1.0)
}

fn linear_field(p: Point2) -> [f64; 2] {
    [
        0.010 * p[0] + 0.004 * p[1] + 0.002,
        -0.003 * p[0] + 0.007 * p[1] - 0.001,
    ]
}

/// Boundary integrals of the linear field over the four edges of the unit square.
fn weak_bcs(n_pieces: usize) -> Vec<EssentialBc> {
    let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut bcs = Vec::new();
    for e in 0..4 {
        bcs.extend(essential_bcs_on_segment(
            corners[e],
            corners[(e + 1) % 4],
            n_pieces,
            3,
            |p| {
                let u = linear_field(p);
                [Some(u[0]), Some(u[1])]
            },
        ));
    }
    bcs
}

/// Point-collocated conditions of the linear field at the boundary nodes.
fn nodal_bcs(domain: &SupportDomain) -> Vec<EssentialBc> {
    let mut bcs = Vec::new();
    for (index, node) in domain.nodes.iter().enumerate() {
        let on_boundary = node
            .pos
            .iter()
            .any(|c| c.abs() < 1e-12 || (c - 1.0).abs() < 1e-12);
        if !on_boundary {
            continue;
        }
        let u = linear_field(node.pos);
        for (dof, value) in u.iter().enumerate() {
            if let Some(bc) = EssentialBc::at_node(domain, index, [0.0, 0.0], 1.0, dof, *value) {
                bcs.push(bc);
            }
        }
    }
    bcs
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Largest difference of the displacement fields `u_h` of two results over a grid of
/// sample points, relative to the largest displacement.
fn field_difference(domain: &SupportDomain, a: &EfgResult, b: &EfgResult) -> f64 {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 1..8 {
        for j in 1..8 {
            let p = [i as f64 / 8.0 + 0.013, j as f64 / 8.0 - 0.011];
            let ua = evaluate_displacement_2d(p, &a.displacements, domain, KERNEL)
                .expect("inside the cloud");
            let ub = evaluate_displacement_2d(p, &b.displacements, domain, KERNEL)
                .expect("inside the cloud");
            for c in 0..2 {
                worst = worst.max((ua[c] - ub[c]).abs());
                scale = scale.max(ua[c].abs());
            }
        }
    }
    worst / scale
}

#[test]
fn sparse_assembly_is_the_dense_assembly_entry_for_entry() {
    let domain = cloud(9);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 12, 12);
    let mat = material();
    let body = [3.0, -1.0];
    let (dense_k, dense_f) =
        efg_assemble_2d(&domain, &cells, &mat, KERNEL, body).expect("assembly");
    let (sparse_k, sparse_f) =
        efg_assemble_2d_sparse(&domain, &cells, &mat, KERNEL, body).expect("assembly");
    let ndof = 2 * domain.node_count();
    assert_eq!((sparse_k.nrows, sparse_k.ncols), (ndof, ndof));
    // one shared integration loop: the entries are identical, not merely close
    assert_eq!(dense_f, sparse_f);
    for row in 0..ndof {
        for col in 0..ndof {
            assert_eq!(
                dense_k[row * ndof + col],
                sparse_k.get(row, col),
                "({row}, {col})"
            );
        }
    }
    // only the couplings of overlapping supports are stored
    let nonzero = dense_k.iter().filter(|v| **v != 0.0).count();
    assert!(sparse_k.nnz() >= nonzero);
    assert!(
        sparse_k.nnz() < ndof * ndof,
        "nnz {} of {}",
        sparse_k.nnz(),
        ndof * ndof
    );
}

#[test]
fn boundary_contributions_are_what_the_dense_functions_add() {
    let domain = cloud(6);
    let mat = material();
    let weak = weak_bcs(5);
    let nodal = nodal_bcs(&domain);
    let ndof = 2 * domain.node_count();

    // replay a contribution on a dense system
    let replay = |c: &BcContribution, n: usize| -> (Vec<f64>, Vec<f64>) {
        let mut k = vec![0.0; n * n];
        let mut f = vec![0.0; n];
        for &(row, col, value) in &c.stiffness {
            k[row * n + col] += value;
        }
        for &(row, value) in &c.force {
            f[row] += value;
        }
        (k, f)
    };

    let penalty = penalty_bc_contribution(&domain, KERNEL, &weak, 1e6).expect("valid");
    assert_eq!(penalty.extra_dofs, 0);
    let (mut k, mut f) = (vec![0.0; ndof * ndof], vec![0.0; ndof]);
    apply_penalty_bc(&mut k, &mut f, &domain, KERNEL, &weak, 1e6).expect("valid");
    assert_eq!((k.clone(), f.clone()), replay(&penalty, ndof));
    // the penalty matrix is symmetric: α w φ_I φ_J
    let scale = max_abs(&k);
    for row in 0..ndof {
        for col in 0..ndof {
            assert!((k[row * ndof + col] - k[col * ndof + row]).abs() <= 1e-14 * scale);
        }
    }

    let nitsche = nitsche_bc_contribution(&domain, KERNEL, &mat, &weak, 40.0).expect("valid");
    assert_eq!(nitsche.extra_dofs, 0);
    let (mut k, mut f) = (vec![0.0; ndof * ndof], vec![0.0; ndof]);
    apply_nitsche_bc(&mut k, &mut f, &domain, KERNEL, &mat, &weak, 40.0).expect("valid");
    assert_eq!((k, f), replay(&nitsche, ndof));

    let lagrange = lagrange_bc_contribution(&domain, KERNEL, &nodal).expect("valid");
    assert_eq!(lagrange.extra_dofs, nodal.len());
    let n_aug = ndof + nodal.len();
    // the constraint block is symmetric and addresses only multiplier rows/columns
    for &(row, col, value) in &lagrange.stiffness {
        assert!(
            (row >= ndof) != (col >= ndof),
            "({row}, {col}) couples primal and multiplier"
        );
        assert!(lagrange.stiffness.contains(&(col, row, value)));
        assert!(row < n_aug && col < n_aug);
    }
    // one right-hand-side entry per constraint: the prescribed value
    assert_eq!(lagrange.force.len(), nodal.len());
    for (c, bc) in nodal.iter().enumerate() {
        assert_eq!(lagrange.force[c], (ndof + c, bc.value));
    }
    let (k_dense, f_dense) = (vec![0.0; ndof * ndof], vec![0.0; ndof]);
    let (k_aug, f_aug) =
        apply_lagrange_bc(&k_dense, &f_dense, &domain, KERNEL, &nodal).expect("valid");
    assert_eq!((k_aug, f_aug), replay(&lagrange, n_aug));

    // invalid input is rejected and nothing is produced
    assert!(penalty_bc_contribution(&domain, KERNEL, &weak, 0.0).is_err());
    assert!(nitsche_bc_contribution(&domain, KERNEL, &mat, &weak, -1.0).is_err());
}

#[test]
fn sparse_and_dense_solutions_agree_on_a_10_by_10_patch() {
    // 100 nodes = 200 DOF, exactly the threshold: this is the first size that
    // `efg_analysis_2d` sends through the sparse path.
    let domain = cloud(10);
    assert_eq!(2 * domain.node_count(), EFG_SPARSE_DOF_THRESHOLD);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 18, 18);
    let mat = material();
    let weak = weak_bcs(9);
    let nodal = nodal_bcs(&domain);
    for (name, method, bcs, parameter) in [
        ("penalty", BcEnforcementMethod::Penalty, &weak, 1e6),
        ("nitsche", BcEnforcementMethod::Nitsche, &weak, 50.0),
        (
            "lagrange",
            BcEnforcementMethod::LagrangeMultiplier,
            &nodal,
            0.0,
        ),
    ] {
        let dense = efg_analysis_2d_dense(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, 0.0],
            bcs,
            method,
            parameter,
        )
        .unwrap_or_else(|| panic!("{name}: dense analysis failed"));
        let sparse = efg_analysis_2d_sparse(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, 0.0],
            bcs,
            method,
            parameter,
        )
        .unwrap_or_else(|| panic!("{name}: sparse analysis failed"));
        assert_eq!(sparse.num_nodes, dense.num_nodes);
        assert_eq!(sparse.displacements.len(), dense.displacements.len());
        let field = field_difference(&domain, &dense, &sparse);
        assert!(field < 1e-10, "{name}: field difference {field:e}");
        let scale = max_abs(&dense.displacements);
        let parameters = dense
            .displacements
            .iter()
            .zip(sparse.displacements.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            parameters < 1e-8 * scale,
            "{name}: parameter difference {:e}",
            parameters / scale
        );
    }
}

#[test]
fn larger_systems_are_routed_through_the_sparse_path_and_match_the_dense_reference() {
    // 14 x 14 = 392 DOF: `efg_analysis_2d` takes the sparse path here.
    let domain = cloud(14);
    assert!(2 * domain.node_count() > EFG_SPARSE_DOF_THRESHOLD);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 26, 26);
    let mat = material();
    let weak = weak_bcs(13);
    let nodal = nodal_bcs(&domain);
    for (name, method, bcs, parameter) in [
        ("penalty", BcEnforcementMethod::Penalty, &weak, 1e6),
        ("nitsche", BcEnforcementMethod::Nitsche, &weak, 50.0),
        (
            "lagrange",
            BcEnforcementMethod::LagrangeMultiplier,
            &nodal,
            0.0,
        ),
    ] {
        let routed = efg_analysis_2d(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, 0.0],
            bcs,
            method,
            parameter,
        )
        .unwrap_or_else(|| panic!("{name}: routed analysis failed"));
        let dense = efg_analysis_2d_dense(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, 0.0],
            bcs,
            method,
            parameter,
        )
        .unwrap_or_else(|| panic!("{name}: dense analysis failed"));
        let field = field_difference(&domain, &routed, &dense);
        assert!(field < 1e-9, "{name}: field difference {field:e}");
        // and the solution solves the problem: the linear field is recovered
        let linear = {
            let mut worst = 0.0_f64;
            for i in 1..8 {
                for j in 1..8 {
                    let p = [i as f64 / 8.0 + 0.013, j as f64 / 8.0 - 0.011];
                    let u = evaluate_displacement_2d(p, &routed.displacements, &domain, KERNEL)
                        .expect("inside the cloud");
                    let e = linear_field(p);
                    worst = worst.max((u[0] - e[0]).abs()).max((u[1] - e[1]).abs());
                }
            }
            worst
        };
        assert!(
            linear < 1e-2 * 0.016,
            "{name}: error against the linear field {linear:e}"
        );
    }
}

#[test]
fn small_systems_keep_the_dense_path_bit_for_bit() {
    let domain = cloud(9); // 162 DOF
    assert!(2 * domain.node_count() < EFG_SPARSE_DOF_THRESHOLD);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 14, 14);
    let mat = material();
    let weak = weak_bcs(8);
    let routed = efg_analysis_2d(
        &domain,
        &cells,
        &mat,
        KERNEL,
        [0.0, 0.0],
        &weak,
        BcEnforcementMethod::Nitsche,
        50.0,
    )
    .expect("analysis succeeds");
    let dense = efg_analysis_2d_dense(
        &domain,
        &cells,
        &mat,
        KERNEL,
        [0.0, 0.0],
        &weak,
        BcEnforcementMethod::Nitsche,
        50.0,
    )
    .expect("analysis succeeds");
    assert_eq!(routed.displacements, dense.displacements);
}

#[test]
fn a_nitsche_parameter_that_is_too_small_is_still_solved_correctly() {
    // For β = 5 the Nitsche matrix is no longer positive definite (its smallest
    // equilibrated eigenvalue is negative): the solver chain (CG with a true-residual
    // check, MINRES, GMRES, dense fallback) must still deliver the dense solution.
    let domain = cloud(11);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 20, 20);
    let mat = material();
    let weak = weak_bcs(10);
    let routed = efg_analysis_2d(
        &domain,
        &cells,
        &mat,
        KERNEL,
        [0.0, 0.0],
        &weak,
        BcEnforcementMethod::Nitsche,
        5.0,
    )
    .expect("analysis succeeds");
    let dense = efg_analysis_2d_dense(
        &domain,
        &cells,
        &mat,
        KERNEL,
        [0.0, 0.0],
        &weak,
        BcEnforcementMethod::Nitsche,
        5.0,
    )
    .expect("analysis succeeds");
    let field = field_difference(&domain, &routed, &dense);
    assert!(field < 1e-9, "field difference {field:e}");
}

#[test]
fn the_sparse_analysis_never_returns_an_unconverged_solution() {
    // A free body under a net load has no static solution (rigid-body modes): the
    // iterations cannot converge, and the sparse path must say so instead of returning
    // the last iterate.
    let domain = cloud(10);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 18, 18);
    let mat = material();
    let sparse = efg_analysis_2d_sparse(
        &domain,
        &cells,
        &mat,
        KERNEL,
        [0.0, -1.0],
        &[],
        BcEnforcementMethod::Penalty,
        1e6,
    );
    assert!(sparse.is_none(), "an unsolvable system returned a solution");
    // the same body held at its boundary is solvable
    let weak = weak_bcs(9);
    assert!(
        efg_analysis_2d_sparse(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, -1.0],
            &weak,
            BcEnforcementMethod::Penalty,
            1e6,
        )
        .is_some()
    );
}

#[test]
fn invalid_boundary_input_fails_both_paths() {
    let domain = cloud(10);
    let cells = generate_tri_cells(0.0, 0.0, 1.0, 1.0, 18, 18);
    let mat = material();
    let weak = weak_bcs(9);
    for method in [BcEnforcementMethod::Penalty, BcEnforcementMethod::Nitsche] {
        // a non-positive parameter is rejected by the contribution functions
        assert!(
            efg_analysis_2d_sparse(
                &domain,
                &cells,
                &mat,
                KERNEL,
                [0.0, 0.0],
                &weak,
                method,
                0.0
            )
            .is_none()
        );
        assert!(
            efg_analysis_2d_dense(
                &domain,
                &cells,
                &mat,
                KERNEL,
                [0.0, 0.0],
                &weak,
                method,
                0.0
            )
            .is_none()
        );
    }
    // a malformed condition (DOF 2)
    let bad = vec![EssentialBc::new([0.0, 0.5], [-1.0, 0.0], 0.1, 2, 0.0)];
    assert!(
        efg_analysis_2d_sparse(
            &domain,
            &cells,
            &mat,
            KERNEL,
            [0.0, 0.0],
            &bad,
            BcEnforcementMethod::LagrangeMultiplier,
            0.0,
        )
        .is_none()
    );
}
