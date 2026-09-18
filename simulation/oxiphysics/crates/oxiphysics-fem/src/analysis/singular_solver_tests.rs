//! Singular and ill-posed systems of the dense solvers.
//!
//! The static analysis, the Cholesky direct solver and the boundary-element solve
//! used to disagree about singular matrices: one returned `NaN`, one left the
//! unknowns at zero and one panicked, each with its own absolute threshold.  They
//! now share one relative test and report [`Error::SingularSystem`] — never `NaN`,
//! a silent zero or a panic.

use super::dense::{cholesky_solve, lu_solve};
use super::{FemError, StaticAnalysis};
use crate::assembly::{StaticSolver, cholesky_solve as assembly_cholesky_solve};
use crate::boundary_element::{BemBc, BemBcType, DenseMatrix, solve_laplace_bem};
use crate::error::Error;

fn assert_close(got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= tolerance, "{got:?} vs {want:?}");
    }
}

// ---- shared kernels ------------------------------------------------------

#[test]
fn lu_solve_solves_regular_systems_that_need_pivoting() {
    // The leading entry is zero: an unpivoted elimination would divide by it.
    let a = [0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 4.0];
    let x_true = [1.0, -2.0, 0.5];
    let b: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| a[i * 3 + j] * x_true[j]).sum())
        .collect();
    let x = lu_solve(&a, &b, 3).expect("regular system");
    assert_close(&x, &x_true, 1e-13);
}

#[test]
fn lu_solve_reports_a_singular_2x2_as_an_error_never_nan() {
    // Rank one: the second pivot vanishes.  The old Gaussian elimination skipped it
    // and divided by zero or left the unknown at zero.
    let result = lu_solve(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0], 2);
    assert!(matches!(result, Err(Error::SingularSystem)), "{result:?}");
    let zero = lu_solve(&[0.0; 4], &[1.0, 1.0], 2);
    assert!(matches!(zero, Err(Error::SingularSystem)));
}

#[test]
fn lu_solve_verdict_does_not_depend_on_the_unit_system() {
    let regular = [2.0, -1.0, -1.0, 3.0];
    let singular = [1.0, 2.0, 2.0, 4.0];
    let rhs = [1.0, 1.0];
    let reference = lu_solve(&regular, &rhs, 2).expect("regular system");
    for scale in [1.0e-30, 1.0e-12, 1.0, 1.0e12, 1.0e30] {
        let scaled_regular: Vec<f64> = regular.iter().map(|v| v * scale).collect();
        let scaled_rhs: Vec<f64> = rhs.iter().map(|v| v * scale).collect();
        let x = lu_solve(&scaled_regular, &scaled_rhs, 2).expect("scale-invariant");
        assert_close(&x, &reference, 1e-12);
        let scaled_singular: Vec<f64> = singular.iter().map(|v| v * scale).collect();
        let result = lu_solve(&scaled_singular, &scaled_rhs, 2);
        assert!(
            matches!(result, Err(Error::SingularSystem)),
            "scale {scale}: {result:?}"
        );
    }
}

#[test]
fn lu_solve_validates_its_input() {
    // Wrong matrix size.
    assert!(matches!(
        lu_solve(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    // Wrong right-hand side size.
    assert!(matches!(
        lu_solve(&[1.0, 0.0, 0.0, 1.0], &[1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    // Non-finite data is rejected instead of being propagated.
    assert!(matches!(
        lu_solve(&[1.0, f64::NAN, 0.0, 1.0], &[1.0, 1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        lu_solve(&[1.0, 0.0, 0.0, 1.0], &[f64::INFINITY, 1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    // The empty system is trivially solved.
    assert_eq!(
        lu_solve(&[], &[], 0).expect("empty system"),
        Vec::<f64>::new()
    );
}

#[test]
fn cholesky_solve_solves_spd_systems() {
    // [[4, 2], [2, 3]]·u = [4, 3]  ⇒  u = [0.75, 0.5]
    let x = cholesky_solve(&[4.0, 2.0, 2.0, 3.0], &[4.0, 3.0], 2).expect("SPD system");
    assert_close(&x, &[0.75, 0.5], 1e-13);
    // A 4x4 diagonally dominant SPD matrix against the LU solution.
    let a = [
        5.0, 1.0, 0.5, 0.0, 1.0, 6.0, 1.0, 0.5, 0.5, 1.0, 7.0, 1.0, 0.0, 0.5, 1.0, 8.0,
    ];
    let b = [1.0, -2.0, 3.0, -4.0];
    let by_cholesky = cholesky_solve(&a, &b, 4).expect("SPD system");
    let by_lu = lu_solve(&a, &b, 4).expect("regular system");
    assert_close(&by_cholesky, &by_lu, 1e-13);
}

#[test]
fn cholesky_solve_reports_non_positive_definite_matrices_as_singular() {
    // A free spring: rigid-body mode, pivot exactly zero.  This used to panic.
    let spring = [1.0, -1.0, -1.0, 1.0];
    let result = cholesky_solve(&spring, &[1.0, -1.0], 2);
    assert!(matches!(result, Err(Error::SingularSystem)), "{result:?}");
    // Indefinite and zero-diagonal matrices.
    let indefinite = cholesky_solve(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0], 2);
    assert!(matches!(indefinite, Err(Error::SingularSystem)));
    let zero_diagonal = cholesky_solve(&[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0], 2);
    assert!(matches!(zero_diagonal, Err(Error::SingularSystem)));
}

#[test]
fn cholesky_solve_is_scale_invariant_and_validates_its_input() {
    let spd = [4.0, 2.0, 2.0, 3.0];
    let rhs = [4.0, 3.0];
    for scale in [1.0e-30, 1.0, 1.0e30] {
        let a: Vec<f64> = spd.iter().map(|v| v * scale).collect();
        let b: Vec<f64> = rhs.iter().map(|v| v * scale).collect();
        let x = cholesky_solve(&a, &b, 2).expect("scale-invariant");
        assert_close(&x, &[0.75, 0.5], 1e-12);
        let spring: Vec<f64> = [1.0, -1.0, -1.0, 1.0].iter().map(|v| v * scale).collect();
        assert!(matches!(
            cholesky_solve(&spring, &b, 2),
            Err(Error::SingularSystem)
        ));
    }
    assert!(matches!(
        cholesky_solve(&spd, &[1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        cholesky_solve(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        cholesky_solve(&[f64::NAN, 0.0, 0.0, 1.0], &[1.0, 1.0], 2),
        Err(Error::InvalidInput(_))
    ));
}

// ---- StaticAnalysis -------------------------------------------------------

fn static_system(stiffness: Vec<f64>, force: Vec<f64>) -> StaticAnalysis {
    let mut analysis = StaticAnalysis::new(force.len());
    analysis.stiffness_matrix = stiffness;
    analysis.force_vector = force;
    analysis
}

#[test]
fn static_analysis_reports_an_unconstrained_structure() {
    // Two DOFs joined by a spring and not tied to the ground.
    let analysis = static_system(vec![1.0, -1.0, -1.0, 1.0], vec![1.0, -1.0]);
    assert_eq!(analysis.solve(), Err(FemError::SingularSystem));
}

#[test]
fn static_analysis_singularity_test_is_relative_to_the_stiffness_scale() {
    // A perfectly regular stiffness matrix of order 1e-20 (a very soft model in SI
    // units) was rejected by the old absolute threshold of 1e-14.
    let soft = static_system(
        vec![2.0e-20, -1.0e-20, -1.0e-20, 3.0e-20],
        vec![1.0e-20, 1.0e-20],
    );
    let u = soft.solve().expect("regular at any scale");
    assert_close(&u, &[0.8, 0.6], 1e-12);

    // A matrix of order 1e12 whose second pivot is round-off (one ulp) is singular,
    // yet the old threshold accepted its 1e-4 pivot and returned a solution of order
    // 1e4 that was pure noise.
    let k = 1.0e12_f64;
    let noisy = static_system(vec![k, -k, -k, k + 1.0e-4], vec![1.0, 1.0]);
    assert_eq!(noisy.solve(), Err(FemError::SingularSystem));
}

#[test]
fn static_analysis_validates_sizes_and_data() {
    let mut analysis = StaticAnalysis::new(2);
    analysis.stiffness_matrix = vec![1.0, 0.0, 0.0];
    assert!(matches!(analysis.solve(), Err(FemError::InvalidInput(_))));
    analysis.stiffness_matrix = vec![1.0, 0.0, 0.0, 1.0];
    analysis.force_vector = vec![1.0];
    assert!(matches!(analysis.solve(), Err(FemError::InvalidInput(_))));
    analysis.force_vector = vec![1.0, f64::NAN];
    assert!(matches!(analysis.solve(), Err(FemError::InvalidInput(_))));
    assert_eq!(StaticAnalysis::new(0).solve(), Ok(Vec::new()));
}

// ---- assembly::cholesky_solve / StaticSolver::solve_direct ---------------

/// Stiffness matrix of a chain of `n` springs between `n` free DOFs.
fn free_spring_chain(n: usize, spring: f64) -> Vec<Vec<f64>> {
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        k[i][i] += spring;
        k[i + 1][i + 1] += spring;
        k[i][i + 1] -= spring;
        k[i + 1][i] -= spring;
    }
    k
}

#[test]
fn direct_solver_reports_an_unconstrained_stiffness_matrix_instead_of_panicking() {
    let k = free_spring_chain(3, 100.0);
    let f = [0.0, 0.0, 1.0];
    let solver = StaticSolver::default();
    // assert!(s > 0.0, ...) used to fire here.
    assert!(matches!(
        solver.solve_direct(&k, &f),
        Err(Error::SingularSystem)
    ));
    assert!(matches!(
        assembly_cholesky_solve(&k, &f),
        Err(Error::SingularSystem)
    ));
}

#[test]
fn direct_solver_solves_the_constrained_chain() {
    // Fix DOF 0 of the chain: [[200, -100], [-100, 100]]·u = [0, 1] ⇒ u = [0.01, 0.02].
    let k = vec![vec![200.0, -100.0], vec![-100.0, 100.0]];
    let u = StaticSolver::default()
        .solve_direct(&k, &[0.0, 1.0])
        .expect("constrained chain");
    assert_close(&u, &[0.01, 0.02], 1e-14);
}

#[test]
fn direct_solver_validates_sizes_and_data() {
    let solver = StaticSolver::default();
    let spd = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
    assert!(matches!(
        solver.solve_direct(&spd, &[1.0, 2.0, 3.0]),
        Err(Error::InvalidInput(_))
    ));
    let ragged = vec![vec![2.0, 0.0], vec![0.0]];
    assert!(matches!(
        solver.solve_direct(&ragged, &[1.0, 2.0]),
        Err(Error::InvalidInput(_))
    ));
    let not_a_number = vec![vec![f64::NAN, 0.0], vec![0.0, 2.0]];
    assert!(matches!(
        solver.solve_direct(&not_a_number, &[1.0, 2.0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        solver.solve_direct(&spd, &[1.0, f64::NAN]),
        Err(Error::InvalidInput(_))
    ));
}

// ---- boundary-element solve ----------------------------------------------

fn matrix(rows: usize, data: &[f64]) -> DenseMatrix {
    DenseMatrix {
        rows,
        cols: rows,
        data: data.to_vec(),
    }
}

fn condition(element_id: usize, bc_type: BemBcType, value: f64) -> BemBc {
    BemBc {
        element_id,
        bc_type,
        values: vec![value],
    }
}

/// `H = I` and a symmetric positive-definite `G`.
fn small_system() -> (DenseMatrix, DenseMatrix) {
    (
        matrix(2, &[1.0, 0.0, 0.0, 1.0]),
        matrix(2, &[2.0, 1.0, 1.0, 3.0]),
    )
}

#[test]
fn bem_solve_matches_hand_solutions() {
    let (h, g) = small_system();
    // All Dirichlet: u = [1, 2] given, G·q = H·u ⇒ q = G⁻¹u = [0.2, 0.6].
    let dirichlet = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(1, BemBcType::Dirichlet, 2.0),
    ];
    let (u, q) = solve_laplace_bem(&h, &g, &dirichlet).expect("well-posed");
    assert_close(&u, &[1.0, 2.0], 1e-14);
    assert_close(&q, &[0.2, 0.6], 1e-14);
    // All Neumann: q = [1, 2] given, H·u = G·q ⇒ u = [4, 7].
    let neumann = [
        condition(0, BemBcType::Neumann, 1.0),
        condition(1, BemBcType::Neumann, 2.0),
    ];
    let (u, q) = solve_laplace_bem(&h, &g, &neumann).expect("well-posed");
    assert_close(&u, &[4.0, 7.0], 1e-14);
    assert_close(&q, &[1.0, 2.0], 1e-14);
}

#[test]
fn bem_solve_reports_singular_systems_instead_of_zero_filling() {
    // A singular G with Dirichlet data: the old elimination skipped the zero pivot
    // and returned q = 0 for that element.
    let h = matrix(2, &[1.0, 0.0, 0.0, 1.0]);
    let g = matrix(2, &[1.0, 2.0, 2.0, 4.0]);
    let dirichlet = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(1, BemBcType::Dirichlet, 1.0),
    ];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &dirichlet),
        Err(Error::SingularSystem)
    ));
    // A pure Neumann problem: H·1 = 0, the potential is only determined up to a
    // constant.
    let h = matrix(2, &[1.0, -1.0, -1.0, 1.0]);
    let (_, g) = small_system();
    let neumann = [
        condition(0, BemBcType::Neumann, 1.0),
        condition(1, BemBcType::Neumann, -1.0),
    ];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &neumann),
        Err(Error::SingularSystem)
    ));
}

#[test]
fn bem_solve_assigns_conditions_by_element_id_not_by_position() {
    let (h, g) = small_system();
    // Element 0: u = 1 (Dirichlet).  Element 1: q = 2 (Neumann).
    // Unknowns q₀, u₁:  -2q₀ = 1  ⇒  q₀ = -0.5;  -q₀ + u₁ = 6  ⇒  u₁ = 5.5.
    let in_order = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(1, BemBcType::Neumann, 2.0),
    ];
    let reversed = [
        condition(1, BemBcType::Neumann, 2.0),
        condition(0, BemBcType::Dirichlet, 1.0),
    ];
    let (u, q) = solve_laplace_bem(&h, &g, &in_order).expect("well-posed");
    assert_close(&u, &[1.0, 5.5], 1e-14);
    assert_close(&q, &[-0.5, 2.0], 1e-14);
    // Listing the same conditions in another order must not change the answer.
    let (u_reversed, q_reversed) = solve_laplace_bem(&h, &g, &reversed).expect("well-posed");
    assert_eq!((u, q), (u_reversed, q_reversed));
}

#[test]
fn bem_solve_validates_its_input_without_panicking() {
    let (h, g) = small_system();
    // Too few conditions: an assert_eq! used to panic.
    let one = [condition(0, BemBcType::Dirichlet, 1.0)];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &one),
        Err(Error::InvalidInput(_))
    ));
    // A condition on an element that does not exist.
    let missing = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(7, BemBcType::Dirichlet, 1.0),
    ];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &missing),
        Err(Error::DofOutOfRange { dof: 7, n: 2 })
    ));
    // Two conditions on one element leave the other without any.
    let twice = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(0, BemBcType::Neumann, 1.0),
    ];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &twice),
        Err(Error::InvalidInput(_))
    ));
    // A condition without a value: values[0] used to panic.
    let valueless = [
        condition(0, BemBcType::Dirichlet, 1.0),
        BemBc {
            element_id: 1,
            bc_type: BemBcType::Neumann,
            values: Vec::new(),
        },
    ];
    assert!(matches!(
        solve_laplace_bem(&h, &g, &valueless),
        Err(Error::InvalidInput(_))
    ));
    // Matrices that are not n x n.
    let wide = DenseMatrix {
        rows: 2,
        cols: 3,
        data: vec![0.0; 6],
    };
    let ok_conditions = [
        condition(0, BemBcType::Dirichlet, 1.0),
        condition(1, BemBcType::Dirichlet, 1.0),
    ];
    assert!(matches!(
        solve_laplace_bem(&wide, &g, &ok_conditions),
        Err(Error::InvalidInput(_))
    ));
}
