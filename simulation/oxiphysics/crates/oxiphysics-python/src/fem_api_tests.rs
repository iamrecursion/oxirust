// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super`] (the Python-facing 2-D FEM API).

use super::*;

mod basic {
    use super::*;
    use crate::PyFemMesh;
    use crate::PyFemSolveResult;
    use crate::PyFemSolver;

    fn simple_mesh() -> PyFemMesh {
        // Two-triangle unit square patch
        let mut mesh = PyFemMesh::new();
        mesh.add_node(0.0, 0.0); // 0
        mesh.add_node(1.0, 0.0); // 1
        mesh.add_node(1.0, 1.0); // 2
        mesh.add_node(0.0, 1.0); // 3
        mesh.add_element(0, 1, 2);
        mesh.add_element(0, 2, 3);
        mesh
    }

    #[test]
    fn test_fem_mesh_creation() {
        let mesh = simple_mesh();
        assert_eq!(mesh.num_nodes(), 4);
        assert_eq!(mesh.num_elements(), 2);
        assert_eq!(mesh.num_dofs(), 8);
    }

    #[test]
    fn test_fem_material_steel_properties() {
        let mat = PyFemMaterial::steel();
        assert!((mat.young_modulus - 200e9).abs() < 1.0);
        assert!((mat.poisson_ratio - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_fem_material_d_matrix_symmetry() {
        let mat = PyFemMaterial::steel();
        let d = mat_plane_stress_d(&mat);
        // D[0][1] == D[1][0]
        assert!((d[1] - d[3]).abs() < 1.0, "D should be symmetric");
    }

    #[test]
    fn test_fem_element_area_positive() {
        let mesh = simple_mesh();
        let area0 = mesh.element_area(0);
        let area1 = mesh.element_area(1);
        assert!(area0 != 0.0, "area = {}", area0);
        assert!(area1 != 0.0, "area = {}", area1);
        // Sum of absolute areas should be 1.0 for the unit square
        assert!((area0.abs() + area1.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fem_dirichlet_bc_pin() {
        let bcs = PyFemDirichletBC::pin(0);
        assert_eq!(bcs.len(), 2);
        assert_eq!(bcs[0].dof, 0);
        assert_eq!(bcs[1].dof, 1);
    }

    #[test]
    fn test_fem_pin_node() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(1);
        assert_eq!(mesh.dirichlet_bcs.len(), 4);
    }

    #[test]
    fn test_fem_apply_nodal_force() {
        let mut mesh = simple_mesh();
        mesh.apply_nodal_force(2, 0.0, -1000.0);
        assert_eq!(mesh.nodal_forces.len(), 1);
        assert!((mesh.nodal_forces[0].force[1] + 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_fem_solver_creation() {
        let solver = PyFemSolver::new();
        assert_eq!(solver.max_iterations, 10_000);
        assert!((solver.tolerance - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_fem_solver_returns_none_for_empty_mesh() {
        let mesh = PyFemMesh::new();
        let solver = PyFemSolver::new();
        assert!(solver.solve(&mesh).expect("valid").is_none());
    }

    #[test]
    fn test_fem_solve_displacement_length() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        mesh.apply_nodal_force(1, 0.0, -1000.0);
        mesh.apply_nodal_force(2, 0.0, -1000.0);
        let solver = PyFemSolver::new();
        let result = solver
            .solve(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.displacements.len(), 8); // 4 nodes * 2 DOFs
    }

    #[test]
    fn test_fem_solve_von_mises_length() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        mesh.apply_nodal_force(1, 1000.0, 0.0);
        let solver = PyFemSolver::new();
        let result = solver
            .solve(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.von_mises_stress.len(), 2); // 2 elements
    }

    #[test]
    fn test_fem_solve_converges() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        mesh.apply_nodal_force(1, 0.0, -500.0);
        let solver = PyFemSolver::new();
        let result = solver
            .solve(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert!(
            result.converged,
            "CG should converge for a well-posed problem"
        );
    }

    #[test]
    fn test_fem_result_node_displacement() {
        let result = PyFemSolveResult {
            displacements: vec![0.0, 0.0, 1e-4, -2e-5, 0.5e-4, -1e-5, 0.0, 0.0],
            von_mises_stress: vec![1.0e6, 0.8e6],
            solver_iterations: 10,
            residual_norm: 1e-12,
            converged: true,
        };
        let d0 = result.node_displacement(0).unwrap();
        assert!((d0[0]).abs() < 1e-15);
        let d1 = result.node_displacement(1).unwrap();
        assert!((d1[0] - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_fem_result_max_von_mises() {
        let result = PyFemSolveResult {
            displacements: vec![0.0; 8],
            von_mises_stress: vec![1.0e6, 2.0e6, 0.5e6],
            solver_iterations: 5,
            residual_norm: 1e-12,
            converged: true,
        };
        assert!((result.max_von_mises() - 2.0e6).abs() < 1.0);
    }

    #[test]
    fn test_fem_result_mean_von_mises() {
        let result = PyFemSolveResult {
            displacements: vec![0.0; 8],
            von_mises_stress: vec![1.0, 2.0, 3.0],
            solver_iterations: 5,
            residual_norm: 1e-12,
            converged: true,
        };
        assert!((result.mean_von_mises() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fem_build_rectangle() {
        let mut mesh = PyFemMesh::new();
        let n_nodes = mesh.build_rectangle(vec![0.0, 0.0, 1.0, 1.0], vec![4, 4], 0, 1.0);
        assert_eq!(n_nodes, 25); // 5×5 = 25 nodes
        assert_eq!(mesh.num_elements(), 32); // 4×4×2 = 32 triangles
    }

    #[test]
    fn test_fem_material_aluminum() {
        let mat = PyFemMaterial::aluminum();
        assert!((mat.young_modulus - 70e9).abs() < 1.0);
    }

    #[test]
    fn test_fem_material_rubber_high_poisson() {
        let mat = PyFemMaterial::rubber();
        assert!(mat.poisson_ratio > 0.4);
    }

    #[test]
    fn test_fem_material_concrete() {
        let mat = PyFemMaterial::concrete();
        assert!((mat.density - 2400.0).abs() < 1.0);
    }

    #[test]
    fn test_fem_nodal_force_new() {
        let nf = PyFemNodalForce::new(3, 100.0, -200.0);
        assert_eq!(nf.node, 3);
        assert!((nf.force[0] - 100.0).abs() < 1e-10);
        assert!((nf.force[1] + 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_fem_dirichlet_fix_x() {
        let bc = PyFemDirichletBC::fix_x(5);
        assert_eq!(bc.node, 5);
        assert_eq!(bc.dof, 0);
        assert!((bc.value).abs() < 1e-15);
    }

    #[test]
    fn test_fem_dirichlet_fix_y() {
        let bc = PyFemDirichletBC::fix_y(2);
        assert_eq!(bc.dof, 1);
    }

    #[test]
    fn test_fem_solve_pinned_both_sides_zero_displacement_at_pin() {
        let mut mesh = simple_mesh();
        // Pin all nodes — no load — displacement should be zero
        for n in 0..4 {
            mesh.pin_node(n);
        }
        let solver = PyFemSolver::new();
        let result = solver
            .solve(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        for &d in &result.displacements {
            assert!(d.abs() < 1e-10, "constrained mesh with no load: d={}", d);
        }
    }

    #[test]
    fn test_fem_add_material_returns_index() {
        let mut mesh = PyFemMesh::new();
        let idx = mesh.add_material(PyFemMaterial::rubber());
        // Default material is at index 0, so new one is at index 1
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_fem_element_stiffness_symmetric() {
        // The element stiffness matrix should be symmetric
        let mut mesh = PyFemMesh::new();
        mesh.add_node(0.0, 0.0);
        mesh.add_node(1.0, 0.0);
        mesh.add_node(0.5, 1.0);
        mesh.add_element(0, 1, 2);

        let elem = &mesh.elements[0];
        let mat = &mesh.materials[0];
        let ke = element_stiffness(&mesh, elem, mat);

        for i in 0..6 {
            for j in 0..6 {
                let diff = (ke[i * 6 + j] - ke[j * 6 + i]).abs();
                assert!(
                    diff < 1e-6,
                    "Ke not symmetric at ({},{}): diff={}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_fem_solver_with_settings() {
        let solver = PyFemSolver::with_settings(500, 1e-8);
        assert_eq!(solver.max_iterations, 500);
        assert!((solver.tolerance - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn test_fem_result_node_displacement_out_of_bounds() {
        let result = PyFemSolveResult {
            displacements: vec![0.0; 4],
            von_mises_stress: vec![],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        assert!(result.node_displacement(10).is_none());
    }

    #[test]
    fn test_fem_mesh_serde_roundtrip() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.apply_nodal_force(2, 100.0, -200.0);
        let json = serde_json::to_string(&mesh).expect("serialize");
        let back: PyFemMesh = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.num_nodes(), 4);
        assert_eq!(back.dirichlet_bcs.len(), 2);
        assert_eq!(back.nodal_forces.len(), 1);
    }
}

mod ext {

    use crate::PyFemMesh;
    use crate::PyFemSolveResult;
    use crate::PyFemSolver;
    use crate::fem_api::ModalAnalysisResult;
    use crate::fem_api::ModalAnalyzer;

    fn simple_mesh() -> PyFemMesh {
        let mut mesh = PyFemMesh::new();
        mesh.add_node(0.0, 0.0);
        mesh.add_node(1.0, 0.0);
        mesh.add_node(1.0, 1.0);
        mesh.add_node(0.0, 1.0);
        mesh.add_element(0, 1, 2);
        mesh.add_element(0, 2, 3);
        mesh
    }

    // --- ModalAnalyzer ---

    #[test]
    fn test_modal_analysis_returns_some() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        let analyzer = ModalAnalyzer::new(2);
        let result = analyzer.analyze(&mesh).expect("valid mesh");
        assert!(result.is_some());
    }

    #[test]
    fn test_modal_analysis_returns_none_empty_mesh() {
        let mesh = PyFemMesh::new();
        let analyzer = ModalAnalyzer::new(2);
        assert!(analyzer.analyze(&mesh).expect("valid").is_none());
    }

    #[test]
    fn test_modal_analysis_num_modes() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        // Two pinned nodes leave 4 free DOFs.
        let result = ModalAnalyzer::new(2)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 2);
        let all = ModalAnalyzer::new(10)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(all.num_modes, 4, "num_modes is clamped to the free DOFs");
    }

    #[test]
    fn test_modal_analysis_frequencies_positive_and_ascending() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        let result = ModalAnalyzer::new(3)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 3);
        for w in result.natural_frequencies.windows(2) {
            assert!(
                w[0] <= w[1],
                "frequencies must ascend: {:?}",
                result.natural_frequencies
            );
        }
        // Steel patch of unit size: frequencies are in the kHz range, never
        // the fabricated 1 rad/s of an identity boundary row.
        for &f in &result.natural_frequencies {
            assert!(f > 100.0, "frequency should be physical: {}", f);
        }
    }

    #[test]
    fn test_modal_analysis_frequency_hz() {
        let mut mesh = simple_mesh();
        mesh.pin_node(0);
        mesh.pin_node(3);
        let result = ModalAnalyzer::new(1)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        let hz = result.frequency_hz(0).expect("one mode");
        let omega = result.natural_frequencies[0];
        assert!((hz - omega / (2.0 * std::f64::consts::PI)).abs() < 1e-9 * omega);
    }

    #[test]
    fn test_modal_period_s() {
        let result = ModalAnalysisResult {
            natural_frequencies: vec![2.0 * std::f64::consts::PI * 100.0], // 100 Hz
            mode_shapes: vec![vec![1.0, 0.0, 0.0, 0.0]],
            num_modes: 1,
        };
        let period = result.period_s(0).unwrap();
        assert!((period - 0.01).abs() < 1e-6, "period = {}", period);
    }

    // --- Result extraction helpers ---

    #[test]
    fn test_displacement_pairs_length() {
        let result = PyFemSolveResult {
            displacements: vec![1.0, 2.0, 3.0, 4.0],
            von_mises_stress: vec![],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        let pairs = result.displacement_pairs();
        assert_eq!(pairs.len(), 2);
        assert!((pairs[0][0] - 1.0).abs() < 1e-15);
        assert!((pairs[1][1] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_max_displacement_magnitude() {
        let result = PyFemSolveResult {
            displacements: vec![3.0, 4.0, 0.0, 0.0],
            von_mises_stress: vec![],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        assert!((result.max_displacement_magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sorted_von_mises() {
        let result = PyFemSolveResult {
            displacements: vec![],
            von_mises_stress: vec![3.0, 1.0, 2.0],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        let sorted = result.sorted_von_mises();
        assert!((sorted[0] - 1.0).abs() < 1e-15);
        assert!((sorted[2] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_von_mises_quantile_min() {
        let result = PyFemSolveResult {
            displacements: vec![],
            von_mises_stress: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        assert!((result.von_mises_quantile(0.0) - 1.0).abs() < 1e-10);
        assert!((result.von_mises_quantile(1.0) - 5.0).abs() < 1e-10);
    }

    // --- Mesh query helpers ---

    #[test]
    fn test_bounding_box_unit_square() {
        let mesh = simple_mesh();
        let bb = mesh.bounding_box().unwrap();
        assert!((bb[0]).abs() < 1e-15); // xmin = 0
        assert!((bb[2] - 1.0).abs() < 1e-15); // xmax = 1
    }

    #[test]
    fn test_bounding_box_empty() {
        let mesh = PyFemMesh::new();
        assert!(mesh.bounding_box().is_none());
    }

    #[test]
    fn test_total_area_unit_square() {
        let mesh = simple_mesh();
        assert!((mesh.total_area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_element_centroid_valid() {
        let mesh = simple_mesh();
        let c = mesh.element_centroid(0).unwrap();
        // Triangle (0,0),(1,0),(1,1): centroid = (2/3, 1/3)
        assert!((c[0] - 2.0 / 3.0).abs() < 1e-10, "cx = {}", c[0]);
    }

    #[test]
    fn test_element_centroid_out_of_bounds() {
        let mesh = simple_mesh();
        assert!(mesh.element_centroid(99).is_none());
    }

    #[test]
    fn test_closest_node_corner() {
        let mesh = simple_mesh();
        let idx = mesh.closest_node(0.01, 0.01).unwrap();
        assert_eq!(idx, 0); // closest to (0,0)
    }

    #[test]
    fn test_closest_node_empty() {
        let mesh = PyFemMesh::new();
        assert!(mesh.closest_node(0.0, 0.0).is_none());
    }

    #[test]
    fn test_left_boundary_nodes() {
        let mesh = simple_mesh();
        let left = mesh.left_boundary_nodes(1e-6);
        assert!(left.contains(&0)); // node at (0,0)
        assert!(left.contains(&3)); // node at (0,1)
    }

    #[test]
    fn test_right_boundary_nodes() {
        let mesh = simple_mesh();
        let right = mesh.right_boundary_nodes(1e-6);
        assert!(right.contains(&1)); // node at (1,0)
        assert!(right.contains(&2)); // node at (1,1)
    }

    #[test]
    fn test_pin_left_boundary() {
        let mut mesh = simple_mesh();
        mesh.pin_left_boundary(1e-6);
        // 2 left nodes × 2 DOFs each = 4 BCs
        assert_eq!(mesh.dirichlet_bcs.len(), 4);
    }

    #[test]
    fn test_fem_solve_with_large_mesh() {
        let mut mesh = PyFemMesh::new();
        mesh.build_rectangle(vec![0.0, 0.0, 1.0, 0.5], vec![4, 2], 0, 0.01);
        mesh.pin_left_boundary(1e-6);
        // Apply load on right boundary
        let right = mesh.right_boundary_nodes(1e-6);
        for n in right {
            mesh.apply_nodal_force(n, 0.0, -500.0);
        }
        let solver = PyFemSolver::new();
        let result = solver
            .solve(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert!(!result.displacements.is_empty());
    }

    #[test]
    fn test_fem_result_displacement_pairs_empty() {
        let result = PyFemSolveResult {
            displacements: vec![],
            von_mises_stress: vec![],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        assert!(result.displacement_pairs().is_empty());
    }

    #[test]
    fn test_fem_max_displacement_empty() {
        let result = PyFemSolveResult {
            displacements: vec![],
            von_mises_stress: vec![],
            solver_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        };
        assert!((result.max_displacement_magnitude()).abs() < 1e-15);
    }
}

// ===========================================================================
// Engine-backed solver behaviour: sparse PCG static solve and modal analysis
// ===========================================================================

mod engine {
    use super::*;

    /// Rectangular strip `[0, lx] × [0, ly]` meshed with `nx × ny` quads.
    fn strip(lx: f64, ly: f64, nx: usize, ny: usize, mat: PyFemMaterial) -> PyFemMesh {
        let mut mesh = PyFemMesh::new();
        mesh.materials[0] = mat;
        mesh.build_rectangle(vec![0.0, 0.0, lx, ly], vec![nx, ny], 0, 1.0);
        mesh
    }

    /// Node index of grid point `(i, j)` in a mesh built by [`strip`].
    fn grid_node(nx: usize, i: usize, j: usize) -> usize {
        j * (nx + 1) + i
    }

    fn solve(mesh: &PyFemMesh, solver: &PyFemSolver) -> PyFemSolveResult {
        solver
            .solve(mesh)
            .expect("valid mesh")
            .expect("non-empty mesh")
    }

    /// Dense Gaussian elimination with partial pivoting (test reference).
    fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))
                .expect("non-empty");
            a.swap(col, pivot);
            b.swap(col, pivot);
            let (upper, lower) = a.split_at_mut(col + 1);
            let pivot_row = &upper[col];
            for (offset, row) in lower.iter_mut().enumerate() {
                let factor = row[col] / pivot_row[col];
                for (entry, &p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *entry -= factor * p;
                }
                b[col + 1 + offset] -= factor * b[col];
            }
        }
        let mut x = vec![0.0; n];
        for row in (0..n).rev() {
            let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
            x[row] = (b[row] - s) / a[row][row];
        }
        x
    }

    #[test]
    fn test_uniaxial_patch_is_exact() {
        // Constant-strain field: CST reproduces u = P L / (E A) exactly.
        let (lx, ly, nx, ny) = (4.0, 1.0, 8, 4);
        let mat = PyFemMaterial::new(2.0e7, 0.3, 1000.0);
        let mut mesh = strip(lx, ly, nx, ny, mat);
        for j in 0..=ny {
            mesh.add_dirichlet_bc(PyFemDirichletBC::fix_x(grid_node(nx, 0, j)));
        }
        mesh.add_dirichlet_bc(PyFemDirichletBC::fix_y(grid_node(nx, 0, 0)));
        let total = 1000.0;
        for j in 0..=ny {
            let share = if j == 0 || j == ny { 0.5 } else { 1.0 };
            mesh.apply_nodal_force(grid_node(nx, nx, j), share * total / ny as f64, 0.0);
        }
        let result = solve(&mesh, &PyFemSolver::new());
        assert!(result.converged, "PCG must converge");
        let expected = total * lx / (2.0e7 * ly * 1.0);
        for j in 0..=ny {
            let ux = result.displacements[grid_node(nx, nx, j) * 2];
            assert!(
                (ux - expected).abs() < 1e-8 * expected,
                "tip ux = {ux}, expected {expected}"
            );
        }
        // Uniform uniaxial stress P / A in every element.
        for &vm in &result.von_mises_stress {
            assert!((vm - total / ly).abs() < 1e-6 * total, "von Mises = {vm}");
        }
    }

    #[test]
    fn test_prescribed_displacement_is_reproduced_exactly() {
        // The old penalty method put 1e30 * g into the load vector and could
        // not converge with an absolute tolerance.
        let (lx, ly, nx, ny) = (2.0, 0.5, 8, 2);
        let mat = PyFemMaterial::new(70e9, 0.33, 2700.0);
        let mut mesh = strip(lx, ly, nx, ny, mat);
        let g = 1e-3;
        for j in 0..=ny {
            mesh.add_dirichlet_bc(PyFemDirichletBC::fix_x(grid_node(nx, 0, j)));
            mesh.add_dirichlet_bc(PyFemDirichletBC::new(grid_node(nx, nx, j), 0, g));
        }
        mesh.add_dirichlet_bc(PyFemDirichletBC::fix_y(grid_node(nx, 0, 0)));
        let result = solve(&mesh, &PyFemSolver::new());
        assert!(result.converged, "PCG must converge with a nonzero BC");
        for j in 0..=ny {
            for i in 0..=nx {
                let x = lx * i as f64 / nx as f64;
                let ux = result.displacements[grid_node(nx, i, j) * 2];
                assert!(
                    (ux - g * x / lx).abs() < 1e-12,
                    "node ({i},{j}): ux = {ux}, expected {}",
                    g * x / lx
                );
            }
            assert_eq!(result.displacements[grid_node(nx, nx, j) * 2], g);
        }
        // Free lateral contraction: uniaxial stress E * strain.
        let sigma = 70e9 * g / lx;
        for &vm in &result.von_mises_stress {
            assert!((vm - sigma).abs() < 1e-6 * sigma, "von Mises = {vm}");
        }
    }

    #[test]
    fn test_cantilever_tip_deflection_matches_beam_theory() {
        let (length, depth, nx, ny) = (10.0, 1.0, 160, 16);
        let (e, nu) = (1.0e7, 0.3);
        let mut mesh = strip(length, depth, nx, ny, PyFemMaterial::new(e, nu, 1000.0));
        mesh.pin_left_boundary(1e-9);
        let load = -100.0;
        for j in 0..=ny {
            let share = if j == 0 || j == ny { 0.5 } else { 1.0 };
            mesh.apply_nodal_force(grid_node(nx, nx, j), 0.0, share * load / ny as f64);
        }
        let result = solve(&mesh, &PyFemSolver::with_settings(20_000, 1e-10));
        assert!(
            result.converged,
            "PCG must converge: {} its",
            result.solver_iterations
        );

        let inertia = depth.powi(3) / 12.0;
        let shear_modulus = e / (2.0 * (1.0 + nu));
        let bending = load * length.powi(3) / (3.0 * e * inertia);
        let shear = load * length / (5.0 / 6.0 * shear_modulus * depth);
        let reference = bending + shear;
        let tip = result.displacements[grid_node(nx, nx, ny / 2) * 2 + 1];
        let rel = (tip - reference).abs() / reference.abs();
        assert!(
            rel < 0.05,
            "tip deflection {tip} vs Timoshenko {reference} (rel err {rel})"
        );
    }

    #[test]
    fn test_large_mesh_solves_sparsely() {
        // 100 × 100 quads = 20 402 DOFs: the dense solver needed 3.3 GB here.
        let (nx, ny) = (100, 100);
        let mut mesh = strip(1.0, 1.0, nx, ny, PyFemMaterial::steel());
        mesh.pin_left_boundary(1e-9);
        for j in 0..=ny {
            mesh.apply_nodal_force(grid_node(nx, nx, j), 1.0e3, -1.0e3);
        }
        assert_eq!(mesh.num_dofs(), 20_402);
        let result = solve(&mesh, &PyFemSolver::with_settings(20_000, 1e-8));
        assert!(result.converged, "{} its", result.solver_iterations);
        assert!(result.max_displacement_magnitude() > 0.0);
        for j in 0..=ny {
            let base = grid_node(nx, 0, j) * 2;
            assert_eq!(result.displacements[base], 0.0);
            assert_eq!(result.displacements[base + 1], 0.0);
        }
    }

    #[test]
    fn test_elimination_matches_dense_reference_solution() {
        // Wrapper (engine CSR + PCG) vs. a dense direct solve of the same
        // system with the Dirichlet rows/columns eliminated by hand.
        let (nx, ny) = (3, 2);
        let mut mesh = strip(3.0, 2.0, nx, ny, PyFemMaterial::new(5.0e6, 0.25, 1.0));
        mesh.pin_node(grid_node(nx, 0, 0));
        mesh.add_dirichlet_bc(PyFemDirichletBC::new(grid_node(nx, 0, ny), 0, 2e-4));
        mesh.add_dirichlet_bc(PyFemDirichletBC::new(grid_node(nx, 0, ny), 1, -1e-4));
        mesh.apply_nodal_force(grid_node(nx, nx, ny), 50.0, -80.0);
        mesh.apply_nodal_force(grid_node(nx, 2, 1), -20.0, 10.0);
        let result = solve(&mesh, &PyFemSolver::new());
        assert!(result.converged);

        let n = mesh.num_dofs();
        let mut k = vec![vec![0.0; n]; n];
        for (i, j, v) in assemble_stiffness_triplets(&mesh) {
            k[i][j] += v;
        }
        let mut f = vec![0.0; n];
        for nf in &mesh.nodal_forces {
            f[nf.node * 2] += nf.force[0];
            f[nf.node * 2 + 1] += nf.force[1];
        }
        let fixed = prescribed_dofs(&mesh).expect("consistent BCs");
        let free: Vec<usize> = (0..n).filter(|&d| fixed[d].is_none()).collect();
        let a: Vec<Vec<f64>> = free
            .iter()
            .map(|&i| free.iter().map(|&j| k[i][j]).collect())
            .collect();
        let b: Vec<f64> = free
            .iter()
            .map(|&i| {
                f[i] - (0..n)
                    .filter_map(|j| fixed[j].map(|g| k[i][j] * g))
                    .sum::<f64>()
            })
            .collect();
        let x = dense_solve(a, b);
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (r, &dof) in free.iter().enumerate() {
            assert!(
                (result.displacements[dof] - x[r]).abs() < 1e-7 * scale,
                "dof {dof}: {} vs {}",
                result.displacements[dof],
                x[r]
            );
        }
        for (dof, g) in fixed.iter().enumerate() {
            if let Some(g) = g {
                assert_eq!(result.displacements[dof], *g);
            }
        }
    }

    #[test]
    fn test_unused_unloaded_node_stays_at_rest() {
        let mut mesh = strip(1.0, 1.0, 2, 2, PyFemMaterial::steel());
        mesh.pin_left_boundary(1e-9);
        mesh.apply_nodal_force(grid_node(2, 2, 1), 1.0e4, 0.0);
        let orphan = mesh.add_node(5.0, 5.0);
        let result = solve(&mesh, &PyFemSolver::new());
        assert!(result.converged);
        assert_eq!(result.displacements[orphan * 2], 0.0);
        assert_eq!(result.displacements[orphan * 2 + 1], 0.0);
        assert!(result.displacements[grid_node(2, 2, 1) * 2] > 0.0);
    }

    #[test]
    fn test_invalid_meshes_are_rejected() {
        let solver = PyFemSolver::new();
        let base = || {
            let mut mesh = strip(1.0, 1.0, 1, 1, PyFemMaterial::steel());
            mesh.pin_left_boundary(1e-9);
            mesh
        };

        let mut bad_node = base();
        bad_node.add_element(0, 1, 9);
        assert!(
            solver.solve(&bad_node).is_err(),
            "element node out of range"
        );

        let mut bad_dof = base();
        bad_dof.add_dirichlet_bc(PyFemDirichletBC::new(1, 2, 0.0));
        assert!(solver.solve(&bad_dof).is_err(), "DOF index 2");

        let mut conflicting = base();
        conflicting.add_dirichlet_bc(PyFemDirichletBC::new(0, 0, 1e-3));
        assert!(solver.solve(&conflicting).is_err(), "conflicting BCs");

        let mut repeated = base();
        repeated.pin_node(0);
        assert!(solver.solve(&repeated).is_ok(), "identical BCs are fine");

        let mut bad_load = base();
        bad_load.apply_nodal_force(17, 1.0, 0.0);
        assert!(solver.solve(&bad_load).is_err(), "load node out of range");

        let mut orphan_load = base();
        let orphan = orphan_load.add_node(3.0, 3.0);
        orphan_load.apply_nodal_force(orphan, 1.0, 0.0);
        assert!(
            solver.solve(&orphan_load).is_err(),
            "load without stiffness"
        );

        let mut bad_material = base();
        bad_material.add_element_with_material(0, 1, 3, 5, 1.0);
        assert!(
            solver.solve(&bad_material).is_err(),
            "material out of range"
        );

        let mut degenerate = base();
        let c = degenerate.add_node(2.0, 0.0);
        degenerate.add_element(1, c, 3);
        assert!(solver.solve(&degenerate).is_ok(), "a proper extra triangle");
        degenerate.add_element(0, 1, c);
        assert!(solver.solve(&degenerate).is_err(), "collinear element");

        let mut bad_poisson = base();
        bad_poisson.materials[0].poisson_ratio = 0.5;
        assert!(solver.solve(&bad_poisson).is_err(), "nu = 0.5");

        assert!(PyFemSolver::with_settings(0, 1e-8).solve(&base()).is_err());
        assert!(PyFemSolver::with_settings(10, 0.0).solve(&base()).is_err());
    }

    // --- Modal analysis ---

    #[test]
    fn test_axial_bar_fundamental_frequency() {
        // nu = 0, all y-DOFs fixed, x fixed at x = 0: a fixed-free rod with
        // omega_1 = (pi / 2L) sqrt(E / rho), omega_2 = 3 omega_1.
        let (length, nx, ny) = (2.0, 40, 2);
        let (e, rho) = (2.0e11, 7800.0);
        let mut mesh = strip(length, 0.2, nx, ny, PyFemMaterial::new(e, 0.0, rho));
        for j in 0..=ny {
            mesh.add_dirichlet_bc(PyFemDirichletBC::fix_x(grid_node(nx, 0, j)));
            for i in 0..=nx {
                mesh.add_dirichlet_bc(PyFemDirichletBC::fix_y(grid_node(nx, i, j)));
            }
        }
        let result = ModalAnalyzer::new(2)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 2);
        let omega1 = std::f64::consts::PI / (2.0 * length) * (e / rho).sqrt();
        let w = &result.natural_frequencies;
        assert!(
            (w[0] - omega1).abs() < 0.05 * omega1,
            "omega_1 = {}, expected {omega1}",
            w[0]
        );
        assert!(
            (w[1] - 3.0 * omega1).abs() < 0.05 * 3.0 * omega1,
            "omega_2 = {}, expected {}",
            w[1],
            3.0 * omega1
        );
        // Mode 1 is axial: y-DOFs are constrained (zero) and the free end
        // moves the most.
        let phi = &result.mode_shapes[0];
        for j in 0..=ny {
            for i in 0..=nx {
                assert_eq!(phi[grid_node(nx, i, j) * 2 + 1], 0.0);
            }
            assert_eq!(phi[grid_node(nx, 0, j) * 2], 0.0);
        }
        let tip = phi[grid_node(nx, nx, 1) * 2].abs();
        let mid = phi[grid_node(nx, nx / 2, 1) * 2].abs();
        assert!(tip > mid && mid > 0.0);
    }

    #[test]
    fn test_cantilever_first_bending_frequency() {
        let (length, depth, nx, ny) = (10.0, 1.0, 40, 4);
        let (e, rho) = (2.0e11, 7850.0);
        let mut mesh = strip(length, depth, nx, ny, PyFemMaterial::new(e, 0.0, rho));
        mesh.pin_left_boundary(1e-9);
        let result = ModalAnalyzer::new(3)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 3);
        // Euler-Bernoulli: omega_1 = 1.8751^2 sqrt(E I / (rho A L^4)).
        let inertia = depth.powi(3) / 12.0;
        let area = depth;
        let beta_l: f64 = 1.875_104_068_7;
        let omega1 = beta_l.powi(2) * (e * inertia / (rho * area * length.powi(4))).sqrt();
        let w0 = result.natural_frequencies[0];
        assert!(
            (w0 - omega1).abs() < 0.10 * omega1,
            "omega_1 = {w0}, Euler-Bernoulli {omega1}"
        );
        for &w in &result.natural_frequencies {
            assert!((w - 1.0).abs() > 1.0, "fabricated unit frequency: {w}");
        }
    }

    #[test]
    fn test_modes_are_mass_normalised_and_mass_orthogonal() {
        let (nx, ny) = (6, 2);
        let mut mesh = strip(3.0, 1.0, nx, ny, PyFemMaterial::new(1.0e6, 0.25, 500.0));
        mesh.materials[0].density = 500.0;
        mesh.pin_left_boundary(1e-9);
        let result = ModalAnalyzer::new(4)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 4);
        // Lumped mass: rho t |A| / 3 per element node and direction.
        let mut mass = vec![0.0; mesh.num_dofs()];
        for (eidx, elem) in mesh.elements.iter().enumerate() {
            let nodal = 500.0 * elem.thickness * mesh.element_area(eidx).abs() / 3.0;
            for dof in element_dofs(elem) {
                mass[dof] += nodal;
            }
        }
        for a in 0..4 {
            for b in 0..4 {
                let m_ab: f64 = (0..mesh.num_dofs())
                    .map(|d| result.mode_shapes[a][d] * mass[d] * result.mode_shapes[b][d])
                    .sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((m_ab - expected).abs() < 1e-6, "M[{a}][{b}] = {m_ab}");
            }
        }
    }

    #[test]
    fn test_modal_parity_with_engine_generalized_solver() {
        // Wrapper vs. the engine's dense Jacobi generalized solver on the
        // identical reduced (K, M_lumped) pencil.
        let mut mesh = PyFemMesh::new();
        mesh.add_node(0.0, 0.0);
        mesh.add_node(1.0, 0.0);
        mesh.add_node(1.0, 1.0);
        mesh.add_node(0.0, 1.0);
        mesh.add_element(0, 1, 2);
        mesh.add_element(0, 2, 3);
        mesh.pin_node(0);
        mesh.pin_node(3);
        let result = ModalAnalyzer::new(4)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 4);

        let free = [2usize, 3, 4, 5];
        let n = mesh.num_dofs();
        let mut k = vec![vec![0.0; n]; n];
        for (i, j, v) in assemble_stiffness_triplets(&mesh) {
            k[i][j] += v;
        }
        let rho = mesh.materials[0].density;
        let mut mass = vec![0.0; n];
        for (eidx, elem) in mesh.elements.iter().enumerate() {
            let nodal = rho * elem.thickness * mesh.element_area(eidx).abs() / 3.0;
            for dof in element_dofs(elem) {
                mass[dof] += nodal;
            }
        }
        let k_red: Vec<Vec<f64>> = free
            .iter()
            .map(|&i| free.iter().map(|&j| k[i][j]).collect())
            .collect();
        let m_red: Vec<f64> = free.iter().map(|&d| mass[d]).collect();
        let reference = oxiphysics::fem::eigen::modal_analysis(&k_red, &m_red, 4);
        for (w, osq) in result.natural_frequencies.iter().zip(&reference.omega_sq) {
            let expected = osq.max(0.0).sqrt();
            assert!(
                (w - expected).abs() < 1e-6 * expected,
                "omega {w} vs engine {expected}"
            );
        }
    }

    #[test]
    fn test_unconstrained_mesh_has_three_rigid_body_modes() {
        let mesh = strip(2.0, 1.0, 4, 2, PyFemMaterial::steel());
        let result = ModalAnalyzer::new(4)
            .analyze(&mesh)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 4);
        let elastic = result.natural_frequencies[3];
        assert!(elastic > 100.0, "first elastic mode {elastic}");
        for &w in &result.natural_frequencies[..3] {
            assert!(w < 1e-4 * elastic, "rigid-body mode should be ~0: {w}");
        }
    }

    #[test]
    fn test_modal_rejects_invalid_input_and_handles_full_constraint() {
        let mut zero_density = strip(1.0, 1.0, 1, 1, PyFemMaterial::new(1.0e6, 0.3, 0.0));
        zero_density.pin_node(0);
        assert!(ModalAnalyzer::new(2).analyze(&zero_density).is_err());

        let mut all_pinned = strip(1.0, 1.0, 1, 1, PyFemMaterial::steel());
        for n in 0..all_pinned.num_nodes() {
            all_pinned.pin_node(n);
        }
        let result = ModalAnalyzer::new(2)
            .analyze(&all_pinned)
            .expect("valid mesh")
            .expect("non-empty mesh");
        assert_eq!(result.num_modes, 0);
        assert!(result.natural_frequencies.is_empty());

        let mut bad = ModalAnalyzer::new(2);
        bad.max_iter = 0;
        assert!(bad.analyze(&all_pinned).is_err());
    }
}
