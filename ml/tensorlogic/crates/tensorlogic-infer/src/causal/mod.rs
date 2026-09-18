//! Causal inference primitives for TensorLogic.
//!
//! Provides do-calculus interventions, backdoor criterion, frontdoor criterion,
//! instrumental variables, and average treatment effect (ATE) estimation.
//!
//! ## Key Types
//! - [`CausalGraph`]: Directed acyclic graph for causal structure
//! - [`ObservationalData`]: Observational data container
//! - [`Intervention`]: do-operator specification
//! - [`TreatmentEffect`]: ATE estimation result
//! - [`BackdoorAdjustment`]: Backdoor adjustment set
//!
//! ## Key Functions
//! - [`backdoor_criterion`]: Check backdoor criterion validity
//! - [`find_backdoor_adjustment`]: Find minimal backdoor adjustment set
//! - [`frontdoor_criterion`]: Check frontdoor criterion validity
//! - [`ate_backdoor`]: Estimate ATE via backdoor adjustment
//! - [`ate_instrumental_variable`]: Estimate ATE via instrumental variable
//! - [`do_intervention`]: Apply do-operator (graph mutilation)
//! - [`propensity_score`]: Compute propensity scores via logistic regression

mod criteria;
mod data;
mod error;
mod estimation;
mod graph;

pub use criteria::{
    backdoor_criterion, do_intervention, find_backdoor_adjustment, frontdoor_criterion,
};
pub use data::{BackdoorAdjustment, Intervention, ObservationalData, TreatmentEffect};
pub use error::CausalError;
pub use estimation::{ate_backdoor, ate_instrumental_variable, propensity_score};
pub use graph::CausalGraph;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper: build a simple chain graph X→Y→Z ---
    fn chain_graph() -> CausalGraph {
        let mut g = CausalGraph::new(vec!["X".into(), "Y".into(), "Z".into()]);
        g.add_edge("X", "Y").unwrap();
        g.add_edge("Y", "Z").unwrap();
        g
    }

    // --- Helper: confounded graph X←C→Y, X→Y ---
    fn confounded_graph() -> CausalGraph {
        let mut g = CausalGraph::new(vec!["C".into(), "X".into(), "Y".into()]);
        g.add_edge("C", "X").unwrap();
        g.add_edge("C", "Y").unwrap();
        g.add_edge("X", "Y").unwrap();
        g
    }

    // --- Helper: build observational data with known ATE = 2.0 ---
    fn simple_treatment_data() -> ObservationalData {
        // Y = 2*T + noise (noise=0 for deterministic test)
        let mut data = ObservationalData::new(vec!["T".into(), "Y".into()]);
        for t in &[0.0_f64, 1.0] {
            for _ in 0..50 {
                data.add_sample(vec![*t, 2.0 * t]).unwrap();
            }
        }
        data
    }

    // --- Test 1: node count ---
    #[test]
    fn test_node_count() {
        let g = CausalGraph::new(vec!["A".into(), "B".into(), "C".into()]);
        assert_eq!(g.node_count(), 3);
    }

    // --- Test 2: add_edge creates correct parent/child ---
    #[test]
    fn test_add_edge_parent_child() {
        let mut g = CausalGraph::new(vec!["A".into(), "B".into()]);
        g.add_edge("A", "B").unwrap();
        assert!(g.parents_of("B").contains(&"A".to_string()));
        assert!(g.children_of("A").contains(&"B".to_string()));
    }

    // --- Test 3: parents_of ---
    #[test]
    fn test_parents_of() {
        let g = confounded_graph();
        let parents = g.parents_of("Y");
        assert!(parents.contains(&"C".to_string()));
        assert!(parents.contains(&"X".to_string()));
    }

    // --- Test 4: ancestors_of (transitive) ---
    #[test]
    fn test_ancestors_transitive() {
        let g = chain_graph();
        let ancs = g.ancestors_of("Z");
        assert!(ancs.contains(&"X".to_string()));
        assert!(ancs.contains(&"Y".to_string()));
    }

    // --- Test 5: is_acyclic true for DAG ---
    #[test]
    fn test_is_acyclic_dag() {
        let g = chain_graph();
        assert!(g.is_acyclic());
    }

    // --- Test 6: is_acyclic false for cycle ---
    #[test]
    fn test_is_acyclic_cycle() {
        let mut g = CausalGraph::new(vec!["A".into(), "B".into(), "C".into()]);
        g.add_edge("A", "B").unwrap();
        g.add_edge("B", "C").unwrap();
        g.add_edge("C", "A").unwrap();
        assert!(!g.is_acyclic());
    }

    // --- Test 7: d-separated in chain given middle node ---
    #[test]
    fn test_d_separated_chain_given_middle() {
        let g = chain_graph();
        // X _||_ Z | Y
        assert!(g.d_separated("X", "Z", &["Y"]));
    }

    // --- Test 8: NOT d-separated in chain without conditioning ---
    #[test]
    fn test_not_d_separated_chain_unconditional() {
        let g = chain_graph();
        // X NOT _||_ Z (no conditioning)
        assert!(!g.d_separated("X", "Z", &[]));
    }

    // --- Test 9: backdoor criterion with empty set (no confounders) ---
    #[test]
    fn test_backdoor_empty_set_no_confounders() {
        // Simple graph: T→Y with no confounders
        let mut g = CausalGraph::new(vec!["T".into(), "Y".into()]);
        g.add_edge("T", "Y").unwrap();
        assert!(backdoor_criterion(&g, "T", "Y", &[]));
    }

    // --- Test 10: backdoor criterion satisfied by parent set ---
    #[test]
    fn test_backdoor_parent_set_valid() {
        let g = confounded_graph(); // C→X, C→Y, X→Y
        assert!(backdoor_criterion(&g, "X", "Y", &["C"]));
    }

    // --- Test 11: backdoor criterion invalid with descendant ---
    #[test]
    fn test_backdoor_invalid_with_descendant() {
        // T→M→Y, C→T, C→Y
        let mut g = CausalGraph::new(vec!["C".into(), "T".into(), "M".into(), "Y".into()]);
        g.add_edge("C", "T").unwrap();
        g.add_edge("C", "Y").unwrap();
        g.add_edge("T", "M").unwrap();
        g.add_edge("M", "Y").unwrap();
        // M is a descendant of T — using it in the adjustment set violates criterion
        assert!(!backdoor_criterion(&g, "T", "Y", &["M"]));
    }

    // --- Test 12: find_backdoor_adjustment returns valid set ---
    #[test]
    fn test_find_backdoor_adjustment() {
        let g = confounded_graph();
        let result = find_backdoor_adjustment(&g, "X", "Y").unwrap();
        assert!(result.valid, "Adjustment set should be valid");
        let refs: Vec<&str> = result.adjustment_set.iter().map(|s| s.as_str()).collect();
        assert!(backdoor_criterion(&g, "X", "Y", &refs));
    }

    // --- Test 13: frontdoor criterion basic structure ---
    #[test]
    fn test_frontdoor_criterion() {
        // X→M→Y, X←U→Y (unobserved confounder U is implicit via structure)
        // Classic frontdoor: X→M→Y with no backdoor to M, X blocks backdoor to Y
        let mut g = CausalGraph::new(vec!["X".into(), "M".into(), "Y".into()]);
        g.add_edge("X", "M").unwrap();
        g.add_edge("M", "Y").unwrap();
        assert!(frontdoor_criterion(&g, "X", "Y", &["M"]));
    }

    // --- Test 14: do_intervention removes incoming edges ---
    #[test]
    fn test_do_intervention_removes_incoming() {
        let g = confounded_graph(); // C→X, C→Y, X→Y
        let int = Intervention {
            variable: "X".to_string(),
            value: 1.0,
        };
        let mutilated = do_intervention(&g, &int);
        // C→X should be removed
        let parents = mutilated.parents_of("X");
        assert!(
            parents.is_empty(),
            "After do(X), X should have no parents; got {:?}",
            parents
        );
    }

    // --- Test 15: do_intervention preserves outgoing edges ---
    #[test]
    fn test_do_intervention_preserves_outgoing() {
        let g = confounded_graph();
        let int = Intervention {
            variable: "X".to_string(),
            value: 1.0,
        };
        let mutilated = do_intervention(&g, &int);
        // X→Y should remain
        assert!(mutilated.children_of("X").contains(&"Y".to_string()));
    }

    // --- Test 16: add_sample validates dimension ---
    #[test]
    fn test_add_sample_dimension_check() {
        let mut data = ObservationalData::new(vec!["A".into(), "B".into()]);
        let result = data.add_sample(vec![1.0, 2.0, 3.0]); // wrong size
        assert!(matches!(result, Err(CausalError::DimensionMismatch)));
    }

    // --- Test 17: mean computation ---
    #[test]
    fn test_mean() {
        let mut data = ObservationalData::new(vec!["X".into()]);
        for v in &[1.0_f64, 2.0, 3.0, 4.0] {
            data.add_sample(vec![*v]).unwrap();
        }
        let m = data.mean("X").unwrap();
        assert!((m - 2.5).abs() < 1e-10);
    }

    // --- Test 18: conditional_mean filters correctly ---
    #[test]
    fn test_conditional_mean() {
        let mut data = ObservationalData::new(vec!["T".into(), "Y".into()]);
        for _ in 0..5 {
            data.add_sample(vec![0.0, 0.0]).unwrap();
            data.add_sample(vec![1.0, 4.0]).unwrap();
        }
        let cm = data.conditional_mean("Y", "T", 1.0).unwrap();
        assert!((cm - 4.0).abs() < 1e-10);
        let cm0 = data.conditional_mean("Y", "T", 0.0).unwrap();
        assert!((cm0 - 0.0).abs() < 1e-10);
    }

    // --- Test 19: ate_backdoor estimates correct effect ---
    #[test]
    fn test_ate_backdoor_known_effect() {
        let mut g = CausalGraph::new(vec!["T".into(), "Y".into()]);
        g.add_edge("T", "Y").unwrap();
        let data = simple_treatment_data();
        let result = ate_backdoor(&g, &data, "T", "Y").unwrap();
        assert!(
            (result.ate - 2.0).abs() < 1e-6,
            "Expected ATE~2.0, got {}",
            result.ate
        );
    }

    // --- Test 20: ate_instrumental_variable ---
    #[test]
    fn test_ate_instrumental_variable() {
        // Z→T→Y: Z is instrument, T is treatment, Y is outcome
        // Y = 3*T, T = Z (perfect instrument)
        let mut data = ObservationalData::new(vec!["Z".into(), "T".into(), "Y".into()]);
        for z in &[0.0_f64, 1.0] {
            for _ in 0..50 {
                data.add_sample(vec![*z, *z, 3.0 * z]).unwrap();
            }
        }
        let result = ate_instrumental_variable(&data, "T", "Y", "Z").unwrap();
        assert!(
            (result.ate - 3.0).abs() < 1e-6,
            "Expected IV ATE~3.0, got {}",
            result.ate
        );
    }

    // --- Test 21: ate_backdoor returns NoCausalPath when treatment not connected ---
    #[test]
    fn test_ate_backdoor_no_causal_path() {
        let g = CausalGraph::new(vec!["T".into(), "Y".into()]);
        // No edge T→Y
        let data = simple_treatment_data();
        let result = ate_backdoor(&g, &data, "T", "Y");
        assert!(matches!(result, Err(CausalError::NoCausalPath)));
    }

    // --- Test 22: TreatmentEffect.ate is finite ---
    #[test]
    fn test_treatment_effect_ate_finite() {
        let mut g = CausalGraph::new(vec!["T".into(), "Y".into()]);
        g.add_edge("T", "Y").unwrap();
        let data = simple_treatment_data();
        let result = ate_backdoor(&g, &data, "T", "Y").unwrap();
        assert!(result.ate.is_finite(), "ATE must be finite");
    }

    // --- Test 23: propensity_score returns n_samples scores ---
    #[test]
    fn test_propensity_score_length() {
        let mut data = ObservationalData::new(vec!["X1".into(), "X2".into(), "T".into()]);
        for i in 0..40 {
            let t = if i % 2 == 0 { 1.0 } else { 0.0 };
            data.add_sample(vec![i as f64, (i as f64).sin(), t])
                .unwrap();
        }
        let scores = propensity_score(&data, "T", &["X1", "X2"]).unwrap();
        assert_eq!(scores.len(), 40);
        for &s in &scores {
            assert!(s > 0.0 && s < 1.0, "Score {} must be in (0,1)", s);
        }
    }

    // --- Test 24: CausalError Display ---
    #[test]
    fn test_causal_error_display() {
        let e = CausalError::NodeNotFound("Foo".into());
        let msg = e.to_string();
        assert!(
            msg.contains("Foo"),
            "Error message should mention the node name"
        );

        let e2 = CausalError::CycleDetected;
        assert!(e2.to_string().contains("cycle"));

        let e3 = CausalError::InsufficientData;
        assert!(!e3.to_string().is_empty());

        let e4 = CausalError::NoCausalPath;
        assert!(e4.to_string().contains("causal path"));

        let e5 = CausalError::NumericalError("test".into());
        assert!(e5.to_string().contains("test"));
    }

    // --- Test 25: edge_count ---
    #[test]
    fn test_edge_count() {
        let g = chain_graph();
        assert_eq!(g.edge_count(), 2);
    }

    // --- Test 26: descendants_of ---
    #[test]
    fn test_descendants_of() {
        let g = chain_graph();
        let descs = g.descendants_of("X");
        assert!(descs.contains(&"Y".to_string()));
        assert!(descs.contains(&"Z".to_string()));
    }

    // --- Test 27: add_edge returns error for missing node ---
    #[test]
    fn test_add_edge_missing_node() {
        let mut g = CausalGraph::new(vec!["A".into()]);
        let result = g.add_edge("A", "B");
        assert!(matches!(result, Err(CausalError::NodeNotFound(_))));
    }

    // --- Test 28: ate_backdoor with confounded data ---
    #[test]
    fn test_ate_backdoor_confounded() {
        let g = confounded_graph(); // C→X, C→Y, X→Y
                                    // Build data: C ∈ {0,1}, X=C, Y = 2*X + C (so naive estimate is biased)
        let mut data = ObservationalData::new(vec!["C".into(), "X".into(), "Y".into()]);
        for c in &[0.0_f64, 1.0] {
            for _ in 0..50 {
                let x = *c;
                let y = 2.0 * x + c; // ATE should be 2.0 after adjustment
                data.add_sample(vec![*c, x, y]).unwrap();
            }
        }
        let result = ate_backdoor(&g, &data, "X", "Y").unwrap();
        assert!(result.ate.is_finite());
        assert_eq!(result.estimator, "backdoor");
    }
}
