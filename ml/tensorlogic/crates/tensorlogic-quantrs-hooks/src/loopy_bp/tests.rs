//! Unit tests for the Loopy Belief Propagation module.

use super::engine::linear_to_assignment;
use super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array, Array1, ArrayD};
use std::collections::HashMap;

use crate::factor::Factor;
use crate::graph::FactorGraph;
use crate::message_passing::MessagePassingAlgorithm;

fn make_chain_graph(n: usize) -> FactorGraph {
    // Chain: X1 - X2 - ... - Xn with pairwise uniform factors.
    let mut g = FactorGraph::new();
    for i in 0..n {
        g.add_variable(format!("x{}", i), "Binary".to_string());
    }
    for i in 0..n - 1 {
        let v1 = format!("x{}", i);
        let v2 = format!("x{}", i + 1);
        let fac = Factor::uniform(format!("f{}_{}", i, i + 1), vec![v1, v2], 2);
        g.add_factor(fac).expect("add factor");
    }
    g
}

fn make_loop_graph() -> FactorGraph {
    // Triangle: X0 - X1 - X2 - X0  (3-cycle)
    let mut g = FactorGraph::new();
    for i in 0..3 {
        g.add_variable(format!("x{}", i), "Binary".to_string());
    }
    // f01, f12, f20
    let pairs = [
        ("x0", "x1", "f01"),
        ("x1", "x2", "f12"),
        ("x2", "x0", "f20"),
    ];
    for (a, b, name) in pairs {
        let fac = Factor::uniform(name.to_string(), vec![a.to_string(), b.to_string()], 2);
        g.add_factor(fac).expect("add factor");
    }
    g
}

// ── LogMessage ────────────────────────────────────────────────────────────

#[test]
fn test_log_message_uniform() {
    let msg = LogMessage::uniform("x", 4);
    assert_eq!(msg.log_values.len(), 4);
    let probs = msg.to_probs();
    assert_abs_diff_eq!(probs.sum(), 1.0, epsilon = 1e-10);
    for &p in probs.iter() {
        assert_abs_diff_eq!(p, 0.25, epsilon = 1e-10);
    }
}

#[test]
fn test_log_message_normalise() {
    let mut msg = LogMessage {
        variable: "x".to_string(),
        log_values: Array1::from(vec![0.0, 1.0, 2.0, 3.0]),
    };
    msg.log_normalise();
    let probs = msg.to_probs();
    assert_abs_diff_eq!(probs.sum(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_log_message_damping() {
    let old = LogMessage::uniform("x", 2);
    let mut new_msg = LogMessage {
        variable: "x".to_string(),
        log_values: Array1::from(vec![0.0_f64.ln(), f64::NEG_INFINITY]),
    };
    new_msg.log_normalise();
    let damped = new_msg.damp(&old, 0.5);
    let probs = damped.to_probs();
    assert_abs_diff_eq!(probs.sum(), 1.0, epsilon = 1e-10);
    // With 50% damping toward uniform, both probabilities should be between 0 and 1.
    assert!(probs[0] > 0.0 && probs[0] < 1.0);
    assert!(probs[1] > 0.0 && probs[1] < 1.0);
}

#[test]
fn test_log_message_residual() {
    let a = LogMessage::uniform("x", 2);
    let b = LogMessage::uniform("x", 2);
    assert_abs_diff_eq!(a.residual_linf(&b), 0.0, epsilon = 1e-10);
}

// ── LbpDampingPolicy ─────────────────────────────────────────────────────

#[test]
fn test_damping_none() {
    let pol = LbpDampingPolicy::None;
    assert_abs_diff_eq!(pol.effective_lambda(0.5), 1.0, epsilon = 1e-10);
}

#[test]
fn test_damping_uniform() {
    let pol = LbpDampingPolicy::Uniform(0.7);
    assert_abs_diff_eq!(pol.effective_lambda(0.0), 0.7, epsilon = 1e-10);
    assert_abs_diff_eq!(pol.effective_lambda(1000.0), 0.7, epsilon = 1e-10);
}

#[test]
fn test_damping_adaptive() {
    let pol = LbpDampingPolicy::Adaptive { base_lambda: 0.1 };
    let lam_small = pol.effective_lambda(0.0);
    let lam_large = pol.effective_lambda(10.0);
    // Small residual → lambda closer to 1 (less damping).
    // Large residual → lambda closer to base (more damping).
    assert!(lam_small > lam_large);
    assert!(lam_large >= 0.1);
    assert!(lam_small <= 1.0);
}

// ── CycleDetector ────────────────────────────────────────────────────────

#[test]
fn test_cycle_detector_tree() {
    let g = make_chain_graph(4);
    let ca = CycleDetector::new(&g).analyse();
    assert!(!ca.has_cycles);
    assert_eq!(ca.girth, None);
    assert!(ca.is_tree);
    assert_eq!(ca.cycle_rank, 0);
}

#[test]
fn test_cycle_detector_loop() {
    let g = make_loop_graph();
    let ca = CycleDetector::new(&g).analyse();
    assert!(ca.has_cycles);
    assert!(ca.cycle_rank > 0);
}

#[test]
fn test_cycle_detector_empty() {
    let g = FactorGraph::new();
    let ca = CycleDetector::new(&g).analyse();
    assert!(!ca.has_cycles);
    assert_eq!(ca.num_components, 0);
}

// ── LoopyBeliefPropagation — synchronous ─────────────────────────────────

#[test]
fn test_lbp_synchronous_chain() {
    let g = make_chain_graph(3);
    let lbp = LoopyBeliefPropagation::new(
        LoopyBpConfig::default().with_schedule(UpdateSchedule::Synchronous),
    );
    let result = lbp.run_full(&g).expect("LBP failed");
    // Beliefs should sum to 1 for every variable.
    for belief in result.beliefs.values() {
        assert_abs_diff_eq!(belief.sum(), 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_lbp_synchronous_loop() {
    let g = make_loop_graph();
    let config = LoopyBpConfig::default()
        .with_schedule(UpdateSchedule::Synchronous)
        .with_max_iterations(500)
        .with_tolerance(1e-5)
        .with_damping(LbpDampingPolicy::Uniform(0.5));
    let lbp = LoopyBeliefPropagation::new(config);
    let result = lbp.run_full(&g).expect("LBP on loop failed");
    // Beliefs should still be valid probability distributions.
    for belief in result.beliefs.values() {
        assert_abs_diff_eq!(belief.sum(), 1.0, epsilon = 1e-5);
        for &p in belief.iter() {
            assert!(p >= 0.0);
        }
    }
}

// ── LoopyBeliefPropagation — sequential ──────────────────────────────────

#[test]
fn test_lbp_sequential_chain() {
    let g = make_chain_graph(3);
    let lbp = LoopyBeliefPropagation::new(
        LoopyBpConfig::default().with_schedule(UpdateSchedule::Sequential),
    );
    let result = lbp.run_full(&g).expect("LBP sequential failed");
    for belief in result.beliefs.values() {
        assert_abs_diff_eq!(belief.sum(), 1.0, epsilon = 1e-6);
    }
}

// ── LoopyBeliefPropagation — residual ────────────────────────────────────

#[test]
fn test_lbp_residual_chain() {
    let g = make_chain_graph(3);
    let lbp = LoopyBeliefPropagation::new(
        LoopyBpConfig::default()
            .with_schedule(UpdateSchedule::Residual)
            .with_damping(LbpDampingPolicy::None),
    );
    let result = lbp.run_full(&g).expect("Residual LBP failed");
    for belief in result.beliefs.values() {
        assert_abs_diff_eq!(belief.sum(), 1.0, epsilon = 1e-6);
    }
}

// ── MessagePassingAlgorithm trait ────────────────────────────────────────

#[test]
fn test_lbp_trait_interface() {
    let g = make_chain_graph(2);
    let lbp = LoopyBeliefPropagation::new(LoopyBpConfig::default());
    assert_eq!(lbp.name(), "LoopyBeliefPropagation");
    let beliefs = lbp.run(&g).expect("trait run failed");
    assert_eq!(beliefs.len(), 2);
}

// ── Bethe free energy ────────────────────────────────────────────────────

#[test]
fn test_bethe_free_energy_single_variable() {
    let mut g = FactorGraph::new();
    g.add_variable("x".to_string(), "Binary".to_string());

    let mut beliefs_var = HashMap::new();
    beliefs_var.insert("x".to_string(), Array1::from(vec![0.5, 0.5]));
    let beliefs_fac: HashMap<String, ArrayD<f64>> = HashMap::new();

    let bfe = bethe_free_energy(&g, &beliefs_var, &beliefs_fac);
    // Factor energy is 0 (no factors), variable entropy = (1-0) * ∑ b ln b = ln(0.5).
    assert!(bfe.total.is_finite());
}

#[test]
fn test_bethe_included_in_result() {
    let g = make_chain_graph(2);
    let config = LoopyBpConfig {
        compute_bethe: true,
        ..Default::default()
    };
    let lbp = LoopyBeliefPropagation::new(config);
    let result = lbp.run_full(&g).expect("LBP failed");
    assert!(result.bethe.is_some());
    let bfe = result.bethe.expect("Bethe missing");
    assert!(bfe.total.is_finite());
}

// ── linear_to_assignment ─────────────────────────────────────────────────

#[test]
fn test_linear_to_assignment() {
    let shape = [2, 3];
    assert_eq!(linear_to_assignment(0, &shape), vec![0, 0]);
    assert_eq!(linear_to_assignment(1, &shape), vec![0, 1]);
    assert_eq!(linear_to_assignment(2, &shape), vec![0, 2]);
    assert_eq!(linear_to_assignment(3, &shape), vec![1, 0]);
    assert_eq!(linear_to_assignment(5, &shape), vec![1, 2]);
}

// ── Convergence monitor ──────────────────────────────────────────────────

#[test]
fn test_convergence_monitor_detects_convergence() {
    let mut mon = LbpConvergenceMonitor::new();
    mon.record(
        LbpIterStats {
            iteration: 0,
            max_residual: 0.1,
            mean_residual: 0.05,
            active_messages: 10,
        },
        1e-3,
    );
    assert!(!mon.is_converged());
    mon.record(
        LbpIterStats {
            iteration: 1,
            max_residual: 1e-7,
            mean_residual: 5e-8,
            active_messages: 0,
        },
        1e-3,
    );
    assert!(mon.is_converged());
    assert_eq!(mon.converged_at, Some(1));
}

#[test]
fn test_convergence_monitor_last_residual() {
    let mut mon = LbpConvergenceMonitor::new();
    assert_eq!(mon.last_residual(), f64::INFINITY);
    mon.record(
        LbpIterStats {
            iteration: 0,
            max_residual: 0.5,
            mean_residual: 0.25,
            active_messages: 5,
        },
        1e-6,
    );
    assert_abs_diff_eq!(mon.last_residual(), 0.5, epsilon = 1e-10);
}

// ── Non-uniform factor — bias test ────────────────────────────────────────

#[test]
fn test_lbp_biased_single_factor() {
    // Single factor strongly biasing X=0.
    let mut g = FactorGraph::new();
    g.add_variable("x".to_string(), "Binary".to_string());
    let vals = Array::from_shape_vec(vec![2], vec![0.9, 0.1])
        .expect("shape vec")
        .into_dyn();
    let fac = Factor::new("f".to_string(), vec!["x".to_string()], vals).expect("factor");
    g.add_factor(fac).expect("add factor");

    let lbp = LoopyBeliefPropagation::new(LoopyBpConfig::default());
    let result = lbp.run_full(&g).expect("LBP biased failed");
    let belief_x = result.beliefs.get("x").expect("x belief");
    // The belief should be close to [0.9, 0.1].
    assert_abs_diff_eq!(belief_x.sum(), 1.0, epsilon = 1e-6);
    assert!(belief_x[0] > belief_x[1], "X=0 should dominate");
}

// ── Adaptive damping ─────────────────────────────────────────────────────

#[test]
fn test_lbp_adaptive_damping() {
    let g = make_loop_graph();
    let config = LoopyBpConfig::default()
        .with_damping(LbpDampingPolicy::Adaptive { base_lambda: 0.3 })
        .with_schedule(UpdateSchedule::Synchronous)
        .with_max_iterations(300);
    let lbp = LoopyBeliefPropagation::new(config);
    let result = lbp.run_full(&g).expect("LBP adaptive damping failed");
    for belief in result.beliefs.values() {
        assert_abs_diff_eq!(belief.sum(), 1.0, epsilon = 1e-5);
    }
}

// ── LoopyBpConfig builder ────────────────────────────────────────────────

#[test]
fn test_config_builder() {
    let cfg = LoopyBpConfig::new()
        .with_max_iterations(50)
        .with_tolerance(1e-4)
        .with_damping(LbpDampingPolicy::None)
        .with_schedule(UpdateSchedule::Sequential);
    assert_eq!(cfg.max_iterations, 50);
    assert_abs_diff_eq!(cfg.tolerance, 1e-4, epsilon = 1e-15);
}
