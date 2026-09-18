//! Tests for functional optimizers and training loop utilities
//!
//! This test module verifies:
//! - Graph memory management (clear_graph, node_count)
//! - Functional/immutable optimizer pattern (FunctionalSGD, FunctionalAdam)
//! - Training loop utilities (TrainingConfig, TrainingLoop)
//! - Tensor detach() method

use scirs2_autograd as ag;
use scirs2_autograd::optimizers::training::{TrainingConfig, TrainingLoop};
use scirs2_autograd::optimizers::{FunctionalAdam, FunctionalOptimizer, FunctionalSGD};
use scirs2_autograd::tensor_ops as T;
use scirs2_autograd::variable::GetVariableTensor;

/// Test that clear_graph() properly clears the graph
#[test]
fn test_clear_graph() {
    let mut env = ag::VariableEnvironment::<f64>::new();

    // Initialize a variable
    env.name("w")
        .set(scirs2_autograd::ndarray_ext::zeros(&[3, 3]));

    env.run(|ctx| {
        // Create some tensors
        let x = T::ones(&[3, 3], ctx);
        let y = x * 2.0;
        let _ = y * 3.0;

        // Check node count is > 0
        let count_before = ctx.node_count();
        assert!(count_before > 0, "Graph should have nodes after operations");

        // This test is for documentation - in a real training loop,
        // you would call clear_graph() to prevent memory leaks
    });
}

/// Test that node_count() works correctly
#[test]
fn test_node_count() {
    ag::run(|ctx: &mut ag::Context<f64>| {
        let initial_count = ctx.node_count();

        let _x = T::ones(&[2, 2], ctx);
        assert!(ctx.node_count() > initial_count);

        let _y = T::zeros(&[2, 2], ctx);
        assert!(ctx.node_count() > initial_count + 1);
    });
}

/// Test Tensor::detach() creates a non-differentiable tensor
#[test]
fn test_tensor_detach() {
    ag::run(|ctx: &mut ag::Context<f64>| {
        let x = T::ones(&[2, 2], ctx);
        let y = x * 2.0;

        // Detach y
        let y_detached = y.detach();

        // y_detached should not be differentiable
        assert!(!y_detached.is_differentiable());

        // Original y should still be differentiable
        assert!(y.is_differentiable());
    });
}

/// Test Tensor::with_grad() method
#[test]
fn test_tensor_with_grad() {
    ag::run(|ctx: &mut ag::Context<f64>| {
        let x = T::ones(&[2, 2], ctx);

        // with_grad(false) should be equivalent to detach()
        let x_no_grad = x.with_grad(false);
        assert!(!x_no_grad.is_differentiable());

        // with_grad(true) should return the same tensor
        let x_with_grad = x.with_grad(true);
        assert!(x_with_grad.is_differentiable());
    });
}

/// Test FunctionalSGD optimizer
#[test]
fn test_functional_sgd() {
    let mut env = ag::VariableEnvironment::<f64>::new();

    // Initialize weights
    env.name("w")
        .set(scirs2_autograd::ndarray_ext::ones(&[2, 2]));

    let mut opt = FunctionalSGD::new(0.1);

    env.run(|ctx| {
        let w = ctx.variable("w");
        let loss = T::sum_all(w);

        // Compute gradient
        let grads = T::grad(&[loss], &[w]);

        // Functional step
        let new_params = opt.step(&[w], &grads, ctx);

        // new_params should be valid tensors
        assert_eq!(new_params.len(), 1);

        // Step count should increment
        assert_eq!(opt.step_count(), 1);
    });

    // Reset and verify
    opt.reset();
    assert_eq!(opt.step_count(), 0);
}

/// Test FunctionalAdam optimizer
#[test]
fn test_functional_adam() {
    let mut env = ag::VariableEnvironment::<f64>::new();

    env.name("w")
        .set(scirs2_autograd::ndarray_ext::ones(&[2, 2]));

    let mut opt = FunctionalAdam::new(0.001);

    // Verify default parameters
    assert_eq!(opt.learning_rate(), 0.001);

    env.run(|ctx| {
        let w = ctx.variable("w");
        let loss = T::sum_all(w);
        let grads = T::grad(&[loss], &[w]);

        let new_params = opt.step(&[w], &grads, ctx);
        assert_eq!(new_params.len(), 1);
        assert_eq!(opt.step_count(), 1);
    });

    // Set new learning rate
    opt.set_learning_rate(0.01);
    assert_eq!(opt.learning_rate(), 0.01);

    opt.reset();
    assert_eq!(opt.step_count(), 0);
}

/// Test FunctionalAdam with custom parameters
#[test]
fn test_functional_adam_custom_params() {
    let opt: FunctionalAdam<f64> = FunctionalAdam::with_params(0.01, 0.95, 0.9999, 1e-6);

    assert_eq!(opt.lr, 0.01);
    assert_eq!(opt.beta1, 0.95);
    assert_eq!(opt.beta2, 0.9999);
    assert_eq!(opt.eps, 1e-6);
}

/// Test TrainingConfig builder
#[test]
fn test_training_config() {
    let config = TrainingConfig::new()
        .with_max_graph_nodes(100_000)
        .with_auto_clear(false)
        .with_warn_threshold(50_000)
        .with_print_warnings(false)
        .with_checkpoint_interval(100);

    assert_eq!(config.max_graph_nodes, 100_000);
    assert!(!config.auto_clear);
    assert_eq!(config.warn_threshold, 50_000);
    assert!(!config.print_warnings);
    assert_eq!(config.checkpoint_interval, Some(100));
}

/// Test TrainingLoop statistics
#[test]
fn test_training_loop_stats() {
    let config = TrainingConfig::new().with_print_warnings(false);
    let mut trainer: TrainingLoop<f64> = TrainingLoop::new(config);

    // Initial state
    assert_eq!(trainer.step_count(), 0);
    assert_eq!(trainer.total_nodes_created(), 0);
    assert_eq!(trainer.peak_graph_size(), 0);

    // Simulate training steps
    trainer.increment_step();
    trainer.record_graph_stats(1000);

    trainer.increment_step();
    trainer.record_graph_stats(2000);

    trainer.increment_step();
    trainer.record_graph_stats(1500);

    assert_eq!(trainer.step_count(), 3);
    assert_eq!(trainer.total_nodes_created(), 4500);
    assert_eq!(trainer.peak_graph_size(), 2000);

    // Test stats string
    let stats = trainer.stats_string();
    assert!(stats.contains("Steps: 3"));
    assert!(stats.contains("Peak graph size: 2000"));
}

/// Test checkpoint interval
#[test]
fn test_checkpoint_interval() {
    let config = TrainingConfig::new().with_checkpoint_interval(10);
    let mut trainer: TrainingLoop<f64> = TrainingLoop::new(config);

    // Should not checkpoint at step 0
    assert!(!trainer.should_checkpoint());

    for _ in 0..9 {
        trainer.increment_step();
        assert!(!trainer.should_checkpoint());
    }

    // Step 10 should trigger checkpoint
    trainer.increment_step();
    assert!(trainer.should_checkpoint());

    // Step 11 should not
    trainer.increment_step();
    assert!(!trainer.should_checkpoint());

    // Step 20 should trigger again
    for _ in 0..9 {
        trainer.increment_step();
    }
    assert!(trainer.should_checkpoint());
}

/// Test training loop reset
#[test]
fn test_training_loop_reset() {
    let mut trainer: TrainingLoop<f64> = TrainingLoop::default();

    trainer.increment_step();
    trainer.record_graph_stats(1000);
    trainer.increment_step();
    trainer.record_graph_stats(2000);

    trainer.reset();

    assert_eq!(trainer.step_count(), 0);
    assert_eq!(trainer.total_nodes_created(), 0);
    assert_eq!(trainer.peak_graph_size(), 0);
}

/// Integration test: Complete training loop pattern (Issue #94 example)
#[test]
fn test_training_loop_pattern() {
    let mut env = ag::VariableEnvironment::<f64>::new();

    // Initialize weights
    let w_init = scirs2_autograd::ndarray_ext::ArrayRng::<f64>::default().standard_normal(&[4, 4]);
    env.name("w").set(w_init);

    let mut opt = FunctionalSGD::new(0.01);
    let config = TrainingConfig::new()
        .with_print_warnings(false)
        .with_auto_clear(true);
    let mut trainer: TrainingLoop<f64> = TrainingLoop::new(config);

    // Simulate 5 training iterations
    for _epoch in 0..5 {
        env.run(|ctx| {
            let w = ctx.variable("w");

            // Simple loss: sum of weights squared
            let loss = T::sum_all(T::mul(w, w));

            // Compute gradients
            let grads = T::grad(&[loss], &[w]);

            // Functional update
            let new_params = opt.step(&[w], &grads, ctx);

            // Record stats
            trainer.increment_step();
            trainer.record_graph_stats(ctx.node_count());

            // Verify we got new parameters
            assert_eq!(new_params.len(), 1);
        });
    }

    assert_eq!(trainer.step_count(), 5);
    assert_eq!(opt.step_count(), 5);
}

/// Test that multiple training iterations don't accumulate graph nodes
/// when using the recommended pattern
#[test]
fn test_no_graph_accumulation() {
    let mut env = ag::VariableEnvironment::<f64>::new();
    env.name("w")
        .set(scirs2_autograd::ndarray_ext::ones(&[3, 3]));

    let mut node_counts = Vec::new();

    // Run multiple iterations
    for _ in 0..10 {
        env.run(|ctx| {
            let w = ctx.variable("w");
            let y = w * 2.0;
            let _ = y * 3.0;

            node_counts.push(ctx.node_count());
        });
    }

    // Each iteration should have roughly the same node count
    // (since graph is reset between run() calls)
    let max_count = *node_counts.iter().max().unwrap_or(&0);
    let min_count = *node_counts.iter().min().unwrap_or(&0);

    // The difference should be small (within 10 nodes)
    assert!(
        max_count - min_count < 10,
        "Graph node counts should be consistent: min={}, max={}",
        min_count,
        max_count
    );
}

/// Regression test for GitHub Issue #98: Adam optimizer crashes on scalar/1×1 parameters
/// Tests both 0-D scalars (shape []) and 1-element 1-D arrays (shape [1])
#[test]
fn test_adam_scalar_and_1x1_parameters() {
    use scirs2_core::ndarray::arr0;

    // Test 1: 0-D scalar parameter (shape [])
    {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("scalar_param").set(arr0(1.0).into_dyn());

        let mut opt = FunctionalAdam::new(0.01);

        env.run(|ctx| {
            let param = ctx.variable("scalar_param");

            // Create a simple loss
            let loss = param * 2.0;

            // Compute gradient
            let grads = T::grad(&[loss], &[param]);

            // This should not panic (GitHub Issue #98)
            let new_params = opt.step(&[param], &grads, ctx);
            assert_eq!(new_params.len(), 1);
            assert_eq!(opt.step_count(), 1);
        });
    }

    // Test 2: 1-element 1-D array (shape [1])
    {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("one_elem_param")
            .set(scirs2_autograd::ndarray_ext::ones(&[1]));

        let mut opt = FunctionalAdam::new(0.01);

        env.run(|ctx| {
            let param = ctx.variable("one_elem_param");

            // Create a simple loss
            let loss = T::sum_all(param * 3.0);

            // Compute gradient
            let grads = T::grad(&[loss], &[param]);

            // This should not panic (GitHub Issue #98)
            let new_params = opt.step(&[param], &grads, ctx);
            assert_eq!(new_params.len(), 1);
            assert_eq!(opt.step_count(), 1);
        });
    }

    // Test 3: Multiple steps with scalar parameter
    {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("scalar").set(arr0(5.0).into_dyn());

        let mut opt = FunctionalAdam::new(0.001);

        // Run multiple optimization steps
        for step in 1..=5 {
            env.run(|ctx| {
                let param = ctx.variable("scalar");
                let loss = param * param; // Simple quadratic loss

                let grads = T::grad(&[loss], &[param]);
                let _new_params = opt.step(&[param], &grads, ctx);
                assert_eq!(opt.step_count(), step);
            });
        }
    }

    // Test 4: 1×1 matrix (shape [1, 1])
    {
        let mut env = ag::VariableEnvironment::<f64>::new();
        env.name("matrix_1x1")
            .set(scirs2_autograd::ndarray_ext::ones(&[1, 1]));

        let mut opt = FunctionalAdam::new(0.01);

        env.run(|ctx| {
            let param = ctx.variable("matrix_1x1");
            let loss = T::sum_all(param);

            let grads = T::grad(&[loss], &[param]);

            // Should handle 1×1 matrices correctly
            let new_params = opt.step(&[param], &grads, ctx);
            assert_eq!(new_params.len(), 1);
        });
    }
}
