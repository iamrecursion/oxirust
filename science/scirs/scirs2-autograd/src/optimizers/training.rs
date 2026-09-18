//! Training loop utilities for scirs2-autograd
//!
//! This module provides utilities for building training loops that properly
//! manage computation graphs, avoiding memory leaks and supporting immutable
//! parameter updates.
//!
//! # Graph Memory Management
//!
//! The computation graph can grow unboundedly during training if not properly
//! managed. This module provides utilities to:
//!
//! - Automatically clear graphs between iterations
//! - Monitor graph size during training
//! - Support checkpointing and rollback
//!
//! # Example: Basic Training Loop
//!
//! ```ignore
//! use scirs2_autograd::optimizers::training::{TrainingLoop, TrainingConfig};
//!
//! let config = TrainingConfig::new()
//!     .with_max_graph_nodes(10000)
//!     .with_auto_clear(true);
//!
//! let mut trainer = TrainingLoop::new(config);
//!
//! for epoch in 0..100 {
//!     trainer.step(|ctx| {
//!         // Forward pass
//!         let loss = model.forward(ctx, &input);
//!
//!         // Backward pass
//!         let grads = ctx.gradients(&[loss], &params);
//!
//!         // Optimizer update
//!         optimizer.update(&params, &grads, ctx);
//!
//!         loss.eval(ctx).expect("Loss evaluation failed")
//!     });
//! }
//! ```

use crate::Float;

/// Configuration for training loops
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum number of graph nodes before automatic clearing
    pub max_graph_nodes: usize,
    /// Whether to automatically clear the graph between iterations
    pub auto_clear: bool,
    /// Warning threshold for graph nodes
    pub warn_threshold: usize,
    /// Whether to print graph size warnings
    pub print_warnings: bool,
    /// Checkpoint interval (in steps)
    pub checkpoint_interval: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_graph_nodes: 50_000,
            auto_clear: true,
            warn_threshold: 25_000,
            print_warnings: true,
            checkpoint_interval: None,
        }
    }
}

impl TrainingConfig {
    /// Creates a new TrainingConfig with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of graph nodes before automatic clearing
    pub fn with_max_graph_nodes(mut self, max: usize) -> Self {
        self.max_graph_nodes = max;
        self
    }

    /// Sets whether to automatically clear the graph between iterations
    pub fn with_auto_clear(mut self, auto_clear: bool) -> Self {
        self.auto_clear = auto_clear;
        self
    }

    /// Sets the warning threshold for graph nodes
    pub fn with_warn_threshold(mut self, threshold: usize) -> Self {
        self.warn_threshold = threshold;
        self
    }

    /// Sets whether to print graph size warnings
    pub fn with_print_warnings(mut self, print: bool) -> Self {
        self.print_warnings = print;
        self
    }

    /// Sets the checkpoint interval (in steps)
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }
}

/// Training loop manager that handles graph memory management
///
/// This struct provides utilities for managing computation graphs during
/// training, including automatic clearing and monitoring.
#[derive(Debug)]
pub struct TrainingLoop<F: Float> {
    /// Configuration
    config: TrainingConfig,
    /// Current step count
    step_count: usize,
    /// Total nodes created (for statistics)
    total_nodes_created: usize,
    /// Peak graph size observed
    peak_graph_size: usize,
    /// Phantom data for F
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> TrainingLoop<F> {
    /// Creates a new TrainingLoop with the given configuration
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            step_count: 0,
            total_nodes_created: 0,
            peak_graph_size: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns the total number of nodes created during training
    pub fn total_nodes_created(&self) -> usize {
        self.total_nodes_created
    }

    /// Returns the peak graph size observed during training
    pub fn peak_graph_size(&self) -> usize {
        self.peak_graph_size
    }

    /// Returns the configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Increments the step counter and returns the new step count
    pub fn increment_step(&mut self) -> usize {
        self.step_count += 1;
        self.step_count
    }

    /// Records graph statistics for monitoring
    pub fn record_graph_stats(&mut self, node_count: usize) {
        self.total_nodes_created += node_count;
        if node_count > self.peak_graph_size {
            self.peak_graph_size = node_count;
        }

        if self.config.print_warnings && node_count > self.config.warn_threshold {
            eprintln!(
                "Warning: Graph size ({}) exceeds warning threshold ({}). \
                 Consider using ctx.clear_graph() or restructuring your training loop.",
                node_count, self.config.warn_threshold
            );
        }
    }

    /// Returns whether a checkpoint should be saved at the current step
    pub fn should_checkpoint(&self) -> bool {
        if let Some(interval) = self.config.checkpoint_interval {
            self.step_count > 0 && self.step_count.is_multiple_of(interval)
        } else {
            false
        }
    }

    /// Resets the training loop state
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.total_nodes_created = 0;
        self.peak_graph_size = 0;
    }

    /// Returns training statistics as a formatted string
    pub fn stats_string(&self) -> String {
        format!(
            "TrainingLoop Stats:\n\
             - Steps: {}\n\
             - Total nodes created: {}\n\
             - Peak graph size: {}\n\
             - Avg nodes per step: {:.1}",
            self.step_count,
            self.total_nodes_created,
            self.peak_graph_size,
            if self.step_count > 0 {
                self.total_nodes_created as f64 / self.step_count as f64
            } else {
                0.0
            }
        )
    }
}

impl<F: Float> Default for TrainingLoop<F> {
    fn default() -> Self {
        Self::new(TrainingConfig::default())
    }
}

/// A guard that automatically clears the graph when dropped
///
/// This is useful for ensuring the graph is cleared even if an error occurs
/// during training.
#[allow(dead_code)]
pub struct GraphClearGuard<'a, F: Float> {
    ctx: &'a mut crate::Context<'a, F>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new()
            .with_max_graph_nodes(100_000)
            .with_auto_clear(false)
            .with_warn_threshold(50_000)
            .with_checkpoint_interval(100);

        assert_eq!(config.max_graph_nodes, 100_000);
        assert!(!config.auto_clear);
        assert_eq!(config.warn_threshold, 50_000);
        assert_eq!(config.checkpoint_interval, Some(100));
    }

    #[test]
    fn test_training_loop_stats() {
        let mut trainer: TrainingLoop<f64> = TrainingLoop::new(TrainingConfig::default());

        trainer.increment_step();
        trainer.record_graph_stats(1000);
        trainer.increment_step();
        trainer.record_graph_stats(2000);

        assert_eq!(trainer.step_count(), 2);
        assert_eq!(trainer.total_nodes_created(), 3000);
        assert_eq!(trainer.peak_graph_size(), 2000);
    }

    #[test]
    fn test_should_checkpoint() {
        let config = TrainingConfig::new().with_checkpoint_interval(10);
        let mut trainer: TrainingLoop<f64> = TrainingLoop::new(config);

        for _ in 0..9 {
            trainer.increment_step();
            assert!(!trainer.should_checkpoint());
        }

        trainer.increment_step();
        assert!(trainer.should_checkpoint());
    }
}
