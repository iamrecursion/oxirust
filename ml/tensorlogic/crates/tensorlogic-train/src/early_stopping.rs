//! Early stopping monitor for training loops.
//!
//! Provides configurable early stopping based on metric monitoring,
//! multi-metric policies, plateau detection, and training progress tracking.

use std::collections::HashMap;

/// Configuration for early stopping.
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of epochs to wait after last improvement before stopping.
    pub patience: usize,
    /// Minimum change to qualify as an improvement.
    pub min_delta: f64,
    /// Whether to minimize (loss) or maximize (accuracy) the metric.
    pub mode: MonitorMode,
    /// If `Some`, the metric must beat this baseline to count as improvement.
    pub baseline: Option<f64>,
    /// Whether to signal restoring the best model state on stop.
    pub restore_best: bool,
    /// Minimum number of epochs before early stopping can trigger.
    pub min_epochs: usize,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            mode: MonitorMode::Minimize,
            baseline: None,
            restore_best: true,
            min_epochs: 1,
        }
    }
}

/// Whether we want to minimize (loss) or maximize (accuracy) the metric.
#[derive(Debug, Clone, PartialEq)]
pub enum MonitorMode {
    /// Improvement means the metric is decreasing (e.g. loss).
    Minimize,
    /// Improvement means the metric is increasing (e.g. accuracy).
    Maximize,
}

/// Decision returned by the early stopping monitor after each step.
#[derive(Debug, Clone, PartialEq)]
pub enum EarlyStoppingDecision {
    /// Keep training, no special event.
    Continue,
    /// Training should stop.
    Stop {
        /// Human-readable reason for stopping.
        reason: String,
    },
    /// A new best metric value was observed (training continues).
    NewBest {
        /// The new best metric value.
        value: f64,
        /// The epoch at which the new best was observed.
        epoch: usize,
    },
}

/// The early stopping monitor.
///
/// Tracks a single metric over epochs and decides when to stop training
/// based on patience, minimum delta, baseline, and minimum epoch constraints.
#[derive(Debug, Clone)]
pub struct EarlyStoppingMonitor {
    config: EarlyStoppingConfig,
    best_value: Option<f64>,
    best_epoch: usize,
    epochs_without_improvement: usize,
    current_epoch: usize,
    history: Vec<f64>,
    stopped: bool,
}

impl EarlyStoppingMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            best_value: None,
            best_epoch: 0,
            epochs_without_improvement: 0,
            current_epoch: 0,
            history: Vec::new(),
            stopped: false,
        }
    }

    /// Create a new monitor with default configuration.
    pub fn with_default() -> Self {
        Self::new(EarlyStoppingConfig::default())
    }

    /// Report the metric for the current epoch and return a decision.
    ///
    /// This advances the epoch counter, records the metric value,
    /// and evaluates whether training should continue or stop.
    pub fn step(&mut self, metric_value: f64) -> EarlyStoppingDecision {
        self.current_epoch += 1;
        self.history.push(metric_value);

        // Check baseline constraint: if a baseline is set and the metric
        // doesn't beat it, we don't consider it an improvement.
        if let Some(baseline) = self.config.baseline {
            let beats_baseline = match self.config.mode {
                MonitorMode::Minimize => metric_value < baseline,
                MonitorMode::Maximize => metric_value > baseline,
            };
            if !beats_baseline {
                self.epochs_without_improvement += 1;
                return self.evaluate_stop();
            }
        }

        // Check if this is an improvement over the best value seen so far.
        let is_new_best = match self.best_value {
            None => true,
            Some(best) => self.is_improvement(metric_value, best),
        };

        if is_new_best {
            self.best_value = Some(metric_value);
            self.best_epoch = self.current_epoch;
            self.epochs_without_improvement = 0;
            EarlyStoppingDecision::NewBest {
                value: metric_value,
                epoch: self.current_epoch,
            }
        } else {
            self.epochs_without_improvement += 1;
            self.evaluate_stop()
        }
    }

    /// Evaluate whether training should stop based on patience and min_epochs.
    fn evaluate_stop(&mut self) -> EarlyStoppingDecision {
        if self.current_epoch < self.config.min_epochs {
            return EarlyStoppingDecision::Continue;
        }

        if self.epochs_without_improvement >= self.config.patience {
            self.stopped = true;
            let best_str = self
                .best_value
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "N/A".to_string());
            EarlyStoppingDecision::Stop {
                reason: format!(
                    "No improvement for {} epochs. Best value: {} at epoch {}.",
                    self.config.patience, best_str, self.best_epoch
                ),
            }
        } else {
            EarlyStoppingDecision::Continue
        }
    }

    /// Check if training should stop (without advancing epoch).
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Best metric value seen so far.
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Epoch at which the best value was seen.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Current epoch number.
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Number of epochs since last improvement.
    pub fn epochs_since_improvement(&self) -> usize {
        self.epochs_without_improvement
    }

    /// Full metric history.
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Reset the monitor to its initial state.
    pub fn reset(&mut self) {
        self.best_value = None;
        self.best_epoch = 0;
        self.epochs_without_improvement = 0;
        self.current_epoch = 0;
        self.history.clear();
        self.stopped = false;
    }

    /// Whether the current value is an improvement over the best.
    ///
    /// For `Minimize` mode: `current < best - min_delta`.
    /// For `Maximize` mode: `current > best + min_delta`.
    fn is_improvement(&self, current: f64, best: f64) -> bool {
        match self.config.mode {
            MonitorMode::Minimize => current < best - self.config.min_delta,
            MonitorMode::Maximize => current > best + self.config.min_delta,
        }
    }

    /// Return a human-readable summary of the monitor's current state.
    pub fn summary(&self) -> String {
        let best_str = self
            .best_value
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "N/A".to_string());
        let mode_str = match self.config.mode {
            MonitorMode::Minimize => "minimize",
            MonitorMode::Maximize => "maximize",
        };
        format!(
            "EarlyStoppingMonitor(mode={}, epoch={}, best={} at epoch {}, \
             patience={}/{}, stopped={})",
            mode_str,
            self.current_epoch,
            best_str,
            self.best_epoch,
            self.epochs_without_improvement,
            self.config.patience,
            self.stopped,
        )
    }
}

/// Policy for combining decisions from multiple metric monitors.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiMetricPolicy {
    /// Stop when ALL monitored metrics signal stop.
    All,
    /// Stop when ANY monitored metric signals stop.
    Any,
}

/// A multi-metric early stopping monitor.
///
/// Monitors multiple metrics simultaneously and applies a policy
/// (`All` or `Any`) to decide when to stop training.
#[derive(Debug, Clone)]
pub struct MultiMetricMonitor {
    monitors: Vec<(String, EarlyStoppingMonitor)>,
    policy: MultiMetricPolicy,
}

impl MultiMetricMonitor {
    /// Create a new multi-metric monitor with the given policy.
    pub fn new(policy: MultiMetricPolicy) -> Self {
        Self {
            monitors: Vec::new(),
            policy,
        }
    }

    /// Register a new metric to monitor.
    pub fn add_metric(&mut self, name: impl Into<String>, config: EarlyStoppingConfig) {
        let name = name.into();
        let monitor = EarlyStoppingMonitor::new(config);
        self.monitors.push((name, monitor));
    }

    /// Report values for all metrics and return a combined decision.
    ///
    /// Keys in `values` must match registered metric names.
    /// Metrics not present in `values` are skipped for that step.
    pub fn step(&mut self, values: &[(String, f64)]) -> EarlyStoppingDecision {
        let value_map: HashMap<&str, f64> = values.iter().map(|(k, v)| (k.as_str(), *v)).collect();

        let mut decisions = Vec::new();

        for (name, monitor) in &mut self.monitors {
            if let Some(&val) = value_map.get(name.as_str()) {
                let decision = monitor.step(val);
                decisions.push(decision);
            }
        }

        // Apply the policy to decide the combined outcome.
        let any_stop = decisions
            .iter()
            .any(|d| matches!(d, EarlyStoppingDecision::Stop { .. }));
        let all_stop = !decisions.is_empty()
            && decisions
                .iter()
                .all(|d| matches!(d, EarlyStoppingDecision::Stop { .. }));

        let should_stop = match self.policy {
            MultiMetricPolicy::All => all_stop,
            MultiMetricPolicy::Any => any_stop,
        };

        if should_stop {
            // Collect reasons from all monitors that signaled stop.
            let reasons: Vec<String> = decisions
                .into_iter()
                .filter_map(|d| {
                    if let EarlyStoppingDecision::Stop { reason } = d {
                        Some(reason)
                    } else {
                        None
                    }
                })
                .collect();
            EarlyStoppingDecision::Stop {
                reason: reasons.join("; "),
            }
        } else {
            // Check if any monitor found a new best.
            let new_best = decisions
                .iter()
                .find(|d| matches!(d, EarlyStoppingDecision::NewBest { .. }));
            match new_best {
                Some(EarlyStoppingDecision::NewBest { value, epoch }) => {
                    EarlyStoppingDecision::NewBest {
                        value: *value,
                        epoch: *epoch,
                    }
                }
                _ => EarlyStoppingDecision::Continue,
            }
        }
    }

    /// Get an individual monitor by name.
    pub fn get_monitor(&self, name: &str) -> Option<&EarlyStoppingMonitor> {
        self.monitors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m)
    }

    /// Number of registered metrics.
    pub fn num_metrics(&self) -> usize {
        self.monitors.len()
    }

    /// Summary across all monitors.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!(
            "MultiMetricMonitor(policy={:?}, metrics={})",
            self.policy,
            self.monitors.len()
        ));
        for (name, monitor) in &self.monitors {
            parts.push(format!("  {}: {}", name, monitor.summary()));
        }
        parts.join("\n")
    }
}

/// Plateau detector: detects when a metric has plateaued.
///
/// A plateau is detected when the variance of the most recent values
/// within a sliding window falls below a configurable threshold.
#[derive(Debug, Clone)]
pub struct PlateauDetector {
    /// Size of the sliding window.
    pub window_size: usize,
    /// Variance threshold below which a plateau is declared.
    pub variance_threshold: f64,
    history: Vec<f64>,
}

impl PlateauDetector {
    /// Create a new plateau detector.
    pub fn new(window_size: usize, variance_threshold: f64) -> Self {
        Self {
            window_size,
            variance_threshold,
            history: Vec::new(),
        }
    }

    /// Push a new metric value.
    pub fn push(&mut self, value: f64) {
        self.history.push(value);
    }

    /// Whether the metric has plateaued (window full and variance below threshold).
    pub fn is_plateau(&self) -> bool {
        if self.history.len() < self.window_size {
            return false;
        }
        match self.current_variance() {
            Some(var) => var < self.variance_threshold,
            None => false,
        }
    }

    /// Compute the variance of the values in the current window.
    ///
    /// Returns `None` if there are fewer values than the window size.
    pub fn current_variance(&self) -> Option<f64> {
        if self.history.len() < self.window_size {
            return None;
        }
        let window = self.values_in_window();
        let n = window.len() as f64;
        if n < 1.0 {
            return None;
        }
        let mean = window.iter().sum::<f64>() / n;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Some(variance)
    }

    /// The values currently in the sliding window.
    pub fn values_in_window(&self) -> &[f64] {
        if self.history.len() < self.window_size {
            &self.history
        } else {
            &self.history[self.history.len() - self.window_size..]
        }
    }
}

/// Training progress tracker.
///
/// Combines epoch tracking with multi-metric recording and progress reporting.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    /// Total number of planned epochs.
    pub total_epochs: usize,
    /// Current epoch (0-indexed, incremented by `advance_epoch`).
    pub current_epoch: usize,
    /// Recorded metrics keyed by name, each with a history of values.
    pub metrics: HashMap<String, Vec<f64>>,
}

impl TrainingProgress {
    /// Create a new training progress tracker.
    pub fn new(total_epochs: usize) -> Self {
        Self {
            total_epochs,
            current_epoch: 0,
            metrics: HashMap::new(),
        }
    }

    /// Record a metric value for the current epoch.
    pub fn record(&mut self, metric_name: impl Into<String>, value: f64) {
        self.metrics
            .entry(metric_name.into())
            .or_default()
            .push(value);
    }

    /// Fraction of training completed (current / total).
    pub fn progress_fraction(&self) -> f64 {
        if self.total_epochs == 0 {
            return 0.0;
        }
        self.current_epoch as f64 / self.total_epochs as f64
    }

    /// Advance to the next epoch.
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Get the latest recorded value for a metric.
    pub fn latest(&self, metric_name: &str) -> Option<f64> {
        self.metrics
            .get(metric_name)
            .and_then(|v| v.last().copied())
    }

    /// Get the best (min or max) recorded value for a metric.
    pub fn best(&self, metric_name: &str, mode: &MonitorMode) -> Option<f64> {
        self.metrics.get(metric_name).and_then(|values| {
            if values.is_empty() {
                return None;
            }
            match mode {
                MonitorMode::Minimize => values
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                MonitorMode::Maximize => values
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            }
        })
    }

    /// Human-readable summary of training progress.
    pub fn summary(&self) -> String {
        let pct = self.progress_fraction() * 100.0;
        let mut parts = vec![format!(
            "TrainingProgress: epoch {}/{} ({:.1}%)",
            self.current_epoch, self.total_epochs, pct
        )];
        for (name, values) in &self.metrics {
            let latest = values.last().map(|v| format!("{v:.6}")).unwrap_or_default();
            parts.push(format!(
                "  {}: latest={}, entries={}",
                name,
                latest,
                values.len()
            ));
        }
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_config_default() {
        let config = EarlyStoppingConfig::default();
        assert_eq!(config.patience, 10);
        assert_eq!(config.min_delta, 0.0);
        assert_eq!(config.mode, MonitorMode::Minimize);
        assert!(config.baseline.is_none());
        assert!(config.restore_best);
        assert_eq!(config.min_epochs, 1);
    }

    #[test]
    fn test_monitor_new_best_on_first_step() {
        let mut monitor = EarlyStoppingMonitor::with_default();
        let decision = monitor.step(1.0);
        assert_eq!(
            decision,
            EarlyStoppingDecision::NewBest {
                value: 1.0,
                epoch: 1
            }
        );
    }

    #[test]
    fn test_monitor_continue_while_improving() {
        let config = EarlyStoppingConfig {
            patience: 3,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        // Decreasing loss values = improvement in Minimize mode.
        let d1 = monitor.step(1.0);
        assert!(matches!(d1, EarlyStoppingDecision::NewBest { .. }));

        let d2 = monitor.step(0.8);
        assert!(matches!(d2, EarlyStoppingDecision::NewBest { .. }));

        let d3 = monitor.step(0.6);
        assert!(matches!(d3, EarlyStoppingDecision::NewBest { .. }));

        let d4 = monitor.step(0.4);
        assert!(matches!(d4, EarlyStoppingDecision::NewBest { .. }));
    }

    #[test]
    fn test_monitor_stop_after_patience() {
        let config = EarlyStoppingConfig {
            patience: 3,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        // First step sets the best.
        monitor.step(1.0);

        // No improvement for 3 epochs.
        let d1 = monitor.step(1.5);
        assert_eq!(d1, EarlyStoppingDecision::Continue);

        let d2 = monitor.step(1.5);
        assert_eq!(d2, EarlyStoppingDecision::Continue);

        let d3 = monitor.step(1.5);
        assert!(matches!(d3, EarlyStoppingDecision::Stop { .. }));
        assert!(monitor.should_stop());
    }

    #[test]
    fn test_monitor_min_delta_threshold() {
        let config = EarlyStoppingConfig {
            patience: 2,
            min_delta: 0.1,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        // Best = 1.0
        monitor.step(1.0);

        // 0.95 is not < 1.0 - 0.1 = 0.9, so NOT an improvement.
        let d = monitor.step(0.95);
        assert_eq!(d, EarlyStoppingDecision::Continue);

        // 0.89 IS < 0.9, so it IS an improvement.
        let d = monitor.step(0.89);
        assert!(matches!(d, EarlyStoppingDecision::NewBest { .. }));
    }

    #[test]
    fn test_monitor_maximize_mode() {
        let config = EarlyStoppingConfig {
            patience: 3,
            mode: MonitorMode::Maximize,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        let d1 = monitor.step(0.5);
        assert!(matches!(d1, EarlyStoppingDecision::NewBest { .. }));

        let d2 = monitor.step(0.7);
        assert!(matches!(d2, EarlyStoppingDecision::NewBest { .. }));

        let d3 = monitor.step(0.9);
        assert!(matches!(d3, EarlyStoppingDecision::NewBest { .. }));

        // Metric goes down — not an improvement in Maximize mode.
        let d4 = monitor.step(0.8);
        assert_eq!(d4, EarlyStoppingDecision::Continue);
    }

    #[test]
    fn test_monitor_baseline_required() {
        let config = EarlyStoppingConfig {
            patience: 5,
            baseline: Some(0.5),
            mode: MonitorMode::Minimize,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        // 0.8 does NOT beat baseline of 0.5 (for Minimize, need < 0.5).
        let d = monitor.step(0.8);
        assert_eq!(d, EarlyStoppingDecision::Continue);
        assert!(monitor.best_value().is_none());

        // 0.4 DOES beat baseline.
        let d = monitor.step(0.4);
        assert!(matches!(d, EarlyStoppingDecision::NewBest { .. }));
    }

    #[test]
    fn test_monitor_min_epochs_prevents_early_stop() {
        let config = EarlyStoppingConfig {
            patience: 1,
            min_epochs: 5,
            ..Default::default()
        };
        let mut monitor = EarlyStoppingMonitor::new(config);

        // First step sets best.
        monitor.step(1.0);

        // No improvement, but we're under min_epochs (5), so Continue.
        let d = monitor.step(2.0);
        assert_eq!(d, EarlyStoppingDecision::Continue);

        let d = monitor.step(2.0);
        assert_eq!(d, EarlyStoppingDecision::Continue);

        let d = monitor.step(2.0);
        assert_eq!(d, EarlyStoppingDecision::Continue);

        // Epoch 5 — now min_epochs is satisfied, patience (1) exceeded.
        let d = monitor.step(2.0);
        assert!(matches!(d, EarlyStoppingDecision::Stop { .. }));
    }

    #[test]
    fn test_monitor_best_value_tracked() {
        let mut monitor = EarlyStoppingMonitor::with_default();
        monitor.step(1.0);
        monitor.step(0.5);
        monitor.step(0.8);
        assert_eq!(monitor.best_value(), Some(0.5));
        assert_eq!(monitor.best_epoch(), 2);
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = EarlyStoppingMonitor::with_default();
        monitor.step(1.0);
        monitor.step(0.5);
        assert!(monitor.best_value().is_some());

        monitor.reset();
        assert!(monitor.best_value().is_none());
        assert_eq!(monitor.current_epoch(), 0);
        assert!(monitor.history().is_empty());
        assert!(!monitor.should_stop());
    }

    #[test]
    fn test_monitor_history() {
        let mut monitor = EarlyStoppingMonitor::with_default();
        monitor.step(1.0);
        monitor.step(0.8);
        monitor.step(0.6);
        assert_eq!(monitor.history().len(), 3);
        assert_eq!(monitor.history(), &[1.0, 0.8, 0.6]);
    }

    #[test]
    fn test_monitor_summary_nonempty() {
        let mut monitor = EarlyStoppingMonitor::with_default();
        monitor.step(1.0);
        let summary = monitor.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("minimize"));
    }

    #[test]
    fn test_multi_metric_any_policy() {
        let mut mm = MultiMetricMonitor::new(MultiMetricPolicy::Any);
        mm.add_metric(
            "loss",
            EarlyStoppingConfig {
                patience: 2,
                ..Default::default()
            },
        );
        mm.add_metric(
            "accuracy",
            EarlyStoppingConfig {
                patience: 100, // very high patience
                mode: MonitorMode::Maximize,
                ..Default::default()
            },
        );

        // Step 1: both get new best.
        let d = mm.step(&[("loss".to_string(), 1.0), ("accuracy".to_string(), 0.5)]);
        assert!(matches!(d, EarlyStoppingDecision::NewBest { .. }));

        // Step 2: loss not improving, accuracy not improving.
        let d = mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.3)]);
        assert_eq!(d, EarlyStoppingDecision::Continue);

        // Step 3: loss patience exhausted (2 epochs without improvement) → Any triggers stop.
        let d = mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.3)]);
        assert!(matches!(d, EarlyStoppingDecision::Stop { .. }));
    }

    #[test]
    fn test_multi_metric_all_policy() {
        let mut mm = MultiMetricMonitor::new(MultiMetricPolicy::All);
        mm.add_metric(
            "loss",
            EarlyStoppingConfig {
                patience: 2,
                ..Default::default()
            },
        );
        mm.add_metric(
            "accuracy",
            EarlyStoppingConfig {
                patience: 2,
                mode: MonitorMode::Maximize,
                ..Default::default()
            },
        );

        // Step 1: both new best.
        mm.step(&[("loss".to_string(), 1.0), ("accuracy".to_string(), 0.5)]);

        // Step 2-3: no improvement in either.
        mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.3)]);

        // Step 3: loss has patience=2 exhausted, accuracy has patience=2 exhausted → All triggers.
        let d = mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.3)]);
        assert!(matches!(d, EarlyStoppingDecision::Stop { .. }));
    }

    #[test]
    fn test_multi_metric_all_policy_no_stop_when_one_improving() {
        let mut mm = MultiMetricMonitor::new(MultiMetricPolicy::All);
        mm.add_metric(
            "loss",
            EarlyStoppingConfig {
                patience: 2,
                ..Default::default()
            },
        );
        mm.add_metric(
            "accuracy",
            EarlyStoppingConfig {
                patience: 2,
                mode: MonitorMode::Maximize,
                ..Default::default()
            },
        );

        // Step 1
        mm.step(&[("loss".to_string(), 1.0), ("accuracy".to_string(), 0.5)]);

        // Step 2: loss stagnant, accuracy improving.
        mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.7)]);

        // Step 3: loss patience exhausted but accuracy still improving → All does NOT stop.
        let d = mm.step(&[("loss".to_string(), 1.5), ("accuracy".to_string(), 0.9)]);
        assert!(!matches!(d, EarlyStoppingDecision::Stop { .. }));
    }

    #[test]
    fn test_multi_metric_get_monitor() {
        let mut mm = MultiMetricMonitor::new(MultiMetricPolicy::Any);
        mm.add_metric("loss", EarlyStoppingConfig::default());
        mm.add_metric(
            "accuracy",
            EarlyStoppingConfig {
                mode: MonitorMode::Maximize,
                ..Default::default()
            },
        );

        assert!(mm.get_monitor("loss").is_some());
        assert!(mm.get_monitor("accuracy").is_some());
        assert!(mm.get_monitor("nonexistent").is_none());
        assert_eq!(mm.num_metrics(), 2);
    }

    #[test]
    fn test_multi_metric_summary() {
        let mut mm = MultiMetricMonitor::new(MultiMetricPolicy::Any);
        mm.add_metric("loss", EarlyStoppingConfig::default());
        let summary = mm.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("loss"));
    }

    #[test]
    fn test_plateau_detector_no_plateau() {
        let mut detector = PlateauDetector::new(3, 0.001);
        detector.push(1.0);
        detector.push(2.0);
        detector.push(3.0);
        assert!(!detector.is_plateau());
        assert!(detector.current_variance().is_some());
    }

    #[test]
    fn test_plateau_detector_plateau() {
        let mut detector = PlateauDetector::new(3, 0.001);
        detector.push(1.0);
        detector.push(1.0);
        detector.push(1.0);
        assert!(detector.is_plateau());
        assert_eq!(detector.current_variance(), Some(0.0));
    }

    #[test]
    fn test_plateau_detector_insufficient_data() {
        let mut detector = PlateauDetector::new(5, 0.001);
        detector.push(1.0);
        detector.push(1.0);
        assert!(!detector.is_plateau());
        assert!(detector.current_variance().is_none());
    }

    #[test]
    fn test_plateau_detector_window_slides() {
        let mut detector = PlateauDetector::new(3, 0.001);
        // Push varying values, then constant ones.
        detector.push(1.0);
        detector.push(5.0);
        detector.push(10.0);
        assert!(!detector.is_plateau()); // high variance

        detector.push(2.0);
        detector.push(2.0);
        detector.push(2.0);
        assert!(detector.is_plateau()); // window is [2.0, 2.0, 2.0]
    }

    #[test]
    fn test_training_progress_advance() {
        let mut progress = TrainingProgress::new(100);
        assert_eq!(progress.current_epoch, 0);
        progress.advance_epoch();
        assert_eq!(progress.current_epoch, 1);
        progress.advance_epoch();
        assert_eq!(progress.current_epoch, 2);
    }

    #[test]
    fn test_training_progress_best_minimize() {
        let mut progress = TrainingProgress::new(10);
        progress.record("loss", 1.0);
        progress.record("loss", 0.5);
        progress.record("loss", 0.8);

        let best = progress.best("loss", &MonitorMode::Minimize);
        assert_eq!(best, Some(0.5));
    }

    #[test]
    fn test_training_progress_best_maximize() {
        let mut progress = TrainingProgress::new(10);
        progress.record("accuracy", 0.6);
        progress.record("accuracy", 0.9);
        progress.record("accuracy", 0.7);

        let best = progress.best("accuracy", &MonitorMode::Maximize);
        assert_eq!(best, Some(0.9));
    }

    #[test]
    fn test_training_progress_latest() {
        let mut progress = TrainingProgress::new(10);
        progress.record("loss", 1.0);
        progress.record("loss", 0.5);
        assert_eq!(progress.latest("loss"), Some(0.5));
        assert_eq!(progress.latest("nonexistent"), None);
    }

    #[test]
    fn test_training_progress_fraction() {
        let mut progress = TrainingProgress::new(10);
        assert_eq!(progress.progress_fraction(), 0.0);
        progress.advance_epoch();
        progress.advance_epoch();
        progress.advance_epoch();
        assert!((progress.progress_fraction() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_training_progress_summary() {
        let mut progress = TrainingProgress::new(10);
        progress.advance_epoch();
        progress.record("loss", 0.5);
        let summary = progress.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("loss"));
    }

    #[test]
    fn test_training_progress_zero_total_epochs() {
        let progress = TrainingProgress::new(0);
        assert_eq!(progress.progress_fraction(), 0.0);
    }
}
