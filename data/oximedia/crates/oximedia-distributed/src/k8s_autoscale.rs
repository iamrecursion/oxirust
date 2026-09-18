//! Kubernetes HPA (Horizontal Pod Autoscaler) integration.
//!
//! Provides a pure-Rust implementation of HPA-style scaling logic:
//! given the current number of replicas, queue depth, and active workers,
//! [`compute_scaling_recommendation`] decides whether to scale up, scale
//! down, or keep the current replica count unchanged.
//!
//! # Algorithm
//!
//! - **Scale up**: `queue_depth / active_workers > target_queue_depth * scale_up_threshold`
//!   → add one replica (capped at `max_replicas`).
//! - **Scale down**: `queue_depth / active_workers < target_queue_depth * scale_down_threshold`
//!   → remove one replica (capped at `min_replicas`).
//! - **No change**: otherwise keep `current_replicas`.
//!
//! Division by zero is guarded: when `active_workers` is 0 the ratio is
//! treated as `f64::MAX` (scale up immediately).
//!
//! # Feature gate
//!
//! This module is gated behind the `k8s` Cargo feature:
//!
//! ```toml
//! oximedia-distributed = { version = "…", features = ["k8s"] }
//! ```

/// Configuration for the HPA scaling logic.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HpaConfig {
    /// Minimum number of replicas (pods) that must always be running.
    pub min_replicas: u32,
    /// Maximum number of replicas allowed.
    pub max_replicas: u32,
    /// Target number of queued jobs per active worker at steady state.
    pub target_queue_depth: usize,
    /// Scale-up trigger: ratio of actual load to target above which a
    /// scale-up is recommended (e.g. `1.2` = 20 % over target).
    pub scale_up_threshold: f64,
    /// Scale-down trigger: ratio of actual load to target below which a
    /// scale-down is recommended (e.g. `0.5` = 50 % of target).
    pub scale_down_threshold: f64,
}

impl Default for HpaConfig {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 10,
            target_queue_depth: 5,
            scale_up_threshold: 1.2,
            scale_down_threshold: 0.5,
        }
    }
}

impl HpaConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error string if the config is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_replicas == 0 {
            return Err("min_replicas must be >= 1".to_string());
        }
        if self.max_replicas < self.min_replicas {
            return Err("max_replicas must be >= min_replicas".to_string());
        }
        if self.scale_up_threshold <= 0.0 {
            return Err("scale_up_threshold must be > 0.0".to_string());
        }
        if self.scale_down_threshold < 0.0 || self.scale_down_threshold >= self.scale_up_threshold {
            return Err("scale_down_threshold must be in [0, scale_up_threshold)".to_string());
        }
        Ok(())
    }
}

/// The scaling decision produced by [`compute_scaling_recommendation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingRecommendation {
    /// Current replica count (input unchanged).
    pub current: u32,
    /// Recommended replica count.
    pub recommended: u32,
    /// Human-readable rationale for the recommendation.
    pub reason: String,
}

impl ScalingRecommendation {
    /// Returns `true` if the recommendation changes the replica count.
    #[must_use]
    pub fn is_change(&self) -> bool {
        self.current != self.recommended
    }

    /// Returns `true` if this is a scale-up recommendation.
    #[must_use]
    pub fn is_scale_up(&self) -> bool {
        self.recommended > self.current
    }

    /// Returns `true` if this is a scale-down recommendation.
    #[must_use]
    pub fn is_scale_down(&self) -> bool {
        self.recommended < self.current
    }

    /// Returns `true` if the replica count is unchanged.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.current == self.recommended
    }
}

/// Compute an HPA scaling recommendation.
///
/// # Arguments
///
/// - `config`          — HPA configuration (thresholds, bounds).
/// - `current_replicas`— Number of replicas currently running.
/// - `queue_depth`     — Total number of queued (pending) jobs.
/// - `active_workers`  — Number of workers currently processing jobs.
///
/// # Returns
///
/// A [`ScalingRecommendation`] describing the change (or stability).
#[must_use]
pub fn compute_scaling_recommendation(
    config: &HpaConfig,
    current_replicas: u32,
    queue_depth: usize,
    active_workers: u32,
) -> ScalingRecommendation {
    // Avoid division by zero: 0 active workers → maximum possible ratio.
    let ratio = if active_workers == 0 {
        f64::MAX
    } else {
        queue_depth as f64 / active_workers as f64
    };

    let target = config.target_queue_depth as f64;
    let up_threshold = target * config.scale_up_threshold;
    let down_threshold = target * config.scale_down_threshold;

    if ratio > up_threshold {
        // Scale up by one, respecting max_replicas.
        let recommended = (current_replicas + 1).min(config.max_replicas);
        let reason = if recommended == current_replicas {
            format!(
                "would scale up but already at max_replicas ({})",
                config.max_replicas
            )
        } else {
            format!(
                "scale up: load ratio {ratio:.2} > threshold {up_threshold:.2}; \
                 {current_replicas} → {recommended}"
            )
        };
        ScalingRecommendation {
            current: current_replicas,
            recommended,
            reason,
        }
    } else if ratio < down_threshold {
        // Scale down by one, respecting min_replicas.
        let recommended = current_replicas.saturating_sub(1).max(config.min_replicas);
        let reason = if recommended == current_replicas {
            format!(
                "would scale down but already at min_replicas ({})",
                config.min_replicas
            )
        } else {
            format!(
                "scale down: load ratio {ratio:.2} < threshold {down_threshold:.2}; \
                 {current_replicas} → {recommended}"
            )
        };
        ScalingRecommendation {
            current: current_replicas,
            recommended,
            reason,
        }
    } else {
        ScalingRecommendation {
            current: current_replicas,
            recommended: current_replicas,
            reason: format!(
                "stable: load ratio {ratio:.2} within [{down_threshold:.2}, {up_threshold:.2}]"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HpaConfig {
        HpaConfig {
            min_replicas: 1,
            max_replicas: 8,
            target_queue_depth: 5,
            scale_up_threshold: 1.2,   // triggers at ratio > 6.0
            scale_down_threshold: 0.5, // triggers at ratio < 2.5
        }
    }

    #[test]
    fn test_hpa_scale_up_on_high_queue() {
        let config = default_config();
        // 30 jobs / 3 workers = ratio 10 > threshold 6.0 → scale up
        let rec = compute_scaling_recommendation(&config, 3, 30, 3);
        assert!(rec.is_scale_up(), "expected scale-up: {}", rec.reason);
        assert_eq!(rec.recommended, 4);
    }

    #[test]
    fn test_hpa_scale_down_on_low_queue() {
        let config = default_config();
        // 2 jobs / 4 workers = ratio 0.5 < threshold 2.5 → scale down
        let rec = compute_scaling_recommendation(&config, 4, 2, 4);
        assert!(rec.is_scale_down(), "expected scale-down: {}", rec.reason);
        assert_eq!(rec.recommended, 3);
    }

    #[test]
    fn test_hpa_stable_in_normal_range() {
        let config = default_config();
        // 20 jobs / 4 workers = ratio 5.0; thresholds [2.5, 6.0] → stable
        let rec = compute_scaling_recommendation(&config, 4, 20, 4);
        assert!(rec.is_stable(), "expected stable: {}", rec.reason);
        assert_eq!(rec.recommended, 4);
    }

    #[test]
    fn test_hpa_respects_max_replicas() {
        let config = HpaConfig {
            max_replicas: 2,
            ..default_config()
        };
        // High load, but already at max_replicas (2)
        let rec = compute_scaling_recommendation(&config, 2, 100, 1);
        assert_eq!(rec.recommended, 2, "should not exceed max_replicas");
    }

    #[test]
    fn test_hpa_respects_min_replicas() {
        let config = HpaConfig {
            min_replicas: 2,
            ..default_config()
        };
        // Very low load, already at min_replicas (2)
        let rec = compute_scaling_recommendation(&config, 2, 0, 10);
        assert_eq!(rec.recommended, 2, "should not go below min_replicas");
    }

    #[test]
    fn test_hpa_zero_active_workers_triggers_scale_up() {
        let config = default_config();
        // 0 active workers → ratio = MAX → always scale up
        let rec = compute_scaling_recommendation(&config, 1, 10, 0);
        assert!(
            rec.is_scale_up(),
            "0 active workers should trigger scale-up"
        );
    }

    #[test]
    fn test_hpa_zero_queue_with_many_workers_triggers_scale_down() {
        let config = default_config();
        // 0 / 5 = 0 < 2.5 → scale down
        let rec = compute_scaling_recommendation(&config, 5, 0, 5);
        assert!(rec.is_scale_down(), "empty queue should trigger scale-down");
    }

    #[test]
    fn test_hpa_config_validation_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_hpa_config_validation_invalid_min() {
        let config = HpaConfig {
            min_replicas: 0,
            ..default_config()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hpa_config_validation_max_less_than_min() {
        let config = HpaConfig {
            min_replicas: 5,
            max_replicas: 3,
            ..default_config()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_scaling_recommendation_flags() {
        let up = ScalingRecommendation {
            current: 2,
            recommended: 3,
            reason: "scale up".to_string(),
        };
        assert!(up.is_scale_up());
        assert!(!up.is_scale_down());
        assert!(!up.is_stable());
        assert!(up.is_change());

        let stable = ScalingRecommendation {
            current: 2,
            recommended: 2,
            reason: "stable".to_string(),
        };
        assert!(stable.is_stable());
        assert!(!stable.is_change());
    }
}
