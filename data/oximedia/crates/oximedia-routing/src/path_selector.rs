//! Network path selection for media routing workflows.
//!
//! Provides quality scoring and selection of the best available network path
//! for real-time media delivery.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Quality and reliability metrics for a network path.
#[derive(Debug, Clone)]
pub struct PathMetrics {
    /// Round-trip latency in milliseconds.
    pub latency_ms: u32,
    /// Jitter (latency variance) in milliseconds.
    pub jitter_ms: u32,
    /// Packet loss as a percentage (0.0–100.0).
    pub packet_loss_pct: f32,
    /// Available bandwidth in Mbit/s.
    pub bandwidth_mbps: f32,
}

impl PathMetrics {
    /// Returns `true` if the path is healthy: loss < 0.5% and jitter < 5 ms.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.packet_loss_pct < 0.5 && self.jitter_ms < 5
    }

    /// Returns a quality score in the range [0.0, 1.0].
    ///
    /// The score penalises high latency, high jitter, and high packet loss.
    #[must_use]
    pub fn quality_score(&self) -> f32 {
        // Latency contribution: ideal is 0 ms, worst considered is 500 ms
        let latency_score = 1.0 - (self.latency_ms as f32 / 500.0).min(1.0);
        // Jitter contribution: ideal is 0 ms, worst considered is 50 ms
        let jitter_score = 1.0 - (self.jitter_ms as f32 / 50.0).min(1.0);
        // Loss contribution: ideal is 0%, worst is 10%
        let loss_score = 1.0 - (self.packet_loss_pct / 10.0).min(1.0);

        // Weighted average
        (latency_score * 0.4 + jitter_score * 0.3 + loss_score * 0.3).clamp(0.0, 1.0)
    }
}

/// A routable network path between two endpoints.
#[derive(Debug, Clone)]
pub struct NetworkPath {
    /// Unique path identifier.
    pub path_id: u32,
    /// Source address or identifier.
    pub source: String,
    /// Destination address or identifier.
    pub destination: String,
    /// Number of network hops.
    pub hops: u8,
    /// Current path quality metrics.
    pub metrics: PathMetrics,
}

impl NetworkPath {
    /// Returns `true` if the path is a direct (0 or 1 hop) connection.
    #[must_use]
    pub fn is_direct(&self) -> bool {
        self.hops <= 1
    }
}

/// Manages a pool of network paths and selects the optimal one.
#[derive(Debug, Default)]
pub struct PathSelector {
    paths: Vec<NetworkPath>,
}

impl PathSelector {
    /// Creates a new, empty `PathSelector`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a path to the pool.
    pub fn add(&mut self, path: NetworkPath) {
        self.paths.push(path);
    }

    /// Returns a reference to the path with the highest quality score.
    #[must_use]
    pub fn best_path(&self) -> Option<&NetworkPath> {
        self.paths.iter().max_by(|a, b| {
            a.metrics
                .quality_score()
                .partial_cmp(&b.metrics.quality_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns all paths whose quality score is at or above `min_score`.
    #[must_use]
    pub fn paths_above_quality(&self, min_score: f32) -> Vec<&NetworkPath> {
        self.paths
            .iter()
            .filter(|p| p.metrics.quality_score() >= min_score)
            .collect()
    }

    /// Removes all unhealthy paths and returns how many were removed.
    pub fn remove_unhealthy(&mut self) -> usize {
        let before = self.paths.len();
        self.paths.retain(|p| p.metrics.is_healthy());
        before - self.paths.len()
    }

    /// Returns the total number of paths in the pool.
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_metrics() -> PathMetrics {
        PathMetrics {
            latency_ms: 10,
            jitter_ms: 2,
            packet_loss_pct: 0.0,
            bandwidth_mbps: 100.0,
        }
    }

    fn unhealthy_metrics() -> PathMetrics {
        PathMetrics {
            latency_ms: 200,
            jitter_ms: 10,
            packet_loss_pct: 2.5,
            bandwidth_mbps: 5.0,
        }
    }

    fn make_path(id: u32, hops: u8, metrics: PathMetrics) -> NetworkPath {
        NetworkPath {
            path_id: id,
            source: "src".to_string(),
            destination: "dst".to_string(),
            hops,
            metrics,
        }
    }

    #[test]
    fn test_is_healthy_true() {
        assert!(healthy_metrics().is_healthy());
    }

    #[test]
    fn test_is_healthy_false_loss() {
        let m = PathMetrics {
            packet_loss_pct: 0.6,
            ..healthy_metrics()
        };
        assert!(!m.is_healthy());
    }

    #[test]
    fn test_is_healthy_false_jitter() {
        let m = PathMetrics {
            jitter_ms: 5,
            ..healthy_metrics()
        };
        assert!(!m.is_healthy());
    }

    #[test]
    fn test_quality_score_perfect() {
        let m = PathMetrics {
            latency_ms: 0,
            jitter_ms: 0,
            packet_loss_pct: 0.0,
            bandwidth_mbps: 1000.0,
        };
        assert!((m.quality_score() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quality_score_clamped() {
        let m = unhealthy_metrics();
        let score = m.quality_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_is_direct_zero_hops() {
        let path = make_path(1, 0, healthy_metrics());
        assert!(path.is_direct());
    }

    #[test]
    fn test_is_direct_one_hop() {
        let path = make_path(1, 1, healthy_metrics());
        assert!(path.is_direct());
    }

    #[test]
    fn test_is_not_direct_two_hops() {
        let path = make_path(1, 2, healthy_metrics());
        assert!(!path.is_direct());
    }

    #[test]
    fn test_selector_add_and_count() {
        let mut sel = PathSelector::new();
        sel.add(make_path(1, 0, healthy_metrics()));
        sel.add(make_path(2, 1, unhealthy_metrics()));
        assert_eq!(sel.path_count(), 2);
    }

    #[test]
    fn test_selector_best_path_picks_highest_score() {
        let mut sel = PathSelector::new();
        sel.add(make_path(1, 0, healthy_metrics()));
        sel.add(make_path(2, 2, unhealthy_metrics()));
        let best = sel.best_path().expect("should succeed in test");
        assert_eq!(best.path_id, 1);
    }

    #[test]
    fn test_selector_paths_above_quality() {
        let mut sel = PathSelector::new();
        sel.add(make_path(1, 0, healthy_metrics()));
        sel.add(make_path(2, 2, unhealthy_metrics()));
        let above = sel.paths_above_quality(0.8);
        assert_eq!(above.len(), 1);
        assert_eq!(above[0].path_id, 1);
    }

    #[test]
    fn test_selector_remove_unhealthy() {
        let mut sel = PathSelector::new();
        sel.add(make_path(1, 0, healthy_metrics()));
        sel.add(make_path(2, 2, unhealthy_metrics()));
        let removed = sel.remove_unhealthy();
        assert_eq!(removed, 1);
        assert_eq!(sel.path_count(), 1);
    }

    #[test]
    fn test_selector_best_path_empty() {
        let sel = PathSelector::new();
        assert!(sel.best_path().is_none());
    }
}
