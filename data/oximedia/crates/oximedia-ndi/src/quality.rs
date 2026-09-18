//! NDI stream quality management.
//!
//! This module provides quality monitoring and adaptive quality selection for NDI streams,
//! including metrics collection, quality scoring, and bandwidth-aware recommendations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// Metrics snapshot for a single NDI stream observation window.
#[derive(Debug, Clone, PartialEq)]
pub struct NdiQualityMetrics {
    /// Number of frames dropped in the observation window.
    pub dropped_frames: u64,
    /// Number of frames that arrived late.
    pub late_frames: u64,
    /// Average end-to-end latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Observed bandwidth in megabits per second.
    pub bandwidth_mbps: f64,
    /// Whether the stream is currently connected.
    pub connected: bool,
}

impl NdiQualityMetrics {
    /// Create a new metrics snapshot.
    #[allow(dead_code)]
    pub fn new(
        dropped_frames: u64,
        late_frames: u64,
        avg_latency_ms: f64,
        bandwidth_mbps: f64,
        connected: bool,
    ) -> Self {
        Self {
            dropped_frames,
            late_frames,
            avg_latency_ms,
            bandwidth_mbps,
            connected,
        }
    }
}

/// NDI quality operating level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiQualityLevel {
    /// Highest quality, maximum bandwidth.
    Best,
    /// Trade-off between quality and bandwidth.
    Balanced,
    /// Reduced quality to save bandwidth.
    LowBandwidth,
    /// Stream is offline or cannot connect.
    Offline,
}

impl NdiQualityLevel {
    /// Return the maximum allowed bandwidth in Mbit/s for this quality level.
    pub fn max_bandwidth_mbps(&self) -> f64 {
        match self {
            Self::Best => 250.0,
            Self::Balanced => 100.0,
            Self::LowBandwidth => 30.0,
            Self::Offline => 0.0,
        }
    }

    /// Return the target end-to-end latency in milliseconds for this quality level.
    pub fn target_latency_ms(&self) -> f64 {
        match self {
            Self::Best => 16.7,     // one frame at 60 fps
            Self::Balanced => 33.3, // one frame at 30 fps
            Self::LowBandwidth => 100.0,
            Self::Offline => f64::MAX,
        }
    }
}

/// Rolling-window quality monitor for an NDI stream.
///
/// Collects [`NdiQualityMetrics`] observations up to `window_size` entries and
/// exposes aggregate statistics and quality recommendations.
#[derive(Debug)]
pub struct NdiQualityMonitor {
    metrics: Vec<NdiQualityMetrics>,
    window_size: usize,
}

impl NdiQualityMonitor {
    /// Create a new monitor with the given sliding-window capacity.
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(1);
        Self {
            metrics: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Record a new metrics observation, evicting the oldest if the window is full.
    pub fn record(&mut self, m: NdiQualityMetrics) {
        if self.metrics.len() >= self.window_size {
            self.metrics.remove(0);
        }
        self.metrics.push(m);
    }

    /// Return the average latency across all observations in the window.
    ///
    /// Returns `0.0` when no observations have been recorded.
    pub fn avg_latency(&self) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.metrics.iter().map(|m| m.avg_latency_ms).sum();
        sum / self.metrics.len() as f64
    }

    /// Return the total number of dropped frames across all observations.
    pub fn total_dropped(&self) -> u64 {
        self.metrics.iter().map(|m| m.dropped_frames).sum()
    }

    /// Compute a quality score in the range `[0.0, 1.0]`.
    ///
    /// A score of `1.0` means perfect quality; `0.0` means completely degraded.
    /// The score is based on drop rate, late-frame rate, and relative latency.
    pub fn quality_score(&self) -> f64 {
        if self.metrics.is_empty() {
            return 1.0;
        }

        let n = self.metrics.len() as f64;

        // Average total frames per observation (approximate: assume 30 fps per second)
        let total_drop_rate =
            self.metrics.iter().map(|m| m.dropped_frames).sum::<u64>() as f64 / (n * 30.0).max(1.0);
        let total_late_rate =
            self.metrics.iter().map(|m| m.late_frames).sum::<u64>() as f64 / (n * 30.0).max(1.0);

        let avg_lat = self.avg_latency();
        // Normalise latency: 0 ms → 1.0, 500 ms → 0.0
        let latency_score = (1.0 - avg_lat / 500.0).clamp(0.0, 1.0);

        let drop_score = (1.0 - total_drop_rate).clamp(0.0, 1.0);
        let late_score = (1.0 - total_late_rate).clamp(0.0, 1.0);

        // Check connectivity
        let all_connected = self.metrics.iter().all(|m| m.connected);
        if !all_connected {
            return 0.0;
        }

        (drop_score * 0.5 + late_score * 0.2 + latency_score * 0.3).clamp(0.0, 1.0)
    }

    /// Recommend the most appropriate [`NdiQualityLevel`] given the current metrics.
    pub fn recommend_quality(&self) -> NdiQualityLevel {
        if self.metrics.is_empty() {
            return NdiQualityLevel::Best;
        }

        let connected = self.metrics.iter().all(|m| m.connected);
        if !connected {
            return NdiQualityLevel::Offline;
        }

        let score = self.quality_score();
        let avg_bw = {
            let sum: f64 = self.metrics.iter().map(|m| m.bandwidth_mbps).sum();
            sum / self.metrics.len() as f64
        };

        if score >= 0.85 && avg_bw >= 100.0 {
            NdiQualityLevel::Best
        } else if score >= 0.6 || avg_bw >= 30.0 {
            NdiQualityLevel::Balanced
        } else {
            NdiQualityLevel::LowBandwidth
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_good() -> NdiQualityMetrics {
        NdiQualityMetrics::new(0, 0, 16.0, 200.0, true)
    }

    fn make_bad() -> NdiQualityMetrics {
        NdiQualityMetrics::new(10, 5, 400.0, 5.0, true)
    }

    #[test]
    fn test_quality_level_best_bandwidth() {
        assert_eq!(NdiQualityLevel::Best.max_bandwidth_mbps(), 250.0);
    }

    #[test]
    fn test_quality_level_offline_bandwidth() {
        assert_eq!(NdiQualityLevel::Offline.max_bandwidth_mbps(), 0.0);
    }

    #[test]
    fn test_quality_level_latency_ordering() {
        assert!(
            NdiQualityLevel::Best.target_latency_ms()
                < NdiQualityLevel::Balanced.target_latency_ms()
        );
        assert!(
            NdiQualityLevel::Balanced.target_latency_ms()
                < NdiQualityLevel::LowBandwidth.target_latency_ms()
        );
    }

    #[test]
    fn test_monitor_empty_avg_latency() {
        let m = NdiQualityMonitor::new(10);
        assert_eq!(m.avg_latency(), 0.0);
    }

    #[test]
    fn test_monitor_avg_latency() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(NdiQualityMetrics::new(0, 0, 20.0, 100.0, true));
        m.record(NdiQualityMetrics::new(0, 0, 40.0, 100.0, true));
        assert!((m.avg_latency() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_monitor_total_dropped() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(NdiQualityMetrics::new(3, 0, 20.0, 100.0, true));
        m.record(NdiQualityMetrics::new(7, 0, 20.0, 100.0, true));
        assert_eq!(m.total_dropped(), 10);
    }

    #[test]
    fn test_monitor_window_eviction() {
        let mut m = NdiQualityMonitor::new(3);
        for _ in 0..5 {
            m.record(make_good());
        }
        assert_eq!(m.metrics.len(), 3);
    }

    #[test]
    fn test_quality_score_perfect() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(make_good());
        let score = m.quality_score();
        assert!(score > 0.9, "expected high score, got {score}");
    }

    #[test]
    fn test_quality_score_degraded() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(make_bad());
        let score = m.quality_score();
        assert!(score < 0.8, "expected low score, got {score}");
    }

    #[test]
    fn test_quality_score_disconnected() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(NdiQualityMetrics::new(0, 0, 10.0, 200.0, false));
        assert_eq!(m.quality_score(), 0.0);
    }

    #[test]
    fn test_recommend_best() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(make_good());
        assert_eq!(m.recommend_quality(), NdiQualityLevel::Best);
    }

    #[test]
    fn test_recommend_offline_when_disconnected() {
        let mut m = NdiQualityMonitor::new(10);
        m.record(NdiQualityMetrics::new(0, 0, 10.0, 200.0, false));
        assert_eq!(m.recommend_quality(), NdiQualityLevel::Offline);
    }

    #[test]
    fn test_recommend_low_bandwidth() {
        let mut m = NdiQualityMonitor::new(10);
        // Very poor conditions
        m.record(NdiQualityMetrics::new(100, 50, 450.0, 2.0, true));
        let rec = m.recommend_quality();
        assert!(
            rec == NdiQualityLevel::LowBandwidth || rec == NdiQualityLevel::Balanced,
            "unexpected recommendation: {rec:?}"
        );
    }

    #[test]
    fn test_empty_monitor_recommend() {
        let m = NdiQualityMonitor::new(5);
        assert_eq!(m.recommend_quality(), NdiQualityLevel::Best);
    }
}
