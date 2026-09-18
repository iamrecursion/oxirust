#![allow(dead_code)]
//! Network quality monitoring and adaptive bitrate for game streams.
//!
//! Tracks round-trip time, packet loss, bandwidth estimates, and provides
//! an adaptive bitrate ladder for live game-stream delivery.

use std::collections::VecDeque;
use std::time::Duration;

/// A single network probe measurement.
#[derive(Debug, Clone, Copy)]
pub struct NetworkProbe {
    /// Round-trip time in microseconds.
    pub rtt_us: u64,
    /// Fraction of packets lost in this measurement window (0.0..1.0).
    pub loss_fraction: f64,
    /// Estimated available bandwidth in kbps.
    pub bandwidth_kbps: u64,
    /// Measurement timestamp (monotonic, microseconds since start).
    pub timestamp_us: u64,
}

/// Quality tier used by the adaptive bitrate ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityTier {
    /// Source quality (no down-scaling).
    Source,
    /// 1080p / high bitrate.
    High,
    /// 720p / medium bitrate.
    Medium,
    /// 480p / low bitrate.
    Low,
    /// 360p / audio-only fallback.
    AudioOnly,
}

/// Bitrate ladder entry.
#[derive(Debug, Clone, Copy)]
pub struct LadderRung {
    /// Quality tier.
    pub tier: QualityTier,
    /// Target video bitrate in kbps.
    pub video_kbps: u32,
    /// Target audio bitrate in kbps.
    pub audio_kbps: u32,
    /// Required minimum bandwidth in kbps.
    pub min_bandwidth_kbps: u64,
}

/// Default bitrate ladder for live game streaming.
#[must_use]
pub fn default_ladder() -> Vec<LadderRung> {
    vec![
        LadderRung {
            tier: QualityTier::Source,
            video_kbps: 8000,
            audio_kbps: 160,
            min_bandwidth_kbps: 10000,
        },
        LadderRung {
            tier: QualityTier::High,
            video_kbps: 4500,
            audio_kbps: 128,
            min_bandwidth_kbps: 6000,
        },
        LadderRung {
            tier: QualityTier::Medium,
            video_kbps: 2500,
            audio_kbps: 128,
            min_bandwidth_kbps: 3500,
        },
        LadderRung {
            tier: QualityTier::Low,
            video_kbps: 1000,
            audio_kbps: 96,
            min_bandwidth_kbps: 1500,
        },
        LadderRung {
            tier: QualityTier::AudioOnly,
            video_kbps: 0,
            audio_kbps: 64,
            min_bandwidth_kbps: 100,
        },
    ]
}

/// Aggregated network quality snapshot.
#[derive(Debug, Clone)]
pub struct QualitySnapshot {
    /// Average RTT in microseconds.
    pub avg_rtt_us: u64,
    /// Minimum RTT in microseconds.
    pub min_rtt_us: u64,
    /// Maximum RTT in microseconds.
    pub max_rtt_us: u64,
    /// Average packet loss fraction.
    pub avg_loss: f64,
    /// Average bandwidth in kbps.
    pub avg_bandwidth_kbps: u64,
    /// Number of probes in the window.
    pub probe_count: usize,
    /// Recommended quality tier.
    pub recommended_tier: QualityTier,
}

/// Configuration for the network quality monitor.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum number of probes to keep in the sliding window.
    pub window_size: usize,
    /// RTT threshold above which we consider the link congested (us).
    pub congestion_rtt_us: u64,
    /// Loss fraction threshold for congestion detection.
    pub congestion_loss: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            congestion_rtt_us: 150_000, // 150 ms
            congestion_loss: 0.05,      // 5 %
        }
    }
}

/// Network quality monitor for game streaming.
#[derive(Debug)]
pub struct NetworkQualityMonitor {
    config: MonitorConfig,
    probes: VecDeque<NetworkProbe>,
    ladder: Vec<LadderRung>,
}

impl NetworkQualityMonitor {
    /// Create a new monitor with default ladder.
    #[must_use]
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            probes: VecDeque::with_capacity(config.window_size),
            config,
            ladder: default_ladder(),
        }
    }

    /// Create a monitor with a custom bitrate ladder.
    #[must_use]
    pub fn with_ladder(config: MonitorConfig, ladder: Vec<LadderRung>) -> Self {
        Self {
            probes: VecDeque::with_capacity(config.window_size),
            config,
            ladder,
        }
    }

    /// Record a network probe measurement.
    pub fn record_probe(&mut self, probe: NetworkProbe) {
        if self.probes.len() >= self.config.window_size {
            self.probes.pop_front();
        }
        self.probes.push_back(probe);
    }

    /// Get the current quality snapshot.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> QualitySnapshot {
        if self.probes.is_empty() {
            return QualitySnapshot {
                avg_rtt_us: 0,
                min_rtt_us: 0,
                max_rtt_us: 0,
                avg_loss: 0.0,
                avg_bandwidth_kbps: 0,
                probe_count: 0,
                recommended_tier: QualityTier::Source,
            };
        }
        let count = self.probes.len();
        let sum_rtt: u64 = self.probes.iter().map(|p| p.rtt_us).sum();
        let sum_loss: f64 = self.probes.iter().map(|p| p.loss_fraction).sum();
        let sum_bw: u64 = self.probes.iter().map(|p| p.bandwidth_kbps).sum();

        let avg_rtt = sum_rtt / count as u64;
        let avg_loss = sum_loss / count as f64;
        let avg_bw = sum_bw / count as u64;
        let min_rtt = self.probes.iter().map(|p| p.rtt_us).min().unwrap_or(0);
        let max_rtt = self.probes.iter().map(|p| p.rtt_us).max().unwrap_or(0);

        let tier = self.select_tier(avg_bw, avg_rtt, avg_loss);

        QualitySnapshot {
            avg_rtt_us: avg_rtt,
            min_rtt_us: min_rtt,
            max_rtt_us: max_rtt,
            avg_loss,
            avg_bandwidth_kbps: avg_bw,
            probe_count: count,
            recommended_tier: tier,
        }
    }

    /// Check if the network is currently in a congested state.
    #[must_use]
    pub fn is_congested(&self) -> bool {
        let snap = self.snapshot();
        snap.avg_rtt_us > self.config.congestion_rtt_us
            || snap.avg_loss > self.config.congestion_loss
    }

    /// Reset all probe history.
    pub fn reset(&mut self) {
        self.probes.clear();
    }

    /// Return the number of probes currently stored.
    #[must_use]
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    // -- private helpers --

    fn select_tier(&self, avg_bw: u64, avg_rtt: u64, avg_loss: f64) -> QualityTier {
        // Sort ladder by min_bandwidth descending so we pick the highest viable.
        let mut sorted = self.ladder.clone();
        sorted.sort_by(|a, b| b.min_bandwidth_kbps.cmp(&a.min_bandwidth_kbps));

        // Apply congestion penalty: if RTT or loss are high, cap bandwidth estimate.
        let effective_bw =
            if avg_rtt > self.config.congestion_rtt_us || avg_loss > self.config.congestion_loss {
                avg_bw / 2
            } else {
                avg_bw
            };

        for rung in &sorted {
            if effective_bw >= rung.min_bandwidth_kbps {
                return rung.tier;
            }
        }

        QualityTier::AudioOnly
    }
}

/// Estimate one-way latency from an RTT measurement.
#[must_use]
pub fn estimate_one_way_latency(rtt: Duration) -> Duration {
    rtt / 2
}

/// Compute exponentially-weighted moving average for a bandwidth series.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ewma_bandwidth(samples: &[u64], alpha: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut avg = samples[0] as f64;
    for &s in &samples[1..] {
        avg = alpha * s as f64 + (1.0 - alpha) * avg;
    }
    avg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_probe(rtt_us: u64, loss: f64, bw: u64, ts: u64) -> NetworkProbe {
        NetworkProbe {
            rtt_us,
            loss_fraction: loss,
            bandwidth_kbps: bw,
            timestamp_us: ts,
        }
    }

    #[test]
    fn test_default_ladder_order() {
        let l = default_ladder();
        assert_eq!(l.len(), 5);
        assert_eq!(l[0].tier, QualityTier::Source);
        assert_eq!(l[4].tier, QualityTier::AudioOnly);
    }

    #[test]
    fn test_monitor_empty_snapshot() {
        let mon = NetworkQualityMonitor::new(MonitorConfig::default());
        let snap = mon.snapshot();
        assert_eq!(snap.probe_count, 0);
        assert_eq!(snap.avg_rtt_us, 0);
    }

    #[test]
    fn test_monitor_single_probe() {
        let mut mon = NetworkQualityMonitor::new(MonitorConfig::default());
        mon.record_probe(make_probe(20_000, 0.0, 20_000, 0));
        let snap = mon.snapshot();
        assert_eq!(snap.probe_count, 1);
        assert_eq!(snap.avg_rtt_us, 20_000);
        assert_eq!(snap.recommended_tier, QualityTier::Source);
    }

    #[test]
    fn test_monitor_selects_medium_tier() {
        let mut mon = NetworkQualityMonitor::new(MonitorConfig::default());
        for i in 0..10 {
            mon.record_probe(make_probe(30_000, 0.01, 4000, i * 1_000_000));
        }
        let snap = mon.snapshot();
        assert_eq!(snap.recommended_tier, QualityTier::Medium);
    }

    #[test]
    fn test_monitor_congestion_detection() {
        let cfg = MonitorConfig {
            congestion_rtt_us: 100_000,
            congestion_loss: 0.05,
            window_size: 10,
        };
        let mut mon = NetworkQualityMonitor::new(cfg);
        mon.record_probe(make_probe(200_000, 0.1, 5000, 0));
        assert!(mon.is_congested());
    }

    #[test]
    fn test_monitor_not_congested() {
        let mut mon = NetworkQualityMonitor::new(MonitorConfig::default());
        mon.record_probe(make_probe(10_000, 0.0, 20_000, 0));
        assert!(!mon.is_congested());
    }

    #[test]
    fn test_monitor_window_eviction() {
        let cfg = MonitorConfig {
            window_size: 3,
            ..MonitorConfig::default()
        };
        let mut mon = NetworkQualityMonitor::new(cfg);
        for i in 0..5 {
            mon.record_probe(make_probe(10_000, 0.0, 10_000, i));
        }
        assert_eq!(mon.probe_count(), 3);
    }

    #[test]
    fn test_monitor_reset() {
        let mut mon = NetworkQualityMonitor::new(MonitorConfig::default());
        mon.record_probe(make_probe(10_000, 0.0, 10_000, 0));
        mon.reset();
        assert_eq!(mon.probe_count(), 0);
    }

    #[test]
    fn test_estimate_one_way_latency() {
        let rtt = Duration::from_millis(40);
        let owt = estimate_one_way_latency(rtt);
        assert_eq!(owt, Duration::from_millis(20));
    }

    #[test]
    fn test_ewma_bandwidth_single() {
        let avg = ewma_bandwidth(&[5000], 0.3);
        assert!((avg - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ewma_bandwidth_converge() {
        let samples = vec![10_000, 10_000, 10_000, 10_000, 10_000];
        let avg = ewma_bandwidth(&samples, 0.3);
        assert!((avg - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_ewma_bandwidth_empty() {
        let avg = ewma_bandwidth(&[], 0.3);
        assert!((avg - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_tier_equality() {
        assert_eq!(QualityTier::Source, QualityTier::Source);
        assert_ne!(QualityTier::Source, QualityTier::Low);
    }

    #[test]
    fn test_monitor_config_default() {
        let cfg = MonitorConfig::default();
        assert_eq!(cfg.window_size, 60);
        assert_eq!(cfg.congestion_rtt_us, 150_000);
    }
}
