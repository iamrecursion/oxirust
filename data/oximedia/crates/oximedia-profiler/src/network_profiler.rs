//! Network performance profiling for media streaming.
//!
//! Provides deterministic simulation of network probes, streaming quality
//! scoring, multi-hop path analysis, and bandwidth estimation.

#![allow(dead_code)]

// ── NetworkProbeResult ────────────────────────────────────────────────────────

/// Result of a single network probe measurement.
#[derive(Debug, Clone)]
pub struct NetworkProbeResult {
    /// Target host/IP that was probed.
    pub target: String,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f32,
    /// Jitter (RTT standard deviation) in milliseconds.
    pub jitter_ms: f32,
    /// Packet loss percentage (0–100).
    pub packet_loss_pct: f32,
    /// Estimated available bandwidth in kbps.
    pub bandwidth_kbps: u32,
}

// ── NetworkProbe ──────────────────────────────────────────────────────────────

/// Simulated network prober.
pub struct NetworkProbe;

impl NetworkProbe {
    /// Deterministically simulate a probe to `target` using a `seed` for reproducibility.
    #[must_use]
    pub fn simulate_probe(target: &str, seed: u64) -> NetworkProbeResult {
        // Simple LCG-based pseudo-random to avoid external dependencies
        let hash = lcg_hash(seed ^ fnv1a(target.as_bytes()));

        let rtt_ms = 5.0 + (hash & 0xFF) as f32 * 0.5; // 5 – 132 ms
        let jitter_ms = (hash >> 8 & 0x1F) as f32 * 0.2; // 0 – 6 ms
        let packet_loss_pct = (hash >> 16 & 0x0F) as f32 * 0.1; // 0 – 1.5 %
        let bandwidth_kbps = 1000 + (hash >> 20 & 0xFFF) as u32 * 10; // 1 – 41 Mbps

        NetworkProbeResult {
            target: target.to_string(),
            rtt_ms,
            jitter_ms,
            packet_loss_pct,
            bandwidth_kbps,
        }
    }
}

// ── StreamingQuality ──────────────────────────────────────────────────────────

/// Measured streaming quality parameters.
#[derive(Debug, Clone, Copy)]
pub struct StreamingQuality {
    /// Number of buffering / re-buffering events.
    pub buffering_events: u32,
    /// Mean re-buffer duration in milliseconds.
    pub avg_rebuffer_ms: f32,
    /// Number of quality-level switches (adaptive bitrate).
    pub quality_switches: u32,
    /// Start-up latency in milliseconds.
    pub startup_ms: u32,
}

/// Computes a 0–100 quality of experience score.
pub struct StreamingQualityScore;

impl StreamingQualityScore {
    /// Compute a quality score from `StreamingQuality` measurements.
    ///
    /// Higher is better; 100 = perfect.
    #[must_use]
    pub fn compute(quality: &StreamingQuality) -> f32 {
        let mut score = 100.0f32;

        // Penalise buffering events (−5 each, max −40)
        score -= (quality.buffering_events as f32 * 5.0).min(40.0);

        // Penalise re-buffer duration (−1 per 500 ms, max −20)
        score -= (quality.avg_rebuffer_ms / 500.0).min(20.0);

        // Penalise quality switches (−2 each, max −20)
        score -= (quality.quality_switches as f32 * 2.0).min(20.0);

        // Penalise startup latency (−1 per 500 ms above 1 s, max −20)
        let startup_penalty = (quality.startup_ms.saturating_sub(1000) as f32 / 500.0).min(20.0);
        score -= startup_penalty;

        score.max(0.0)
    }
}

// ── NetworkHop ────────────────────────────────────────────────────────────────

/// A single hop in a network path.
#[derive(Debug, Clone)]
pub struct NetworkHop {
    /// IP address (or hostname) of this hop.
    pub ip: String,
    /// RTT to this hop in milliseconds.
    pub rtt_ms: f32,
    /// Packet loss percentage at this hop.
    pub packet_loss_pct: f32,
}

// ── NetworkPathAnalysis ───────────────────────────────────────────────────────

/// Analysis of a multi-hop network path (traceroute-like).
#[derive(Debug, Clone)]
pub struct NetworkPathAnalysis {
    /// Ordered list of network hops.
    pub hops: Vec<NetworkHop>,
    /// Sum of all hop latencies.
    pub total_latency_ms: f32,
    /// Index of the hop with the highest RTT (the bottleneck), if any.
    pub bottleneck_hop: Option<usize>,
}

impl NetworkPathAnalysis {
    /// Build a `NetworkPathAnalysis` from a slice of hops.
    #[must_use]
    pub fn from_hops(hops: Vec<NetworkHop>) -> Self {
        let total_latency_ms = hops.iter().map(|h| h.rtt_ms).sum();
        let bottleneck_hop = if hops.is_empty() {
            None
        } else {
            hops.iter()
                .enumerate()
                .max_by(|(_i, a), (_j, b)| {
                    a.rtt_ms
                        .partial_cmp(&b.rtt_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        };
        Self {
            hops,
            total_latency_ms,
            bottleneck_hop,
        }
    }
}

// ── NetworkBandwidthEstimator ─────────────────────────────────────────────────

/// Estimates available bandwidth using the harmonic mean of measurements.
#[derive(Debug, Default)]
pub struct NetworkBandwidthEstimator {
    /// (bytes, duration_ms) pairs
    measurements: Vec<(u64, u64)>,
}

impl NetworkBandwidthEstimator {
    /// Create a new estimator with no measurements.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a measurement: `bytes` transferred in `duration_ms` milliseconds.
    pub fn add_measurement(&mut self, bytes: u64, duration_ms: u64) {
        if duration_ms > 0 {
            self.measurements.push((bytes, duration_ms));
        }
    }

    /// Estimate bandwidth in kbps using the harmonic mean of per-measurement rates.
    ///
    /// Returns 0 if no measurements have been recorded.
    #[must_use]
    pub fn estimated_kbps(&self) -> u32 {
        if self.measurements.is_empty() {
            return 0;
        }
        // Harmonic mean of (bytes/ms * 8 / 1000) kbps values
        // HM = n / Σ(1/x_i)
        let n = self.measurements.len() as f64;
        let reciprocal_sum: f64 = self
            .measurements
            .iter()
            .map(|(bytes, ms)| {
                let kbps = (*bytes as f64 * 8.0) / (*ms as f64); // bits per ms = kbps
                if kbps > 0.0 {
                    1.0 / kbps
                } else {
                    0.0
                }
            })
            .sum();
        if reciprocal_sum == 0.0 {
            return 0;
        }
        (n / reciprocal_sum) as u32
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// FNV-1a hash of a byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Simple LCG step.
fn lcg_hash(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_deterministic() {
        let a = NetworkProbe::simulate_probe("192.168.1.1", 42);
        let b = NetworkProbe::simulate_probe("192.168.1.1", 42);
        // Same seed + target → same result
        assert!((a.rtt_ms - b.rtt_ms).abs() < 1e-6);
        assert!((a.bandwidth_kbps as i64 - b.bandwidth_kbps as i64).abs() == 0);
    }

    #[test]
    fn test_probe_different_seeds() {
        let a = NetworkProbe::simulate_probe("host", 1);
        let b = NetworkProbe::simulate_probe("host", 999);
        // Different seeds should produce different results (overwhelmingly likely)
        assert!(
            (a.rtt_ms - b.rtt_ms).abs() > 0.0
                || (a.bandwidth_kbps as i64 - b.bandwidth_kbps as i64).abs() > 0
        );
    }

    #[test]
    fn test_probe_rtt_positive() {
        let result = NetworkProbe::simulate_probe("example.com", 12345);
        assert!(result.rtt_ms > 0.0);
        assert!(result.bandwidth_kbps > 0);
    }

    #[test]
    fn test_streaming_quality_perfect() {
        let q = StreamingQuality {
            buffering_events: 0,
            avg_rebuffer_ms: 0.0,
            quality_switches: 0,
            startup_ms: 500,
        };
        let score = StreamingQualityScore::compute(&q);
        assert!((score - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_streaming_quality_poor() {
        let q = StreamingQuality {
            buffering_events: 20,
            avg_rebuffer_ms: 30_000.0,
            quality_switches: 50,
            startup_ms: 20_000,
        };
        let score = StreamingQualityScore::compute(&q);
        assert!(score <= 0.0);
    }

    #[test]
    fn test_streaming_quality_moderate() {
        let q = StreamingQuality {
            buffering_events: 2,
            avg_rebuffer_ms: 200.0,
            quality_switches: 3,
            startup_ms: 1500,
        };
        let score = StreamingQualityScore::compute(&q);
        assert!(score > 0.0 && score < 100.0);
    }

    #[test]
    fn test_path_analysis_empty() {
        let analysis = NetworkPathAnalysis::from_hops(vec![]);
        assert_eq!(analysis.total_latency_ms, 0.0);
        assert!(analysis.bottleneck_hop.is_none());
    }

    #[test]
    fn test_path_analysis_bottleneck() {
        let hops = vec![
            NetworkHop {
                ip: "10.0.0.1".to_string(),
                rtt_ms: 5.0,
                packet_loss_pct: 0.0,
            },
            NetworkHop {
                ip: "10.0.0.2".to_string(),
                rtt_ms: 120.0,
                packet_loss_pct: 0.0,
            },
            NetworkHop {
                ip: "10.0.0.3".to_string(),
                rtt_ms: 10.0,
                packet_loss_pct: 0.0,
            },
        ];
        let analysis = NetworkPathAnalysis::from_hops(hops);
        assert_eq!(analysis.bottleneck_hop, Some(1));
        assert!((analysis.total_latency_ms - 135.0).abs() < 1e-5);
    }

    #[test]
    fn test_bandwidth_estimator_empty() {
        let est = NetworkBandwidthEstimator::new();
        assert_eq!(est.estimated_kbps(), 0);
    }

    #[test]
    fn test_bandwidth_estimator_single() {
        let mut est = NetworkBandwidthEstimator::new();
        // 1 MB in 1000 ms → 8000 kbps
        est.add_measurement(1_000_000, 1000);
        assert_eq!(est.estimated_kbps(), 8000);
    }

    #[test]
    fn test_bandwidth_estimator_harmonic_mean() {
        let mut est = NetworkBandwidthEstimator::new();
        // Fast: 1MB in 100ms = 80000 kbps
        // Slow: 1MB in 1000ms = 8000 kbps
        est.add_measurement(1_000_000, 100);
        est.add_measurement(1_000_000, 1000);
        let kbps = est.estimated_kbps();
        // HM(80000, 8000) = 2/(1/80000 + 1/8000) ≈ 14545
        assert!(
            kbps > 8000 && kbps < 80000,
            "expected HM between extremes, got {}",
            kbps
        );
    }

    #[test]
    fn test_bandwidth_estimator_ignores_zero_duration() {
        let mut est = NetworkBandwidthEstimator::new();
        est.add_measurement(1_000_000, 0); // should be ignored
        est.add_measurement(1_000_000, 1000);
        assert_eq!(est.estimated_kbps(), 8000);
    }
}
