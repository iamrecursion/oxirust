#![allow(dead_code)]
//! Bandwidth estimation for video-over-IP streams.
//!
//! Implements multiple bandwidth estimation algorithms including
//! exponentially-weighted moving average (EWMA), Google's GCC-style
//! delay-based estimation, and loss-based estimation. These are used
//! to drive adaptive bitrate decisions in real-time streaming.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Probe result
// ---------------------------------------------------------------------------

/// A single bandwidth probe / measurement.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthProbe {
    /// Timestamp of the measurement in microseconds (monotonic).
    pub timestamp_us: u64,
    /// Observed throughput in bits per second.
    pub throughput_bps: f64,
    /// One-way delay gradient (positive = increasing delay).
    pub delay_gradient_ms: f64,
    /// Packet loss ratio observed in this interval (0.0..1.0).
    pub loss_ratio: f64,
}

// ---------------------------------------------------------------------------
// Estimation state
// ---------------------------------------------------------------------------

/// Current bandwidth estimate.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthEstimate {
    /// Estimated available bandwidth in bits per second.
    pub bandwidth_bps: f64,
    /// Confidence in the estimate (0.0..1.0).
    pub confidence: f64,
    /// Whether the network is congested.
    pub congested: bool,
    /// Recommended send bitrate in bits per second.
    pub recommended_bitrate_bps: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Bandwidth estimator configuration.
#[derive(Debug, Clone)]
pub struct BandwidthEstConfig {
    /// EWMA smoothing factor for throughput (0.0..1.0, higher = more responsive).
    pub ewma_alpha: f64,
    /// Window size for delay-gradient analysis (number of probes).
    pub delay_window: usize,
    /// Threshold for delay gradient to be considered congestion signal (ms).
    pub delay_threshold_ms: f64,
    /// Loss ratio threshold above which congestion is declared.
    pub loss_threshold: f64,
    /// Multiplicative decrease factor on congestion (0.0..1.0).
    pub decrease_factor: f64,
    /// Additive increase rate (bps per probe interval).
    pub increase_rate_bps: f64,
    /// Minimum bandwidth floor (bps).
    pub min_bandwidth_bps: f64,
    /// Maximum bandwidth ceiling (bps).
    pub max_bandwidth_bps: f64,
    /// Safety margin: recommended bitrate = estimate * (1 - margin).
    pub safety_margin: f64,
}

impl Default for BandwidthEstConfig {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.1,
            delay_window: 20,
            delay_threshold_ms: 5.0,
            loss_threshold: 0.02,
            decrease_factor: 0.85,
            increase_rate_bps: 100_000.0,
            min_bandwidth_bps: 500_000.0,
            max_bandwidth_bps: 100_000_000.0,
            safety_margin: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Bandwidth estimator combining EWMA, delay-based, and loss-based signals.
#[derive(Debug)]
pub struct BandwidthEstimator {
    /// Configuration.
    config: BandwidthEstConfig,
    /// EWMA smoothed throughput.
    ewma_throughput: f64,
    /// Current estimated bandwidth.
    estimated_bw: f64,
    /// Recent delay gradients for trend analysis.
    delay_history: VecDeque<f64>,
    /// Recent loss ratios.
    loss_history: VecDeque<f64>,
    /// Total probes processed.
    probe_count: usize,
    /// Whether we detected congestion last cycle.
    congested: bool,
    /// Last probe timestamp.
    last_timestamp_us: u64,
}

impl BandwidthEstimator {
    /// Create a new bandwidth estimator with the given configuration.
    #[must_use]
    pub fn new(config: BandwidthEstConfig) -> Self {
        let initial_bw = config.max_bandwidth_bps * 0.5;
        Self {
            delay_history: VecDeque::with_capacity(config.delay_window),
            loss_history: VecDeque::with_capacity(config.delay_window),
            ewma_throughput: initial_bw,
            estimated_bw: initial_bw,
            config,
            probe_count: 0,
            congested: false,
            last_timestamp_us: 0,
        }
    }

    /// Create an estimator with default config and a specified initial bandwidth.
    #[must_use]
    pub fn with_initial_bandwidth(initial_bps: f64) -> Self {
        let mut est = Self::new(BandwidthEstConfig::default());
        est.ewma_throughput = initial_bps;
        est.estimated_bw = initial_bps;
        est
    }

    /// Feed a bandwidth probe measurement.
    pub fn push_probe(&mut self, probe: BandwidthProbe) {
        self.probe_count += 1;
        self.last_timestamp_us = probe.timestamp_us;

        // EWMA throughput update
        if self.probe_count == 1 {
            self.ewma_throughput = probe.throughput_bps;
        } else {
            self.ewma_throughput = self.config.ewma_alpha * probe.throughput_bps
                + (1.0 - self.config.ewma_alpha) * self.ewma_throughput;
        }

        // Maintain delay history
        if self.delay_history.len() == self.config.delay_window {
            self.delay_history.pop_front();
        }
        self.delay_history.push_back(probe.delay_gradient_ms);

        // Maintain loss history
        if self.loss_history.len() == self.config.delay_window {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(probe.loss_ratio);

        // Detect congestion
        let delay_congested = self.is_delay_congested();
        let loss_congested = self.is_loss_congested();
        self.congested = delay_congested || loss_congested;

        // Update estimate
        if self.congested {
            self.estimated_bw *= self.config.decrease_factor;
        } else {
            self.estimated_bw += self.config.increase_rate_bps;
        }

        // Clamp
        self.estimated_bw = self
            .estimated_bw
            .max(self.config.min_bandwidth_bps)
            .min(self.config.max_bandwidth_bps);

        // Don't exceed observed throughput by too much
        self.estimated_bw = self.estimated_bw.min(self.ewma_throughput * 1.5);
        self.estimated_bw = self.estimated_bw.max(self.config.min_bandwidth_bps);
    }

    /// Get the current bandwidth estimate.
    #[must_use]
    pub fn estimate(&self) -> BandwidthEstimate {
        let recommended = self.estimated_bw * (1.0 - self.config.safety_margin);
        #[allow(clippy::cast_precision_loss)]
        let confidence = (self.probe_count.min(50) as f64 / 50.0).min(1.0);

        BandwidthEstimate {
            bandwidth_bps: self.estimated_bw,
            confidence,
            congested: self.congested,
            recommended_bitrate_bps: recommended.max(self.config.min_bandwidth_bps),
        }
    }

    /// Get the EWMA-smoothed throughput.
    #[must_use]
    pub fn smoothed_throughput(&self) -> f64 {
        self.ewma_throughput
    }

    /// Return the total number of probes processed.
    #[must_use]
    pub fn probe_count(&self) -> usize {
        self.probe_count
    }

    /// Check if the estimator currently considers the network congested.
    #[must_use]
    pub fn is_congested(&self) -> bool {
        self.congested
    }

    /// Reset the estimator to initial state.
    pub fn reset(&mut self) {
        let initial_bw = self.config.max_bandwidth_bps * 0.5;
        self.ewma_throughput = initial_bw;
        self.estimated_bw = initial_bw;
        self.delay_history.clear();
        self.loss_history.clear();
        self.probe_count = 0;
        self.congested = false;
        self.last_timestamp_us = 0;
    }

    // -- internal --

    /// Check delay-based congestion: positive trend in delay gradients.
    fn is_delay_congested(&self) -> bool {
        if self.delay_history.len() < 3 {
            return false;
        }
        // Average recent delay gradients
        #[allow(clippy::cast_precision_loss)]
        let avg = self.delay_history.iter().sum::<f64>() / self.delay_history.len() as f64;
        avg > self.config.delay_threshold_ms
    }

    /// Check loss-based congestion.
    fn is_loss_congested(&self) -> bool {
        if self.loss_history.is_empty() {
            return false;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64;
        avg > self.config.loss_threshold
    }
}

// ---------------------------------------------------------------------------
// Simple helper: one-shot estimation from a series of throughput measurements
// ---------------------------------------------------------------------------

/// Estimate bandwidth from a series of throughput observations using EWMA.
///
/// Returns `None` if `observations` is empty.
#[must_use]
pub fn estimate_from_observations(observations: &[f64], alpha: f64) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    let mut ewma = observations[0];
    for &obs in &observations[1..] {
        ewma = alpha * obs + (1.0 - alpha) * ewma;
    }
    Some(ewma)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn good_probe(ts: u64, throughput: f64) -> BandwidthProbe {
        BandwidthProbe {
            timestamp_us: ts,
            throughput_bps: throughput,
            delay_gradient_ms: 0.5,
            loss_ratio: 0.0,
        }
    }

    fn congested_probe(ts: u64) -> BandwidthProbe {
        BandwidthProbe {
            timestamp_us: ts,
            throughput_bps: 2_000_000.0,
            delay_gradient_ms: 20.0,
            loss_ratio: 0.05,
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = BandwidthEstConfig::default();
        assert!((cfg.ewma_alpha - 0.1).abs() < 0.001);
        assert_eq!(cfg.delay_window, 20);
        assert!(cfg.min_bandwidth_bps > 0.0);
    }

    #[test]
    fn test_initial_estimate() {
        let est = BandwidthEstimator::new(BandwidthEstConfig::default());
        let e = est.estimate();
        assert!(e.bandwidth_bps > 0.0);
        assert!(!e.congested);
    }

    #[test]
    fn test_with_initial_bandwidth() {
        let est = BandwidthEstimator::with_initial_bandwidth(5_000_000.0);
        assert!((est.smoothed_throughput() - 5_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_single_probe() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        est.push_probe(good_probe(1000, 10_000_000.0));
        assert_eq!(est.probe_count(), 1);
        let e = est.estimate();
        assert!(e.bandwidth_bps > 0.0);
    }

    #[test]
    fn test_stable_throughput() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        for i in 0..50 {
            est.push_probe(good_probe(i * 1000, 10_000_000.0));
        }
        let e = est.estimate();
        assert!(!e.congested);
        assert!(e.confidence > 0.9);
        // Smoothed throughput should be near 10 Mbps
        assert!((est.smoothed_throughput() - 10_000_000.0).abs() < 1_000_000.0);
    }

    #[test]
    fn test_congestion_detected() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        for i in 0..30 {
            est.push_probe(congested_probe(i * 1000));
        }
        assert!(est.is_congested());
        let e = est.estimate();
        assert!(e.congested);
    }

    #[test]
    fn test_decrease_on_congestion() {
        let mut est = BandwidthEstimator::with_initial_bandwidth(10_000_000.0);
        let initial = est.estimate().bandwidth_bps;
        for i in 0..30 {
            est.push_probe(congested_probe(i * 1000));
        }
        let after = est.estimate().bandwidth_bps;
        assert!(after < initial);
    }

    #[test]
    fn test_recovery_after_congestion() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        // Congestion phase
        for i in 0..20 {
            est.push_probe(congested_probe(i * 1000));
        }
        let congested_bw = est.estimate().bandwidth_bps;
        // Recovery phase
        for i in 20..80 {
            est.push_probe(good_probe(i * 1000, 20_000_000.0));
        }
        let recovered_bw = est.estimate().bandwidth_bps;
        assert!(recovered_bw > congested_bw);
    }

    #[test]
    fn test_recommended_bitrate_has_margin() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        for i in 0..10 {
            est.push_probe(good_probe(i * 1000, 10_000_000.0));
        }
        let e = est.estimate();
        assert!(e.recommended_bitrate_bps < e.bandwidth_bps);
    }

    #[test]
    fn test_reset() {
        let mut est = BandwidthEstimator::new(BandwidthEstConfig::default());
        for i in 0..10 {
            est.push_probe(good_probe(i * 1000, 10_000_000.0));
        }
        est.reset();
        assert_eq!(est.probe_count(), 0);
        assert!(!est.is_congested());
    }

    #[test]
    fn test_estimate_from_observations_empty() {
        assert!(estimate_from_observations(&[], 0.1).is_none());
    }

    #[test]
    fn test_estimate_from_observations_single() {
        let result = estimate_from_observations(&[5.0], 0.1);
        assert!((result.expect("should succeed in test") - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_from_observations_convergence() {
        let obs = vec![10.0; 100];
        let result = estimate_from_observations(&obs, 0.3).expect("should succeed in test");
        assert!((result - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_min_bandwidth_floor() {
        let cfg = BandwidthEstConfig {
            min_bandwidth_bps: 1_000_000.0,
            ..Default::default()
        };
        let mut est = BandwidthEstimator::new(cfg);
        // Push very low throughput + congestion
        for i in 0..50 {
            est.push_probe(BandwidthProbe {
                timestamp_us: i * 1000,
                throughput_bps: 100_000.0,
                delay_gradient_ms: 50.0,
                loss_ratio: 0.1,
            });
        }
        let e = est.estimate();
        assert!(e.bandwidth_bps >= 1_000_000.0);
    }
}
