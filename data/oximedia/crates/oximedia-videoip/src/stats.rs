//! Network statistics and monitoring.

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Network statistics for monitoring stream health.
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Packets lost (detected by sequence gaps).
    pub packets_lost: u64,
    /// Packets recovered by FEC.
    pub packets_recovered: u64,
    /// Packets arrived out of order.
    pub packets_out_of_order: u64,
    /// Duplicate packets received.
    pub packets_duplicate: u64,
    /// Current bitrate in bits per second (exponential moving average).
    pub current_bitrate: u64,
    /// Average round-trip time in microseconds.
    pub avg_rtt_us: u64,
    /// Packet jitter in microseconds (variation in packet arrival times).
    pub jitter_us: u64,
    /// Packet loss rate (0.0 - 1.0).
    pub loss_rate: f64,
}

impl NetworkStats {
    /// Creates new network statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            packets_sent: 0,
            packets_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            packets_lost: 0,
            packets_recovered: 0,
            packets_out_of_order: 0,
            packets_duplicate: 0,
            current_bitrate: 0,
            avg_rtt_us: 0,
            jitter_us: 0,
            loss_rate: 0.0,
        }
    }

    /// Updates packet loss rate.
    pub fn update_loss_rate(&mut self) {
        let total_expected = self.packets_received + self.packets_lost;
        if total_expected > 0 {
            self.loss_rate = self.packets_lost as f64 / total_expected as f64;
        }
    }

    /// Returns true if the stream is healthy (low packet loss, low jitter).
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.loss_rate < 0.01 && self.jitter_us < 10000 // < 1% loss, < 10ms jitter
    }

    /// Returns a health score from 0.0 (bad) to 1.0 (perfect).
    #[must_use]
    pub fn health_score(&self) -> f64 {
        let loss_score = (1.0 - self.loss_rate).max(0.0);
        let jitter_score = (1.0 - (self.jitter_us as f64 / 50000.0)).max(0.0);
        (loss_score + jitter_score) / 2.0
    }
}

/// Statistics tracker with time-based calculations.
pub struct StatsTracker {
    /// Current statistics.
    stats: Arc<RwLock<NetworkStats>>,
    /// Last update time for bitrate calculation.
    last_update: Instant,
    /// Bytes sent/received since last update.
    bytes_since_update: u64,
    /// Update interval for bitrate calculation.
    update_interval: Duration,
}

impl StatsTracker {
    /// Creates a new statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(NetworkStats::new())),
            last_update: Instant::now(),
            bytes_since_update: 0,
            update_interval: Duration::from_secs(1),
        }
    }

    /// Records a packet sent.
    pub fn record_sent(&mut self, bytes: usize) {
        let mut stats = self.stats.write();
        stats.packets_sent += 1;
        stats.bytes_sent += bytes as u64;
        drop(stats);

        self.bytes_since_update += bytes as u64;
        self.update_bitrate_stats();
    }

    /// Records a received packet and updates byte statistics.
    pub fn record_received(&mut self, bytes: usize) {
        let mut stats = self.stats.write();
        stats.packets_received += 1;
        stats.bytes_received += bytes as u64;
        drop(stats);

        self.bytes_since_update += bytes as u64;
        self.update_bitrate_stats();
    }

    /// Records a lost packet and updates loss statistics.
    pub fn record_lost(&self) {
        let mut stats = self.stats.write();
        stats.packets_lost += 1;
        stats.update_loss_rate();
    }

    /// Records a packet recovered by FEC.
    pub fn record_recovered(&self) {
        let mut stats = self.stats.write();
        stats.packets_recovered += 1;
    }

    /// Records an out-of-order packet.
    pub fn record_out_of_order(&self) {
        let mut stats = self.stats.write();
        stats.packets_out_of_order += 1;
    }

    /// Records a duplicate packet.
    pub fn record_duplicate(&self) {
        let mut stats = self.stats.write();
        stats.packets_duplicate += 1;
    }

    /// Updates the jitter measurement.
    pub fn update_jitter(&self, jitter_us: u64) {
        let mut stats = self.stats.write();
        // Exponential moving average
        const ALPHA: f64 = 0.125;
        stats.jitter_us =
            ((1.0 - ALPHA) * stats.jitter_us as f64 + ALPHA * jitter_us as f64) as u64;
    }

    /// Updates the round-trip time measurement.
    pub fn update_rtt(&self, rtt_us: u64) {
        let mut stats = self.stats.write();
        // Exponential moving average
        const ALPHA: f64 = 0.125;
        stats.avg_rtt_us = ((1.0 - ALPHA) * stats.avg_rtt_us as f64 + ALPHA * rtt_us as f64) as u64;
    }

    /// Updates the current bitrate (internal version that doesn't hold lock).
    fn update_bitrate_stats(&mut self) {
        let elapsed = self.last_update.elapsed();
        if elapsed >= self.update_interval {
            let bits = self.bytes_since_update * 8;
            let bitrate = (bits as f64 / elapsed.as_secs_f64()) as u64;

            // Exponential moving average
            const ALPHA: f64 = 0.2;
            let mut stats = self.stats.write();
            stats.current_bitrate =
                ((1.0 - ALPHA) * stats.current_bitrate as f64 + ALPHA * bitrate as f64) as u64;

            self.last_update = Instant::now();
            self.bytes_since_update = 0;
        }
    }

    /// Updates the current bitrate using exponential moving average.
    #[allow(dead_code)]
    fn update_bitrate(&mut self, stats: &mut NetworkStats) {
        let elapsed = self.last_update.elapsed();
        if elapsed >= self.update_interval {
            let bits = self.bytes_since_update * 8;
            let bitrate = (bits as f64 / elapsed.as_secs_f64()) as u64;

            // Exponential moving average
            const ALPHA: f64 = 0.2;
            stats.current_bitrate =
                ((1.0 - ALPHA) * stats.current_bitrate as f64 + ALPHA * bitrate as f64) as u64;

            self.last_update = Instant::now();
            self.bytes_since_update = 0;
        }
    }

    /// Returns a clone of the current statistics.
    #[must_use]
    pub fn get_stats(&self) -> NetworkStats {
        self.stats.read().clone()
    }

    /// Returns a shared reference to the statistics.
    #[must_use]
    pub fn stats(&self) -> Arc<RwLock<NetworkStats>> {
        Arc::clone(&self.stats)
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        let mut stats = self.stats.write();
        *stats = NetworkStats::new();
        self.last_update = Instant::now();
        self.bytes_since_update = 0;
    }
}

impl Default for StatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_stats_creation() {
        let stats = NetworkStats::new();
        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.packets_received, 0);
    }

    #[test]
    fn test_loss_rate_calculation() {
        let mut stats = NetworkStats::new();
        stats.packets_received = 95;
        stats.packets_lost = 5;
        stats.update_loss_rate();
        assert!((stats.loss_rate - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_check() {
        let mut stats = NetworkStats::new();
        stats.packets_received = 1000;
        stats.packets_lost = 5;
        stats.jitter_us = 5000;
        stats.update_loss_rate();
        assert!(stats.is_healthy());
    }

    #[test]
    fn test_unhealthy_stream() {
        let mut stats = NetworkStats::new();
        stats.packets_received = 100;
        stats.packets_lost = 10;
        stats.update_loss_rate();
        assert!(!stats.is_healthy());
    }

    #[test]
    fn test_health_score() {
        let stats = NetworkStats::new();
        let score = stats.health_score();
        assert!((score - 1.0).abs() < 0.01); // Perfect health
    }

    #[test]
    fn test_stats_tracker() {
        let mut tracker = StatsTracker::new();
        tracker.record_sent(1000);
        tracker.record_received(1000);

        let stats = tracker.get_stats();
        assert_eq!(stats.packets_sent, 1);
        assert_eq!(stats.packets_received, 1);
        assert_eq!(stats.bytes_sent, 1000);
        assert_eq!(stats.bytes_received, 1000);
    }

    #[test]
    fn test_stats_tracker_lost_packets() {
        let tracker = StatsTracker::new();
        tracker.record_lost();
        tracker.record_recovered();

        let stats = tracker.get_stats();
        assert_eq!(stats.packets_lost, 1);
        assert_eq!(stats.packets_recovered, 1);
    }

    #[test]
    fn test_jitter_update() {
        let tracker = StatsTracker::new();
        tracker.update_jitter(1000);
        tracker.update_jitter(2000);

        let stats = tracker.get_stats();
        assert!(stats.jitter_us > 0);
    }

    #[test]
    fn test_rtt_update() {
        let tracker = StatsTracker::new();
        tracker.update_rtt(5000);
        tracker.update_rtt(6000);

        let stats = tracker.get_stats();
        assert!(stats.avg_rtt_us > 0);
    }

    #[test]
    fn test_stats_reset() {
        let mut tracker = StatsTracker::new();
        tracker.record_sent(1000);
        tracker.reset();

        let stats = tracker.get_stats();
        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.bytes_sent, 0);
    }
}
