#![allow(dead_code)]
//! Quality-of-Service monitoring for network media streams.
//!
//! Collects per-stream QoS metrics (jitter, packet loss, round-trip time) and
//! derives an overall health score that can drive adaptive bitrate decisions.

use std::collections::VecDeque;
use std::fmt;

/// A single QoS observation at a point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QosSample {
    /// Timestamp of the sample in milliseconds since stream start.
    pub timestamp_ms: u64,
    /// One-way jitter in microseconds.
    pub jitter_us: u64,
    /// Fraction of packets lost in this interval (0.0..=1.0).
    pub loss_fraction: f64,
    /// Round-trip time in milliseconds (`None` if not measured).
    pub rtt_ms: Option<u64>,
}

impl QosSample {
    /// Creates a new QoS sample.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(timestamp_ms: u64, jitter_us: u64, loss_fraction: f64, rtt_ms: Option<u64>) -> Self {
        Self {
            timestamp_ms,
            jitter_us,
            loss_fraction: loss_fraction.clamp(0.0, 1.0),
            rtt_ms,
        }
    }
}

/// Aggregate QoS health rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthRating {
    /// Stream quality is excellent.
    Excellent,
    /// Stream quality is good — minor impairments.
    Good,
    /// Stream quality is fair — noticeable degradation.
    Fair,
    /// Stream quality is poor — likely visible/audible artefacts.
    Poor,
    /// Stream has failed or is unusable.
    Critical,
}

impl fmt::Display for HealthRating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Critical => "Critical",
        };
        f.write_str(label)
    }
}

/// Summary statistics derived from the QoS window.
#[derive(Debug, Clone, PartialEq)]
pub struct QosSummary {
    /// Number of samples in the window.
    pub sample_count: usize,
    /// Mean jitter in microseconds.
    pub mean_jitter_us: f64,
    /// Maximum jitter in microseconds.
    pub max_jitter_us: u64,
    /// Mean loss fraction.
    pub mean_loss: f64,
    /// Mean RTT in milliseconds (only from samples that have RTT).
    pub mean_rtt_ms: Option<f64>,
    /// Composite health score in 0.0..=100.0.
    pub health_score: f64,
    /// Derived health rating.
    pub rating: HealthRating,
}

/// Sliding-window QoS monitor.
///
/// Holds up to `capacity` samples and derives aggregate statistics on demand.
#[derive(Debug)]
pub struct QosMonitor {
    samples: VecDeque<QosSample>,
    capacity: usize,
}

impl QosMonitor {
    /// Creates a new monitor with the given window capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Pushes a new sample, evicting the oldest if the window is full.
    pub fn push(&mut self, sample: QosSample) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Returns the number of samples currently held.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if no samples are present.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Clears all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Computes a [`QosSummary`] from the current window.
    #[allow(clippy::cast_precision_loss)]
    pub fn summary(&self) -> QosSummary {
        if self.samples.is_empty() {
            return QosSummary {
                sample_count: 0,
                mean_jitter_us: 0.0,
                max_jitter_us: 0,
                mean_loss: 0.0,
                mean_rtt_ms: None,
                health_score: 100.0,
                rating: HealthRating::Excellent,
            };
        }

        let n = self.samples.len() as f64;
        let mean_jitter_us: f64 = self.samples.iter().map(|s| s.jitter_us as f64).sum::<f64>() / n;
        let max_jitter_us = self.samples.iter().map(|s| s.jitter_us).max().unwrap_or(0);
        let mean_loss: f64 = self.samples.iter().map(|s| s.loss_fraction).sum::<f64>() / n;

        let rtt_samples: Vec<u64> = self.samples.iter().filter_map(|s| s.rtt_ms).collect();
        let mean_rtt_ms = if rtt_samples.is_empty() {
            None
        } else {
            Some(rtt_samples.iter().map(|&r| r as f64).sum::<f64>() / rtt_samples.len() as f64)
        };

        let health_score = Self::compute_health(mean_jitter_us, mean_loss, mean_rtt_ms);
        let rating = Self::rating_from_score(health_score);

        QosSummary {
            sample_count: self.samples.len(),
            mean_jitter_us,
            max_jitter_us,
            mean_loss,
            mean_rtt_ms,
            health_score,
            rating,
        }
    }

    /// Computes a composite 0..100 health score.
    ///
    /// Higher is better.  Penalises jitter, loss, and latency.
    fn compute_health(mean_jitter_us: f64, mean_loss: f64, mean_rtt_ms: Option<f64>) -> f64 {
        // Jitter penalty: 1 ms jitter → −5 points (capped at −40)
        let jitter_penalty = (mean_jitter_us / 1000.0 * 5.0).min(40.0);
        // Loss penalty: 1 % loss → −20 points (capped at −50)
        let loss_penalty = (mean_loss * 100.0 * 20.0).min(50.0);
        // RTT penalty: 100 ms → −5 points (capped at −30)
        let rtt_penalty = mean_rtt_ms
            .map(|rtt| (rtt / 100.0 * 5.0).min(30.0))
            .unwrap_or(0.0);

        (100.0 - jitter_penalty - loss_penalty - rtt_penalty).max(0.0)
    }

    /// Maps a numeric health score to a [`HealthRating`].
    fn rating_from_score(score: f64) -> HealthRating {
        if score >= 90.0 {
            HealthRating::Excellent
        } else if score >= 70.0 {
            HealthRating::Good
        } else if score >= 50.0 {
            HealthRating::Fair
        } else if score >= 20.0 {
            HealthRating::Poor
        } else {
            HealthRating::Critical
        }
    }

    /// Returns the latest sample, if any.
    pub fn latest(&self) -> Option<&QosSample> {
        self.samples.back()
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(ts: u64, jitter: u64, loss: f64, rtt: Option<u64>) -> QosSample {
        QosSample::new(ts, jitter, loss, rtt)
    }

    // 1. new monitor is empty
    #[test]
    fn test_monitor_starts_empty() {
        let m = QosMonitor::new(10);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    // 2. push increases len
    #[test]
    fn test_push_increases_len() {
        let mut m = QosMonitor::new(10);
        m.push(make_sample(0, 100, 0.0, None));
        assert_eq!(m.len(), 1);
    }

    // 3. eviction at capacity
    #[test]
    fn test_eviction() {
        let mut m = QosMonitor::new(3);
        for i in 0..5 {
            m.push(make_sample(i, 100, 0.0, None));
        }
        assert_eq!(m.len(), 3);
    }

    // 4. summary of empty monitor
    #[test]
    fn test_summary_empty() {
        let m = QosMonitor::new(5);
        let s = m.summary();
        assert_eq!(s.sample_count, 0);
        assert_eq!(s.rating, HealthRating::Excellent);
    }

    // 5. perfect stream → Excellent
    #[test]
    fn test_perfect_stream() {
        let mut m = QosMonitor::new(5);
        for i in 0..5 {
            m.push(make_sample(i * 100, 50, 0.0, Some(10)));
        }
        let s = m.summary();
        assert!(s.health_score >= 90.0);
        assert_eq!(s.rating, HealthRating::Excellent);
    }

    // 6. high loss → Poor/Critical
    #[test]
    fn test_high_loss_poor() {
        let mut m = QosMonitor::new(4);
        for i in 0..4 {
            m.push(make_sample(i * 100, 100, 0.05, Some(50)));
        }
        let s = m.summary();
        assert!(s.health_score < 50.0, "score={}", s.health_score);
    }

    // 7. mean jitter calculated correctly
    #[test]
    fn test_mean_jitter() {
        let mut m = QosMonitor::new(2);
        m.push(make_sample(0, 200, 0.0, None));
        m.push(make_sample(100, 400, 0.0, None));
        let s = m.summary();
        assert!((s.mean_jitter_us - 300.0).abs() < f64::EPSILON);
    }

    // 8. max jitter tracked
    #[test]
    fn test_max_jitter() {
        let mut m = QosMonitor::new(3);
        m.push(make_sample(0, 100, 0.0, None));
        m.push(make_sample(100, 500, 0.0, None));
        m.push(make_sample(200, 200, 0.0, None));
        let s = m.summary();
        assert_eq!(s.max_jitter_us, 500);
    }

    // 9. mean RTT with partial data
    #[test]
    fn test_mean_rtt_partial() {
        let mut m = QosMonitor::new(3);
        m.push(make_sample(0, 0, 0.0, Some(100)));
        m.push(make_sample(100, 0, 0.0, None));
        m.push(make_sample(200, 0, 0.0, Some(200)));
        let s = m.summary();
        // mean of 100 and 200 = 150
        assert!((s.mean_rtt_ms.expect("should succeed in test") - 150.0).abs() < f64::EPSILON);
    }

    // 10. loss fraction clamped
    #[test]
    fn test_loss_clamped() {
        let s = QosSample::new(0, 0, 2.0, None);
        assert!((s.loss_fraction - 1.0).abs() < f64::EPSILON);
        let s2 = QosSample::new(0, 0, -0.5, None);
        assert!(s2.loss_fraction.abs() < f64::EPSILON);
    }

    // 11. reset clears samples
    #[test]
    fn test_reset() {
        let mut m = QosMonitor::new(5);
        m.push(make_sample(0, 100, 0.0, None));
        m.reset();
        assert!(m.is_empty());
    }

    // 12. latest returns newest sample
    #[test]
    fn test_latest() {
        let mut m = QosMonitor::new(5);
        m.push(make_sample(0, 100, 0.0, None));
        m.push(make_sample(100, 200, 0.01, Some(50)));
        let latest = m.latest().expect("should succeed in test");
        assert_eq!(latest.timestamp_ms, 100);
    }

    // 13. HealthRating display
    #[test]
    fn test_health_rating_display() {
        assert_eq!(format!("{}", HealthRating::Critical), "Critical");
        assert_eq!(format!("{}", HealthRating::Good), "Good");
    }

    // 14. rating_from_score boundaries
    #[test]
    fn test_rating_boundaries() {
        assert_eq!(
            QosMonitor::rating_from_score(100.0),
            HealthRating::Excellent
        );
        assert_eq!(QosMonitor::rating_from_score(90.0), HealthRating::Excellent);
        assert_eq!(QosMonitor::rating_from_score(89.9), HealthRating::Good);
        assert_eq!(QosMonitor::rating_from_score(70.0), HealthRating::Good);
        assert_eq!(QosMonitor::rating_from_score(50.0), HealthRating::Fair);
        assert_eq!(QosMonitor::rating_from_score(20.0), HealthRating::Poor);
        assert_eq!(QosMonitor::rating_from_score(0.0), HealthRating::Critical);
    }
}
