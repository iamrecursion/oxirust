#![allow(dead_code)]
//! Jitter buffer for handling network timing jitter in time synchronization.
//!
//! Provides adaptive and fixed-size buffers that absorb packet timing
//! variations, ensuring smooth clock updates even over unreliable networks.

use std::collections::VecDeque;

/// Configuration for the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Minimum buffer depth in samples.
    pub min_depth: usize,
    /// Maximum buffer depth in samples.
    pub max_depth: usize,
    /// Target buffer depth in samples.
    pub target_depth: usize,
    /// Whether to adapt the depth automatically.
    pub adaptive: bool,
    /// Maximum allowed jitter in nanoseconds before discarding.
    pub max_jitter_ns: i64,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            min_depth: 2,
            max_depth: 64,
            target_depth: 8,
            adaptive: true,
            max_jitter_ns: 10_000_000, // 10ms
        }
    }
}

impl JitterBufferConfig {
    /// Creates a new configuration with the given target depth.
    #[must_use]
    pub fn new(target_depth: usize) -> Self {
        Self {
            target_depth,
            min_depth: target_depth.saturating_sub(2).max(1),
            max_depth: target_depth * 4,
            ..Default::default()
        }
    }

    /// Sets whether the buffer adapts automatically.
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Sets the maximum jitter threshold in nanoseconds.
    #[must_use]
    pub fn with_max_jitter_ns(mut self, ns: i64) -> Self {
        self.max_jitter_ns = ns;
        self
    }
}

/// A single timestamped sample in the buffer.
#[derive(Debug, Clone, Copy)]
pub struct TimingSample {
    /// Sequence number of this sample.
    pub sequence: u64,
    /// Measured offset in nanoseconds.
    pub offset_ns: i64,
    /// One-way delay in nanoseconds.
    pub delay_ns: i64,
    /// Receive timestamp in nanoseconds since epoch.
    pub receive_time_ns: u64,
}

impl TimingSample {
    /// Creates a new timing sample.
    #[must_use]
    pub fn new(sequence: u64, offset_ns: i64, delay_ns: i64, receive_time_ns: u64) -> Self {
        Self {
            sequence,
            offset_ns,
            delay_ns,
            receive_time_ns,
        }
    }
}

/// Statistics about jitter buffer performance.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Total samples received.
    pub total_received: u64,
    /// Samples consumed (played out).
    pub total_consumed: u64,
    /// Samples discarded due to overflow or excessive jitter.
    pub total_discarded: u64,
    /// Current buffer depth.
    pub current_depth: usize,
    /// Running mean jitter in nanoseconds.
    pub mean_jitter_ns: f64,
    /// Peak jitter observed in nanoseconds.
    pub peak_jitter_ns: i64,
    /// Number of underruns (buffer empty when read attempted).
    pub underruns: u64,
    /// Number of overruns (buffer full when write attempted).
    pub overruns: u64,
}

/// Adaptive jitter buffer for time synchronization samples.
#[derive(Debug)]
pub struct JitterBuffer {
    /// Configuration.
    config: JitterBufferConfig,
    /// Buffered samples, ordered by receive time.
    buffer: VecDeque<TimingSample>,
    /// Running statistics.
    stats: JitterStats,
    /// Last consumed offset for jitter calculation.
    last_offset_ns: Option<i64>,
    /// Exponentially weighted jitter estimate.
    jitter_estimate_ns: f64,
    /// Current adaptive depth.
    adaptive_depth: usize,
}

impl JitterBuffer {
    /// Creates a new jitter buffer with the given configuration.
    #[must_use]
    pub fn new(config: JitterBufferConfig) -> Self {
        let adaptive_depth = config.target_depth;
        Self {
            config,
            buffer: VecDeque::new(),
            stats: JitterStats::default(),
            last_offset_ns: None,
            jitter_estimate_ns: 0.0,
            adaptive_depth,
        }
    }

    /// Creates a jitter buffer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(JitterBufferConfig::default())
    }

    /// Pushes a new timing sample into the buffer.
    ///
    /// Returns `true` if the sample was accepted, `false` if discarded.
    pub fn push(&mut self, sample: TimingSample) -> bool {
        self.stats.total_received += 1;

        // Check jitter threshold
        if let Some(last) = self.last_offset_ns {
            let jitter = (sample.offset_ns - last).abs();
            if jitter > self.config.max_jitter_ns {
                self.stats.total_discarded += 1;
                return false;
            }
            self.update_jitter_estimate(jitter);
        }

        // Check overflow
        let max = if self.config.adaptive {
            self.adaptive_depth
        } else {
            self.config.max_depth
        };

        if self.buffer.len() >= max {
            self.stats.overruns += 1;
            self.stats.total_discarded += 1;
            // Drop oldest
            self.buffer.pop_front();
        }

        self.buffer.push_back(sample);
        self.stats.current_depth = self.buffer.len();
        true
    }

    /// Consumes the next sample from the buffer.
    ///
    /// Returns `None` if the buffer is empty (underrun).
    pub fn pop(&mut self) -> Option<TimingSample> {
        if self.buffer.is_empty() {
            self.stats.underruns += 1;
            return None;
        }
        let sample = self.buffer.pop_front()?;
        self.last_offset_ns = Some(sample.offset_ns);
        self.stats.total_consumed += 1;
        self.stats.current_depth = self.buffer.len();
        Some(sample)
    }

    /// Peeks at the next sample without consuming it.
    #[must_use]
    pub fn peek(&self) -> Option<&TimingSample> {
        self.buffer.front()
    }

    /// Returns the current buffer depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the current statistics.
    #[must_use]
    pub fn stats(&self) -> &JitterStats {
        &self.stats
    }

    /// Returns the current jitter estimate in nanoseconds.
    #[must_use]
    pub fn jitter_estimate_ns(&self) -> f64 {
        self.jitter_estimate_ns
    }

    /// Returns the current adaptive depth target.
    #[must_use]
    pub fn adaptive_depth(&self) -> usize {
        self.adaptive_depth
    }

    /// Resets the buffer, clearing all samples and statistics.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.stats = JitterStats::default();
        self.last_offset_ns = None;
        self.jitter_estimate_ns = 0.0;
        self.adaptive_depth = self.config.target_depth;
    }

    /// Computes the median offset of currently buffered samples.
    #[must_use]
    pub fn median_offset_ns(&self) -> Option<i64> {
        if self.buffer.is_empty() {
            return None;
        }
        let mut offsets: Vec<i64> = self.buffer.iter().map(|s| s.offset_ns).collect();
        offsets.sort_unstable();
        let mid = offsets.len() / 2;
        if offsets.len() % 2 == 0 && offsets.len() >= 2 {
            Some((offsets[mid - 1] + offsets[mid]) / 2)
        } else {
            Some(offsets[mid])
        }
    }

    /// Computes the mean delay of currently buffered samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_delay_ns(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            return None;
        }
        let sum: i64 = self.buffer.iter().map(|s| s.delay_ns).sum();
        Some(sum as f64 / self.buffer.len() as f64)
    }

    /// Updates the exponentially weighted jitter estimate.
    #[allow(clippy::cast_precision_loss)]
    fn update_jitter_estimate(&mut self, jitter: i64) {
        // RFC 3550 style jitter calculation: J(i) = J(i-1) + (|D(i)| - J(i-1)) / 16
        let j = jitter as f64;
        self.jitter_estimate_ns += (j - self.jitter_estimate_ns) / 16.0;
        self.stats.mean_jitter_ns = self.jitter_estimate_ns;
        if jitter > self.stats.peak_jitter_ns {
            self.stats.peak_jitter_ns = jitter;
        }
        // Adapt depth if needed
        if self.config.adaptive {
            self.adapt_depth();
        }
    }

    /// Adapts the buffer depth based on observed jitter.
    fn adapt_depth(&mut self) {
        #[allow(clippy::cast_precision_loss)]
        let jitter_ratio = self.jitter_estimate_ns / self.config.max_jitter_ns as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let desired = ((self.config.target_depth as f64) * (1.0 + jitter_ratio)) as usize;
        self.adaptive_depth = desired.clamp(self.config.min_depth, self.config.max_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(seq: u64, offset: i64, delay: i64) -> TimingSample {
        TimingSample::new(seq, offset, delay, seq * 1_000_000)
    }

    #[test]
    fn test_config_default() {
        let cfg = JitterBufferConfig::default();
        assert_eq!(cfg.target_depth, 8);
        assert!(cfg.adaptive);
        assert_eq!(cfg.max_jitter_ns, 10_000_000);
    }

    #[test]
    fn test_config_new() {
        let cfg = JitterBufferConfig::new(16);
        assert_eq!(cfg.target_depth, 16);
        assert_eq!(cfg.min_depth, 14);
        assert_eq!(cfg.max_depth, 64);
    }

    #[test]
    fn test_config_builder() {
        let cfg = JitterBufferConfig::new(4)
            .with_adaptive(false)
            .with_max_jitter_ns(5_000_000);
        assert!(!cfg.adaptive);
        assert_eq!(cfg.max_jitter_ns, 5_000_000);
    }

    #[test]
    fn test_push_and_pop() {
        let mut buf = JitterBuffer::new(JitterBufferConfig::new(4));
        assert!(buf.is_empty());

        assert!(buf.push(sample(1, 100, 50)));
        assert_eq!(buf.depth(), 1);

        let s = buf.pop().expect("should succeed in test");
        assert_eq!(s.sequence, 1);
        assert_eq!(s.offset_ns, 100);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_underrun_tracking() {
        let mut buf = JitterBuffer::with_defaults();
        assert!(buf.pop().is_none());
        assert_eq!(buf.stats().underruns, 1);
    }

    #[test]
    fn test_overflow_drops_oldest() {
        let cfg = JitterBufferConfig {
            max_depth: 3,
            adaptive: false,
            max_jitter_ns: i64::MAX,
            ..JitterBufferConfig::new(2)
        };
        let mut buf = JitterBuffer::new(cfg);
        buf.push(sample(1, 0, 10));
        buf.push(sample(2, 0, 10));
        buf.push(sample(3, 0, 10));
        // This should trigger overflow
        buf.push(sample(4, 0, 10));

        assert_eq!(buf.depth(), 3);
        let first = buf.pop().expect("should succeed in test");
        assert_eq!(first.sequence, 2); // seq 1 was dropped
    }

    #[test]
    fn test_discard_excessive_jitter() {
        let cfg = JitterBufferConfig::default().with_max_jitter_ns(100);
        let mut buf = JitterBuffer::new(cfg);
        // First sample is always accepted (no previous)
        assert!(buf.push(sample(1, 0, 10)));
        buf.pop(); // consume so last_offset_ns is set
                   // Next sample with huge jitter should be discarded
        assert!(!buf.push(sample(2, 5000, 10)));
        assert_eq!(buf.stats().total_discarded, 1);
    }

    #[test]
    fn test_peek() {
        let mut buf = JitterBuffer::with_defaults();
        assert!(buf.peek().is_none());
        buf.push(sample(1, 100, 50));
        assert_eq!(buf.peek().expect("should succeed in test").sequence, 1);
        assert_eq!(buf.depth(), 1); // peek doesn't consume
    }

    #[test]
    fn test_median_offset() {
        let cfg = JitterBufferConfig::default().with_max_jitter_ns(i64::MAX);
        let mut buf = JitterBuffer::new(cfg);
        buf.push(sample(1, 100, 10));
        buf.push(sample(2, 300, 10));
        buf.push(sample(3, 200, 10));

        assert_eq!(buf.median_offset_ns(), Some(200));
    }

    #[test]
    fn test_median_offset_even() {
        let cfg = JitterBufferConfig::default().with_max_jitter_ns(i64::MAX);
        let mut buf = JitterBuffer::new(cfg);
        buf.push(sample(1, 100, 10));
        buf.push(sample(2, 200, 10));

        assert_eq!(buf.median_offset_ns(), Some(150));
    }

    #[test]
    fn test_mean_delay() {
        let cfg = JitterBufferConfig::default().with_max_jitter_ns(i64::MAX);
        let mut buf = JitterBuffer::new(cfg);
        buf.push(sample(1, 0, 100));
        buf.push(sample(2, 0, 200));
        buf.push(sample(3, 0, 300));

        let mean = buf.mean_delay_ns().expect("should succeed in test");
        assert!((mean - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut buf = JitterBuffer::with_defaults();
        buf.push(sample(1, 100, 50));
        buf.push(sample(2, 200, 60));
        assert_eq!(buf.depth(), 2);

        buf.reset();
        assert!(buf.is_empty());
        assert_eq!(buf.stats().total_received, 0);
    }

    #[test]
    fn test_jitter_estimate_converges() {
        let cfg = JitterBufferConfig::default().with_max_jitter_ns(i64::MAX);
        let mut buf = JitterBuffer::new(cfg);
        // Feed samples with consistent small jitter
        for i in 0..32 {
            buf.push(sample(i, (i as i64) * 10, 50));
            if i > 0 {
                buf.pop();
            }
        }
        // Jitter estimate should be positive but bounded
        assert!(buf.jitter_estimate_ns() > 0.0);
        assert!(buf.jitter_estimate_ns() < 200.0);
    }
}
