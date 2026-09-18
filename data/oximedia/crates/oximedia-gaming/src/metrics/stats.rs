//! Stream statistics.

use std::time::Duration;

/// Stream statistics collector.
pub struct StatisticsCollector {
    stats: StreamStatistics,
}

/// Stream statistics.
#[derive(Debug, Clone, Default)]
pub struct StreamStatistics {
    /// Total frames captured
    pub frames_captured: u64,
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Total frames dropped
    pub frames_dropped: u64,
    /// Average bitrate (kbps)
    pub average_bitrate: f64,
    /// Peak bitrate (kbps)
    pub peak_bitrate: f64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Stream duration
    pub duration: Duration,
}

impl StatisticsCollector {
    /// Create a new statistics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: StreamStatistics::default(),
        }
    }

    /// Update statistics.
    pub fn update(&mut self) {
        // In a real implementation, this would collect streaming stats
    }

    /// Get current statistics.
    #[must_use]
    pub fn statistics(&self) -> &StreamStatistics {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.stats = StreamStatistics::default();
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let collector = StatisticsCollector::new();
        assert_eq!(collector.statistics().frames_captured, 0);
    }

    #[test]
    fn test_reset() {
        let mut collector = StatisticsCollector::new();
        collector.stats.frames_captured = 100;
        collector.reset();
        assert_eq!(collector.statistics().frames_captured, 0);
    }
}
