//! Stream-specific metrics tracking.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Stream metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    /// Stream key.
    pub stream_key: String,

    /// Application name.
    pub app_name: String,

    /// Total bytes ingested.
    pub bytes_ingested: u64,

    /// Total bytes served.
    pub bytes_served: u64,

    /// Total frames processed.
    pub frames_processed: u64,

    /// Current bitrate (bits per second).
    pub current_bitrate: u64,

    /// Average bitrate (bits per second).
    pub average_bitrate: u64,

    /// Peak bitrate (bits per second).
    pub peak_bitrate: u64,

    /// Stream duration.
    pub duration: Duration,

    /// Viewer metrics.
    pub viewers: ViewerMetrics,

    /// Bandwidth metrics.
    pub bandwidth: BandwidthMetrics,
}

impl StreamMetrics {
    /// Creates new stream metrics.
    #[must_use]
    pub fn new(app_name: impl Into<String>, stream_key: impl Into<String>) -> Self {
        Self {
            stream_key: stream_key.into(),
            app_name: app_name.into(),
            bytes_ingested: 0,
            bytes_served: 0,
            frames_processed: 0,
            current_bitrate: 0,
            average_bitrate: 0,
            peak_bitrate: 0,
            duration: Duration::from_secs(0),
            viewers: ViewerMetrics::default(),
            bandwidth: BandwidthMetrics::default(),
        }
    }

    /// Updates bitrate metrics.
    pub fn update_bitrate(&mut self, bytes: u64, duration: Duration) {
        if duration.as_secs() > 0 {
            self.current_bitrate = (bytes * 8) / duration.as_secs();

            if self.current_bitrate > self.peak_bitrate {
                self.peak_bitrate = self.current_bitrate;
            }

            if self.duration.as_secs() > 0 {
                self.average_bitrate = (self.bytes_ingested * 8) / self.duration.as_secs();
            }
        }
    }
}

/// Viewer metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViewerMetrics {
    /// Current viewer count.
    pub current: u64,

    /// Peak viewer count.
    pub peak: u64,

    /// Total unique viewers.
    pub total_unique: u64,

    /// Average concurrent viewers.
    pub average_concurrent: f64,
}

impl ViewerMetrics {
    /// Increments viewer count.
    pub fn add_viewer(&mut self) {
        self.current += 1;
        if self.current > self.peak {
            self.peak = self.current;
        }
    }

    /// Decrements viewer count.
    pub fn remove_viewer(&mut self) {
        if self.current > 0 {
            self.current -= 1;
        }
    }
}

/// Bandwidth metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BandwidthMetrics {
    /// Total ingress (bytes).
    pub ingress: u64,

    /// Total egress (bytes).
    pub egress: u64,

    /// Peak ingress rate (bytes per second).
    pub peak_ingress_rate: u64,

    /// Peak egress rate (bytes per second).
    pub peak_egress_rate: u64,

    /// Average ingress rate (bytes per second).
    pub average_ingress_rate: u64,

    /// Average egress rate (bytes per second).
    pub average_egress_rate: u64,
}

impl BandwidthMetrics {
    /// Records ingress.
    pub fn record_ingress(&mut self, bytes: u64) {
        self.ingress += bytes;
    }

    /// Records egress.
    pub fn record_egress(&mut self, bytes: u64) {
        self.egress += bytes;
    }

    /// Updates rates.
    pub fn update_rates(&mut self, duration: Duration) {
        if duration.as_secs() > 0 {
            self.average_ingress_rate = self.ingress / duration.as_secs();
            self.average_egress_rate = self.egress / duration.as_secs();
        }
    }
}
