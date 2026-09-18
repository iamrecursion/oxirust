//! Signal status and error monitoring.

pub mod errors;
pub mod signal;

use serde::{Deserialize, Serialize};

pub use errors::ErrorLogger;
pub use signal::SignalMonitor;

/// Signal quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalQuality {
    /// Excellent signal.
    Excellent,

    /// Good signal.
    Good,

    /// Fair signal.
    Fair,

    /// Poor signal.
    Poor,

    /// No signal.
    NoSignal,
}

/// Monitoring status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitoringStatus {
    /// Signal quality.
    pub signal_quality: Option<SignalQuality>,

    /// Video present.
    pub video_present: bool,

    /// Audio present.
    pub audio_present: bool,

    /// Frame count.
    pub frame_count: u64,

    /// Error count.
    pub error_count: u64,

    /// Warning count.
    pub warning_count: u64,
}

/// Signal status tracker.
#[derive(Default)]
pub struct SignalStatus {
    status: MonitoringStatus,
}

impl SignalStatus {
    /// Update video status.
    pub fn update_video(&mut self, _width: u32, _height: u32, frame_count: u64) {
        self.status.video_present = true;
        self.status.frame_count = frame_count;
        self.status.signal_quality = Some(SignalQuality::Good);
    }

    /// Update audio status.
    pub fn update_audio(&mut self, _sample_count: u64) {
        self.status.audio_present = true;
    }

    /// Get current status.
    #[must_use]
    pub const fn status(&self) -> &MonitoringStatus {
        &self.status
    }
}

/// Error log.
#[derive(Default)]
pub struct ErrorLog {
    errors: Vec<String>,
}

impl ErrorLog {
    /// Create a new error log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Get errors.
    #[must_use]
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Clear errors.
    pub fn clear(&mut self) {
        self.errors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_status() {
        let mut status = SignalStatus::default();
        status.update_video(1920, 1080, 100);
        assert!(status.status().video_present);
    }

    #[test]
    fn test_error_log() {
        let mut log = ErrorLog::new();
        log.add_error("Test error".to_string());
        assert_eq!(log.errors().len(), 1);
    }
}
