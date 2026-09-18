//! Synchronization methods for multi-camera production.

pub mod audio;
pub mod cross_correlate;
pub mod drift;
pub mod genlock;
pub mod temporal;
pub mod timecode;
pub mod visual;

use crate::{AngleId, Result};

/// Synchronization method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethod {
    /// Audio cross-correlation
    Audio,
    /// Timecode-based (LTC/VITC/SMPTE)
    Timecode,
    /// Visual markers (flash, clapper)
    Visual,
    /// Manual offset
    Manual,
    /// Genlock (hardware sync)
    Genlock,
    /// Combined methods
    Hybrid,
}

/// Synchronization result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Reference angle (usually angle 0)
    pub reference_angle: AngleId,
    /// Offsets for each angle in frames (positive = delay this angle)
    pub offsets: Vec<SyncOffset>,
    /// Synchronization confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Method used for synchronization
    pub method: SyncMethod,
}

/// Offset for a single camera angle
#[derive(Debug, Clone, Copy)]
pub struct SyncOffset {
    /// Angle identifier
    pub angle: AngleId,
    /// Offset in frames (positive = this angle is ahead, needs delay)
    pub frames: i64,
    /// Sub-frame offset (0.0 to 1.0)
    pub sub_frame: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl SyncOffset {
    /// Create a new sync offset
    #[must_use]
    pub fn new(angle: AngleId, frames: i64, sub_frame: f64, confidence: f64) -> Self {
        Self {
            angle,
            frames,
            sub_frame,
            confidence,
        }
    }

    /// Get total offset in frames (including sub-frame)
    #[must_use]
    pub fn total_frames(&self) -> f64 {
        self.frames as f64 + self.sub_frame
    }

    /// Convert offset to samples at given sample rate
    #[must_use]
    pub fn to_samples(&self, sample_rate: u32, frame_rate: f64) -> i64 {
        let samples_per_frame = f64::from(sample_rate) / frame_rate;
        (self.total_frames() * samples_per_frame) as i64
    }

    /// Convert offset to seconds
    #[must_use]
    pub fn to_seconds(&self, frame_rate: f64) -> f64 {
        self.total_frames() / frame_rate
    }
}

/// Synchronization configuration
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Preferred synchronization method
    pub method: SyncMethod,
    /// Maximum offset to search (frames)
    pub max_offset: u32,
    /// Sub-frame accuracy enabled
    pub sub_frame_accuracy: bool,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Sample rate for audio sync
    pub sample_rate: u32,
    /// Frame rate
    pub frame_rate: f64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            method: SyncMethod::Audio,
            max_offset: 600, // ±25 seconds at 24fps
            sub_frame_accuracy: true,
            min_confidence: 0.7,
            sample_rate: 48000,
            frame_rate: 24.0,
        }
    }
}

/// Synchronizer trait for different sync methods
pub trait Synchronizer {
    /// Synchronize multiple camera angles
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    fn synchronize(&self, config: &SyncConfig) -> Result<SyncResult>;

    /// Get the synchronization method
    fn method(&self) -> SyncMethod;

    /// Check if synchronization is reliable
    fn is_reliable(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_offset() {
        let offset = SyncOffset::new(1, 10, 0.5, 0.95);
        assert_eq!(offset.angle, 1);
        assert_eq!(offset.frames, 10);
        assert_eq!(offset.sub_frame, 0.5);
        assert!((offset.total_frames() - 10.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_offset_conversion() {
        let offset = SyncOffset::new(1, 24, 0.0, 0.95);
        assert!((offset.to_seconds(24.0) - 1.0).abs() < f64::EPSILON);
        assert_eq!(offset.to_samples(48000, 24.0), 48000);
    }

    #[test]
    fn test_default_config() {
        let config = SyncConfig::default();
        assert_eq!(config.method, SyncMethod::Audio);
        assert_eq!(config.max_offset, 600);
        assert!(config.sub_frame_accuracy);
        assert_eq!(config.sample_rate, 48000);
    }
}
