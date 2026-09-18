//! Virtual genlock simulation for multi-camera post-production.
//!
//! This module simulates hardware genlock for frame-accurate synchronization.

use super::{SyncConfig, SyncMethod, SyncOffset, SyncResult, Synchronizer};
use crate::{AngleId, FrameNumber, Result};

/// Virtual genlock synchronizer
#[derive(Debug)]
pub struct GenlockSync {
    /// Master clock reference (frames per second)
    master_clock: f64,
    /// Phase offsets for each angle (in frames)
    phase_offsets: Vec<f64>,
    /// Frame timestamps for each angle
    timestamps: Vec<Vec<f64>>,
}

impl GenlockSync {
    /// Create a new genlock synchronizer
    #[must_use]
    pub fn new(angle_count: usize, master_clock: f64) -> Self {
        Self {
            master_clock,
            phase_offsets: vec![0.0; angle_count],
            timestamps: vec![Vec::new(); angle_count],
        }
    }

    /// Set master clock frequency
    pub fn set_master_clock(&mut self, clock: f64) {
        self.master_clock = clock;
    }

    /// Get master clock frequency
    #[must_use]
    pub fn master_clock(&self) -> f64 {
        self.master_clock
    }

    /// Add timestamps for an angle
    pub fn add_timestamps(&mut self, angle: AngleId, timestamps: Vec<f64>) {
        if angle < self.timestamps.len() {
            self.timestamps[angle] = timestamps;
        }
    }

    /// Calculate phase offset for an angle
    ///
    /// # Errors
    ///
    /// Returns an error if calculation fails
    pub fn calculate_phase_offset(&mut self, angle: AngleId) -> Result<f64> {
        if angle >= self.timestamps.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle));
        }

        let timestamps = &self.timestamps[angle];
        if timestamps.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timestamps for angle".to_string(),
            ));
        }

        // Calculate phase offset relative to master clock
        let frame_period = 1.0 / self.master_clock;
        let first_timestamp = timestamps[0];
        let phase = (first_timestamp % frame_period) / frame_period;

        self.phase_offsets[angle] = phase;
        Ok(phase)
    }

    /// Synchronize angle to master clock
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    pub fn sync_to_master(&self, angle: AngleId) -> Result<SyncOffset> {
        if angle >= self.phase_offsets.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle));
        }

        let phase = self.phase_offsets[angle];
        let frame_offset = phase.floor() as i64;
        let sub_frame = phase - frame_offset as f64;

        Ok(SyncOffset::new(angle, frame_offset, sub_frame, 1.0))
    }

    /// Generate virtual genlock signal
    #[must_use]
    pub fn generate_signal(&self, duration_frames: u64) -> Vec<f64> {
        let mut signal = Vec::with_capacity(duration_frames as usize);
        let frame_period = 1.0 / self.master_clock;

        for frame in 0..duration_frames {
            let timestamp = frame as f64 * frame_period;
            signal.push(timestamp);
        }

        signal
    }

    /// Detect genlock drift
    #[must_use]
    pub fn detect_drift(&self, angle: AngleId) -> f64 {
        if angle >= self.timestamps.len() {
            return 0.0;
        }

        let timestamps = &self.timestamps[angle];
        if timestamps.len() < 2 {
            return 0.0;
        }

        // Calculate actual frame rate from timestamps
        let duration = timestamps[timestamps.len() - 1] - timestamps[0];
        let actual_rate = (timestamps.len() - 1) as f64 / duration;

        // Calculate drift in parts per million (PPM)

        (actual_rate - self.master_clock) / self.master_clock * 1_000_000.0
    }

    /// Correct genlock drift
    pub fn correct_drift(&mut self, angle: AngleId, drift_ppm: f64) {
        if angle >= self.timestamps.len() {
            return;
        }

        let correction_factor = 1.0 + (drift_ppm / 1_000_000.0);
        let timestamps = &mut self.timestamps[angle];

        for timestamp in timestamps {
            *timestamp *= correction_factor;
        }
    }

    /// Lock angle to specific phase
    pub fn lock_phase(&mut self, angle: AngleId, phase: f64) {
        if angle < self.phase_offsets.len() {
            self.phase_offsets[angle] = phase.rem_euclid(1.0);
        }
    }

    /// Get phase offset for angle
    #[must_use]
    pub fn get_phase(&self, angle: AngleId) -> f64 {
        self.phase_offsets.get(angle).copied().unwrap_or(0.0)
    }

    /// Check if angle is locked to master
    #[must_use]
    pub fn is_locked(&self, angle: AngleId) -> bool {
        if angle >= self.timestamps.len() {
            return false;
        }

        let drift = self.detect_drift(angle).abs();
        drift < 10.0 // Less than 10 PPM drift
    }

    /// Align frame to genlock signal
    #[must_use]
    pub fn align_frame(&self, frame: FrameNumber, angle: AngleId) -> FrameNumber {
        if angle >= self.phase_offsets.len() {
            return frame;
        }

        let phase = self.phase_offsets[angle];

        (frame as f64 + phase).round() as FrameNumber
    }

    /// Calculate vsync timing
    #[must_use]
    pub fn calculate_vsync(&self, frame: FrameNumber) -> f64 {
        frame as f64 / self.master_clock
    }

    /// Generate tri-level sync signal (for HD/3G-SDI)
    #[must_use]
    pub fn generate_trilevel_sync(&self, samples_per_frame: usize) -> Vec<f32> {
        let mut signal = Vec::with_capacity(samples_per_frame);

        // Simplified tri-level sync pattern
        let sync_width = samples_per_frame / 20; // ~5% of line

        for i in 0..samples_per_frame {
            let value = if i < sync_width {
                -0.3 // Negative sync
            } else if i < sync_width * 2 {
                0.3 // Positive sync
            } else if i < sync_width * 3 {
                -0.3 // Negative sync
            } else {
                0.0 // Blanking level
            };
            signal.push(value);
        }

        signal
    }
}

impl Synchronizer for GenlockSync {
    fn synchronize(&self, _config: &SyncConfig) -> Result<SyncResult> {
        if self.timestamps.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timestamps available".to_string(),
            ));
        }

        let mut offsets = Vec::new();
        let reference_angle = 0;

        // Calculate offsets for each angle
        for angle in 0..self.timestamps.len() {
            let offset = self.sync_to_master(angle)?;
            offsets.push(offset);
        }

        // Check if all angles are locked
        let all_locked = (0..self.timestamps.len()).all(|a| self.is_locked(a));
        let confidence = if all_locked { 1.0 } else { 0.8 };

        Ok(SyncResult {
            reference_angle,
            offsets,
            confidence,
            method: SyncMethod::Genlock,
        })
    }

    fn method(&self) -> SyncMethod {
        SyncMethod::Genlock
    }

    fn is_reliable(&self) -> bool {
        !self.timestamps.is_empty() && (0..self.timestamps.len()).all(|a| self.is_locked(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genlock_creation() {
        let genlock = GenlockSync::new(3, 25.0);
        assert_eq!(genlock.phase_offsets.len(), 3);
        assert_eq!(genlock.master_clock, 25.0);
    }

    #[test]
    fn test_set_master_clock() {
        let mut genlock = GenlockSync::new(2, 25.0);
        genlock.set_master_clock(30.0);
        assert_eq!(genlock.master_clock(), 30.0);
    }

    #[test]
    fn test_generate_signal() {
        let genlock = GenlockSync::new(1, 25.0);
        let signal = genlock.generate_signal(100);
        assert_eq!(signal.len(), 100);
        assert!((signal[25] - 1.0).abs() < 0.01); // Frame 25 = 1 second at 25fps
    }

    #[test]
    fn test_lock_phase() {
        let mut genlock = GenlockSync::new(1, 25.0);
        genlock.lock_phase(0, 0.5);
        assert_eq!(genlock.get_phase(0), 0.5);
    }

    #[test]
    fn test_calculate_vsync() {
        let genlock = GenlockSync::new(1, 25.0);
        let vsync = genlock.calculate_vsync(25);
        assert!((vsync - 1.0).abs() < 0.01); // 25 frames at 25fps = 1 second
    }

    #[test]
    fn test_generate_trilevel_sync() {
        let genlock = GenlockSync::new(1, 25.0);
        let sync = genlock.generate_trilevel_sync(1000);
        assert_eq!(sync.len(), 1000);
        assert!(sync[0] < 0.0); // Starts with negative sync
    }

    #[test]
    fn test_align_frame() {
        let mut genlock = GenlockSync::new(1, 25.0);
        genlock.lock_phase(0, 0.5);
        let aligned = genlock.align_frame(100, 0);
        assert_eq!(aligned, 101); // Rounded with 0.5 phase offset
    }
}
