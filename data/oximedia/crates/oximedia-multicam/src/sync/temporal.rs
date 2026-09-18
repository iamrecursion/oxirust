//! Temporal synchronization for multi-camera production.
//!
//! This module provides frame-accurate temporal synchronization across multiple cameras.

use super::{SyncConfig, SyncMethod, SyncOffset, SyncResult, Synchronizer};
use crate::{AngleId, FrameNumber, Result};

/// Temporal synchronizer
#[derive(Debug)]
pub struct TemporalSync {
    /// Frame timestamps for each angle
    timestamps: Vec<Vec<f64>>,
    /// Frame rate for each angle
    frame_rates: Vec<f64>,
}

impl TemporalSync {
    /// Create a new temporal synchronizer
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            timestamps: vec![Vec::new(); angle_count],
            frame_rates: vec![0.0; angle_count],
        }
    }

    /// Add timestamps for an angle
    pub fn add_timestamps(&mut self, angle: AngleId, timestamps: Vec<f64>, frame_rate: f64) {
        if angle < self.timestamps.len() {
            self.timestamps[angle] = timestamps;
            self.frame_rates[angle] = frame_rate;
        }
    }

    /// Find temporal offset between two angles using timestamp correlation
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    pub fn find_offset(&self, angle_a: AngleId, angle_b: AngleId) -> Result<SyncOffset> {
        if angle_a >= self.timestamps.len() || angle_b >= self.timestamps.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle_a.max(angle_b)));
        }

        let ts_a = &self.timestamps[angle_a];
        let ts_b = &self.timestamps[angle_b];

        if ts_a.is_empty() || ts_b.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timestamps available".to_string(),
            ));
        }

        // Find the time offset by matching timestamp patterns
        let offset = self.correlate_timestamps(ts_a, ts_b);

        Ok(SyncOffset::new(angle_b, offset.0, offset.1, offset.2))
    }

    /// Correlate timestamps between two sequences
    fn correlate_timestamps(&self, ts_a: &[f64], ts_b: &[f64]) -> (i64, f64, f64) {
        // Simple correlation: find the offset that minimizes timestamp differences
        let mut best_offset = 0i64;
        let mut best_score = 0.0f64;
        let max_offset = 600; // Search range

        for offset in -max_offset..=max_offset {
            let score = self.compute_correlation_score(ts_a, ts_b, offset);
            if score > best_score {
                best_score = score;
                best_offset = offset;
            }
        }

        // Refine with sub-frame accuracy
        let sub_frame = self.refine_offset(ts_a, ts_b, best_offset);
        let confidence = (best_score.min(1.0)).max(0.0);

        (best_offset, sub_frame, confidence)
    }

    /// Compute correlation score for a given offset
    fn compute_correlation_score(&self, ts_a: &[f64], ts_b: &[f64], offset: i64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for (i, &t_a) in ts_a.iter().enumerate() {
            let b_idx = (i as i64 + offset) as usize;
            if b_idx < ts_b.len() {
                let diff = (t_a - ts_b[b_idx]).abs();
                sum += (-diff * 10.0).exp(); // Gaussian-like scoring
                count += 1;
            }
        }

        if count > 0 {
            sum / f64::from(count)
        } else {
            0.0
        }
    }

    /// Refine offset to sub-frame accuracy
    fn refine_offset(&self, ts_a: &[f64], ts_b: &[f64], offset: i64) -> f64 {
        let mut best_sub = 0.0;
        let mut best_score = 0.0;

        // Search sub-frame offsets in 0.1 frame increments
        for sub_offset in -10..=10 {
            let sub = f64::from(sub_offset) * 0.1;
            let score = self.compute_subframe_score(ts_a, ts_b, offset, sub);
            if score > best_score {
                best_score = score;
                best_sub = sub;
            }
        }

        best_sub
    }

    /// Compute sub-frame correlation score
    fn compute_subframe_score(&self, ts_a: &[f64], ts_b: &[f64], offset: i64, sub: f64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        let total_offset = offset as f64 + sub;

        for (i, &t_a) in ts_a.iter().enumerate() {
            let b_idx = (i as f64 + total_offset) as usize;
            if b_idx < ts_b.len() {
                let b_frac = (i as f64 + total_offset) - b_idx as f64;
                let t_b_interp = if b_idx + 1 < ts_b.len() {
                    ts_b[b_idx] * (1.0 - b_frac) + ts_b[b_idx + 1] * b_frac
                } else {
                    ts_b[b_idx]
                };
                let diff = (t_a - t_b_interp).abs();
                sum += (-diff * 10.0).exp();
                count += 1;
            }
        }

        if count > 0 {
            sum / f64::from(count)
        } else {
            0.0
        }
    }

    /// Detect frame rate drift between angles
    #[must_use]
    pub fn detect_drift(&self, angle_a: AngleId, angle_b: AngleId) -> f64 {
        if angle_a >= self.frame_rates.len() || angle_b >= self.frame_rates.len() {
            return 0.0;
        }

        let rate_a = self.frame_rates[angle_a];
        let rate_b = self.frame_rates[angle_b];

        (rate_a - rate_b) / rate_a
    }

    /// Align frame numbers between two angles
    #[must_use]
    pub fn align_frames(
        &self,
        frame: FrameNumber,
        from_angle: AngleId,
        to_angle: AngleId,
    ) -> FrameNumber {
        if from_angle >= self.timestamps.len() || to_angle >= self.timestamps.len() {
            return frame;
        }

        let rate_from = self.frame_rates[from_angle];
        let rate_to = self.frame_rates[to_angle];

        if rate_from == 0.0 || rate_to == 0.0 {
            return frame;
        }

        // Convert frame number through time
        let time = frame as f64 / rate_from;
        (time * rate_to) as FrameNumber
    }
}

impl Synchronizer for TemporalSync {
    fn synchronize(&self, _config: &SyncConfig) -> Result<SyncResult> {
        if self.timestamps.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timestamps available".to_string(),
            ));
        }

        let mut offsets = Vec::new();
        let reference_angle = 0;

        // Calculate offsets relative to first angle
        for angle in 1..self.timestamps.len() {
            let offset = self.find_offset(reference_angle, angle)?;
            offsets.push(offset);
        }

        // Add zero offset for reference angle
        offsets.insert(0, SyncOffset::new(reference_angle, 0, 0.0, 1.0));

        // Calculate average confidence
        let confidence = offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64;

        Ok(SyncResult {
            reference_angle,
            offsets,
            confidence,
            method: SyncMethod::Audio, // Will be overridden by actual method
        })
    }

    fn method(&self) -> SyncMethod {
        SyncMethod::Audio
    }

    fn is_reliable(&self) -> bool {
        !self.timestamps.is_empty() && self.timestamps.iter().all(|ts| !ts.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_sync_creation() {
        let sync = TemporalSync::new(3);
        assert_eq!(sync.timestamps.len(), 3);
        assert_eq!(sync.frame_rates.len(), 3);
    }

    #[test]
    fn test_add_timestamps() {
        let mut sync = TemporalSync::new(2);
        let timestamps = vec![0.0, 0.04, 0.08, 0.12]; // 25fps
        sync.add_timestamps(0, timestamps.clone(), 25.0);
        assert_eq!(sync.timestamps[0], timestamps);
        assert_eq!(sync.frame_rates[0], 25.0);
    }

    #[test]
    fn test_drift_detection() {
        let mut sync = TemporalSync::new(2);
        sync.frame_rates[0] = 25.0;
        sync.frame_rates[1] = 24.0;
        let drift = sync.detect_drift(0, 1);
        assert!((drift - 0.04).abs() < 0.01); // ~4% drift
    }

    #[test]
    fn test_align_frames() {
        let mut sync = TemporalSync::new(2);
        sync.frame_rates[0] = 25.0;
        sync.frame_rates[1] = 30.0;
        let aligned = sync.align_frames(25, 0, 1);
        assert_eq!(aligned, 30); // 1 second at different rates
    }
}
