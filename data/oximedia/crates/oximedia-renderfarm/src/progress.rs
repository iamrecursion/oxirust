// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Progress tracking for jobs.

use crate::job::JobId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Progress tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTracker {
    job_id: JobId,
    total_frames: u32,
    completed_frames: u32,
    failed_frames: u32,
    start_time: DateTime<Utc>,
    frame_times: Vec<f64>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    #[must_use]
    pub fn new(job_id: JobId, total_frames: u32) -> Self {
        Self {
            job_id,
            total_frames,
            completed_frames: 0,
            failed_frames: 0,
            start_time: Utc::now(),
            frame_times: Vec::new(),
        }
    }

    /// Record completed frame
    pub fn record_completed(&mut self, frame_time: f64) {
        self.completed_frames += 1;
        self.frame_times.push(frame_time);
    }

    /// Record failed frame
    pub fn record_failed(&mut self) {
        self.failed_frames += 1;
    }

    /// Get progress (0.0 to 1.0)
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        f64::from(self.completed_frames) / f64::from(self.total_frames)
    }

    /// Get ETA
    #[must_use]
    pub fn eta(&self) -> Option<DateTime<Utc>> {
        if self.completed_frames == 0 || self.frame_times.is_empty() {
            return None;
        }

        let avg_frame_time = self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
        let remaining_frames = self.total_frames - self.completed_frames;
        let remaining_seconds = f64::from(remaining_frames) * avg_frame_time;

        Some(Utc::now() + chrono::Duration::seconds(remaining_seconds as i64))
    }

    /// Get throughput (frames per hour)
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }

        let avg_frame_time = self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
        if avg_frame_time > 0.0 {
            3600.0 / avg_frame_time
        } else {
            0.0
        }
    }

    /// Get elapsed time (seconds)
    #[must_use]
    pub fn elapsed_time(&self) -> f64 {
        (Utc::now() - self.start_time).num_seconds() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(JobId::new(), 100);
        assert_eq!(tracker.total_frames, 100);
        assert_eq!(tracker.progress(), 0.0);
    }

    #[test]
    fn test_record_completed() {
        let mut tracker = ProgressTracker::new(JobId::new(), 100);
        tracker.record_completed(1.0);
        assert_eq!(tracker.completed_frames, 1);
        assert_eq!(tracker.progress(), 0.01);
    }

    #[test]
    fn test_throughput() {
        let mut tracker = ProgressTracker::new(JobId::new(), 100);
        tracker.record_completed(1.0);
        tracker.record_completed(1.0);
        tracker.record_completed(1.0);

        let throughput = tracker.throughput();
        assert_eq!(throughput, 3600.0);
    }
}
