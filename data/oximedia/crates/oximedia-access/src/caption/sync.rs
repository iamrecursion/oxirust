//! Caption synchronization with real-time user-adjustable offset and drift detection.

use crate::caption::Caption;
use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};

/// Synchronization quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncQuality {
    /// Basic frame-level sync (within 1 frame).
    Frame,
    /// Word-level sync (each word timed).
    Word,
    /// Phoneme-level sync (highest precision).
    Phoneme,
}

/// Describes a user's sync adjustment request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncAdjustment {
    /// Global offset in milliseconds (positive = captions later, negative = earlier).
    pub offset_ms: i64,
    /// Per-caption time scaling factor (1.0 = no change, >1.0 = stretch).
    pub time_scale: f64,
    /// Whether to snap results to frame boundaries.
    pub snap_to_frames: bool,
    /// Maximum allowed offset before warning (in ms).
    pub max_offset_ms: i64,
}

impl Default for SyncAdjustment {
    fn default() -> Self {
        Self {
            offset_ms: 0,
            time_scale: 1.0,
            snap_to_frames: true,
            max_offset_ms: 10_000,
        }
    }
}

impl SyncAdjustment {
    /// Create with a specific offset.
    #[must_use]
    pub fn with_offset(offset_ms: i64) -> Self {
        Self {
            offset_ms,
            ..Self::default()
        }
    }

    /// Validate adjustment parameters.
    pub fn validate(&self) -> AccessResult<()> {
        if self.offset_ms.abs() > self.max_offset_ms {
            return Err(AccessError::SyncError(format!(
                "Offset {}ms exceeds maximum allowed {}ms",
                self.offset_ms, self.max_offset_ms
            )));
        }
        if self.time_scale <= 0.0 {
            return Err(AccessError::SyncError(
                "Time scale must be positive".to_string(),
            ));
        }
        if self.time_scale < 0.1 || self.time_scale > 10.0 {
            return Err(AccessError::SyncError(format!(
                "Time scale {} is out of range [0.1, 10.0]",
                self.time_scale
            )));
        }
        Ok(())
    }
}

/// Analysis of caption timing drift relative to a reference timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDriftAnalysis {
    /// Average drift in milliseconds across all captions.
    pub mean_drift_ms: f64,
    /// Standard deviation of drift.
    pub std_dev_ms: f64,
    /// Maximum drift observed (absolute value).
    pub max_drift_ms: f64,
    /// Number of captions analyzed.
    pub caption_count: usize,
    /// Suggested offset to correct drift.
    pub suggested_offset_ms: i64,
    /// Whether drift appears to be linear (accumulating over time).
    pub is_linear_drift: bool,
    /// Estimated drift rate in ms per second (if linear).
    pub drift_rate_ms_per_sec: f64,
}

/// Caption synchronizer for timing adjustments with real-time offset control.
pub struct CaptionSynchronizer {
    quality: SyncQuality,
    frame_rate: f64,
    /// Current user-adjustable offset in milliseconds.
    user_offset_ms: i64,
    /// Accumulated adjustment history for undo support.
    adjustment_history: Vec<i64>,
}

impl CaptionSynchronizer {
    /// Create a new synchronizer.
    #[must_use]
    pub fn new(quality: SyncQuality, frame_rate: f64) -> Self {
        Self {
            quality,
            frame_rate,
            user_offset_ms: 0,
            adjustment_history: Vec::new(),
        }
    }

    /// Get the current sync quality.
    #[must_use]
    pub fn quality(&self) -> SyncQuality {
        self.quality
    }

    /// Get the current user offset.
    #[must_use]
    pub fn user_offset_ms(&self) -> i64 {
        self.user_offset_ms
    }

    /// Set user offset directly (for real-time slider control).
    pub fn set_user_offset(&mut self, offset_ms: i64) {
        self.adjustment_history.push(self.user_offset_ms);
        self.user_offset_ms = offset_ms;
    }

    /// Nudge user offset by a delta (for +/- button control).
    pub fn nudge_offset(&mut self, delta_ms: i64) {
        self.adjustment_history.push(self.user_offset_ms);
        self.user_offset_ms += delta_ms;
    }

    /// Undo the last offset adjustment.
    pub fn undo_offset(&mut self) -> bool {
        if let Some(prev) = self.adjustment_history.pop() {
            self.user_offset_ms = prev;
            true
        } else {
            false
        }
    }

    /// Reset offset to zero.
    pub fn reset_offset(&mut self) {
        self.adjustment_history.push(self.user_offset_ms);
        self.user_offset_ms = 0;
    }

    /// Apply the full sync adjustment (offset + scaling + snapping) to captions.
    pub fn apply_adjustment(
        &self,
        captions: &mut [Caption],
        adjustment: &SyncAdjustment,
    ) -> AccessResult<()> {
        adjustment.validate()?;

        // Find the reference point (first caption start) for time scaling
        let reference_time = captions.first().map(|c| c.subtitle.start_time).unwrap_or(0);

        for caption in captions.iter_mut() {
            // Apply time scaling relative to reference point
            let start_offset = caption.subtitle.start_time - reference_time;
            let end_offset = caption.subtitle.end_time - reference_time;

            let scaled_start =
                reference_time + (start_offset as f64 * adjustment.time_scale).round() as i64;
            let scaled_end =
                reference_time + (end_offset as f64 * adjustment.time_scale).round() as i64;

            // Apply global offset
            caption.subtitle.start_time = scaled_start + adjustment.offset_ms;
            caption.subtitle.end_time = scaled_end + adjustment.offset_ms;

            // Snap to frame boundaries if requested
            if adjustment.snap_to_frames {
                caption.subtitle.start_time = self.snap_to_frame(caption.subtitle.start_time);
                caption.subtitle.end_time = self.snap_to_frame(caption.subtitle.end_time);
            }

            // Ensure minimum duration of 1 frame after adjustment
            let min_duration = (1000.0 / self.frame_rate).ceil() as i64;
            if caption.subtitle.end_time - caption.subtitle.start_time < min_duration {
                caption.subtitle.end_time = caption.subtitle.start_time + min_duration;
            }
        }

        Ok(())
    }

    /// Apply only the user offset to captions (for real-time preview).
    pub fn apply_user_offset(&self, captions: &mut [Caption]) {
        if self.user_offset_ms != 0 {
            for caption in captions {
                caption.subtitle.start_time += self.user_offset_ms;
                caption.subtitle.end_time += self.user_offset_ms;
            }
        }
    }

    /// Analyze timing drift between captions and a set of reference timestamps.
    ///
    /// Reference timestamps represent the expected start times (from audio analysis).
    /// Returns an analysis with mean drift, suggested correction, and drift rate.
    pub fn analyze_drift(
        &self,
        captions: &[Caption],
        reference_start_times_ms: &[i64],
    ) -> AccessResult<SyncDriftAnalysis> {
        let count = captions.len().min(reference_start_times_ms.len());
        if count == 0 {
            return Err(AccessError::SyncError(
                "No captions or reference times to analyze".to_string(),
            ));
        }

        let drifts: Vec<f64> = captions
            .iter()
            .zip(reference_start_times_ms.iter())
            .map(|(c, &r)| (c.subtitle.start_time - r) as f64)
            .collect();

        let mean = drifts.iter().sum::<f64>() / drifts.len() as f64;
        let variance = drifts.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / drifts.len() as f64;
        let std_dev = variance.sqrt();
        let max_drift = drifts.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

        // Detect linear drift by comparing first-half vs second-half mean
        let (is_linear, drift_rate) = if count >= 4 {
            let half = count / 2;
            let first_half_mean: f64 = drifts[..half].iter().sum::<f64>() / half as f64;
            let second_half_mean: f64 = drifts[half..].iter().sum::<f64>() / (count - half) as f64;

            // Time span for rate calculation
            let first_ref = reference_start_times_ms[0];
            let last_ref = reference_start_times_ms[count - 1];
            let time_span_sec = (last_ref - first_ref) as f64 / 1000.0;

            let drift_diff = second_half_mean - first_half_mean;
            let is_linear = drift_diff.abs() > std_dev * 0.5; // Significant trend

            let rate = if time_span_sec > 0.0 {
                (drifts[count - 1] - drifts[0]) / time_span_sec
            } else {
                0.0
            };

            (is_linear, rate)
        } else {
            (false, 0.0)
        };

        Ok(SyncDriftAnalysis {
            mean_drift_ms: mean,
            std_dev_ms: std_dev,
            max_drift_ms: max_drift,
            caption_count: count,
            suggested_offset_ms: (-mean).round() as i64,
            is_linear_drift: is_linear,
            drift_rate_ms_per_sec: drift_rate,
        })
    }

    /// Auto-correct drift by applying the suggested offset from drift analysis.
    pub fn auto_correct_drift(
        &mut self,
        captions: &mut [Caption],
        reference_start_times_ms: &[i64],
    ) -> AccessResult<SyncDriftAnalysis> {
        let analysis = self.analyze_drift(captions, reference_start_times_ms)?;

        if analysis.is_linear_drift {
            // For linear drift, apply per-caption correction
            let count = captions.len().min(reference_start_times_ms.len());
            for (i, caption) in captions.iter_mut().enumerate().take(count) {
                let drift = caption.subtitle.start_time - reference_start_times_ms[i];
                let duration = caption.subtitle.end_time - caption.subtitle.start_time;
                caption.subtitle.start_time -= drift;
                caption.subtitle.end_time = caption.subtitle.start_time + duration;
            }
        } else {
            // For constant drift, apply uniform offset
            self.adjust_offset(captions, analysis.suggested_offset_ms);
        }

        Ok(analysis)
    }

    /// Synchronize captions to frame boundaries.
    pub fn sync_to_frames(&self, captions: &mut [Caption]) -> AccessResult<()> {
        for caption in captions {
            caption.subtitle.start_time = self.snap_to_frame(caption.subtitle.start_time);
            caption.subtitle.end_time = self.snap_to_frame(caption.subtitle.end_time);
        }

        Ok(())
    }

    /// Snap time to nearest frame boundary using floating-point precision.
    fn snap_to_frame(&self, time_ms: i64) -> i64 {
        let frame_duration = 1000.0 / self.frame_rate;
        let frame_num = (time_ms as f64 / frame_duration).round() as i64;
        (frame_num as f64 * frame_duration).round() as i64
    }

    /// Adjust timing offset for all captions.
    pub fn adjust_offset(&self, captions: &mut [Caption], offset_ms: i64) {
        for caption in captions {
            caption.subtitle.start_time += offset_ms;
            caption.subtitle.end_time += offset_ms;
        }
    }

    /// Detect and fix timing gaps.
    pub fn fix_gaps(&self, captions: &mut [Caption], max_gap_ms: i64) -> AccessResult<usize> {
        let mut fixed_count = 0;

        for i in 0..captions.len().saturating_sub(1) {
            let gap = captions[i + 1].subtitle.start_time - captions[i].subtitle.end_time;

            if gap > max_gap_ms {
                captions[i].subtitle.end_time = captions[i + 1].subtitle.start_time - 100;
                fixed_count += 1;
            }
        }

        Ok(fixed_count)
    }

    /// Detect and fix overlapping captions.
    pub fn fix_overlaps(&self, captions: &mut [Caption]) -> AccessResult<usize> {
        let mut fixed_count = 0;

        for i in 0..captions.len().saturating_sub(1) {
            if captions[i].subtitle.end_time > captions[i + 1].subtitle.start_time {
                captions[i].subtitle.end_time = captions[i + 1].subtitle.start_time;
                fixed_count += 1;
            }
        }

        Ok(fixed_count)
    }

    /// Validate caption timing.
    pub fn validate(&self, captions: &[Caption]) -> AccessResult<()> {
        for (i, caption) in captions.iter().enumerate() {
            if caption.subtitle.end_time <= caption.subtitle.start_time {
                return Err(AccessError::SyncError(format!(
                    "Caption {} has invalid timing: start={}, end={}",
                    i, caption.subtitle.start_time, caption.subtitle.end_time
                )));
            }
        }

        for i in 0..captions.len().saturating_sub(1) {
            if captions[i].subtitle.end_time > captions[i + 1].subtitle.start_time {
                return Err(AccessError::SyncError(format!(
                    "Captions {} and {} overlap",
                    i,
                    i + 1
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::CaptionType;
    use oximedia_subtitle::Subtitle;

    fn make_caption(start: i64, end: i64, text: &str) -> Caption {
        Caption::new(
            Subtitle::new(start, end, text.to_string()),
            CaptionType::Closed,
        )
    }

    #[test]
    fn test_sync_to_frames() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![make_caption(1003, 2007, "Test")];

        synchronizer
            .sync_to_frames(&mut captions)
            .expect("sync_to_frames should succeed");

        // Should snap to 24fps frame boundaries (41.67ms per frame)
        assert_eq!(captions[0].subtitle.start_time, 1000);
    }

    #[test]
    fn test_adjust_offset() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![make_caption(1000, 2000, "Test")];

        synchronizer.adjust_offset(&mut captions, 500);

        assert_eq!(captions[0].subtitle.start_time, 1500);
        assert_eq!(captions[0].subtitle.end_time, 2500);
    }

    #[test]
    fn test_fix_overlaps() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![
            make_caption(1000, 3000, "First"),
            make_caption(2000, 4000, "Second"),
        ];

        let fixed = synchronizer
            .fix_overlaps(&mut captions)
            .expect("fixed should be valid");
        assert_eq!(fixed, 1);
        assert_eq!(captions[0].subtitle.end_time, 2000);
    }

    // ============================================================
    // Real-time caption synchronization adjustment tests
    // ============================================================

    #[test]
    fn test_user_offset_set_and_get() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        assert_eq!(sync.user_offset_ms(), 0);

        sync.set_user_offset(500);
        assert_eq!(sync.user_offset_ms(), 500);

        sync.set_user_offset(-200);
        assert_eq!(sync.user_offset_ms(), -200);
    }

    #[test]
    fn test_nudge_offset() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        sync.nudge_offset(100);
        assert_eq!(sync.user_offset_ms(), 100);

        sync.nudge_offset(100);
        assert_eq!(sync.user_offset_ms(), 200);

        sync.nudge_offset(-50);
        assert_eq!(sync.user_offset_ms(), 150);
    }

    #[test]
    fn test_undo_offset() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        sync.set_user_offset(100);
        sync.set_user_offset(200);
        sync.set_user_offset(300);

        assert!(sync.undo_offset());
        assert_eq!(sync.user_offset_ms(), 200);

        assert!(sync.undo_offset());
        assert_eq!(sync.user_offset_ms(), 100);

        assert!(sync.undo_offset());
        assert_eq!(sync.user_offset_ms(), 0);

        // No more history
        assert!(!sync.undo_offset());
    }

    #[test]
    fn test_reset_offset() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        sync.set_user_offset(500);
        sync.reset_offset();
        assert_eq!(sync.user_offset_ms(), 0);

        // Should be able to undo the reset
        assert!(sync.undo_offset());
        assert_eq!(sync.user_offset_ms(), 500);
    }

    #[test]
    fn test_apply_user_offset() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        sync.set_user_offset(300);

        let mut captions = vec![
            make_caption(1000, 2000, "First"),
            make_caption(3000, 4000, "Second"),
        ];

        sync.apply_user_offset(&mut captions);

        assert_eq!(captions[0].subtitle.start_time, 1300);
        assert_eq!(captions[0].subtitle.end_time, 2300);
        assert_eq!(captions[1].subtitle.start_time, 3300);
        assert_eq!(captions[1].subtitle.end_time, 4300);
    }

    #[test]
    fn test_apply_user_offset_zero() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        let mut captions = vec![make_caption(1000, 2000, "Test")];

        sync.apply_user_offset(&mut captions);

        // No change when offset is zero
        assert_eq!(captions[0].subtitle.start_time, 1000);
    }

    #[test]
    fn test_sync_adjustment_validation() {
        let adj = SyncAdjustment::default();
        assert!(adj.validate().is_ok());

        let bad_offset = SyncAdjustment {
            offset_ms: 20_000,
            max_offset_ms: 10_000,
            ..SyncAdjustment::default()
        };
        assert!(bad_offset.validate().is_err());

        let bad_scale = SyncAdjustment {
            time_scale: -1.0,
            ..SyncAdjustment::default()
        };
        assert!(bad_scale.validate().is_err());

        let extreme_scale = SyncAdjustment {
            time_scale: 100.0,
            ..SyncAdjustment::default()
        };
        assert!(extreme_scale.validate().is_err());
    }

    #[test]
    fn test_apply_adjustment_with_offset() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        let mut captions = vec![
            make_caption(1000, 2000, "First"),
            make_caption(3000, 4000, "Second"),
        ];

        let adj = SyncAdjustment {
            offset_ms: 500,
            time_scale: 1.0,
            snap_to_frames: false,
            max_offset_ms: 10_000,
        };

        sync.apply_adjustment(&mut captions, &adj)
            .expect("adjustment should succeed");

        assert_eq!(captions[0].subtitle.start_time, 1500);
        assert_eq!(captions[1].subtitle.start_time, 3500);
    }

    #[test]
    fn test_apply_adjustment_with_time_scale() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        let mut captions = vec![
            make_caption(0, 1000, "First"),
            make_caption(2000, 3000, "Second"),
        ];

        let adj = SyncAdjustment {
            offset_ms: 0,
            time_scale: 2.0, // Double the time spread
            snap_to_frames: false,
            max_offset_ms: 10_000,
        };

        sync.apply_adjustment(&mut captions, &adj)
            .expect("adjustment should succeed");

        // Second caption's start was at offset 2000 from first caption
        // After 2x scaling: offset becomes 4000
        assert_eq!(captions[0].subtitle.start_time, 0);
        assert_eq!(captions[1].subtitle.start_time, 4000);
    }

    #[test]
    fn test_drift_analysis_constant() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        // Captions are all 200ms late
        let captions = vec![
            make_caption(1200, 2200, "A"),
            make_caption(3200, 4200, "B"),
            make_caption(5200, 6200, "C"),
            make_caption(7200, 8200, "D"),
        ];
        let references = vec![1000, 3000, 5000, 7000];

        let analysis = sync
            .analyze_drift(&captions, &references)
            .expect("drift analysis should succeed");

        assert!((analysis.mean_drift_ms - 200.0).abs() < 1.0);
        assert_eq!(analysis.suggested_offset_ms, -200);
        assert_eq!(analysis.caption_count, 4);
        // Constant drift should not be detected as linear
        assert!(!analysis.is_linear_drift);
    }

    #[test]
    fn test_drift_analysis_linear() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        // Drift increases linearly: 0ms, 100ms, 200ms, 300ms
        let captions = vec![
            make_caption(1000, 2000, "A"),
            make_caption(3100, 4100, "B"),
            make_caption(5200, 6200, "C"),
            make_caption(7300, 8300, "D"),
        ];
        let references = vec![1000, 3000, 5000, 7000];

        let analysis = sync
            .analyze_drift(&captions, &references)
            .expect("drift analysis should succeed");

        assert!(analysis.is_linear_drift);
        assert!(analysis.drift_rate_ms_per_sec > 0.0);
    }

    #[test]
    fn test_drift_analysis_empty() {
        let sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);
        let result = sync.analyze_drift(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_correct_constant_drift() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![
            make_caption(1200, 2200, "A"),
            make_caption(3200, 4200, "B"),
            make_caption(5200, 6200, "C"),
            make_caption(7200, 8200, "D"),
        ];
        let references = vec![1000, 3000, 5000, 7000];

        let analysis = sync
            .auto_correct_drift(&mut captions, &references)
            .expect("auto-correct should succeed");

        assert!((analysis.mean_drift_ms - 200.0).abs() < 1.0);

        // After correction, captions should be close to reference
        assert_eq!(captions[0].subtitle.start_time, 1000);
        assert_eq!(captions[1].subtitle.start_time, 3000);
    }

    #[test]
    fn test_auto_correct_linear_drift() {
        let mut sync = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![
            make_caption(1000, 2000, "A"),
            make_caption(3100, 4100, "B"),
            make_caption(5200, 6200, "C"),
            make_caption(7300, 8300, "D"),
        ];
        let references = vec![1000, 3000, 5000, 7000];

        let _ = sync
            .auto_correct_drift(&mut captions, &references)
            .expect("auto-correct should succeed");

        // Each caption should be individually corrected
        assert_eq!(captions[0].subtitle.start_time, 1000);
        assert_eq!(captions[1].subtitle.start_time, 3000);
        assert_eq!(captions[2].subtitle.start_time, 5000);
        assert_eq!(captions[3].subtitle.start_time, 7000);
    }
}
