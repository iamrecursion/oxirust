//! Multi-clip synchronization utilities.
//!
//! Provides types and algorithms for aligning multiple clips to a common
//! timeline using various synchronization methods.

#![allow(dead_code)]

/// The method used to synchronize clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethod {
    /// Align using embedded timecode values.
    Timecode,
    /// Align using a clapper-board audio transient.
    AudioClap,
    /// Align using full waveform cross-correlation.
    WaveformMatch,
    /// Alignment was set manually by the user.
    Manual,
}

impl SyncMethod {
    /// Return the expected accuracy in frames for this method.
    #[must_use]
    pub fn accuracy_frames(&self) -> f32 {
        match self {
            Self::Timecode => 0.0,
            Self::AudioClap => 1.0,
            Self::WaveformMatch => 0.5,
            Self::Manual => 2.0,
        }
    }
}

/// A synchronization point that maps one clip frame to the shared timeline.
#[derive(Debug, Clone)]
pub struct SyncPoint {
    /// Identifier of the clip this point belongs to.
    pub clip_id: String,
    /// Frame number of the sync event within the clip.
    pub frame: u64,
    /// Confidence of the detected sync (0.0 – 1.0).
    pub confidence: f32,
}

impl SyncPoint {
    /// Create a new sync point.
    #[must_use]
    pub fn new(clip_id: &str, frame: u64, confidence: f32) -> Self {
        Self {
            clip_id: clip_id.to_owned(),
            frame,
            confidence,
        }
    }

    /// Return `true` when confidence is high enough to be considered reliable.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.7
    }
}

/// Detects the frame of a clap event in a mono audio signal.
pub struct ClapDetector;

impl ClapDetector {
    /// Detect a clap onset by finding the maximum energy transient in the first
    /// 10 seconds of audio.
    ///
    /// Returns the sample index of the peak, or `None` if `energy` is empty
    /// or `sample_rate` is zero.
    #[must_use]
    pub fn detect(energy: &[f32], sample_rate: u32) -> Option<u64> {
        if energy.is_empty() || sample_rate == 0 {
            return None;
        }
        // Limit search to the first 10 seconds
        let search_len = ((sample_rate as usize) * 10).min(energy.len());
        let window = &energy[..search_len];

        let mut best_idx = 0usize;
        let mut best_val = window[0];
        for (i, &v) in window.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Some(best_idx as u64)
    }
}

/// Tracks sync points for multiple clips relative to a single reference clip.
#[derive(Debug, Clone)]
pub struct MultiClipSync {
    /// ID of the clip used as the reference (offset 0).
    pub reference_id: String,
    /// All registered sync points.
    pub sync_points: Vec<SyncPoint>,
}

impl MultiClipSync {
    /// Create a new `MultiClipSync` with the given reference clip.
    #[must_use]
    pub fn new(reference_id: &str) -> Self {
        Self {
            reference_id: reference_id.to_owned(),
            sync_points: Vec::new(),
        }
    }

    /// Change the reference clip.
    pub fn set_reference(&mut self, id: &str) {
        self.reference_id = id.to_owned();
    }

    /// Register a sync point.
    pub fn add_sync(&mut self, point: SyncPoint) {
        self.sync_points.push(point);
    }

    /// Return the signed frame offset for the given clip relative to the reference,
    /// or `None` if the clip or the reference has no sync point.
    #[must_use]
    pub fn offset_for(&self, clip_id: &str) -> Option<i64> {
        let ref_frame = self
            .sync_points
            .iter()
            .find(|sp| sp.clip_id == self.reference_id)
            .map(|sp| sp.frame)?;
        let clip_frame = self
            .sync_points
            .iter()
            .find(|sp| sp.clip_id == clip_id)
            .map(|sp| sp.frame)?;
        Some(clip_frame as i64 - ref_frame as i64)
    }

    /// Return the IDs of all clips that have a sync point registered.
    #[must_use]
    pub fn synchronized_clips(&self) -> Vec<&str> {
        self.sync_points
            .iter()
            .map(|sp| sp.clip_id.as_str())
            .collect()
    }
}

/// A summary report produced after a synchronization attempt.
#[derive(Debug, Clone)]
pub struct SyncReport {
    /// Whether synchronization succeeded.
    pub success: bool,
    /// Which method was used.
    pub method: SyncMethod,
    /// Maximum offset (in frames) between any two clips.
    pub max_offset_frames: f32,
    /// Overall confidence of the synchronization.
    pub confidence: f32,
}

impl SyncReport {
    /// Create a new sync report.
    #[must_use]
    pub fn new(success: bool, method: SyncMethod, max_offset_frames: f32, confidence: f32) -> Self {
        Self {
            success,
            method,
            max_offset_frames,
            confidence,
        }
    }

    /// Return `true` if the sync is within acceptable accuracy for its method
    /// and confidence is reliable.
    #[must_use]
    pub fn is_accurate(&self) -> bool {
        self.success
            && self.confidence >= 0.7
            && self.max_offset_frames <= self.method.accuracy_frames() + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SyncMethod ---

    #[test]
    fn test_timecode_accuracy() {
        assert_eq!(SyncMethod::Timecode.accuracy_frames(), 0.0);
    }

    #[test]
    fn test_audio_clap_accuracy() {
        assert_eq!(SyncMethod::AudioClap.accuracy_frames(), 1.0);
    }

    #[test]
    fn test_waveform_accuracy() {
        assert_eq!(SyncMethod::WaveformMatch.accuracy_frames(), 0.5);
    }

    #[test]
    fn test_manual_accuracy() {
        assert_eq!(SyncMethod::Manual.accuracy_frames(), 2.0);
    }

    // --- SyncPoint ---

    #[test]
    fn test_sync_point_reliable_above_threshold() {
        let sp = SyncPoint::new("clip_a", 100, 0.9);
        assert!(sp.is_reliable());
    }

    #[test]
    fn test_sync_point_reliable_at_threshold() {
        let sp = SyncPoint::new("clip_a", 100, 0.7);
        assert!(sp.is_reliable());
    }

    #[test]
    fn test_sync_point_not_reliable() {
        let sp = SyncPoint::new("clip_a", 100, 0.5);
        assert!(!sp.is_reliable());
    }

    // --- ClapDetector ---

    #[test]
    fn test_clap_detector_empty() {
        assert!(ClapDetector::detect(&[], 48000).is_none());
    }

    #[test]
    fn test_clap_detector_zero_sample_rate() {
        let energy = vec![0.1f32; 100];
        assert!(ClapDetector::detect(&energy, 0).is_none());
    }

    #[test]
    fn test_clap_detector_finds_peak() {
        let mut energy = vec![0.1f32; 1000];
        energy[300] = 10.0; // sharp transient at index 300
        let result = ClapDetector::detect(&energy, 100);
        assert_eq!(result, Some(300));
    }

    #[test]
    fn test_clap_detector_respects_10s_window() {
        // Place the true peak beyond 10 s and a smaller peak within 10 s.
        let mut energy = vec![0.0f32; 20_000];
        energy[500] = 5.0; // within first 10 s at 1000 Hz sample rate
        energy[15_000] = 100.0; // beyond 10 s — should be ignored
        let result = ClapDetector::detect(&energy, 1000);
        assert_eq!(result, Some(500));
    }

    // --- MultiClipSync ---

    #[test]
    fn test_multi_clip_sync_new() {
        let mcs = MultiClipSync::new("ref");
        assert_eq!(mcs.reference_id, "ref");
        assert!(mcs.sync_points.is_empty());
    }

    #[test]
    fn test_set_reference() {
        let mut mcs = MultiClipSync::new("old");
        mcs.set_reference("new");
        assert_eq!(mcs.reference_id, "new");
    }

    #[test]
    fn test_offset_for_no_reference() {
        let mcs = MultiClipSync::new("ref");
        assert!(mcs.offset_for("clip_a").is_none());
    }

    #[test]
    fn test_offset_for_returns_correct_value() {
        let mut mcs = MultiClipSync::new("ref");
        mcs.add_sync(SyncPoint::new("ref", 100, 1.0));
        mcs.add_sync(SyncPoint::new("cam2", 120, 0.9));
        assert_eq!(mcs.offset_for("cam2"), Some(20));
    }

    #[test]
    fn test_offset_for_negative() {
        let mut mcs = MultiClipSync::new("ref");
        mcs.add_sync(SyncPoint::new("ref", 200, 1.0));
        mcs.add_sync(SyncPoint::new("cam2", 150, 0.8));
        assert_eq!(mcs.offset_for("cam2"), Some(-50));
    }

    #[test]
    fn test_synchronized_clips_list() {
        let mut mcs = MultiClipSync::new("ref");
        mcs.add_sync(SyncPoint::new("ref", 0, 1.0));
        mcs.add_sync(SyncPoint::new("cam_a", 5, 0.9));
        mcs.add_sync(SyncPoint::new("cam_b", 3, 0.8));
        let clips = mcs.synchronized_clips();
        assert_eq!(clips.len(), 3);
        assert!(clips.contains(&"cam_a"));
        assert!(clips.contains(&"cam_b"));
    }

    // --- SyncReport ---

    #[test]
    fn test_sync_report_accurate_timecode() {
        let r = SyncReport::new(true, SyncMethod::Timecode, 0.0, 1.0);
        assert!(r.is_accurate());
    }

    #[test]
    fn test_sync_report_not_accurate_low_confidence() {
        let r = SyncReport::new(true, SyncMethod::Timecode, 0.0, 0.5);
        assert!(!r.is_accurate());
    }

    #[test]
    fn test_sync_report_not_accurate_large_offset() {
        let r = SyncReport::new(true, SyncMethod::AudioClap, 5.0, 0.9);
        // accuracy_frames = 1.0, threshold = 1.0 + 1.0 = 2.0; 5.0 > 2.0 → not accurate
        assert!(!r.is_accurate());
    }

    #[test]
    fn test_sync_report_not_accurate_failed() {
        let r = SyncReport::new(false, SyncMethod::Timecode, 0.0, 1.0);
        assert!(!r.is_accurate());
    }
}
