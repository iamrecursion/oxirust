//! Segment-based transcoding for parallel and distributed workloads.
//!
//! Provides time-segment decomposition, status tracking, and a multi-worker
//! transcoding queue for processing large media files in parallel chunks.

/// Specification for a single time-segment to transcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSpec {
    /// Segment start time in milliseconds.
    pub start_ms: u64,
    /// Segment end time in milliseconds.
    pub end_ms: u64,
    /// Name of the encoding profile to apply.
    pub profile_name: String,
}

impl SegmentSpec {
    /// Creates a new segment specification.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, profile_name: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            profile_name: profile_name.into(),
        }
    }

    /// Returns the segment duration in milliseconds.
    ///
    /// Returns 0 if `end_ms` is not greater than `start_ms`.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Status of a single transcode segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentStatus {
    /// Waiting to be processed.
    Pending,
    /// Currently encoding; value is progress percentage (0–100).
    Encoding(u8),
    /// Successfully completed.
    Done,
    /// Failed with an error message.
    Failed(String),
}

impl SegmentStatus {
    /// Returns `true` if the segment has finished (either done or failed).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Done | Self::Failed(_))
    }

    /// Returns the progress percentage.
    ///
    /// - `Pending` → 0
    /// - `Encoding(p)` → p
    /// - `Done` → 100
    /// - `Failed(_)` → 0
    #[must_use]
    pub fn progress_pct(&self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Encoding(p) => *p,
            Self::Done => 100,
            Self::Failed(_) => 0,
        }
    }
}

/// A single segment job: specification, file paths, and current status.
#[derive(Debug, Clone)]
pub struct TranscodeSegment {
    /// Specification describing the time range and profile.
    pub spec: SegmentSpec,
    /// Absolute path of the input file.
    pub input_path: String,
    /// Absolute path of the output file.
    pub output_path: String,
    /// Current processing status.
    pub status: SegmentStatus,
}

impl TranscodeSegment {
    /// Creates a new segment in `Pending` state.
    #[must_use]
    pub fn new(
        spec: SegmentSpec,
        input_path: impl Into<String>,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            spec,
            input_path: input_path.into(),
            output_path: output_path.into(),
            status: SegmentStatus::Pending,
        }
    }
}

/// Orchestrates multi-segment transcoding with configurable worker concurrency.
#[derive(Debug, Clone)]
pub struct SegmentTranscoder {
    /// All queued segments.
    pub segments: Vec<TranscodeSegment>,
    /// Number of parallel worker threads to use.
    pub workers: u32,
}

impl SegmentTranscoder {
    /// Creates a new transcoder with a specified worker count.
    ///
    /// # Panics
    ///
    /// Does not panic; `workers` must be at least 1 (caller responsibility).
    #[must_use]
    pub fn new(workers: u32) -> Self {
        Self {
            segments: Vec::new(),
            workers,
        }
    }

    /// Queues a new segment for transcoding.
    pub fn queue_segment(
        &mut self,
        spec: SegmentSpec,
        input: impl Into<String>,
        output: impl Into<String>,
    ) {
        self.segments
            .push(TranscodeSegment::new(spec, input, output));
    }

    /// Returns the number of segments still waiting to be processed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| s.status == SegmentStatus::Pending)
            .count()
    }

    /// Returns the number of successfully completed segments.
    #[must_use]
    pub fn complete_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| s.status == SegmentStatus::Done)
            .count()
    }

    /// Returns references to all segments that have failed.
    #[must_use]
    pub fn failed_segments(&self) -> Vec<&TranscodeSegment> {
        self.segments
            .iter()
            .filter(|s| matches!(s.status, SegmentStatus::Failed(_)))
            .collect()
    }

    /// Returns the sum of all segment durations in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.segments.iter().map(|s| s.spec.duration_ms()).sum()
    }

    /// Returns the total number of queued segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the number of segments currently encoding.
    #[must_use]
    pub fn encoding_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| matches!(s.status, SegmentStatus::Encoding(_)))
            .count()
    }

    /// Returns the overall progress across all segments as a percentage (0–100).
    #[must_use]
    pub fn overall_progress_pct(&self) -> u8 {
        if self.segments.is_empty() {
            return 0;
        }
        let total: u32 = self
            .segments
            .iter()
            .map(|s| u32::from(s.status.progress_pct()))
            .sum();
        #[allow(clippy::cast_possible_truncation)]
        let avg = (total / self.segments.len() as u32) as u8;
        avg
    }
}

impl Default for SegmentTranscoder {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SegmentSpec ──────────────────────────────────────────────────────────

    #[test]
    fn test_spec_duration_basic() {
        let spec = SegmentSpec::new(1000, 5000, "720p");
        assert_eq!(spec.duration_ms(), 4000);
    }

    #[test]
    fn test_spec_duration_zero_when_equal() {
        let spec = SegmentSpec::new(3000, 3000, "720p");
        assert_eq!(spec.duration_ms(), 0);
    }

    #[test]
    fn test_spec_duration_saturating_when_reversed() {
        // end < start saturates to 0
        let spec = SegmentSpec::new(5000, 3000, "720p");
        assert_eq!(spec.duration_ms(), 0);
    }

    #[test]
    fn test_spec_profile_name() {
        let spec = SegmentSpec::new(0, 10_000, "4k-hevc");
        assert_eq!(spec.profile_name, "4k-hevc");
    }

    // ── SegmentStatus ────────────────────────────────────────────────────────

    #[test]
    fn test_status_pending_not_complete() {
        assert!(!SegmentStatus::Pending.is_complete());
    }

    #[test]
    fn test_status_encoding_not_complete() {
        assert!(!SegmentStatus::Encoding(50).is_complete());
    }

    #[test]
    fn test_status_done_is_complete() {
        assert!(SegmentStatus::Done.is_complete());
    }

    #[test]
    fn test_status_failed_is_complete() {
        assert!(SegmentStatus::Failed("oom".to_string()).is_complete());
    }

    #[test]
    fn test_status_progress_pending() {
        assert_eq!(SegmentStatus::Pending.progress_pct(), 0);
    }

    #[test]
    fn test_status_progress_encoding() {
        assert_eq!(SegmentStatus::Encoding(73).progress_pct(), 73);
    }

    #[test]
    fn test_status_progress_done() {
        assert_eq!(SegmentStatus::Done.progress_pct(), 100);
    }

    #[test]
    fn test_status_progress_failed() {
        assert_eq!(SegmentStatus::Failed("err".to_string()).progress_pct(), 0);
    }

    // ── SegmentTranscoder ────────────────────────────────────────────────────

    #[test]
    fn test_transcoder_initial_counts() {
        let tc = SegmentTranscoder::new(2);
        assert_eq!(tc.segment_count(), 0);
        assert_eq!(tc.pending_count(), 0);
        assert_eq!(tc.complete_count(), 0);
        assert!(tc.failed_segments().is_empty());
        assert_eq!(tc.total_duration_ms(), 0);
    }

    #[test]
    fn test_queue_segment_increments_count() {
        let mut tc = SegmentTranscoder::new(2);
        let spec = SegmentSpec::new(0, 30_000, "1080p");
        tc.queue_segment(spec, "/in/a.mp4", "/out/a.mp4");
        assert_eq!(tc.segment_count(), 1);
        assert_eq!(tc.pending_count(), 1);
    }

    #[test]
    fn test_complete_count_after_marking_done() {
        let mut tc = SegmentTranscoder::new(1);
        let spec = SegmentSpec::new(0, 10_000, "720p");
        tc.queue_segment(spec, "/in/b.mp4", "/out/b.mp4");
        tc.segments[0].status = SegmentStatus::Done;
        assert_eq!(tc.complete_count(), 1);
        assert_eq!(tc.pending_count(), 0);
    }

    #[test]
    fn test_failed_segments_returns_correct_refs() {
        let mut tc = SegmentTranscoder::new(2);
        let s1 = SegmentSpec::new(0, 5000, "360p");
        let s2 = SegmentSpec::new(5000, 10_000, "360p");
        tc.queue_segment(s1, "/in/c.mp4", "/out/c1.mp4");
        tc.queue_segment(s2, "/in/c.mp4", "/out/c2.mp4");
        tc.segments[0].status = SegmentStatus::Failed("codec error".to_string());
        let failed = tc.failed_segments();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].output_path, "/out/c1.mp4");
    }

    #[test]
    fn test_total_duration_ms_sums_all_segments() {
        let mut tc = SegmentTranscoder::new(4);
        tc.queue_segment(SegmentSpec::new(0, 10_000, "p"), "/i", "/o1");
        tc.queue_segment(SegmentSpec::new(10_000, 25_000, "p"), "/i", "/o2");
        tc.queue_segment(SegmentSpec::new(25_000, 30_000, "p"), "/i", "/o3");
        assert_eq!(tc.total_duration_ms(), 30_000);
    }

    #[test]
    fn test_overall_progress_empty() {
        let tc = SegmentTranscoder::new(2);
        assert_eq!(tc.overall_progress_pct(), 0);
    }

    #[test]
    fn test_workers_stored() {
        let tc = SegmentTranscoder::new(8);
        assert_eq!(tc.workers, 8);
    }

    #[test]
    fn test_encoding_count() {
        let mut tc = SegmentTranscoder::new(2);
        tc.queue_segment(SegmentSpec::new(0, 5000, "p"), "/i", "/o1");
        tc.queue_segment(SegmentSpec::new(5000, 10_000, "p"), "/i", "/o2");
        tc.segments[0].status = SegmentStatus::Encoding(42);
        assert_eq!(tc.encoding_count(), 1);
        assert_eq!(tc.pending_count(), 1);
    }
}
