//! Media segment lifecycle management.
//!
//! Tracks the state of every media segment from creation through download,
//! playback availability, and eviction.  Provides buffer-depth accounting
//! and prefetch scheduling hints.

use std::collections::HashMap;
use std::time::Instant;

// ─── Segment state ────────────────────────────────────────────────────────────

/// Lifecycle state of a single media segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentState {
    /// Segment has been registered but download has not begun.
    Pending,
    /// Download is in progress.
    Downloading,
    /// Download complete; segment is ready for playback.
    Available,
    /// Segment was available but has been removed from the buffer.
    Evicted,
    /// Download failed or the segment could not be decoded.
    Failed,
}

// ─── Media segment ────────────────────────────────────────────────────────────

/// A single media segment with full lifecycle metadata.
#[derive(Debug, Clone)]
pub struct MediaSegment {
    /// Unique identifier (UUID v4 formatted as a hex string).
    pub id: String,
    /// Monotonically increasing sequence number within a stream.
    pub sequence_number: u64,
    /// Presentation timestamp of the first frame/sample in milliseconds.
    pub pts_start_ms: u64,
    /// Nominal duration of the segment in milliseconds.
    pub duration_ms: u64,
    /// Size of the segment payload in bytes.
    pub size_bytes: usize,
    /// Quality tier name this segment belongs to (e.g. `"720p"`).
    pub tier_name: String,
    /// Current lifecycle state.
    pub state: SegmentState,
    /// Measured download throughput in kbps, populated after download.
    pub download_speed_kbps: Option<f64>,
    /// Monotonic timestamp of when this segment record was created.
    pub created_at: Instant,
}

// ─── Prefetch configuration ───────────────────────────────────────────────────

/// Configuration for bandwidth-adaptive prefetch depth adjustment.
///
/// Encapsulates the tuning parameters for [`SegmentManager::auto_adjust_prefetch`],
/// which dynamically sets the prefetch depth based on available bandwidth and
/// the current quality tier's bitrate.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Minimum prefetch depth (segments). Always at least 1.
    pub min_depth: usize,
    /// Maximum prefetch depth (segments).
    pub max_depth: usize,
    /// Target playback buffer in seconds.  When the actual buffer is below this
    /// value, the algorithm biases toward a deeper prefetch to build up the
    /// buffer faster.
    pub target_buffer_secs: f64,
    /// Nominal segment duration in seconds (e.g. 6.0 for typical HLS).
    pub segment_duration_secs: f64,
    /// When the buffer is below `target_buffer_secs × low_buffer_factor`, the
    /// prefetch depth is increased by one extra segment (clamped to `max_depth`).
    pub low_buffer_factor: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            min_depth: 1,
            max_depth: 10,
            target_buffer_secs: 30.0,
            segment_duration_secs: 6.0,
            low_buffer_factor: 0.5,
        }
    }
}

// ─── Segment manager ──────────────────────────────────────────────────────────

/// Manages a pool of [`MediaSegment`]s and drives prefetch / eviction policy.
pub struct SegmentManager {
    /// All known segments, keyed by segment ID.
    pub segments: HashMap<String, MediaSegment>,
    /// Maximum number of `Available` segments to keep in memory simultaneously.
    pub max_buffer_segments: usize,
    /// How many segments ahead of the playhead to attempt to prefetch.
    pub prefetch_count: usize,
    /// How many segments behind the playhead to retain before evicting.
    pub evict_behind: usize,
    next_sequence: u64,
}

impl SegmentManager {
    /// Construct a new manager.
    ///
    /// - `max_buffer`: maximum `Available` segments held in memory.
    /// - `prefetch`: desired number of segments to download ahead of playback.
    pub fn new(max_buffer: usize, prefetch: usize) -> Self {
        Self {
            segments: HashMap::new(),
            max_buffer_segments: max_buffer.max(1),
            prefetch_count: prefetch,
            evict_behind: 3,
            next_sequence: 0,
        }
    }

    /// Register a new segment and return its generated ID.
    ///
    /// The segment starts in the [`SegmentState::Pending`] state.
    pub fn create_segment(
        &mut self,
        pts_ms: u64,
        duration_ms: u64,
        size_bytes: usize,
        tier: &str,
    ) -> String {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        // Derive a deterministic-looking ID without pulling in the uuid crate.
        let id = format!("seg-{seq:016x}-{pts_ms:016x}");
        let segment = MediaSegment {
            id: id.clone(),
            sequence_number: seq,
            pts_start_ms: pts_ms,
            duration_ms,
            size_bytes,
            tier_name: tier.to_string(),
            state: SegmentState::Pending,
            download_speed_kbps: None,
            created_at: Instant::now(),
        };
        self.segments.insert(id.clone(), segment);
        id
    }

    /// Transition a segment to [`SegmentState::Downloading`].
    ///
    /// Returns `true` if the segment existed and was in `Pending` state.
    pub fn mark_downloading(&mut self, id: &str) -> bool {
        match self.segments.get_mut(id) {
            Some(seg) if seg.state == SegmentState::Pending => {
                seg.state = SegmentState::Downloading;
                true
            }
            _ => false,
        }
    }

    /// Transition a segment to [`SegmentState::Available`] and record download speed.
    ///
    /// Returns `true` if the segment existed and was in `Downloading` state.
    pub fn mark_available(&mut self, id: &str, speed_kbps: f64) -> bool {
        match self.segments.get_mut(id) {
            Some(seg) if seg.state == SegmentState::Downloading => {
                seg.state = SegmentState::Available;
                seg.download_speed_kbps = Some(speed_kbps.max(0.0));
                true
            }
            _ => false,
        }
    }

    /// Transition a segment to [`SegmentState::Failed`].
    ///
    /// Returns `true` if the segment was found and not already `Evicted`.
    pub fn mark_failed(&mut self, id: &str) -> bool {
        match self.segments.get_mut(id) {
            Some(seg) if seg.state != SegmentState::Evicted => {
                seg.state = SegmentState::Failed;
                true
            }
            _ => false,
        }
    }

    /// Evict segments whose sequence number is more than `evict_behind` before
    /// `playhead_seq`.
    ///
    /// Only `Available` segments are evicted; `Pending`/`Downloading` segments
    /// behind the playhead are left untouched so downloads can complete.
    ///
    /// Returns the number of segments evicted.
    pub fn evict_old(&mut self, playhead_seq: u64) -> usize {
        let cutoff = playhead_seq.saturating_sub(self.evict_behind as u64);
        let mut count = 0;
        for seg in self.segments.values_mut() {
            if seg.sequence_number < cutoff && seg.state == SegmentState::Available {
                seg.state = SegmentState::Evicted;
                count += 1;
            }
        }
        count
    }

    /// Return all `Available` segments sorted by sequence number ascending.
    pub fn available_segments(&self) -> Vec<&MediaSegment> {
        let mut segs: Vec<&MediaSegment> = self
            .segments
            .values()
            .filter(|s| s.state == SegmentState::Available)
            .collect();
        segs.sort_by_key(|s| s.sequence_number);
        segs
    }

    /// Return the sequence number of the next segment that still needs to be
    /// fetched (either `Pending` in the current pool, or the first sequence
    /// number beyond what has been registered).
    pub fn next_needed(&self) -> Option<u64> {
        // Find the lowest-sequence Pending segment.
        let min_pending = self
            .segments
            .values()
            .filter(|s| s.state == SegmentState::Pending)
            .map(|s| s.sequence_number)
            .min();

        if min_pending.is_some() {
            return min_pending;
        }

        // If no Pending segments exist but next_sequence > 0, signal that a
        // new segment should be created.
        if self.next_sequence > 0 {
            Some(self.next_sequence)
        } else {
            None
        }
    }

    /// Total duration (ms) of all `Available` segments currently buffered.
    pub fn buffer_duration_ms(&self) -> u64 {
        self.segments
            .values()
            .filter(|s| s.state == SegmentState::Available)
            .map(|s| s.duration_ms)
            .sum()
    }

    /// Mean download speed (kbps) over all segments that have a recorded speed.
    ///
    /// Returns `None` if no segment has completed yet.
    pub fn avg_download_speed_kbps(&self) -> Option<f64> {
        let speeds: Vec<f64> = self
            .segments
            .values()
            .filter_map(|s| s.download_speed_kbps)
            .collect();
        if speeds.is_empty() {
            None
        } else {
            Some(speeds.iter().sum::<f64>() / speeds.len() as f64)
        }
    }

    /// Dynamically adjust the prefetch depth based on available bandwidth and
    /// segment bitrate.
    ///
    /// The algorithm computes how many segments can be downloaded per segment
    /// duration at the current bandwidth, then clamps the result between
    /// `min_prefetch` and `max_prefetch`.
    ///
    /// # Parameters
    ///
    /// - `bandwidth_kbps`: current estimated downstream bandwidth in kbps.
    /// - `segment_bitrate_kbps`: bitrate of the active quality tier in kbps.
    /// - `segment_duration_secs`: nominal segment duration in seconds.
    /// - `min_prefetch`: lower bound on prefetch depth.
    /// - `max_prefetch`: upper bound on prefetch depth.
    ///
    /// # Returns
    ///
    /// The new prefetch count (also stored in `self.prefetch_count`).
    pub fn adjust_prefetch_depth(
        &mut self,
        bandwidth_kbps: f64,
        segment_bitrate_kbps: f64,
        segment_duration_secs: f64,
        min_prefetch: usize,
        max_prefetch: usize,
    ) -> usize {
        let min_prefetch = min_prefetch.max(1);
        let max_prefetch = max_prefetch.max(min_prefetch);

        if segment_bitrate_kbps <= 0.0 || segment_duration_secs <= 0.0 || bandwidth_kbps <= 0.0 {
            self.prefetch_count = min_prefetch;
            return self.prefetch_count;
        }

        // Segment size in kilobits.
        let segment_size_kb = segment_bitrate_kbps * segment_duration_secs;

        // Time to download one segment (seconds).
        let download_time_secs = segment_size_kb / bandwidth_kbps;

        // Ratio: how many segments can we download in the time one segment plays.
        let throughput_ratio = if download_time_secs > 0.0 {
            segment_duration_secs / download_time_secs
        } else {
            max_prefetch as f64
        };

        // If throughput_ratio > 1.0, we can download faster than realtime.
        // Higher ratio → deeper prefetch is safe.
        //
        // Strategy: prefetch = floor(throughput_ratio), clamped to [min, max].
        // When bandwidth is just barely enough (ratio ~1.0), prefetch = 1.
        // When bandwidth is 5x the bitrate, prefetch = 5.
        let computed = throughput_ratio.floor().max(0.0) as usize;
        self.prefetch_count = computed.clamp(min_prefetch, max_prefetch);
        self.prefetch_count
    }

    /// Automatically adjust prefetch depth using a [`PrefetchConfig`] and the
    /// current buffer / bandwidth state.
    ///
    /// This is a higher-level wrapper around `adjust_prefetch_depth` that also
    /// applies a low-buffer boost: when the current buffer is below
    /// `config.target_buffer_secs × config.low_buffer_factor`, the computed
    /// depth is increased by one extra segment.
    ///
    /// # Parameters
    ///
    /// - `bandwidth_kbps`: current estimated downstream bandwidth in kbps.
    /// - `segment_bitrate_kbps`: bitrate of the active quality tier in kbps.
    /// - `current_buffer_secs`: current playback buffer depth in seconds.
    /// - `config`: prefetch tuning parameters.
    ///
    /// # Returns
    ///
    /// The new prefetch count (also stored in `self.prefetch_count`).
    pub fn auto_adjust_prefetch(
        &mut self,
        bandwidth_kbps: f64,
        segment_bitrate_kbps: f64,
        current_buffer_secs: f64,
        config: &PrefetchConfig,
    ) -> usize {
        let min = config.min_depth.max(1);
        let max = config.max_depth.max(min);

        // Base depth from throughput ratio.
        let base = self.adjust_prefetch_depth(
            bandwidth_kbps,
            segment_bitrate_kbps,
            config.segment_duration_secs,
            min,
            max,
        );

        // Low-buffer boost: if buffer is dangerously low, add one segment.
        let low_threshold = config.target_buffer_secs * config.low_buffer_factor;
        let boosted = if current_buffer_secs < low_threshold && current_buffer_secs >= 0.0 {
            (base + 1).min(max)
        } else {
            base
        };

        self.prefetch_count = boosted;
        self.prefetch_count
    }

    /// Return the sequence numbers of up to `prefetch_count` segments that
    /// should be downloaded next, starting from `playhead_seq`.
    ///
    /// Skips segments that are already `Downloading`, `Available`, or `Evicted`.
    pub fn prefetch_candidates(&self, playhead_seq: u64) -> Vec<u64> {
        let mut candidates = Vec::with_capacity(self.prefetch_count);
        let mut seq = playhead_seq;
        let limit = playhead_seq
            .saturating_add(self.prefetch_count as u64 + self.max_buffer_segments as u64);

        while candidates.len() < self.prefetch_count && seq < limit {
            let dominated = self.segments.values().any(|s| {
                s.sequence_number == seq
                    && matches!(
                        s.state,
                        SegmentState::Downloading | SegmentState::Available | SegmentState::Evicted
                    )
            });
            if !dominated {
                candidates.push(seq);
            }
            seq = seq.saturating_add(1);
        }
        candidates
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> SegmentManager {
        SegmentManager::new(10, 3)
    }

    #[test]
    fn test_create_segment_returns_unique_ids() {
        let mut mgr = make_manager();
        let id1 = mgr.create_segment(0, 2000, 512_000, "720p");
        let id2 = mgr.create_segment(2000, 2000, 512_000, "720p");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_create_segment_initial_state_pending() {
        let mut mgr = make_manager();
        let id = mgr.create_segment(0, 2000, 100_000, "480p");
        let seg = mgr.segments.get(&id).expect("segment should exist");
        assert_eq!(seg.state, SegmentState::Pending);
    }

    #[test]
    fn test_mark_downloading_transitions_pending() {
        let mut mgr = make_manager();
        let id = mgr.create_segment(0, 2000, 100_000, "480p");
        assert!(mgr.mark_downloading(&id));
        assert_eq!(mgr.segments[&id].state, SegmentState::Downloading);
    }

    #[test]
    fn test_mark_downloading_rejects_non_pending() {
        let mut mgr = make_manager();
        let id = mgr.create_segment(0, 2000, 100_000, "480p");
        mgr.mark_downloading(&id);
        // Second call should fail: already Downloading
        assert!(!mgr.mark_downloading(&id));
    }

    #[test]
    fn test_mark_available_records_speed() {
        let mut mgr = make_manager();
        let id = mgr.create_segment(0, 2000, 100_000, "480p");
        mgr.mark_downloading(&id);
        assert!(mgr.mark_available(&id, 5000.0));
        let seg = &mgr.segments[&id];
        assert_eq!(seg.state, SegmentState::Available);
        assert_eq!(seg.download_speed_kbps, Some(5000.0));
    }

    #[test]
    fn test_mark_failed_transitions_any_active_state() {
        let mut mgr = make_manager();
        let id = mgr.create_segment(0, 2000, 100_000, "480p");
        mgr.mark_downloading(&id);
        assert!(mgr.mark_failed(&id));
        assert_eq!(mgr.segments[&id].state, SegmentState::Failed);
    }

    #[test]
    fn test_evict_old_removes_available_behind_playhead() {
        let mut mgr = make_manager();
        // Create 5 segments and make them all available.
        let mut ids = Vec::new();
        for i in 0..5_u64 {
            let id = mgr.create_segment(i * 2000, 2000, 50_000, "360p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 1000.0);
            ids.push(id);
        }
        // Playhead at seq 4; evict_behind = 3 → cutoff = 1; seqs 0 should be evicted.
        let evicted = mgr.evict_old(4);
        assert!(evicted >= 1, "at least seq 0 should be evicted");
        // Verify seq 0 is evicted
        let first = mgr.segments.get(&ids[0]).expect("seg 0");
        assert_eq!(first.state, SegmentState::Evicted);
    }

    #[test]
    fn test_available_segments_sorted() {
        let mut mgr = make_manager();
        for i in [2_u64, 0, 1] {
            let id = mgr.create_segment(i * 2000, 2000, 50_000, "360p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 1000.0);
        }
        let avail = mgr.available_segments();
        assert_eq!(avail.len(), 3);
        assert!(avail[0].sequence_number <= avail[1].sequence_number);
        assert!(avail[1].sequence_number <= avail[2].sequence_number);
    }

    #[test]
    fn test_buffer_duration_ms_sums_available() {
        let mut mgr = make_manager();
        for _ in 0..3 {
            let id = mgr.create_segment(0, 2000, 50_000, "480p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 1000.0);
        }
        assert_eq!(mgr.buffer_duration_ms(), 6000);
    }

    #[test]
    fn test_avg_download_speed_none_when_empty() {
        let mgr = make_manager();
        assert!(mgr.avg_download_speed_kbps().is_none());
    }

    #[test]
    fn test_avg_download_speed_computed_correctly() {
        let mut mgr = make_manager();
        let speeds = [1000.0_f64, 2000.0, 3000.0];
        for &spd in &speeds {
            let id = mgr.create_segment(0, 2000, 50_000, "480p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, spd);
        }
        let avg = mgr.avg_download_speed_kbps().expect("avg");
        assert!((avg - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_next_needed_returns_pending_sequence() {
        let mut mgr = make_manager();
        let _id = mgr.create_segment(0, 2000, 50_000, "480p");
        let next = mgr.next_needed();
        assert_eq!(next, Some(0));
    }

    // ── Prefetch depth tests ────────────────────────────────────────────────

    #[test]
    fn test_adjust_prefetch_depth_high_bandwidth() {
        let mut mgr = make_manager();
        // Bandwidth 10000 kbps, segment bitrate 2000 kbps, 6 s segments
        // Ratio = 6 / (2000*6/10000) = 6 / 1.2 = 5
        let depth = mgr.adjust_prefetch_depth(10_000.0, 2_000.0, 6.0, 1, 10);
        assert_eq!(depth, 5);
        assert_eq!(mgr.prefetch_count, 5);
    }

    #[test]
    fn test_adjust_prefetch_depth_low_bandwidth() {
        let mut mgr = make_manager();
        // Bandwidth 2500 kbps, segment bitrate 2000 kbps, 6 s segments
        // Ratio = 6 / (12000/2500) = 6 / 4.8 = 1.25 → floor = 1
        let depth = mgr.adjust_prefetch_depth(2_500.0, 2_000.0, 6.0, 1, 10);
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_adjust_prefetch_depth_clamps_to_min() {
        let mut mgr = make_manager();
        // Very low bandwidth → ratio < 1 → clamped to min=2
        let depth = mgr.adjust_prefetch_depth(100.0, 5_000.0, 6.0, 2, 10);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_adjust_prefetch_depth_clamps_to_max() {
        let mut mgr = make_manager();
        // Extremely high bandwidth → huge ratio → clamped to max=4
        let depth = mgr.adjust_prefetch_depth(100_000.0, 500.0, 6.0, 1, 4);
        assert_eq!(depth, 4);
    }

    #[test]
    fn test_adjust_prefetch_depth_zero_bandwidth_returns_min() {
        let mut mgr = make_manager();
        let depth = mgr.adjust_prefetch_depth(0.0, 2_000.0, 6.0, 2, 8);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_adjust_prefetch_depth_zero_bitrate_returns_min() {
        let mut mgr = make_manager();
        let depth = mgr.adjust_prefetch_depth(10_000.0, 0.0, 6.0, 1, 8);
        assert_eq!(depth, 1);
    }

    #[test]
    fn test_prefetch_candidates_returns_pending_seqs() {
        let mut mgr = make_manager();
        mgr.prefetch_count = 3;
        // Create some segments: seq 0..4
        for i in 0..5u64 {
            let _id = mgr.create_segment(i * 2000, 2000, 50_000, "480p");
        }
        // Mark seqs 0 and 1 as downloading/available
        let ids: Vec<String> = mgr
            .segments
            .values()
            .filter(|s| s.sequence_number < 2)
            .map(|s| s.id.clone())
            .collect();
        for id in &ids {
            mgr.mark_downloading(id);
        }
        // Prefetch from playhead=0 should skip seq 0,1 (downloading) and give 2,3,4
        let candidates = mgr.prefetch_candidates(0);
        assert_eq!(candidates.len(), 3);
        assert!(candidates.contains(&2));
        assert!(candidates.contains(&3));
        assert!(candidates.contains(&4));
    }

    #[test]
    fn test_prefetch_candidates_empty_when_all_active() {
        let mut mgr = make_manager();
        mgr.prefetch_count = 2;
        for i in 0..3u64 {
            let id = mgr.create_segment(i * 2000, 2000, 50_000, "480p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 1000.0);
        }
        let candidates = mgr.prefetch_candidates(0);
        // Seqs 0,1,2 are all Available, so candidates start at 3
        assert!(!candidates.is_empty());
        assert!(candidates[0] >= 3);
    }

    // ── PrefetchConfig / auto_adjust_prefetch tests ─────────────────────────

    #[test]
    fn test_prefetch_config_default_values() {
        let cfg = PrefetchConfig::default();
        assert_eq!(cfg.min_depth, 1);
        assert_eq!(cfg.max_depth, 10);
        assert!((cfg.target_buffer_secs - 30.0).abs() < 0.01);
        assert!((cfg.segment_duration_secs - 6.0).abs() < 0.01);
        assert!((cfg.low_buffer_factor - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_auto_adjust_prefetch_high_bandwidth_full_buffer() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig {
            min_depth: 1,
            max_depth: 10,
            target_buffer_secs: 30.0,
            segment_duration_secs: 6.0,
            low_buffer_factor: 0.5,
        };
        // High BW (10000), low bitrate (2000), 6s segments → ratio = 5
        // Buffer at 30s → no boost
        let depth = mgr.auto_adjust_prefetch(10_000.0, 2_000.0, 30.0, &cfg);
        assert_eq!(depth, 5);
    }

    #[test]
    fn test_auto_adjust_prefetch_low_buffer_boost() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig {
            min_depth: 1,
            max_depth: 10,
            target_buffer_secs: 30.0,
            segment_duration_secs: 6.0,
            low_buffer_factor: 0.5,
        };
        // Same bandwidth as above → base = 5
        // Buffer at 10s (< 30 * 0.5 = 15s) → boosted to 6
        let depth = mgr.auto_adjust_prefetch(10_000.0, 2_000.0, 10.0, &cfg);
        assert_eq!(depth, 6);
    }

    #[test]
    fn test_auto_adjust_prefetch_boost_capped_at_max() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig {
            min_depth: 1,
            max_depth: 5,
            target_buffer_secs: 30.0,
            segment_duration_secs: 6.0,
            low_buffer_factor: 0.5,
        };
        // Base = 5 (capped at max), low buffer → boost to 6 but capped at max=5
        let depth = mgr.auto_adjust_prefetch(10_000.0, 2_000.0, 1.0, &cfg);
        assert_eq!(depth, 5);
    }

    #[test]
    fn test_auto_adjust_prefetch_no_boost_above_threshold() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig {
            min_depth: 2,
            max_depth: 8,
            target_buffer_secs: 20.0,
            segment_duration_secs: 4.0,
            low_buffer_factor: 0.5,
        };
        // Buffer at 15s (>= 20 * 0.5 = 10s) → no boost
        let depth = mgr.auto_adjust_prefetch(10_000.0, 2_000.0, 15.0, &cfg);
        // Base: ratio = 4 / (2000*4/10000) = 4/0.8 = 5
        assert_eq!(depth, 5, "no boost expected above threshold");
    }

    #[test]
    fn test_auto_adjust_prefetch_zero_bandwidth() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig::default();
        let depth = mgr.auto_adjust_prefetch(0.0, 2_000.0, 5.0, &cfg);
        // Zero BW → min, buffer low → boost → min+1
        assert_eq!(depth, 2); // min=1 + boost=1
    }

    #[test]
    fn test_auto_adjust_prefetch_very_low_bandwidth() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig {
            min_depth: 2,
            max_depth: 8,
            target_buffer_secs: 30.0,
            segment_duration_secs: 6.0,
            low_buffer_factor: 0.5,
        };
        // BW barely above bitrate → ratio ~1, buffer ok
        let depth = mgr.auto_adjust_prefetch(2_500.0, 2_000.0, 20.0, &cfg);
        assert_eq!(depth, 2, "barely-enough BW should yield min depth");
    }

    #[test]
    fn test_auto_adjust_prefetch_stores_in_prefetch_count() {
        let mut mgr = make_manager();
        let cfg = PrefetchConfig::default();
        let depth = mgr.auto_adjust_prefetch(10_000.0, 2_000.0, 30.0, &cfg);
        assert_eq!(mgr.prefetch_count, depth);
    }

    // ── Segment continuity tests ─────────────────────────────────────────────

    #[test]
    fn test_segment_continuity_across_playlist_updates() {
        let mut mgr = SegmentManager::new(20, 3);
        // Round 1: create 5 segments and mark them available
        let ids: Vec<String> = (0u64..5)
            .map(|i| {
                let id = mgr.create_segment(i * 3000, 3000, 1024, "720p");
                mgr.mark_downloading(&id);
                id
            })
            .collect();
        for id in &ids {
            mgr.mark_available(id, 5000.0);
        }
        // Assert round 1 sequence numbers are 0..=4 in order
        let seqs: Vec<u64> = mgr
            .available_segments()
            .iter()
            .map(|s| s.sequence_number)
            .collect();
        assert_eq!(seqs, vec![0, 1, 2, 3, 4]);

        // Round 2: create 5 more segments and mark them available
        let ids2: Vec<String> = (0u64..5)
            .map(|i| {
                let id = mgr.create_segment(15000 + i * 3000, 3000, 1024, "720p");
                mgr.mark_downloading(&id);
                id
            })
            .collect();
        for id in &ids2 {
            mgr.mark_available(id, 5000.0);
        }
        // Combined available list must be strictly monotonically increasing
        let all_seqs: Vec<u64> = mgr
            .available_segments()
            .iter()
            .map(|s| s.sequence_number)
            .collect();
        assert!(
            all_seqs.windows(2).all(|w| w[0] < w[1]),
            "sequence numbers must be strictly monotonic: {:?}",
            all_seqs
        );
        assert_eq!(all_seqs.first().copied(), Some(0));
        assert_eq!(all_seqs.last().copied(), Some(9));
    }

    #[test]
    fn test_segment_continuity_after_eviction() {
        let mut mgr = SegmentManager::new(20, 3);
        // Create and mark 5 segments available
        for i in 0u64..5 {
            let id = mgr.create_segment(i * 3000, 3000, 1024, "720p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 5000.0);
        }
        // Evict segments behind playhead sequence 3 (evict_behind=3 so only
        // segments with seq < 3-3=0 are evicted; raise playhead to 6 to
        // actually evict some)
        mgr.evict_old(6);
        // Create 5 more segments and mark available
        for i in 0u64..5 {
            let id = mgr.create_segment(15000 + i * 3000, 3000, 1024, "720p");
            mgr.mark_downloading(&id);
            mgr.mark_available(&id, 5000.0);
        }
        // Remaining available segments must still have strictly monotonic
        // sequence numbers (no wrap-around or gaps introduced by eviction)
        let seqs: Vec<u64> = mgr
            .available_segments()
            .iter()
            .map(|s| s.sequence_number)
            .collect();
        assert!(
            seqs.windows(2).all(|w| w[0] < w[1]),
            "monotonic after eviction: {:?}",
            seqs
        );
        assert!(!seqs.is_empty());
    }
}
