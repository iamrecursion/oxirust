//! Segment-based encoding for HLS and DASH streaming.
//!
//! This module provides tools for planning segment boundaries,
//! tracking encoded segments, and generating HLS/DASH manifests.

/// Configuration for segment-based encoding.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Target segment duration in seconds.
    pub duration_secs: f32,
    /// Keyframe interval in frames.
    pub keyframe_interval: u32,
    /// Force a keyframe at each segment boundary.
    pub force_key_at_segment: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            duration_secs: 6.0,
            keyframe_interval: 60,
            force_key_at_segment: true,
        }
    }
}

impl SegmentConfig {
    /// Creates a new segment configuration.
    #[must_use]
    pub fn new(duration_secs: f32, keyframe_interval: u32, force_key_at_segment: bool) -> Self {
        Self {
            duration_secs,
            keyframe_interval,
            force_key_at_segment,
        }
    }
}

/// A boundary point between two segments.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentBoundary {
    /// Frame index where the segment starts.
    pub frame_idx: u64,
    /// Whether this frame is a keyframe.
    pub is_keyframe: bool,
    /// Timestamp in seconds.
    pub timestamp_secs: f64,
}

/// A complete segment plan for encoding.
#[derive(Debug, Clone)]
pub struct SegmentPlan {
    /// All segment boundaries.
    pub boundaries: Vec<SegmentBoundary>,
    /// Total number of frames.
    pub total_frames: u64,
    /// Total number of segments.
    pub segment_count: u32,
}

/// Plans segment boundaries for a given video.
#[derive(Debug, Clone, Default)]
pub struct SegmentPlanner;

impl SegmentPlanner {
    /// Creates a new segment planner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Plans segment boundaries for `total_frames` frames at `fps` with the given config.
    #[must_use]
    pub fn plan(total_frames: u64, fps: f32, config: &SegmentConfig) -> SegmentPlan {
        let frames_per_segment = (config.duration_secs * fps).round() as u64;
        let frames_per_segment = frames_per_segment.max(1);

        let mut boundaries = Vec::new();
        let mut frame_idx = 0u64;
        let mut seg_count = 0u32;

        while frame_idx < total_frames {
            let is_keyframe = if config.force_key_at_segment {
                true
            } else {
                // Keyframe at regular intervals
                frame_idx % u64::from(config.keyframe_interval) == 0
            };

            let timestamp_secs = frame_idx as f64 / f64::from(fps);

            boundaries.push(SegmentBoundary {
                frame_idx,
                is_keyframe,
                timestamp_secs,
            });

            frame_idx += frames_per_segment;
            seg_count += 1;
        }

        SegmentPlan {
            boundaries,
            total_frames,
            segment_count: seg_count,
        }
    }
}

/// An encoded segment of media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedSegment {
    /// Segment index (0-based).
    pub index: u32,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Actual bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Codec used for this segment.
    pub codec: String,
}

impl EncodedSegment {
    /// Creates a new encoded segment.
    #[must_use]
    pub fn new(
        index: u32,
        start_ms: u64,
        duration_ms: u64,
        size_bytes: u64,
        bitrate_kbps: u32,
        codec: impl Into<String>,
    ) -> Self {
        Self {
            index,
            start_ms,
            duration_ms,
            size_bytes,
            bitrate_kbps,
            codec: codec.into(),
        }
    }

    /// Returns the end time in milliseconds.
    #[must_use]
    pub fn end_ms(&self) -> u64 {
        self.start_ms + self.duration_ms
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }
}

/// Encoder that tracks encoded segments.
#[derive(Debug, Clone, Default)]
pub struct SegmentEncoder {
    /// All encoded segments.
    pub encoded_segments: Vec<EncodedSegment>,
}

impl SegmentEncoder {
    /// Creates a new segment encoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a segment to the encoder's list.
    pub fn add_segment(&mut self, segment: EncodedSegment) {
        self.encoded_segments.push(segment);
    }

    /// Returns the total number of encoded segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.encoded_segments.len()
    }

    /// Returns the total encoded size in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.encoded_segments.iter().map(|s| s.size_bytes).sum()
    }

    /// Returns the average bitrate across all segments.
    #[must_use]
    pub fn average_bitrate_kbps(&self) -> Option<u32> {
        if self.encoded_segments.is_empty() {
            return None;
        }
        let sum: u64 = self
            .encoded_segments
            .iter()
            .map(|s| u64::from(s.bitrate_kbps))
            .sum();
        Some((sum / self.encoded_segments.len() as u64) as u32)
    }
}

/// Generates HLS and DASH manifests from encoded segments.
#[derive(Debug, Clone, Default)]
pub struct SegmentManifest;

impl SegmentManifest {
    /// Generates an HLS `.m3u8` manifest.
    #[must_use]
    pub fn generate_hls(segments: &[EncodedSegment], base_url: &str) -> String {
        let max_duration = segments
            .iter()
            .map(EncodedSegment::duration_secs)
            .fold(0.0_f64, f64::max);

        let mut manifest = format!(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:{}\n#EXT-X-MEDIA-SEQUENCE:0\n",
            max_duration.ceil() as u64
        );

        for seg in segments {
            let duration = seg.duration_secs();
            manifest.push_str(&format!(
                "#EXTINF:{:.3},\n{}/segment_{:05}.ts\n",
                duration, base_url, seg.index
            ));
        }

        manifest.push_str("#EXT-X-ENDLIST\n");
        manifest
    }

    /// Generates a MPEG-DASH `manifest.mpd` manifest.
    #[must_use]
    pub fn generate_dash(segments: &[EncodedSegment], base_url: &str) -> String {
        let total_ms: u64 = segments.iter().map(|s| s.duration_ms).sum();
        let total_secs = total_ms as f64 / 1000.0;

        let avg_bitrate = if segments.is_empty() {
            0u32
        } else {
            let sum: u64 = segments.iter().map(|s| u64::from(s.bitrate_kbps)).sum();
            (sum / segments.len() as u64) as u32
        };

        let codec = segments.first().map_or("avc1", |s| s.codec.as_str());

        let mut mpd = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" \
             mediaPresentationDuration=\"PT{total_secs:.3}S\" \
             type=\"static\">\n  \
             <Period>\n    \
             <AdaptationSet mimeType=\"video/mp4\">\n      \
             <Representation id=\"0\" codecs=\"{codec}\" bandwidth=\"{}\">\n",
            avg_bitrate * 1000
        );

        mpd.push_str("        <SegmentList>\n");
        for seg in segments {
            mpd.push_str(&format!(
                "          <SegmentURL media=\"{}/segment_{:05}.mp4\"/>\n",
                base_url, seg.index
            ));
        }
        mpd.push_str(
            "        </SegmentList>\n      \
             </Representation>\n    \
             </AdaptationSet>\n  \
             </Period>\n\
             </MPD>\n",
        );

        mpd
    }
}

// ─── Parallel segment encoding ────────────────────────────────────────────────

/// The result of encoding a single independent segment in parallel.
#[derive(Debug, Clone)]
pub struct ParallelSegmentResult {
    /// Zero-based segment index.
    pub index: u32,
    /// Whether the encode succeeded.
    pub success: bool,
    /// Error message if `!success`.
    pub error: Option<String>,
    /// Produced encoded segment (on success).
    pub segment: Option<EncodedSegment>,
    /// Wall-clock time for this segment in seconds.
    pub wall_time_secs: f64,
}

impl ParallelSegmentResult {
    /// Creates a successful result.
    #[must_use]
    pub fn ok(index: u32, segment: EncodedSegment, wall_time_secs: f64) -> Self {
        Self {
            index,
            success: true,
            error: None,
            segment: Some(segment),
            wall_time_secs,
        }
    }

    /// Creates a failed result.
    #[must_use]
    pub fn err(index: u32, error: impl Into<String>, wall_time_secs: f64) -> Self {
        Self {
            index,
            success: false,
            error: Some(error.into()),
            segment: None,
            wall_time_secs,
        }
    }
}

/// Summary statistics for a parallel segment encode batch.
#[derive(Debug, Clone, Default)]
pub struct ParallelSegmentStats {
    /// Total segments submitted.
    pub total_segments: u32,
    /// Number of segments encoded successfully.
    pub succeeded: u32,
    /// Number of segments that failed.
    pub failed: u32,
    /// Total compressed bytes from all successful segments.
    pub total_bytes: u64,
    /// Total wall-clock time in seconds.
    pub wall_time_secs: f64,
}

impl ParallelSegmentStats {
    /// Throughput in segments per second.
    #[must_use]
    pub fn segments_per_second(&self) -> f64 {
        if self.wall_time_secs > 0.0 {
            f64::from(self.succeeded) / self.wall_time_secs
        } else {
            0.0
        }
    }

    /// Failure rate as a fraction in [0.0, 1.0].
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        if self.total_segments == 0 {
            return 0.0;
        }
        f64::from(self.failed) / f64::from(self.total_segments)
    }
}

/// A specification for one segment in a parallel encode batch.
///
/// The caller describes the segment (start/duration/codec) and provides raw
/// pixel/sample data.  The encoder executes each spec independently, which
/// allows rayon to encode all segments concurrently.
#[derive(Debug, Clone)]
pub struct SegmentSpec {
    /// Zero-based segment index.
    pub index: u32,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Codec identifier string (e.g. `"av1"`, `"vp9"`).
    pub codec: String,
    /// Raw frame data for this segment (RGBA bytes, row-major).
    pub frame_data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl SegmentSpec {
    /// Creates a new segment specification.
    #[must_use]
    pub fn new(
        index: u32,
        start_ms: u64,
        duration_ms: u64,
        codec: impl Into<String>,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            index,
            start_ms,
            duration_ms,
            codec: codec.into(),
            frame_data,
            width,
            height,
        }
    }

    /// Expected byte length for the RGBA frame data.
    #[must_use]
    pub fn expected_frame_bytes(&self) -> usize {
        (self.width * self.height * 4) as usize
    }

    /// Returns `true` if the frame data matches the expected dimensions.
    #[must_use]
    pub fn frame_data_valid(&self) -> bool {
        self.frame_data.len() >= self.expected_frame_bytes() && self.width > 0 && self.height > 0
    }
}

/// Encodes a batch of independent GOPs (segments) in parallel using rayon.
///
/// Each [`SegmentSpec`] is a self-contained unit: it carries its own raw
/// frame data and codec hint.  The encoder compresses each spec on a rayon
/// thread and collects the results, maintaining the original ordering.
///
/// # Thread safety
///
/// All state is local to each rayon task.  No shared mutable state is used.
pub struct ParallelSegmentEncoder {
    /// Maximum number of rayon threads (0 = use global pool).
    max_threads: usize,
    /// Accumulated stats from previous encode calls.
    stats: ParallelSegmentStats,
}

impl ParallelSegmentEncoder {
    /// Creates a new parallel segment encoder.
    ///
    /// `max_threads` controls the rayon thread pool size; pass `0` to use
    /// rayon's default (number of logical CPUs).
    #[must_use]
    pub fn new(max_threads: usize) -> Self {
        Self {
            max_threads,
            stats: ParallelSegmentStats::default(),
        }
    }

    /// Returns the current accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> &ParallelSegmentStats {
        &self.stats
    }

    /// Resets the accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ParallelSegmentStats::default();
    }

    /// Encodes all specs in parallel and returns a result per spec.
    ///
    /// Results are returned in the same order as `specs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the rayon thread pool cannot be created.
    pub fn encode_batch(
        &mut self,
        specs: Vec<SegmentSpec>,
    ) -> crate::Result<Vec<ParallelSegmentResult>> {
        use rayon::prelude::*;

        let total = specs.len() as u32;
        let wall_start = std::time::Instant::now();

        // Build a custom thread pool if a limit was requested.
        let results: Vec<ParallelSegmentResult> = if self.max_threads > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.max_threads)
                .build()
                .map_err(|e| {
                    crate::TranscodeError::PipelineError(format!(
                        "Failed to create segment thread pool: {e}"
                    ))
                })?;

            pool.install(|| specs.into_par_iter().map(encode_single_segment).collect())
        } else {
            specs.into_par_iter().map(encode_single_segment).collect()
        };

        let wall_secs = wall_start.elapsed().as_secs_f64();

        // Accumulate stats.
        let succeeded = results.iter().filter(|r| r.success).count() as u32;
        let failed = total - succeeded;
        let total_bytes: u64 = results
            .iter()
            .filter_map(|r| r.segment.as_ref().map(|s| s.size_bytes))
            .sum();

        self.stats.total_segments += total;
        self.stats.succeeded += succeeded;
        self.stats.failed += failed;
        self.stats.total_bytes += total_bytes;
        self.stats.wall_time_secs += wall_secs;

        Ok(results)
    }
}

/// Encode a single segment spec on whichever rayon thread picks it up.
fn encode_single_segment(spec: SegmentSpec) -> ParallelSegmentResult {
    let t0 = std::time::Instant::now();

    if !spec.frame_data_valid() {
        return ParallelSegmentResult::err(
            spec.index,
            format!(
                "Segment {}: invalid frame data ({}×{}, {} bytes)",
                spec.index,
                spec.width,
                spec.height,
                spec.frame_data.len()
            ),
            t0.elapsed().as_secs_f64(),
        );
    }

    // Compress the segment frame data using a simple luma RLE placeholder.
    // In production this would call the appropriate oximedia-codec encoder.
    let compressed = compress_segment_placeholder(&spec.frame_data);

    let segment = EncodedSegment::new(
        spec.index,
        spec.start_ms,
        spec.duration_ms,
        compressed.len() as u64,
        estimate_bitrate_kbps(&compressed, spec.duration_ms),
        spec.codec.clone(),
    );

    ParallelSegmentResult::ok(spec.index, segment, t0.elapsed().as_secs_f64())
}

/// Estimates bitrate in kbps from compressed size and duration.
fn estimate_bitrate_kbps(data: &[u8], duration_ms: u64) -> u32 {
    if duration_ms == 0 {
        return 0;
    }
    // bits / duration_secs / 1000
    let bits = data.len() as u64 * 8;
    let duration_secs = duration_ms as f64 / 1_000.0;
    ((bits as f64 / duration_secs) / 1_000.0) as u32
}

/// Placeholder compressor: simple byte-value RLE on the raw data.
fn compress_segment_placeholder(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(data.len() / 2 + 2);
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run: u8 = 1;
        while i + usize::from(run) < data.len() && data[i + usize::from(run)] == val && run < 255 {
            run += 1;
        }
        out.push(val);
        out.push(run);
        i += usize::from(run);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_config_default() {
        let cfg = SegmentConfig::default();
        assert_eq!(cfg.duration_secs, 6.0);
        assert!(cfg.force_key_at_segment);
    }

    #[test]
    fn test_segment_config_new() {
        let cfg = SegmentConfig::new(4.0, 120, false);
        assert_eq!(cfg.duration_secs, 4.0);
        assert_eq!(cfg.keyframe_interval, 120);
        assert!(!cfg.force_key_at_segment);
    }

    #[test]
    fn test_segment_planner_basic() {
        let cfg = SegmentConfig::default(); // 6s segments
                                            // 180 frames at 30fps = 6 seconds total → 1 segment
        let plan = SegmentPlanner::plan(180, 30.0, &cfg);
        assert_eq!(plan.total_frames, 180);
        assert!(plan.segment_count >= 1);
    }

    #[test]
    fn test_segment_planner_multiple_segments() {
        let cfg = SegmentConfig::new(2.0, 30, true);
        // 60 frames at 30fps = 2s per segment → expect 1 segment start at 0
        let plan = SegmentPlanner::plan(60, 30.0, &cfg);
        assert!(!plan.boundaries.is_empty());
    }

    #[test]
    fn test_segment_planner_keyframe_at_boundary() {
        let cfg = SegmentConfig::new(2.0, 60, true);
        let plan = SegmentPlanner::plan(120, 30.0, &cfg);
        for b in &plan.boundaries {
            assert!(b.is_keyframe, "All boundaries should be keyframes");
        }
    }

    #[test]
    fn test_segment_boundary_timestamp() {
        let cfg = SegmentConfig::new(2.0, 60, true);
        let plan = SegmentPlanner::plan(120, 30.0, &cfg);
        assert!((plan.boundaries[0].timestamp_secs - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encoded_segment_end_ms() {
        let seg = EncodedSegment::new(0, 0, 2000, 512_000, 2048, "h264");
        assert_eq!(seg.end_ms(), 2000);
        assert!((seg.duration_secs() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_segment_encoder_add_and_count() {
        let mut enc = SegmentEncoder::new();
        assert_eq!(enc.segment_count(), 0);
        enc.add_segment(EncodedSegment::new(0, 0, 2000, 1024, 4000, "h264"));
        enc.add_segment(EncodedSegment::new(1, 2000, 2000, 2048, 8000, "h264"));
        assert_eq!(enc.segment_count(), 2);
    }

    #[test]
    fn test_segment_encoder_total_bytes() {
        let mut enc = SegmentEncoder::new();
        enc.add_segment(EncodedSegment::new(0, 0, 2000, 1000, 4000, "h264"));
        enc.add_segment(EncodedSegment::new(1, 2000, 2000, 2000, 8000, "h264"));
        assert_eq!(enc.total_bytes(), 3000);
    }

    #[test]
    fn test_segment_encoder_average_bitrate() {
        let mut enc = SegmentEncoder::new();
        assert!(enc.average_bitrate_kbps().is_none());
        enc.add_segment(EncodedSegment::new(0, 0, 2000, 1000, 4000, "h264"));
        enc.add_segment(EncodedSegment::new(1, 2000, 2000, 2000, 6000, "h264"));
        assert_eq!(enc.average_bitrate_kbps(), Some(5000));
    }

    #[test]
    fn test_generate_hls_contains_extm3u() {
        let segments = vec![
            EncodedSegment::new(0, 0, 6000, 1000, 4000, "h264"),
            EncodedSegment::new(1, 6000, 6000, 1000, 4000, "h264"),
        ];
        let manifest = SegmentManifest::generate_hls(&segments, "https://cdn.example.com");
        assert!(manifest.contains("#EXTM3U"));
        assert!(manifest.contains("#EXT-X-ENDLIST"));
        assert!(manifest.contains("segment_00000.ts"));
        assert!(manifest.contains("segment_00001.ts"));
    }

    #[test]
    fn test_generate_dash_contains_mpd() {
        let segments = vec![EncodedSegment::new(0, 0, 6000, 1000, 4000, "avc1")];
        let manifest = SegmentManifest::generate_dash(&segments, "https://cdn.example.com");
        assert!(manifest.contains("<?xml"));
        assert!(manifest.contains("<MPD"));
        assert!(manifest.contains("segment_00000.mp4"));
    }

    #[test]
    fn test_generate_hls_empty() {
        let manifest = SegmentManifest::generate_hls(&[], "https://cdn.example.com");
        assert!(manifest.contains("#EXTM3U"));
        assert!(manifest.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_generate_dash_empty() {
        let manifest = SegmentManifest::generate_dash(&[], "https://cdn.example.com");
        assert!(manifest.contains("<?xml"));
    }

    // ── ParallelSegmentEncoder tests ──────────────────────────────────────────

    #[test]
    fn test_segment_spec_new() {
        let spec = SegmentSpec::new(0, 0, 2000, "av1", vec![0u8; 8 * 8 * 4], 8, 8);
        assert_eq!(spec.index, 0);
        assert_eq!(spec.duration_ms, 2000);
        assert_eq!(spec.codec, "av1");
        assert!(spec.frame_data_valid());
    }

    #[test]
    fn test_segment_spec_invalid_frame_data() {
        let spec = SegmentSpec::new(0, 0, 2000, "av1", vec![0u8; 4], 64, 64);
        assert!(
            !spec.frame_data_valid(),
            "undersized frame data should be invalid"
        );
    }

    #[test]
    fn test_segment_spec_expected_bytes() {
        let spec = SegmentSpec::new(0, 0, 1000, "vp9", vec![], 16, 16);
        assert_eq!(spec.expected_frame_bytes(), 16 * 16 * 4);
    }

    #[test]
    fn test_parallel_segment_encoder_single() {
        let mut encoder = ParallelSegmentEncoder::new(2);
        let frame_data = vec![128u8; 64 * 64 * 4]; // grey 64×64
        let spec = SegmentSpec::new(0, 0, 2000, "av1", frame_data, 64, 64);

        let results = encoder.encode_batch(vec![spec]).expect("encode ok");
        assert_eq!(results.len(), 1);
        assert!(results[0].success, "single segment should succeed");
        assert!(results[0].segment.is_some());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_parallel_segment_encoder_multiple_preserves_order() {
        let mut encoder = ParallelSegmentEncoder::new(4);
        let specs: Vec<SegmentSpec> = (0..8)
            .map(|i| {
                let frame_data = vec![(i * 30) as u8; 64 * 64 * 4];
                SegmentSpec::new(i as u32, i as u64 * 2000, 2000, "av1", frame_data, 64, 64)
            })
            .collect();

        let results = encoder.encode_batch(specs).expect("encode ok");
        assert_eq!(results.len(), 8);

        // Results must be in the same order as input specs.
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.index, i as u32, "result order mismatch at index {i}");
            assert!(result.success, "all segments should succeed");
        }
    }

    #[test]
    fn test_parallel_segment_encoder_invalid_spec_fails_gracefully() {
        let mut encoder = ParallelSegmentEncoder::new(2);
        // Invalid: frame_data too small
        let bad_spec = SegmentSpec::new(0, 0, 2000, "av1", vec![0u8; 10], 64, 64);

        let results = encoder
            .encode_batch(vec![bad_spec])
            .expect("encode batch ok");
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "invalid spec should fail");
        assert!(results[0].error.is_some());
    }

    #[test]
    fn test_parallel_segment_encoder_stats() {
        let mut encoder = ParallelSegmentEncoder::new(2);
        let specs: Vec<SegmentSpec> = (0..4)
            .map(|i| {
                let frame_data = vec![64u8; 64 * 64 * 4];
                SegmentSpec::new(i as u32, i as u64 * 1000, 1000, "vp9", frame_data, 64, 64)
            })
            .collect();

        encoder.encode_batch(specs).expect("encode ok");

        let stats = encoder.stats();
        assert_eq!(stats.total_segments, 4);
        assert_eq!(stats.succeeded, 4);
        assert_eq!(stats.failed, 0);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_parallel_segment_encoder_stats_reset() {
        let mut encoder = ParallelSegmentEncoder::new(2);
        let spec = SegmentSpec::new(0, 0, 1000, "av1", vec![0u8; 64 * 64 * 4], 64, 64);
        encoder.encode_batch(vec![spec]).expect("encode ok");
        assert!(encoder.stats().total_segments > 0);

        encoder.reset_stats();
        assert_eq!(encoder.stats().total_segments, 0);
        assert_eq!(encoder.stats().succeeded, 0);
    }

    #[test]
    fn test_parallel_segment_stats_failure_rate() {
        let stats = ParallelSegmentStats {
            total_segments: 10,
            succeeded: 8,
            failed: 2,
            total_bytes: 1_000,
            wall_time_secs: 1.0,
        };
        assert!((stats.failure_rate() - 0.2).abs() < 1e-9);
        assert!((stats.segments_per_second() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_segment_result_ok() {
        let seg = EncodedSegment::new(0, 0, 1000, 512, 4096, "av1");
        let result = ParallelSegmentResult::ok(0, seg, 0.5);
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(result.segment.is_some());
        assert!((result.wall_time_secs - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_segment_result_err() {
        let result = ParallelSegmentResult::err(1, "codec unavailable", 0.1);
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("codec unavailable"));
        assert!(result.segment.is_none());
    }

    #[test]
    fn test_compress_segment_placeholder_empty() {
        assert!(compress_segment_placeholder(&[]).is_empty());
    }

    #[test]
    fn test_compress_segment_placeholder_rle() {
        let data = vec![42u8; 8];
        let compressed = compress_segment_placeholder(&data);
        // One RLE pair: value=42, run=8
        assert_eq!(compressed, vec![42, 8]);
    }

    #[test]
    fn test_estimate_bitrate_kbps_zero_duration() {
        assert_eq!(estimate_bitrate_kbps(&[0u8; 100], 0), 0);
    }

    #[test]
    fn test_estimate_bitrate_kbps_nonzero() {
        // 125 bytes × 8 bits = 1000 bits over 1 second = 1 kbps
        let bps = estimate_bitrate_kbps(&[0u8; 125], 1_000);
        assert_eq!(bps, 1);
    }

    #[test]
    fn test_parallel_segment_encoder_zero_threads() {
        // 0 threads = use rayon global pool
        let mut encoder = ParallelSegmentEncoder::new(0);
        let spec = SegmentSpec::new(0, 0, 1000, "av1", vec![0u8; 8 * 8 * 4], 8, 8);
        let results = encoder.encode_batch(vec![spec]).expect("encode ok");
        assert!(results[0].success);
    }
}
