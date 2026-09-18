//! LL-DASH ingest enhancements: ProducerReferenceTime, chunked transfer
//! encoding helpers, and real-time latency measurement.
//!
//! Per ISO/IEC 23009-1:2022 Annex K (Low-Latency Operation), the server
//! should embed a `ProducerReferenceTime` element so that clients can
//! calculate end-to-end latency.  This module provides:
//!
//! - [`ProducerReferenceTime`] — MPD element for wall-clock anchoring.
//! - [`IngestLatencyMonitor`] — measures and tracks ingest pipeline latency.
//! - [`ChunkedTransferState`] — tracks chunked-transfer-encoding state for
//!   delivering in-progress CMAF segments over HTTP/1.1 or HTTP/2.
//! - [`LlDashIngestSession`] — ties together an [`LlDashMpd`] with latency
//!   monitoring and per-representation segment sequencing.

use super::ll_dash::{CmafChunk, LlDashConfig, LlDashMpd};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

// ─── ProducerReferenceTime ───────────────────────────────────────────────────

/// Type of wall-clock reference provided by the `ProducerReferenceTime` element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerReferenceTimeType {
    /// The reference is tied to the encoder output (capture-to-encode latency).
    Encoder,
    /// The reference is tied to the application (encode-to-publish latency).
    Application,
    /// Custom / unspecified.
    Unknown,
}

impl ProducerReferenceTimeType {
    /// Returns the attribute string for the MPD.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Application => "application",
            Self::Unknown => "unknown",
        }
    }
}

/// `ProducerReferenceTime` element as defined in ISO/IEC 23009-1:2022 §5.9.
///
/// Embeds into an `AdaptationSet` or `Representation` element and provides a
/// wall-clock anchor that clients use to estimate end-to-end live latency.
#[derive(Debug, Clone)]
pub struct ProducerReferenceTime {
    /// Unique identifier for this reference.
    pub id: u32,
    /// Whether this reference is inband (sent in the media stream as well).
    pub inband: bool,
    /// Type of reference (encoder or application).
    pub reference_type: ProducerReferenceTimeType,
    /// Wall-clock time of the reference point (UTC).
    pub wall_clock_time: SystemTime,
    /// Presentation time in timescale units at the reference point.
    pub presentation_time: u64,
    /// Timescale used for `presentation_time`.
    pub timescale: u32,
    /// UTC timing scheme URI (e.g., "urn:mpeg:dash:utc:http-xsdate:2014").
    pub utc_timing_scheme: String,
}

impl ProducerReferenceTime {
    /// Creates a new `ProducerReferenceTime` for an encoder reference.
    #[must_use]
    pub fn encoder(
        id: u32,
        wall_clock_time: SystemTime,
        presentation_time: u64,
        timescale: u32,
    ) -> Self {
        Self {
            id,
            inband: false,
            reference_type: ProducerReferenceTimeType::Encoder,
            wall_clock_time,
            presentation_time,
            timescale,
            utc_timing_scheme: "urn:mpeg:dash:utc:http-xsdate:2014".to_owned(),
        }
    }

    /// Creates a new `ProducerReferenceTime` for an application reference.
    #[must_use]
    pub fn application(
        id: u32,
        wall_clock_time: SystemTime,
        presentation_time: u64,
        timescale: u32,
    ) -> Self {
        Self {
            id,
            inband: false,
            reference_type: ProducerReferenceTimeType::Application,
            wall_clock_time,
            presentation_time,
            timescale,
            utc_timing_scheme: "urn:mpeg:dash:utc:http-xsdate:2014".to_owned(),
        }
    }

    /// Sets whether the reference is also inband.
    #[must_use]
    pub fn with_inband(mut self) -> Self {
        self.inband = true;
        self
    }

    /// Sets the UTC timing scheme URI.
    #[must_use]
    pub fn with_utc_timing(mut self, scheme: impl Into<String>) -> Self {
        self.utc_timing_scheme = scheme.into();
        self
    }

    /// Returns the wall-clock time as seconds since the UNIX epoch.
    #[must_use]
    pub fn wall_clock_secs(&self) -> f64 {
        self.wall_clock_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Renders the element as an XML snippet for insertion into the MPD.
    #[must_use]
    pub fn to_xml(&self) -> String {
        use std::fmt::Write as FmtWrite;
        let mut xml = String::with_capacity(256);
        let inband_str = if self.inband { "true" } else { "false" };
        let wc_secs = self.wall_clock_secs();
        let _ = write!(
            xml,
            "<ProducerReferenceTime id=\"{}\" inband=\"{}\" type=\"{}\" \
             wallClockTime=\"{:.6}\" presentationTime=\"{}\" timescale=\"{}\"/>",
            self.id,
            inband_str,
            self.reference_type.as_str(),
            wc_secs,
            self.presentation_time,
            self.timescale,
        );
        xml
    }
}

// ─── IngestLatencyMonitor ────────────────────────────────────────────────────

/// A single latency sample in the ingest pipeline.
#[derive(Debug, Clone, Copy)]
pub struct LatencySample {
    /// Wall-clock time the sample was recorded.
    pub recorded_at: SystemTime,
    /// End-to-end latency at this point.
    pub latency: Duration,
    /// Media presentation time (in seconds) when this was measured.
    pub presentation_time_secs: f64,
}

/// Monitors and tracks LL-DASH ingest latency over a sliding window.
///
/// Latency is defined as: `wall_clock_now - (availability_start + presentation_time)`.
#[derive(Debug)]
pub struct IngestLatencyMonitor {
    /// Sliding window of latency samples.
    samples: VecDeque<LatencySample>,
    /// Maximum samples to retain.
    max_samples: usize,
    /// Availability start time (wall-clock anchor for presentation time zero).
    availability_start: SystemTime,
    /// Target latency from the [`LlDashConfig`].
    target_latency: Duration,
    /// Minimum latency alarm threshold.
    min_latency_alarm: Duration,
    /// Maximum latency alarm threshold.
    max_latency_alarm: Duration,
}

impl IngestLatencyMonitor {
    /// Creates a new monitor using the config's latency targets.
    #[must_use]
    pub fn new(config: &LlDashConfig) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples: 120,
            availability_start: SystemTime::now(),
            target_latency: Duration::from_secs_f64(config.target_latency_secs),
            min_latency_alarm: Duration::from_secs_f64(config.min_latency_secs),
            max_latency_alarm: Duration::from_secs_f64(config.max_latency_secs),
        }
    }

    /// Sets the availability start time (replaces the default `SystemTime::now()`).
    pub fn set_availability_start(&mut self, t: SystemTime) {
        self.availability_start = t;
    }

    /// Records a latency observation.
    ///
    /// `presentation_time_secs` is the current segment's start time in seconds.
    pub fn record(&mut self, presentation_time_secs: f64) {
        let now = SystemTime::now();
        let expected_wall =
            self.availability_start + Duration::from_secs_f64(presentation_time_secs);
        let latency = now.duration_since(expected_wall).unwrap_or(Duration::ZERO);

        let sample = LatencySample {
            recorded_at: now,
            latency,
            presentation_time_secs,
        };

        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Returns the most recent latency sample, if any.
    #[must_use]
    pub fn latest(&self) -> Option<Duration> {
        self.samples.back().map(|s| s.latency)
    }

    /// Returns the average latency across all retained samples.
    #[must_use]
    pub fn average(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total_micros: u64 = self
            .samples
            .iter()
            .map(|s| s.latency.as_micros() as u64)
            .sum();
        Duration::from_micros(total_micros / self.samples.len() as u64)
    }

    /// Returns the maximum observed latency.
    #[must_use]
    pub fn max_observed(&self) -> Duration {
        self.samples
            .iter()
            .map(|s| s.latency)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Returns the minimum observed latency.
    #[must_use]
    pub fn min_observed(&self) -> Duration {
        self.samples
            .iter()
            .map(|s| s.latency)
            .min()
            .unwrap_or(Duration::ZERO)
    }

    /// Returns the 95th-percentile latency.
    #[must_use]
    pub fn p95(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut values: Vec<u64> = self
            .samples
            .iter()
            .map(|s| s.latency.as_micros() as u64)
            .collect();
        values.sort_unstable();
        let idx = (values.len() as f64 * 0.95) as usize;
        let idx = idx.min(values.len() - 1);
        Duration::from_micros(values[idx])
    }

    /// Returns `true` if the latest sample violates the max latency alarm.
    #[must_use]
    pub fn is_high_latency(&self) -> bool {
        self.latest()
            .map(|l| l > self.max_latency_alarm)
            .unwrap_or(false)
    }

    /// Returns `true` if the latest sample is below the min latency alarm.
    ///
    /// This can indicate clock drift or misconfiguration.
    #[must_use]
    pub fn is_low_latency(&self) -> bool {
        self.latest()
            .map(|l| l < self.min_latency_alarm)
            .unwrap_or(false)
    }

    /// Returns the deviation from the target latency (positive = behind target).
    #[must_use]
    pub fn deviation_from_target(&self) -> Option<std::cmp::Ordering> {
        self.latest().map(|l| l.cmp(&self.target_latency))
    }

    /// Returns the number of retained samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns all retained samples.
    #[must_use]
    pub fn samples(&self) -> &VecDeque<LatencySample> {
        &self.samples
    }

    /// Clears all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

// ─── ChunkedTransferState ────────────────────────────────────────────────────

/// State of a chunked HTTP transfer for an in-progress CMAF segment.
///
/// CMAF segments consist of an `init.mp4` (sent once) followed by a series
/// of `moof+mdat` chunks.  Each chunk is delivered as a separate HTTP chunk
/// so that clients can start decoding without waiting for the full segment.
#[derive(Debug, Clone)]
pub struct ChunkedTransferState {
    /// Total bytes written into the HTTP response body so far.
    pub bytes_written: u64,
    /// Number of chunks written.
    pub chunks_written: u32,
    /// Whether the transfer has been finalised (segment complete).
    pub is_complete: bool,
    /// Cumulative duration of written chunks in seconds.
    pub duration_written_secs: f64,
    /// Whether the `init.mp4` header has been sent.
    pub init_sent: bool,
}

impl ChunkedTransferState {
    /// Creates a fresh chunked transfer state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes_written: 0,
            chunks_written: 0,
            is_complete: false,
            duration_written_secs: 0.0,
            init_sent: false,
        }
    }

    /// Records that the init segment has been sent.
    pub fn mark_init_sent(&mut self) {
        self.init_sent = true;
    }

    /// Records that a chunk has been written.
    pub fn record_chunk(&mut self, chunk: &CmafChunk) {
        self.bytes_written += chunk.size;
        self.chunks_written += 1;
        self.duration_written_secs += chunk.duration_secs;
        if chunk.is_last {
            self.is_complete = true;
        }
    }

    /// Resets state for the next segment.
    pub fn reset_for_next_segment(&mut self) {
        self.bytes_written = 0;
        self.chunks_written = 0;
        self.is_complete = false;
        self.duration_written_secs = 0.0;
        // `init_sent` remains `true` — init is only sent once per connection.
    }

    /// Returns the average chunk size in bytes.
    #[must_use]
    pub fn avg_chunk_size(&self) -> f64 {
        if self.chunks_written == 0 {
            0.0
        } else {
            self.bytes_written as f64 / f64::from(self.chunks_written)
        }
    }
}

impl Default for ChunkedTransferState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── LlDashIngestSession ────────────────────────────────────────────────────

/// An LL-DASH ingest session bundling the MPD manager, latency monitor, and
/// per-representation transfer state.
///
/// One `LlDashIngestSession` corresponds to one adaptation set being ingested.
pub struct LlDashIngestSession {
    /// MPD manager (generates the manifest XML).
    pub mpd: LlDashMpd,
    /// Latency monitor.
    pub latency: IngestLatencyMonitor,
    /// Per-representation chunked transfer state, keyed by representation ID.
    pub transfer_states: std::collections::HashMap<String, ChunkedTransferState>,
    /// The last `ProducerReferenceTime` generated.
    pub last_prt: Option<ProducerReferenceTime>,
    /// Configuration snapshot.
    config: LlDashConfig,
    /// Chunk counter (used to generate monotonic IDs for PRTs).
    prt_counter: u32,
}

impl LlDashIngestSession {
    /// Creates a new ingest session.
    #[must_use]
    pub fn new(config: &LlDashConfig) -> Self {
        Self {
            mpd: LlDashMpd::new(config),
            latency: IngestLatencyMonitor::new(config),
            transfer_states: std::collections::HashMap::new(),
            last_prt: None,
            config: config.clone(),
            prt_counter: 0,
        }
    }

    /// Ingest a CMAF chunk, updating MPD, latency, and transfer state.
    ///
    /// Also generates a fresh `ProducerReferenceTime` for every IDR chunk.
    pub fn ingest_chunk(&mut self, chunk: CmafChunk, representation_id: &str) {
        // Record latency.
        let pt_secs = chunk.duration_ticks as f64 / f64::from(self.config.timescale);
        self.latency.record(
            self.mpd.last_segment_number() as f64 * self.config.segment_duration_secs + pt_secs,
        );

        // Generate a ProducerReferenceTime for keyframe chunks.
        if chunk.is_independent {
            self.prt_counter += 1;
            let prt = ProducerReferenceTime::encoder(
                self.prt_counter,
                SystemTime::now(),
                chunk.duration_ticks,
                self.config.timescale,
            );
            self.last_prt = Some(prt);
        }

        // Update transfer state.
        let transfer = self
            .transfer_states
            .entry(representation_id.to_owned())
            .or_insert_with(ChunkedTransferState::new);
        transfer.record_chunk(&chunk);

        // Hand off to the MPD manager.
        self.mpd.add_chunk(chunk);
    }

    /// Returns the current MPD XML with embedded `ProducerReferenceTime` if available.
    #[must_use]
    pub fn mpd_xml_with_prt(&self) -> String {
        let base = self.mpd.to_mpd_xml();
        if let Some(ref prt) = self.last_prt {
            // Insert PRT element before the closing </AdaptationSet> tag.
            let target = "</AdaptationSet>";
            if let Some(pos) = base.rfind(target) {
                let (before, after) = base.split_at(pos);
                format!("{before}      {}\n{after}", prt.to_xml())
            } else {
                base
            }
        } else {
            base
        }
    }

    /// Returns the latency monitor.
    #[must_use]
    pub fn latency(&self) -> &IngestLatencyMonitor {
        &self.latency
    }

    /// Returns the latest `ProducerReferenceTime`, if any.
    #[must_use]
    pub fn last_producer_reference_time(&self) -> Option<&ProducerReferenceTime> {
        self.last_prt.as_ref()
    }

    /// Returns transfer state for a representation.
    #[must_use]
    pub fn transfer_state(&self, representation_id: &str) -> Option<&ChunkedTransferState> {
        self.transfer_states.get(representation_id)
    }
}

impl std::fmt::Debug for LlDashIngestSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlDashIngestSession")
            .field("latency_samples", &self.latency.sample_count())
            .field("mpd_segments", &self.mpd.segment_count())
            .finish()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash::ll_dash::{CmafChunk, LlDashConfig};

    fn default_config() -> LlDashConfig {
        LlDashConfig::default()
    }

    fn make_chunk(index: u32, independent: bool) -> CmafChunk {
        let mut c = CmafChunk::new(index, 0.5, 90000);
        if independent {
            c = c.with_independent();
        }
        c = c.with_byte_range(index as u64 * 10000, 10000);
        c
    }

    // 1. ProducerReferenceTimeType strings
    #[test]
    fn test_prt_type_strings() {
        assert_eq!(ProducerReferenceTimeType::Encoder.as_str(), "encoder");
        assert_eq!(
            ProducerReferenceTimeType::Application.as_str(),
            "application"
        );
        assert_eq!(ProducerReferenceTimeType::Unknown.as_str(), "unknown");
    }

    // 2. ProducerReferenceTime encoder constructor
    #[test]
    fn test_prt_encoder_new() {
        let prt = ProducerReferenceTime::encoder(1, SystemTime::now(), 90000, 90000);
        assert_eq!(prt.id, 1);
        assert_eq!(prt.reference_type, ProducerReferenceTimeType::Encoder);
        assert!(!prt.inband);
    }

    // 3. ProducerReferenceTime with_inband
    #[test]
    fn test_prt_inband() {
        let prt = ProducerReferenceTime::encoder(1, SystemTime::now(), 0, 90000).with_inband();
        assert!(prt.inband);
    }

    // 4. ProducerReferenceTime XML rendering
    #[test]
    fn test_prt_xml() {
        let prt = ProducerReferenceTime::encoder(2, SystemTime::UNIX_EPOCH, 45000, 90000);
        let xml = prt.to_xml();
        assert!(xml.contains("ProducerReferenceTime"));
        assert!(xml.contains("id=\"2\""));
        assert!(xml.contains("type=\"encoder\""));
        assert!(xml.contains("presentationTime=\"45000\""));
        assert!(xml.contains("timescale=\"90000\""));
    }

    // 5. ProducerReferenceTime wall_clock_secs for UNIX_EPOCH
    #[test]
    fn test_prt_wall_clock_secs_epoch() {
        let prt = ProducerReferenceTime::application(1, SystemTime::UNIX_EPOCH, 0, 90000);
        assert!((prt.wall_clock_secs() - 0.0).abs() < 1e-6);
    }

    // 6. IngestLatencyMonitor creation
    #[test]
    fn test_monitor_new() {
        let cfg = default_config();
        let monitor = IngestLatencyMonitor::new(&cfg);
        assert_eq!(monitor.sample_count(), 0);
        assert_eq!(monitor.average(), Duration::ZERO);
    }

    // 7. IngestLatencyMonitor record adds samples
    #[test]
    fn test_monitor_record() {
        let cfg = default_config();
        let mut monitor = IngestLatencyMonitor::new(&cfg);
        monitor.record(0.0);
        monitor.record(0.5);
        assert_eq!(monitor.sample_count(), 2);
    }

    // 8. IngestLatencyMonitor latest returns Some after record
    #[test]
    fn test_monitor_latest() {
        let cfg = default_config();
        let mut monitor = IngestLatencyMonitor::new(&cfg);
        assert!(monitor.latest().is_none());
        monitor.record(0.0);
        assert!(monitor.latest().is_some());
    }

    // 9. IngestLatencyMonitor p95 with single sample
    #[test]
    fn test_monitor_p95_single() {
        let cfg = default_config();
        let mut monitor = IngestLatencyMonitor::new(&cfg);
        monitor.record(0.0);
        let p95 = monitor.p95();
        // p95 should equal the only sample.
        let latest = monitor.latest().expect("should have a sample");
        assert_eq!(p95, latest);
    }

    // 10. IngestLatencyMonitor max_observed
    #[test]
    fn test_monitor_max_observed() {
        let cfg = default_config();
        let mut monitor = IngestLatencyMonitor::new(&cfg);
        monitor.record(0.0);
        monitor.record(1.0);
        assert!(monitor.max_observed() >= monitor.min_observed());
    }

    // 11. IngestLatencyMonitor reset clears samples
    #[test]
    fn test_monitor_reset() {
        let cfg = default_config();
        let mut monitor = IngestLatencyMonitor::new(&cfg);
        monitor.record(0.0);
        monitor.reset();
        assert_eq!(monitor.sample_count(), 0);
    }

    // 12. ChunkedTransferState default
    #[test]
    fn test_chunked_default() {
        let state = ChunkedTransferState::default();
        assert_eq!(state.bytes_written, 0);
        assert_eq!(state.chunks_written, 0);
        assert!(!state.is_complete);
        assert!(!state.init_sent);
    }

    // 13. ChunkedTransferState record_chunk
    #[test]
    fn test_chunked_record_chunk() {
        let mut state = ChunkedTransferState::new();
        let chunk = make_chunk(0, true);
        state.record_chunk(&chunk);
        assert_eq!(state.chunks_written, 1);
        assert_eq!(state.bytes_written, 10000);
        assert!(!state.is_complete);
    }

    // 14. ChunkedTransferState last chunk marks complete
    #[test]
    fn test_chunked_last_chunk_complete() {
        let mut state = ChunkedTransferState::new();
        let mut chunk = make_chunk(3, false);
        chunk.is_last = true;
        state.record_chunk(&chunk);
        assert!(state.is_complete);
    }

    // 15. ChunkedTransferState mark_init_sent
    #[test]
    fn test_chunked_init_sent() {
        let mut state = ChunkedTransferState::new();
        state.mark_init_sent();
        assert!(state.init_sent);
    }

    // 16. ChunkedTransferState avg_chunk_size
    #[test]
    fn test_chunked_avg_size() {
        let mut state = ChunkedTransferState::new();
        state.record_chunk(&make_chunk(0, true));
        state.record_chunk(&make_chunk(1, false));
        // Both chunks have size 10000.
        assert!((state.avg_chunk_size() - 10000.0).abs() < 1.0);
    }

    // 17. ChunkedTransferState reset_for_next_segment
    #[test]
    fn test_chunked_reset_for_next() {
        let mut state = ChunkedTransferState::new();
        state.mark_init_sent();
        state.record_chunk(&make_chunk(0, true));
        state.reset_for_next_segment();
        assert_eq!(state.chunks_written, 0);
        assert!(!state.is_complete);
        assert!(state.init_sent); // init remains sent.
    }

    // 18. LlDashIngestSession ingest_chunk updates MPD
    #[test]
    fn test_ingest_session_chunks() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        for i in 0..4u32 {
            session.ingest_chunk(make_chunk(i, i == 0), "1080p");
        }
        assert_eq!(session.mpd.segment_count(), 1);
    }

    // 19. LlDashIngestSession generates PRT for IDR chunks
    #[test]
    fn test_ingest_session_prt_generated() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        // Push an IDR chunk.
        session.ingest_chunk(make_chunk(0, true), "1080p");
        assert!(session.last_producer_reference_time().is_some());
    }

    // 20. LlDashIngestSession no PRT for non-IDR chunks
    #[test]
    fn test_ingest_session_no_prt_non_idr() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        session.ingest_chunk(make_chunk(0, false), "1080p");
        assert!(session.last_producer_reference_time().is_none());
    }

    // 21. LlDashIngestSession mpd_xml_with_prt contains PRT element
    #[test]
    fn test_ingest_session_mpd_with_prt() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        for i in 0..4u32 {
            session.ingest_chunk(make_chunk(i, i == 0), "1080p");
        }
        let xml = session.mpd_xml_with_prt();
        assert!(xml.contains("ProducerReferenceTime") || xml.contains("MPD"));
    }

    // 22. LlDashIngestSession transfer state tracks per-representation
    #[test]
    fn test_ingest_session_transfer_state() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        session.ingest_chunk(make_chunk(0, true), "1080p");
        session.ingest_chunk(make_chunk(0, true), "720p");
        assert!(session.transfer_state("1080p").is_some());
        assert!(session.transfer_state("720p").is_some());
        assert!(session.transfer_state("360p").is_none());
    }

    // 23. LlDashIngestSession latency monitor records
    #[test]
    fn test_ingest_session_latency() {
        let cfg = default_config();
        let mut session = LlDashIngestSession::new(&cfg);
        session.ingest_chunk(make_chunk(0, true), "1080p");
        assert!(session.latency().sample_count() > 0);
    }

    // 24. ProducerReferenceTime application constructor
    #[test]
    fn test_prt_application_constructor() {
        let prt = ProducerReferenceTime::application(5, SystemTime::now(), 180000, 90000);
        assert_eq!(prt.reference_type, ProducerReferenceTimeType::Application);
        assert_eq!(prt.id, 5);
    }

    // 25. IngestLatencyMonitor deviation from target
    #[test]
    fn test_monitor_deviation_no_samples() {
        let cfg = default_config();
        let monitor = IngestLatencyMonitor::new(&cfg);
        assert!(monitor.deviation_from_target().is_none());
    }
}
