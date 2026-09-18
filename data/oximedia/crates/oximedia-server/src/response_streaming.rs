//! Response streaming for large media file downloads.
//!
//! Provides chunked streaming with back-pressure, progress tracking,
//! bandwidth throttling, and adaptive chunk sizing for efficient
//! delivery of large media files.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for response streaming.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Initial chunk size in bytes.
    pub initial_chunk_size: usize,
    /// Minimum chunk size.
    pub min_chunk_size: usize,
    /// Maximum chunk size.
    pub max_chunk_size: usize,
    /// Whether to use adaptive chunk sizing.
    pub adaptive_chunking: bool,
    /// Maximum bandwidth per stream (bytes/sec), 0 = unlimited.
    pub max_bandwidth: u64,
    /// Buffer size for the streaming pipeline.
    pub buffer_size: usize,
    /// Whether to send Content-Length header (if known).
    pub send_content_length: bool,
    /// Whether to support range requests.
    pub support_range: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            initial_chunk_size: 64 * 1024, // 64 KB
            min_chunk_size: 8 * 1024,      // 8 KB
            max_chunk_size: 1024 * 1024,   // 1 MB
            adaptive_chunking: true,
            max_bandwidth: 0, // unlimited
            buffer_size: 8,
            send_content_length: true,
            support_range: true,
        }
    }
}

/// A byte range for partial content delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Start byte (inclusive).
    pub start: u64,
    /// End byte (inclusive).
    pub end: u64,
}

impl ByteRange {
    /// Creates a new byte range.
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Length of the range in bytes.
    pub fn length(&self) -> u64 {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    /// Whether this is a valid range.
    pub fn is_valid(&self) -> bool {
        self.end >= self.start
    }

    /// Whether this range is satisfiable for a file of the given size.
    pub fn is_satisfiable(&self, file_size: u64) -> bool {
        self.start < file_size
    }

    /// Clamps the range to the file size.
    pub fn clamp_to_size(&self, file_size: u64) -> Self {
        Self {
            start: self.start.min(file_size.saturating_sub(1)),
            end: self.end.min(file_size.saturating_sub(1)),
        }
    }

    /// Formats the Content-Range header value.
    pub fn content_range_header(&self, total_size: u64) -> String {
        format!("bytes {}-{}/{}", self.start, self.end, total_size)
    }
}

/// Parses a Range header into byte ranges.
pub fn parse_range_header(header: &str, file_size: u64) -> Vec<ByteRange> {
    let header = header.trim();
    let rest = match header.strip_prefix("bytes=") {
        Some(r) => r,
        None => return vec![],
    };

    let mut ranges = Vec::new();

    for part in rest.split(',') {
        let part = part.trim();
        if let Some((start_str, end_str)) = part.split_once('-') {
            if start_str.is_empty() {
                // Suffix range: -500 = last 500 bytes
                if let Ok(suffix) = end_str.parse::<u64>() {
                    let start = file_size.saturating_sub(suffix);
                    ranges.push(ByteRange::new(start, file_size - 1));
                }
            } else if end_str.is_empty() {
                // Open-ended: 100- = from byte 100 to end
                if let Ok(start) = start_str.parse::<u64>() {
                    if start < file_size {
                        ranges.push(ByteRange::new(start, file_size - 1));
                    }
                }
            } else {
                // Closed range: 100-200
                if let (Ok(start), Ok(end)) = (start_str.parse::<u64>(), end_str.parse::<u64>()) {
                    let end = end.min(file_size - 1);
                    if start <= end && start < file_size {
                        ranges.push(ByteRange::new(start, end));
                    }
                }
            }
        }
    }

    ranges
}

/// Streaming session state.
#[derive(Debug, Clone)]
pub struct StreamSession {
    /// Session ID.
    pub id: String,
    /// File path or resource key.
    pub resource: String,
    /// Total file size.
    pub total_size: u64,
    /// Bytes sent so far.
    pub bytes_sent: u64,
    /// Current chunk size.
    pub current_chunk_size: usize,
    /// When streaming started.
    pub started_at: Instant,
    /// Last chunk sent time.
    pub last_chunk_at: Option<Instant>,
    /// Number of chunks sent.
    pub chunks_sent: u64,
    /// Client IP.
    pub client_ip: Option<String>,
    /// Whether the session has completed.
    pub completed: bool,
    /// Whether the session was aborted.
    pub aborted: bool,
}

impl StreamSession {
    /// Creates a new streaming session.
    pub fn new(id: impl Into<String>, resource: impl Into<String>, total_size: u64) -> Self {
        Self {
            id: id.into(),
            resource: resource.into(),
            total_size,
            bytes_sent: 0,
            current_chunk_size: 64 * 1024,
            started_at: Instant::now(),
            last_chunk_at: None,
            chunks_sent: 0,
            client_ip: None,
            completed: false,
            aborted: false,
        }
    }

    /// Progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.total_size == 0 {
            return 1.0;
        }
        self.bytes_sent as f64 / self.total_size as f64
    }

    /// Remaining bytes.
    pub fn remaining(&self) -> u64 {
        self.total_size.saturating_sub(self.bytes_sent)
    }

    /// Elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Average throughput in bytes/sec.
    pub fn throughput_bps(&self) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed < 0.001 {
            return 0.0;
        }
        self.bytes_sent as f64 / elapsed
    }

    /// Estimated time remaining.
    pub fn eta(&self) -> Duration {
        let bps = self.throughput_bps();
        if bps < 1.0 {
            return Duration::from_secs(u64::MAX);
        }
        let remaining_secs = self.remaining() as f64 / bps;
        Duration::from_secs_f64(remaining_secs)
    }

    /// Records a chunk being sent.
    pub fn record_chunk(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
        self.chunks_sent += 1;
        self.last_chunk_at = Some(Instant::now());
        if self.bytes_sent >= self.total_size {
            self.completed = true;
        }
    }

    /// Marks the session as aborted.
    pub fn abort(&mut self) {
        self.aborted = true;
    }
}

/// Adaptive chunk sizer.
pub struct AdaptiveChunkSizer {
    config: StreamingConfig,
    /// Current chunk size.
    current: usize,
    /// Recent throughput samples.
    throughput_samples: Vec<f64>,
    /// Maximum samples to keep.
    max_samples: usize,
}

impl AdaptiveChunkSizer {
    /// Creates a new sizer.
    pub fn new(config: StreamingConfig) -> Self {
        let current = config.initial_chunk_size;
        Self {
            config,
            current,
            throughput_samples: Vec::new(),
            max_samples: 20,
        }
    }

    /// Returns the current recommended chunk size.
    pub fn chunk_size(&self) -> usize {
        self.current
    }

    /// Records a throughput measurement and adapts chunk size.
    pub fn record_throughput(&mut self, bytes_per_sec: f64) {
        if self.throughput_samples.len() >= self.max_samples {
            self.throughput_samples.remove(0);
        }
        self.throughput_samples.push(bytes_per_sec);

        if !self.config.adaptive_chunking || self.throughput_samples.len() < 3 {
            return;
        }

        let avg: f64 =
            self.throughput_samples.iter().sum::<f64>() / self.throughput_samples.len() as f64;

        // Target: 100ms worth of data per chunk
        let target = (avg * 0.1) as usize;
        self.current = target
            .max(self.config.min_chunk_size)
            .min(self.config.max_chunk_size);
    }

    /// Resets the sizer.
    pub fn reset(&mut self) {
        self.current = self.config.initial_chunk_size;
        self.throughput_samples.clear();
    }
}

/// Statistics for all streaming sessions.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total sessions.
    pub total_sessions: u64,
    /// Active sessions.
    pub active_sessions: usize,
    /// Completed sessions.
    pub completed_sessions: u64,
    /// Aborted sessions.
    pub aborted_sessions: u64,
    /// Total bytes streamed.
    pub total_bytes: u64,
}

impl StreamingStats {
    /// Completion rate.
    pub fn completion_rate(&self) -> f64 {
        let finished = self.completed_sessions + self.aborted_sessions;
        if finished == 0 {
            return 1.0;
        }
        self.completed_sessions as f64 / finished as f64
    }
}

/// Manages active streaming sessions.
pub struct StreamingManager {
    config: StreamingConfig,
    sessions: HashMap<String, StreamSession>,
    stats: StreamingStats,
    next_id: u64,
}

impl StreamingManager {
    /// Creates a new manager.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            sessions: HashMap::new(),
            stats: StreamingStats::default(),
            next_id: 1,
        }
    }

    /// Creates a new streaming session.
    pub fn create_session(&mut self, resource: &str, total_size: u64) -> String {
        let id = format!("stream-{}", self.next_id);
        self.next_id += 1;

        let mut session = StreamSession::new(id.clone(), resource, total_size);
        session.current_chunk_size = self.config.initial_chunk_size;

        self.sessions.insert(id.clone(), session);
        self.stats.total_sessions += 1;
        self.stats.active_sessions = self.sessions.len();

        id
    }

    /// Gets a session.
    pub fn get_session(&self, id: &str) -> Option<&StreamSession> {
        self.sessions.get(id)
    }

    /// Records a chunk sent in a session.
    pub fn record_chunk(&mut self, session_id: &str, bytes: u64) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.record_chunk(bytes);
            self.stats.total_bytes += bytes;
            if session.completed {
                self.stats.completed_sessions += 1;
            }
            true
        } else {
            false
        }
    }

    /// Aborts a session.
    pub fn abort_session(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.abort();
            self.stats.aborted_sessions += 1;
            true
        } else {
            false
        }
    }

    /// Removes completed/aborted sessions.
    pub fn cleanup(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|_, s| !s.completed && !s.aborted);
        self.stats.active_sessions = self.sessions.len();
        before - self.sessions.len()
    }

    /// Returns statistics.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Returns the number of active sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for StreamingManager {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // StreamingConfig

    #[test]
    fn test_default_config() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.initial_chunk_size, 64 * 1024);
        assert!(cfg.adaptive_chunking);
        assert!(cfg.support_range);
    }

    // ByteRange

    #[test]
    fn test_byte_range_length() {
        let r = ByteRange::new(0, 99);
        assert_eq!(r.length(), 100);
    }

    #[test]
    fn test_byte_range_valid() {
        assert!(ByteRange::new(0, 99).is_valid());
        assert!(!ByteRange::new(100, 50).is_valid());
    }

    #[test]
    fn test_byte_range_satisfiable() {
        assert!(ByteRange::new(0, 99).is_satisfiable(100));
        assert!(!ByteRange::new(100, 200).is_satisfiable(100));
    }

    #[test]
    fn test_byte_range_clamp() {
        let r = ByteRange::new(0, 999);
        let clamped = r.clamp_to_size(500);
        assert_eq!(clamped.end, 499);
    }

    #[test]
    fn test_byte_range_content_range_header() {
        let r = ByteRange::new(0, 99);
        assert_eq!(r.content_range_header(1000), "bytes 0-99/1000");
    }

    // parse_range_header

    #[test]
    fn test_parse_closed_range() {
        let ranges = parse_range_header("bytes=0-99", 1000);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges[0].end, 99);
    }

    #[test]
    fn test_parse_open_ended_range() {
        let ranges = parse_range_header("bytes=500-", 1000);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 500);
        assert_eq!(ranges[0].end, 999);
    }

    #[test]
    fn test_parse_suffix_range() {
        let ranges = parse_range_header("bytes=-200", 1000);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 800);
        assert_eq!(ranges[0].end, 999);
    }

    #[test]
    fn test_parse_multiple_ranges() {
        let ranges = parse_range_header("bytes=0-99, 200-299", 1000);
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn test_parse_invalid_range_header() {
        let ranges = parse_range_header("invalid", 1000);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_parse_unsatisfiable_range() {
        let ranges = parse_range_header("bytes=1000-2000", 500);
        assert!(ranges.is_empty());
    }

    // StreamSession

    #[test]
    fn test_session_creation() {
        let session = StreamSession::new("s1", "/media/file.mp4", 1_000_000);
        assert_eq!(session.id, "s1");
        assert_eq!(session.total_size, 1_000_000);
        assert!((session.progress() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_session_record_chunk() {
        let mut session = StreamSession::new("s1", "file", 1000);
        session.record_chunk(500);
        assert_eq!(session.bytes_sent, 500);
        assert_eq!(session.chunks_sent, 1);
        assert!((session.progress() - 0.5).abs() < 1e-9);
        assert!(!session.completed);
    }

    #[test]
    fn test_session_completion() {
        let mut session = StreamSession::new("s1", "file", 100);
        session.record_chunk(100);
        assert!(session.completed);
        assert!((session.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_session_remaining() {
        let mut session = StreamSession::new("s1", "file", 1000);
        session.record_chunk(300);
        assert_eq!(session.remaining(), 700);
    }

    #[test]
    fn test_session_abort() {
        let mut session = StreamSession::new("s1", "file", 1000);
        session.abort();
        assert!(session.aborted);
    }

    #[test]
    fn test_session_zero_size() {
        let session = StreamSession::new("s1", "file", 0);
        assert!((session.progress() - 1.0).abs() < 1e-9);
    }

    // AdaptiveChunkSizer

    #[test]
    fn test_initial_chunk_size() {
        let sizer = AdaptiveChunkSizer::new(StreamingConfig::default());
        assert_eq!(sizer.chunk_size(), 64 * 1024);
    }

    #[test]
    fn test_adaptive_sizing() {
        let mut sizer = AdaptiveChunkSizer::new(StreamingConfig::default());
        // Record high throughput
        for _ in 0..5 {
            sizer.record_throughput(10_000_000.0); // 10 MB/s
        }
        // Should increase chunk size
        let cs = sizer.chunk_size();
        assert!(cs > 64 * 1024);
    }

    #[test]
    fn test_adaptive_sizing_low_throughput() {
        let mut sizer = AdaptiveChunkSizer::new(StreamingConfig::default());
        for _ in 0..5 {
            sizer.record_throughput(50_000.0); // 50 KB/s
        }
        let cs = sizer.chunk_size();
        assert!(cs <= 64 * 1024);
    }

    #[test]
    fn test_sizer_reset() {
        let mut sizer = AdaptiveChunkSizer::new(StreamingConfig::default());
        for _ in 0..5 {
            sizer.record_throughput(10_000_000.0);
        }
        sizer.reset();
        assert_eq!(sizer.chunk_size(), 64 * 1024);
    }

    // StreamingStats

    #[test]
    fn test_stats_completion_rate() {
        let mut stats = StreamingStats::default();
        stats.completed_sessions = 9;
        stats.aborted_sessions = 1;
        assert!((stats.completion_rate() - 0.9).abs() < 1e-9);
    }

    // StreamingManager

    #[test]
    fn test_manager_create_session() {
        let mut mgr = StreamingManager::default();
        let id = mgr.create_session("/media/file.mp4", 1_000_000);
        assert!(mgr.get_session(&id).is_some());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_manager_record_chunk() {
        let mut mgr = StreamingManager::default();
        let id = mgr.create_session("file", 1000);
        assert!(mgr.record_chunk(&id, 500));
        let session = mgr.get_session(&id).expect("should exist");
        assert_eq!(session.bytes_sent, 500);
    }

    #[test]
    fn test_manager_abort() {
        let mut mgr = StreamingManager::default();
        let id = mgr.create_session("file", 1000);
        assert!(mgr.abort_session(&id));
        assert_eq!(mgr.stats().aborted_sessions, 1);
    }

    #[test]
    fn test_manager_cleanup() {
        let mut mgr = StreamingManager::default();
        let id1 = mgr.create_session("a", 100);
        let _id2 = mgr.create_session("b", 100);
        mgr.record_chunk(&id1, 100); // completes
        let cleaned = mgr.cleanup();
        assert_eq!(cleaned, 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_manager_stats_bytes() {
        let mut mgr = StreamingManager::default();
        let id = mgr.create_session("file", 1000);
        mgr.record_chunk(&id, 300);
        mgr.record_chunk(&id, 200);
        assert_eq!(mgr.stats().total_bytes, 500);
    }
}
