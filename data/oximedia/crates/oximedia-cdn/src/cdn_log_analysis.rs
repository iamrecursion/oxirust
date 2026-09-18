//! CDN access log analysis — parse structured access-log lines, compute
//! per-edge traffic patterns, detect anomalies via statistical thresholds,
//! and export aggregated reports.
//!
//! # Overview
//!
//! `LogAnalyzer` ingests raw access-log lines conforming to a configurable
//! format, stores them in a rolling time-window, and provides:
//!
//! - Per-path and per-edge request counters.
//! - Bandwidth aggregation (bytes transferred).
//! - Cache-hit-ratio per edge PoP.
//! - Status-code distribution.
//! - Anomaly detection via z-score on the rolling request rate.
//! - Top-N URL ranking by request count.
//!
//! No regex crate is used — parsing relies on split-based tokenisation.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by log-analysis operations.
#[derive(Debug, Error)]
pub enum LogAnalysisError {
    /// The log line could not be parsed (bad format).
    #[error("malformed log line: {0}")]
    MalformedLine(String),
    /// A numeric field could not be parsed.
    #[error("invalid numeric field '{field}' in line: {reason}")]
    InvalidNumeric {
        /// Field name.
        field: String,
        /// Reason.
        reason: String,
    },
}

// ─── LogEntry ─────────────────────────────────────────────────────────────────

/// Cache hit/miss classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// The response was served from cache.
    Hit,
    /// The response required a fetch from origin.
    Miss,
    /// Cache status was not determined.
    Unknown,
}

impl CacheStatus {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "HIT" => Self::Hit,
            "MISS" => Self::Miss,
            _ => Self::Unknown,
        }
    }
}

/// A single parsed CDN access-log record.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Edge PoP that handled the request.
    pub edge_id: String,
    /// HTTP method (GET, POST, …).
    pub method: String,
    /// Request path (no query string).
    pub path: String,
    /// HTTP status code (e.g. 200, 404).
    pub status: u16,
    /// Response body size in bytes.
    pub bytes: u64,
    /// Cache status.
    pub cache_status: CacheStatus,
    /// Client IP address (string, no validation).
    pub client_ip: String,
    /// Unix timestamp (seconds) when the request was received.
    pub timestamp_secs: u64,
    /// Time taken to serve the request in milliseconds.
    pub response_time_ms: u32,
}

impl LogEntry {
    /// Parse a tab-separated log line with the following field order:
    ///
    /// ```text
    /// timestamp_secs  edge_id  client_ip  method  path  status  bytes  cache_status  response_time_ms
    /// ```
    ///
    /// Returns [`LogAnalysisError::MalformedLine`] if the line has fewer than
    /// 9 tab-separated fields.
    pub fn parse(line: &str) -> Result<Self, LogAnalysisError> {
        let fields: Vec<&str> = line.splitn(9, '\t').collect();
        if fields.len() < 9 {
            return Err(LogAnalysisError::MalformedLine(line.to_string()));
        }
        let timestamp_secs =
            fields[0]
                .parse::<u64>()
                .map_err(|e| LogAnalysisError::InvalidNumeric {
                    field: "timestamp_secs".into(),
                    reason: e.to_string(),
                })?;
        let edge_id = fields[1].to_string();
        let client_ip = fields[2].to_string();
        let method = fields[3].to_string();
        let path = fields[4].to_string();
        let status = fields[5]
            .parse::<u16>()
            .map_err(|e| LogAnalysisError::InvalidNumeric {
                field: "status".into(),
                reason: e.to_string(),
            })?;
        let bytes = fields[6]
            .parse::<u64>()
            .map_err(|e| LogAnalysisError::InvalidNumeric {
                field: "bytes".into(),
                reason: e.to_string(),
            })?;
        let cache_status = CacheStatus::from_str(fields[7]);
        let response_time_ms =
            fields[8]
                .trim()
                .parse::<u32>()
                .map_err(|e| LogAnalysisError::InvalidNumeric {
                    field: "response_time_ms".into(),
                    reason: e.to_string(),
                })?;

        Ok(Self {
            edge_id,
            method,
            path,
            status,
            bytes,
            cache_status,
            client_ip,
            timestamp_secs,
            response_time_ms,
        })
    }
}

// ─── PathStats ────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single URL path.
#[derive(Debug, Clone, Default)]
pub struct PathStats {
    /// Total requests for this path.
    pub requests: u64,
    /// Total bytes transferred.
    pub bytes: u64,
    /// Cache-hit count.
    pub hits: u64,
    /// Cache-miss count.
    pub misses: u64,
    /// Sum of response times for average computation.
    pub response_time_sum_ms: u64,
    /// 2xx responses.
    pub status_2xx: u64,
    /// 3xx responses.
    pub status_3xx: u64,
    /// 4xx responses.
    pub status_4xx: u64,
    /// 5xx responses.
    pub status_5xx: u64,
}

impl PathStats {
    /// Cache-hit ratio in [0, 1].  Returns 0.0 if no requests.
    pub fn hit_ratio(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.hits as f64 / self.requests as f64
    }

    /// Average response time in milliseconds.  Returns 0.0 if no requests.
    pub fn avg_response_time_ms(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.response_time_sum_ms as f64 / self.requests as f64
    }
}

// ─── EdgeStats ────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single edge PoP.
#[derive(Debug, Clone, Default)]
pub struct EdgeStats {
    /// Total requests handled.
    pub requests: u64,
    /// Total bytes transferred.
    pub bytes: u64,
    /// Cache-hit count.
    pub hits: u64,
    /// Cache-miss count.
    pub misses: u64,
    /// Sum of response times.
    pub response_time_sum_ms: u64,
    /// Status-code distribution.
    pub status_distribution: HashMap<u16, u64>,
}

impl EdgeStats {
    /// Cache-hit ratio in [0, 1].
    pub fn hit_ratio(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.hits as f64 / self.requests as f64
    }

    /// Average response time in milliseconds.
    pub fn avg_response_time_ms(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.response_time_sum_ms as f64 / self.requests as f64
    }
}

// ─── AnomalyDetector ─────────────────────────────────────────────────────────

/// Request rate anomaly detector using a rolling z-score.
///
/// Keeps a sliding window of per-second request counts and flags epochs where
/// the request rate deviates more than `z_threshold` standard deviations from
/// the rolling mean.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Time-bucketed request counts (each bucket = 1 second).
    buckets: VecDeque<u64>,
    /// Maximum window size in seconds.
    window_secs: usize,
    /// Z-score threshold for anomaly flagging.
    pub z_threshold: f64,
    /// Current bucket accumulator.
    current_bucket: u64,
    /// Timestamp (seconds) of the current bucket.
    current_bucket_ts: u64,
}

impl AnomalyDetector {
    /// Create a new detector with `window_secs` rolling history and the given
    /// z-score threshold (default: 3.0 standard deviations).
    pub fn new(window_secs: usize, z_threshold: f64) -> Self {
        Self {
            buckets: VecDeque::with_capacity(window_secs),
            window_secs,
            z_threshold,
            current_bucket: 0,
            current_bucket_ts: 0,
        }
    }

    /// Record a request that arrived at `timestamp_secs`.
    ///
    /// Returns `true` if the current request rate is anomalously high.
    pub fn record(&mut self, timestamp_secs: u64) -> bool {
        if self.current_bucket_ts == 0 {
            self.current_bucket_ts = timestamp_secs;
        }
        if timestamp_secs == self.current_bucket_ts {
            self.current_bucket += 1;
        } else {
            // Advance buckets for each skipped second.
            let skipped = (timestamp_secs - self.current_bucket_ts) as usize;
            for _ in 0..skipped {
                if self.buckets.len() >= self.window_secs {
                    self.buckets.pop_front();
                }
                self.buckets.push_back(self.current_bucket);
                self.current_bucket = 0;
            }
            self.current_bucket_ts = timestamp_secs;
            self.current_bucket = 1;
        }
        self.is_anomalous(self.current_bucket as f64)
    }

    /// Check whether `rate` is anomalous given the rolling window statistics.
    fn is_anomalous(&self, rate: f64) -> bool {
        if self.buckets.len() < 2 {
            return false;
        }
        let mean = self.mean();
        let std = self.std_dev(mean);
        if std < 1e-10 {
            return false;
        }
        let z = (rate - mean) / std;
        z > self.z_threshold
    }

    /// Rolling mean request rate (requests/second).
    pub fn mean(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.buckets.iter().sum();
        sum as f64 / self.buckets.len() as f64
    }

    /// Rolling standard deviation of the request rate.
    pub fn std_dev(&self, mean: f64) -> f64 {
        if self.buckets.len() < 2 {
            return 0.0;
        }
        let variance: f64 = self
            .buckets
            .iter()
            .map(|&b| {
                let diff = b as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.buckets.len() - 1) as f64;
        variance.sqrt()
    }
}

// ─── AnalysisReport ───────────────────────────────────────────────────────────

/// Snapshot report produced by [`LogAnalyzer::report`].
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Total entries processed.
    pub total_entries: u64,
    /// Total bytes transferred across all edges.
    pub total_bytes: u64,
    /// Overall cache-hit ratio.
    pub overall_hit_ratio: f64,
    /// Per-edge statistics.
    pub edge_stats: HashMap<String, EdgeStats>,
    /// Per-path statistics.
    pub path_stats: HashMap<String, PathStats>,
    /// Top-N paths by request count.
    pub top_paths: Vec<(String, u64)>,
    /// Number of anomalous epochs detected.
    pub anomaly_count: u64,
}

// ─── LogAnalyzer ─────────────────────────────────────────────────────────────

/// Ingests CDN access-log entries and maintains rolling statistics.
pub struct LogAnalyzer {
    /// Per-edge statistics.
    edge_stats: HashMap<String, EdgeStats>,
    /// Per-path statistics.
    path_stats: HashMap<String, PathStats>,
    /// Total entries ingested.
    total_entries: u64,
    /// Total bytes across all edges.
    total_bytes: u64,
    /// Total cache hits.
    total_hits: u64,
    /// Total cache misses.
    total_misses: u64,
    /// Anomaly detector.
    anomaly: AnomalyDetector,
    /// Total anomaly epochs detected.
    anomaly_count: u64,
    /// Configurable top-N for URL ranking.
    top_n: usize,
    /// Rolling window for the log buffer (oldest entries evicted beyond window).
    window: Duration,
    /// Log buffer for time-window eviction (entries kept in arrival order).
    buffer: VecDeque<(Instant, LogEntry)>,
}

impl LogAnalyzer {
    /// Create a new analyzer with the given rolling window and top-N setting.
    ///
    /// - `window` — time-window for buffered entries.
    /// - `top_n`  — number of top URLs returned in reports.
    /// - `anomaly_window_secs` — window for the anomaly detector.
    /// - `z_threshold` — z-score threshold for anomaly flagging.
    pub fn new(
        window: Duration,
        top_n: usize,
        anomaly_window_secs: usize,
        z_threshold: f64,
    ) -> Self {
        Self {
            edge_stats: HashMap::new(),
            path_stats: HashMap::new(),
            total_entries: 0,
            total_bytes: 0,
            total_hits: 0,
            total_misses: 0,
            anomaly: AnomalyDetector::new(anomaly_window_secs, z_threshold),
            anomaly_count: 0,
            top_n,
            window,
            buffer: VecDeque::new(),
        }
    }

    /// Ingest a pre-parsed [`LogEntry`].
    ///
    /// Returns `true` if the entry's request rate triggered an anomaly alert.
    pub fn ingest(&mut self, entry: LogEntry) -> bool {
        self.evict_stale();

        // Update edge stats.
        let edge = self.edge_stats.entry(entry.edge_id.clone()).or_default();
        edge.requests += 1;
        edge.bytes += entry.bytes;
        *edge.status_distribution.entry(entry.status).or_insert(0) += 1;
        edge.response_time_sum_ms += entry.response_time_ms as u64;
        match entry.cache_status {
            CacheStatus::Hit => edge.hits += 1,
            CacheStatus::Miss => edge.misses += 1,
            CacheStatus::Unknown => {}
        }

        // Update path stats.
        let ps = self.path_stats.entry(entry.path.clone()).or_default();
        ps.requests += 1;
        ps.bytes += entry.bytes;
        ps.response_time_sum_ms += entry.response_time_ms as u64;
        match entry.cache_status {
            CacheStatus::Hit => ps.hits += 1,
            CacheStatus::Miss => ps.misses += 1,
            CacheStatus::Unknown => {}
        }
        match entry.status {
            200..=299 => ps.status_2xx += 1,
            300..=399 => ps.status_3xx += 1,
            400..=499 => ps.status_4xx += 1,
            500..=599 => ps.status_5xx += 1,
            _ => {}
        }

        // Update global counters.
        self.total_entries += 1;
        self.total_bytes += entry.bytes;
        match entry.cache_status {
            CacheStatus::Hit => self.total_hits += 1,
            CacheStatus::Miss => self.total_misses += 1,
            CacheStatus::Unknown => {}
        }

        // Anomaly detection.
        let anomalous = self.anomaly.record(entry.timestamp_secs);
        if anomalous {
            self.anomaly_count += 1;
        }

        self.buffer.push_back((Instant::now(), entry));
        anomalous
    }

    /// Ingest a raw log line, parsing it first.
    ///
    /// Returns `Ok(anomalous)` or a parse error.
    pub fn ingest_line(&mut self, line: &str) -> Result<bool, LogAnalysisError> {
        let entry = LogEntry::parse(line)?;
        Ok(self.ingest(entry))
    }

    /// Produce a snapshot [`AnalysisReport`].
    pub fn report(&self) -> AnalysisReport {
        let overall_hit_ratio = if self.total_hits + self.total_misses > 0 {
            self.total_hits as f64 / (self.total_hits + self.total_misses) as f64
        } else {
            0.0
        };

        let mut path_counts: Vec<(String, u64)> = self
            .path_stats
            .iter()
            .map(|(p, s)| (p.clone(), s.requests))
            .collect();
        path_counts.sort_by(|a, b| b.1.cmp(&a.1));
        path_counts.truncate(self.top_n);

        AnalysisReport {
            total_entries: self.total_entries,
            total_bytes: self.total_bytes,
            overall_hit_ratio,
            edge_stats: self.edge_stats.clone(),
            path_stats: self.path_stats.clone(),
            top_paths: path_counts,
            anomaly_count: self.anomaly_count,
        }
    }

    /// Request count per edge PoP.
    pub fn requests_per_edge(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self
            .edge_stats
            .iter()
            .map(|(k, s)| (k.clone(), s.requests))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Return the cache-hit ratio for a specific edge PoP.
    pub fn edge_hit_ratio(&self, edge_id: &str) -> Option<f64> {
        self.edge_stats.get(edge_id).map(|s| s.hit_ratio())
    }

    /// Total number of entries currently in the rolling buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Evict entries older than `window` from the buffer.
    fn evict_stale(&mut self) {
        let cutoff = Instant::now()
            .checked_sub(self.window)
            .unwrap_or_else(Instant::now);
        while let Some((ts, _)) = self.buffer.front() {
            if *ts < cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        edge: &str,
        path: &str,
        status: u16,
        bytes: u64,
        cache: CacheStatus,
        ts: u64,
        rt_ms: u32,
    ) -> LogEntry {
        LogEntry {
            edge_id: edge.to_string(),
            method: "GET".to_string(),
            path: path.to_string(),
            status,
            bytes,
            cache_status: cache,
            client_ip: "1.2.3.4".to_string(),
            timestamp_secs: ts,
            response_time_ms: rt_ms,
        }
    }

    // 1. LogEntry::parse round-trip
    #[test]
    fn test_log_entry_parse_ok() {
        let line = "1700000000\tpop-iad\t1.2.3.4\tGET\t/video/intro.mp4\t200\t5242880\tHIT\t12";
        let entry = LogEntry::parse(line).expect("should parse");
        assert_eq!(entry.timestamp_secs, 1_700_000_000);
        assert_eq!(entry.edge_id, "pop-iad");
        assert_eq!(entry.client_ip, "1.2.3.4");
        assert_eq!(entry.method, "GET");
        assert_eq!(entry.path, "/video/intro.mp4");
        assert_eq!(entry.status, 200);
        assert_eq!(entry.bytes, 5_242_880);
        assert_eq!(entry.cache_status, CacheStatus::Hit);
        assert_eq!(entry.response_time_ms, 12);
    }

    // 2. LogEntry::parse rejects short lines
    #[test]
    fn test_log_entry_parse_short_line() {
        let line = "1700000000\tpop-iad\tGET";
        let err = LogEntry::parse(line).unwrap_err();
        assert!(matches!(err, LogAnalysisError::MalformedLine(_)));
    }

    // 3. LogEntry::parse rejects bad status code
    #[test]
    fn test_log_entry_parse_bad_status() {
        let line = "1700000000\tpop-iad\t1.2.3.4\tGET\t/x\tXXX\t100\tHIT\t5";
        let err = LogEntry::parse(line).unwrap_err();
        assert!(matches!(err, LogAnalysisError::InvalidNumeric { field, .. } if field == "status"));
    }

    // 4. CacheStatus::from_str
    #[test]
    fn test_cache_status_from_str() {
        assert_eq!(CacheStatus::from_str("HIT"), CacheStatus::Hit);
        assert_eq!(CacheStatus::from_str("MISS"), CacheStatus::Miss);
        assert_eq!(CacheStatus::from_str("hit"), CacheStatus::Hit);
        assert_eq!(CacheStatus::from_str("STALE"), CacheStatus::Unknown);
    }

    // 5. PathStats hit_ratio
    #[test]
    fn test_path_stats_hit_ratio() {
        let mut ps = PathStats::default();
        assert!((ps.hit_ratio() - 0.0).abs() < 1e-10);
        ps.requests = 10;
        ps.hits = 7;
        ps.misses = 3;
        assert!((ps.hit_ratio() - 0.7).abs() < 1e-10);
    }

    // 6. EdgeStats avg_response_time_ms
    #[test]
    fn test_edge_stats_avg_response_time() {
        let mut es = EdgeStats::default();
        assert!((es.avg_response_time_ms() - 0.0).abs() < 1e-10);
        es.requests = 4;
        es.response_time_sum_ms = 200;
        assert!((es.avg_response_time_ms() - 50.0).abs() < 1e-10);
    }

    // 7. LogAnalyzer ingests entries and accumulates edge stats
    #[test]
    fn test_log_analyzer_ingest() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        analyzer.ingest(make_entry(
            "pop-iad",
            "/a",
            200,
            1000,
            CacheStatus::Hit,
            1,
            10,
        ));
        analyzer.ingest(make_entry(
            "pop-iad",
            "/b",
            200,
            2000,
            CacheStatus::Miss,
            2,
            20,
        ));
        analyzer.ingest(make_entry(
            "pop-lon",
            "/a",
            200,
            500,
            CacheStatus::Hit,
            3,
            5,
        ));

        let report = analyzer.report();
        assert_eq!(report.total_entries, 3);
        assert_eq!(report.total_bytes, 3500);

        let iad = report.edge_stats.get("pop-iad").expect("iad stats");
        assert_eq!(iad.requests, 2);
        assert_eq!(iad.bytes, 3000);
        assert_eq!(iad.hits, 1);
        assert_eq!(iad.misses, 1);

        let lon = report.edge_stats.get("pop-lon").expect("lon stats");
        assert_eq!(lon.requests, 1);
    }

    // 8. Overall hit ratio calculation
    #[test]
    fn test_overall_hit_ratio() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        for _ in 0..7 {
            analyzer.ingest(make_entry("e", "/x", 200, 100, CacheStatus::Hit, 1, 5));
        }
        for _ in 0..3 {
            analyzer.ingest(make_entry("e", "/x", 200, 100, CacheStatus::Miss, 1, 5));
        }
        let report = analyzer.report();
        assert!((report.overall_hit_ratio - 0.7).abs() < 1e-10);
    }

    // 9. Top-N paths by request count
    #[test]
    fn test_top_paths() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 3, 60, 3.0);
        for _ in 0..10 {
            analyzer.ingest(make_entry("e", "/popular", 200, 1, CacheStatus::Hit, 1, 1));
        }
        for _ in 0..5 {
            analyzer.ingest(make_entry("e", "/medium", 200, 1, CacheStatus::Hit, 1, 1));
        }
        analyzer.ingest(make_entry("e", "/rare", 200, 1, CacheStatus::Hit, 1, 1));
        let report = analyzer.report();
        assert_eq!(report.top_paths.len(), 3);
        assert_eq!(report.top_paths[0].0, "/popular");
        assert_eq!(report.top_paths[0].1, 10);
        assert_eq!(report.top_paths[1].0, "/medium");
    }

    // 10. requests_per_edge sorted descending
    #[test]
    fn test_requests_per_edge_sorted() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        for _ in 0..3 {
            analyzer.ingest(make_entry("pop-a", "/x", 200, 1, CacheStatus::Hit, 1, 1));
        }
        analyzer.ingest(make_entry("pop-b", "/x", 200, 1, CacheStatus::Hit, 1, 1));
        let sorted = analyzer.requests_per_edge();
        assert_eq!(sorted[0].0, "pop-a");
        assert_eq!(sorted[0].1, 3);
        assert_eq!(sorted[1].0, "pop-b");
    }

    // 11. edge_hit_ratio returns None for unknown edge
    #[test]
    fn test_edge_hit_ratio_unknown() {
        let analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        assert!(analyzer.edge_hit_ratio("ghost").is_none());
    }

    // 12. AnomalyDetector: no anomaly below threshold
    #[test]
    fn test_anomaly_detector_no_anomaly() {
        let mut det = AnomalyDetector::new(10, 3.0);
        // Steady rate: 10 req/s for 10 seconds
        for ts in 1u64..=10 {
            for _ in 0..10 {
                det.record(ts);
            }
        }
        // At ts=11, same rate → no anomaly
        let anomalous = det.record(11);
        assert!(!anomalous, "steady rate should not be anomalous");
    }

    // 13. AnomalyDetector: spike triggers anomaly
    #[test]
    fn test_anomaly_detector_spike() {
        let mut det = AnomalyDetector::new(20, 2.0);
        // Build a varied baseline so std_dev > 0.
        // Alternate 1 and 3 req/s to create variance.
        for ts in 1u64..=20 {
            let reqs = if ts % 2 == 0 { 1u64 } else { 3u64 };
            for _ in 0..reqs {
                det.record(ts);
            }
        }
        // Advance to ts=21 to flush the last bucket.
        det.record(21);
        // Massive spike: add 200 requests at ts=22
        let mut anomalous = false;
        for _ in 0..200 {
            anomalous |= det.record(22);
        }
        assert!(
            anomalous,
            "spike of 200 req should be anomalous vs baseline mean ~2"
        );
    }

    // 14. AnomalyDetector mean / std_dev
    #[test]
    fn test_anomaly_detector_statistics() {
        let mut det = AnomalyDetector::new(5, 3.0);
        // Add 5 buckets of exactly 10 requests.
        for ts in 1u64..=5 {
            for _ in 0..10 {
                det.record(ts);
            }
        }
        // Force bucket flush by advancing to ts=6.
        det.record(6);
        let mean = det.mean();
        // Buckets: 10, 10, 10, 10, 10 (ts 1-5), and ts=6 still being accumulated.
        assert!(mean > 5.0, "mean should be around 10, got {mean}");
    }

    // 15. ingest_line parses and accumulates
    #[test]
    fn test_ingest_line() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        let line = "1700000100\tpop-syd\t10.0.0.1\tGET\t/stream/live.m3u8\t200\t8192\tMISS\t45";
        analyzer.ingest_line(line).expect("should parse");
        let report = analyzer.report();
        assert_eq!(report.total_entries, 1);
        let syd = report.edge_stats.get("pop-syd").expect("syd");
        assert_eq!(syd.misses, 1);
        assert_eq!(syd.bytes, 8192);
    }

    // 16. Status code distribution
    #[test]
    fn test_status_code_distribution() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        analyzer.ingest(make_entry("e", "/a", 200, 1, CacheStatus::Hit, 1, 1));
        analyzer.ingest(make_entry("e", "/a", 404, 1, CacheStatus::Miss, 1, 1));
        analyzer.ingest(make_entry("e", "/a", 500, 1, CacheStatus::Unknown, 1, 1));
        let report = analyzer.report();
        let ps = report.path_stats.get("/a").expect("path stats");
        assert_eq!(ps.status_2xx, 1);
        assert_eq!(ps.status_4xx, 1);
        assert_eq!(ps.status_5xx, 1);
    }

    // 17. ingest_line returns error on bad input
    #[test]
    fn test_ingest_line_error() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        let err = analyzer.ingest_line("bad data").unwrap_err();
        assert!(matches!(err, LogAnalysisError::MalformedLine(_)));
    }

    // 18. PathStats avg_response_time
    #[test]
    fn test_path_stats_avg_response_time() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 60, 3.0);
        analyzer.ingest(make_entry("e", "/v", 200, 1, CacheStatus::Hit, 1, 100));
        analyzer.ingest(make_entry("e", "/v", 200, 1, CacheStatus::Hit, 2, 200));
        let report = analyzer.report();
        let ps = report.path_stats.get("/v").expect("stats");
        assert!((ps.avg_response_time_ms() - 150.0).abs() < 1e-6);
    }

    // 19. anomaly_count increments on anomalies
    #[test]
    fn test_anomaly_count() {
        let mut analyzer = LogAnalyzer::new(Duration::from_secs(3600), 5, 20, 1.5);
        // Build a varied baseline so std_dev > 0 (alternating 1 and 3 req/s).
        for ts in 1u64..=20 {
            let reqs = if ts % 2 == 0 { 1u64 } else { 3u64 };
            for _ in 0..reqs {
                analyzer.ingest(make_entry("e", "/x", 200, 1, CacheStatus::Hit, ts, 1));
            }
        }
        // Flush the last bucket.
        analyzer.ingest(make_entry("e", "/x", 200, 1, CacheStatus::Hit, 21, 1));
        // Massive spike at ts=22.
        let spike_ts = 22u64;
        for _ in 0..300 {
            analyzer.ingest(make_entry("e", "/x", 200, 1, CacheStatus::Hit, spike_ts, 1));
        }
        let report = analyzer.report();
        assert!(report.anomaly_count > 0, "should have detected anomalies");
    }

    // 20. buffer eviction respects window
    #[test]
    fn test_buffer_len_bounded() {
        let mut analyzer = LogAnalyzer::new(Duration::from_millis(1), 5, 60, 3.0);
        for ts in 0u64..10 {
            analyzer.ingest(make_entry("e", "/x", 200, 1, CacheStatus::Hit, ts, 1));
        }
        // Sleep to let the window expire.
        std::thread::sleep(Duration::from_millis(5));
        // The next ingest call evicts stale entries.
        analyzer.ingest(make_entry("e", "/x", 200, 1, CacheStatus::Hit, 100, 1));
        // After eviction, only the latest entry should remain.
        assert_eq!(analyzer.buffer_len(), 1);
    }
}
