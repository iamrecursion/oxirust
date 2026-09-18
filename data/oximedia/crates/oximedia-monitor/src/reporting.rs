//! Monitoring report generation.
//!
//! Provides structured report types and serialisation for stream monitoring.

/// Audio quality statistics for a monitoring period.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AudioStats {
    /// Average loudness in LUFS.
    pub avg_loudness_lufs: f64,
    /// Peak loudness in LUFS.
    pub peak_loudness_lufs: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Loudness range in LU.
    pub loudness_range_lu: f64,
    /// Number of audio dropouts detected.
    pub dropout_count: u32,
}

impl AudioStats {
    /// Create default audio stats (all zeros).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            avg_loudness_lufs: 0.0,
            peak_loudness_lufs: 0.0,
            true_peak_dbtp: 0.0,
            loudness_range_lu: 0.0,
            dropout_count: 0,
        }
    }

    /// Returns `true` if any dropout was detected.
    #[must_use]
    pub fn has_dropouts(&self) -> bool {
        self.dropout_count > 0
    }
}

/// Video quality statistics for a monitoring period.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VideoStats {
    /// Average bitrate in kbps.
    pub avg_bitrate_kbps: u64,
    /// Peak bitrate in kbps.
    pub peak_bitrate_kbps: u64,
    /// Number of freeze events detected.
    pub freeze_count: u32,
    /// Total freeze duration in milliseconds.
    pub total_freeze_ms: u64,
    /// Number of dropped frames.
    pub frame_drop_count: u64,
}

impl VideoStats {
    /// Create default video stats (all zeros).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            avg_bitrate_kbps: 0,
            peak_bitrate_kbps: 0,
            freeze_count: 0,
            total_freeze_ms: 0,
            frame_drop_count: 0,
        }
    }

    /// Returns `true` if any freeze event occurred.
    #[must_use]
    pub fn has_freezes(&self) -> bool {
        self.freeze_count > 0
    }
}

/// A single anomaly entry in a report.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReportAnomaly {
    /// String identifier for the anomaly type.
    pub anomaly_type: String,
    /// Timestamp of the anomaly (ms since epoch).
    pub timestamp: u64,
    /// Severity score in [0.0, 1.0].
    pub severity: f64,
}

impl ReportAnomaly {
    /// Create a new report anomaly.
    #[must_use]
    pub fn new(anomaly_type: impl Into<String>, timestamp: u64, severity: f64) -> Self {
        Self {
            anomaly_type: anomaly_type.into(),
            timestamp,
            severity: severity.clamp(0.0, 1.0),
        }
    }
}

/// A complete monitoring report for a stream over a time range.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MonitorReport {
    /// Report start time (ms since epoch).
    pub start_time: u64,
    /// Report end time (ms since epoch).
    pub end_time: u64,
    /// Stream identifier.
    pub stream_id: String,
    /// Audio quality statistics.
    pub audio_stats: AudioStats,
    /// Video quality statistics.
    pub video_stats: VideoStats,
    /// Anomalies detected during the report period.
    pub anomalies: Vec<ReportAnomaly>,
    /// Stream uptime as a percentage [0.0, 100.0].
    pub uptime_pct: f64,
}

impl MonitorReport {
    /// Create a new report.
    #[must_use]
    pub fn new(
        stream_id: impl Into<String>,
        start_time: u64,
        end_time: u64,
        audio_stats: AudioStats,
        video_stats: VideoStats,
        anomalies: Vec<ReportAnomaly>,
        uptime_pct: f64,
    ) -> Self {
        Self {
            start_time,
            end_time,
            stream_id: stream_id.into(),
            audio_stats,
            video_stats,
            anomalies,
            uptime_pct: uptime_pct.clamp(0.0, 100.0),
        }
    }

    /// Return the duration of the report period in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Return the total number of anomalies.
    #[must_use]
    pub fn anomaly_count(&self) -> usize {
        self.anomalies.len()
    }

    /// Return whether the report is considered healthy (no critical anomalies,
    /// uptime above 99%).
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.uptime_pct >= 99.0
            && self.anomalies.iter().all(|a| a.severity < 0.8)
            && !self.video_stats.has_freezes()
            && !self.audio_stats.has_dropouts()
    }
}

/// Generate a basic monitoring report for a stream.
///
/// In a real system this would pull data from a database or ring buffer.
/// Here we construct a default-healthy report as a skeleton.
#[must_use]
pub fn generate_report(stream_id: &str, start: u64, end: u64) -> MonitorReport {
    MonitorReport::new(
        stream_id,
        start,
        end,
        AudioStats::zero(),
        VideoStats::zero(),
        Vec::new(),
        100.0,
    )
}

/// Serialise a `MonitorReport` to a JSON string.
///
/// The implementation is hand-rolled to avoid pulling in `serde` derive on
/// these types; it produces a compact JSON object.
#[must_use]
pub fn report_to_json(report: &MonitorReport) -> String {
    let anomaly_json: Vec<String> = report
        .anomalies
        .iter()
        .map(|a| {
            format!(
                r#"{{"type":"{}","timestamp":{},"severity":{:.4}}}"#,
                escape_json(&a.anomaly_type),
                a.timestamp,
                a.severity
            )
        })
        .collect();

    format!(
        r#"{{"stream_id":"{sid}","start_time":{start},"end_time":{end},"duration_ms":{dur},"uptime_pct":{uptime:.2},"audio":{{"avg_loudness_lufs":{avg_l:.4},"peak_loudness_lufs":{peak_l:.4},"true_peak_dbtp":{tp:.4},"loudness_range_lu":{lr:.4},"dropout_count":{dc}}},"video":{{"avg_bitrate_kbps":{abr},"peak_bitrate_kbps":{pbr},"freeze_count":{fc},"total_freeze_ms":{tfms},"frame_drop_count":{fdc}}},"anomalies":[{anomalies}]}}"#,
        sid = escape_json(&report.stream_id),
        start = report.start_time,
        end = report.end_time,
        dur = report.duration_ms(),
        uptime = report.uptime_pct,
        avg_l = report.audio_stats.avg_loudness_lufs,
        peak_l = report.audio_stats.peak_loudness_lufs,
        tp = report.audio_stats.true_peak_dbtp,
        lr = report.audio_stats.loudness_range_lu,
        dc = report.audio_stats.dropout_count,
        abr = report.video_stats.avg_bitrate_kbps,
        pbr = report.video_stats.peak_bitrate_kbps,
        fc = report.video_stats.freeze_count,
        tfms = report.video_stats.total_freeze_ms,
        fdc = report.video_stats.frame_drop_count,
        anomalies = anomaly_json.join(","),
    )
}

/// Generate a human-readable text summary of a report.
#[must_use]
pub fn report_summary(report: &MonitorReport) -> String {
    let health = if report.is_healthy() {
        "HEALTHY"
    } else {
        "DEGRADED"
    };
    let duration_s = report.duration_ms() / 1000;

    let mut lines = vec![
        format!("=== Monitor Report: {} ===", report.stream_id),
        format!("Status  : {health}"),
        format!("Duration: {duration_s}s"),
        format!("Uptime  : {:.2}%", report.uptime_pct),
        String::from("--- Audio ---"),
        format!(
            "  Avg loudness : {:.1} LUFS",
            report.audio_stats.avg_loudness_lufs
        ),
        format!(
            "  Peak loudness: {:.1} LUFS",
            report.audio_stats.peak_loudness_lufs
        ),
        format!(
            "  True peak    : {:.1} dBTP",
            report.audio_stats.true_peak_dbtp
        ),
        format!(
            "  Loudness range: {:.1} LU",
            report.audio_stats.loudness_range_lu
        ),
        format!("  Dropouts      : {}", report.audio_stats.dropout_count),
        String::from("--- Video ---"),
        format!(
            "  Avg bitrate  : {} kbps",
            report.video_stats.avg_bitrate_kbps
        ),
        format!(
            "  Peak bitrate : {} kbps",
            report.video_stats.peak_bitrate_kbps
        ),
        format!("  Freeze count : {}", report.video_stats.freeze_count),
        format!("  Total freeze : {} ms", report.video_stats.total_freeze_ms),
        format!("  Frame drops  : {}", report.video_stats.frame_drop_count),
        format!("--- Anomalies ({}) ---", report.anomalies.len()),
    ];

    for a in &report.anomalies {
        lines.push(format!(
            "  [{}ms] {} (severity: {:.2})",
            a.timestamp, a.anomaly_type, a.severity
        ));
    }

    lines.join("\n")
}

/// Escape a string for safe embedding in JSON.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_audio() -> AudioStats {
        AudioStats {
            avg_loudness_lufs: -23.0,
            peak_loudness_lufs: -18.0,
            true_peak_dbtp: -1.0,
            loudness_range_lu: 8.5,
            dropout_count: 0,
        }
    }

    fn sample_video() -> VideoStats {
        VideoStats {
            avg_bitrate_kbps: 4000,
            peak_bitrate_kbps: 6000,
            freeze_count: 0,
            total_freeze_ms: 0,
            frame_drop_count: 0,
        }
    }

    fn make_report() -> MonitorReport {
        MonitorReport::new(
            "stream-01",
            1_000_000,
            1_060_000,
            sample_audio(),
            sample_video(),
            Vec::new(),
            99.9,
        )
    }

    #[test]
    fn test_report_duration_ms() {
        let r = make_report();
        assert_eq!(r.duration_ms(), 60_000);
    }

    #[test]
    fn test_report_is_healthy_when_no_issues() {
        let r = make_report();
        assert!(r.is_healthy());
    }

    #[test]
    fn test_report_unhealthy_low_uptime() {
        let mut r = make_report();
        r.uptime_pct = 95.0;
        assert!(!r.is_healthy());
    }

    #[test]
    fn test_report_unhealthy_with_freeze() {
        let mut r = make_report();
        r.video_stats.freeze_count = 2;
        assert!(!r.is_healthy());
    }

    #[test]
    fn test_report_unhealthy_with_dropout() {
        let mut r = make_report();
        r.audio_stats.dropout_count = 1;
        assert!(!r.is_healthy());
    }

    #[test]
    fn test_report_unhealthy_critical_anomaly() {
        let mut r = make_report();
        r.anomalies
            .push(ReportAnomaly::new("BitrateSpike", 1000, 0.9));
        assert!(!r.is_healthy());
    }

    #[test]
    fn test_report_anomaly_count() {
        let mut r = make_report();
        r.anomalies.push(ReportAnomaly::new("A", 0, 0.1));
        r.anomalies.push(ReportAnomaly::new("B", 1, 0.2));
        assert_eq!(r.anomaly_count(), 2);
    }

    #[test]
    fn test_generate_report_defaults() {
        let r = generate_report("test-stream", 0, 10_000);
        assert_eq!(r.stream_id, "test-stream");
        assert_eq!(r.start_time, 0);
        assert_eq!(r.end_time, 10_000);
        assert!((r.uptime_pct - 100.0).abs() < f64::EPSILON);
        assert!(r.anomalies.is_empty());
    }

    #[test]
    fn test_report_to_json_contains_stream_id() {
        let r = make_report();
        let json = report_to_json(&r);
        assert!(json.contains("stream-01"), "JSON should contain stream_id");
    }

    #[test]
    fn test_report_to_json_contains_uptime() {
        let r = make_report();
        let json = report_to_json(&r);
        assert!(
            json.contains("uptime_pct"),
            "JSON should contain uptime_pct"
        );
    }

    #[test]
    fn test_report_to_json_anomaly_included() {
        let mut r = make_report();
        r.anomalies
            .push(ReportAnomaly::new("AudioClipping", 5000, 0.7));
        let json = report_to_json(&r);
        assert!(
            json.contains("AudioClipping"),
            "JSON should contain anomaly type"
        );
    }

    #[test]
    fn test_report_summary_contains_stream_id() {
        let r = make_report();
        let summary = report_summary(&r);
        assert!(summary.contains("stream-01"));
    }

    #[test]
    fn test_report_summary_health_status() {
        let r = make_report();
        let summary = report_summary(&r);
        assert!(summary.contains("HEALTHY"));
    }

    #[test]
    fn test_report_summary_degraded_status() {
        let mut r = make_report();
        r.uptime_pct = 90.0;
        let summary = report_summary(&r);
        assert!(summary.contains("DEGRADED"));
    }

    #[test]
    fn test_audio_stats_has_dropouts() {
        let mut a = sample_audio();
        assert!(!a.has_dropouts());
        a.dropout_count = 3;
        assert!(a.has_dropouts());
    }

    #[test]
    fn test_video_stats_has_freezes() {
        let mut v = sample_video();
        assert!(!v.has_freezes());
        v.freeze_count = 1;
        assert!(v.has_freezes());
    }

    #[test]
    fn test_report_anomaly_severity_clamped() {
        let a = ReportAnomaly::new("test", 0, 99.0);
        assert!((a.severity - 1.0).abs() < f64::EPSILON);
    }
}
