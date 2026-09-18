//! Atomic-counter based metrics collector for the OxiMedia REST server.
//!
//! Exposes a `ServerMetricsCollector` that tracks request/transfer/transcode
//! counters and latency histograms using lock-free atomics.  The
//! `render_prometheus` method produces output compatible with the Prometheus
//! text exposition format 0.0.4.
//!
//! # Route
//!
//! `GET /metrics` → `text/plain; version=0.0.4`
//!
//! Wrap `ServerMetricsCollector` in `Arc` and store it in `AppState` (or share
//! it via `axum` extension) to collect metrics from handlers.

use axum::{extract::State, http::header::HeaderValue, response::Response};
use std::{
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

// ── Histogram bucket boundaries (request latency in milliseconds) ─────────────

/// Bucket upper bounds in milliseconds (exclusive).
const LATENCY_BUCKETS_MS: [u64; 6] = [10, 50, 100, 500, 1_000, u64::MAX];

/// Human-readable bucket labels for Prometheus `le` labels.
const LATENCY_BUCKET_LABELS: [&str; 6] = ["0.01", "0.05", "0.1", "0.5", "1.0", "+Inf"];

/// Number of histogram buckets.
const BUCKET_COUNT: usize = LATENCY_BUCKETS_MS.len();

// ── ServerMetricsCollector ────────────────────────────────────────────────────

/// Lock-free metrics collector for the OxiMedia REST server.
///
/// All fields are `AtomicU64` / `AtomicI64` so they can be updated from any
/// number of async tasks without acquiring a mutex.
pub struct ServerMetricsCollector {
    // ── Counters ──────────────────────────────────────────────────────────────
    /// Total HTTP requests handled.
    pub requests_total: AtomicU64,
    /// Total bytes received in upload bodies.
    pub upload_bytes_total: AtomicU64,
    /// Total bytes sent to clients (streaming / download).
    pub download_bytes_total: AtomicU64,
    /// Total transcode jobs submitted.
    pub transcode_jobs_total: AtomicU64,
    /// Total transcode jobs that ended in failure.
    pub transcode_jobs_failed: AtomicU64,

    // ── Gauges ────────────────────────────────────────────────────────────────
    /// Number of currently open HTTP connections.
    pub active_connections: AtomicI64,
    /// Number of jobs currently in the pending queue.
    pub queue_depth: AtomicI64,
    /// Total bytes consumed by stored media files.
    pub storage_bytes_used: AtomicI64,

    // ── Latency histogram ─────────────────────────────────────────────────────
    /// Cumulative request counts per latency bucket.
    ///
    /// `latency_buckets[i]` counts requests that completed in ≤ `LATENCY_BUCKETS_MS[i]` ms.
    latency_buckets: [AtomicU64; BUCKET_COUNT],
    /// Sum of all request latencies in milliseconds (for `_sum`).
    latency_sum_ms: AtomicU64,
    /// Total number of latency observations (for `_count`).
    latency_count: AtomicU64,

    /// Process start time (for uptime calculation).
    start_time: Instant,
}

impl ServerMetricsCollector {
    /// Creates a new collector with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            upload_bytes_total: AtomicU64::new(0),
            download_bytes_total: AtomicU64::new(0),
            transcode_jobs_total: AtomicU64::new(0),
            transcode_jobs_failed: AtomicU64::new(0),
            active_connections: AtomicI64::new(0),
            queue_depth: AtomicI64::new(0),
            storage_bytes_used: AtomicI64::new(0),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_sum_ms: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    // ── Counter helpers ───────────────────────────────────────────────────────

    /// Increments the total request counter by 1.
    pub fn inc_requests(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds `bytes` to the upload byte counter.
    pub fn add_upload_bytes(&self, bytes: u64) {
        self.upload_bytes_total.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Adds `bytes` to the download byte counter.
    pub fn add_download_bytes(&self, bytes: u64) {
        self.download_bytes_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increments the transcode job counter by 1.
    pub fn inc_transcode_jobs(&self) {
        self.transcode_jobs_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the failed transcode job counter by 1.
    pub fn inc_transcode_jobs_failed(&self) {
        self.transcode_jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    // ── Gauge helpers ─────────────────────────────────────────────────────────

    /// Increments the active-connection gauge by 1.
    pub fn connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the active-connection gauge by 1 (floor: 0).
    pub fn connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Sets the job-queue depth gauge.
    pub fn set_queue_depth(&self, depth: i64) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Sets the storage-bytes-used gauge.
    pub fn set_storage_bytes_used(&self, bytes: i64) {
        self.storage_bytes_used.store(bytes, Ordering::Relaxed);
    }

    // ── Histogram helpers ─────────────────────────────────────────────────────

    /// Records a single request latency observation (duration in milliseconds).
    ///
    /// Updates all cumulative buckets for which `latency_ms ≤ upper_bound`.
    pub fn observe_latency_ms(&self, latency_ms: u64) {
        for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if latency_ms <= bound {
                self.latency_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.latency_sum_ms.fetch_add(latency_ms, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
    }

    // ── Prometheus text rendering ─────────────────────────────────────────────

    /// Renders all metrics in Prometheus text format (exposition format 0.0.4).
    ///
    /// Output structure for each metric family:
    /// ```text
    /// # HELP <name> <description>
    /// # TYPE <name> <type>
    /// <name>[{labels}] <value>
    /// ```
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // ── Counters ──────────────────────────────────────────────────────────
        write_counter(
            &mut out,
            "oximedia_requests_total",
            "Total HTTP requests handled by the media server.",
            self.requests_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut out,
            "oximedia_upload_bytes_total",
            "Total bytes received in upload request bodies.",
            self.upload_bytes_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut out,
            "oximedia_download_bytes_total",
            "Total bytes sent to clients in streaming and download responses.",
            self.download_bytes_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut out,
            "oximedia_transcode_jobs_total",
            "Total transcode jobs submitted.",
            self.transcode_jobs_total.load(Ordering::Relaxed),
        );
        write_counter(
            &mut out,
            "oximedia_transcode_jobs_failed_total",
            "Total transcode jobs that ended in failure.",
            self.transcode_jobs_failed.load(Ordering::Relaxed),
        );

        // ── Gauges ────────────────────────────────────────────────────────────
        write_gauge_i64(
            &mut out,
            "oximedia_active_connections",
            "Number of currently open HTTP connections.",
            self.active_connections.load(Ordering::Relaxed),
        );
        write_gauge_i64(
            &mut out,
            "oximedia_queue_depth",
            "Number of transcode jobs currently pending in the queue.",
            self.queue_depth.load(Ordering::Relaxed),
        );
        write_gauge_i64(
            &mut out,
            "oximedia_storage_bytes_used",
            "Total bytes consumed by stored media files.",
            self.storage_bytes_used.load(Ordering::Relaxed),
        );

        // ── Uptime gauge ──────────────────────────────────────────────────────
        let uptime_secs = self.start_time.elapsed().as_secs();
        write_gauge_u64(
            &mut out,
            "oximedia_uptime_seconds",
            "Seconds since the server process started.",
            uptime_secs,
        );

        // ── Request latency histogram ─────────────────────────────────────────
        let hist_name = "oximedia_request_duration_seconds";
        out.push_str(&format!(
            "# HELP {} Histogram of HTTP request latencies in seconds.\n",
            hist_name
        ));
        out.push_str(&format!("# TYPE {} histogram\n", hist_name));

        for (i, label) in LATENCY_BUCKET_LABELS.iter().enumerate() {
            let count = self.latency_buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{}_bucket{{le=\"{}\"}} {}\n",
                hist_name, label, count
            ));
        }

        // Prometheus expects _sum in seconds; convert from milliseconds.
        let sum_ms = self.latency_sum_ms.load(Ordering::Relaxed);
        let count = self.latency_count.load(Ordering::Relaxed);
        out.push_str(&format!(
            "{}_sum {:.6}\n",
            hist_name,
            sum_ms as f64 / 1000.0
        ));
        out.push_str(&format!("{}_count {}\n", hist_name, count));

        out
    }
}

impl Default for ServerMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Prometheus text format helpers ────────────────────────────────────────────

fn write_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} counter\n", name));
    out.push_str(&format!("{} {}\n", name, value));
}

fn write_gauge_u64(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} gauge\n", name));
    out.push_str(&format!("{} {}\n", name, value));
}

fn write_gauge_i64(out: &mut String, name: &str, help: &str, value: i64) {
    out.push_str(&format!("# HELP {} {}\n", name, help));
    out.push_str(&format!("# TYPE {} gauge\n", name));
    out.push_str(&format!("{} {}\n", name, value));
}

// ── Route handler ─────────────────────────────────────────────────────────────

/// `GET /metrics` — return Prometheus-format metrics.
///
/// Content-Type is `text/plain; version=0.0.4` as required by the Prometheus
/// data model specification.
pub async fn metrics_handler(State(collector): State<Arc<ServerMetricsCollector>>) -> Response {
    let body = collector.render_prometheus();
    let mut response = axum::response::Response::new(axum::body::Body::from(body));
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    response
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collector() -> ServerMetricsCollector {
        ServerMetricsCollector::new()
    }

    #[test]
    fn test_counters_start_at_zero() {
        let c = make_collector();
        assert_eq!(c.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(c.upload_bytes_total.load(Ordering::Relaxed), 0);
        assert_eq!(c.download_bytes_total.load(Ordering::Relaxed), 0);
        assert_eq!(c.transcode_jobs_total.load(Ordering::Relaxed), 0);
        assert_eq!(c.transcode_jobs_failed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_inc_requests() {
        let c = make_collector();
        c.inc_requests();
        c.inc_requests();
        assert_eq!(c.requests_total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_add_upload_bytes() {
        let c = make_collector();
        c.add_upload_bytes(1024);
        c.add_upload_bytes(2048);
        assert_eq!(c.upload_bytes_total.load(Ordering::Relaxed), 3072);
    }

    #[test]
    fn test_add_download_bytes() {
        let c = make_collector();
        c.add_download_bytes(500);
        assert_eq!(c.download_bytes_total.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn test_transcode_counters() {
        let c = make_collector();
        c.inc_transcode_jobs();
        c.inc_transcode_jobs();
        c.inc_transcode_jobs_failed();
        assert_eq!(c.transcode_jobs_total.load(Ordering::Relaxed), 2);
        assert_eq!(c.transcode_jobs_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_active_connections_gauge() {
        let c = make_collector();
        c.connection_opened();
        c.connection_opened();
        assert_eq!(c.active_connections.load(Ordering::Relaxed), 2);
        c.connection_closed();
        assert_eq!(c.active_connections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_set_queue_depth() {
        let c = make_collector();
        c.set_queue_depth(42);
        assert_eq!(c.queue_depth.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_set_storage_bytes_used() {
        let c = make_collector();
        c.set_storage_bytes_used(1_000_000);
        assert_eq!(c.storage_bytes_used.load(Ordering::Relaxed), 1_000_000);
    }

    #[test]
    fn test_latency_histogram_buckets() {
        let c = make_collector();
        // 5 ms fits in ALL buckets (cumulative): ≤10, ≤50, ≤100, ≤500, ≤1000, +Inf
        c.observe_latency_ms(5);
        assert_eq!(c.latency_buckets[0].load(Ordering::Relaxed), 1); // ≤10ms
        assert_eq!(c.latency_buckets[1].load(Ordering::Relaxed), 1); // ≤50ms
        assert_eq!(c.latency_buckets[3].load(Ordering::Relaxed), 1); // ≤500ms
        assert_eq!(c.latency_buckets[5].load(Ordering::Relaxed), 1); // +Inf

        // 200 ms: does NOT fit in ≤10, ≤50, ≤100; DOES fit in ≤500, ≤1000, +Inf
        c.observe_latency_ms(200);
        assert_eq!(c.latency_buckets[0].load(Ordering::Relaxed), 1); // ≤10ms unchanged
        assert_eq!(c.latency_buckets[1].load(Ordering::Relaxed), 1); // ≤50ms unchanged
        assert_eq!(c.latency_buckets[2].load(Ordering::Relaxed), 1); // ≤100ms unchanged
        assert_eq!(c.latency_buckets[3].load(Ordering::Relaxed), 2); // ≤500ms: 5ms + 200ms
        assert_eq!(c.latency_buckets[4].load(Ordering::Relaxed), 2); // ≤1000ms: both
        assert_eq!(c.latency_buckets[5].load(Ordering::Relaxed), 2); // +Inf: both

        assert_eq!(c.latency_count.load(Ordering::Relaxed), 2);
        assert_eq!(c.latency_sum_ms.load(Ordering::Relaxed), 205);
    }

    #[test]
    fn test_render_prometheus_contains_help_and_type() {
        let c = make_collector();
        c.inc_requests();
        let output = c.render_prometheus();
        assert!(
            output.contains("# HELP oximedia_requests_total"),
            "missing HELP"
        );
        assert!(
            output.contains("# TYPE oximedia_requests_total counter"),
            "missing TYPE"
        );
        assert!(output.contains("oximedia_requests_total 1"), "wrong value");
    }

    #[test]
    fn test_render_prometheus_histogram_buckets_present() {
        let c = make_collector();
        c.observe_latency_ms(7);
        let output = c.render_prometheus();
        assert!(output.contains("oximedia_request_duration_seconds_bucket{le=\"0.01\"}"));
        assert!(output.contains("oximedia_request_duration_seconds_bucket{le=\"+Inf\"}"));
        assert!(output.contains("oximedia_request_duration_seconds_sum"));
        assert!(output.contains("oximedia_request_duration_seconds_count 1"));
    }

    #[test]
    fn test_render_prometheus_sum_in_seconds() {
        let c = make_collector();
        c.observe_latency_ms(1000); // 1000 ms → 1.0 s
        let output = c.render_prometheus();
        // Sum should be rendered as seconds.
        assert!(output.contains("oximedia_request_duration_seconds_sum 1.000000"));
    }

    #[test]
    fn test_render_prometheus_uptime_present() {
        let c = make_collector();
        let output = c.render_prometheus();
        assert!(output.contains("oximedia_uptime_seconds"));
    }

    #[test]
    fn test_bucket_label_count_matches_bounds() {
        assert_eq!(LATENCY_BUCKETS_MS.len(), LATENCY_BUCKET_LABELS.len());
        assert_eq!(LATENCY_BUCKETS_MS.len(), BUCKET_COUNT);
    }
}
