//! OpenTelemetry Protocol (OTLP) export support.
//!
//! Provides serialization of metric samples into the OTLP JSON format and
//! an HTTP exporter that POSTs metric batches to an OTLP collector endpoint.
//! This module complements the existing Prometheus exporter in
//! [`crate::metric_export`], providing dual-export capability.
//!
//! # Protocol
//!
//! Implements a subset of the OTLP metrics JSON encoding defined in:
//! <https://opentelemetry.io/docs/specs/otlp/>
//!
//! The exporter encodes metrics as `gauge` data points (all metric types
//! map to gauge unless the type is `Counter`, which maps to `sum`).
//!
//! # Example (no-run)
//!
//! ```no_run
//! use oximedia_monitor::otlp_export::{OtlpExporterConfig, OtlpMetricExporter, OtlpResourceAttributes};
//! use oximedia_monitor::metric_export::{MetricBatch, MetricSample, MetricType};
//!
//! let cfg = OtlpExporterConfig::new("http://localhost:4318");
//! let exporter = OtlpMetricExporter::new(cfg);
//! let mut batch = MetricBatch::new();
//! batch.push(MetricSample::new("cpu_usage", 72.5, MetricType::Gauge));
//! // exporter.export_json(&batch).unwrap(); // network call
//! ```

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

// `MonitorError`/`MonitorResult` are only used by `export()` below, which
// performs a real HTTP POST via `reqwest` and is therefore gated out on
// `wasm32` (no blocking-friendly HTTP client there).
#[cfg(not(target_arch = "wasm32"))]
use crate::error::{MonitorError, MonitorResult};
use crate::metric_export::{MetricBatch, MetricType};

// ---------------------------------------------------------------------------
// Resource attributes
// ---------------------------------------------------------------------------

/// OpenTelemetry resource attributes attached to every export.
#[derive(Debug, Clone)]
pub struct OtlpResourceAttributes {
    /// Key-value attributes (e.g. `service.name`, `service.version`).
    pub attributes: BTreeMap<String, String>,
}

impl OtlpResourceAttributes {
    /// Create a resource with a service name.
    #[must_use]
    pub fn with_service(service_name: impl Into<String>) -> Self {
        let mut attrs = BTreeMap::new();
        attrs.insert("service.name".to_string(), service_name.into());
        Self { attributes: attrs }
    }

    /// Add an attribute.
    #[must_use]
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

impl Default for OtlpResourceAttributes {
    fn default() -> Self {
        Self::with_service("oximedia-monitor")
    }
}

// ---------------------------------------------------------------------------
// Exporter configuration
// ---------------------------------------------------------------------------

/// Configuration for the OTLP metric exporter.
#[derive(Debug, Clone)]
pub struct OtlpExporterConfig {
    /// OTLP HTTP endpoint (e.g. `http://localhost:4318`).
    /// The path `/v1/metrics` will be appended automatically.
    pub endpoint: String,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Additional HTTP headers (e.g. authentication tokens).
    pub headers: BTreeMap<String, String>,
    /// Resource attributes for this exporter instance.
    pub resource: OtlpResourceAttributes,
}

impl OtlpExporterConfig {
    /// Create a new config pointing to the given endpoint.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout_ms: 5_000,
            headers: BTreeMap::new(),
            resource: OtlpResourceAttributes::default(),
        }
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Add an HTTP header.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set the resource attributes.
    #[must_use]
    pub fn with_resource(mut self, resource: OtlpResourceAttributes) -> Self {
        self.resource = resource;
        self
    }

    /// Build the full OTLP metrics endpoint URL.
    #[must_use]
    pub fn metrics_url(&self) -> String {
        let base = self.endpoint.trim_end_matches('/');
        format!("{base}/v1/metrics")
    }
}

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------

/// Get current time as unix nanoseconds.
fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Escape a string for JSON embedding.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Serialize resource attributes to OTLP JSON `attributes` array.
fn attrs_to_json(attrs: &BTreeMap<String, String>) -> String {
    let items: Vec<String> = attrs
        .iter()
        .map(|(k, v)| {
            format!(
                "{{\"key\":\"{}\",\"value\":{{\"stringValue\":\"{}\"}}}}",
                json_escape(k),
                json_escape(v)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// OtlpMetricExporter
// ---------------------------------------------------------------------------

/// OTLP metric exporter.
///
/// Serializes [`MetricBatch`] data to OTLP JSON format.  The actual HTTP
/// transport is handled via the `export_to_url` method which is only compiled
/// on non-WASM targets where `reqwest` is available.
#[derive(Debug, Clone)]
pub struct OtlpMetricExporter {
    config: OtlpExporterConfig,
}

impl OtlpMetricExporter {
    /// Create a new exporter with the given configuration.
    #[must_use]
    pub fn new(config: OtlpExporterConfig) -> Self {
        Self { config }
    }

    /// Serialize the batch to OTLP metrics JSON.
    ///
    /// The output is a valid OTLP ExportMetricsServiceRequest JSON body.
    #[must_use]
    pub fn serialize_to_json(&self, batch: &MetricBatch) -> String {
        let now = now_nanos();
        let resource_attrs = attrs_to_json(&self.config.resource.attributes);

        // Build scope_metrics array — one metric per sample for simplicity.
        let metrics_json: Vec<String> = batch.samples.iter().map(|s| {
            let (data_key, data_content) = match s.metric_type {
                MetricType::Counter => {
                    let data = format!(
                        "{{\"dataPoints\":[{{\"attributes\":{},\"startTimeUnixNano\":\"{}\",\"timeUnixNano\":\"{}\",\"asDouble\":{}}}],\"aggregationTemporality\":2,\"isMonotonic\":true}}",
                        attrs_to_json(&s.labels),
                        now,
                        now,
                        s.value
                    );
                    ("sum", data)
                }
                _ => {
                    let data = format!(
                        "{{\"dataPoints\":[{{\"attributes\":{},\"timeUnixNano\":\"{}\",\"asDouble\":{}}}]}}",
                        attrs_to_json(&s.labels),
                        now,
                        s.value
                    );
                    ("gauge", data)
                }
            };

            let help_escaped = json_escape(&s.help);
            format!(
                "{{\"name\":\"{}\",\"description\":\"{}\",\"unit\":\"\",\"{}\":{}}}",
                json_escape(&s.name),
                help_escaped,
                data_key,
                data_content
            )
        }).collect();

        let scope_metrics = format!(
            "{{\"scope\":{{\"name\":\"oximedia-monitor\",\"version\":\"0.1.2\"}},\"metrics\":[{}]}}",
            metrics_json.join(",")
        );

        format!(
            "{{\"resourceMetrics\":[{{\"resource\":{{\"attributes\":{}}},\"scopeMetrics\":[{}]}}]}}",
            resource_attrs,
            scope_metrics
        )
    }

    /// Reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &OtlpExporterConfig {
        &self.config
    }

    /// Export the batch by POSTing to the configured OTLP endpoint.
    ///
    /// This method is only available on non-WASM targets where HTTP networking
    /// is available.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the server returns a
    /// non-2xx status code.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn export(&self, batch: &MetricBatch) -> MonitorResult<()> {
        use std::time::Duration;

        let body = self.serialize_to_json(batch);
        let url = self.config.metrics_url();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(self.config.timeout_ms))
            .build()
            .map_err(MonitorError::Http)?;

        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body);

        for (k, v) in &self.config.headers {
            request = request.header(k.as_str(), v.as_str());
        }

        let response = request.send().await.map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable>"));
            return Err(MonitorError::Other(format!(
                "OTLP export failed: HTTP {status} — {text}"
            )));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric_export::{MetricSample, MetricType};

    fn make_batch() -> MetricBatch {
        let mut batch = MetricBatch::new();
        batch.push(
            MetricSample::new("cpu_usage", 72.5, MetricType::Gauge)
                .with_help("CPU usage percent")
                .with_label("host", "server1"),
        );
        batch.push(
            MetricSample::new("requests_total", 1000.0, MetricType::Counter)
                .with_help("Total requests")
                .with_label("method", "GET"),
        );
        batch
    }

    // -- OtlpResourceAttributes --

    #[test]
    fn test_resource_default_has_service_name() {
        let res = OtlpResourceAttributes::default();
        assert_eq!(
            res.attributes.get("service.name").map(String::as_str),
            Some("oximedia-monitor")
        );
    }

    #[test]
    fn test_resource_with_service() {
        let res = OtlpResourceAttributes::with_service("my-service");
        assert_eq!(
            res.attributes.get("service.name").map(String::as_str),
            Some("my-service")
        );
    }

    #[test]
    fn test_resource_with_attr() {
        let res = OtlpResourceAttributes::default()
            .with_attr("service.version", "1.0.0")
            .with_attr("deployment.environment", "production");
        assert_eq!(
            res.attributes.get("service.version").map(String::as_str),
            Some("1.0.0")
        );
        assert_eq!(
            res.attributes
                .get("deployment.environment")
                .map(String::as_str),
            Some("production")
        );
    }

    // -- OtlpExporterConfig --

    #[test]
    fn test_config_metrics_url_no_trailing_slash() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        assert_eq!(cfg.metrics_url(), "http://localhost:4318/v1/metrics");
    }

    #[test]
    fn test_config_metrics_url_with_trailing_slash() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318/");
        assert_eq!(cfg.metrics_url(), "http://localhost:4318/v1/metrics");
    }

    #[test]
    fn test_config_with_timeout() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318").with_timeout_ms(10_000);
        assert_eq!(cfg.timeout_ms, 10_000);
    }

    #[test]
    fn test_config_with_header() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318")
            .with_header("Authorization", "Bearer token123");
        assert_eq!(
            cfg.headers.get("Authorization").map(String::as_str),
            Some("Bearer token123")
        );
    }

    // -- OtlpMetricExporter serialization --

    #[test]
    fn test_serialize_contains_resource_metrics() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let batch = make_batch();
        let json = exporter.serialize_to_json(&batch);
        assert!(
            json.contains("resourceMetrics"),
            "Output must contain 'resourceMetrics'"
        );
    }

    #[test]
    fn test_serialize_contains_metric_names() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let batch = make_batch();
        let json = exporter.serialize_to_json(&batch);
        assert!(
            json.contains("cpu_usage"),
            "Output must contain 'cpu_usage'"
        );
        assert!(
            json.contains("requests_total"),
            "Output must contain 'requests_total'"
        );
    }

    #[test]
    fn test_serialize_counter_uses_sum() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let mut batch = MetricBatch::new();
        batch.push(MetricSample::new("reqs", 500.0, MetricType::Counter));
        let json = exporter.serialize_to_json(&batch);
        assert!(
            json.contains("\"sum\""),
            "Counter metric must use 'sum' data point type"
        );
        assert!(
            json.contains("isMonotonic"),
            "Counter sum must be marked monotonic"
        );
    }

    #[test]
    fn test_serialize_gauge_uses_gauge() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let mut batch = MetricBatch::new();
        batch.push(MetricSample::new("cpu", 50.0, MetricType::Gauge));
        let json = exporter.serialize_to_json(&batch);
        assert!(
            json.contains("\"gauge\""),
            "Gauge metric must use 'gauge' data point type"
        );
    }

    #[test]
    fn test_serialize_resource_attributes_present() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318").with_resource(
            OtlpResourceAttributes::with_service("media-encoder")
                .with_attr("service.version", "0.1.2"),
        );
        let exporter = OtlpMetricExporter::new(cfg);
        let batch = make_batch();
        let json = exporter.serialize_to_json(&batch);
        assert!(
            json.contains("media-encoder"),
            "Resource service name should appear in output"
        );
        assert!(
            json.contains("0.1.2"),
            "Resource service version should appear in output"
        );
    }

    #[test]
    fn test_serialize_label_attributes() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let batch = make_batch();
        let json = exporter.serialize_to_json(&batch);
        // The sample has label host=server1.
        assert!(
            json.contains("server1"),
            "Data point attributes must include label values"
        );
    }

    #[test]
    fn test_serialize_empty_batch() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let batch = MetricBatch::new();
        let json = exporter.serialize_to_json(&batch);
        assert!(json.contains("resourceMetrics"));
        assert!(json.contains("scopeMetrics"));
    }

    #[test]
    fn test_serialize_json_escaping() {
        let cfg = OtlpExporterConfig::new("http://localhost:4318");
        let exporter = OtlpMetricExporter::new(cfg);
        let mut batch = MetricBatch::new();
        // Metric name with characters that might trip up naive JSON builders.
        batch.push(
            MetricSample::new("metric_with_desc", 1.0, MetricType::Gauge)
                .with_help("Description with \"quotes\" and\\backslash"),
        );
        let json = exporter.serialize_to_json(&batch);
        // JSON should be valid enough that it doesn't contain raw unescaped double-quote
        // inside a string value.
        assert!(json.contains("\\\"quotes\\\""));
    }

    #[test]
    fn test_attrs_to_json_empty() {
        let attrs = BTreeMap::new();
        let json = attrs_to_json(&attrs);
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_attrs_to_json_single() {
        let mut attrs = BTreeMap::new();
        attrs.insert("key".to_string(), "val".to_string());
        let json = attrs_to_json(&attrs);
        assert!(json.contains("\"key\""));
        assert!(json.contains("\"val\""));
        assert!(json.contains("stringValue"));
    }

    #[test]
    fn test_json_escape_special_chars() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(json_escape("path\\to"), "path\\\\to");
        assert_eq!(json_escape("line\nnewline"), "line\\nnewline");
    }

    #[test]
    fn test_exporter_config_accessor() {
        let cfg = OtlpExporterConfig::new("http://collector:4318").with_timeout_ms(3000);
        let exporter = OtlpMetricExporter::new(cfg.clone());
        assert_eq!(exporter.config().timeout_ms, 3000);
        assert_eq!(exporter.config().endpoint, "http://collector:4318");
    }
}
