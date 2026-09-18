//! Prometheus metrics exporter.

use crate::metrics::MetricsCollector;
use std::sync::Arc;

/// Prometheus metrics exporter.
pub struct PrometheusExporter {
    /// Metrics collector.
    collector: Arc<MetricsCollector>,
}

impl PrometheusExporter {
    /// Creates a new Prometheus exporter.
    #[must_use]
    pub fn new(collector: Arc<MetricsCollector>) -> Self {
        Self { collector }
    }

    /// Exports metrics in Prometheus format.
    #[must_use]
    pub fn export(&self) -> String {
        let mut output = String::new();

        let metrics = self.collector.get_metrics();

        for (name, metric) in metrics {
            // Add metric type comment
            let type_str = match metric.metric_type {
                crate::metrics::MetricType::Counter => "counter",
                crate::metrics::MetricType::Gauge => "gauge",
                crate::metrics::MetricType::Histogram => "histogram",
            };

            output.push_str(&format!("# TYPE {} {}\n", name, type_str));

            // Add labels if present
            let labels = if metric.labels.is_empty() {
                String::new()
            } else {
                let label_pairs: Vec<String> = metric
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", label_pairs.join(","))
            };

            output.push_str(&format!("{}{} {}\n", name, labels, metric.value));
        }

        // Add uptime
        let uptime = self.collector.uptime().as_secs();
        output.push_str(&format!("# TYPE uptime_seconds gauge\n"));
        output.push_str(&format!("uptime_seconds {}\n", uptime));

        output
    }
}
