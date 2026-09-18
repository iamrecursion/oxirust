//! Monitoring Module
//!
//! Provides monitoring capabilities for distributed optimization systems.

use crate::error::{MetricsError, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Monitoring manager
#[derive(Debug, Clone)]
pub struct MonitoringManager {
    node_id: String,
    metrics: HashMap<String, MetricValue>,
    alerts: Vec<Alert>,
    thresholds: HashMap<String, Threshold>,
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timer(Duration),
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub metric_name: String,
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Threshold {
    pub warning_level: f64,
    pub critical_level: f64,
}

impl MonitoringManager {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            metrics: HashMap::new(),
            alerts: Vec::new(),
            thresholds: HashMap::new(),
        }
    }

    pub fn record_metric(&mut self, name: String, value: MetricValue) -> Result<()> {
        self.metrics.insert(name.clone(), value);
        self.check_thresholds(&name)?;
        Ok(())
    }

    pub fn set_threshold(&mut self, metric_name: String, threshold: Threshold) {
        self.thresholds.insert(metric_name, threshold);
    }

    pub fn get_metric(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.get(name)
    }

    pub fn get_alerts(&self) -> &[Alert] {
        &self.alerts
    }

    fn check_thresholds(&mut self, metric_name: &str) -> Result<()> {
        if let Some(threshold) = self.thresholds.get(metric_name) {
            if let Some(metric) = self.metrics.get(metric_name) {
                let value = match metric {
                    MetricValue::Gauge(v) => *v,
                    MetricValue::Counter(v) => *v as f64,
                    MetricValue::Timer(d) => d.as_secs_f64(),
                    _ => return Ok(()),
                };

                if value >= threshold.critical_level {
                    self.alerts.push(Alert {
                        metric_name: metric_name.to_string(),
                        level: AlertLevel::Critical,
                        message: format!("Critical threshold exceeded: {}", value),
                        timestamp: Instant::now(),
                    });
                } else if value >= threshold.warning_level {
                    self.alerts.push(Alert {
                        metric_name: metric_name.to_string(),
                        level: AlertLevel::Warning,
                        message: format!("Warning threshold exceeded: {}", value),
                        timestamp: Instant::now(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }
}
