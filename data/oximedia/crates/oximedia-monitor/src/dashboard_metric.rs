//! Dashboard metric widgets and panel composition for `OxiMedia` monitoring.
//!
//! Provides building blocks for assembling real-time monitoring dashboards
//! from typed metric widgets grouped into named panels.

#![allow(dead_code)]

/// The kind of value a metric widget displays.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricWidget {
    /// A single numeric gauge (e.g. CPU %).
    Gauge {
        /// Widget label.
        label: String,
        /// Current value.
        value: f64,
        /// Optional unit string (e.g. "%", "MB/s").
        unit: Option<String>,
        /// Threshold above which the value is considered critical.
        critical_threshold: Option<f64>,
    },
    /// A sparkline showing recent history.
    Sparkline {
        /// Widget label.
        label: String,
        /// Ordered series of recent values.
        history: Vec<f64>,
        /// Maximum expected value for scaling.
        max_value: f64,
    },
    /// A text status badge.
    StatusBadge {
        /// Widget label.
        label: String,
        /// Current status text.
        status: String,
        /// `true` when the status indicates healthy.
        healthy: bool,
    },
    /// A percentage bar (0–100).
    ProgressBar {
        /// Widget label.
        label: String,
        /// Current percentage in `[0.0, 100.0]`.
        percent: f64,
        /// Threshold at which the bar turns orange.
        warn_threshold: f64,
    },
    /// A count of discrete events.
    Counter {
        /// Widget label.
        label: String,
        /// Accumulated count.
        count: u64,
        /// Optional rate per second.
        rate_per_sec: Option<f64>,
    },
}

impl MetricWidget {
    /// Returns the human-readable label of this widget.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Gauge { label, .. }
            | Self::Sparkline { label, .. }
            | Self::StatusBadge { label, .. }
            | Self::ProgressBar { label, .. }
            | Self::Counter { label, .. } => label,
        }
    }

    /// Returns `true` when the widget currently represents a critical state.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        match self {
            Self::Gauge {
                value,
                critical_threshold,
                ..
            } => critical_threshold.is_some_and(|t| *value >= t),
            Self::ProgressBar { percent, .. } => *percent >= 100.0,
            Self::StatusBadge { healthy, .. } => !healthy,
            _ => false,
        }
    }
}

/// A named metric value with a widget type hint.
#[derive(Debug, Clone)]
pub struct DashboardMetric {
    /// Unique identifier within the dashboard.
    pub id: String,
    /// The widget that renders this metric.
    pub widget: MetricWidget,
    /// Display order within the parent panel (lower = first).
    pub order: u32,
}

impl DashboardMetric {
    /// Create a new `DashboardMetric`.
    #[must_use]
    pub fn new(id: impl Into<String>, widget: MetricWidget, order: u32) -> Self {
        Self {
            id: id.into(),
            widget,
            order,
        }
    }

    /// Returns `true` when this metric is in a critical state.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.widget.is_critical()
    }
}

/// A named group of dashboard metrics that forms a single panel.
#[derive(Debug, Clone, Default)]
pub struct DashboardPanel {
    /// Panel title shown to operators.
    pub title: String,
    /// Metrics contained in this panel.
    widgets: Vec<DashboardMetric>,
}

impl DashboardPanel {
    /// Create an empty panel with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            widgets: Vec::new(),
        }
    }

    /// Append a widget to the panel.
    pub fn add_widget(&mut self, metric: DashboardMetric) {
        self.widgets.push(metric);
        self.widgets.sort_by_key(|m| m.order);
    }

    /// Returns all widgets in display order.
    #[must_use]
    pub fn widgets(&self) -> &[DashboardMetric] {
        &self.widgets
    }

    /// Returns the number of widgets in the panel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.widgets.len()
    }

    /// Returns `true` when the panel contains no widgets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.widgets.is_empty()
    }

    /// Returns the number of widgets that are currently critical.
    #[must_use]
    pub fn critical_count(&self) -> usize {
        self.widgets.iter().filter(|m| m.is_critical()).count()
    }

    /// Remove a widget by its identifier. Returns `true` if removed.
    pub fn remove_widget(&mut self, id: &str) -> bool {
        let before = self.widgets.len();
        self.widgets.retain(|m| m.id != id);
        self.widgets.len() < before
    }

    /// Find a widget by identifier.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&DashboardMetric> {
        self.widgets.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauge(label: &str, value: f64, threshold: Option<f64>) -> MetricWidget {
        MetricWidget::Gauge {
            label: label.to_string(),
            value,
            unit: Some("%".to_string()),
            critical_threshold: threshold,
        }
    }

    fn badge(label: &str, status: &str, healthy: bool) -> MetricWidget {
        MetricWidget::StatusBadge {
            label: label.to_string(),
            status: status.to_string(),
            healthy,
        }
    }

    #[test]
    fn test_gauge_label() {
        let w = gauge("CPU", 42.0, None);
        assert_eq!(w.label(), "CPU");
    }

    #[test]
    fn test_gauge_not_critical_below_threshold() {
        let w = gauge("CPU", 50.0, Some(90.0));
        assert!(!w.is_critical());
    }

    #[test]
    fn test_gauge_critical_at_threshold() {
        let w = gauge("CPU", 90.0, Some(90.0));
        assert!(w.is_critical());
    }

    #[test]
    fn test_gauge_no_threshold_not_critical() {
        let w = gauge("CPU", 99.9, None);
        assert!(!w.is_critical());
    }

    #[test]
    fn test_badge_healthy_not_critical() {
        let w = badge("Service", "OK", true);
        assert!(!w.is_critical());
    }

    #[test]
    fn test_badge_unhealthy_is_critical() {
        let w = badge("Service", "DOWN", false);
        assert!(w.is_critical());
    }

    #[test]
    fn test_progress_bar_at_100_is_critical() {
        let w = MetricWidget::ProgressBar {
            label: "Queue".to_string(),
            percent: 100.0,
            warn_threshold: 80.0,
        };
        assert!(w.is_critical());
    }

    #[test]
    fn test_counter_not_critical() {
        let w = MetricWidget::Counter {
            label: "Frames".to_string(),
            count: 1_000_000,
            rate_per_sec: Some(30.0),
        };
        assert!(!w.is_critical());
    }

    #[test]
    fn test_panel_add_widget_ordering() {
        let mut panel = DashboardPanel::new("System");
        panel.add_widget(DashboardMetric::new("b", gauge("B", 1.0, None), 2));
        panel.add_widget(DashboardMetric::new("a", gauge("A", 1.0, None), 1));
        assert_eq!(panel.widgets()[0].id, "a");
        assert_eq!(panel.widgets()[1].id, "b");
    }

    #[test]
    fn test_panel_len_and_is_empty() {
        let mut panel = DashboardPanel::new("Test");
        assert!(panel.is_empty());
        panel.add_widget(DashboardMetric::new("x", gauge("X", 0.0, None), 0));
        assert_eq!(panel.len(), 1);
        assert!(!panel.is_empty());
    }

    #[test]
    fn test_panel_critical_count() {
        let mut panel = DashboardPanel::new("Alerts");
        panel.add_widget(DashboardMetric::new("ok", gauge("OK", 10.0, Some(80.0)), 0));
        panel.add_widget(DashboardMetric::new(
            "crit",
            gauge("CRIT", 90.0, Some(80.0)),
            1,
        ));
        assert_eq!(panel.critical_count(), 1);
    }

    #[test]
    fn test_panel_remove_widget() {
        let mut panel = DashboardPanel::new("Test");
        panel.add_widget(DashboardMetric::new("one", gauge("One", 1.0, None), 0));
        panel.add_widget(DashboardMetric::new("two", gauge("Two", 2.0, None), 1));
        assert!(panel.remove_widget("one"));
        assert_eq!(panel.len(), 1);
        assert_eq!(panel.widgets()[0].id, "two");
    }

    #[test]
    fn test_panel_remove_nonexistent() {
        let mut panel = DashboardPanel::new("Test");
        assert!(!panel.remove_widget("missing"));
    }

    #[test]
    fn test_panel_find() {
        let mut panel = DashboardPanel::new("Test");
        panel.add_widget(DashboardMetric::new("m1", gauge("M1", 5.0, None), 0));
        assert!(panel.find("m1").is_some());
        assert!(panel.find("missing").is_none());
    }

    #[test]
    fn test_dashboard_metric_is_critical_delegates() {
        let metric = DashboardMetric::new("crit", gauge("CPU", 95.0, Some(90.0)), 0);
        assert!(metric.is_critical());
    }

    #[test]
    fn test_sparkline_not_critical() {
        let w = MetricWidget::Sparkline {
            label: "Net".to_string(),
            history: vec![1.0, 2.0, 3.0],
            max_value: 100.0,
        };
        assert!(!w.is_critical());
        assert_eq!(w.label(), "Net");
    }
}
