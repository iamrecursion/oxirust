//! Dashboard panel and widget model for building monitoring views.
#![allow(dead_code)]

/// A widget that can appear in a monitoring dashboard panel.
#[derive(Clone, Debug, PartialEq)]
pub enum DashboardWidget {
    /// A numeric gauge showing a single value with optional label.
    Gauge {
        /// Label shown above the gauge.
        label: String,
        /// Current numeric value.
        value: f64,
        /// Optional unit string (e.g. "%", "fps").
        unit: Option<String>,
    },
    /// A time-series line graph for a named metric.
    LineGraph {
        /// Title of the graph.
        title: String,
        /// Metric key whose history is plotted.
        metric_key: String,
    },
    /// A simple text status readout.
    StatusText {
        /// Label for the status entry.
        label: String,
        /// Status string (e.g. "OK", "DEGRADED").
        status: String,
    },
    /// An alert summary box showing the count of active alerts.
    AlertSummary {
        /// Number of active (firing) alerts.
        active_count: usize,
    },
    /// A heat-map tile grid.
    HeatMap {
        /// Title for the heat map.
        title: String,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
}

impl DashboardWidget {
    /// Return a human-readable name for the widget type.
    #[must_use]
    pub fn widget_name(&self) -> &'static str {
        match self {
            Self::Gauge { .. } => "Gauge",
            Self::LineGraph { .. } => "LineGraph",
            Self::StatusText { .. } => "StatusText",
            Self::AlertSummary { .. } => "AlertSummary",
            Self::HeatMap { .. } => "HeatMap",
        }
    }

    /// `true` if this is a gauge widget.
    #[must_use]
    pub fn is_gauge(&self) -> bool {
        matches!(self, Self::Gauge { .. })
    }

    /// `true` if this is an alert-summary widget.
    #[must_use]
    pub fn is_alert_summary(&self) -> bool {
        matches!(self, Self::AlertSummary { .. })
    }
}

/// A rectangular panel that contains one or more widgets.
#[derive(Clone, Debug)]
pub struct DashboardPanel {
    /// Panel title shown in the UI.
    pub title: String,
    widgets: Vec<DashboardWidget>,
}

impl DashboardPanel {
    /// Create a new, empty panel with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            widgets: Vec::new(),
        }
    }

    /// Add a widget to this panel.
    pub fn add_widget(&mut self, widget: DashboardWidget) {
        self.widgets.push(widget);
    }

    /// Number of widgets currently in this panel.
    #[must_use]
    pub fn widget_count(&self) -> usize {
        self.widgets.len()
    }

    /// Immutable slice of all widgets in this panel.
    #[must_use]
    pub fn widgets(&self) -> &[DashboardWidget] {
        &self.widgets
    }

    /// Remove all widgets from the panel.
    pub fn clear_widgets(&mut self) {
        self.widgets.clear();
    }

    /// `true` when the panel has no widgets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.widgets.is_empty()
    }
}

/// A top-level dashboard composed of named panels.
#[derive(Debug, Default)]
pub struct Dashboard {
    /// Dashboard name.
    pub name: String,
    panels: Vec<DashboardPanel>,
}

impl Dashboard {
    /// Create a new dashboard with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            panels: Vec::new(),
        }
    }

    /// Add a panel to this dashboard.
    pub fn add_panel(&mut self, panel: DashboardPanel) {
        self.panels.push(panel);
    }

    /// Number of panels in this dashboard.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Immutable slice of all panels.
    #[must_use]
    pub fn panels(&self) -> &[DashboardPanel] {
        &self.panels
    }

    /// Total number of widgets across all panels.
    #[must_use]
    pub fn total_widget_count(&self) -> usize {
        self.panels.iter().map(DashboardPanel::widget_count).sum()
    }

    /// Remove all panels from the dashboard.
    pub fn clear(&mut self) {
        self.panels.clear();
    }

    /// `true` if the dashboard has no panels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }

    /// Find the first panel with the given title.
    #[must_use]
    pub fn find_panel(&self, title: &str) -> Option<&DashboardPanel> {
        self.panels.iter().find(|p| p.title == title)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DashboardWidget ──────────────────────────────────────────────────────

    #[test]
    fn widget_name_gauge() {
        let w = DashboardWidget::Gauge {
            label: "CPU".into(),
            value: 42.0,
            unit: Some("%".into()),
        };
        assert_eq!(w.widget_name(), "Gauge");
    }

    #[test]
    fn widget_name_line_graph() {
        let w = DashboardWidget::LineGraph {
            title: "FPS".into(),
            metric_key: "encode_fps".into(),
        };
        assert_eq!(w.widget_name(), "LineGraph");
    }

    #[test]
    fn widget_name_status_text() {
        let w = DashboardWidget::StatusText {
            label: "System".into(),
            status: "OK".into(),
        };
        assert_eq!(w.widget_name(), "StatusText");
    }

    #[test]
    fn widget_name_alert_summary() {
        let w = DashboardWidget::AlertSummary { active_count: 3 };
        assert_eq!(w.widget_name(), "AlertSummary");
    }

    #[test]
    fn widget_name_heat_map() {
        let w = DashboardWidget::HeatMap {
            title: "Grid".into(),
            rows: 4,
            cols: 4,
        };
        assert_eq!(w.widget_name(), "HeatMap");
    }

    #[test]
    fn widget_is_gauge_true_for_gauge() {
        let w = DashboardWidget::Gauge {
            label: "X".into(),
            value: 1.0,
            unit: None,
        };
        assert!(w.is_gauge());
    }

    #[test]
    fn widget_is_gauge_false_for_others() {
        let w = DashboardWidget::AlertSummary { active_count: 0 };
        assert!(!w.is_gauge());
    }

    #[test]
    fn widget_is_alert_summary() {
        let w = DashboardWidget::AlertSummary { active_count: 5 };
        assert!(w.is_alert_summary());
    }

    // ── DashboardPanel ───────────────────────────────────────────────────────

    #[test]
    fn panel_starts_empty() {
        let p = DashboardPanel::new("Overview");
        assert!(p.is_empty());
        assert_eq!(p.widget_count(), 0);
    }

    #[test]
    fn panel_add_widget_increments_count() {
        let mut p = DashboardPanel::new("Stats");
        p.add_widget(DashboardWidget::AlertSummary { active_count: 0 });
        p.add_widget(DashboardWidget::AlertSummary { active_count: 1 });
        assert_eq!(p.widget_count(), 2);
    }

    #[test]
    fn panel_clear_widgets_empties_panel() {
        let mut p = DashboardPanel::new("Temp");
        p.add_widget(DashboardWidget::AlertSummary { active_count: 0 });
        p.clear_widgets();
        assert!(p.is_empty());
    }

    #[test]
    fn panel_widgets_slice_matches() {
        let mut p = DashboardPanel::new("P");
        let w = DashboardWidget::AlertSummary { active_count: 2 };
        p.add_widget(w.clone());
        assert_eq!(p.widgets()[0], w);
    }

    // ── Dashboard ────────────────────────────────────────────────────────────

    #[test]
    fn dashboard_starts_empty() {
        let d = Dashboard::new("Main");
        assert!(d.is_empty());
        assert_eq!(d.panel_count(), 0);
    }

    #[test]
    fn dashboard_add_panel() {
        let mut d = Dashboard::new("Main");
        d.add_panel(DashboardPanel::new("Overview"));
        assert_eq!(d.panel_count(), 1);
    }

    #[test]
    fn dashboard_total_widget_count() {
        let mut d = Dashboard::new("D");
        let mut p1 = DashboardPanel::new("P1");
        p1.add_widget(DashboardWidget::AlertSummary { active_count: 0 });
        p1.add_widget(DashboardWidget::AlertSummary { active_count: 1 });
        let mut p2 = DashboardPanel::new("P2");
        p2.add_widget(DashboardWidget::AlertSummary { active_count: 2 });
        d.add_panel(p1);
        d.add_panel(p2);
        assert_eq!(d.total_widget_count(), 3);
    }

    #[test]
    fn dashboard_find_panel_by_title() {
        let mut d = Dashboard::new("D");
        d.add_panel(DashboardPanel::new("Encoding"));
        d.add_panel(DashboardPanel::new("Storage"));
        assert!(d.find_panel("Encoding").is_some());
        assert!(d.find_panel("Missing").is_none());
    }

    #[test]
    fn dashboard_clear_removes_all_panels() {
        let mut d = Dashboard::new("D");
        d.add_panel(DashboardPanel::new("P"));
        d.clear();
        assert!(d.is_empty());
    }
}
