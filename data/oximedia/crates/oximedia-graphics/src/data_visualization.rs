#![allow(dead_code)]
//! Data visualization for live broadcast charts and graphs.
//!
//! Provides production-quality chart rendering suitable for:
//! - Live election results, sports statistics, financial tickers
//! - Bar charts (horizontal and vertical), line charts, pie/donut charts
//! - Animated data transitions with interpolation
//! - Axis labeling, gridlines, legends
//! - Broadcast-safe color palettes

use crate::error::{GraphicsError, Result};
use std::f64::consts::PI;

/// Color for chart elements [R, G, B, A].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChartColor {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel.
    pub a: u8,
}

impl ChartColor {
    /// Create a new color.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Linear interpolation between two colors.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0) as f32;
        Self::new(
            (f32::from(self.r) + (f32::from(other.r) - f32::from(self.r)) * t) as u8,
            (f32::from(self.g) + (f32::from(other.g) - f32::from(self.g)) * t) as u8,
            (f32::from(self.b) + (f32::from(other.b) - f32::from(self.b)) * t) as u8,
            (f32::from(self.a) + (f32::from(other.a) - f32::from(self.a)) * t) as u8,
        )
    }
}

/// Broadcast-safe color palette (high contrast, accessible).
pub const PALETTE_BROADCAST: [ChartColor; 8] = [
    ChartColor::rgb(65, 105, 225),  // Royal blue
    ChartColor::rgb(220, 50, 50),   // Red
    ChartColor::rgb(50, 180, 50),   // Green
    ChartColor::rgb(255, 165, 0),   // Orange
    ChartColor::rgb(148, 103, 189), // Purple
    ChartColor::rgb(0, 190, 190),   // Teal
    ChartColor::rgb(255, 215, 0),   // Gold
    ChartColor::rgb(200, 200, 200), // Light gray
];

/// A single data point for charts.
#[derive(Clone, Debug)]
pub struct DataPoint {
    /// Label for this data point.
    pub label: String,
    /// Numeric value.
    pub value: f64,
    /// Optional color override.
    pub color: Option<ChartColor>,
}

impl DataPoint {
    /// Create a new data point.
    pub fn new(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
            color: None,
        }
    }

    /// Set a custom color.
    pub fn with_color(mut self, color: ChartColor) -> Self {
        self.color = Some(color);
        self
    }
}

/// A data series for line/bar charts.
#[derive(Clone, Debug)]
pub struct DataSeries {
    /// Series name.
    pub name: String,
    /// Data points.
    pub points: Vec<DataPoint>,
    /// Series color.
    pub color: ChartColor,
}

impl DataSeries {
    /// Create a new data series.
    pub fn new(name: impl Into<String>, color: ChartColor) -> Self {
        Self {
            name: name.into(),
            points: Vec::new(),
            color,
        }
    }

    /// Add a data point.
    pub fn add_point(&mut self, label: impl Into<String>, value: f64) {
        self.points.push(DataPoint::new(label, value));
    }

    /// Get the minimum value in the series.
    pub fn min_value(&self) -> Option<f64> {
        self.points.iter().map(|p| p.value).reduce(f64::min)
    }

    /// Get the maximum value in the series.
    pub fn max_value(&self) -> Option<f64> {
        self.points.iter().map(|p| p.value).reduce(f64::max)
    }

    /// Get the sum of all values.
    pub fn sum(&self) -> f64 {
        self.points.iter().map(|p| p.value).sum()
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Chart orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Vertical bars / standard layout.
    Vertical,
    /// Horizontal bars.
    Horizontal,
}

/// Axis configuration.
#[derive(Clone, Debug)]
pub struct AxisConfig {
    /// Whether to show this axis.
    pub visible: bool,
    /// Number of gridlines (0 = auto).
    pub gridlines: u32,
    /// Minimum value (None = auto from data).
    pub min_value: Option<f64>,
    /// Maximum value (None = auto from data).
    pub max_value: Option<f64>,
    /// Whether to show gridlines.
    pub show_gridlines: bool,
    /// Gridline color.
    pub gridline_color: ChartColor,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            visible: true,
            gridlines: 5,
            min_value: None,
            max_value: None,
            show_gridlines: true,
            gridline_color: ChartColor::new(80, 80, 80, 128),
        }
    }
}

/// Legend position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegendPosition {
    /// No legend displayed.
    None,
    /// Top of chart area.
    Top,
    /// Bottom of chart area.
    Bottom,
    /// Left of chart area.
    Left,
    /// Right of chart area.
    Right,
}

impl Default for LegendPosition {
    fn default() -> Self {
        Self::Bottom
    }
}

// ===========================================================================
// Bar chart
// ===========================================================================

/// Bar chart configuration and renderer.
#[derive(Clone, Debug)]
pub struct BarChart {
    /// Data series.
    pub series: Vec<DataSeries>,
    /// Chart width in pixels.
    pub width: u32,
    /// Chart height in pixels.
    pub height: u32,
    /// Bar orientation.
    pub orientation: Orientation,
    /// Spacing between bars as a fraction of bar width (0.0..1.0).
    pub bar_spacing: f64,
    /// Value axis configuration.
    pub value_axis: AxisConfig,
    /// Legend position.
    pub legend: LegendPosition,
    /// Background color.
    pub background: ChartColor,
    /// Chart padding in pixels [top, right, bottom, left].
    pub padding: [u32; 4],
}

impl BarChart {
    /// Create a new bar chart.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            series: Vec::new(),
            width,
            height,
            orientation: Orientation::Vertical,
            bar_spacing: 0.2,
            value_axis: AxisConfig::default(),
            legend: LegendPosition::default(),
            background: ChartColor::new(0, 0, 0, 0),
            padding: [20, 20, 30, 40],
        }
    }

    /// Add a data series.
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Set bar orientation.
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set bar spacing.
    pub fn with_bar_spacing(mut self, spacing: f64) -> Self {
        self.bar_spacing = spacing.clamp(0.0, 0.9);
        self
    }

    /// Set background color.
    pub fn with_background(mut self, color: ChartColor) -> Self {
        self.background = color;
        self
    }

    /// Compute the bars for rendering.
    ///
    /// Returns a list of `BarRect` describing each bar's position and color.
    pub fn compute_bars(&self) -> Result<Vec<BarRect>> {
        if self.series.is_empty() {
            return Ok(Vec::new());
        }

        let plot_x = self.padding[3];
        let plot_y = self.padding[0];
        let plot_w = self.width.saturating_sub(self.padding[1] + self.padding[3]);
        let plot_h = self
            .height
            .saturating_sub(self.padding[0] + self.padding[2]);

        if plot_w == 0 || plot_h == 0 {
            return Err(GraphicsError::InvalidDimensions(self.width, self.height));
        }

        // Find global min/max
        let data_min = self.value_axis.min_value.unwrap_or_else(|| {
            self.series
                .iter()
                .filter_map(|s| s.min_value())
                .fold(0.0_f64, f64::min)
        });
        let data_max = self.value_axis.max_value.unwrap_or_else(|| {
            self.series
                .iter()
                .filter_map(|s| s.max_value())
                .fold(1.0_f64, f64::max)
        });

        let value_range = if (data_max - data_min).abs() < f64::EPSILON {
            1.0
        } else {
            data_max - data_min
        };

        let max_points = self.series.iter().map(|s| s.len()).max().unwrap_or(0);
        if max_points == 0 {
            return Ok(Vec::new());
        }

        let num_series = self.series.len();
        let group_count = max_points;

        let mut bars = Vec::new();

        match self.orientation {
            Orientation::Vertical => {
                let group_width = plot_w as f64 / group_count as f64;
                let bar_area = group_width * (1.0 - self.bar_spacing);
                let bar_w = bar_area / num_series as f64;
                let spacing_offset = group_width * self.bar_spacing / 2.0;

                for (si, series) in self.series.iter().enumerate() {
                    for (pi, point) in series.points.iter().enumerate() {
                        let normalized = (point.value - data_min) / value_range;
                        let bar_h = normalized * plot_h as f64;

                        let x = plot_x as f64
                            + pi as f64 * group_width
                            + spacing_offset
                            + si as f64 * bar_w;
                        let y = plot_y as f64 + (plot_h as f64 - bar_h);

                        let color = point.color.unwrap_or(series.color);

                        bars.push(BarRect {
                            x: x as f32,
                            y: y as f32,
                            width: bar_w as f32,
                            height: bar_h as f32,
                            color,
                            label: point.label.clone(),
                            value: point.value,
                            series_index: si,
                        });
                    }
                }
            }
            Orientation::Horizontal => {
                let group_height = plot_h as f64 / group_count as f64;
                let bar_area = group_height * (1.0 - self.bar_spacing);
                let bar_h = bar_area / num_series as f64;
                let spacing_offset = group_height * self.bar_spacing / 2.0;

                for (si, series) in self.series.iter().enumerate() {
                    for (pi, point) in series.points.iter().enumerate() {
                        let normalized = (point.value - data_min) / value_range;
                        let bar_w = normalized * plot_w as f64;

                        let x = plot_x as f64;
                        let y = plot_y as f64
                            + pi as f64 * group_height
                            + spacing_offset
                            + si as f64 * bar_h;

                        let color = point.color.unwrap_or(series.color);

                        bars.push(BarRect {
                            x: x as f32,
                            y: y as f32,
                            width: bar_w as f32,
                            height: bar_h as f32,
                            color,
                            label: point.label.clone(),
                            value: point.value,
                            series_index: si,
                        });
                    }
                }
            }
        }

        Ok(bars)
    }

    /// Compute gridlines for the value axis.
    pub fn compute_gridlines(&self) -> Vec<GridLine> {
        if !self.value_axis.show_gridlines || self.value_axis.gridlines == 0 {
            return Vec::new();
        }

        let plot_x = self.padding[3] as f32;
        let plot_y = self.padding[0] as f32;
        let plot_w = self.width.saturating_sub(self.padding[1] + self.padding[3]) as f32;
        let plot_h = self
            .height
            .saturating_sub(self.padding[0] + self.padding[2]) as f32;

        let data_min = self.value_axis.min_value.unwrap_or(0.0);
        let data_max = self.value_axis.max_value.unwrap_or_else(|| {
            self.series
                .iter()
                .filter_map(|s| s.max_value())
                .fold(1.0_f64, f64::max)
        });

        let n = self.value_axis.gridlines;
        let mut lines = Vec::with_capacity(n as usize + 1);

        for i in 0..=n {
            let t = i as f64 / n as f64;
            let value = data_min + (data_max - data_min) * t;

            match self.orientation {
                Orientation::Vertical => {
                    let y = plot_y + plot_h * (1.0 - t as f32);
                    lines.push(GridLine {
                        x1: plot_x,
                        y1: y,
                        x2: plot_x + plot_w,
                        y2: y,
                        value,
                        color: self.value_axis.gridline_color,
                    });
                }
                Orientation::Horizontal => {
                    let x = plot_x + plot_w * t as f32;
                    lines.push(GridLine {
                        x1: x,
                        y1: plot_y,
                        x2: x,
                        y2: plot_y + plot_h,
                        value,
                        color: self.value_axis.gridline_color,
                    });
                }
            }
        }

        lines
    }
}

/// A computed bar rectangle ready for rendering.
#[derive(Clone, Debug)]
pub struct BarRect {
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
    /// Color.
    pub color: ChartColor,
    /// Label text.
    pub label: String,
    /// Data value.
    pub value: f64,
    /// Index of the series this bar belongs to.
    pub series_index: usize,
}

/// A gridline for rendering.
#[derive(Clone, Debug)]
pub struct GridLine {
    /// Start X.
    pub x1: f32,
    /// Start Y.
    pub y1: f32,
    /// End X.
    pub x2: f32,
    /// End Y.
    pub y2: f32,
    /// Value at this gridline.
    pub value: f64,
    /// Color.
    pub color: ChartColor,
}

// ===========================================================================
// Line chart
// ===========================================================================

/// Line chart configuration and renderer.
#[derive(Clone, Debug)]
pub struct LineChart {
    /// Data series.
    pub series: Vec<DataSeries>,
    /// Chart width in pixels.
    pub width: u32,
    /// Chart height in pixels.
    pub height: u32,
    /// Line thickness in pixels.
    pub line_width: f32,
    /// Whether to show data point markers.
    pub show_markers: bool,
    /// Marker radius in pixels.
    pub marker_radius: f32,
    /// Whether to fill area under the line.
    pub fill_area: bool,
    /// Fill opacity (0.0..1.0).
    pub fill_opacity: f32,
    /// Value axis configuration.
    pub value_axis: AxisConfig,
    /// Chart padding [top, right, bottom, left].
    pub padding: [u32; 4],
}

impl LineChart {
    /// Create a new line chart.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            series: Vec::new(),
            width,
            height,
            line_width: 2.0,
            show_markers: true,
            marker_radius: 4.0,
            fill_area: false,
            fill_opacity: 0.3,
            value_axis: AxisConfig::default(),
            padding: [20, 20, 30, 40],
        }
    }

    /// Add a data series.
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Set line width.
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width.max(0.5);
        self
    }

    /// Enable area fill.
    pub fn with_fill(mut self, opacity: f32) -> Self {
        self.fill_area = true;
        self.fill_opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Compute line segments for rendering.
    pub fn compute_lines(&self) -> Result<Vec<LineSegment>> {
        if self.series.is_empty() {
            return Ok(Vec::new());
        }

        let plot_x = self.padding[3] as f64;
        let plot_y = self.padding[0] as f64;
        let plot_w = self.width.saturating_sub(self.padding[1] + self.padding[3]) as f64;
        let plot_h = self
            .height
            .saturating_sub(self.padding[0] + self.padding[2]) as f64;

        if plot_w <= 0.0 || plot_h <= 0.0 {
            return Err(GraphicsError::InvalidDimensions(self.width, self.height));
        }

        let data_min = self.value_axis.min_value.unwrap_or_else(|| {
            self.series
                .iter()
                .filter_map(|s| s.min_value())
                .fold(0.0_f64, f64::min)
        });
        let data_max = self.value_axis.max_value.unwrap_or_else(|| {
            self.series
                .iter()
                .filter_map(|s| s.max_value())
                .fold(1.0_f64, f64::max)
        });
        let value_range = if (data_max - data_min).abs() < f64::EPSILON {
            1.0
        } else {
            data_max - data_min
        };

        let mut segments = Vec::new();

        for (si, series) in self.series.iter().enumerate() {
            if series.len() < 2 {
                continue;
            }

            let n = series.len();
            for i in 0..n - 1 {
                let t1 = i as f64 / (n - 1) as f64;
                let t2 = (i + 1) as f64 / (n - 1) as f64;

                let v1 = (series.points[i].value - data_min) / value_range;
                let v2 = (series.points[i + 1].value - data_min) / value_range;

                segments.push(LineSegment {
                    x1: (plot_x + t1 * plot_w) as f32,
                    y1: (plot_y + (1.0 - v1) * plot_h) as f32,
                    x2: (plot_x + t2 * plot_w) as f32,
                    y2: (plot_y + (1.0 - v2) * plot_h) as f32,
                    color: series.color,
                    width: self.line_width,
                    series_index: si,
                });
            }
        }

        Ok(segments)
    }

    /// Compute marker positions for rendering.
    pub fn compute_markers(&self) -> Result<Vec<MarkerPoint>> {
        if !self.show_markers || self.series.is_empty() {
            return Ok(Vec::new());
        }

        let plot_x = self.padding[3] as f64;
        let plot_y = self.padding[0] as f64;
        let plot_w = self.width.saturating_sub(self.padding[1] + self.padding[3]) as f64;
        let plot_h = self
            .height
            .saturating_sub(self.padding[0] + self.padding[2]) as f64;

        let data_min = self.value_axis.min_value.unwrap_or(0.0);
        let data_max = self.value_axis.max_value.unwrap_or(1.0);
        let value_range = if (data_max - data_min).abs() < f64::EPSILON {
            1.0
        } else {
            data_max - data_min
        };

        let mut markers = Vec::new();

        for (si, series) in self.series.iter().enumerate() {
            let n = series.len().max(1);
            for (pi, point) in series.points.iter().enumerate() {
                let t = if n > 1 {
                    pi as f64 / (n - 1) as f64
                } else {
                    0.5
                };
                let v = (point.value - data_min) / value_range;

                markers.push(MarkerPoint {
                    x: (plot_x + t * plot_w) as f32,
                    y: (plot_y + (1.0 - v) * plot_h) as f32,
                    radius: self.marker_radius,
                    color: point.color.unwrap_or(series.color),
                    label: point.label.clone(),
                    value: point.value,
                    series_index: si,
                });
            }
        }

        Ok(markers)
    }
}

/// A computed line segment for rendering.
#[derive(Clone, Debug)]
pub struct LineSegment {
    /// Start X.
    pub x1: f32,
    /// Start Y.
    pub y1: f32,
    /// End X.
    pub x2: f32,
    /// End Y.
    pub y2: f32,
    /// Color.
    pub color: ChartColor,
    /// Line width.
    pub width: f32,
    /// Series index.
    pub series_index: usize,
}

/// A computed marker point for rendering.
#[derive(Clone, Debug)]
pub struct MarkerPoint {
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Marker radius.
    pub radius: f32,
    /// Color.
    pub color: ChartColor,
    /// Label.
    pub label: String,
    /// Value.
    pub value: f64,
    /// Series index.
    pub series_index: usize,
}

// ===========================================================================
// Pie chart
// ===========================================================================

/// Pie/donut chart configuration and renderer.
#[derive(Clone, Debug)]
pub struct PieChart {
    /// Data points (each slice).
    pub slices: Vec<DataPoint>,
    /// Chart width.
    pub width: u32,
    /// Chart height.
    pub height: u32,
    /// Inner radius ratio (0.0 = pie, >0 = donut). Range: 0.0..1.0.
    pub inner_radius_ratio: f64,
    /// Start angle in degrees (0 = top).
    pub start_angle_deg: f64,
    /// Whether to sort slices by value (descending).
    pub sort_descending: bool,
    /// Gap between slices in degrees.
    pub slice_gap_deg: f64,
    /// Explode distance for highlighted slices (pixels).
    pub explode_distance: f32,
    /// Index of exploded slice (None = no explode).
    pub explode_index: Option<usize>,
}

impl PieChart {
    /// Create a new pie chart.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            slices: Vec::new(),
            width,
            height,
            inner_radius_ratio: 0.0,
            start_angle_deg: -90.0,
            sort_descending: false,
            slice_gap_deg: 1.0,
            explode_distance: 10.0,
            explode_index: None,
        }
    }

    /// Create a donut chart.
    pub fn donut(width: u32, height: u32, inner_ratio: f64) -> Self {
        Self {
            inner_radius_ratio: inner_ratio.clamp(0.0, 0.95),
            ..Self::new(width, height)
        }
    }

    /// Add a slice.
    pub fn add_slice(&mut self, label: impl Into<String>, value: f64) {
        self.slices.push(DataPoint::new(label, value));
    }

    /// Add a colored slice.
    pub fn add_colored_slice(&mut self, label: impl Into<String>, value: f64, color: ChartColor) {
        self.slices
            .push(DataPoint::new(label, value).with_color(color));
    }

    /// Set explode on a slice index.
    pub fn with_explode(mut self, index: usize) -> Self {
        self.explode_index = Some(index);
        self
    }

    /// Compute pie slices for rendering.
    pub fn compute_slices(&self) -> Result<Vec<PieSlice>> {
        if self.slices.is_empty() {
            return Ok(Vec::new());
        }

        let total: f64 = self.slices.iter().map(|s| s.value.max(0.0)).sum();
        if total <= 0.0 {
            return Err(GraphicsError::InvalidParameter(
                "Pie chart total value must be > 0".to_string(),
            ));
        }

        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let outer_r = (self.width.min(self.height) as f64 / 2.0) * 0.85;
        let inner_r = outer_r * self.inner_radius_ratio;

        let mut ordered: Vec<(usize, &DataPoint)> = self.slices.iter().enumerate().collect();
        if self.sort_descending {
            ordered.sort_by(|a, b| {
                b.1.value
                    .partial_cmp(&a.1.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let gap_rad = self.slice_gap_deg.to_radians();
        let total_gap = gap_rad * ordered.len() as f64;
        let available_angle = 2.0 * PI - total_gap;

        let start = self.start_angle_deg.to_radians();
        let mut angle = start;
        let mut result = Vec::with_capacity(ordered.len());

        for (idx, (original_idx, point)) in ordered.iter().enumerate() {
            let fraction = point.value.max(0.0) / total;
            let sweep = fraction * available_angle;

            let mid_angle = angle + sweep / 2.0;
            let (explode_dx, explode_dy) = if self.explode_index == Some(*original_idx) {
                (
                    mid_angle.cos() * self.explode_distance as f64,
                    mid_angle.sin() * self.explode_distance as f64,
                )
            } else {
                (0.0, 0.0)
            };

            let color = point
                .color
                .unwrap_or(PALETTE_BROADCAST[idx % PALETTE_BROADCAST.len()]);

            result.push(PieSlice {
                center_x: (cx + explode_dx) as f32,
                center_y: (cy + explode_dy) as f32,
                outer_radius: outer_r as f32,
                inner_radius: inner_r as f32,
                start_angle: angle as f32,
                end_angle: (angle + sweep) as f32,
                color,
                label: point.label.clone(),
                value: point.value,
                percentage: fraction * 100.0,
                original_index: *original_idx,
            });

            angle += sweep + gap_rad;
        }

        Ok(result)
    }

    /// Compute the total value.
    pub fn total(&self) -> f64 {
        self.slices.iter().map(|s| s.value.max(0.0)).sum()
    }
}

/// A computed pie slice for rendering.
#[derive(Clone, Debug)]
pub struct PieSlice {
    /// Center X (may be offset for exploded slices).
    pub center_x: f32,
    /// Center Y.
    pub center_y: f32,
    /// Outer radius.
    pub outer_radius: f32,
    /// Inner radius (0 for pie, >0 for donut).
    pub inner_radius: f32,
    /// Start angle in radians.
    pub start_angle: f32,
    /// End angle in radians.
    pub end_angle: f32,
    /// Slice color.
    pub color: ChartColor,
    /// Label.
    pub label: String,
    /// Data value.
    pub value: f64,
    /// Percentage of total.
    pub percentage: f64,
    /// Original index in the data.
    pub original_index: usize,
}

impl PieSlice {
    /// Sweep angle of this slice.
    pub fn sweep_angle(&self) -> f32 {
        self.end_angle - self.start_angle
    }

    /// Mid-angle of this slice (useful for label placement).
    pub fn mid_angle(&self) -> f32 {
        (self.start_angle + self.end_angle) / 2.0
    }

    /// Point at a given radius along the mid-angle (for label placement).
    pub fn label_point(&self, radius_factor: f32) -> (f32, f32) {
        let r = self.outer_radius * radius_factor;
        let a = self.mid_angle();
        (self.center_x + r * a.cos(), self.center_y + r * a.sin())
    }
}

// ===========================================================================
// Data transition (for animated charts)
// ===========================================================================

/// Interpolator for animating between two data snapshots.
#[derive(Clone, Debug)]
pub struct DataTransition {
    /// Previous values.
    from: Vec<f64>,
    /// Target values.
    to: Vec<f64>,
    /// Current progress (0.0..=1.0).
    progress: f64,
}

impl DataTransition {
    /// Create a new transition from one data set to another.
    pub fn new(from: Vec<f64>, to: Vec<f64>) -> Self {
        Self {
            from,
            to,
            progress: 0.0,
        }
    }

    /// Set the transition progress.
    pub fn set_progress(&mut self, t: f64) {
        self.progress = t.clamp(0.0, 1.0);
    }

    /// Get the current interpolated values.
    pub fn current_values(&self) -> Vec<f64> {
        let len = self.from.len().max(self.to.len());
        let t = self.progress;
        (0..len)
            .map(|i| {
                let a = self.from.get(i).copied().unwrap_or(0.0);
                let b = self.to.get(i).copied().unwrap_or(0.0);
                a + (b - a) * t
            })
            .collect()
    }

    /// Whether the transition is complete.
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_color_lerp() {
        let a = ChartColor::rgb(0, 0, 0);
        let b = ChartColor::rgb(255, 255, 255);
        let mid = a.lerp(&b, 0.5);
        assert!(mid.r > 120 && mid.r < 135);
    }

    #[test]
    fn test_data_point_creation() {
        let dp = DataPoint::new("Q1", 100.0);
        assert_eq!(dp.label, "Q1");
        assert!((dp.value - 100.0).abs() < f64::EPSILON);
        assert!(dp.color.is_none());
    }

    #[test]
    fn test_data_point_with_color() {
        let dp = DataPoint::new("Q1", 50.0).with_color(ChartColor::rgb(255, 0, 0));
        assert!(dp.color.is_some());
    }

    #[test]
    fn test_data_series() {
        let mut series = DataSeries::new("Sales", ChartColor::rgb(65, 105, 225));
        series.add_point("Jan", 100.0);
        series.add_point("Feb", 150.0);
        series.add_point("Mar", 120.0);
        assert_eq!(series.len(), 3);
        assert!(!series.is_empty());
        assert_eq!(series.min_value(), Some(100.0));
        assert_eq!(series.max_value(), Some(150.0));
        assert!((series.sum() - 370.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_data_series_empty() {
        let series = DataSeries::new("Empty", ChartColor::rgb(0, 0, 0));
        assert!(series.is_empty());
        assert_eq!(series.min_value(), None);
        assert_eq!(series.max_value(), None);
    }

    #[test]
    fn test_bar_chart_empty() {
        let chart = BarChart::new(400, 300);
        let bars = chart.compute_bars().expect("should succeed");
        assert!(bars.is_empty());
    }

    #[test]
    fn test_bar_chart_single_series() {
        let mut chart = BarChart::new(400, 300);
        let mut series = DataSeries::new("Revenue", PALETTE_BROADCAST[0]);
        series.add_point("Q1", 100.0);
        series.add_point("Q2", 200.0);
        series.add_point("Q3", 150.0);
        chart.add_series(series);

        let bars = chart.compute_bars().expect("should succeed");
        assert_eq!(bars.len(), 3);

        // Bars should have positive dimensions
        for bar in &bars {
            assert!(bar.width > 0.0);
            assert!(bar.height >= 0.0);
        }
    }

    #[test]
    fn test_bar_chart_multi_series() {
        let mut chart = BarChart::new(600, 400);
        let mut s1 = DataSeries::new("2023", PALETTE_BROADCAST[0]);
        s1.add_point("Q1", 100.0);
        s1.add_point("Q2", 200.0);

        let mut s2 = DataSeries::new("2024", PALETTE_BROADCAST[1]);
        s2.add_point("Q1", 120.0);
        s2.add_point("Q2", 180.0);

        chart.add_series(s1);
        chart.add_series(s2);

        let bars = chart.compute_bars().expect("should succeed");
        assert_eq!(bars.len(), 4); // 2 series * 2 points
    }

    #[test]
    fn test_bar_chart_horizontal() {
        let mut chart = BarChart::new(400, 300).with_orientation(Orientation::Horizontal);
        let mut series = DataSeries::new("Votes", PALETTE_BROADCAST[0]);
        series.add_point("A", 1000.0);
        series.add_point("B", 800.0);
        chart.add_series(series);

        let bars = chart.compute_bars().expect("should succeed");
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn test_bar_chart_gridlines() {
        let mut chart = BarChart::new(400, 300);
        chart.value_axis.max_value = Some(100.0);
        chart.value_axis.gridlines = 4;

        let lines = chart.compute_gridlines();
        assert_eq!(lines.len(), 5); // 4 + 1 (includes 0)
    }

    #[test]
    fn test_line_chart_empty() {
        let chart = LineChart::new(400, 300);
        let lines = chart.compute_lines().expect("should succeed");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_line_chart_segments() {
        let mut chart = LineChart::new(400, 300);
        let mut series = DataSeries::new("Temperature", PALETTE_BROADCAST[0]);
        series.add_point("Mon", 20.0);
        series.add_point("Tue", 22.0);
        series.add_point("Wed", 19.0);
        series.add_point("Thu", 25.0);
        chart.add_series(series);

        let segments = chart.compute_lines().expect("should succeed");
        assert_eq!(segments.len(), 3); // 4 points = 3 segments
    }

    #[test]
    fn test_line_chart_markers() {
        let mut chart = LineChart::new(400, 300);
        chart.value_axis.min_value = Some(0.0);
        chart.value_axis.max_value = Some(100.0);

        let mut series = DataSeries::new("Score", PALETTE_BROADCAST[0]);
        series.add_point("A", 50.0);
        series.add_point("B", 75.0);
        chart.add_series(series);

        let markers = chart.compute_markers().expect("should succeed");
        assert_eq!(markers.len(), 2);
    }

    #[test]
    fn test_line_chart_no_markers() {
        let mut chart = LineChart::new(400, 300);
        chart.show_markers = false;
        let mut series = DataSeries::new("X", PALETTE_BROADCAST[0]);
        series.add_point("A", 1.0);
        chart.add_series(series);

        let markers = chart.compute_markers().expect("should succeed");
        assert!(markers.is_empty());
    }

    #[test]
    fn test_pie_chart_basic() {
        let mut pie = PieChart::new(400, 400);
        pie.add_slice("A", 50.0);
        pie.add_slice("B", 30.0);
        pie.add_slice("C", 20.0);

        let slices = pie.compute_slices().expect("should succeed");
        assert_eq!(slices.len(), 3);

        // Percentages should sum to ~100
        let total_pct: f64 = slices.iter().map(|s| s.percentage).sum();
        assert!((total_pct - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_pie_chart_donut() {
        let mut pie = PieChart::donut(400, 400, 0.5);
        pie.add_slice("X", 60.0);
        pie.add_slice("Y", 40.0);

        let slices = pie.compute_slices().expect("should succeed");
        assert_eq!(slices.len(), 2);
        assert!(slices[0].inner_radius > 0.0);
    }

    #[test]
    fn test_pie_chart_explode() {
        let mut pie = PieChart::new(400, 400).with_explode(0);
        pie.add_slice("Big", 70.0);
        pie.add_slice("Small", 30.0);

        let slices = pie.compute_slices().expect("should succeed");
        // Exploded slice should have offset center
        let normal_cx = slices[1].center_x;
        let exploded_cx = slices[0].center_x;
        // They should differ (exploded moves outward)
        assert!((normal_cx - exploded_cx).abs() > 0.1 || slices[0].center_y != slices[1].center_y);
    }

    #[test]
    fn test_pie_chart_empty() {
        let pie = PieChart::new(400, 400);
        let slices = pie.compute_slices().expect("should succeed");
        assert!(slices.is_empty());
    }

    #[test]
    fn test_pie_chart_zero_total() {
        let mut pie = PieChart::new(400, 400);
        pie.add_slice("Zero", 0.0);
        assert!(pie.compute_slices().is_err());
    }

    #[test]
    fn test_pie_slice_geometry() {
        let mut pie = PieChart::new(400, 400);
        pie.add_slice("Half", 50.0);
        pie.add_slice("Half", 50.0);

        let slices = pie.compute_slices().expect("should succeed");
        for s in &slices {
            assert!(s.sweep_angle() > 0.0);
        }
    }

    #[test]
    fn test_pie_slice_label_point() {
        let slice = PieSlice {
            center_x: 200.0,
            center_y: 200.0,
            outer_radius: 150.0,
            inner_radius: 0.0,
            start_angle: 0.0,
            end_angle: std::f32::consts::PI,
            color: ChartColor::rgb(255, 0, 0),
            label: "Test".to_string(),
            value: 50.0,
            percentage: 50.0,
            original_index: 0,
        };
        let (lx, ly) = slice.label_point(0.7);
        // Label should be outside center
        let dist = ((lx - 200.0).powi(2) + (ly - 200.0).powi(2)).sqrt();
        assert!(dist > 50.0);
    }

    #[test]
    fn test_data_transition() {
        let mut t = DataTransition::new(vec![0.0, 10.0, 20.0], vec![100.0, 50.0, 80.0]);

        t.set_progress(0.0);
        let vals = t.current_values();
        assert!((vals[0]).abs() < 0.01);

        t.set_progress(0.5);
        let vals = t.current_values();
        assert!((vals[0] - 50.0).abs() < 0.01);
        assert!((vals[1] - 30.0).abs() < 0.01);

        t.set_progress(1.0);
        assert!(t.is_complete());
        let vals = t.current_values();
        assert!((vals[0] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_data_transition_different_lengths() {
        let mut t = DataTransition::new(vec![10.0], vec![20.0, 30.0]);
        t.set_progress(0.5);
        let vals = t.current_values();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 15.0).abs() < 0.01);
        assert!((vals[1] - 15.0).abs() < 0.01); // 0 -> 30 at 0.5
    }

    #[test]
    fn test_palette_has_8_colors() {
        assert_eq!(PALETTE_BROADCAST.len(), 8);
    }

    #[test]
    fn test_bar_chart_builder() {
        let chart = BarChart::new(800, 600)
            .with_orientation(Orientation::Horizontal)
            .with_bar_spacing(0.3)
            .with_background(ChartColor::rgb(30, 30, 30));
        assert_eq!(chart.orientation, Orientation::Horizontal);
        assert!((chart.bar_spacing - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_line_chart_builder() {
        let chart = LineChart::new(800, 600).with_line_width(3.0).with_fill(0.5);
        assert!((chart.line_width - 3.0).abs() < f32::EPSILON);
        assert!(chart.fill_area);
        assert!((chart.fill_opacity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pie_chart_total() {
        let mut pie = PieChart::new(400, 400);
        pie.add_slice("A", 30.0);
        pie.add_slice("B", 70.0);
        assert!((pie.total() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_colored_slices() {
        let mut pie = PieChart::new(400, 400);
        pie.add_colored_slice("Custom", 100.0, ChartColor::rgb(255, 128, 0));
        let slices = pie.compute_slices().expect("should succeed");
        assert_eq!(slices[0].color, ChartColor::rgb(255, 128, 0));
    }

    #[test]
    fn test_axis_config_defaults() {
        let axis = AxisConfig::default();
        assert!(axis.visible);
        assert!(axis.show_gridlines);
        assert_eq!(axis.gridlines, 5);
    }

    #[test]
    fn test_bar_chart_values_in_range() {
        let mut chart = BarChart::new(400, 300);
        let mut series = DataSeries::new("Test", PALETTE_BROADCAST[0]);
        series.add_point("A", 10.0);
        series.add_point("B", 90.0);
        chart.add_series(series);

        let bars = chart.compute_bars().expect("should succeed");
        for bar in &bars {
            assert!(bar.x >= 0.0);
            assert!(bar.y >= 0.0);
            assert!(bar.x + bar.width <= chart.width as f32 + 1.0);
            assert!(bar.y + bar.height <= chart.height as f32 + 1.0);
        }
    }
}
