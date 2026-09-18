//! Performance heads-up display (HUD) for live game streaming.
//!
//! Collects and renders real-time performance metrics (FPS, frame-time,
//! CPU / GPU utilisation, bitrate, dropped frames) as a lightweight
//! text-based overlay that can be composited onto the output stream.
//! Includes a frame-time graph renderer and usage bar displays.

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Metric sample
// ---------------------------------------------------------------------------

/// A single timestamped performance sample.
#[derive(Debug, Clone, Copy)]
pub struct PerfSample {
    /// Frames per second at the moment of capture.
    pub fps: f32,
    /// Frame time in milliseconds.
    pub frame_time_ms: f32,
    /// CPU utilisation as a fraction in `[0.0, 1.0]`.
    pub cpu_usage: f32,
    /// GPU utilisation as a fraction in `[0.0, 1.0]`.
    pub gpu_usage: f32,
    /// Encoding bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Number of frames dropped since the last sample.
    pub dropped_frames: u32,
    /// VRAM usage in megabytes (0 if not available).
    pub vram_mb: u32,
    /// RAM usage in megabytes (0 if not available).
    pub ram_mb: u32,
}

impl Default for PerfSample {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            cpu_usage: 0.0,
            gpu_usage: 0.0,
            bitrate_kbps: 0,
            dropped_frames: 0,
            vram_mb: 0,
            ram_mb: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// HUD position / colour
// ---------------------------------------------------------------------------

/// Screen corner where the HUD is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
}

/// Simple RGBA colour value for the HUD text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudColor {
    /// Red channel 0-255.
    pub r: u8,
    /// Green channel 0-255.
    pub g: u8,
    /// Blue channel 0-255.
    pub b: u8,
    /// Alpha channel 0-255.
    pub a: u8,
}

impl HudColor {
    /// Create a new opaque colour.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a colour with custom alpha.
    #[must_use]
    pub const fn with_alpha(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Pre-defined green colour for "good" status.
    #[must_use]
    pub const fn green() -> Self {
        Self::new(0, 255, 0)
    }

    /// Pre-defined yellow colour for "warning" status.
    #[must_use]
    pub const fn yellow() -> Self {
        Self::new(255, 255, 0)
    }

    /// Pre-defined red colour for "critical" status.
    #[must_use]
    pub const fn red() -> Self {
        Self::new(255, 0, 0)
    }

    /// Convert to `[u8; 4]` RGBA array.
    #[must_use]
    pub const fn to_rgba(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

// ---------------------------------------------------------------------------
// HUD configuration
// ---------------------------------------------------------------------------

/// Configuration for the performance HUD overlay.
#[derive(Debug, Clone)]
pub struct PerfHudConfig {
    /// Where to place the HUD.
    pub position: HudPosition,
    /// Text colour.
    pub text_color: HudColor,
    /// Background colour (may be semi-transparent).
    pub bg_color: HudColor,
    /// Font size in points.
    pub font_size: u32,
    /// How many historical samples to keep for averaging.
    pub history_size: usize,
    /// Show FPS counter.
    pub show_fps: bool,
    /// Show frame-time graph.
    pub show_frame_time: bool,
    /// Show CPU usage.
    pub show_cpu: bool,
    /// Show GPU usage.
    pub show_gpu: bool,
    /// Show bitrate.
    pub show_bitrate: bool,
    /// Show dropped frames.
    pub show_dropped: bool,
    /// Show frame-time sparkline graph.
    pub show_frame_time_graph: bool,
    /// Show memory usage (RAM/VRAM).
    pub show_memory: bool,
    /// Update interval.
    pub update_interval: Duration,
    /// Graph width in samples (for the frame-time sparkline).
    pub graph_width: usize,
    /// Graph height in text lines.
    pub graph_height: usize,
}

impl Default for PerfHudConfig {
    fn default() -> Self {
        Self {
            position: HudPosition::TopLeft,
            text_color: HudColor::green(),
            bg_color: HudColor::with_alpha(0, 0, 0, 128),
            font_size: 14,
            history_size: 120,
            show_fps: true,
            show_frame_time: true,
            show_cpu: true,
            show_gpu: true,
            show_bitrate: true,
            show_dropped: true,
            show_frame_time_graph: true,
            show_memory: false,
            update_interval: Duration::from_millis(250),
            graph_width: 60,
            graph_height: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// FrameTimeGraph -- renders frame-time as a sparkline
// ---------------------------------------------------------------------------

/// A sparkline-style frame-time graph using Unicode block characters.
pub struct FrameTimeGraph {
    /// Width in columns.
    width: usize,
    /// Height in rows.
    height: usize,
}

impl FrameTimeGraph {
    /// Create a new frame-time graph renderer.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// Render the frame-time graph from the given samples.
    ///
    /// Returns a vector of strings, one per row (top to bottom).
    /// Each column represents one sample, scaled to [0, max_ms].
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn render(&self, samples: &[f32], target_ms: f32) -> Vec<String> {
        if samples.is_empty() {
            return vec![String::new(); self.height];
        }

        // Use target_ms * 2 as the graph ceiling
        let max_val = target_ms * 2.0;

        // Take the last `width` samples
        let start = samples.len().saturating_sub(self.width);
        let visible = &samples[start..];

        // For each column, compute normalised height [0, height]
        let total_cells = self.height;
        let bar_heights: Vec<usize> = visible
            .iter()
            .map(|&v| {
                let normalised = (v / max_val).clamp(0.0, 1.0);
                (normalised * total_cells as f32).round() as usize
            })
            .collect();

        // Build rows from top (highest) to bottom (lowest)
        let mut rows = Vec::with_capacity(self.height);
        for row in 0..self.height {
            let threshold = self.height - row;
            let mut line = String::with_capacity(self.width);
            for &bh in &bar_heights {
                if bh >= threshold {
                    line.push('#');
                } else {
                    line.push(' ');
                }
            }
            // Pad remaining columns
            while line.len() < self.width {
                line.push(' ');
            }
            rows.push(line);
        }

        rows
    }
}

// ---------------------------------------------------------------------------
// UsageBar -- renders CPU/GPU usage as horizontal bar
// ---------------------------------------------------------------------------

/// Renders a usage percentage as a horizontal text bar.
pub struct UsageBar;

impl UsageBar {
    /// Render a usage bar for the given percentage (0.0 - 1.0).
    ///
    /// Returns a string like `[########    ] 67%`
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn render(usage: f32, width: usize, label: &str) -> String {
        let clamped = usage.clamp(0.0, 1.0);
        let filled = (clamped * width as f32).round() as usize;
        let empty = width.saturating_sub(filled);

        let bar: String = "#".repeat(filled) + &" ".repeat(empty);
        format!("{label}: [{bar}] {:.0}%", clamped * 100.0)
    }

    /// Determine colour for a usage value.
    #[must_use]
    pub fn usage_color(usage: f32) -> HudColor {
        if usage < 0.6 {
            HudColor::green()
        } else if usage < 0.85 {
            HudColor::yellow()
        } else {
            HudColor::red()
        }
    }
}

// ---------------------------------------------------------------------------
// PerfHud
// ---------------------------------------------------------------------------

/// The performance HUD collects samples and produces text lines for overlay.
pub struct PerfHud {
    config: PerfHudConfig,
    history: VecDeque<PerfSample>,
    graph: FrameTimeGraph,
}

impl PerfHud {
    /// Create a new HUD with the given configuration.
    #[must_use]
    pub fn new(config: PerfHudConfig) -> Self {
        let cap = config.history_size;
        let graph = FrameTimeGraph::new(config.graph_width, config.graph_height);
        Self {
            config,
            history: VecDeque::with_capacity(cap),
            graph,
        }
    }

    /// Record a new performance sample.
    pub fn push_sample(&mut self, sample: PerfSample) {
        if self.history.len() == self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(sample);
    }

    /// Number of samples currently stored.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Average FPS over the stored history.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_fps(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.fps).sum();
        sum / self.history.len() as f32
    }

    /// Average frame time in milliseconds over the history.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_frame_time_ms(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.frame_time_ms).sum();
        sum / self.history.len() as f32
    }

    /// Average CPU usage over the history.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_cpu(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.cpu_usage).sum();
        sum / self.history.len() as f32
    }

    /// Average GPU usage over the history.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_gpu(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.gpu_usage).sum();
        sum / self.history.len() as f32
    }

    /// Total dropped frames across all stored samples.
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        self.history
            .iter()
            .map(|s| u64::from(s.dropped_frames))
            .sum()
    }

    /// Determine colour for a given FPS value relative to a target.
    #[must_use]
    pub fn fps_color(&self, fps: f32, target_fps: f32) -> HudColor {
        if fps >= target_fps * 0.95 {
            HudColor::green()
        } else if fps >= target_fps * 0.75 {
            HudColor::yellow()
        } else {
            HudColor::red()
        }
    }

    /// Render the HUD as a vector of display lines.
    #[must_use]
    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.config.show_fps {
            lines.push(format!("FPS: {:.1}", self.avg_fps()));
        }
        if self.config.show_frame_time {
            lines.push(format!("Frame: {:.2} ms", self.avg_frame_time_ms()));
        }
        if self.config.show_cpu {
            lines.push(UsageBar::render(self.avg_cpu(), 20, "CPU"));
        }
        if self.config.show_gpu {
            lines.push(UsageBar::render(self.avg_gpu(), 20, "GPU"));
        }
        if self.config.show_bitrate {
            if let Some(last) = self.history.back() {
                lines.push(format!("Bitrate: {} kbps", last.bitrate_kbps));
            }
        }
        if self.config.show_dropped {
            lines.push(format!("Dropped: {}", self.total_dropped()));
        }
        if self.config.show_memory {
            if let Some(last) = self.history.back() {
                lines.push(format!(
                    "RAM: {} MB  VRAM: {} MB",
                    last.ram_mb, last.vram_mb
                ));
            }
        }
        if self.config.show_frame_time_graph {
            let frame_times: Vec<f32> = self.history.iter().map(|s| s.frame_time_ms).collect();
            let target_ms = if self.avg_fps() > 0.0 {
                1000.0 / self.avg_fps()
            } else {
                16.67
            };
            let graph_rows = self.graph.render(&frame_times, target_ms);
            lines.push("Frame time:".to_string());
            for row in &graph_rows {
                lines.push(format!("|{row}|"));
            }
        }
        lines
    }

    /// Clear all stored samples.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &PerfHudConfig {
        &self.config
    }

    /// The 1st-percentile (worst-case) FPS across the history.
    #[must_use]
    pub fn percentile_1_fps(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut fps_vals: Vec<f32> = self.history.iter().map(|s| s.fps).collect();
        fps_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        #[allow(clippy::cast_precision_loss)]
        let idx = ((fps_vals.len() as f32) * 0.01).ceil() as usize;
        let idx = idx.max(1).min(fps_vals.len()) - 1;
        fps_vals[idx]
    }

    /// Maximum frame time in the history.
    #[must_use]
    pub fn max_frame_time_ms(&self) -> f32 {
        self.history
            .iter()
            .map(|s| s.frame_time_ms)
            .fold(0.0_f32, f32::max)
    }

    /// Minimum frame time in the history.
    #[must_use]
    pub fn min_frame_time_ms(&self) -> f32 {
        self.history
            .iter()
            .map(|s| s.frame_time_ms)
            .fold(f32::INFINITY, f32::min)
    }

    /// Frame-time jitter (standard deviation) across the history.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_time_jitter(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_frame_time_ms();
        let variance: f32 = self
            .history
            .iter()
            .map(|s| {
                let diff = s.frame_time_ms - mean;
                diff * diff
            })
            .sum::<f32>()
            / (self.history.len() as f32 - 1.0);
        variance.sqrt()
    }

    /// Get the latest sample, if any.
    #[must_use]
    pub fn latest_sample(&self) -> Option<&PerfSample> {
        self.history.back()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(fps: f32, cpu: f32, gpu: f32, dropped: u32) -> PerfSample {
        PerfSample {
            fps,
            frame_time_ms: if fps > 0.0 { 1000.0 / fps } else { 0.0 },
            cpu_usage: cpu,
            gpu_usage: gpu,
            bitrate_kbps: 6000,
            dropped_frames: dropped,
            vram_mb: 512,
            ram_mb: 2048,
        }
    }

    #[test]
    fn test_hud_creation_default() {
        let hud = PerfHud::new(PerfHudConfig::default());
        assert_eq!(hud.sample_count(), 0);
    }

    #[test]
    fn test_push_sample() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(PerfSample::default());
        assert_eq!(hud.sample_count(), 1);
    }

    #[test]
    fn test_history_cap_eviction() {
        let cfg = PerfHudConfig {
            history_size: 3,
            ..PerfHudConfig::default()
        };
        let mut hud = PerfHud::new(cfg);
        for i in 0..5 {
            #[allow(clippy::cast_precision_loss)]
            hud.push_sample(sample(60.0 + i as f32, 0.5, 0.5, 0));
        }
        assert_eq!(hud.sample_count(), 3);
    }

    #[test]
    fn test_avg_fps_empty() {
        let hud = PerfHud::new(PerfHudConfig::default());
        assert!((hud.avg_fps() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_avg_fps_single() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.5, 0.5, 0));
        assert!((hud.avg_fps() - 60.0).abs() < 1e-3);
    }

    #[test]
    fn test_avg_fps_multiple() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(50.0, 0.0, 0.0, 0));
        hud.push_sample(sample(70.0, 0.0, 0.0, 0));
        assert!((hud.avg_fps() - 60.0).abs() < 1e-3);
    }

    #[test]
    fn test_avg_frame_time() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.0, 0.0, 0));
        // 1000/60 ~= 16.667
        assert!((hud.avg_frame_time_ms() - 16.6667).abs() < 0.1);
    }

    #[test]
    fn test_avg_cpu_gpu() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.4, 0.8, 0));
        hud.push_sample(sample(60.0, 0.6, 0.6, 0));
        assert!((hud.avg_cpu() - 0.5).abs() < 1e-5);
        assert!((hud.avg_gpu() - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_total_dropped() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.0, 0.0, 5));
        hud.push_sample(sample(60.0, 0.0, 0.0, 3));
        assert_eq!(hud.total_dropped(), 8);
    }

    #[test]
    fn test_fps_color_green() {
        let hud = PerfHud::new(PerfHudConfig::default());
        let c = hud.fps_color(59.0, 60.0);
        assert_eq!(c, HudColor::green());
    }

    #[test]
    fn test_fps_color_yellow() {
        let hud = PerfHud::new(PerfHudConfig::default());
        let c = hud.fps_color(50.0, 60.0);
        assert_eq!(c, HudColor::yellow());
    }

    #[test]
    fn test_fps_color_red() {
        let hud = PerfHud::new(PerfHudConfig::default());
        let c = hud.fps_color(30.0, 60.0);
        assert_eq!(c, HudColor::red());
    }

    #[test]
    fn test_render_lines_non_empty() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.5, 0.7, 0));
        let lines = hud.render_lines();
        assert!(!lines.is_empty());
        assert!(lines[0].contains("FPS"));
    }

    #[test]
    fn test_render_lines_cpu_gpu_bars() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.5, 0.7, 0));
        let lines = hud.render_lines();
        let cpu_line = lines.iter().find(|l| l.starts_with("CPU:"));
        assert!(cpu_line.is_some());
        let gpu_line = lines.iter().find(|l| l.starts_with("GPU:"));
        assert!(gpu_line.is_some());
    }

    #[test]
    fn test_render_lines_frame_time_graph() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        for i in 0..20 {
            #[allow(clippy::cast_precision_loss)]
            hud.push_sample(sample(60.0 + (i % 5) as f32, 0.0, 0.0, 0));
        }
        let lines = hud.render_lines();
        let graph_header = lines.iter().any(|l| l.contains("Frame time"));
        assert!(graph_header);
    }

    #[test]
    fn test_render_lines_memory() {
        let cfg = PerfHudConfig {
            show_memory: true,
            ..PerfHudConfig::default()
        };
        let mut hud = PerfHud::new(cfg);
        hud.push_sample(sample(60.0, 0.5, 0.5, 0));
        let lines = hud.render_lines();
        let mem_line = lines
            .iter()
            .any(|l| l.contains("RAM:") && l.contains("VRAM:"));
        assert!(mem_line);
    }

    #[test]
    fn test_clear() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(PerfSample::default());
        hud.clear();
        assert_eq!(hud.sample_count(), 0);
    }

    #[test]
    fn test_percentile_1_fps() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        for i in 0..100 {
            #[allow(clippy::cast_precision_loss)]
            hud.push_sample(sample(30.0 + i as f32, 0.0, 0.0, 0));
        }
        let p1 = hud.percentile_1_fps();
        // lowest values are 30, 31, ... - 1% of 100 = index 0
        assert!(p1 >= 30.0 && p1 <= 32.0);
    }

    #[test]
    fn test_hud_position_variants() {
        let positions = [
            HudPosition::TopLeft,
            HudPosition::TopRight,
            HudPosition::BottomLeft,
            HudPosition::BottomRight,
        ];
        for pos in positions {
            let cfg = PerfHudConfig {
                position: pos,
                ..PerfHudConfig::default()
            };
            let hud = PerfHud::new(cfg);
            assert_eq!(hud.config().position, pos);
        }
    }

    // FrameTimeGraph tests

    #[test]
    fn test_frame_time_graph_empty() {
        let graph = FrameTimeGraph::new(10, 3);
        let rows = graph.render(&[], 16.67);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_frame_time_graph_single_sample() {
        let graph = FrameTimeGraph::new(5, 3);
        let rows = graph.render(&[16.67], 16.67);
        assert_eq!(rows.len(), 3);
        // The single bar should have some '#' characters
        let has_bar = rows.iter().any(|r| r.contains('#'));
        assert!(has_bar);
    }

    #[test]
    fn test_frame_time_graph_full() {
        let graph = FrameTimeGraph::new(5, 3);
        let samples = vec![16.0, 17.0, 33.0, 10.0, 20.0];
        let rows = graph.render(&samples, 16.67);
        assert_eq!(rows.len(), 3);
        // All rows should be `width` characters long
        for row in &rows {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn test_frame_time_graph_overflow_clamped() {
        let graph = FrameTimeGraph::new(5, 3);
        // Very high value should be clamped to max height
        let samples = vec![1000.0];
        let rows = graph.render(&samples, 16.67);
        // All rows should have '#' in the first column
        for row in &rows {
            assert_eq!(row.chars().next(), Some('#'));
        }
    }

    // UsageBar tests

    #[test]
    fn test_usage_bar_zero() {
        let bar = UsageBar::render(0.0, 10, "CPU");
        assert!(bar.contains("CPU"));
        assert!(bar.contains("0%"));
    }

    #[test]
    fn test_usage_bar_full() {
        let bar = UsageBar::render(1.0, 10, "GPU");
        assert!(bar.contains("100%"));
        assert!(bar.contains("##########"));
    }

    #[test]
    fn test_usage_bar_half() {
        let bar = UsageBar::render(0.5, 10, "CPU");
        assert!(bar.contains("50%"));
    }

    #[test]
    fn test_usage_color_green() {
        let c = UsageBar::usage_color(0.3);
        assert_eq!(c, HudColor::green());
    }

    #[test]
    fn test_usage_color_yellow() {
        let c = UsageBar::usage_color(0.7);
        assert_eq!(c, HudColor::yellow());
    }

    #[test]
    fn test_usage_color_red() {
        let c = UsageBar::usage_color(0.95);
        assert_eq!(c, HudColor::red());
    }

    // Additional PerfHud tests

    #[test]
    fn test_max_frame_time() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.0, 0.0, 0));
        hud.push_sample(sample(30.0, 0.0, 0.0, 0));
        // 30 fps => 33.33ms frame time
        assert!(hud.max_frame_time_ms() > 30.0);
    }

    #[test]
    fn test_min_frame_time() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(60.0, 0.0, 0.0, 0));
        hud.push_sample(sample(120.0, 0.0, 0.0, 0));
        // 120 fps => ~8.33ms
        assert!(hud.min_frame_time_ms() < 10.0);
    }

    #[test]
    fn test_frame_time_jitter_constant() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        for _ in 0..10 {
            hud.push_sample(sample(60.0, 0.0, 0.0, 0));
        }
        // All same FPS => jitter ~0
        assert!(hud.frame_time_jitter() < 0.01);
    }

    #[test]
    fn test_frame_time_jitter_variable() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        hud.push_sample(sample(30.0, 0.0, 0.0, 0));
        hud.push_sample(sample(120.0, 0.0, 0.0, 0));
        // Very different frame times => non-zero jitter
        assert!(hud.frame_time_jitter() > 1.0);
    }

    #[test]
    fn test_latest_sample() {
        let mut hud = PerfHud::new(PerfHudConfig::default());
        assert!(hud.latest_sample().is_none());
        hud.push_sample(sample(60.0, 0.5, 0.7, 2));
        let latest = hud.latest_sample().expect("should have sample");
        assert!((latest.fps - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hud_color_to_rgba() {
        let c = HudColor::new(10, 20, 30);
        assert_eq!(c.to_rgba(), [10, 20, 30, 255]);
    }
}
