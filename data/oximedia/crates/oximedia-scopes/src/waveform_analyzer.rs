#![allow(dead_code)]
//! High-level waveform analysis API.
//!
//! Provides:
//! * [`WaveformMode`]     – display mode selection (Luma, Rgb, Parade).
//! * [`WaveformConfig`]   – display configuration with height accessor.
//! * [`WaveformAnalyzer`] – per-line analysis producing luma/RGB columns.
//! * [`WaveformReport`]   – aggregated statistics across the full frame.

// ---------------------------------------------------------------------------
// WaveformMode
// ---------------------------------------------------------------------------

/// Selects which signal(s) to plot on the waveform monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformMode {
    /// Plot the luma (Y′) channel only.
    Luma,
    /// Plot all three RGB channels superimposed.
    Rgb,
    /// Plot R, G, B as separate side-by-side parade columns.
    Parade,
}

impl WaveformMode {
    /// Returns the number of parade columns required for this mode.
    #[must_use]
    pub const fn parade_columns(self) -> usize {
        match self {
            Self::Luma | Self::Rgb => 1,
            Self::Parade => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// WaveformConfig
// ---------------------------------------------------------------------------

/// Configuration for the waveform monitor display.
#[derive(Debug, Clone)]
pub struct WaveformConfig {
    /// Display height in pixels (scope resolution, not source frame).
    display_height: u32,
    /// Display width in pixels.
    display_width: u32,
    /// Which signal(s) to render.
    pub mode: WaveformMode,
    /// If `true`, scale values to IRE (0–100) instead of normalised (0.0–1.0).
    pub ire_scale: bool,
    /// Graticule step size (0 = no graticule).
    pub graticule_step: u32,
}

impl WaveformConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(display_width: u32, display_height: u32, mode: WaveformMode) -> Self {
        Self {
            display_height,
            display_width,
            mode,
            ire_scale: false,
            graticule_step: 10,
        }
    }

    /// Display height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.display_height
    }

    /// Display width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.display_width
    }
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self::new(512, 256, WaveformMode::Luma)
    }
}

// ---------------------------------------------------------------------------
// WaveformAnalyzer
// ---------------------------------------------------------------------------

/// Analyses video frames and produces waveform column data.
pub struct WaveformAnalyzer {
    config: WaveformConfig,
}

impl WaveformAnalyzer {
    /// Create an analyser with the given configuration.
    #[must_use]
    pub fn new(config: WaveformConfig) -> Self {
        Self { config }
    }

    /// Analyse a single scanline of RGB24 pixel data.
    ///
    /// `line` must contain `width * 3` bytes in `[R, G, B, …]` order.
    ///
    /// Returns a `Vec<f32>` where each element is the normalised luma value
    /// `[0.0, 1.0]` of the corresponding pixel when `mode` is [`WaveformMode::Luma`],
    /// or the channel average when `mode` is [`WaveformMode::Rgb`].
    /// For [`WaveformMode::Parade`] the three channel values are interleaved:
    /// `[r0, g0, b0, r1, g1, b1, …]`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze_line(&self, line: &[u8], width: usize) -> Vec<f32> {
        if line.len() < width * 3 {
            return vec![0.0; width];
        }
        match self.config.mode {
            WaveformMode::Luma => (0..width)
                .map(|x| {
                    let r = line[x * 3] as f32 / 255.0;
                    let g = line[x * 3 + 1] as f32 / 255.0;
                    let b = line[x * 3 + 2] as f32 / 255.0;
                    // BT.709 luma coefficients
                    0.2126 * r + 0.7152 * g + 0.0722 * b
                })
                .collect(),
            WaveformMode::Rgb => (0..width)
                .map(|x| {
                    let r = line[x * 3] as f32 / 255.0;
                    let g = line[x * 3 + 1] as f32 / 255.0;
                    let b = line[x * 3 + 2] as f32 / 255.0;
                    (r + g + b) / 3.0
                })
                .collect(),
            WaveformMode::Parade => (0..width)
                .flat_map(|x| {
                    let r = line[x * 3] as f32 / 255.0;
                    let g = line[x * 3 + 1] as f32 / 255.0;
                    let b = line[x * 3 + 2] as f32 / 255.0;
                    [r, g, b]
                })
                .collect(),
        }
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub const fn config(&self) -> &WaveformConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// WaveformReport
// ---------------------------------------------------------------------------

/// Aggregated waveform statistics for a complete frame.
#[derive(Debug, Clone)]
pub struct WaveformReport {
    /// Minimum luma value observed (0.0 – 1.0).
    min_luma: f32,
    /// Maximum luma value observed (0.0 – 1.0).
    max_luma: f32,
    /// Arithmetic mean luma value.
    mean_luma: f32,
    /// Number of pixels analysed.
    pixel_count: u64,
}

impl WaveformReport {
    /// Build a report from a flat slice of normalised luma values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_luma_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                min_luma: 0.0,
                max_luma: 0.0,
                mean_luma: 0.0,
                pixel_count: 0,
            };
        }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0_f32;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        Self {
            min_luma: min,
            max_luma: max,
            mean_luma: sum / values.len() as f32,
            pixel_count: values.len() as u64,
        }
    }

    /// Minimum luma value in the frame.
    #[must_use]
    pub fn min_luma(&self) -> f32 {
        self.min_luma
    }

    /// Maximum luma value in the frame.
    #[must_use]
    pub fn max_luma(&self) -> f32 {
        self.max_luma
    }

    /// Mean luma value.
    #[must_use]
    pub fn mean_luma(&self) -> f32 {
        self.mean_luma
    }

    /// Dynamic range (max − min).
    #[must_use]
    pub fn luma_range(&self) -> f32 {
        self.max_luma - self.min_luma
    }

    /// Number of pixels included in this report.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        self.pixel_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_mode_parade_columns() {
        assert_eq!(WaveformMode::Luma.parade_columns(), 1);
        assert_eq!(WaveformMode::Rgb.parade_columns(), 1);
        assert_eq!(WaveformMode::Parade.parade_columns(), 3);
    }

    #[test]
    fn test_config_height() {
        let cfg = WaveformConfig::new(512, 256, WaveformMode::Luma);
        assert_eq!(cfg.height(), 256);
    }

    #[test]
    fn test_config_width() {
        let cfg = WaveformConfig::new(512, 256, WaveformMode::Rgb);
        assert_eq!(cfg.width(), 512);
    }

    #[test]
    fn test_config_default() {
        let cfg = WaveformConfig::default();
        assert_eq!(cfg.mode, WaveformMode::Luma);
        assert_eq!(cfg.height(), 256);
    }

    #[test]
    fn test_analyze_line_luma_black() {
        let cfg = WaveformConfig::new(4, 256, WaveformMode::Luma);
        let analyzer = WaveformAnalyzer::new(cfg);
        let line = vec![0u8; 4 * 3];
        let result = analyzer.analyze_line(&line, 4);
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert!(v.abs() < 1e-4);
        }
    }

    #[test]
    fn test_analyze_line_luma_white() {
        let cfg = WaveformConfig::new(2, 256, WaveformMode::Luma);
        let analyzer = WaveformAnalyzer::new(cfg);
        let line = vec![255u8; 2 * 3];
        let result = analyzer.analyze_line(&line, 2);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_analyze_line_rgb_mode() {
        let cfg = WaveformConfig::new(1, 256, WaveformMode::Rgb);
        let analyzer = WaveformAnalyzer::new(cfg);
        // R=255, G=0, B=0 => average = 1/3 ≈ 0.333
        let line = vec![255u8, 0, 0];
        let result = analyzer.analyze_line(&line, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_line_parade_mode() {
        let cfg = WaveformConfig::new(1, 256, WaveformMode::Parade);
        let analyzer = WaveformAnalyzer::new(cfg);
        let line = vec![255u8, 128, 0];
        let result = analyzer.analyze_line(&line, 1);
        assert_eq!(result.len(), 3); // interleaved r, g, b
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_line_insufficient_data() {
        let cfg = WaveformConfig::new(4, 256, WaveformMode::Luma);
        let analyzer = WaveformAnalyzer::new(cfg);
        let result = analyzer.analyze_line(&[0u8; 3], 4); // only 1 pixel
        assert_eq!(result, vec![0.0; 4]);
    }

    #[test]
    fn test_report_empty() {
        let report = WaveformReport::from_luma_values(&[]);
        assert_eq!(report.pixel_count(), 0);
        assert_eq!(report.min_luma(), 0.0);
        assert_eq!(report.max_luma(), 0.0);
    }

    #[test]
    fn test_report_min_max() {
        let values = vec![0.1, 0.5, 0.9];
        let report = WaveformReport::from_luma_values(&values);
        assert!((report.min_luma() - 0.1).abs() < 1e-5);
        assert!((report.max_luma() - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_report_mean() {
        let values = vec![0.2, 0.4, 0.6];
        let report = WaveformReport::from_luma_values(&values);
        assert!((report.mean_luma() - 0.4).abs() < 1e-4);
    }

    #[test]
    fn test_report_luma_range() {
        let values = vec![0.1, 0.9];
        let report = WaveformReport::from_luma_values(&values);
        assert!((report.luma_range() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_report_pixel_count() {
        let values: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let report = WaveformReport::from_luma_values(&values);
        assert_eq!(report.pixel_count(), 100);
    }
}
