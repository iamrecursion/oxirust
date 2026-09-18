#![allow(dead_code)]
//! LED volume calibration for virtual production stages.
//!
//! Implements per-panel colour uniformity, brightness matching,
//! geometric alignment, and verification routines used to prepare
//! an LED volume before a shoot.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Calibration target – what aspect of the volume is being calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalibrationTarget {
    /// Per-panel colour uniformity.
    ColorUniformity,
    /// Brightness matching across panels.
    Brightness,
    /// Geometric alignment (seam correction).
    Geometry,
    /// Black level calibration.
    BlackLevel,
    /// White point calibration.
    WhitePoint,
    /// Full calibration (all targets).
    Full,
}

impl CalibrationTarget {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::ColorUniformity => "Color Uniformity",
            Self::Brightness => "Brightness",
            Self::Geometry => "Geometry",
            Self::BlackLevel => "Black Level",
            Self::WhitePoint => "White Point",
            Self::Full => "Full Calibration",
        }
    }

    /// Returns `true` for targets that affect colour output.
    #[must_use]
    pub fn is_color_related(&self) -> bool {
        matches!(
            self,
            Self::ColorUniformity | Self::WhitePoint | Self::BlackLevel | Self::Full
        )
    }
}

/// Measured colour value in CIE xy + Y space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMeasurement {
    /// CIE x chromaticity.
    pub x: f64,
    /// CIE y chromaticity.
    pub y: f64,
    /// Luminance in cd/m^2.
    pub luminance: f64,
}

impl ColorMeasurement {
    /// Creates a new measurement.
    #[must_use]
    pub fn new(x: f64, y: f64, luminance: f64) -> Self {
        Self { x, y, luminance }
    }

    /// D65 reference white.
    #[must_use]
    pub fn d65() -> Self {
        Self {
            x: 0.3127,
            y: 0.3290,
            luminance: 100.0,
        }
    }

    /// Euclidean distance in CIE xy plane.
    #[must_use]
    pub fn chromaticity_distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Luminance difference as a ratio (other / self).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn luminance_ratio(&self, other: &Self) -> f64 {
        if self.luminance.abs() < f64::EPSILON {
            return 0.0;
        }
        other.luminance / self.luminance
    }
}

/// Per-panel calibration data.
#[derive(Debug, Clone)]
pub struct PanelCalibration {
    /// Panel identifier.
    pub panel_id: String,
    /// Row index in the wall.
    pub row: u32,
    /// Column index in the wall.
    pub col: u32,
    /// Measured white point.
    pub white_point: ColorMeasurement,
    /// Measured black level.
    pub black_level: ColorMeasurement,
    /// Correction gain per RGB channel.
    pub gain: [f64; 3],
    /// Correction offset per RGB channel.
    pub offset: [f64; 3],
    /// Whether this panel passed calibration.
    pub passed: bool,
}

impl PanelCalibration {
    /// Creates a new uncalibrated panel entry.
    pub fn new(panel_id: impl Into<String>, row: u32, col: u32) -> Self {
        Self {
            panel_id: panel_id.into(),
            row,
            col,
            white_point: ColorMeasurement::d65(),
            black_level: ColorMeasurement::new(0.0, 0.0, 0.0),
            gain: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            passed: false,
        }
    }

    /// Applies the gain/offset correction to an input RGB triple (0.0–1.0).
    #[must_use]
    pub fn correct(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            (rgb[0] * self.gain[0] + self.offset[0]).clamp(0.0, 1.0),
            (rgb[1] * self.gain[1] + self.offset[1]).clamp(0.0, 1.0),
            (rgb[2] * self.gain[2] + self.offset[2]).clamp(0.0, 1.0),
        ]
    }

    /// Contrast ratio (white luminance / black luminance).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn contrast_ratio(&self) -> f64 {
        if self.black_level.luminance.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        self.white_point.luminance / self.black_level.luminance
    }
}

/// Overall calibration status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStatus {
    /// Not yet started.
    Pending,
    /// In progress.
    Running,
    /// Completed successfully – all panels passed.
    Passed,
    /// Completed with some panels failing.
    PartialFail,
    /// Calibration failed.
    Failed,
}

/// Configuration for a calibration run.
#[derive(Debug, Clone)]
pub struct VolumeCalibrationConfig {
    /// Which aspect to calibrate.
    pub target: CalibrationTarget,
    /// Maximum acceptable chromaticity drift from reference.
    pub max_chroma_drift: f64,
    /// Maximum acceptable luminance deviation (fraction, e.g. 0.05 = 5%).
    pub max_luma_deviation: f64,
    /// Reference white point.
    pub reference_white: ColorMeasurement,
    /// Number of measurement samples per panel.
    pub samples_per_panel: u32,
}

impl Default for VolumeCalibrationConfig {
    fn default() -> Self {
        Self {
            target: CalibrationTarget::Full,
            max_chroma_drift: 0.005,
            max_luma_deviation: 0.03,
            reference_white: ColorMeasurement::d65(),
            samples_per_panel: 5,
        }
    }
}

/// Statistics for a calibration run.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Total panels calibrated.
    pub panels_total: u32,
    /// Panels that passed.
    pub panels_passed: u32,
    /// Panels that failed.
    pub panels_failed: u32,
    /// Average chromaticity drift.
    pub avg_chroma_drift: f64,
    /// Maximum chromaticity drift observed.
    pub max_chroma_drift_observed: f64,
    /// Average luminance deviation.
    pub avg_luma_deviation: f64,
    /// Calibration duration.
    pub duration: Duration,
}

impl CalibrationStats {
    /// Creates zeroed stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            panels_total: 0,
            panels_passed: 0,
            panels_failed: 0,
            avg_chroma_drift: 0.0,
            max_chroma_drift_observed: 0.0,
            avg_luma_deviation: 0.0,
            duration: Duration::ZERO,
        }
    }

    /// Pass rate as a fraction.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.panels_total == 0 {
            return 0.0;
        }
        f64::from(self.panels_passed) / f64::from(self.panels_total)
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume calibration manager.
pub struct VolumeCalibrator {
    /// Configuration.
    config: VolumeCalibrationConfig,
    /// Per-panel calibration data, keyed by `panel_id`.
    panels: HashMap<String, PanelCalibration>,
    /// Current status.
    status: CalibrationStatus,
    /// Statistics.
    stats: CalibrationStats,
    /// When the current run started.
    started_at: Option<Instant>,
}

impl VolumeCalibrator {
    /// Creates a new calibrator.
    #[must_use]
    pub fn new(config: VolumeCalibrationConfig) -> Self {
        Self {
            config,
            panels: HashMap::new(),
            status: CalibrationStatus::Pending,
            stats: CalibrationStats::new(),
            started_at: None,
        }
    }

    /// Registers a panel for calibration.
    pub fn add_panel(&mut self, panel: PanelCalibration) {
        self.panels.insert(panel.panel_id.clone(), panel);
    }

    /// Returns the number of registered panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Gets a panel by ID.
    #[must_use]
    pub fn get_panel(&self, panel_id: &str) -> Option<&PanelCalibration> {
        self.panels.get(panel_id)
    }

    /// Starts a calibration run.
    pub fn start(&mut self) {
        self.status = CalibrationStatus::Running;
        self.started_at = Some(Instant::now());
        self.stats = CalibrationStats::new();
    }

    /// Evaluates a single panel against the configuration thresholds.
    pub fn evaluate_panel(&mut self, panel_id: &str) -> bool {
        let config = self.config.clone();
        if let Some(panel) = self.panels.get_mut(panel_id) {
            let chroma_drift = panel
                .white_point
                .chromaticity_distance(&config.reference_white);
            let luma_ratio = config.reference_white.luminance_ratio(&panel.white_point);
            let luma_dev = (luma_ratio - 1.0).abs();

            let pass =
                chroma_drift <= config.max_chroma_drift && luma_dev <= config.max_luma_deviation;

            panel.passed = pass;
            self.stats.panels_total += 1;
            if pass {
                self.stats.panels_passed += 1;
            } else {
                self.stats.panels_failed += 1;
            }

            if chroma_drift > self.stats.max_chroma_drift_observed {
                self.stats.max_chroma_drift_observed = chroma_drift;
            }

            pass
        } else {
            false
        }
    }

    /// Finishes calibration and computes final stats.
    pub fn finish(&mut self) {
        if let Some(start) = self.started_at.take() {
            self.stats.duration = start.elapsed();
        }
        if self.stats.panels_failed == 0 && self.stats.panels_total > 0 {
            self.status = CalibrationStatus::Passed;
        } else if self.stats.panels_passed > 0 {
            self.status = CalibrationStatus::PartialFail;
        } else if self.stats.panels_total > 0 {
            self.status = CalibrationStatus::Failed;
        }
    }

    /// Returns current calibration status.
    #[must_use]
    pub fn status(&self) -> CalibrationStatus {
        self.status
    }

    /// Returns current stats.
    #[must_use]
    pub fn stats(&self) -> &CalibrationStats {
        &self.stats
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &VolumeCalibrationConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Thermal drift compensation
// ---------------------------------------------------------------------------

/// A timestamped color measurement for tracking drift over time.
#[derive(Debug, Clone)]
pub struct TimestampedMeasurement {
    /// When the measurement was taken (seconds since calibration start).
    pub elapsed_secs: f64,
    /// Panel temperature in degrees Celsius (estimated or measured).
    pub temperature_c: f64,
    /// Color measurement at this point in time.
    pub measurement: ColorMeasurement,
}

/// Configuration for thermal drift compensation.
#[derive(Debug, Clone)]
pub struct ThermalDriftConfig {
    /// Maximum number of samples to retain per panel.
    pub max_samples: usize,
    /// Temperature coefficient: expected chromaticity shift per degree C.
    /// Typical LED panels drift ~0.001 in CIE xy per 10 degrees.
    pub chroma_drift_per_degree: f64,
    /// Temperature coefficient: expected luminance change per degree C
    /// as a fraction (e.g. 0.003 = 0.3% per degree).
    pub luma_drift_per_degree: f64,
    /// Reference temperature at calibration time (degrees C).
    pub reference_temperature_c: f64,
    /// Minimum samples needed before drift estimation is active.
    pub min_samples_for_estimation: usize,
    /// Whether to auto-apply gain corrections based on drift model.
    pub auto_correct: bool,
}

impl Default for ThermalDriftConfig {
    fn default() -> Self {
        Self {
            max_samples: 128,
            chroma_drift_per_degree: 0.0001,
            luma_drift_per_degree: 0.003,
            reference_temperature_c: 25.0,
            min_samples_for_estimation: 5,
            auto_correct: true,
        }
    }
}

/// Linear regression result for a single variable.
#[derive(Debug, Clone, Copy)]
pub struct LinearFit {
    /// Slope (change per unit of independent variable).
    pub slope: f64,
    /// Intercept (value at zero).
    pub intercept: f64,
    /// R-squared goodness-of-fit.
    pub r_squared: f64,
}

/// Per-panel thermal drift tracker.
#[derive(Debug, Clone)]
pub struct PanelDriftTracker {
    /// Panel identifier.
    pub panel_id: String,
    /// Time-series of measurements.
    samples: Vec<TimestampedMeasurement>,
    /// Maximum samples to retain.
    max_samples: usize,
    /// Estimated luminance drift model (luminance vs temperature).
    luma_fit: Option<LinearFit>,
    /// Estimated CIE x drift model (x vs temperature).
    x_fit: Option<LinearFit>,
    /// Estimated CIE y drift model (y vs temperature).
    y_fit: Option<LinearFit>,
}

impl PanelDriftTracker {
    /// Create a new tracker for a panel.
    #[must_use]
    pub fn new(panel_id: impl Into<String>, max_samples: usize) -> Self {
        Self {
            panel_id: panel_id.into(),
            samples: Vec::new(),
            max_samples,
            luma_fit: None,
            x_fit: None,
            y_fit: None,
        }
    }

    /// Record a new measurement.
    pub fn record(&mut self, measurement: TimestampedMeasurement) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(measurement);
    }

    /// Number of recorded samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Fit linear regression models for luminance, CIE x, and CIE y
    /// as functions of temperature.
    pub fn fit_drift_model(&mut self) {
        if self.samples.len() < 2 {
            return;
        }

        let temps: Vec<f64> = self.samples.iter().map(|s| s.temperature_c).collect();
        let lumas: Vec<f64> = self
            .samples
            .iter()
            .map(|s| s.measurement.luminance)
            .collect();
        let xs: Vec<f64> = self.samples.iter().map(|s| s.measurement.x).collect();
        let ys: Vec<f64> = self.samples.iter().map(|s| s.measurement.y).collect();

        self.luma_fit = Some(linear_regression(&temps, &lumas));
        self.x_fit = Some(linear_regression(&temps, &xs));
        self.y_fit = Some(linear_regression(&temps, &ys));
    }

    /// Predict the color measurement at a given temperature using the fitted model.
    ///
    /// Returns `None` if no model has been fitted yet.
    #[must_use]
    pub fn predict_at_temperature(&self, temp_c: f64) -> Option<ColorMeasurement> {
        let luma_fit = self.luma_fit.as_ref()?;
        let x_fit = self.x_fit.as_ref()?;
        let y_fit = self.y_fit.as_ref()?;

        Some(ColorMeasurement {
            x: x_fit.slope * temp_c + x_fit.intercept,
            y: y_fit.slope * temp_c + y_fit.intercept,
            luminance: luma_fit.slope * temp_c + luma_fit.intercept,
        })
    }

    /// Compute the correction gain needed to compensate for thermal drift
    /// at the given temperature, relative to the reference measurement.
    ///
    /// Returns per-channel gain adjustments `[r_gain, g_gain, b_gain]` that
    /// should be multiplied with the panel output to compensate for drift.
    #[must_use]
    pub fn correction_gain(
        &self,
        current_temp_c: f64,
        reference: &ColorMeasurement,
    ) -> Option<[f64; 3]> {
        let predicted = self.predict_at_temperature(current_temp_c)?;

        // Luminance compensation: ratio of reference to predicted
        let luma_ratio = if predicted.luminance.abs() > 1e-10 {
            reference.luminance / predicted.luminance
        } else {
            1.0
        };

        // Chromaticity shift compensation
        // Map CIE xy shift to approximate RGB gain adjustments
        // Using simplified Bradford-like approach:
        //   dx > 0 means shift toward red => reduce red, boost blue
        //   dy > 0 means shift toward green => reduce green, boost red+blue
        let dx = predicted.x - reference.x;
        let dy = predicted.y - reference.y;

        let r_gain = luma_ratio * (1.0 - 1.5 * dx + 0.5 * dy);
        let g_gain = luma_ratio * (1.0 + 0.5 * dx - 1.5 * dy);
        let b_gain = luma_ratio * (1.0 + 1.0 * dx + 1.0 * dy);

        Some([
            r_gain.max(0.5).min(2.0),
            g_gain.max(0.5).min(2.0),
            b_gain.max(0.5).min(2.0),
        ])
    }

    /// Get the luminance drift model fit.
    #[must_use]
    pub fn luma_fit(&self) -> Option<&LinearFit> {
        self.luma_fit.as_ref()
    }

    /// Get the CIE x drift model fit.
    #[must_use]
    pub fn x_fit(&self) -> Option<&LinearFit> {
        self.x_fit.as_ref()
    }

    /// Get the CIE y drift model fit.
    #[must_use]
    pub fn y_fit(&self) -> Option<&LinearFit> {
        self.y_fit.as_ref()
    }

    /// Get the recorded samples.
    #[must_use]
    pub fn samples(&self) -> &[TimestampedMeasurement] {
        &self.samples
    }
}

/// Thermal drift compensation manager that tracks multiple panels.
pub struct ThermalDriftCompensator {
    config: ThermalDriftConfig,
    /// Per-panel drift trackers.
    trackers: HashMap<String, PanelDriftTracker>,
}

impl ThermalDriftCompensator {
    /// Create a new thermal drift compensator.
    #[must_use]
    pub fn new(config: ThermalDriftConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
        }
    }

    /// Record a measurement for a panel.
    pub fn record(
        &mut self,
        panel_id: &str,
        elapsed_secs: f64,
        temperature_c: f64,
        measurement: ColorMeasurement,
    ) {
        let tracker = self
            .trackers
            .entry(panel_id.to_string())
            .or_insert_with(|| PanelDriftTracker::new(panel_id, self.config.max_samples));

        tracker.record(TimestampedMeasurement {
            elapsed_secs,
            temperature_c,
            measurement,
        });

        // Re-fit the model if we have enough samples
        if tracker.sample_count() >= self.config.min_samples_for_estimation {
            tracker.fit_drift_model();
        }
    }

    /// Get the correction gain for a panel at its current temperature.
    #[must_use]
    pub fn correction_gain(&self, panel_id: &str, current_temp_c: f64) -> Option<[f64; 3]> {
        let tracker = self.trackers.get(panel_id)?;
        let ref_measurement = ColorMeasurement::new(0.3127, 0.3290, 100.0); // D65 reference
        tracker.correction_gain(current_temp_c, &ref_measurement)
    }

    /// Get a panel's drift tracker.
    #[must_use]
    pub fn tracker(&self, panel_id: &str) -> Option<&PanelDriftTracker> {
        self.trackers.get(panel_id)
    }

    /// Number of tracked panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.trackers.len()
    }

    /// Check if a panel's drift exceeds acceptable thresholds.
    #[must_use]
    pub fn is_drift_excessive(&self, panel_id: &str, current_temp_c: f64) -> bool {
        if let Some(tracker) = self.trackers.get(panel_id) {
            if let Some(predicted) = tracker.predict_at_temperature(current_temp_c) {
                let d65 = ColorMeasurement::d65();
                let chroma_drift = predicted.chromaticity_distance(&d65);
                let luma_dev = if d65.luminance.abs() > 1e-10 {
                    (predicted.luminance / d65.luminance - 1.0).abs()
                } else {
                    0.0
                };

                let temp_delta = (current_temp_c - self.config.reference_temperature_c).abs();
                let expected_chroma = self.config.chroma_drift_per_degree * temp_delta;
                let expected_luma = self.config.luma_drift_per_degree * temp_delta;

                // Excessive if actual drift > 2x expected
                return chroma_drift > expected_chroma * 2.0 || luma_dev > expected_luma * 2.0;
            }
        }
        false
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &ThermalDriftConfig {
        &self.config
    }
}

/// Simple ordinary least-squares linear regression: y = slope * x + intercept.
fn linear_regression(x: &[f64], y: &[f64]) -> LinearFit {
    let n = x.len() as f64;
    if n < 2.0 {
        return LinearFit {
            slope: 0.0,
            intercept: y.first().copied().unwrap_or(0.0),
            r_squared: 0.0,
        };
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return LinearFit {
            slope: 0.0,
            intercept: sum_y / n,
            r_squared: 0.0,
        };
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // R-squared
    let y_mean = sum_y / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let pred = slope * xi + intercept;
            (yi - pred).powi(2)
        })
        .sum();

    let r_squared = if ss_tot.abs() > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    LinearFit {
        slope,
        intercept,
        r_squared,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_target_label() {
        assert_eq!(CalibrationTarget::Full.label(), "Full Calibration");
        assert_eq!(CalibrationTarget::Brightness.label(), "Brightness");
    }

    #[test]
    fn test_calibration_target_is_color_related() {
        assert!(CalibrationTarget::ColorUniformity.is_color_related());
        assert!(CalibrationTarget::Full.is_color_related());
        assert!(!CalibrationTarget::Geometry.is_color_related());
        assert!(!CalibrationTarget::Brightness.is_color_related());
    }

    #[test]
    fn test_color_measurement_distance_zero() {
        let d65 = ColorMeasurement::d65();
        assert!(d65.chromaticity_distance(&d65) < f64::EPSILON);
    }

    #[test]
    fn test_color_measurement_distance_nonzero() {
        let a = ColorMeasurement::new(0.3, 0.3, 100.0);
        let b = ColorMeasurement::new(0.4, 0.3, 100.0);
        assert!((a.chromaticity_distance(&b) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_luminance_ratio() {
        let a = ColorMeasurement::new(0.3, 0.3, 100.0);
        let b = ColorMeasurement::new(0.3, 0.3, 95.0);
        assert!((a.luminance_ratio(&b) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_panel_correction() {
        let mut panel = PanelCalibration::new("P01", 0, 0);
        panel.gain = [1.1, 0.9, 1.0];
        panel.offset = [0.0, 0.05, 0.0];
        let corrected = panel.correct([0.5, 0.5, 0.5]);
        assert!((corrected[0] - 0.55).abs() < 1e-9);
        assert!((corrected[1] - 0.50).abs() < 1e-9);
        assert!((corrected[2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_panel_contrast_ratio() {
        let mut panel = PanelCalibration::new("P01", 0, 0);
        panel.white_point = ColorMeasurement::new(0.31, 0.33, 500.0);
        panel.black_level = ColorMeasurement::new(0.31, 0.33, 0.5);
        assert!((panel.contrast_ratio() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_panel_correction_clamping() {
        let mut panel = PanelCalibration::new("P02", 0, 1);
        panel.gain = [2.0, 2.0, 2.0];
        let corrected = panel.correct([0.8, 0.9, 1.0]);
        assert!((corrected[0] - 1.0).abs() < 1e-9);
        assert!((corrected[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_calibrator_add_panels() {
        let mut cal = VolumeCalibrator::new(VolumeCalibrationConfig::default());
        cal.add_panel(PanelCalibration::new("A1", 0, 0));
        cal.add_panel(PanelCalibration::new("A2", 0, 1));
        assert_eq!(cal.panel_count(), 2);
        assert!(cal.get_panel("A1").is_some());
    }

    #[test]
    fn test_calibrator_evaluate_passing_panel() {
        let mut cal = VolumeCalibrator::new(VolumeCalibrationConfig::default());
        let mut panel = PanelCalibration::new("P1", 0, 0);
        panel.white_point = ColorMeasurement::d65();
        cal.add_panel(panel);
        cal.start();
        let pass = cal.evaluate_panel("P1");
        assert!(pass);
        assert_eq!(cal.stats().panels_passed, 1);
    }

    #[test]
    fn test_calibrator_evaluate_failing_panel() {
        let mut cal = VolumeCalibrator::new(VolumeCalibrationConfig::default());
        let mut panel = PanelCalibration::new("P1", 0, 0);
        panel.white_point = ColorMeasurement::new(0.4, 0.4, 50.0);
        cal.add_panel(panel);
        cal.start();
        let pass = cal.evaluate_panel("P1");
        assert!(!pass);
        assert_eq!(cal.stats().panels_failed, 1);
    }

    #[test]
    fn test_calibrator_finish_all_passed() {
        let mut cal = VolumeCalibrator::new(VolumeCalibrationConfig::default());
        cal.add_panel(PanelCalibration::new("P1", 0, 0));
        cal.start();
        cal.evaluate_panel("P1");
        cal.finish();
        assert_eq!(cal.status(), CalibrationStatus::Passed);
    }

    #[test]
    fn test_calibration_stats_pass_rate() {
        let mut stats = CalibrationStats::new();
        stats.panels_total = 10;
        stats.panels_passed = 8;
        stats.panels_failed = 2;
        assert!((stats.pass_rate() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_default_config() {
        let cfg = VolumeCalibrationConfig::default();
        assert_eq!(cfg.target, CalibrationTarget::Full);
        assert!((cfg.max_chroma_drift - 0.005).abs() < f64::EPSILON);
        assert_eq!(cfg.samples_per_panel, 5);
    }

    // --- Linear regression tests ---

    #[test]
    fn test_linear_regression_perfect_fit() {
        // y = 2x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let fit = linear_regression(&x, &y);
        assert!((fit.slope - 2.0).abs() < 1e-10, "slope: {}", fit.slope);
        assert!(
            (fit.intercept - 1.0).abs() < 1e-10,
            "intercept: {}",
            fit.intercept
        );
        assert!((fit.r_squared - 1.0).abs() < 1e-10, "r²: {}", fit.r_squared);
    }

    #[test]
    fn test_linear_regression_constant() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0];
        let fit = linear_regression(&x, &y);
        assert!(fit.slope.abs() < 1e-10, "slope should be ~0: {}", fit.slope);
        assert!((fit.intercept - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_single_point() {
        let x = vec![3.0];
        let y = vec![7.0];
        let fit = linear_regression(&x, &y);
        assert_eq!(fit.slope, 0.0);
        assert!((fit.intercept - 7.0).abs() < 1e-10);
    }

    // --- Panel drift tracker tests ---

    #[test]
    fn test_panel_drift_tracker_creation() {
        let tracker = PanelDriftTracker::new("P1", 64);
        assert_eq!(tracker.panel_id, "P1");
        assert_eq!(tracker.sample_count(), 0);
        assert!(tracker.luma_fit().is_none());
    }

    #[test]
    fn test_panel_drift_tracker_record_and_fit() {
        let mut tracker = PanelDriftTracker::new("P1", 64);

        // Simulate: as temperature rises, luminance drops linearly
        for i in 0..10 {
            let temp = 25.0 + i as f64 * 2.0; // 25°C to 43°C
            let luma = 100.0 - 0.3 * (temp - 25.0); // drops 0.3 cd/m² per °C
            tracker.record(TimestampedMeasurement {
                elapsed_secs: i as f64 * 60.0,
                temperature_c: temp,
                measurement: ColorMeasurement::new(0.3127, 0.3290, luma),
            });
        }

        tracker.fit_drift_model();
        let luma_fit = tracker.luma_fit().expect("should have fit");
        assert!(
            (luma_fit.slope - (-0.3)).abs() < 0.01,
            "luma slope: {}",
            luma_fit.slope
        );
        assert!(luma_fit.r_squared > 0.99, "r²: {}", luma_fit.r_squared);
    }

    #[test]
    fn test_panel_drift_prediction() {
        let mut tracker = PanelDriftTracker::new("P2", 64);

        // Linear drift in CIE x with temperature
        for i in 0..10 {
            let temp = 25.0 + i as f64;
            tracker.record(TimestampedMeasurement {
                elapsed_secs: i as f64 * 60.0,
                temperature_c: temp,
                measurement: ColorMeasurement::new(0.3127 + 0.0001 * (temp - 25.0), 0.3290, 100.0),
            });
        }

        tracker.fit_drift_model();
        let predicted = tracker
            .predict_at_temperature(35.0)
            .expect("should predict");
        // At 35°C, x should be ~0.3127 + 0.001
        assert!(
            (predicted.x - 0.3137).abs() < 0.001,
            "predicted x: {}",
            predicted.x
        );
    }

    #[test]
    fn test_panel_drift_correction_gain() {
        let mut tracker = PanelDriftTracker::new("P3", 64);

        // Panel gets warmer and luminance drops
        for i in 0..10 {
            let temp = 25.0 + i as f64 * 2.0;
            let luma = 100.0 - 0.5 * (temp - 25.0);
            tracker.record(TimestampedMeasurement {
                elapsed_secs: i as f64 * 60.0,
                temperature_c: temp,
                measurement: ColorMeasurement::new(0.3127, 0.3290, luma),
            });
        }

        tracker.fit_drift_model();
        let reference = ColorMeasurement::d65();
        let gain = tracker
            .correction_gain(35.0, &reference)
            .expect("should compute gain");

        // At 35°C, luminance predicted ~95, reference 100 => gain ~1.05
        assert!(gain[0] > 1.0, "r gain should compensate: {}", gain[0]);
        assert!(gain[1] > 1.0, "g gain should compensate: {}", gain[1]);
        assert!(gain[2] > 1.0, "b gain should compensate: {}", gain[2]);
    }

    #[test]
    fn test_panel_drift_no_fit_returns_none() {
        let tracker = PanelDriftTracker::new("P4", 64);
        assert!(tracker.predict_at_temperature(30.0).is_none());
        assert!(tracker
            .correction_gain(30.0, &ColorMeasurement::d65())
            .is_none());
    }

    // --- Thermal drift compensator tests ---

    #[test]
    fn test_thermal_compensator_creation() {
        let comp = ThermalDriftCompensator::new(ThermalDriftConfig::default());
        assert_eq!(comp.panel_count(), 0);
    }

    #[test]
    fn test_thermal_compensator_record_and_track() {
        let mut comp = ThermalDriftCompensator::new(ThermalDriftConfig {
            min_samples_for_estimation: 3,
            ..ThermalDriftConfig::default()
        });

        for i in 0..5 {
            let temp = 25.0 + i as f64;
            comp.record(
                "P1",
                i as f64 * 60.0,
                temp,
                ColorMeasurement::new(0.3127, 0.3290, 100.0 - 0.2 * (temp - 25.0)),
            );
        }

        assert_eq!(comp.panel_count(), 1);
        assert!(comp.tracker("P1").is_some());

        let gain = comp.correction_gain("P1", 30.0);
        assert!(gain.is_some());
    }

    #[test]
    fn test_thermal_compensator_multiple_panels() {
        let mut comp = ThermalDriftCompensator::new(ThermalDriftConfig {
            min_samples_for_estimation: 2,
            ..ThermalDriftConfig::default()
        });

        for i in 0..5 {
            let temp = 25.0 + i as f64;
            comp.record("A1", i as f64 * 60.0, temp, ColorMeasurement::d65());
            comp.record("A2", i as f64 * 60.0, temp, ColorMeasurement::d65());
            comp.record("B1", i as f64 * 60.0, temp, ColorMeasurement::d65());
        }

        assert_eq!(comp.panel_count(), 3);
    }

    #[test]
    fn test_thermal_compensator_no_data_returns_none() {
        let comp = ThermalDriftCompensator::new(ThermalDriftConfig::default());
        assert!(comp.correction_gain("nonexistent", 30.0).is_none());
    }

    #[test]
    fn test_drift_excessive_detection() {
        let mut comp = ThermalDriftCompensator::new(ThermalDriftConfig {
            min_samples_for_estimation: 3,
            chroma_drift_per_degree: 0.0001,
            reference_temperature_c: 25.0,
            ..ThermalDriftConfig::default()
        });

        // Record measurements with excessive chromaticity drift
        for i in 0..5 {
            let temp = 25.0 + i as f64 * 5.0;
            comp.record(
                "P1",
                i as f64 * 60.0,
                temp,
                ColorMeasurement::new(
                    0.3127 + 0.005 * (temp - 25.0), // very large drift
                    0.3290,
                    100.0,
                ),
            );
        }

        assert!(comp.is_drift_excessive("P1", 45.0));
    }

    #[test]
    fn test_drift_not_excessive_for_normal_panel() {
        let mut comp = ThermalDriftCompensator::new(ThermalDriftConfig {
            min_samples_for_estimation: 3,
            ..ThermalDriftConfig::default()
        });

        // Record stable measurements (no drift)
        for i in 0..5 {
            let temp = 25.0 + i as f64;
            comp.record("P1", i as f64 * 60.0, temp, ColorMeasurement::d65());
        }

        assert!(!comp.is_drift_excessive("P1", 27.0));
    }

    #[test]
    fn test_panel_drift_sample_trimming() {
        let mut tracker = PanelDriftTracker::new("P1", 4);
        for i in 0..10 {
            tracker.record(TimestampedMeasurement {
                elapsed_secs: i as f64,
                temperature_c: 25.0 + i as f64,
                measurement: ColorMeasurement::d65(),
            });
        }
        assert_eq!(tracker.sample_count(), 4);
    }
}
