//! Stereo correlation and phase meters.
//!
//! Provides correlation coefficient measurement and goniometer visualization.

use crate::frame::AudioFrame;
use std::collections::VecDeque;

/// Stereo correlation meter.
///
/// Measures the phase relationship between stereo channels:
/// - +1.0: Perfect in-phase (mono)
/// - 0.0: Uncorrelated (wide stereo)
/// - -1.0: Perfect out-of-phase (phase issues)
pub struct CorrelationMeter {
    /// Sample buffer for left channel.
    left_buffer: VecDeque<f64>,
    /// Sample buffer for right channel.
    right_buffer: VecDeque<f64>,
    /// Integration window size in samples.
    window_size: usize,
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: f64,
    /// Current correlation coefficient.
    correlation: f64,
    /// Minimum correlation seen.
    min_correlation: f64,
    /// Maximum correlation seen.
    max_correlation: f64,
    /// Running sums for correlation calculation.
    sum_left_right: f64,
    /// Sum of left squared.
    sum_left_squared: f64,
    /// Sum of right squared.
    sum_right_squared: f64,
}

impl CorrelationMeter {
    /// Create a new correlation meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `integration_time` - Integration window in seconds
    #[must_use]
    pub fn new(sample_rate: f64, integration_time: f64) -> Self {
        let window_size = (sample_rate * integration_time) as usize;

        Self {
            left_buffer: VecDeque::with_capacity(window_size),
            right_buffer: VecDeque::with_capacity(window_size),
            window_size,
            sample_rate,
            correlation: 0.0,
            min_correlation: 1.0,
            max_correlation: -1.0,
            sum_left_right: 0.0,
            sum_left_squared: 0.0,
            sum_right_squared: 0.0,
        }
    }

    /// Process an audio frame and update correlation.
    ///
    /// # Arguments
    ///
    /// * `frame` - Stereo audio frame to process
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        let channels = frame.channels.count();

        if channels < 2 {
            // Mono signal is perfectly correlated
            self.correlation = 1.0;
            return;
        }

        let num_frames = samples.len() / channels;

        for i in 0..num_frames {
            let left = samples.get(i * channels).copied().unwrap_or(0.0);
            let right = samples.get(i * channels + 1).copied().unwrap_or(0.0);

            // Add to running sums
            self.sum_left_right += left * right;
            self.sum_left_squared += left * left;
            self.sum_right_squared += right * right;

            self.left_buffer.push_back(left);
            self.right_buffer.push_back(right);

            // Remove oldest samples if window is full
            if self.left_buffer.len() > self.window_size {
                if let (Some(old_left), Some(old_right)) =
                    (self.left_buffer.pop_front(), self.right_buffer.pop_front())
                {
                    self.sum_left_right -= old_left * old_right;
                    self.sum_left_squared -= old_left * old_left;
                    self.sum_right_squared -= old_right * old_right;
                }
            }

            // Calculate correlation coefficient
            if self.left_buffer.len() > 1 {
                let denominator = (self.sum_left_squared * self.sum_right_squared).sqrt();
                if denominator > 1e-10 {
                    self.correlation = self.sum_left_right / denominator;
                    self.correlation = self.correlation.clamp(-1.0, 1.0);

                    // Update min/max
                    self.min_correlation = self.min_correlation.min(self.correlation);
                    self.max_correlation = self.max_correlation.max(self.correlation);
                }
            }
        }
    }

    /// Get current correlation coefficient.
    ///
    /// Returns:
    /// - +1.0: Perfect in-phase
    /// - 0.0: Uncorrelated
    /// - -1.0: Perfect out-of-phase
    #[must_use]
    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    /// Get minimum correlation seen.
    #[must_use]
    pub fn min_correlation(&self) -> f64 {
        self.min_correlation
    }

    /// Get maximum correlation seen.
    #[must_use]
    pub fn max_correlation(&self) -> f64 {
        self.max_correlation
    }

    /// Check if signal has phase issues (correlation < -0.5).
    #[must_use]
    pub fn has_phase_issues(&self) -> bool {
        self.correlation < -0.5
    }

    /// Check if signal is mono (correlation > 0.95).
    #[must_use]
    pub fn is_mono(&self) -> bool {
        self.correlation > 0.95
    }

    /// Get stereo width estimate (0.0 = mono, 1.0 = wide stereo).
    #[must_use]
    pub fn stereo_width(&self) -> f64 {
        // Map correlation to width: +1 -> 0, 0 -> 1, -1 -> 1
        if self.correlation > 0.0 {
            1.0 - self.correlation
        } else {
            1.0
        }
    }

    /// Get visualization data.
    #[must_use]
    pub fn visualization_data(&self) -> CorrelationVisualization {
        CorrelationVisualization {
            correlation: self.correlation,
            normalized: (self.correlation + 1.0) / 2.0, // Map -1..1 to 0..1
            min_correlation: self.min_correlation,
            max_correlation: self.max_correlation,
            stereo_width: self.stereo_width(),
            phase_warning: self.has_phase_issues(),
            mono_warning: self.is_mono(),
        }
    }

    /// Reset correlation meter.
    pub fn reset(&mut self) {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.correlation = 0.0;
        self.min_correlation = 1.0;
        self.max_correlation = -1.0;
        self.sum_left_right = 0.0;
        self.sum_left_squared = 0.0;
        self.sum_right_squared = 0.0;
    }

    /// Reset min/max only.
    pub fn reset_extrema(&mut self) {
        self.min_correlation = self.correlation;
        self.max_correlation = self.correlation;
    }
}

/// Goniometer (stereo vectorscope).
///
/// Visualizes stereo field by plotting M (mid) vs S (side) or L vs R.
pub struct Goniometer {
    /// Sample buffer for visualization.
    point_buffer: VecDeque<GonioPoint>,
    /// Maximum buffer size.
    max_points: usize,
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: f64,
    /// Decay factor for point fading.
    decay_factor: f64,
    /// Display mode.
    mode: GoniometerMode,
    /// Stereo width measurement.
    width: f64,
    /// Balance measurement (-1.0 = left, 0.0 = center, 1.0 = right).
    balance: f64,
}

impl Goniometer {
    /// Create a new goniometer.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `max_points` - Maximum number of points to display
    /// * `mode` - Display mode (LR or MS)
    #[must_use]
    pub fn new(sample_rate: f64, max_points: usize, mode: GoniometerMode) -> Self {
        Self {
            point_buffer: VecDeque::with_capacity(max_points),
            max_points,
            sample_rate,
            decay_factor: 0.95,
            mode,
            width: 0.0,
            balance: 0.0,
        }
    }

    /// Process an audio frame and update goniometer.
    ///
    /// # Arguments
    ///
    /// * `frame` - Stereo audio frame to process
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        let channels = frame.channels.count();

        if channels < 2 {
            return;
        }

        let num_frames = samples.len() / channels;
        let mut total_width = 0.0;
        let mut total_balance = 0.0;

        for i in 0..num_frames {
            let left = samples.get(i * channels).copied().unwrap_or(0.0);
            let right = samples.get(i * channels + 1).copied().unwrap_or(0.0);

            let point = match self.mode {
                GoniometerMode::LR => GonioPoint {
                    x: left,
                    y: right,
                    intensity: 1.0,
                },
                GoniometerMode::MS => {
                    // Convert L/R to M/S
                    let mid = (left + right) / 2.0;
                    let side = (left - right) / 2.0;
                    GonioPoint {
                        x: mid,
                        y: side,
                        intensity: 1.0,
                    }
                }
            };

            // Calculate width and balance
            let magnitude = (left * left + right * right).sqrt();
            if magnitude > 1e-10 {
                total_width += (left - right).abs();
                total_balance += (right - left) / magnitude;
            }

            self.point_buffer.push_back(point);

            if self.point_buffer.len() > self.max_points {
                self.point_buffer.pop_front();
            }
        }

        // Update width and balance
        if num_frames > 0 {
            self.width = (total_width / num_frames as f64).min(1.0);
            self.balance = (total_balance / num_frames as f64).clamp(-1.0, 1.0);
        }

        // Apply decay to existing points
        for point in &mut self.point_buffer {
            point.intensity *= self.decay_factor;
        }
    }

    /// Get points for visualization.
    #[must_use]
    pub fn points(&self) -> &VecDeque<GonioPoint> {
        &self.point_buffer
    }

    /// Get stereo width (0.0 = mono, 1.0 = wide).
    #[must_use]
    pub fn width(&self) -> f64 {
        self.width
    }

    /// Get stereo balance (-1.0 = left, 0.0 = center, 1.0 = right).
    #[must_use]
    pub fn balance(&self) -> f64 {
        self.balance
    }

    /// Get display mode.
    #[must_use]
    pub fn mode(&self) -> GoniometerMode {
        self.mode
    }

    /// Set display mode.
    pub fn set_mode(&mut self, mode: GoniometerMode) {
        if self.mode != mode {
            self.mode = mode;
            self.clear();
        }
    }

    /// Get visualization data.
    #[must_use]
    pub fn visualization_data(&self) -> GoniometerVisualization {
        GoniometerVisualization {
            points: self.point_buffer.iter().copied().collect(),
            width: self.width,
            balance: self.balance,
            mode: self.mode,
        }
    }

    /// Clear goniometer display.
    pub fn clear(&mut self) {
        self.point_buffer.clear();
        self.width = 0.0;
        self.balance = 0.0;
    }

    /// Set decay factor (0.0 = instant decay, 1.0 = no decay).
    pub fn set_decay_factor(&mut self, factor: f64) {
        self.decay_factor = factor.clamp(0.0, 1.0);
    }
}

/// Goniometer display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GoniometerMode {
    /// L/R mode (Lissajous figure).
    #[default]
    LR,
    /// M/S mode (Mid/Side).
    MS,
}

/// Point on goniometer display.
#[derive(Clone, Copy, Debug)]
pub struct GonioPoint {
    /// X coordinate (-1.0 to 1.0).
    pub x: f64,
    /// Y coordinate (-1.0 to 1.0).
    pub y: f64,
    /// Point intensity (0.0 to 1.0, for fading).
    pub intensity: f64,
}

/// Correlation meter visualization data.
#[derive(Clone, Debug)]
pub struct CorrelationVisualization {
    /// Correlation coefficient (-1.0 to 1.0).
    pub correlation: f64,
    /// Normalized correlation (0.0 to 1.0).
    pub normalized: f64,
    /// Minimum correlation seen.
    pub min_correlation: f64,
    /// Maximum correlation seen.
    pub max_correlation: f64,
    /// Stereo width (0.0 = mono, 1.0 = wide).
    pub stereo_width: f64,
    /// Phase warning indicator.
    pub phase_warning: bool,
    /// Mono warning indicator.
    pub mono_warning: bool,
}

impl CorrelationVisualization {
    /// Get scale markings for correlation display.
    #[must_use]
    pub fn scale_markings() -> Vec<(f64, String)> {
        vec![
            (-1.0, "-1".to_string()),
            (-0.5, "-0.5".to_string()),
            (0.0, "0".to_string()),
            (0.5, "0.5".to_string()),
            (1.0, "1".to_string()),
        ]
    }

    /// Get correlation status description.
    #[must_use]
    pub fn status_description(&self) -> &str {
        if self.correlation > 0.95 {
            "Mono"
        } else if self.correlation > 0.5 {
            "Narrow Stereo"
        } else if self.correlation > 0.0 {
            "Normal Stereo"
        } else if self.correlation > -0.5 {
            "Wide Stereo"
        } else {
            "Phase Issues"
        }
    }
}

/// Goniometer visualization data.
#[derive(Clone, Debug)]
pub struct GoniometerVisualization {
    /// Points to display.
    pub points: Vec<GonioPoint>,
    /// Stereo width (0.0 = mono, 1.0 = wide).
    pub width: f64,
    /// Stereo balance (-1.0 = left, 0.0 = center, 1.0 = right).
    pub balance: f64,
    /// Display mode.
    pub mode: GoniometerMode,
}

impl GoniometerVisualization {
    /// Get axis labels for current mode.
    #[must_use]
    pub fn axis_labels(&self) -> (&str, &str) {
        match self.mode {
            GoniometerMode::LR => ("Left", "Right"),
            GoniometerMode::MS => ("Mid", "Side"),
        }
    }
}

/// Extract samples from audio frame as f64.
#[allow(dead_code)]
fn extract_samples_f64(frame: &AudioFrame) -> Vec<f64> {
    match &frame.samples {
        crate::frame::AudioBuffer::Interleaved(data) => bytes_to_samples_f64(data),
        crate::frame::AudioBuffer::Planar(planes) => {
            if planes.is_empty() {
                return Vec::new();
            }

            let channels = planes.len();
            let sample_size = std::mem::size_of::<f32>();
            let frames = planes[0].len() / sample_size;
            let mut interleaved = Vec::with_capacity(frames * channels);

            for frame_idx in 0..frames {
                for plane in planes {
                    let samples = bytes_to_samples_f64(plane);
                    if let Some(&sample) = samples.get(frame_idx) {
                        interleaved.push(sample);
                    }
                }
            }

            interleaved
        }
    }
}

/// Convert bytes to f64 samples (assumes f32 format).
fn bytes_to_samples_f64(bytes: &bytes::Bytes) -> Vec<f64> {
    let sample_count = bytes.len() / 4;
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let offset = i * 4;
        if offset + 4 <= bytes.len() {
            let bytes_array = [
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ];
            let sample = f32::from_le_bytes(bytes_array);
            samples.push(f64::from(sample));
        }
    }

    samples
}
