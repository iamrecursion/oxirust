//! Digital peak meters (dBFS and RMS).
//!
//! Provides sample-accurate peak detection and RMS level measurement.

use super::ballistics::{linear_to_db, OverloadDetector, PeakDetector, RmsWindow};
use crate::frame::AudioFrame;

/// Digital peak meter (dBFS).
///
/// Sample-accurate peak detection with configurable hold time.
pub struct DigitalPeakMeter {
    /// Peak detectors (one per channel).
    peak_detectors: Vec<PeakDetector>,
    /// Overload detectors (one per channel).
    overload_detectors: Vec<OverloadDetector>,
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: f64,
    /// Number of channels.
    #[allow(dead_code)]
    channels: usize,
    /// Current peak readings per channel (dBFS).
    peak_readings: Vec<f64>,
    /// Maximum peaks per channel (dBFS).
    max_peaks: Vec<f64>,
    /// Peak hold time in seconds.
    #[allow(dead_code)]
    peak_hold_time: f64,
    /// Overload threshold in dBFS.
    #[allow(dead_code)]
    overload_threshold: f64,
}

impl DigitalPeakMeter {
    /// Create a new digital peak meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `peak_hold_seconds` - Peak hold time in seconds
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize, peak_hold_seconds: f64) -> Self {
        Self::with_threshold(sample_rate, channels, peak_hold_seconds, -0.1)
    }

    /// Create a digital peak meter with custom overload threshold.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `peak_hold_seconds` - Peak hold time in seconds
    /// * `overload_threshold_dbfs` - Overload threshold in dBFS
    #[must_use]
    pub fn with_threshold(
        sample_rate: f64,
        channels: usize,
        peak_hold_seconds: f64,
        overload_threshold_dbfs: f64,
    ) -> Self {
        let peak_detectors = (0..channels)
            .map(|_| PeakDetector::new(peak_hold_seconds, 0.3, sample_rate))
            .collect();

        let overload_detectors = (0..channels)
            .map(|_| OverloadDetector::new(overload_threshold_dbfs, 1.0, 1000.0, sample_rate))
            .collect();

        Self {
            peak_detectors,
            overload_detectors,
            sample_rate,
            channels,
            peak_readings: vec![f64::NEG_INFINITY; channels],
            max_peaks: vec![f64::NEG_INFINITY; channels],
            peak_hold_time: peak_hold_seconds,
            overload_threshold: overload_threshold_dbfs,
        }
    }

    /// Process an audio frame and update peak readings.
    ///
    /// # Arguments
    ///
    /// * `frame` - Audio frame to process
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        let num_samples = samples.len() / self.channels;

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if let Some(&sample) = samples.get(idx) {
                    let abs_sample = sample.abs();

                    // Update peak detector
                    let peak = self.peak_detectors[ch].process(abs_sample);

                    // Update overload detector
                    self.overload_detectors[ch].process(abs_sample);

                    // Convert to dBFS
                    let db_fs = linear_to_db(peak);

                    // Update readings
                    self.peak_readings[ch] = db_fs;
                    if db_fs > self.max_peaks[ch] {
                        self.max_peaks[ch] = db_fs;
                    }
                }
            }
        }
    }

    /// Get current peak reading for a channel in dBFS.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn peak_dbfs(&self, channel: usize) -> f64 {
        self.peak_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get maximum peak reading for a channel in dBFS.
    #[must_use]
    pub fn max_peak_dbfs(&self, channel: usize) -> f64 {
        self.max_peaks
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get stereo peak reading (max of L/R).
    #[must_use]
    pub fn stereo_peak_dbfs(&self) -> f64 {
        if self.channels == 1 {
            self.peak_dbfs(0)
        } else if self.channels >= 2 {
            self.peak_dbfs(0).max(self.peak_dbfs(1))
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get normalized peak reading (0.0 to 1.0) for visualization.
    ///
    /// Maps -60 dBFS to 0.0 and 0 dBFS to 1.0.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn normalized_reading(&self, channel: usize) -> f64 {
        let db_fs = self.peak_dbfs(channel);
        normalize_dbfs(db_fs, -60.0, 0.0)
    }

    /// Get normalized max peak.
    #[must_use]
    pub fn normalized_max_peak(&self, channel: usize) -> f64 {
        let db_fs = self.max_peak_dbfs(channel);
        normalize_dbfs(db_fs, -60.0, 0.0)
    }

    /// Check if channel is in overload.
    #[must_use]
    pub fn is_overload(&self, channel: usize) -> bool {
        self.overload_detectors
            .get(channel)
            .map_or(false, OverloadDetector::is_overload)
    }

    /// Get visualization data for a channel.
    #[must_use]
    pub fn visualization_data(&self, channel: usize) -> PeakVisualization {
        let peak = self.peak_dbfs(channel);
        let normalized = self.normalized_reading(channel);
        let max_peak = self.max_peak_dbfs(channel);
        let normalized_max = self.normalized_max_peak(channel);

        PeakVisualization {
            peak_dbfs: peak,
            normalized,
            max_peak_dbfs: max_peak,
            normalized_max,
            overload: self.is_overload(channel),
            color_zone: get_dbfs_color_zone(peak),
        }
    }

    /// Get all channels visualization data.
    #[must_use]
    pub fn all_channels_visualization(&self) -> Vec<PeakVisualization> {
        (0..self.channels)
            .map(|ch| self.visualization_data(ch))
            .collect()
    }

    /// Reset all peak readings.
    pub fn reset(&mut self) {
        for detector in &mut self.peak_detectors {
            detector.reset();
        }
        for detector in &mut self.overload_detectors {
            detector.reset();
        }
        self.peak_readings.fill(f64::NEG_INFINITY);
        self.max_peaks.fill(f64::NEG_INFINITY);
    }

    /// Reset peak hold only.
    pub fn reset_peak_hold(&mut self) {
        for detector in &mut self.peak_detectors {
            detector.reset();
        }
    }

    /// Reset max peaks only.
    pub fn reset_max_peaks(&mut self) {
        self.max_peaks.fill(f64::NEG_INFINITY);
    }

    /// Reset overload indicators.
    pub fn reset_overload(&mut self) {
        for detector in &mut self.overload_detectors {
            detector.reset();
        }
    }
}

/// RMS (Root Mean Square) level meter.
///
/// Measures average signal level over a time window.
pub struct RmsLevelMeter {
    /// RMS windows (one per channel).
    rms_windows: Vec<RmsWindow>,
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: f64,
    /// Number of channels.
    #[allow(dead_code)]
    channels: usize,
    /// Integration time in seconds.
    integration_time: f64,
    /// Current RMS readings per channel (dBFS).
    rms_readings: Vec<f64>,
    /// Maximum RMS readings per channel (dBFS).
    max_rms_readings: Vec<f64>,
}

impl RmsLevelMeter {
    /// Create a new RMS level meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `integration_time` - RMS window duration in seconds
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize, integration_time: f64) -> Self {
        let rms_windows = (0..channels)
            .map(|_| RmsWindow::new(integration_time, sample_rate))
            .collect();

        Self {
            rms_windows,
            sample_rate,
            channels,
            integration_time,
            rms_readings: vec![f64::NEG_INFINITY; channels],
            max_rms_readings: vec![f64::NEG_INFINITY; channels],
        }
    }

    /// Process an audio frame and update RMS readings.
    ///
    /// # Arguments
    ///
    /// * `frame` - Audio frame to process
    pub fn process(&mut self, frame: &AudioFrame) {
        let samples = extract_samples_f64(frame);
        let num_samples = samples.len() / self.channels;

        for i in 0..num_samples {
            for ch in 0..self.channels {
                let idx = i * self.channels + ch;
                if let Some(&sample) = samples.get(idx) {
                    // Compute RMS
                    let rms = self.rms_windows[ch].process(sample);

                    // Convert to dBFS
                    let db_fs = linear_to_db(rms);

                    // Update readings
                    self.rms_readings[ch] = db_fs;
                    if db_fs > self.max_rms_readings[ch] {
                        self.max_rms_readings[ch] = db_fs;
                    }
                }
            }
        }
    }

    /// Get current RMS reading for a channel in dBFS.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn rms_dbfs(&self, channel: usize) -> f64 {
        self.rms_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get maximum RMS reading for a channel in dBFS.
    #[must_use]
    pub fn max_rms_dbfs(&self, channel: usize) -> f64 {
        self.max_rms_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get stereo RMS reading (average of L/R).
    #[must_use]
    pub fn stereo_rms_dbfs(&self) -> f64 {
        if self.channels == 1 {
            self.rms_dbfs(0)
        } else if self.channels >= 2 {
            let left = self.rms_dbfs(0);
            let right = self.rms_dbfs(1);
            if left.is_finite() && right.is_finite() {
                // Average in linear domain, then convert to dB
                let left_lin = super::ballistics::db_to_linear(left);
                let right_lin = super::ballistics::db_to_linear(right);
                linear_to_db((left_lin + right_lin) / 2.0)
            } else if left.is_finite() {
                left
            } else {
                right
            }
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get normalized RMS reading (0.0 to 1.0) for visualization.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn normalized_reading(&self, channel: usize) -> f64 {
        let db_fs = self.rms_dbfs(channel);
        normalize_dbfs(db_fs, -60.0, 0.0)
    }

    /// Get visualization data for a channel.
    #[must_use]
    pub fn visualization_data(&self, channel: usize) -> RmsVisualization {
        let rms = self.rms_dbfs(channel);
        let normalized = self.normalized_reading(channel);
        let max_rms = self.max_rms_dbfs(channel);

        RmsVisualization {
            rms_dbfs: rms,
            normalized,
            max_rms_dbfs: max_rms,
            color_zone: get_dbfs_color_zone(rms),
        }
    }

    /// Get all channels visualization data.
    #[must_use]
    pub fn all_channels_visualization(&self) -> Vec<RmsVisualization> {
        (0..self.channels)
            .map(|ch| self.visualization_data(ch))
            .collect()
    }

    /// Reset all RMS readings.
    pub fn reset(&mut self) {
        for window in &mut self.rms_windows {
            window.reset();
        }
        self.rms_readings.fill(f64::NEG_INFINITY);
        self.max_rms_readings.fill(f64::NEG_INFINITY);
    }

    /// Reset max RMS readings only.
    pub fn reset_max(&mut self) {
        self.max_rms_readings.fill(f64::NEG_INFINITY);
    }

    /// Get integration time.
    #[must_use]
    pub fn integration_time(&self) -> f64 {
        self.integration_time
    }
}

/// Peak meter visualization data.
#[derive(Clone, Debug)]
pub struct PeakVisualization {
    /// Peak reading in dBFS.
    pub peak_dbfs: f64,
    /// Normalized value (0.0 to 1.0).
    pub normalized: f64,
    /// Maximum peak in dBFS.
    pub max_peak_dbfs: f64,
    /// Normalized maximum peak.
    pub normalized_max: f64,
    /// Overload indicator.
    pub overload: bool,
    /// Color zone.
    pub color_zone: ColorZone,
}

impl PeakVisualization {
    /// Get scale markings for dBFS display.
    #[must_use]
    pub fn scale_markings() -> Vec<(f64, String)> {
        vec![
            (-60.0, "-60".to_string()),
            (-48.0, "-48".to_string()),
            (-36.0, "-36".to_string()),
            (-24.0, "-24".to_string()),
            (-18.0, "-18".to_string()),
            (-12.0, "-12".to_string()),
            (-9.0, "-9".to_string()),
            (-6.0, "-6".to_string()),
            (-3.0, "-3".to_string()),
            (0.0, "0".to_string()),
        ]
    }
}

/// RMS meter visualization data.
#[derive(Clone, Debug)]
pub struct RmsVisualization {
    /// RMS reading in dBFS.
    pub rms_dbfs: f64,
    /// Normalized value (0.0 to 1.0).
    pub normalized: f64,
    /// Maximum RMS in dBFS.
    pub max_rms_dbfs: f64,
    /// Color zone.
    pub color_zone: ColorZone,
}

/// Color zone for meter display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorZone {
    /// Green zone (safe operation).
    Green,
    /// Yellow zone (approaching limits).
    Yellow,
    /// Red zone (overload).
    Red,
}

/// Normalize dBFS reading to 0.0-1.0 range.
#[must_use]
fn normalize_dbfs(db_fs: f64, min_db: f64, max_db: f64) -> f64 {
    if db_fs.is_infinite() && db_fs.is_sign_negative() {
        0.0
    } else {
        ((db_fs - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
    }
}

/// Get color zone for dBFS reading.
#[must_use]
fn get_dbfs_color_zone(db_fs: f64) -> ColorZone {
    if db_fs > -3.0 {
        ColorZone::Red
    } else if db_fs > -9.0 {
        ColorZone::Yellow
    } else {
        ColorZone::Green
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
