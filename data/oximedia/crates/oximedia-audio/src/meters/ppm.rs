//! PPM (Peak Programme Meter) implementations.
//!
//! Supports multiple PPM standards:
//! - BBC PPM (BS.6840)
//! - EBU PPM (IEC 60268-10 Type IIa)
//! - Nordic PPM (NRK/SR/DR/YLE)
//! - DIN PPM (IEC 60268-10 Type I)

use super::ballistics::{linear_to_db, BallisticsConfig, BallisticsProcessor, PeakDetector};
use crate::frame::AudioFrame;

/// PPM standard type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PpmStandard {
    /// BBC PPM (BS.6840).
    ///
    /// - Scale: PPM 1-7 (PPM 4 = 0 dBu)
    /// - Attack: 10ms (fast)
    /// - Decay: 2.8s (slow)
    /// - Peak hold: 1.0s
    Bbc,

    /// EBU PPM (IEC 60268-10 Type IIa).
    ///
    /// - Scale: -12 to +12 dBu (0 dBu = test level)
    /// - Attack: 5ms
    /// - Decay: 1.7s (20dB in 1.7s)
    #[default]
    Ebu,

    /// Nordic PPM (NRK/SR/DR/YLE).
    ///
    /// - Scale: -36 to +5 dBu
    /// - Attack: 5ms
    /// - Decay: 1.5s
    Nordic,

    /// DIN PPM (IEC 60268-10 Type I).
    ///
    /// - Scale: -50 to +5 dBu
    /// - Attack: 10ms (quasi-instantaneous)
    /// - Decay: 1.5s (20dB in 1.5s)
    Din,
}

impl PpmStandard {
    /// Get standard name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Bbc => "BBC PPM (BS.6840)",
            Self::Ebu => "EBU PPM (IEC 60268-10 Type IIa)",
            Self::Nordic => "Nordic PPM",
            Self::Din => "DIN PPM (IEC 60268-10 Type I)",
        }
    }

    /// Get minimum scale value in dB.
    #[must_use]
    pub fn min_db(&self) -> f64 {
        match self {
            Self::Bbc => -24.0, // PPM 1
            Self::Ebu => -12.0,
            Self::Nordic => -36.0,
            Self::Din => -50.0,
        }
    }

    /// Get maximum scale value in dB.
    #[must_use]
    pub fn max_db(&self) -> f64 {
        match self {
            Self::Bbc => 12.0, // PPM 7
            Self::Ebu => 12.0,
            Self::Nordic => 5.0,
            Self::Din => 5.0,
        }
    }

    /// Get reference level in dBFS (0 dB on meter = this dBFS).
    #[must_use]
    pub fn reference_dbfs(&self) -> f64 {
        match self {
            Self::Bbc => -18.0, // PPM 4 = -18 dBFS
            Self::Ebu => -18.0, // 0 dB = -18 dBFS (EBU R68)
            Self::Nordic => -18.0,
            Self::Din => -9.0, // DIN uses higher reference
        }
    }

    /// Get ballistics configuration.
    #[must_use]
    pub fn ballistics(&self, sample_rate: f64) -> BallisticsConfig {
        match self {
            Self::Bbc => BallisticsConfig::bbc_ppm(sample_rate),
            Self::Ebu => BallisticsConfig::ebu_ppm(sample_rate),
            Self::Nordic => BallisticsConfig::nordic_ppm(sample_rate),
            Self::Din => BallisticsConfig::din_ppm(sample_rate),
        }
    }

    /// Convert dB to PPM number (BBC only).
    #[must_use]
    pub fn db_to_ppm(&self, db: f64) -> f64 {
        match self {
            Self::Bbc => {
                // PPM scale: 1-7, where PPM 4 = 0 dB
                // Each PPM is 4 dB
                4.0 + (db / 4.0)
            }
            _ => db, // Other standards use dB directly
        }
    }

    /// Convert PPM number to dB (BBC only).
    #[must_use]
    pub fn ppm_to_db(&self, ppm: f64) -> f64 {
        match self {
            Self::Bbc => (ppm - 4.0) * 4.0,
            _ => ppm,
        }
    }
}

/// PPM (Peak Programme Meter).
///
/// Implements various PPM standards with proper ballistics.
pub struct PpmMeter {
    /// PPM standard.
    standard: PpmStandard,
    /// Ballistics processors (one per channel).
    processors: Vec<BallisticsProcessor>,
    /// Peak detectors (one per channel).
    peak_detectors: Vec<PeakDetector>,
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
    /// Current PPM readings per channel (dB or PPM units).
    ppm_readings: Vec<f64>,
    /// Peak readings per channel.
    peak_readings: Vec<f64>,
}

impl PpmMeter {
    /// Create a new PPM meter.
    ///
    /// # Arguments
    ///
    /// * `standard` - PPM standard to use
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(standard: PpmStandard, sample_rate: f64, channels: usize) -> Self {
        let config = standard.ballistics(sample_rate);
        let processors = (0..channels)
            .map(|_| BallisticsProcessor::new(config.clone()))
            .collect();

        let peak_detectors = (0..channels)
            .map(|_| PeakDetector::new(1.0, 0.0, sample_rate))
            .collect();

        Self {
            standard,
            processors,
            peak_detectors,
            sample_rate,
            channels,
            ppm_readings: vec![f64::NEG_INFINITY; channels],
            peak_readings: vec![f64::NEG_INFINITY; channels],
        }
    }

    /// Process an audio frame and update PPM readings.
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

                    // Detect peak
                    let peak = self.peak_detectors[ch].process(abs_sample);

                    // Convert to dBFS
                    let db_fs = linear_to_db(peak);

                    // Convert to PPM scale (relative to reference)
                    let db_relative = db_fs - self.standard.reference_dbfs();

                    // Apply ballistics
                    let reading = self.processors[ch].process(db_relative);

                    // Update readings
                    self.ppm_readings[ch] = reading;
                    self.peak_readings[ch] = self.peak_readings[ch].max(reading);
                }
            }
        }
    }

    /// Get current PPM reading for a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    ///
    /// # Returns
    ///
    /// PPM reading in dB (or PPM units for BBC)
    #[must_use]
    pub fn ppm_reading(&self, channel: usize) -> f64 {
        self.ppm_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get PPM reading in PPM units (BBC standard only).
    #[must_use]
    pub fn ppm_units(&self, channel: usize) -> f64 {
        let db = self.ppm_reading(channel);
        self.standard.db_to_ppm(db)
    }

    /// Get peak PPM reading for a channel.
    #[must_use]
    pub fn peak_reading(&self, channel: usize) -> f64 {
        self.peak_readings
            .get(channel)
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Get stereo PPM reading (max of L/R for stereo, or mono for mono).
    #[must_use]
    pub fn stereo_ppm_reading(&self) -> f64 {
        if self.channels == 1 {
            self.ppm_reading(0)
        } else if self.channels >= 2 {
            let left = self.ppm_reading(0);
            let right = self.ppm_reading(1);
            left.max(right)
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get normalized PPM reading (0.0 to 1.0) for visualization.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index
    #[must_use]
    pub fn normalized_reading(&self, channel: usize) -> f64 {
        let reading = self.ppm_reading(channel);
        normalize_ppm(reading, self.standard.min_db(), self.standard.max_db())
    }

    /// Get peak normalized PPM reading.
    #[must_use]
    pub fn normalized_peak(&self, channel: usize) -> f64 {
        let peak = self.peak_reading(channel);
        normalize_ppm(peak, self.standard.min_db(), self.standard.max_db())
    }

    /// Check if channel is in overload (> permitted maximum).
    #[must_use]
    pub fn is_overload(&self, channel: usize) -> bool {
        let reading = self.ppm_reading(channel);
        match self.standard {
            PpmStandard::Bbc => reading > 8.0, // > PPM 6
            _ => reading > 9.0,                // > +9 dB
        }
    }

    /// Get visualization data for a channel.
    #[must_use]
    pub fn visualization_data(&self, channel: usize) -> PpmVisualization {
        let reading = self.ppm_reading(channel);
        let normalized = self.normalized_reading(channel);
        let peak = self.peak_reading(channel);
        let normalized_peak = self.normalized_peak(channel);

        PpmVisualization {
            db_reading: reading,
            ppm_units: self.standard.db_to_ppm(reading),
            normalized,
            peak_db: peak,
            normalized_peak,
            overload: self.is_overload(channel),
            color_zone: get_ppm_color_zone(reading, &self.standard),
            standard: self.standard,
        }
    }

    /// Get all channels visualization data.
    #[must_use]
    pub fn all_channels_visualization(&self) -> Vec<PpmVisualization> {
        (0..self.channels)
            .map(|ch| self.visualization_data(ch))
            .collect()
    }

    /// Reset all PPM readings.
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
        for detector in &mut self.peak_detectors {
            detector.reset();
        }
        self.ppm_readings.fill(f64::NEG_INFINITY);
        self.peak_readings.fill(f64::NEG_INFINITY);
    }

    /// Reset peak readings only.
    pub fn reset_peaks(&mut self) {
        for processor in &mut self.processors {
            processor.reset_peak_hold();
        }
        for detector in &mut self.peak_detectors {
            detector.reset();
        }
        self.peak_readings.fill(f64::NEG_INFINITY);
    }

    /// Get PPM standard.
    #[must_use]
    pub fn standard(&self) -> PpmStandard {
        self.standard
    }

    /// Set PPM standard (recreates processors).
    pub fn set_standard(&mut self, standard: PpmStandard) {
        if self.standard != standard {
            self.standard = standard;
            let config = standard.ballistics(self.sample_rate);
            self.processors = (0..self.channels)
                .map(|_| BallisticsProcessor::new(config.clone()))
                .collect();
            self.reset();
        }
    }
}

/// PPM visualization data.
#[derive(Clone, Debug)]
pub struct PpmVisualization {
    /// PPM reading in dB.
    pub db_reading: f64,
    /// PPM reading in PPM units (BBC) or dB (others).
    pub ppm_units: f64,
    /// Normalized value (0.0 to 1.0).
    pub normalized: f64,
    /// Peak reading in dB.
    pub peak_db: f64,
    /// Normalized peak value (0.0 to 1.0).
    pub normalized_peak: f64,
    /// Overload indicator.
    pub overload: bool,
    /// Color zone for the current reading.
    pub color_zone: ColorZone,
    /// PPM standard used.
    pub standard: PpmStandard,
}

impl PpmVisualization {
    /// Get scale markings for PPM display.
    #[must_use]
    pub fn scale_markings(&self) -> Vec<(f64, String)> {
        match self.standard {
            PpmStandard::Bbc => vec![
                (1.0, "1".to_string()),
                (2.0, "2".to_string()),
                (3.0, "3".to_string()),
                (4.0, "4".to_string()),
                (5.0, "5".to_string()),
                (6.0, "6".to_string()),
                (7.0, "7".to_string()),
            ],
            PpmStandard::Ebu => vec![
                (-12.0, "-12".to_string()),
                (-9.0, "-9".to_string()),
                (-6.0, "-6".to_string()),
                (-3.0, "-3".to_string()),
                (0.0, "0".to_string()),
                (3.0, "+3".to_string()),
                (6.0, "+6".to_string()),
                (9.0, "+9".to_string()),
            ],
            PpmStandard::Nordic => vec![
                (-36.0, "-36".to_string()),
                (-24.0, "-24".to_string()),
                (-18.0, "-18".to_string()),
                (-12.0, "-12".to_string()),
                (-6.0, "-6".to_string()),
                (0.0, "0".to_string()),
                (5.0, "+5".to_string()),
            ],
            PpmStandard::Din => vec![
                (-50.0, "-50".to_string()),
                (-40.0, "-40".to_string()),
                (-30.0, "-30".to_string()),
                (-20.0, "-20".to_string()),
                (-10.0, "-10".to_string()),
                (0.0, "0".to_string()),
                (5.0, "+5".to_string()),
            ],
        }
    }
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

/// Normalize PPM reading to 0.0-1.0 range.
#[must_use]
fn normalize_ppm(db: f64, min_db: f64, max_db: f64) -> f64 {
    if db.is_infinite() && db.is_sign_negative() {
        0.0
    } else {
        ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
    }
}

/// Get color zone for PPM reading.
#[must_use]
fn get_ppm_color_zone(db: f64, standard: &PpmStandard) -> ColorZone {
    match standard {
        PpmStandard::Bbc => {
            let ppm = standard.db_to_ppm(db);
            if ppm > 6.0 {
                ColorZone::Red
            } else if ppm > 5.0 {
                ColorZone::Yellow
            } else {
                ColorZone::Green
            }
        }
        _ => {
            if db > 9.0 {
                ColorZone::Red
            } else if db > 6.0 {
                ColorZone::Yellow
            } else {
                ColorZone::Green
            }
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
