//! Professional audio metering for mixer channels and buses.
//!
//! Provides peak, RMS, VU, LUFS, and phase correlation meters.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Meter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeterType {
    /// Digital peak meter (dBFS).
    Peak,
    /// RMS level meter.
    Rms,
    /// VU meter (IEC 60268-10).
    Vu,
    /// LUFS meter (EBU R128).
    Lufs,
    /// Phase correlation meter.
    PhaseCorrelation,
    /// Spectrum analyzer.
    Spectrum,
}

/// Metering ballistics (response time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeterBallistics {
    /// Fast (100ms integration).
    Fast,
    /// Medium (300ms integration).
    Medium,
    /// Slow (600ms integration).
    Slow,
    /// Custom integration time in milliseconds.
    Custom(u32),
}

impl MeterBallistics {
    /// Get integration time in milliseconds.
    #[must_use]
    pub fn integration_time_ms(&self) -> u32 {
        match self {
            Self::Fast => 100,
            Self::Medium => 300,
            Self::Slow => 600,
            Self::Custom(ms) => *ms,
        }
    }
}

/// Peak meter reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakMeterReading {
    /// Current peak level in dBFS.
    pub current_db: f32,
    /// Maximum peak level in dBFS.
    pub max_db: f32,
    /// Clip detected.
    pub clipped: bool,
    /// Peak hold value in dBFS.
    pub hold_db: f32,
}

impl Default for PeakMeterReading {
    fn default() -> Self {
        Self {
            current_db: -f32::INFINITY,
            max_db: -f32::INFINITY,
            clipped: false,
            hold_db: -f32::INFINITY,
        }
    }
}

/// RMS meter reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsMeterReading {
    /// Current RMS level in dBFS.
    pub current_db: f32,
    /// Maximum RMS level in dBFS.
    pub max_db: f32,
}

impl Default for RmsMeterReading {
    fn default() -> Self {
        Self {
            current_db: -f32::INFINITY,
            max_db: -f32::INFINITY,
        }
    }
}

/// VU meter reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VuMeterReading {
    /// Current VU level (0 VU = reference level).
    pub vu_level: f32,
}

impl Default for VuMeterReading {
    fn default() -> Self {
        Self { vu_level: -20.0 }
    }
}

/// LUFS meter reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LufsMeterReading {
    /// Momentary loudness (400ms).
    pub momentary: f32,
    /// Short-term loudness (3s).
    pub short_term: f32,
    /// Integrated loudness.
    pub integrated: f32,
    /// Loudness range (LRA).
    pub range: f32,
    /// True peak in dBTP.
    pub true_peak_db: f32,
}

impl Default for LufsMeterReading {
    fn default() -> Self {
        Self {
            momentary: -f32::INFINITY,
            short_term: -f32::INFINITY,
            integrated: -f32::INFINITY,
            range: 0.0,
            true_peak_db: -f32::INFINITY,
        }
    }
}

/// Phase correlation reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCorrelationReading {
    /// Current phase correlation (-1.0 to 1.0).
    pub correlation: f32,
    /// Minimum correlation seen.
    pub min_correlation: f32,
}

impl Default for PhaseCorrelationReading {
    fn default() -> Self {
        Self {
            correlation: 0.0,
            min_correlation: 1.0,
        }
    }
}

/// Combined metering data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeteringData {
    /// Peak meter readings (one per channel).
    pub peak: Vec<PeakMeterReading>,
    /// RMS meter readings (one per channel).
    pub rms: Vec<RmsMeterReading>,
    /// VU meter readings (one per channel).
    pub vu: Vec<VuMeterReading>,
    /// LUFS meter reading.
    pub lufs: LufsMeterReading,
    /// Phase correlation (stereo only).
    pub phase_correlation: Option<PhaseCorrelationReading>,
}

impl MeteringData {
    /// Create new metering data for specified channel count.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        Self {
            peak: vec![PeakMeterReading::default(); channels],
            rms: vec![RmsMeterReading::default(); channels],
            vu: vec![VuMeterReading::default(); channels],
            lufs: LufsMeterReading::default(),
            phase_correlation: if channels == 2 {
                Some(PhaseCorrelationReading::default())
            } else {
                None
            },
        }
    }

    /// Reset all meters.
    pub fn reset(&mut self) {
        for peak in &mut self.peak {
            *peak = PeakMeterReading::default();
        }
        for rms in &mut self.rms {
            *rms = RmsMeterReading::default();
        }
        for vu in &mut self.vu {
            *vu = VuMeterReading::default();
        }
        self.lufs = LufsMeterReading::default();
        if let Some(ref mut pc) = self.phase_correlation {
            *pc = PhaseCorrelationReading::default();
        }
    }
}

/// Professional audio meter.
#[derive(Debug, Clone)]
pub struct Meter {
    /// Number of channels.
    channels: usize,

    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: u32,

    /// Metering ballistics.
    #[allow(dead_code)]
    ballistics: MeterBallistics,

    /// Peak detectors (one per channel).
    peak_detectors: Vec<PeakDetector>,

    /// RMS calculators (one per channel).
    rms_calculators: Vec<RmsCalculator>,

    /// VU meters (one per channel).
    vu_meters: Vec<VuMeter>,

    /// Phase correlation meter (for stereo).
    #[allow(clippy::struct_field_names)]
    phase_meter: Option<PhaseCorrelationMeter>,

    /// Current metering data.
    data: MeteringData,
}

impl Meter {
    /// Create a new meter.
    #[must_use]
    pub fn new(channels: usize, sample_rate: u32, ballistics: MeterBallistics) -> Self {
        let peak_detectors = (0..channels)
            .map(|_| PeakDetector::new(sample_rate, 1.0))
            .collect();

        let integration_time = ballistics.integration_time_ms();
        let rms_calculators = (0..channels)
            .map(|_| RmsCalculator::new(sample_rate, integration_time))
            .collect();

        let vu_meters = (0..channels).map(|_| VuMeter::new(sample_rate)).collect();

        let phase_meter = if channels == 2 {
            Some(PhaseCorrelationMeter::new(sample_rate))
        } else {
            None
        };

        Self {
            channels,
            sample_rate,
            ballistics,
            peak_detectors,
            rms_calculators,
            vu_meters,
            phase_meter,
            data: MeteringData::new(channels),
        }
    }

    /// Process audio samples and update meters.
    pub fn process(&mut self, samples: &[f32]) {
        let samples_per_channel = samples.len() / self.channels;

        for ch in 0..self.channels {
            // Extract channel samples
            let channel_samples: Vec<f32> = (0..samples_per_channel)
                .map(|i| samples[i * self.channels + ch])
                .collect();

            // Update peak detector
            let peak = self.peak_detectors[ch].process(&channel_samples);
            self.data.peak[ch].current_db = linear_to_db(peak);
            if peak > db_to_linear(self.data.peak[ch].max_db) {
                self.data.peak[ch].max_db = linear_to_db(peak);
            }
            if peak >= 1.0 {
                self.data.peak[ch].clipped = true;
            }

            // Update RMS calculator
            let rms = self.rms_calculators[ch].process(&channel_samples);
            self.data.rms[ch].current_db = linear_to_db(rms);
            if rms > db_to_linear(self.data.rms[ch].max_db) {
                self.data.rms[ch].max_db = linear_to_db(rms);
            }

            // Update VU meter
            let vu = self.vu_meters[ch].process(&channel_samples);
            self.data.vu[ch].vu_level = vu;
        }

        // Update phase correlation (stereo only)
        if self.channels == 2 && samples.len() >= 2 {
            if let Some(ref mut phase_meter) = self.phase_meter {
                let left: Vec<f32> = (0..samples_per_channel).map(|i| samples[i * 2]).collect();
                let right: Vec<f32> = (0..samples_per_channel)
                    .map(|i| samples[i * 2 + 1])
                    .collect();

                let correlation = phase_meter.process(&left, &right);
                if let Some(ref mut pc) = self.data.phase_correlation {
                    pc.correlation = correlation;
                    if correlation < pc.min_correlation {
                        pc.min_correlation = correlation;
                    }
                }
            }
        }
    }

    /// Get current metering data.
    #[must_use]
    pub fn data(&self) -> &MeteringData {
        &self.data
    }

    /// Reset all meters.
    pub fn reset(&mut self) {
        for detector in &mut self.peak_detectors {
            detector.reset();
        }
        for calculator in &mut self.rms_calculators {
            calculator.reset();
        }
        for vu_meter in &mut self.vu_meters {
            vu_meter.reset();
        }
        if let Some(ref mut phase_meter) = self.phase_meter {
            phase_meter.reset();
        }
        self.data.reset();
    }
}

/// Peak detector with hold.
#[derive(Debug, Clone)]
struct PeakDetector {
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: u32,
    /// Peak hold time in seconds.
    hold_time: f32,
    /// Current peak value.
    current_peak: f32,
    /// Hold counter.
    hold_counter: usize,
}

impl PeakDetector {
    fn new(sample_rate: u32, hold_time: f32) -> Self {
        Self {
            sample_rate,
            hold_time,
            current_peak: 0.0,
            hold_counter: 0,
        }
    }

    fn process(&mut self, samples: &[f32]) -> f32 {
        let max_sample = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if max_sample > self.current_peak {
            self.current_peak = max_sample;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            {
                self.hold_counter =
                    (f64::from(self.sample_rate) * f64::from(self.hold_time)) as usize;
            }
        } else if self.hold_counter > 0 {
            self.hold_counter = self.hold_counter.saturating_sub(samples.len());
        } else {
            // Decay
            self.current_peak *= 0.95;
        }

        self.current_peak
    }

    fn reset(&mut self) {
        self.current_peak = 0.0;
        self.hold_counter = 0;
    }
}

/// RMS calculator with sliding window.
#[derive(Debug, Clone)]
struct RmsCalculator {
    /// Window size in samples.
    window_size: usize,
    /// Sample buffer.
    buffer: VecDeque<f32>,
    /// Sum of squares.
    sum_squares: f64,
}

impl RmsCalculator {
    fn new(sample_rate: u32, integration_time_ms: u32) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let window_size =
            (f64::from(sample_rate) * f64::from(integration_time_ms) / 1000.0) as usize;
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum_squares: 0.0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn process(&mut self, samples: &[f32]) -> f32 {
        for &sample in samples {
            #[allow(clippy::cast_precision_loss)]
            let square = f64::from(sample * sample);
            self.sum_squares += square;
            self.buffer.push_back(sample);

            if self.buffer.len() > self.window_size {
                if let Some(old_sample) = self.buffer.pop_front() {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        self.sum_squares -= f64::from(old_sample * old_sample);
                    }
                }
            }
        }

        if self.buffer.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = self.sum_squares / self.buffer.len() as f64;
            #[allow(clippy::cast_possible_truncation)]
            {
                mean.sqrt() as f32
            }
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sum_squares = 0.0;
    }
}

/// VU meter (IEC 60268-10 standard).
#[derive(Debug, Clone)]
struct VuMeter {
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: u32,
    /// Current VU level.
    vu_level: f32,
    /// Attack coefficient.
    attack_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
}

impl VuMeter {
    fn new(sample_rate: u32) -> Self {
        // VU meter has 300ms integration time
        let attack_time = 0.3; // 300ms
        let release_time = 0.3;

        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let attack_coeff = (-1.0 / (f64::from(sample_rate) * attack_time)).exp() as f32;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let release_coeff = (-1.0 / (f64::from(sample_rate) * release_time)).exp() as f32;

        Self {
            sample_rate,
            vu_level: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    fn process(&mut self, samples: &[f32]) -> f32 {
        for &sample in samples {
            let abs_sample = sample.abs();

            if abs_sample > self.vu_level {
                self.vu_level += (abs_sample - self.vu_level) * (1.0 - self.attack_coeff);
            } else {
                self.vu_level += (abs_sample - self.vu_level) * (1.0 - self.release_coeff);
            }
        }

        // Convert to VU (0 VU = +4 dBu = -18 dBFS for broadcast)
        linear_to_db(self.vu_level) + 18.0
    }

    fn reset(&mut self) {
        self.vu_level = 0.0;
    }
}

/// Phase correlation meter.
#[derive(Debug, Clone)]
struct PhaseCorrelationMeter {
    /// Sample rate in Hz.
    #[allow(dead_code)]
    sample_rate: u32,
    /// Buffer size for correlation calculation.
    buffer_size: usize,
    /// Left channel buffer.
    left_buffer: VecDeque<f32>,
    /// Right channel buffer.
    right_buffer: VecDeque<f32>,
}

impl PhaseCorrelationMeter {
    fn new(sample_rate: u32) -> Self {
        let buffer_size = 1024; // Arbitrary window for correlation
        Self {
            sample_rate,
            buffer_size,
            left_buffer: VecDeque::with_capacity(buffer_size),
            right_buffer: VecDeque::with_capacity(buffer_size),
        }
    }

    #[allow(clippy::similar_names)]
    fn process(&mut self, left: &[f32], right: &[f32]) -> f32 {
        // Add samples to buffers
        for (&l, &r) in left.iter().zip(right.iter()) {
            self.left_buffer.push_back(l);
            self.right_buffer.push_back(r);

            if self.left_buffer.len() > self.buffer_size {
                self.left_buffer.pop_front();
            }
            if self.right_buffer.len() > self.buffer_size {
                self.right_buffer.pop_front();
            }
        }

        // Calculate correlation
        if self.left_buffer.len() < self.buffer_size {
            return 0.0;
        }

        #[allow(clippy::similar_names)]
        let (sum_lr, sum_ll, sum_rr) = {
            let mut sum_lr = 0.0;
            let mut sum_ll = 0.0;
            let mut sum_rr = 0.0;

            for (l, r) in self.left_buffer.iter().zip(self.right_buffer.iter()) {
                sum_lr += l * r;
                sum_ll += l * l;
                sum_rr += r * r;
            }

            (sum_lr, sum_ll, sum_rr)
        };

        if sum_ll == 0.0 || sum_rr == 0.0 {
            0.0
        } else {
            sum_lr / (sum_ll * sum_rr).sqrt()
        }
    }

    fn reset(&mut self) {
        self.left_buffer.clear();
        self.right_buffer.clear();
    }
}

/// Convert linear amplitude to dB.
#[must_use]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        -f32::INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Convert dB to linear amplitude.
#[must_use]
pub fn db_to_linear(db: f32) -> f32 {
    if db == -f32::INFINITY {
        0.0
    } else {
        10.0_f32.powf(db / 20.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_db() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 0.01);
        assert!((linear_to_db(0.5) - (-6.02)).abs() < 0.01);
        assert_eq!(linear_to_db(0.0), -f32::INFINITY);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.01);
        assert!((db_to_linear(-6.0) - 0.501).abs() < 0.01);
        assert_eq!(db_to_linear(-f32::INFINITY), 0.0);
    }

    #[test]
    fn test_meter_creation() {
        let meter = Meter::new(2, 48000, MeterBallistics::Medium);
        assert_eq!(meter.channels, 2);
        assert_eq!(meter.data.peak.len(), 2);
    }

    #[test]
    fn test_peak_detector() {
        let mut detector = PeakDetector::new(48000, 1.0);
        let samples = vec![0.5, 0.3, 0.8, 0.2];
        let peak = detector.process(&samples);
        assert!((peak - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_rms_calculator() {
        let mut calculator = RmsCalculator::new(48000, 100);
        let samples = vec![0.5; 1000];
        let rms = calculator.process(&samples);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_phase_correlation() {
        let mut meter = PhaseCorrelationMeter::new(48000);

        // Identical signals should have correlation ~1.0
        let left = vec![0.5; 1024];
        let right = vec![0.5; 1024];
        let corr = meter.process(&left, &right);
        assert!(corr > 0.9);

        meter.reset();

        // Inverted signals should have correlation ~-1.0
        let left2 = vec![0.5; 1024];
        let right2 = vec![-0.5; 1024];
        let corr2 = meter.process(&left2, &right2);
        assert!(corr2 < -0.9);
    }

    #[test]
    fn test_metering_data() {
        let mut data = MeteringData::new(2);
        assert_eq!(data.peak.len(), 2);
        assert!(data.phase_correlation.is_some());

        data.reset();
        assert_eq!(data.peak[0].current_db, -f32::INFINITY);
    }

    #[test]
    fn test_meter_ballistics() {
        assert_eq!(MeterBallistics::Fast.integration_time_ms(), 100);
        assert_eq!(MeterBallistics::Medium.integration_time_ms(), 300);
        assert_eq!(MeterBallistics::Slow.integration_time_ms(), 600);
        assert_eq!(MeterBallistics::Custom(500).integration_time_ms(), 500);
    }
}
