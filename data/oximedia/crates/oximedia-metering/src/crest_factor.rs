#![allow(dead_code)]
//! Crest factor meter for broadcast audio analysis.
//!
//! Measures the peak-to-RMS ratio (crest factor) of audio signals,
//! which is a key metric for understanding dynamic range, compression,
//! and loudness characteristics. Used in mastering and broadcast
//! quality control workflows.

/// Crest factor measurement result for a single channel.
#[derive(Clone, Debug)]
pub struct CrestFactorResult {
    /// Crest factor in dB (peak - RMS).
    pub crest_factor_db: f64,
    /// Peak level in dBFS.
    pub peak_dbfs: f64,
    /// RMS level in dBFS.
    pub rms_dbfs: f64,
    /// Peak level in linear scale.
    pub peak_linear: f64,
    /// RMS level in linear scale.
    pub rms_linear: f64,
    /// Channel index.
    pub channel: usize,
}

impl CrestFactorResult {
    /// Check if the signal is heavily compressed (crest factor < 6 dB).
    pub fn is_heavily_compressed(&self) -> bool {
        self.crest_factor_db < 6.0
    }

    /// Check if the signal has good dynamic range (crest factor 10-20 dB).
    pub fn has_good_dynamics(&self) -> bool {
        self.crest_factor_db >= 10.0 && self.crest_factor_db <= 20.0
    }

    /// Describe the dynamic character of the signal.
    pub fn character(&self) -> &'static str {
        if self.crest_factor_db < 3.0 {
            "Brick-wall limited"
        } else if self.crest_factor_db < 6.0 {
            "Heavily compressed"
        } else if self.crest_factor_db < 10.0 {
            "Moderately compressed"
        } else if self.crest_factor_db < 14.0 {
            "Natural dynamics"
        } else if self.crest_factor_db < 20.0 {
            "Wide dynamic range"
        } else {
            "Very wide dynamic range"
        }
    }
}

impl Default for CrestFactorResult {
    fn default() -> Self {
        Self {
            crest_factor_db: 0.0,
            peak_dbfs: f64::NEG_INFINITY,
            rms_dbfs: f64::NEG_INFINITY,
            peak_linear: 0.0,
            rms_linear: 0.0,
            channel: 0,
        }
    }
}

/// Windowed crest factor measurement mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrestMode {
    /// Measure over the entire signal (integrated).
    #[default]
    Integrated,
    /// Measure over a sliding window.
    Windowed,
    /// Measure per-block (non-overlapping).
    Block,
}

/// Configuration for the crest factor meter.
#[derive(Clone, Debug)]
pub struct CrestFactorConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Measurement mode.
    pub mode: CrestMode,
    /// Window size in seconds (for windowed/block modes).
    pub window_seconds: f64,
    /// Minimum RMS threshold: below this, crest factor is not meaningful.
    pub rms_floor_db: f64,
}

impl Default for CrestFactorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000.0,
            channels: 2,
            mode: CrestMode::Integrated,
            window_seconds: 0.4,
            rms_floor_db: -80.0,
        }
    }
}

impl CrestFactorConfig {
    /// Create a config for integrated measurement.
    pub fn integrated(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            mode: CrestMode::Integrated,
            ..Default::default()
        }
    }

    /// Create a config for windowed measurement.
    pub fn windowed(sample_rate: f64, channels: usize, window_seconds: f64) -> Self {
        Self {
            sample_rate,
            channels,
            mode: CrestMode::Windowed,
            window_seconds,
            ..Default::default()
        }
    }

    /// Get the window size in samples.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn window_samples(&self) -> usize {
        (self.sample_rate * self.window_seconds) as usize
    }
}

/// Per-channel state for crest factor measurement.
#[derive(Clone, Debug)]
struct ChannelState {
    /// Peak value (absolute).
    peak: f64,
    /// Sum of squared samples (for RMS).
    sum_sq: f64,
    /// Number of samples.
    count: u64,
    /// Circular buffer for windowed mode.
    buffer: Vec<f64>,
    /// Buffer write position.
    buf_pos: usize,
    /// Buffer peak (tracked).
    buf_peak: f64,
}

impl ChannelState {
    /// Create new channel state.
    fn new(window_size: usize) -> Self {
        Self {
            peak: 0.0,
            sum_sq: 0.0,
            count: 0,
            buffer: vec![0.0; window_size.max(1)],
            buf_pos: 0,
            buf_peak: 0.0,
        }
    }

    /// Reset the channel state.
    fn reset(&mut self) {
        self.peak = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
        self.buffer.fill(0.0);
        self.buf_pos = 0;
        self.buf_peak = 0.0;
    }
}

/// Crest factor meter.
///
/// Measures the peak-to-RMS ratio of audio signals across one or more
/// channels, providing crest factor data useful for mastering decisions
/// and broadcast quality control.
pub struct CrestFactorMeter {
    /// Configuration.
    config: CrestFactorConfig,
    /// Per-channel measurement state.
    channel_states: Vec<ChannelState>,
}

impl CrestFactorMeter {
    /// Create a new crest factor meter.
    pub fn new(config: CrestFactorConfig) -> Self {
        let window = config.window_samples();
        let channel_states = (0..config.channels)
            .map(|_| ChannelState::new(window))
            .collect();
        Self {
            config,
            channel_states,
        }
    }

    /// Create with default stereo configuration.
    pub fn stereo(sample_rate: f64) -> Self {
        Self::new(CrestFactorConfig::integrated(sample_rate, 2))
    }

    /// Get the configuration.
    pub fn config(&self) -> &CrestFactorConfig {
        &self.config
    }

    /// Process interleaved audio samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        if samples.is_empty() || self.config.channels == 0 {
            return;
        }
        let channels = self.config.channels;
        let frame_count = samples.len() / channels;

        for frame in 0..frame_count {
            for ch in 0..channels {
                let idx = frame * channels + ch;
                if idx >= samples.len() || ch >= self.channel_states.len() {
                    continue;
                }
                let sample = samples[idx];
                let abs_sample = sample.abs();
                let state = &mut self.channel_states[ch];

                match self.config.mode {
                    CrestMode::Integrated => {
                        state.peak = state.peak.max(abs_sample);
                        state.sum_sq += sample * sample;
                        state.count += 1;
                    }
                    CrestMode::Windowed | CrestMode::Block => {
                        // Write to buffer.
                        if !state.buffer.is_empty() {
                            state.buffer[state.buf_pos] = sample;
                            state.buf_pos = (state.buf_pos + 1) % state.buffer.len();
                        }
                        state.peak = state.peak.max(abs_sample);
                        state.sum_sq += sample * sample;
                        state.count += 1;
                    }
                }
            }
        }
    }

    /// Process mono samples.
    pub fn process_mono(&mut self, samples: &[f64]) {
        if samples.is_empty() || self.channel_states.is_empty() {
            return;
        }
        for &sample in samples {
            let abs_sample = sample.abs();
            let state = &mut self.channel_states[0];
            state.peak = state.peak.max(abs_sample);
            state.sum_sq += sample * sample;
            state.count += 1;

            if self.config.mode != CrestMode::Integrated && !state.buffer.is_empty() {
                state.buffer[state.buf_pos] = sample;
                state.buf_pos = (state.buf_pos + 1) % state.buffer.len();
            }
        }
    }

    /// Get the crest factor result for a specific channel.
    #[allow(clippy::cast_precision_loss)]
    pub fn result_for_channel(&self, channel: usize) -> CrestFactorResult {
        if channel >= self.channel_states.len() {
            return CrestFactorResult::default();
        }

        let state = &self.channel_states[channel];

        let (peak, rms) = match self.config.mode {
            CrestMode::Integrated => {
                if state.count == 0 {
                    return CrestFactorResult {
                        channel,
                        ..Default::default()
                    };
                }
                let rms = (state.sum_sq / state.count as f64).sqrt();
                (state.peak, rms)
            }
            CrestMode::Windowed | CrestMode::Block => {
                // Calculate from buffer.
                let buf = &state.buffer;
                if buf.is_empty() {
                    return CrestFactorResult {
                        channel,
                        ..Default::default()
                    };
                }
                let mut peak = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                for &s in buf {
                    peak = peak.max(s.abs());
                    sum_sq += s * s;
                }
                let rms = (sum_sq / buf.len() as f64).sqrt();
                (peak, rms)
            }
        };

        let peak_dbfs = if peak < 1e-20 {
            f64::NEG_INFINITY
        } else {
            20.0 * peak.log10()
        };
        let rms_dbfs = if rms < 1e-20 {
            f64::NEG_INFINITY
        } else {
            20.0 * rms.log10()
        };

        let crest_db = if rms_dbfs > self.config.rms_floor_db {
            peak_dbfs - rms_dbfs
        } else {
            0.0
        };

        CrestFactorResult {
            crest_factor_db: crest_db,
            peak_dbfs,
            rms_dbfs,
            peak_linear: peak,
            rms_linear: rms,
            channel,
        }
    }

    /// Get crest factor results for all channels.
    pub fn results(&self) -> Vec<CrestFactorResult> {
        (0..self.config.channels)
            .map(|ch| self.result_for_channel(ch))
            .collect()
    }

    /// Get the maximum crest factor across all channels.
    pub fn max_crest_factor_db(&self) -> f64 {
        self.results()
            .iter()
            .map(|r| r.crest_factor_db)
            .fold(0.0_f64, f64::max)
    }

    /// Get the average crest factor across all channels.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_crest_factor_db(&self) -> f64 {
        let results = self.results();
        if results.is_empty() {
            return 0.0;
        }
        let sum: f64 = results.iter().map(|r| r.crest_factor_db).sum();
        sum / results.len() as f64
    }

    /// Check if the signal appears over-compressed (any channel < 6 dB crest).
    pub fn is_over_compressed(&self) -> bool {
        self.results()
            .iter()
            .any(|r| r.rms_dbfs > self.config.rms_floor_db && r.crest_factor_db < 6.0)
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.reset();
        }
    }

    /// Get total samples processed on channel 0.
    pub fn samples_processed(&self) -> u64 {
        self.channel_states.first().map_or(0, |s| s.count)
    }

    /// Get duration processed in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.samples_processed() as f64 / self.config.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crest_factor_result_default() {
        let r = CrestFactorResult::default();
        assert!(r.peak_dbfs.is_infinite());
        assert!(r.rms_dbfs.is_infinite());
        assert_eq!(r.channel, 0);
    }

    #[test]
    fn test_crest_factor_result_character() {
        let mut r = CrestFactorResult::default();

        r.crest_factor_db = 2.0;
        assert_eq!(r.character(), "Brick-wall limited");

        r.crest_factor_db = 5.0;
        assert_eq!(r.character(), "Heavily compressed");
        assert!(r.is_heavily_compressed());

        r.crest_factor_db = 8.0;
        assert_eq!(r.character(), "Moderately compressed");
        assert!(!r.is_heavily_compressed());

        r.crest_factor_db = 12.0;
        assert_eq!(r.character(), "Natural dynamics");
        assert!(r.has_good_dynamics());

        r.crest_factor_db = 16.0;
        assert_eq!(r.character(), "Wide dynamic range");

        r.crest_factor_db = 25.0;
        assert_eq!(r.character(), "Very wide dynamic range");
    }

    #[test]
    fn test_crest_mode_default() {
        assert_eq!(CrestMode::default(), CrestMode::Integrated);
    }

    #[test]
    fn test_crest_config_default() {
        let cfg = CrestFactorConfig::default();
        assert!((cfg.sample_rate - 48000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.mode, CrestMode::Integrated);
    }

    #[test]
    fn test_crest_config_window_samples() {
        let cfg = CrestFactorConfig::windowed(48000.0, 2, 0.4);
        assert_eq!(cfg.window_samples(), 19200);
    }

    #[test]
    fn test_crest_meter_creation() {
        let meter = CrestFactorMeter::stereo(48000.0);
        assert_eq!(meter.config().channels, 2);
        assert_eq!(meter.samples_processed(), 0);
    }

    #[test]
    fn test_crest_meter_silence() {
        let meter = CrestFactorMeter::stereo(48000.0);
        let results = meter.results();
        assert_eq!(results.len(), 2);
        // With no signal, crest factor should be 0.
        assert!((results[0].crest_factor_db).abs() < f64::EPSILON);
    }

    #[test]
    fn test_crest_meter_sine_wave() {
        let mut meter = CrestFactorMeter::new(CrestFactorConfig::integrated(48000.0, 1));
        // Generate a sine wave: crest factor should be ~3 dB (sqrt(2) ratio).
        let samples: Vec<f64> = (0..48000)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
            })
            .collect();
        meter.process_mono(&samples);
        let result = meter.result_for_channel(0);
        // Sine wave crest factor = 20*log10(sqrt(2)) ~= 3.01 dB
        assert!((result.crest_factor_db - 3.01).abs() < 0.1);
    }

    #[test]
    fn test_crest_meter_interleaved() {
        let mut meter = CrestFactorMeter::stereo(48000.0);
        let num_frames = 4800;
        let mut samples = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / 48000.0;
            let val = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5;
            samples.push(val);
            samples.push(val);
        }
        meter.process_interleaved(&samples);
        let results = meter.results();
        assert_eq!(results.len(), 2);
        assert!(results[0].crest_factor_db > 0.0);
        assert!(results[1].crest_factor_db > 0.0);
    }

    #[test]
    fn test_crest_meter_max_and_avg() {
        let mut meter = CrestFactorMeter::stereo(48000.0);
        let num_frames = 4800;
        let mut samples = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / 48000.0;
            let val = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.8;
            samples.push(val);
            samples.push(val * 0.5);
        }
        meter.process_interleaved(&samples);
        let max_cf = meter.max_crest_factor_db();
        let avg_cf = meter.avg_crest_factor_db();
        assert!(max_cf > 0.0);
        assert!(avg_cf > 0.0);
        assert!(max_cf >= avg_cf);
    }

    #[test]
    fn test_crest_meter_reset() {
        let mut meter = CrestFactorMeter::stereo(48000.0);
        let samples = vec![0.5; 9600];
        meter.process_interleaved(&samples);
        assert!(meter.samples_processed() > 0);

        meter.reset();
        assert_eq!(meter.samples_processed(), 0);
    }

    #[test]
    fn test_crest_meter_windowed() {
        let cfg = CrestFactorConfig::windowed(48000.0, 1, 0.1);
        let mut meter = CrestFactorMeter::new(cfg);
        let samples: Vec<f64> = (0..4800)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.7
            })
            .collect();
        meter.process_mono(&samples);
        let result = meter.result_for_channel(0);
        assert!(result.crest_factor_db > 0.0);
    }

    #[test]
    fn test_crest_meter_duration() {
        let mut meter = CrestFactorMeter::new(CrestFactorConfig::integrated(48000.0, 1));
        let samples: Vec<f64> = vec![0.1; 48000]; // 1 second of audio.
        meter.process_mono(&samples);
        assert!((meter.duration_seconds() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_over_compressed_detection() {
        let mut meter = CrestFactorMeter::new(CrestFactorConfig::integrated(48000.0, 1));
        // Square wave: crest factor = 0 dB (peak = RMS).
        let samples: Vec<f64> = (0..48000)
            .map(|i| if i % 100 < 50 { 0.5 } else { -0.5 })
            .collect();
        meter.process_mono(&samples);
        // Square wave has crest factor ~0 dB, definitely over-compressed.
        assert!(meter.is_over_compressed());
    }

    #[test]
    fn test_crest_result_for_invalid_channel() {
        let meter = CrestFactorMeter::stereo(48000.0);
        let result = meter.result_for_channel(99);
        // Out-of-range channel returns default result (channel=0)
        assert_eq!(result.channel, 0);
        assert!(result.peak_dbfs.is_infinite());
    }
}
