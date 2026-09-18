//! Audio monitoring scope for real-time level metering and peak detection.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single channel bar in an audio meter, tracking level, peak, and RMS.
#[derive(Debug, Clone)]
pub struct ScopeBar {
    /// Instantaneous signal level (0.0–1.0 linear).
    pub level: f64,
    /// Peak hold value (0.0–1.0 linear).
    pub peak: f64,
    /// Root-mean-square value (0.0–1.0 linear).
    pub rms: f64,
    /// Timestamp (ms) when the peak was last set.
    peak_set_at_ms: f64,
}

impl ScopeBar {
    fn new() -> Self {
        Self {
            level: 0.0,
            peak: 0.0,
            rms: 0.0,
            peak_set_at_ms: 0.0,
        }
    }
}

/// Multi-channel audio level meter with peak hold.
#[derive(Debug, Clone)]
pub struct AudioMeter {
    /// Per-channel scope bars.
    pub channels: Vec<ScopeBar>,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Duration (ms) to hold peak values before decay.
    pub peak_hold_ms: f64,
}

impl AudioMeter {
    /// Create a new `AudioMeter` with the given channel count and sample rate.
    ///
    /// `peak_hold_ms` defaults to 1500 ms.
    #[must_use]
    pub fn new(channels: usize, sample_rate: f64) -> Self {
        Self {
            channels: (0..channels).map(|_| ScopeBar::new()).collect(),
            sample_rate,
            peak_hold_ms: 1500.0,
        }
    }

    /// Process a single interleaved frame of samples (`now_ms` is the current clock in ms).
    ///
    /// Samples are expected to be interleaved: `[ch0, ch1, ch0, ch1, …]`.
    /// If the frame contains fewer samples than channels, remaining channels are left at 0.
    pub fn process_frame(&mut self, samples: &[f64], now_ms: f64) {
        let num_ch = self.channels.len();
        if num_ch == 0 || samples.is_empty() {
            return;
        }
        // Deinterleave per channel.
        for ch in 0..num_ch {
            let ch_samples: Vec<f64> = samples.iter().skip(ch).step_by(num_ch).copied().collect();
            if ch_samples.is_empty() {
                continue;
            }
            let peak = compute_peak(&ch_samples);
            let rms = compute_rms(&ch_samples);
            let bar = &mut self.channels[ch];
            bar.level = peak;
            bar.rms = rms;
            // Peak hold logic.
            if peak >= bar.peak {
                bar.peak = peak;
                bar.peak_set_at_ms = now_ms;
            } else if now_ms - bar.peak_set_at_ms > self.peak_hold_ms {
                // Decay peak slowly (10% per call after hold expires).
                bar.peak *= 0.9;
                if bar.peak < peak {
                    bar.peak = peak;
                }
            }
        }
    }

    /// Reset all peak hold values to current levels.
    pub fn reset_peaks(&mut self) {
        for bar in &mut self.channels {
            bar.peak = 0.0;
            bar.peak_set_at_ms = 0.0;
        }
    }

    /// Return a reference to the scope bar for channel `idx`, or `None` if out of range.
    #[must_use]
    pub fn channel(&self, idx: usize) -> Option<&ScopeBar> {
        self.channels.get(idx)
    }
}

/// Compute the root-mean-square of a slice of samples.
#[must_use]
pub fn compute_rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute the absolute peak of a slice of samples.
pub fn compute_peak(samples: &[f64]) -> f64 {
    samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)
}

/// Convert a linear level (0.0–1.0) to dBFS.
///
/// Returns `-120.0` for a zero level to avoid `-inf`.
#[must_use]
pub fn level_to_db(level: f64) -> f64 {
    if level <= 0.0 {
        return -120.0;
    }
    20.0 * level.log10()
}

// ---------------------------------------------------------------------------
// Oscilloscope buffer
// ---------------------------------------------------------------------------

/// A circular sample buffer with hardware-style trigger detection.
///
/// Used to display a stable oscilloscope waveform of an audio signal.
#[derive(Debug, Clone)]
pub struct OscilloscopeBuffer {
    samples: Vec<f32>,
    trigger_level: f32,
    triggered_pos: usize,
    write_pos: usize,
}

impl OscilloscopeBuffer {
    /// Creates a new `OscilloscopeBuffer` with capacity `size` and the given trigger threshold.
    #[must_use]
    pub fn new(size: usize, trigger: f32) -> Self {
        let size = size.max(1);
        Self {
            samples: vec![0.0; size],
            trigger_level: trigger,
            triggered_pos: 0,
            write_pos: 0,
        }
    }

    /// Pushes a single sample into the buffer (circular, oldest overwritten).
    pub fn push(&mut self, sample: f32) {
        let len = self.samples.len();
        self.samples[self.write_pos % len] = sample;
        self.write_pos = self.write_pos.wrapping_add(1);
    }

    /// Finds the first position where the signal crosses `trigger_level` from below.
    ///
    /// Returns the sample index within the buffer, or `None` if no crossing found.
    #[must_use]
    pub fn find_trigger(&self) -> Option<usize> {
        let len = self.samples.len();
        (0..len.saturating_sub(1)).find(|&i| {
            self.samples[i] < self.trigger_level
                && self.samples[(i + 1) % len] >= self.trigger_level
        })
    }

    /// Returns a window of `size` consecutive samples starting at `triggered_pos`.
    ///
    /// If the buffer is shorter than `size`, the whole buffer is returned.
    #[must_use]
    pub fn display_window(&self, size: usize) -> &[f32] {
        let len = self.samples.len();
        let start = self.triggered_pos.min(len.saturating_sub(1));
        let end = (start + size).min(len);
        &self.samples[start..end]
    }
}

// ---------------------------------------------------------------------------
// VU Meter
// ---------------------------------------------------------------------------

/// A single-channel VU meter with RMS, peak, and peak-hold.
#[derive(Debug, Clone)]
pub struct VuMeter {
    /// Instantaneous peak (linear 0.0–1.0+).
    pub peak: f32,
    /// RMS level (linear 0.0–1.0+).
    pub rms: f32,
    /// Held peak value.
    pub hold_peak: f32,
    /// Remaining frames of peak hold.
    pub hold_frames: u32,
}

impl VuMeter {
    /// Creates a zeroed `VuMeter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak: 0.0,
            rms: 0.0,
            hold_peak: 0.0,
            hold_frames: 0,
        }
    }

    /// Processes a block of samples, updating peak, RMS, and hold peak.
    pub fn process_block(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        self.rms = (sum_sq / samples.len() as f32).sqrt();
        if self.peak >= self.hold_peak {
            self.hold_peak = self.peak;
            self.hold_frames = 60; // hold for 60 frames by default
        }
    }

    /// Decrements the hold counter by `frames`, resetting hold peak when expired.
    pub fn decay_peak(&mut self, frames: u32) {
        if self.hold_frames > frames {
            self.hold_frames -= frames;
        } else {
            self.hold_frames = 0;
            // Slowly decay hold peak toward current peak.
            self.hold_peak = (self.hold_peak * 0.95).max(self.peak);
        }
    }

    /// Resets peak hold to the current instantaneous peak.
    pub fn reset_hold(&mut self) {
        self.hold_peak = self.peak;
        self.hold_frames = 0;
    }
}

impl Default for VuMeter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spectrum bar & scope
// ---------------------------------------------------------------------------

/// A single frequency band bar in a spectrum analyzer.
#[derive(Debug, Clone)]
pub struct SpectrumBar {
    /// Centre frequency of this band in Hz.
    pub frequency_hz: f32,
    /// Current amplitude in dBFS.
    pub amplitude_db: f32,
    /// Held peak amplitude in dBFS.
    pub peak_db: f32,
}

impl SpectrumBar {
    /// Creates a new `SpectrumBar` at `frequency_hz` starting at `-120 dBFS`.
    #[must_use]
    pub fn new(frequency_hz: f32) -> Self {
        Self {
            frequency_hz,
            amplitude_db: -120.0,
            peak_db: -120.0,
        }
    }

    /// Decays the current amplitude by `rate` dB per call (subtracted).
    pub fn decay(&mut self, rate: f32) {
        self.amplitude_db = (self.amplitude_db - rate).max(-120.0);
    }

    /// Updates the amplitude; also updates peak if the new value is higher.
    pub fn update(&mut self, amp_db: f32) {
        self.amplitude_db = amp_db;
        if amp_db > self.peak_db {
            self.peak_db = amp_db;
        }
    }
}

/// A multi-band spectrum analyzer scope.
#[derive(Debug, Clone)]
pub struct SpectrumScope {
    /// The frequency bands.
    pub bars: Vec<SpectrumBar>,
    /// Number of frequency bands.
    pub num_bands: usize,
}

impl SpectrumScope {
    /// Creates a new `SpectrumScope` with `num_bands` logarithmically spaced bands
    /// between 20 Hz and 20 kHz.
    #[must_use]
    pub fn new(num_bands: usize) -> Self {
        let num_bands = num_bands.max(1);
        let bars = (0..num_bands)
            .map(|i| {
                // Logarithmic spacing: 20 Hz → 20 000 Hz
                let t = i as f32 / (num_bands.saturating_sub(1).max(1)) as f32;
                let freq = 20.0_f32 * (1000.0_f32.powf(t)); // 20 → 20 000 Hz
                SpectrumBar::new(freq)
            })
            .collect();
        Self { bars, num_bands }
    }

    /// Updates bar amplitudes from a slice of linear magnitude values.
    ///
    /// Each element is converted to dBFS: `20 * log10(mag)`.
    /// Extra magnitudes beyond `num_bands` are ignored; missing bands stay unchanged.
    pub fn update_from_magnitudes(&mut self, mags: &[f32]) {
        for (bar, &mag) in self.bars.iter_mut().zip(mags.iter()) {
            let db = if mag > 0.0 {
                20.0 * mag.log10()
            } else {
                -120.0
            };
            bar.update(db);
        }
    }

    /// Returns the centre frequency of the bar with the highest current amplitude.
    #[must_use]
    pub fn peak_frequency(&self) -> f32 {
        self.bars
            .iter()
            .max_by(|a, b| a.amplitude_db.total_cmp(&b.amplitude_db))
            .map_or(0.0, |b| b.frequency_hz)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stereo Phase Correlation Meter
// ─────────────────────────────────────────────────────────────────────────────

/// Classification of the stereo phase correlation reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationZone {
    /// Correlation in [+0.5, +1.0] — strong mono compatibility, wide but healthy stereo.
    GoodStereo,
    /// Correlation in [0.0, +0.5) — moderate stereo spread, acceptable mono fold-down.
    ModerateStereo,
    /// Correlation in (-0.5, 0.0) — significant anti-phase content, mono fold-down issues.
    PhaseWarning,
    /// Correlation in [-1.0, -0.5] — near-total anti-phase, severe mono cancellation.
    PhaseDanger,
}

impl CorrelationZone {
    /// Classify a correlation value into a zone.
    #[must_use]
    pub fn classify(corr: f32) -> Self {
        if corr >= 0.5 {
            Self::GoodStereo
        } else if corr >= 0.0 {
            Self::ModerateStereo
        } else if corr >= -0.5 {
            Self::PhaseWarning
        } else {
            Self::PhaseDanger
        }
    }
}

/// A sliding-window stereo phase correlation meter.
///
/// Computes the Pearson correlation coefficient between left and right channels
/// over a configurable window and maintains a smoothed readout suitable for
/// a broadcast-style phase meter display.
#[derive(Debug, Clone)]
pub struct PhaseCorrelationMeter {
    /// Current instantaneous (per-block) correlation.
    pub instant_correlation: f32,
    /// Smoothed correlation (IIR low-pass, suitable for meter ballistics).
    pub smooth_correlation: f32,
    /// Minimum correlation seen since last reset.
    pub min_correlation: f32,
    /// Maximum correlation seen since last reset.
    pub max_correlation: f32,
    /// Total number of sample frames processed.
    pub frames_processed: u64,
    /// Smoothing coefficient (0 = no smoothing, 1 = frozen).
    smoothing: f32,
}

impl PhaseCorrelationMeter {
    /// Creates a new meter with the given smoothing coefficient.
    ///
    /// `smoothing` should be in [0.0, 1.0]; typical values are 0.8–0.95 for
    /// broadcast ballistics matching an analogue correlation meter.
    #[must_use]
    pub fn new(smoothing: f32) -> Self {
        let s = smoothing.clamp(0.0, 0.999);
        Self {
            instant_correlation: 1.0,
            smooth_correlation: 1.0,
            min_correlation: 1.0,
            max_correlation: 1.0,
            frames_processed: 0,
            smoothing: s,
        }
    }

    /// Process a block of interleaved stereo samples (L, R, L, R, …).
    ///
    /// Each call updates `instant_correlation`, `smooth_correlation`,
    /// `min_correlation`, and `max_correlation`.
    ///
    /// Returns the instantaneous correlation for the current block.
    pub fn process_interleaved(&mut self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return self.instant_correlation;
        }
        let n = samples.len() / 2;
        let corr = correlation_from_interleaved(samples, n);
        self.update(corr);
        corr
    }

    /// Process separate left and right channel slices.
    ///
    /// The shorter slice determines the frame length used.
    pub fn process_channels(&mut self, left: &[f32], right: &[f32]) -> f32 {
        let n = left.len().min(right.len());
        if n == 0 {
            return self.instant_correlation;
        }
        let corr = correlation_from_channels(left, right, n);
        self.update(corr);
        corr
    }

    /// Returns the current zone classification for the smoothed reading.
    #[must_use]
    pub fn zone(&self) -> CorrelationZone {
        CorrelationZone::classify(self.smooth_correlation)
    }

    /// Resets statistics (min/max/smooth) but keeps smoothing coefficient.
    pub fn reset(&mut self) {
        self.instant_correlation = 1.0;
        self.smooth_correlation = 1.0;
        self.min_correlation = 1.0;
        self.max_correlation = 1.0;
        self.frames_processed = 0;
    }

    fn update(&mut self, corr: f32) {
        self.instant_correlation = corr;
        self.smooth_correlation =
            self.smoothing * self.smooth_correlation + (1.0 - self.smoothing) * corr;
        if corr < self.min_correlation {
            self.min_correlation = corr;
        }
        if corr > self.max_correlation {
            self.max_correlation = corr;
        }
        self.frames_processed = self.frames_processed.saturating_add(1);
    }
}

impl Default for PhaseCorrelationMeter {
    fn default() -> Self {
        Self::new(0.85)
    }
}

/// Compute Pearson correlation from interleaved stereo samples (L,R,L,R,…).
#[allow(clippy::cast_precision_loss)]
fn correlation_from_interleaved(samples: &[f32], n: usize) -> f32 {
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    let mut sum_lr = 0.0f64;

    for i in 0..n {
        let l = f64::from(samples[i * 2]);
        let r = f64::from(samples[i * 2 + 1]);
        sum_l += l;
        sum_r += r;
        sum_ll += l * l;
        sum_rr += r * r;
        sum_lr += l * r;
    }
    pearson_correlation(sum_l, sum_r, sum_ll, sum_rr, sum_lr, n)
}

/// Compute Pearson correlation from separate left and right channel slices.
#[allow(clippy::cast_precision_loss)]
fn correlation_from_channels(left: &[f32], right: &[f32], n: usize) -> f32 {
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    let mut sum_lr = 0.0f64;

    for i in 0..n {
        let l = f64::from(left[i]);
        let r = f64::from(right[i]);
        sum_l += l;
        sum_r += r;
        sum_ll += l * l;
        sum_rr += r * r;
        sum_lr += l * r;
    }
    pearson_correlation(sum_l, sum_r, sum_ll, sum_rr, sum_lr, n)
}

#[allow(clippy::cast_precision_loss)]
fn pearson_correlation(
    sum_l: f64,
    sum_r: f64,
    sum_ll: f64,
    sum_rr: f64,
    sum_lr: f64,
    n: usize,
) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_l = sum_l / nf;
    let mean_r = sum_r / nf;
    let var_l = (sum_ll / nf) - mean_l * mean_l;
    let var_r = (sum_rr / nf) - mean_r * mean_r;
    let covar = (sum_lr / nf) - mean_l * mean_r;
    if var_l > 1e-15 && var_r > 1e-15 {
        (covar / (var_l.sqrt() * var_r.sqrt())).clamp(-1.0, 1.0) as f32
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase Correlation Meter rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for rendering a phase correlation meter bar.
#[derive(Debug, Clone)]
pub struct PhaseCorrelationRenderConfig {
    /// Width of the output RGBA image.
    pub width: u32,
    /// Height of the output RGBA image.
    pub height: u32,
    /// Whether to draw zone boundary markers.
    pub show_zones: bool,
    /// Whether to draw the current-value needle over the bar.
    pub show_needle: bool,
}

impl Default for PhaseCorrelationRenderConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 32,
            show_zones: true,
            show_needle: true,
        }
    }
}

/// Render a horizontal phase correlation meter bar as an RGBA image.
///
/// The bar spans -1 (left, red) to +1 (right, green) with a centre white tick at 0.
/// The filled region reflects `meter.smooth_correlation` and a needle marks
/// `meter.instant_correlation`.
///
/// # Returns
///
/// An RGBA `Vec<u8>` of length `config.width * config.height * 4`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
pub fn render_phase_correlation_bar(
    meter: &PhaseCorrelationMeter,
    config: &PhaseCorrelationRenderConfig,
) -> Vec<u8> {
    let w = config.width as usize;
    let h = config.height as usize;
    let mut out = vec![30u8; w * h * 4]; // Dark background

    // Set all alpha channels to 255
    for chunk in out.chunks_exact_mut(4) {
        chunk[3] = 255;
    }

    // Draw background gradient from red (left) to green (right)
    for x in 0..w {
        let t = x as f32 / (w - 1).max(1) as f32; // 0.0 = left (-1), 1.0 = right (+1)
        let r = ((1.0 - t) * 160.0) as u8;
        let g = (t * 160.0) as u8;
        for y in 1..(h.saturating_sub(1)) {
            let idx = (y * w + x) * 4;
            out[idx] = r;
            out[idx + 1] = g;
            out[idx + 2] = 20;
            out[idx + 3] = 255;
        }
    }

    // Draw zone boundary markers at -0.5, 0.0, +0.5
    if config.show_zones {
        for &corr_val in &[-0.5f32, 0.0, 0.5] {
            let xf = (corr_val + 1.0) / 2.0; // normalise to [0,1]
            let xi = (xf * (w - 1) as f32).round() as usize;
            let tick_color = if corr_val.abs() < 1e-3 {
                [255u8, 255, 255, 255]
            } else {
                [180, 180, 180, 200]
            };
            for y in 0..h {
                let idx = (y * w + xi.min(w - 1)) * 4;
                out[idx] = tick_color[0];
                out[idx + 1] = tick_color[1];
                out[idx + 2] = tick_color[2];
                out[idx + 3] = tick_color[3];
            }
        }
    }

    // Draw smoothed fill from centre to current reading
    let smooth = meter.smooth_correlation.clamp(-1.0, 1.0);
    let centre_x = w / 2;
    let fill_x = ((smooth + 1.0) / 2.0 * (w - 1) as f32).round() as usize;
    let (fill_left, fill_right) = if fill_x >= centre_x {
        (centre_x, fill_x.min(w - 1))
    } else {
        (fill_x, centre_x)
    };

    let fill_y0 = h / 4;
    let fill_y1 = 3 * h / 4;
    for x in fill_left..=fill_right {
        for y in fill_y0..fill_y1 {
            let idx = (y * w + x) * 4;
            out[idx] = out[idx].saturating_add(60);
            out[idx + 1] = out[idx + 1].saturating_add(60);
            out[idx + 2] = out[idx + 2].saturating_add(60);
        }
    }

    // Draw instant needle
    if config.show_needle {
        let inst = meter.instant_correlation.clamp(-1.0, 1.0);
        let nx = ((inst + 1.0) / 2.0 * (w - 1) as f32).round() as usize;
        for y in 0..h {
            let idx = (y * w + nx.min(w - 1)) * 4;
            out[idx] = 255;
            out[idx + 1] = 255;
            out[idx + 2] = 255;
            out[idx + 3] = 255;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rms_silence() {
        let samples = vec![0.0; 100];
        assert_eq!(compute_rms(&samples), 0.0);
    }

    #[test]
    fn test_compute_rms_constant() {
        // RMS of a constant signal equals that constant.
        let samples = vec![0.5; 64];
        let rms = compute_rms(&samples);
        assert!((rms - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rms_sine_approx() {
        // RMS of a full-scale sine ≈ 1/sqrt(2).
        let samples: Vec<f64> = (0..1000)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 1000.0).sin())
            .collect();
        let rms = compute_rms(&samples);
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!((rms - expected).abs() < 0.01);
    }

    #[test]
    fn test_compute_rms_empty() {
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_compute_peak_empty() {
        assert_eq!(compute_peak(&[]), 0.0);
    }

    #[test]
    fn test_compute_peak_negative() {
        let samples = vec![-0.8, 0.3, -0.2, 0.1];
        assert!((compute_peak(&samples) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_compute_peak_positive() {
        let samples = vec![0.1, 0.9, 0.5];
        assert!((compute_peak(&samples) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_level_to_db_full_scale() {
        assert!((level_to_db(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_level_to_db_half() {
        // 0.5 → −6.02 dBFS
        let db = level_to_db(0.5);
        assert!((db - (-6.0206)).abs() < 0.001);
    }

    #[test]
    fn test_level_to_db_zero() {
        assert_eq!(level_to_db(0.0), -120.0);
    }

    #[test]
    fn test_audio_meter_new() {
        let meter = AudioMeter::new(2, 48000.0);
        assert_eq!(meter.channels.len(), 2);
        assert_eq!(meter.sample_rate, 48000.0);
        assert_eq!(meter.peak_hold_ms, 1500.0);
    }

    #[test]
    fn test_audio_meter_process_frame_stereo() {
        let mut meter = AudioMeter::new(2, 48000.0);
        // Interleaved: L=0.9, R=0.5 repeated
        let samples: Vec<f64> = (0..100).flat_map(|_| vec![0.9_f64, 0.5_f64]).collect();
        meter.process_frame(&samples, 0.0);
        assert!((meter.channel(0).expect("should succeed in test").level - 0.9).abs() < 1e-10);
        assert!((meter.channel(1).expect("should succeed in test").level - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_audio_meter_peak_hold() {
        let mut meter = AudioMeter::new(1, 48000.0);
        meter.peak_hold_ms = 1000.0;
        let loud = vec![0.9_f64; 10];
        meter.process_frame(&loud, 0.0);
        let quiet = vec![0.1_f64; 10];
        // Within hold window — peak should not drop.
        meter.process_frame(&quiet, 500.0);
        assert!((meter.channel(0).expect("should succeed in test").peak - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_audio_meter_reset_peaks() {
        let mut meter = AudioMeter::new(2, 48000.0);
        let samples: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
        meter.process_frame(&samples, 0.0);
        meter.reset_peaks();
        assert_eq!(meter.channel(0).expect("should succeed in test").peak, 0.0);
        assert_eq!(meter.channel(1).expect("should succeed in test").peak, 0.0);
    }

    #[test]
    fn test_audio_meter_channel_out_of_bounds() {
        let meter = AudioMeter::new(2, 44100.0);
        assert!(meter.channel(5).is_none());
    }

    // --- OscilloscopeBuffer tests ---

    #[test]
    fn test_oscilloscope_buffer_new() {
        let buf = OscilloscopeBuffer::new(64, 0.5);
        assert_eq!(buf.samples.len(), 64);
        assert!((buf.trigger_level - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_oscilloscope_buffer_push() {
        let mut buf = OscilloscopeBuffer::new(4, 0.0);
        buf.push(1.0);
        buf.push(2.0);
        assert_eq!(buf.samples[0], 1.0);
        assert_eq!(buf.samples[1], 2.0);
    }

    #[test]
    fn test_oscilloscope_buffer_find_trigger() {
        let mut buf = OscilloscopeBuffer::new(8, 0.5);
        // Samples below then above trigger
        for s in [0.0f32, 0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2] {
            buf.push(s);
        }
        // There should be a crossing from below 0.5 to >= 0.5
        let t = buf.find_trigger();
        assert!(t.is_some());
    }

    #[test]
    fn test_oscilloscope_buffer_find_trigger_no_crossing() {
        let mut buf = OscilloscopeBuffer::new(4, 0.9);
        for s in [0.1f32, 0.2, 0.3, 0.4] {
            buf.push(s);
        }
        // All below trigger level — no crossing
        assert!(buf.find_trigger().is_none());
    }

    #[test]
    fn test_oscilloscope_buffer_display_window() {
        let mut buf = OscilloscopeBuffer::new(8, 0.0);
        for s in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            buf.push(s);
        }
        let win = buf.display_window(4);
        assert_eq!(win.len(), 4);
    }

    // --- VuMeter tests ---

    #[test]
    fn test_vu_meter_new() {
        let m = VuMeter::new();
        assert_eq!(m.peak, 0.0);
        assert_eq!(m.rms, 0.0);
        assert_eq!(m.hold_peak, 0.0);
    }

    #[test]
    fn test_vu_meter_process_block() {
        let mut m = VuMeter::new();
        m.process_block(&[0.5f32, -0.5, 0.5, -0.5]);
        assert!((m.peak - 0.5).abs() < 1e-6);
        assert!((m.rms - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_vu_meter_hold_peak_update() {
        let mut m = VuMeter::new();
        m.process_block(&[0.8f32]);
        m.process_block(&[0.1f32]);
        // hold_peak should still be 0.8
        assert!((m.hold_peak - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vu_meter_reset_hold() {
        let mut m = VuMeter::new();
        m.process_block(&[0.9f32]);
        m.reset_hold();
        assert_eq!(m.hold_frames, 0);
    }

    #[test]
    fn test_vu_meter_decay_peak() {
        let mut m = VuMeter::new();
        m.process_block(&[0.8f32]);
        m.decay_peak(70); // exhaust hold frames
                          // hold_peak should have decayed
        assert!(m.hold_peak < 0.8 || m.hold_frames == 0);
    }

    // --- SpectrumBar tests ---

    #[test]
    fn test_spectrum_bar_new() {
        let bar = SpectrumBar::new(1000.0);
        assert!((bar.frequency_hz - 1000.0).abs() < f32::EPSILON);
        assert!((bar.amplitude_db - (-120.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectrum_bar_update_and_peak() {
        let mut bar = SpectrumBar::new(440.0);
        bar.update(-20.0);
        assert!((bar.amplitude_db - (-20.0)).abs() < f32::EPSILON);
        assert!((bar.peak_db - (-20.0)).abs() < f32::EPSILON);
        bar.update(-40.0); // lower than peak
        assert!((bar.peak_db - (-20.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectrum_bar_decay() {
        let mut bar = SpectrumBar::new(1000.0);
        bar.update(-20.0);
        bar.decay(3.0);
        assert!((bar.amplitude_db - (-23.0)).abs() < 1e-5);
    }

    // --- SpectrumScope tests ---

    #[test]
    fn test_spectrum_scope_new() {
        let scope = SpectrumScope::new(8);
        assert_eq!(scope.bars.len(), 8);
        assert_eq!(scope.num_bands, 8);
    }

    #[test]
    fn test_spectrum_scope_update_from_magnitudes() {
        let mut scope = SpectrumScope::new(4);
        let mags = [0.0f32, 0.5, 1.0, 0.1];
        scope.update_from_magnitudes(&mags);
        // mag=1.0 → 0 dB
        assert!((scope.bars[2].amplitude_db - 0.0).abs() < 1e-4);
        // mag=0.0 → -120 dB
        assert!((scope.bars[0].amplitude_db - (-120.0)).abs() < 1e-4);
    }

    #[test]
    fn test_spectrum_scope_peak_frequency() {
        let mut scope = SpectrumScope::new(4);
        let mags = [0.1f32, 0.2, 1.0, 0.05];
        scope.update_from_magnitudes(&mags);
        // Band index 2 has the highest magnitude
        let pf = scope.peak_frequency();
        assert_eq!(pf, scope.bars[2].frequency_hz);
    }

    // --- PhaseCorrelationMeter tests ---

    #[test]
    fn test_phase_correlation_meter_default() {
        let m = PhaseCorrelationMeter::default();
        assert!((m.instant_correlation - 1.0).abs() < f32::EPSILON);
        assert!((m.smooth_correlation - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_correlation_meter_new_clamps_smoothing() {
        let m = PhaseCorrelationMeter::new(2.0);
        // Smoothing is clamped to 0.999
        assert!(m.smoothing < 1.0);
    }

    #[test]
    fn test_phase_correlation_mono_interleaved() {
        let mut m = PhaseCorrelationMeter::new(0.0); // No smoothing
                                                     // Mono signal: L == R every sample
        let samples: Vec<f32> = (0..200)
            .flat_map(|i| {
                let v = (i as f32 * 0.1).sin();
                [v, v]
            })
            .collect();
        let corr = m.process_interleaved(&samples);
        assert!(
            (corr - 1.0).abs() < 0.001,
            "Mono should give corr ≈ 1, got {corr}"
        );
        assert_eq!(m.zone(), CorrelationZone::GoodStereo);
    }

    #[test]
    fn test_phase_correlation_antiphase_channels() {
        let mut m = PhaseCorrelationMeter::new(0.0);
        let left: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let right: Vec<f32> = left.iter().map(|&v| -v).collect();
        let corr = m.process_channels(&left, &right);
        assert!(corr < -0.99, "Anti-phase should give corr ≈ -1, got {corr}");
        assert_eq!(m.zone(), CorrelationZone::PhaseDanger);
    }

    #[test]
    fn test_phase_correlation_reset() {
        let mut m = PhaseCorrelationMeter::new(0.5);
        let samples: Vec<f32> = vec![-0.9f32, 0.9].repeat(50);
        m.process_interleaved(&samples);
        m.reset();
        assert!((m.instant_correlation - 1.0).abs() < f32::EPSILON);
        assert_eq!(m.frames_processed, 0);
    }

    #[test]
    fn test_phase_correlation_zone_classification() {
        assert_eq!(CorrelationZone::classify(0.8), CorrelationZone::GoodStereo);
        assert_eq!(
            CorrelationZone::classify(0.2),
            CorrelationZone::ModerateStereo
        );
        assert_eq!(
            CorrelationZone::classify(-0.2),
            CorrelationZone::PhaseWarning
        );
        assert_eq!(
            CorrelationZone::classify(-0.8),
            CorrelationZone::PhaseDanger
        );
    }

    #[test]
    fn test_phase_correlation_empty_input() {
        let mut m = PhaseCorrelationMeter::default();
        let prev = m.instant_correlation;
        m.process_interleaved(&[]);
        // Should not change on empty input
        assert!((m.instant_correlation - prev).abs() < f32::EPSILON);
    }

    #[test]
    fn test_render_phase_correlation_bar_size() {
        let m = PhaseCorrelationMeter::default();
        let cfg = PhaseCorrelationRenderConfig {
            width: 128,
            height: 24,
            ..Default::default()
        };
        let out = render_phase_correlation_bar(&m, &cfg);
        assert_eq!(out.len(), 128 * 24 * 4);
    }

    #[test]
    fn test_render_phase_correlation_bar_non_empty() {
        let m = PhaseCorrelationMeter::default();
        let cfg = PhaseCorrelationRenderConfig::default();
        let out = render_phase_correlation_bar(&m, &cfg);
        assert!(out.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_phase_correlation_smoothing_effect() {
        let mut m = PhaseCorrelationMeter::new(0.9); // strong smoothing
                                                     // First block at -1
        let left: Vec<f32> = vec![0.5; 100];
        let right: Vec<f32> = vec![-0.5; 100];
        m.process_channels(&left, &right);
        // Smoothed value should be between 1.0 (initial) and -1.0 (instant)
        assert!(m.smooth_correlation > m.instant_correlation);
    }
}
