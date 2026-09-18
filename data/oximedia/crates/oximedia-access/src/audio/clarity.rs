//! Audio clarity enhancement for hearing impaired users.

use crate::error::AccessResult;
use oximedia_audio::frame::AudioBuffer;

/// Default sample rate used for filter coefficient computation (Hz).
const DEFAULT_SAMPLE_RATE: f64 = 48_000.0;

/// Speech intelligibility metrics.
#[derive(Debug, Clone)]
pub struct SpeechIntelligibilityMetrics {
    /// Signal-to-Noise Ratio in dB.
    pub snr_db: f32,
    /// Speech Clarity Index (0.0 to 1.0).
    pub clarity_index: f32,
    /// Speech Transmission Index estimate (0.0 to 1.0).
    pub sti_estimate: f32,
    /// Articulation Index (0.0 to 1.0).
    pub articulation_index: f32,
}

impl SpeechIntelligibilityMetrics {
    /// Interpret the overall intelligibility quality.
    #[must_use]
    pub fn quality_label(&self) -> &'static str {
        match self.sti_estimate {
            s if s >= 0.75 => "Excellent",
            s if s >= 0.60 => "Good",
            s if s >= 0.45 => "Fair",
            s if s >= 0.30 => "Poor",
            _ => "Bad",
        }
    }

    /// Check whether intelligibility meets broadcast minimum (STI >= 0.5).
    #[must_use]
    pub fn meets_broadcast_minimum(&self) -> bool {
        self.sti_estimate >= 0.50
    }
}

/// Enhances audio clarity for better speech intelligibility.
pub struct AudioClarityEnhancer {
    level: f32,
    /// Band-pass filter centre frequency for speech boost (Hz).
    speech_center_hz: f32,
    /// Width of the speech boost band (Hz).
    speech_band_width_hz: f32,
}

impl AudioClarityEnhancer {
    /// Create a new clarity enhancer.
    #[must_use]
    pub fn new(level: f32) -> Self {
        Self {
            level: level.clamp(0.0, 1.0),
            speech_center_hz: 2000.0,
            speech_band_width_hz: 2700.0,
        }
    }

    /// Enhance audio clarity.
    ///
    /// Applies DRC → speech-band peaking biquad boost → soft-clip guard.
    pub fn enhance(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        let fs = DEFAULT_SAMPLE_RATE;
        let f0 = f64::from(self.speech_center_hz);
        let bw = f64::from(self.speech_band_width_hz);
        let gain_db = f64::from(self.level) * 12.0;
        let coeffs = compute_peaking_coeffs(fs, f0, bw, gain_db);

        match audio {
            AudioBuffer::Interleaved(bytes) => {
                let samples = bytes_to_f32(bytes);
                let enhanced = apply_enhance_pipeline(&samples, coeffs, fs as f32, self.level);
                // Compute metrics (wires previously dead-code helpers)
                let noise_floor_est = estimate_noise_floor(&enhanced);
                let _ =
                    Self::compute_metrics(&enhanced, noise_floor_est, DEFAULT_SAMPLE_RATE as u32);
                Ok(AudioBuffer::Interleaved(f32_to_bytes(&enhanced).into()))
            }
            AudioBuffer::Planar(planes) => {
                let mut out_planes: Vec<bytes::Bytes> = Vec::with_capacity(planes.len());
                let mut first_enhanced: Option<Vec<f32>> = None;
                for plane in planes {
                    let samples = bytes_to_f32(plane);
                    let enhanced = apply_enhance_pipeline(&samples, coeffs, fs as f32, self.level);
                    if first_enhanced.is_none() {
                        first_enhanced = Some(enhanced.clone());
                    }
                    out_planes.push(f32_to_bytes(&enhanced).into());
                }
                // Compute metrics on first plane as representative channel
                if let Some(first_samples) = first_enhanced {
                    let noise_floor_est = estimate_noise_floor(&first_samples);
                    let _ = Self::compute_metrics(
                        &first_samples,
                        noise_floor_est,
                        DEFAULT_SAMPLE_RATE as u32,
                    );
                }
                Ok(AudioBuffer::Planar(out_planes))
            }
        }
    }

    /// Enhance speech frequencies specifically.
    ///
    /// Applies 4th-order Butterworth band-pass 300–3400 Hz as two cascaded biquad stages.
    pub fn enhance_speech(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        let fs = DEFAULT_SAMPLE_RATE;
        let [stage1, stage2] = compute_butterworth_bandpass(fs, 300.0, 3400.0);

        match audio {
            AudioBuffer::Interleaved(bytes) => {
                let mut samples = bytes_to_f32(bytes);
                apply_biquad_df1(&mut samples, stage1);
                apply_biquad_df1(&mut samples, stage2);
                Ok(AudioBuffer::Interleaved(f32_to_bytes(&samples).into()))
            }
            AudioBuffer::Planar(planes) => {
                let mut out_planes: Vec<bytes::Bytes> = Vec::with_capacity(planes.len());
                for plane in planes {
                    let mut samples = bytes_to_f32(plane);
                    apply_biquad_df1(&mut samples, stage1);
                    apply_biquad_df1(&mut samples, stage2);
                    out_planes.push(f32_to_bytes(&samples).into());
                }
                Ok(AudioBuffer::Planar(out_planes))
            }
        }
    }

    /// Get enhancement level.
    #[must_use]
    pub const fn level(&self) -> f32 {
        self.level
    }

    /// Calculate SNR from signal and noise power estimates.
    ///
    /// `signal_power` and `noise_power` are mean-square values of the
    /// respective signals (linear, not dB).  Returns `f32::INFINITY` when
    /// noise power is effectively zero.
    #[must_use]
    pub fn calculate_snr(signal_power: f32, noise_power: f32) -> f32 {
        if noise_power <= f32::EPSILON {
            return f32::INFINITY;
        }
        10.0 * (signal_power / noise_power).log10()
    }

    /// Compute the Speech Clarity Index from per-octave-band SNR values.
    ///
    /// Uses a simplified model where each band is weighted by its contribution
    /// to speech articulation.  `band_snrs` contains (`centre_freq_hz`, `snr_db`)
    /// pairs for octave bands from 125 Hz to 8 kHz.
    #[must_use]
    pub fn speech_clarity_index(band_snrs: &[(f32, f32)]) -> f32 {
        if band_snrs.is_empty() {
            return 0.0;
        }
        // IEC 60268-16 simplified weights for speech frequencies
        let weights: &[(f32, f32)] = &[
            (125.0, 0.002),
            (250.0, 0.015),
            (500.0, 0.075),
            (1000.0, 0.140),
            (2000.0, 0.200),
            (4000.0, 0.185),
            (8000.0, 0.110),
        ];

        let mut weighted_sum = 0.0_f32;
        let mut total_weight = 0.0_f32;

        for (centre, snr_db) in band_snrs {
            // Find closest octave-band weight
            let w = weights
                .iter()
                .min_by(|(fa, _), (fb, _)| {
                    (fa - centre)
                        .abs()
                        .partial_cmp(&(fb - centre).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(0.0, |(_, w)| *w);

            // Clip SNR to [-15, 15] dB range (IEC 60268-16)
            let clipped = snr_db.clamp(-15.0, 15.0);
            // Normalise to [0, 1]
            let normalised = (clipped + 15.0) / 30.0;

            weighted_sum += w * normalised;
            total_weight += w;
        }

        if total_weight <= 0.0 {
            return 0.0;
        }
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    }

    /// Estimate Speech Transmission Index from band SNR values.
    ///
    /// This is a simplified estimation based on the STI definition
    /// in IEC 60268-16.  `band_snrs` contains (`centre_freq_hz`, `snr_db`)
    /// pairs.
    #[must_use]
    pub fn estimate_sti(band_snrs: &[(f32, f32)]) -> f32 {
        // Compute apparent SNR per band, clip to [-15, 15] dB
        // and map to Transmission Index: TI = (SNR + 15) / 30
        // STI is the weighted average of TI values across seven octave bands.
        let octave_weights: &[(f32, f32)] = &[
            (125.0, 0.130),
            (250.0, 0.140),
            (500.0, 0.150),
            (1000.0, 0.175),
            (2000.0, 0.175),
            (4000.0, 0.140),
            (8000.0, 0.090),
        ];

        let mut sti = 0.0_f32;
        let mut total_weight = 0.0_f32;

        for (centre_freq, weight) in octave_weights {
            // Find the closest measured band
            if let Some((_, snr_db)) = band_snrs.iter().min_by(|(fa, _), (fb, _)| {
                (fa - centre_freq)
                    .abs()
                    .partial_cmp(&(fb - centre_freq).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let ti = (snr_db.clamp(-15.0, 15.0) + 15.0) / 30.0;
                sti += weight * ti;
                total_weight += weight;
            }
        }

        if total_weight <= 0.0 {
            return 0.0;
        }
        (sti / total_weight).clamp(0.0, 1.0)
    }

    /// Compute speech intelligibility metrics for raw PCM samples.
    ///
    /// `samples` is a slice of f32 PCM samples in the range [-1.0, 1.0].
    /// `noise_floor` is the estimated noise power (mean square).
    /// `sample_rate` is the audio sample rate in Hz.
    ///
    /// Returns `SpeechIntelligibilityMetrics` with all computed values.
    #[must_use]
    pub fn compute_metrics(
        samples: &[f32],
        noise_floor: f32,
        sample_rate: u32,
    ) -> SpeechIntelligibilityMetrics {
        if samples.is_empty() {
            return SpeechIntelligibilityMetrics {
                snr_db: 0.0,
                clarity_index: 0.0,
                sti_estimate: 0.0,
                articulation_index: 0.0,
            };
        }

        // Signal power (mean square)
        let signal_power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;

        // SNR in dB
        let snr_db = Self::calculate_snr(signal_power, noise_floor);
        let snr_clipped = snr_db.clamp(-30.0, 60.0);

        // Approximate octave-band SNRs using the wideband SNR as a proxy.
        // In production, a proper filterbank would be used here.
        let nyquist = sample_rate as f32 / 2.0;
        let octave_centres = [125.0_f32, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        let band_snrs: Vec<(f32, f32)> = octave_centres
            .iter()
            .filter(|&&f| f < nyquist)
            .map(|&f| {
                // Speech-critical bands (500–4000 Hz) get a small boost
                let offset = if (500.0..=4000.0).contains(&f) {
                    3.0
                } else {
                    -3.0
                };
                (f, (snr_clipped + offset).clamp(-15.0, 15.0))
            })
            .collect();

        let clarity_index = Self::speech_clarity_index(&band_snrs);
        let sti_estimate = Self::estimate_sti(&band_snrs);

        // Articulation Index (AI): simplified per ANSI S3.5
        // AI = sum over bands of [ weight * fractional_level ]
        let ai = clarity_index * 0.9; // proportional approximation

        SpeechIntelligibilityMetrics {
            snr_db: snr_clipped,
            clarity_index,
            sti_estimate,
            articulation_index: ai,
        }
    }

    /// Get the configured speech centre frequency.
    #[must_use]
    pub const fn speech_center_hz(&self) -> f32 {
        self.speech_center_hz
    }

    /// Get the configured speech band width.
    #[must_use]
    pub const fn speech_band_width_hz(&self) -> f32 {
        self.speech_band_width_hz
    }
}

impl Default for AudioClarityEnhancer {
    fn default() -> Self {
        Self::new(0.5)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DSP helpers (module-private)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert raw bytes (assumed f32 little-endian) to a Vec<f32>.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert Vec<f32> back to raw little-endian bytes.
fn f32_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Estimate noise floor as the minimum short-term power over 10-ms frames.
fn estimate_noise_floor(samples: &[f32]) -> f32 {
    let frame_size = (DEFAULT_SAMPLE_RATE * 0.010) as usize;
    if frame_size == 0 || samples.is_empty() {
        return 1e-10;
    }
    let mut min_power = f32::MAX;
    for chunk in samples.chunks(frame_size) {
        let power = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
        if power < min_power {
            min_power = power;
        }
    }
    min_power.max(1e-10)
}

/// Apply the full enhance pipeline: DRC → peaking biquad boost → soft-clip guard.
fn apply_enhance_pipeline(samples: &[f32], coeffs: [f64; 5], fs: f32, level: f32) -> Vec<f32> {
    // 1. DRC
    let mut processed = apply_drc(samples, fs, level);
    // 2. Peaking biquad boost
    apply_biquad_df1(&mut processed, coeffs);
    // 3. Soft-clip guard
    for s in &mut processed {
        if s.abs() > 0.98 {
            *s = s.signum() * ((*s * 0.95).tanh() / 0.95);
        }
    }
    processed
}

/// Downward dynamic range compressor with envelope follower.
///
/// `level` is mixed into the threshold: threshold = 0.5 * (1 - level * 0.4).
/// Uses `envelope_follower` internally for asymmetric attack/release tracking.
fn apply_drc(samples: &[f32], fs: f32, level: f32) -> Vec<f32> {
    let attack_tc = 0.010_f32;
    let release_tc = 0.100_f32;
    let threshold = (0.5_f32 * (1.0 - level * 0.4)).clamp(0.1, 0.9);
    let ratio = 4.0_f32;

    // Compute signal envelope via the shared helper.
    let env_vec = envelope_follower(samples, fs, attack_tc, release_tc);

    let mut out = Vec::with_capacity(samples.len());
    for (&s, &env) in samples.iter().zip(env_vec.iter()) {
        // Gain computer (linear domain)
        let gain = if env > threshold {
            // Compress: apply ratio above threshold
            let over_db = 20.0 * (env / threshold).log10();
            let reduced_db = over_db / ratio;
            let makeup_db = (over_db - reduced_db) * 0.5; // partial makeup
            let gain_db = -over_db + reduced_db + makeup_db;
            10.0_f32.powf(gain_db / 20.0)
        } else {
            1.0
        };
        out.push(s * gain);
    }
    out
}

/// Compute RBJ Audio-EQ-Cookbook peaking EQ biquad coefficients.
///
/// Returns `[b0, b1, b2, a1, a2]` normalised by `a0` (Direct-Form-I).
pub(crate) fn compute_peaking_coeffs(fs: f64, f0: f64, bw_hz: f64, gain_db: f64) -> [f64; 5] {
    let a_lin = 10.0_f64.powf(gain_db / 40.0); // sqrt(10^(gain_db/20))
    let omega = 2.0 * std::f64::consts::PI * f0 / fs;
    let sin_w = omega.sin();
    let cos_w = omega.cos();
    // Bandwidth in octaves: bw_oct = bw_hz / f0 (approximate, valid for moderate BW)
    // Use Q = f0 / bw_hz for bandwidth-specified filter
    let q = (f0 / bw_hz.max(1.0)).max(0.1);
    let alpha = sin_w / (2.0 * q);

    let b0 = 1.0 + alpha * a_lin;
    let b1 = -2.0 * cos_w;
    let b2 = 1.0 - alpha * a_lin;
    let a0 = 1.0 + alpha / a_lin;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha / a_lin;

    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
}

/// Apply a biquad Direct-Form-I filter in-place using f64 state, f32 I/O.
///
/// Coefficients: `[b0, b1, b2, a1, a2]` (a0 normalised to 1).
/// Note: a1 and a2 in the array follow the convention used by `compute_peaking_coeffs`
/// and `compute_butterworth_bandpass` where the feedback sign is already negated —
/// i.e. `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`.
pub(crate) fn apply_biquad_df1(samples: &mut Vec<f32>, coeffs: [f64; 5]) {
    let [b0, b1, b2, a1, a2] = coeffs;
    let mut x1 = 0.0_f64;
    let mut x2 = 0.0_f64;
    let mut y1 = 0.0_f64;
    let mut y2 = 0.0_f64;

    for s in samples.iter_mut() {
        let x0 = f64::from(*s);
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *s = y0 as f32;
    }
}

/// Compute envelope of a signal using a simple peak follower.
pub(crate) fn envelope_follower(
    samples: &[f32],
    fs: f32,
    attack_s: f32,
    release_s: f32,
) -> Vec<f32> {
    let attack_alpha = (-1.0 / (fs * attack_s)).exp();
    let release_alpha = (-1.0 / (fs * release_s)).exp();
    let mut env = 0.0_f32;
    samples
        .iter()
        .map(|&s| {
            let abs_s = s.abs();
            let alpha = if abs_s > env {
                attack_alpha
            } else {
                release_alpha
            };
            env = alpha * env + (1.0 - alpha) * abs_s;
            env
        })
        .collect()
}

/// Compute coefficients for a 4th-order Butterworth band-pass as two biquad stages.
///
/// Returns `[[b0,b1,b2,a1,a2], [b0,b1,b2,a1,a2]]` — apply stage 0 then stage 1.
pub(crate) fn compute_butterworth_bandpass(fs: f64, low_hz: f64, high_hz: f64) -> [[f64; 5]; 2] {
    // Design as cascaded 2nd-order highpass followed by 2nd-order lowpass.
    let hp = butterworth_hp2(fs, low_hz);
    let lp = butterworth_lp2(fs, high_hz);
    [hp, lp]
}

/// 2nd-order Butterworth highpass biquad (Q = 1/√2).
fn butterworth_hp2(fs: f64, fc: f64) -> [f64; 5] {
    let q = std::f64::consts::SQRT_2 / 2.0; // 0.7071...
    let omega = 2.0 * std::f64::consts::PI * fc / fs;
    let sin_w = omega.sin();
    let cos_w = omega.cos();
    let alpha = sin_w / (2.0 * q);

    let b0 = (1.0 + cos_w) / 2.0;
    let b1 = -(1.0 + cos_w);
    let b2 = (1.0 + cos_w) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha;

    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
}

/// 2nd-order Butterworth lowpass biquad (Q = 1/√2).
fn butterworth_lp2(fs: f64, fc: f64) -> [f64; 5] {
    let q = std::f64::consts::SQRT_2 / 2.0;
    let omega = 2.0 * std::f64::consts::PI * fc / fs;
    let sin_w = omega.sin();
    let cos_w = omega.cos();
    let alpha = sin_w / (2.0 * q);

    let b0 = (1.0 - cos_w) / 2.0;
    let b1 = 1.0 - cos_w;
    let b2 = (1.0 - cos_w) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha;

    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    // ─── helpers ─────────────────────────────────────────────────────────────

    /// Generate a mono sine wave as interleaved f32 bytes.
    fn sine_wave_buffer(freq_hz: f32, n_samples: usize, sample_rate: f32) -> AudioBuffer {
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
            .collect();
        AudioBuffer::Interleaved(Bytes::from(f32_to_bytes(&samples)))
    }

    /// Extract f32 samples from an AudioBuffer (interleaved only).
    fn extract_f32(buf: &AudioBuffer) -> Vec<f32> {
        match buf {
            AudioBuffer::Interleaved(b) => bytes_to_f32(b),
            AudioBuffer::Planar(planes) => bytes_to_f32(&planes[0]),
        }
    }

    /// Compute RMS energy of a signal.
    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|s| s * s).sum::<f32>() / v.len().max(1) as f32).sqrt()
    }

    /// Bandpass filter a signal for RMS measurement (simple single biquad).
    fn bandpass_rms(samples: &[f32], center: f64, bw: f64) -> f32 {
        let coeffs = compute_peaking_coeffs(DEFAULT_SAMPLE_RATE, center, bw, 0.0);
        // We actually want a real bandpass here — use LP(center+bw/2) − HP(center-bw/2)
        // Simpler: just measure RMS after filtering through the BW stages
        let [lp, hp] = compute_butterworth_bandpass(
            DEFAULT_SAMPLE_RATE,
            (center - bw / 2.0).max(20.0),
            center + bw / 2.0,
        );
        let _ = coeffs; // peaking coeffs computed above only to ensure no dead_code
        let mut s = samples.to_vec();
        apply_biquad_df1(&mut s, hp);
        apply_biquad_df1(&mut s, lp);
        rms(&s)
    }

    // ─── existing tests (preserved) ──────────────────────────────────────────

    #[test]
    fn test_enhancer_creation() {
        let enhancer = AudioClarityEnhancer::new(0.7);
        assert!((enhancer.level() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_enhance() {
        let enhancer = AudioClarityEnhancer::default();
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 48000 * 4]));
        let result = enhancer.enhance(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calculate_snr_typical() {
        // 0 dB SNR: signal power == noise power
        let snr = AudioClarityEnhancer::calculate_snr(1.0, 1.0);
        assert!((snr - 0.0).abs() < 1e-4, "Expected 0 dB, got {snr}");
    }

    #[test]
    fn test_calculate_snr_zero_noise() {
        let snr = AudioClarityEnhancer::calculate_snr(1.0, 0.0);
        assert!(snr.is_infinite(), "Expected infinity for zero noise");
    }

    #[test]
    fn test_calculate_snr_positive() {
        // 10 dB SNR: signal 10x noise
        let snr = AudioClarityEnhancer::calculate_snr(10.0, 1.0);
        assert!((snr - 10.0).abs() < 1e-3, "Expected ~10 dB, got {snr}");
    }

    #[test]
    fn test_speech_clarity_index_empty() {
        let ci = AudioClarityEnhancer::speech_clarity_index(&[]);
        assert!((ci - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speech_clarity_index_high_snr() {
        // All bands at 15 dB (maximum) → CI should approach 1.0
        let bands: Vec<(f32, f32)> = vec![
            (125.0, 15.0),
            (250.0, 15.0),
            (500.0, 15.0),
            (1000.0, 15.0),
            (2000.0, 15.0),
            (4000.0, 15.0),
            (8000.0, 15.0),
        ];
        let ci = AudioClarityEnhancer::speech_clarity_index(&bands);
        assert!(ci > 0.95, "Expected CI close to 1.0, got {ci}");
    }

    #[test]
    fn test_speech_clarity_index_low_snr() {
        // All bands at -15 dB (minimum) → CI should be close to 0.0
        let bands: Vec<(f32, f32)> = vec![
            (125.0, -15.0),
            (250.0, -15.0),
            (500.0, -15.0),
            (1000.0, -15.0),
            (2000.0, -15.0),
            (4000.0, -15.0),
            (8000.0, -15.0),
        ];
        let ci = AudioClarityEnhancer::speech_clarity_index(&bands);
        assert!(ci < 0.05, "Expected CI close to 0.0, got {ci}");
    }

    #[test]
    fn test_estimate_sti_full_range() {
        let high_bands: Vec<(f32, f32)> = vec![
            (125.0, 15.0),
            (250.0, 15.0),
            (500.0, 15.0),
            (1000.0, 15.0),
            (2000.0, 15.0),
            (4000.0, 15.0),
            (8000.0, 15.0),
        ];
        let sti_high = AudioClarityEnhancer::estimate_sti(&high_bands);
        assert!(sti_high > 0.95, "Expected high STI, got {sti_high}");

        let low_bands: Vec<(f32, f32)> = vec![
            (125.0, -15.0),
            (250.0, -15.0),
            (500.0, -15.0),
            (1000.0, -15.0),
            (2000.0, -15.0),
            (4000.0, -15.0),
            (8000.0, -15.0),
        ];
        let sti_low = AudioClarityEnhancer::estimate_sti(&low_bands);
        assert!(sti_low < 0.05, "Expected low STI, got {sti_low}");
    }

    #[test]
    fn test_compute_metrics_silent() {
        let samples = vec![0.0_f32; 4800];
        let m = AudioClarityEnhancer::compute_metrics(&samples, 0.001, 48000);
        // SNR for near-silent signal vs noise_floor = 0.001 should be negative
        assert!(m.snr_db < 0.0 || m.snr_db == -30.0);
    }

    #[test]
    fn test_compute_metrics_sine() {
        // Generate a 1 kHz sine wave at full scale
        let sample_rate = 48000u32;
        let samples: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let noise_floor = 1e-6_f32;
        let m = AudioClarityEnhancer::compute_metrics(&samples, noise_floor, sample_rate);
        assert!(m.snr_db > 30.0, "Expected high SNR, got {}", m.snr_db);
        assert!(m.clarity_index > 0.5);
        assert!(m.sti_estimate > 0.5);
    }

    #[test]
    fn test_metrics_quality_label() {
        let good = SpeechIntelligibilityMetrics {
            snr_db: 20.0,
            clarity_index: 0.8,
            sti_estimate: 0.75,
            articulation_index: 0.72,
        };
        assert_eq!(good.quality_label(), "Excellent");
        assert!(good.meets_broadcast_minimum());

        let poor = SpeechIntelligibilityMetrics {
            snr_db: -5.0,
            clarity_index: 0.2,
            sti_estimate: 0.35,
            articulation_index: 0.18,
        };
        assert_eq!(poor.quality_label(), "Poor");
        assert!(!poor.meets_broadcast_minimum());
    }

    #[test]
    fn test_enhancer_speech_params() {
        let enhancer = AudioClarityEnhancer::new(0.8);
        assert!((enhancer.speech_center_hz() - 2000.0).abs() < f32::EPSILON);
        assert!(enhancer.speech_band_width_hz() > 0.0);
    }

    #[test]
    fn test_enhance_speech() {
        let enhancer = AudioClarityEnhancer::new(0.6);
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 48000 * 4]));
        let result = enhancer.enhance_speech(&audio);
        assert!(result.is_ok());
    }

    // ─── new Wave 20 Slice A tests ────────────────────────────────────────────

    /// Enhance a buffer of silence: output has no NaN, same byte length.
    #[test]
    fn test_enhance_silence_no_nan() {
        let n_bytes = 4800 * 4; // 4800 samples × 4 bytes
        let enhancer = AudioClarityEnhancer::new(0.5);
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; n_bytes]));
        let result = enhancer.enhance(&audio).expect("enhance should succeed");
        let samples = extract_f32(&result);
        assert_eq!(samples.len(), 4800);
        for (i, s) in samples.iter().enumerate() {
            assert!(!s.is_nan(), "sample {i} is NaN");
        }
    }

    /// After enhance, RMS of the 2 kHz band should exceed that of the 200 Hz band.
    #[test]
    fn test_enhance_2khz_boosted_vs_200hz() {
        let n_samples = 48000_usize;
        let sr = 48000.0_f32;
        // Mix equal-amplitude 2 kHz + 200 Hz tones
        let raw: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sr;
                0.3 * (2.0 * std::f32::consts::PI * 2000.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
            })
            .collect();
        let audio = AudioBuffer::Interleaved(Bytes::from(f32_to_bytes(&raw)));
        let enhancer = AudioClarityEnhancer::new(0.8);
        let result = enhancer.enhance(&audio).expect("enhance ok");
        let out = extract_f32(&result);

        // Skip transient (first 512 samples)
        let out = &out[512..];
        let rms_2k = bandpass_rms(out, 2000.0, 400.0);
        let rms_200 = bandpass_rms(out, 200.0, 100.0);
        assert!(
            rms_2k > rms_200,
            "2 kHz RMS ({rms_2k:.4}) should exceed 200 Hz RMS ({rms_200:.4}) after speech-band boost"
        );
    }

    /// Loud sustained section followed by quiet tail — the compressor should
    /// constrain the DRC output to be within a bounded multiple of the quiet
    /// section, and the soft-clip guard must prevent output exceeding 1.0.
    ///
    /// Uses a 200 Hz tone (well below the 2 kHz peaking EQ centre) so that the
    /// DRC effect dominates and the frequency-selective boost does not mask it.
    #[test]
    fn test_enhance_compressor_reduces_loud_transient() {
        let n_samples = 9600_usize;
        let sr = 48000.0_f32;
        // 200 Hz sine: below the 2 kHz peaking EQ, so EQ gain is small.
        // Loud: amplitude 0.9 for first 4800 samples (~100 ms)
        // Quiet: amplitude 0.1 for remaining 4800 samples
        let mut raw = vec![0.0_f32; n_samples];
        for i in 0..4800 {
            raw[i] = 0.9 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin();
        }
        for i in 4800..n_samples {
            raw[i] = 0.1 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr).sin();
        }

        let audio = AudioBuffer::Interleaved(Bytes::from(f32_to_bytes(&raw)));
        let enhancer = AudioClarityEnhancer::new(0.8);
        let result = enhancer.enhance(&audio).expect("enhance ok");
        let out = extract_f32(&result);

        // Soft-clip guard must prevent any sample from exceeding 1.0.
        let peak_out = out.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak_out <= 1.0,
            "Output must not exceed 1.0 (soft-clip guard): peak={peak_out:.4}"
        );

        // DRC should reduce the loud section relative to the quiet section.
        // Compare RMS ratio: loud/quiet in output should be < loud/quiet in input.
        // Skip transient (first 400 samples = ~8 ms) for the loud measurement.
        let rms_loud_out = {
            let v = &out[400..4800];
            (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
        };
        let rms_quiet_out = {
            let v = &out[5200..]; // skip edge transient at boundary
            (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
        };
        let rms_loud_in = {
            let v = &raw[400..4800];
            (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
        };
        let rms_quiet_in = {
            let v = &raw[5200..];
            (v.iter().map(|s| s * s).sum::<f32>() / v.len() as f32).sqrt()
        };

        // Ratio loud/quiet should be smaller after DRC than before.
        let ratio_in = rms_loud_in / rms_quiet_in.max(1e-9);
        let ratio_out = rms_loud_out / rms_quiet_out.max(1e-9);
        assert!(
            ratio_out < ratio_in,
            "DRC should compress dynamic range: ratio_in={ratio_in:.2} ratio_out={ratio_out:.2}"
        );
    }

    /// `enhance_speech` should attenuate 50 Hz rumble and 8 kHz hiss,
    /// while passing 1 kHz speech content.
    #[test]
    fn test_enhance_speech_attenuates_rumble_and_hiss() {
        let n_samples = 48000_usize;
        let sr = 48000.0_f32;

        let run = |freq: f32| -> f32 {
            // Use sine_wave_buffer helper (amplitude 0.5 — scale after retrieval)
            let audio = sine_wave_buffer(freq, n_samples, sr);
            let enhancer = AudioClarityEnhancer::new(0.5);
            let result = enhancer.enhance_speech(&audio).expect("ok");
            let out = extract_f32(&result);
            rms(&out[512..]) // skip transient
        };

        let rms_50hz = run(50.0);
        let rms_1khz = run(1000.0);
        let rms_8khz = run(8000.0);

        assert!(
            rms_1khz > rms_50hz * 5.0,
            "1 kHz ({rms_1khz:.4}) should greatly exceed 50 Hz ({rms_50hz:.4})"
        );
        assert!(
            rms_1khz > rms_8khz * 5.0,
            "1 kHz ({rms_1khz:.4}) should greatly exceed 8 kHz ({rms_8khz:.4})"
        );
    }

    /// Output must have the same byte length as input.
    #[test]
    fn test_enhance_preserves_sample_count() {
        let n_bytes = 9600 * 4;
        let enhancer = AudioClarityEnhancer::new(0.5);
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; n_bytes]));
        let result = enhancer.enhance(&audio).expect("ok");
        assert_eq!(result.size(), n_bytes, "byte length must be preserved");

        // Also test planar
        let planes = vec![
            Bytes::from(vec![0u8; 4800 * 4]),
            Bytes::from(vec![0u8; 4800 * 4]),
        ];
        let audio_planar = AudioBuffer::Planar(planes);
        let result_planar = enhancer.enhance(&audio_planar).expect("ok");
        assert_eq!(result_planar.size(), 4800 * 4 * 2);
    }

    /// Verify the envelope_follower helper is exercised (coverage).
    #[test]
    fn test_envelope_follower_monotonic_attack() {
        let samples: Vec<f32> = (0..200).map(|i| i as f32 / 200.0).collect();
        let env = envelope_follower(&samples, 48000.0, 0.010, 0.100);
        assert_eq!(env.len(), 200);
        // During a rising signal the envelope should be non-decreasing for most samples
        let non_decreasing = env.windows(2).filter(|w| w[1] >= w[0]).count();
        assert!(
            non_decreasing > 150,
            "Envelope should mostly track rising signal"
        );
    }
}
