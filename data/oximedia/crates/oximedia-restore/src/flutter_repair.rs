//! Wow and flutter correction for tape recordings.
//!
//! "Wow" refers to slow (< 6 Hz) speed variations; "flutter" refers to faster
//! (6–100 Hz) variations.  Both create pitch modulation that degrades
//! perceived quality.  This module provides detection via short-time
//! autocorrelation and correction via time-domain resampling.
//!
//! This module also provides pilot-tone (bias-tone) detection via FFT-based
//! peak finding in the 3–4 kHz range, allowing automatic reference frequency
//! determination for wow/flutter correction.

use oxifft::Complex;

// ---------------------------------------------------------------------------
// Pilot-tone detection
// ---------------------------------------------------------------------------

/// Detect the dominant pilot-tone (bias-tone) frequency in the range 3–4 kHz.
///
/// Returns the detected frequency in Hz, or `None` if no sustained narrow-band
/// peak is found above the noise floor.
///
/// A typical analog tape recorder uses a pilot tone (or "bias tone") in the
/// 3–4 kHz range to allow servo-based wow/flutter measurement.  This function
/// locates the strongest narrow peak in that range and checks its prominence
/// above the local spectral mean.
///
/// # Arguments
///
/// * `samples`     - Mono audio samples (at least 4096 samples required).
/// * `sample_rate` - Sample rate in Hz.
///
/// # Returns
///
/// The estimated pilot-tone frequency in Hz, or `None` if no prominent peak
/// is found.
#[must_use]
pub fn detect_pilot_tone(samples: &[f32], sample_rate: u32) -> Option<f32> {
    if samples.len() < 4096 {
        return None;
    }

    // Choose FFT size: largest power-of-two up to 65536.
    let n = samples.len().next_power_of_two().min(65_536);
    // Use a Hann-windowed slice for spectral leakage reduction.
    let window_len = n.min(samples.len());

    let input: Vec<Complex<f32>> = (0..window_len)
        .map(|i| {
            // Hann window
            let w = 0.5_f32
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window_len - 1) as f32).cos());
            Complex::new(samples[i] * w, 0.0)
        })
        .chain((window_len..n).map(|_| Complex::new(0.0_f32, 0.0_f32)))
        .collect();

    let fft_out = oxifft::fft(&input);

    // Only the positive-frequency bins are meaningful (0 .. n/2).
    let bin_width = sample_rate as f64 / n as f64;
    let lo_bin = (3000.0 / bin_width) as usize;
    let hi_bin = ((4000.0 / bin_width) as usize).min(fft_out.len() / 2);

    if lo_bin >= hi_bin {
        return None;
    }

    // Find the peak magnitude bin in [lo_bin, hi_bin).
    let (peak_rel_idx, peak_mag) = fft_out[lo_bin..hi_bin]
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mag = ((c.re as f64).powi(2) + (c.im as f64).powi(2)).sqrt() as f32;
            (i, mag)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    let peak_bin = lo_bin + peak_rel_idx;

    // Compute local mean magnitude in the 3–4 kHz band.
    let local_mean: f32 = fft_out[lo_bin..hi_bin]
        .iter()
        .map(|c| ((c.re as f64).powi(2) + (c.im as f64).powi(2)).sqrt() as f32)
        .sum::<f32>()
        / (hi_bin - lo_bin) as f32;

    // The peak must be at least 10× the local mean to qualify as a sustained
    // narrow-band pilot tone (filters out broadband noise).
    if local_mean < f32::EPSILON || peak_mag < local_mean * 10.0 {
        return None;
    }

    Some(peak_bin as f32 * bin_width as f32)
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Measures of wow and flutter for a recording.
#[derive(Debug, Clone)]
pub struct FlutterMeasurement {
    /// Dominant wow frequency in Hz (< 6 Hz).
    pub wow_hz: f32,
    /// Dominant flutter frequency in Hz (6–100 Hz).
    pub flutter_hz: f32,
    /// Combined depth of modulation as a percentage (0–100).
    pub depth_pct: f32,
    /// Wow severity score (0 = perfect, 1 = severe).
    pub wow_score: f32,
    /// Flutter severity score (0 = perfect, 1 = severe).
    pub flutter_score: f32,
}

/// Qualitative classification of wow/flutter severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlutterSeverity {
    /// Depth < 0.1 % – inaudible.
    Excellent,
    /// Depth 0.1–0.3 % – barely audible.
    Good,
    /// Depth 0.3–1.0 % – clearly audible.
    Fair,
    /// Depth > 1.0 % – strongly audible.
    Poor,
}

impl FlutterMeasurement {
    /// Classify the severity of the measured wow/flutter.
    #[must_use]
    pub fn classify(&self) -> FlutterSeverity {
        match self.depth_pct {
            d if d < 0.1 => FlutterSeverity::Excellent,
            d if d < 0.3 => FlutterSeverity::Good,
            d if d < 1.0 => FlutterSeverity::Fair,
            _ => FlutterSeverity::Poor,
        }
    }
}

/// Detects wow and flutter via frame-wise instantaneous frequency estimation.
pub struct FlutterDetector;

impl FlutterDetector {
    /// Estimate the instantaneous frequency per frame using short-time
    /// autocorrelation.
    ///
    /// Returns a vector with one frequency estimate (Hz) per analysis frame.
    /// The frame step is fixed at 512 samples; frames shorter than the lag
    /// range are skipped.
    ///
    /// # Arguments
    /// * `samples`     - Mono input buffer.
    /// * `sample_rate` - Sample rate in Hz (e.g. 44100).
    #[must_use]
    pub fn detect_instantaneous_frequency(samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let frame_size = 1024_usize;
        let hop = 512_usize;
        let sr = sample_rate as f32;

        // Search lags corresponding to 50 Hz – 8 kHz
        let lag_min = (sr / 8000.0).ceil() as usize;
        let lag_max = (sr / 50.0).ceil() as usize;

        if samples.len() < frame_size || lag_min >= lag_max {
            return Vec::new();
        }

        let mut freqs = Vec::new();
        let mut pos = 0;

        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];

            // Short-time autocorrelation
            let mut best_lag = lag_min;
            let mut best_corr = f32::NEG_INFINITY;

            let max_lag = lag_max.min(frame_size - 1);
            for lag in lag_min..=max_lag {
                let corr: f32 = frame[..frame_size - lag]
                    .iter()
                    .zip(&frame[lag..])
                    .map(|(&a, &b)| a * b)
                    .sum();
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }

            let freq = if best_lag > 0 {
                sr / best_lag as f32
            } else {
                0.0
            };
            freqs.push(freq);
            pos += hop;
        }

        freqs
    }
}

/// Estimates fractional speed variation relative to nominal.
pub struct SpeedVariationEstimator;

impl SpeedVariationEstimator {
    /// Estimate per-frame speed as a fraction of nominal (1.0 = correct speed).
    ///
    /// The method computes the instantaneous frequency via autocorrelation and
    /// divides by the expected `reference_freq`.
    ///
    /// # Arguments
    /// * `samples`       - Mono input buffer.
    /// * `reference_freq`- The expected fundamental frequency (Hz).
    /// * `sample_rate`   - Sample rate in Hz.
    #[must_use]
    pub fn estimate(samples: &[f32], reference_freq: f32, sample_rate: u32) -> Vec<f32> {
        if reference_freq <= 0.0 {
            return Vec::new();
        }
        let inst_freqs = FlutterDetector::detect_instantaneous_frequency(samples, sample_rate);
        inst_freqs
            .iter()
            .map(|&f| if f > 0.0 { f / reference_freq } else { 1.0 })
            .collect()
    }
}

/// Corrects wow and flutter via time-domain resampling.
pub struct FlutterCorrector;

impl FlutterCorrector {
    /// Apply flutter correction to `samples`.
    ///
    /// Uses the [`FlutterMeasurement`] to build a sinusoidal time-warp
    /// function that counteracts the measured frequency modulation, then
    /// resamples accordingly using linear interpolation.
    ///
    /// # Arguments
    /// * `samples`     - Input audio samples.
    /// * `sample_rate` - Sample rate in Hz.
    /// * `measurement` - Measurement from [`FlutterDetector`] or similar.
    #[must_use]
    pub fn correct(
        samples: &[f32],
        sample_rate: u32,
        measurement: &FlutterMeasurement,
    ) -> Vec<f32> {
        let n = samples.len();
        if n == 0 {
            return Vec::new();
        }

        let sr = sample_rate as f32;

        // Build a time-warp map: for each output sample i, the source position
        // is offset by a sinusoidal correction proportional to depth.
        let depth = measurement.depth_pct / 100.0;
        let wow_omega = 2.0 * std::f32::consts::PI * measurement.wow_hz / sr;
        let flutter_omega = 2.0 * std::f32::consts::PI * measurement.flutter_hz / sr;

        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32;
            // Correction offset (in samples): counteract by negating measured modulation
            let warp =
                -depth * sr * (wow_omega * t).sin() - depth * sr * 0.5 * (flutter_omega * t).sin();

            let src_pos = (t + warp).max(0.0).min((n - 1) as f32);
            let lo = src_pos.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = src_pos - lo as f32;
            let val = samples[lo] * (1.0 - frac) + samples[hi] * frac;
            output.push(val);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq_hz: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    // -------------------------------------------------------------------
    // Pilot-tone detection tests
    // -------------------------------------------------------------------

    /// A pure 3150 Hz sine wave in mild white noise must return ~3150 Hz (±10 Hz).
    #[test]
    fn test_detect_pilot_tone_pure_sine() {
        use rand::RngExt;
        let sr = 44100_u32;
        // 65536 samples guarantees bin width < 1 Hz → fine resolution
        let n = 65536_usize;
        let mut signal = sine(3150.0, sr, n);

        // Add mild white noise at amplitude 0.01 (SNR ≈ 40 dB vs unit sine)
        let mut rng = rand::rng();
        for s in signal.iter_mut() {
            *s += rng.random_range(-0.01_f32..0.01_f32);
        }

        let detected = detect_pilot_tone(&signal, sr);
        assert!(
            detected.is_some(),
            "should detect pilot tone in 3150 Hz + noise signal"
        );
        let freq = detected.expect("checked above");
        assert!(
            (freq - 3150.0).abs() < 10.0,
            "detected {freq} Hz, expected 3150 Hz ± 10 Hz"
        );
    }

    /// Broadband white noise has no dominant narrow peak → `None`.
    #[test]
    fn test_detect_pilot_tone_no_peak() {
        use rand::RngExt;
        let sr = 44100_u32;
        let n = 65536_usize;
        let mut rng = rand::rng();
        let noise: Vec<f32> = (0..n)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();

        let result = detect_pilot_tone(&noise, sr);
        assert!(
            result.is_none(),
            "broadband noise should not trigger pilot-tone detection"
        );
    }

    /// Auto-detecting on a 3150 Hz pilot signal should return roughly the same
    /// reference frequency as manually passing 3150.0.
    #[test]
    fn test_wow_corrector_auto_detect_matches_manual() {
        use rand::RngExt;
        let sr = 44100_u32;
        let n = 65536_usize;
        let mut signal = sine(3150.0, sr, n);
        let mut rng = rand::rng();
        for s in signal.iter_mut() {
            *s += rng.random_range(-0.01_f32..0.01_f32);
        }

        let manual_ref = 3150.0_f32;
        let auto_ref = detect_pilot_tone(&signal, sr).expect("auto-detect must find 3150 Hz pilot");

        // Both references must agree within 10 Hz.
        assert!(
            (auto_ref - manual_ref).abs() < 10.0,
            "auto={auto_ref} Hz, manual={manual_ref} Hz — should agree within 10 Hz"
        );
    }

    /// Input shorter than 4096 samples returns `None`.
    #[test]
    fn test_detect_pilot_tone_too_short() {
        let samples = sine(3150.0, 44100, 100);
        assert!(detect_pilot_tone(&samples, 44100).is_none());
    }

    #[test]
    fn test_detect_instantaneous_frequency_sine() {
        let sr = 44100u32;
        let samples = sine(440.0, sr, 44100);
        let freqs = FlutterDetector::detect_instantaneous_frequency(&samples, sr);
        assert!(!freqs.is_empty(), "should return frequency estimates");
        // Each estimate should be in a reasonable audio range
        for f in &freqs {
            assert!(*f > 0.0 && *f < 20000.0, "frequency {f} Hz out of range");
        }
    }

    #[test]
    fn test_detect_instantaneous_frequency_empty() {
        let freqs = FlutterDetector::detect_instantaneous_frequency(&[], 44100);
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_detect_instantaneous_frequency_short() {
        let samples = vec![0.0f32; 100];
        let freqs = FlutterDetector::detect_instantaneous_frequency(&samples, 44100);
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_flutter_severity_excellent() {
        let m = FlutterMeasurement {
            wow_hz: 0.5,
            flutter_hz: 10.0,
            depth_pct: 0.05,
            wow_score: 0.1,
            flutter_score: 0.1,
        };
        assert_eq!(m.classify(), FlutterSeverity::Excellent);
    }

    #[test]
    fn test_flutter_severity_good() {
        let m = FlutterMeasurement {
            wow_hz: 1.0,
            flutter_hz: 15.0,
            depth_pct: 0.2,
            wow_score: 0.2,
            flutter_score: 0.2,
        };
        assert_eq!(m.classify(), FlutterSeverity::Good);
    }

    #[test]
    fn test_flutter_severity_fair() {
        let m = FlutterMeasurement {
            wow_hz: 2.0,
            flutter_hz: 20.0,
            depth_pct: 0.7,
            wow_score: 0.5,
            flutter_score: 0.5,
        };
        assert_eq!(m.classify(), FlutterSeverity::Fair);
    }

    #[test]
    fn test_flutter_severity_poor() {
        let m = FlutterMeasurement {
            wow_hz: 3.0,
            flutter_hz: 30.0,
            depth_pct: 2.0,
            wow_score: 0.9,
            flutter_score: 0.9,
        };
        assert_eq!(m.classify(), FlutterSeverity::Poor);
    }

    #[test]
    fn test_flutter_corrector_preserves_length() {
        let sr = 44100u32;
        let samples = sine(440.0, sr, 4410);
        let measurement = FlutterMeasurement {
            wow_hz: 1.0,
            flutter_hz: 10.0,
            depth_pct: 0.3,
            wow_score: 0.3,
            flutter_score: 0.3,
        };
        let corrected = FlutterCorrector::correct(&samples, sr, &measurement);
        assert_eq!(corrected.len(), samples.len());
    }

    #[test]
    fn test_flutter_corrector_empty() {
        let measurement = FlutterMeasurement {
            wow_hz: 1.0,
            flutter_hz: 10.0,
            depth_pct: 0.3,
            wow_score: 0.3,
            flutter_score: 0.3,
        };
        let result = FlutterCorrector::correct(&[], 44100, &measurement);
        assert!(result.is_empty());
    }

    #[test]
    fn test_flutter_corrector_zero_depth_identity() {
        let sr = 44100u32;
        let samples = sine(440.0, sr, 4410);
        let measurement = FlutterMeasurement {
            wow_hz: 1.0,
            flutter_hz: 10.0,
            depth_pct: 0.0, // no modulation
            wow_score: 0.0,
            flutter_score: 0.0,
        };
        let corrected = FlutterCorrector::correct(&samples, sr, &measurement);
        // With zero depth the warp is zero so output ≈ input
        for (a, b) in samples.iter().zip(corrected.iter()) {
            assert!((a - b).abs() < 1e-5, "deviation {}", (a - b).abs());
        }
    }

    #[test]
    fn test_speed_variation_estimator_unity() {
        let sr = 44100u32;
        // Pure 440 Hz sine
        let samples = sine(440.0, sr, 44100);
        let speeds = SpeedVariationEstimator::estimate(&samples, 440.0, sr);
        assert!(!speeds.is_empty());
        // All speed estimates should be positive and finite
        for s in &speeds {
            assert!(
                s.is_finite() && *s > 0.0,
                "speed {s} should be positive and finite"
            );
        }
    }

    #[test]
    fn test_speed_variation_estimator_invalid_ref() {
        let result = SpeedVariationEstimator::estimate(&[0.0; 100], 0.0, 44100);
        assert!(result.is_empty());
    }
}
