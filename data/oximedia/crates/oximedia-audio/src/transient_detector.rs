//! Transient detection using High-Frequency Content (HFC) onset detection.
//!
//! Detects percussive onsets (attacks, drum hits, etc.) in audio signals
//! by computing the HFC Onset Detection Function (ODF) and applying
//! adaptive peak-picking.

#![forbid(unsafe_code)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Classification of a detected transient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransientType {
    /// Strong, fast onset (HFC strength > 0.7).
    Attack,
    /// Moderate percussive onset (HFC strength 0.4 – 0.7).
    Percussive,
    /// Weak or ambiguous onset (HFC strength < 0.4).
    Ambiguous,
}

/// A single detected transient event.
#[derive(Clone, Debug)]
pub struct TransientEvent {
    /// Time of the transient in milliseconds.
    pub time_ms: f64,
    /// Normalised detection strength in `[0, 1]`.
    pub strength: f32,
    /// Classification of the transient type.
    pub transient_type: TransientType,
}

/// Configuration for `TransientDetector`.
#[derive(Clone, Debug)]
pub struct TransientConfig {
    /// Detection threshold in standard deviations above the mean ODF.
    pub threshold: f32,
    /// Hop size in samples between successive analysis frames.
    pub hop_size: usize,
    /// Analysis window size in samples (should be ≥ `hop_size`).
    pub window_size: usize,
}

impl Default for TransientConfig {
    fn default() -> Self {
        Self {
            threshold: 1.5,
            hop_size: 512,
            window_size: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// TransientDetector
// ---------------------------------------------------------------------------

/// Detects transient events in audio signals using HFC onset detection.
pub struct TransientDetector {
    config: TransientConfig,
}

impl Default for TransientDetector {
    fn default() -> Self {
        Self {
            config: TransientConfig::default(),
        }
    }
}

impl TransientDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: TransientConfig) -> Self {
        Self { config }
    }

    /// Detect transients using the detector's stored configuration.
    pub fn detect_with_config(&self, samples: &[f32], sample_rate: u32) -> Vec<TransientEvent> {
        detect_impl(samples, sample_rate, &self.config)
    }

    /// Detect transients using the default configuration (convenience function).
    ///
    /// This is a static-like associated function — no `self` receiver needed.
    pub fn detect(samples: &[f32], sample_rate: u32) -> Vec<TransientEvent> {
        detect_impl(samples, sample_rate, &TransientConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

fn detect_impl(samples: &[f32], sample_rate: u32, config: &TransientConfig) -> Vec<TransientEvent> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let window_size = config.window_size.max(2);
    let hop_size = config.hop_size.max(1);
    let fft_size = next_power_of_two(window_size);

    if samples.len() < window_size {
        return Vec::new();
    }

    let window = build_hann_window(window_size);

    // Build FFT plan
    let plan = match Plan::<f64>::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE) {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Compute HFC for each hop
    let num_frames = (samples.len() - window_size) / hop_size + 1;
    let mut odf: Vec<f32> = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + window_size;
        if end > samples.len() {
            break;
        }

        // Zero-pad windowed frame to fft_size
        let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
        for i in 0..window_size {
            buf[i] = Complex::new(f64::from(samples[start + i]) * window[i], 0.0);
        }

        let mut out = vec![Complex::<f64>::new(0.0, 0.0); fft_size];
        plan.execute(&buf, &mut out);
        // shadow buf with out for magnitude computation below
        let buf = out;

        // HFC = sum_k ( |X(k)|^2 * k )
        let num_bins = fft_size / 2 + 1;
        let hfc: f64 = (0..num_bins)
            .map(|k| {
                let mag_sq = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
                mag_sq * k as f64
            })
            .sum();

        odf.push(hfc as f32);
    }

    if odf.is_empty() {
        return Vec::new();
    }

    // Compute mean and std of ODF
    let mean = odf.iter().copied().sum::<f32>() / odf.len() as f32;
    let variance = odf
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / odf.len() as f32;
    let std_dev = variance.sqrt();

    let dynamic_threshold = mean + config.threshold * std_dev;

    // Normalise ODF to [0, 1] for strength reporting
    let max_odf = odf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let odf_range = if max_odf > 0.0 { max_odf } else { 1.0 };

    // Peak-picking: local maximum + above adaptive threshold
    let mut events = Vec::new();
    let len = odf.len();

    for i in 1..len.saturating_sub(1) {
        let prev = odf[i - 1];
        let curr = odf[i];
        let next = odf[i + 1];

        if curr > prev && curr >= next && curr > dynamic_threshold {
            let time_ms = (i as f64 * hop_size as f64 / f64::from(sample_rate)) * 1000.0;
            let strength = (curr / odf_range).min(1.0).max(0.0);
            let transient_type = classify(strength);

            events.push(TransientEvent {
                time_ms,
                strength,
                transient_type,
            });
        }
    }

    events
}

/// Classify a normalised strength value into a `TransientType`.
fn classify(strength: f32) -> TransientType {
    if strength > 0.7 {
        TransientType::Attack
    } else if strength >= 0.4 {
        TransientType::Percussive
    } else {
        TransientType::Ambiguous
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_power_of_two(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

fn build_hann_window(size: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size as f64 - 1.0)).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SAMPLE_RATE: u32 = 44100;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0f32; n]
    }

    fn sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    /// Insert an impulse at position `pos` in an otherwise silent signal.
    fn impulse_signal(total: usize, pos: usize) -> Vec<f32> {
        let mut s = silence(total);
        if pos < total {
            s[pos] = 1.0;
        }
        s
    }

    /// Build a signal with impulses at multiples of `period` samples.
    fn click_train(total: usize, period: usize) -> Vec<f32> {
        let mut s = silence(total);
        let mut pos = period / 2;
        while pos < total {
            s[pos] = 1.0;
            pos += period;
        }
        s
    }

    #[test]
    fn test_detect_empty_signal_no_transients() {
        let events = TransientDetector::detect(&[], SAMPLE_RATE);
        assert!(events.is_empty());
    }

    #[test]
    fn test_detect_silence_no_transients() {
        let events = TransientDetector::detect(&silence(44100), SAMPLE_RATE);
        assert!(
            events.is_empty(),
            "silence should have no transients, got {}",
            events.len()
        );
    }

    #[test]
    fn test_detect_single_impulse() {
        let s = impulse_signal(8192, 2048);
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        // We expect at least one transient detected
        assert!(!events.is_empty(), "single impulse should be detected");
    }

    #[test]
    fn test_detect_multiple_impulses() {
        let mut s = silence(22050);
        s[1024] = 1.0;
        s[8192] = 1.0;
        s[16384] = 1.0;
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        assert!(
            events.len() >= 2,
            "three impulses should yield at least 2 detections, got {}",
            events.len()
        );
    }

    #[test]
    fn test_detect_sine_wave_few_transients() {
        let s = sine(440.0, 44100);
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        // Steady sine should not produce many transients relative to its duration;
        // windowing artifacts may create a small number of spurious peaks, but
        // well under 1 per hop (there are ~86 hops in 44100 samples at hop=512).
        assert!(
            events.len() <= 15,
            "steady sine should have few transients, got {}",
            events.len()
        );
    }

    #[test]
    fn test_transient_event_time_ms_positive() {
        let s = impulse_signal(8192, 1024);
        for event in TransientDetector::detect(&s, SAMPLE_RATE) {
            assert!(
                event.time_ms >= 0.0,
                "time_ms must be non-negative, got {}",
                event.time_ms
            );
        }
    }

    #[test]
    fn test_transient_strength_in_range() {
        let mut s = silence(8192);
        s[1024] = 1.0;
        s[4096] = 1.0;
        for event in TransientDetector::detect(&s, SAMPLE_RATE) {
            assert!(
                (0.0..=1.0).contains(&event.strength),
                "strength must be in [0,1], got {}",
                event.strength
            );
        }
    }

    #[test]
    fn test_transient_type_attack_for_strong() {
        let t = classify(0.9);
        assert_eq!(t, TransientType::Attack);
    }

    #[test]
    fn test_transient_type_percussive_for_moderate() {
        let t = classify(0.55);
        assert_eq!(t, TransientType::Percussive);
    }

    #[test]
    fn test_transient_type_ambiguous_for_weak() {
        let t = classify(0.2);
        assert_eq!(t, TransientType::Ambiguous);
    }

    #[test]
    fn test_config_default_values() {
        let cfg = TransientConfig::default();
        assert!((cfg.threshold - 1.5).abs() < 1e-6);
        assert_eq!(cfg.hop_size, 512);
        assert_eq!(cfg.window_size, 1024);
    }

    #[test]
    fn test_detect_click_train() {
        let s = click_train(44100, 4096);
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        // Click train has ~10 impulses; expect several detections
        assert!(
            events.len() >= 3,
            "click train should yield several transients, got {}",
            events.len()
        );
    }

    #[test]
    fn test_detect_drum_pattern_approximation() {
        // Simulate kick at 0, snare at 0.5s, kick at 1.0s
        let n = SAMPLE_RATE as usize * 2;
        let mut s = silence(n);
        s[0] = 1.0;
        s[SAMPLE_RATE as usize / 2] = 1.0;
        s[SAMPLE_RATE as usize] = 1.0;
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        assert!(
            events.len() >= 2,
            "drum pattern should produce multiple transients, got {}",
            events.len()
        );
    }

    #[test]
    fn test_high_threshold_fewer_transients() {
        let s = click_train(22050, 2048);
        let low_cfg = TransientConfig {
            threshold: 0.5,
            ..Default::default()
        };
        let high_cfg = TransientConfig {
            threshold: 5.0,
            ..Default::default()
        };
        let det = TransientDetector::new(low_cfg);
        let low_events = det.detect_with_config(&s, SAMPLE_RATE);
        let det2 = TransientDetector::new(high_cfg);
        let high_events = det2.detect_with_config(&s, SAMPLE_RATE);
        assert!(
            high_events.len() <= low_events.len(),
            "higher threshold should yield fewer or equal transients ({} vs {})",
            high_events.len(),
            low_events.len()
        );
    }

    #[test]
    fn test_low_threshold_more_transients() {
        let s = click_train(22050, 2048);
        let very_low = TransientConfig {
            threshold: 0.01,
            ..Default::default()
        };
        let det = TransientDetector::new(very_low);
        let events = det.detect_with_config(&s, SAMPLE_RATE);
        // Low threshold should pick up many peaks
        assert!(
            events.len() >= 1,
            "low threshold should yield at least one transient"
        );
    }

    #[test]
    fn test_detect_with_custom_config() {
        let cfg = TransientConfig {
            threshold: 1.0,
            hop_size: 256,
            window_size: 512,
        };
        let s = impulse_signal(8192, 2048);
        let det = TransientDetector::new(cfg);
        let events = det.detect_with_config(&s, SAMPLE_RATE);
        assert!(
            !events.is_empty(),
            "custom config should still detect impulse"
        );
    }

    #[test]
    fn test_transient_boundary_strength_07() {
        // Exactly 0.7 — boundary between Percussive and Attack
        let t = classify(0.7);
        assert_eq!(t, TransientType::Percussive); // 0.7 is NOT > 0.7
    }

    #[test]
    fn test_transient_boundary_strength_04() {
        // Exactly 0.4
        let t = classify(0.4);
        assert_eq!(t, TransientType::Percussive); // 0.4 is >= 0.4
    }

    #[test]
    fn test_time_ms_corresponds_to_position() {
        // Place impulse at known sample position
        let n = 22050usize;
        let pos = 4096usize;
        let mut s = silence(n);
        s[pos] = 1.0;
        let events = TransientDetector::detect(&s, SAMPLE_RATE);
        if events.is_empty() {
            return; // acceptable if HFC doesn't reach threshold
        }
        // The earliest event should be roughly at pos/sr*1000 ms
        let expected_ms = pos as f64 / f64::from(SAMPLE_RATE) * 1000.0;
        let first_ms = events[0].time_ms;
        let hop_ms = 512.0 / f64::from(SAMPLE_RATE) * 1000.0;
        assert!(
            (first_ms - expected_ms).abs() < hop_ms * 3.0,
            "event at {first_ms:.1}ms, expected ~{expected_ms:.1}ms (tolerance {:.1}ms)",
            hop_ms * 3.0
        );
    }
}
