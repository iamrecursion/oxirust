//! Instrument identification from audio features.

use crate::spectral::SpectralFeatures;
use crate::transient::TransientResult;

/// Musical instrument classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instrument {
    /// Piano
    Piano,
    /// Guitar (acoustic or electric)
    Guitar,
    /// Violin or string instrument
    Violin,
    /// Flute or wind instrument
    Flute,
    /// Trumpet or brass instrument
    Trumpet,
    /// Drums or percussion
    Drums,
    /// Bass guitar or bass
    Bass,
    /// Synthesizer
    Synthesizer,
    /// Vocals
    Vocals,
    /// Unknown/unclassified
    Unknown,
}

/// Detect instrument from audio features.
///
/// Uses spectral and temporal features to classify instruments:
/// - Piano: Sharp attacks, wide spectral range, harmonic
/// - Guitar: Moderate attacks, characteristic formants
/// - Violin: Smooth attacks, strong harmonics
/// - Flute: Pure tone, low harmonics
/// - Drums: Very sharp attacks, noise-like spectrum
/// - Bass: Low spectral centroid
/// - Vocals: Formant structure, vibrato
///
/// # Arguments
/// * `spectral` - Spectral features
/// * `transients` - Transient detection result
/// * `f0` - Fundamental frequency (if detected)
///
/// # Returns
/// Detected instrument
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn detect_instrument(
    spectral: &SpectralFeatures,
    transients: &TransientResult,
    f0: Option<f32>,
) -> Instrument {
    // Feature extraction
    let is_harmonic = spectral.flatness < 0.3;
    let is_noisy = spectral.flatness > 0.7;
    let has_strong_transients = transients.avg_strength > 0.5;
    let low_centroid = spectral.centroid < 500.0;
    let high_centroid = spectral.centroid > 2000.0;

    // Decision tree classification
    if is_noisy && has_strong_transients {
        return Instrument::Drums;
    }

    if low_centroid && is_harmonic {
        return Instrument::Bass;
    }

    if let Some(fundamental) = f0 {
        // Voice characteristics
        if (80.0..=1000.0).contains(&fundamental) {
            // Check for formant-like structure in spectrum
            let has_formants = check_formant_structure(&spectral.magnitude_spectrum);
            if has_formants {
                return Instrument::Vocals;
            }
        }

        // Flute characteristics
        if fundamental >= 250.0 && is_harmonic && spectral.flatness < 0.15 {
            return Instrument::Flute;
        }

        // Piano characteristics
        if has_strong_transients && is_harmonic && spectral.bandwidth > 1000.0 {
            return Instrument::Piano;
        }

        // Guitar characteristics
        if is_harmonic && !has_strong_transients && fundamental >= 80.0 {
            return Instrument::Guitar;
        }

        // Violin characteristics
        if is_harmonic && high_centroid && fundamental >= 200.0 {
            return Instrument::Violin;
        }

        // Trumpet characteristics
        if is_harmonic && spectral.centroid > 800.0 && spectral.centroid < 2000.0 {
            return Instrument::Trumpet;
        }
    }

    // Synthesizer (often has non-natural spectral characteristics)
    if !is_noisy && !check_formant_structure(&spectral.magnitude_spectrum) {
        return Instrument::Synthesizer;
    }

    Instrument::Unknown
}

/// Check for formant-like structure in spectrum.
fn check_formant_structure(spectrum: &[f32]) -> bool {
    if spectrum.len() < 20 {
        return false;
    }

    // Look for multiple peaks in spectrum
    let mut peaks = 0;
    for i in 2..(spectrum.len() - 2) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 {
            peaks += 1;
        }
    }

    // Formants typically have 2-4 prominent peaks
    (2..=4).contains(&peaks)
}

/// Detect instrument with confidence scores.
#[must_use]
pub fn detect_instrument_scores(
    spectral: &SpectralFeatures,
    transients: &TransientResult,
    f0: Option<f32>,
) -> Vec<(Instrument, f32)> {
    let mut scores = vec![
        (Instrument::Piano, 0.0),
        (Instrument::Guitar, 0.0),
        (Instrument::Violin, 0.0),
        (Instrument::Flute, 0.0),
        (Instrument::Trumpet, 0.0),
        (Instrument::Drums, 0.0),
        (Instrument::Bass, 0.0),
        (Instrument::Vocals, 0.0),
        (Instrument::Synthesizer, 0.0),
    ];

    // Drums score
    if spectral.flatness > 0.5 && transients.avg_strength > 0.4 {
        scores[5].1 = 0.8;
    }

    // Bass score
    if spectral.centroid < 500.0 {
        scores[6].1 = 0.7;
    }

    if let Some(fundamental) = f0 {
        // Vocals score
        if (80.0..=1000.0).contains(&fundamental)
            && check_formant_structure(&spectral.magnitude_spectrum)
        {
            scores[7].1 = 0.8;
        }

        // Flute score
        if fundamental >= 250.0 && spectral.flatness < 0.15 {
            scores[3].1 = 0.7;
        }

        // Piano score
        if transients.avg_strength > 0.5 && spectral.bandwidth > 1000.0 {
            scores[0].1 = 0.7;
        }

        // Guitar score
        if spectral.flatness < 0.3 && fundamental >= 80.0 {
            scores[1].1 = 0.6;
        }

        // Violin score
        if spectral.centroid > 2000.0 && spectral.flatness < 0.3 {
            scores[2].1 = 0.6;
        }

        // Trumpet score
        if spectral.centroid > 800.0 && spectral.centroid < 2000.0 {
            scores[4].1 = 0.6;
        }
    }

    scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

// ---------------------------------------------------------------------------
// Per-instrument onset detection
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Frequency band associated with a class of instruments.
///
/// Variants carry no data; see [`InstrumentBand::range_hz`] for the Hz bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrumentBand {
    /// Kick drum band: 0–200 Hz.
    Kick,
    /// Bass guitar / bass synth band: 20–300 Hz.
    Bass,
    /// Mid-range instruments (snare, voice, guitar): 300–3 000 Hz.
    MidRange,
    /// High-range / treble instruments: 3 000–20 000 Hz.
    Treble,
    /// Hi-hat / cymbal band: 8 000–20 000 Hz.
    HiHat,
}

impl InstrumentBand {
    /// Return the `(low_hz, high_hz)` frequency range for this band.
    #[must_use]
    pub fn range_hz(self) -> (f64, f64) {
        match self {
            Self::Kick => (0.0, 200.0),
            Self::Bass => (20.0, 300.0),
            Self::MidRange => (300.0, 3_000.0),
            Self::Treble => (3_000.0, 20_000.0),
            Self::HiHat => (8_000.0, 20_000.0),
        }
    }

    /// All instrument bands in a stable order.
    #[must_use]
    pub fn all() -> [Self; 5] {
        [
            Self::Kick,
            Self::Bass,
            Self::MidRange,
            Self::Treble,
            Self::HiHat,
        ]
    }

    /// Per-band onset detection threshold (normalised spectral flux units).
    ///
    /// Low-frequency bands need a lower threshold because their spectral flux
    /// values are intrinsically smaller (fewer bins).
    fn threshold(self) -> f64 {
        match self {
            Self::Kick => 0.02,
            Self::Bass => 0.015,
            Self::MidRange => 0.025,
            Self::Treble => 0.02,
            Self::HiHat => 0.018,
        }
    }
}

/// Per-instrument onset detector using band-limited spectral flux.
///
/// For each [`InstrumentBand`] the detector computes the half-wave-rectified
/// spectral flux restricted to the band's frequency range, applies a simple
/// adaptive mean threshold, and returns onset times in seconds.
pub struct InstrumentOnsetDetector {
    /// FFT size used for analysis.
    pub window_size: usize,
    /// Hop size (samples) between successive analysis frames.
    pub hop_size: usize,
}

impl Default for InstrumentOnsetDetector {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 256,
        }
    }
}

impl InstrumentOnsetDetector {
    /// Create a new detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect onset times (seconds) per instrument band.
    ///
    /// Returns a `HashMap` mapping each [`InstrumentBand`] to a (possibly empty)
    /// `Vec<f64>` of onset times in seconds.
    #[must_use]
    pub fn detect_onsets_per_instrument(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> HashMap<InstrumentBand, Vec<f64>> {
        let mut result: HashMap<InstrumentBand, Vec<f64>> = HashMap::new();
        for band in InstrumentBand::all() {
            let onsets = self.detect_band_onsets(samples, sample_rate, band);
            result.insert(band, onsets);
        }
        result
    }

    /// Detect onsets in a single frequency band.
    fn detect_band_onsets(
        &self,
        samples: &[f32],
        sample_rate: u32,
        band: InstrumentBand,
    ) -> Vec<f64> {
        let (low_hz, high_hz) = band.range_hz();
        let sr = sample_rate as f64;
        let n = self.window_size;

        // Map band edges to FFT bin indices.
        let low_bin = ((low_hz * n as f64 / sr).round() as usize).min(n / 2);
        let high_bin = ((high_hz * n as f64 / sr).round() as usize).min(n / 2);

        if high_bin <= low_bin {
            return Vec::new();
        }

        // Compute magnitude spectra per frame.
        let frames = compute_magnitude_frames(samples, n, self.hop_size);
        if frames.len() < 2 {
            return Vec::new();
        }

        // Band-limited spectral flux (HWR) per frame transition.
        let mut flux_values: Vec<f64> = Vec::with_capacity(frames.len() - 1);
        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];
            let mut flux = 0.0_f64;
            let end = high_bin.min(curr.len()).min(prev.len());
            let start = low_bin.min(end);
            for k in start..end {
                let diff = f64::from(curr[k]) - f64::from(prev[k]);
                if diff > 0.0 {
                    flux += diff;
                }
            }
            flux_values.push(flux);
        }

        // Adaptive mean threshold over a sliding window.
        let window_frames = 8_usize;
        let base_threshold = band.threshold();
        let mut onsets = Vec::new();

        for i in 0..flux_values.len() {
            let start = i.saturating_sub(window_frames);
            let slice = &flux_values[start..i + 1];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            let adaptive_threshold = (mean + base_threshold).max(base_threshold);

            if flux_values[i] > adaptive_threshold {
                // Simple peak-picking: only record if it's a local max.
                let prev_flux = if i > 0 { flux_values[i - 1] } else { 0.0 };
                let next_flux = flux_values.get(i + 1).copied().unwrap_or(0.0);
                if flux_values[i] >= prev_flux && flux_values[i] >= next_flux {
                    let time_s = (i + 1) as f64 * self.hop_size as f64 / sr;
                    onsets.push(time_s);
                }
            }
        }

        onsets
    }
}

/// Compute a sequence of magnitude spectra (one per hop) via a simple DFT.
fn compute_magnitude_frames(samples: &[f32], window_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    use std::f64::consts::PI;

    let num_bins = window_size / 2 + 1;
    let mut frames = Vec::new();

    let mut pos = 0_usize;
    while pos + window_size <= samples.len() {
        let frame = &samples[pos..pos + window_size];
        let mut magnitudes = vec![0.0_f32; num_bins];

        for k in 0..num_bins {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (j, &s) in frame.iter().enumerate() {
                // Hann window
                let w = 0.5 * (1.0 - (2.0 * PI * j as f64 / (window_size - 1) as f64).cos());
                let angle = -2.0 * PI * k as f64 * j as f64 / window_size as f64;
                let sv = f64::from(s) * w;
                re += sv * angle.cos();
                im += sv * angle.sin();
            }
            magnitudes[k] = (re * re + im * im).sqrt() as f32;
        }

        frames.push(magnitudes);
        pos += hop_size;
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_detection() {
        // Create features for drums (noisy, transient)
        let spectral = SpectralFeatures {
            centroid: 1000.0,
            flatness: 0.8,
            crest: 5.0,
            bandwidth: 3000.0,
            rolloff: 5000.0,
            flux: 0.0,
            magnitude_spectrum: vec![0.5; 100],
        };

        let transients = TransientResult {
            transient_times: vec![0.1, 0.2, 0.3],
            onset_strength: vec![0.8, 0.7, 0.9],
            num_transients: 3,
            avg_strength: 0.8,
        };

        let instrument = detect_instrument(&spectral, &transients, None);
        assert_eq!(instrument, Instrument::Drums);
    }

    #[test]
    fn test_instrument_scores() {
        let spectral = SpectralFeatures {
            centroid: 300.0,
            flatness: 0.2,
            crest: 3.0,
            bandwidth: 500.0,
            rolloff: 800.0,
            flux: 0.0,
            magnitude_spectrum: vec![0.5; 100],
        };

        let transients = TransientResult::default();

        let scores = detect_instrument_scores(&spectral, &transients, Some(100.0));

        // Bass should have high score due to low centroid
        let bass_score = scores
            .iter()
            .find(|(i, _)| *i == Instrument::Bass)
            .expect("unexpected None/Err")
            .1;
        assert!(bass_score > 0.5);
    }

    // -----------------------------------------------------------------------
    // Per-instrument onset detection tests (Item 2)
    // -----------------------------------------------------------------------

    /// Generate a burst of low-frequency sine waves separated by silence —
    /// simulating kick drum hits — then verify onsets appear in the Kick band.
    /// Uses a lower sample rate to keep test fast.
    #[test]
    fn test_kick_onset_in_low_band() {
        // Use 8000 Hz sample rate to keep signal short while preserving 80 Hz.
        let sample_rate: u32 = 8000;
        let sr_f = sample_rate as f64;

        // Three 80 Hz bursts (50 ms each) separated by 100 ms silence.
        let burst_len = (0.05 * sr_f) as usize;
        let silence_len = (0.1 * sr_f) as usize;
        let freq = 80.0_f64; // inside Kick band (0-200 Hz)

        let mut samples: Vec<f32> = Vec::new();
        for _ in 0..3 {
            for i in 0..burst_len {
                let t = i as f64 / sr_f;
                samples.push((2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.8);
            }
            for _ in 0..silence_len {
                samples.push(0.0);
            }
        }

        // Use small window/hop to stay fast.
        let detector = InstrumentOnsetDetector {
            window_size: 256,
            hop_size: 64,
        };
        let onsets = detector.detect_onsets_per_instrument(&samples, sample_rate);

        let kick_onsets = onsets
            .get(&InstrumentBand::Kick)
            .expect("Kick band missing");

        // We expect at least one onset detected in the kick band.
        assert!(
            !kick_onsets.is_empty(),
            "Expected at least one kick onset, got none"
        );
        // All detected times should be within the signal duration.
        let duration = samples.len() as f64 / sr_f;
        for &t in kick_onsets {
            assert!(
                t <= duration + 0.1,
                "Onset time {t} s is out of signal range"
            );
        }
    }

    /// Verify that different bands can detect onsets independently:
    /// a hi-hat signal (high-frequency) should yield detections in HiHat/Treble
    /// but not in the Kick band.
    #[test]
    fn test_instrument_onset_bands_independent() {
        // 22050 Hz sample rate so 8 kHz hi-hat is representable.
        let sample_rate: u32 = 22050;
        let sr_f = sample_rate as f64;

        // Three 10 kHz bursts (20 ms) — inside HiHat band (8-20 kHz).
        let burst_len = (0.02 * sr_f) as usize;
        let silence_len = (0.08 * sr_f) as usize;
        let freq = 10_000.0_f64;

        let mut samples: Vec<f32> = Vec::new();
        for _ in 0..3 {
            for i in 0..burst_len {
                let t = i as f64 / sr_f;
                samples.push((2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.8);
            }
            for _ in 0..silence_len {
                samples.push(0.0);
            }
        }

        let detector = InstrumentOnsetDetector {
            window_size: 256,
            hop_size: 64,
        };
        let onsets = detector.detect_onsets_per_instrument(&samples, sample_rate);

        // HiHat band should have onsets.
        let hihat_onsets = onsets.get(&InstrumentBand::HiHat).expect("HiHat missing");
        assert!(
            !hihat_onsets.is_empty(),
            "Expected hi-hat onsets in HiHat band, got none"
        );

        // Kick band (0-200 Hz) should have fewer (ideally zero) onsets from a
        // 12 kHz signal. We don't assert exactly zero to allow for DFT leakage,
        // but it should be fewer than HiHat.
        let kick_onsets = onsets.get(&InstrumentBand::Kick).expect("Kick missing");
        assert!(
            kick_onsets.len() <= hihat_onsets.len(),
            "Kick band ({}) should have ≤ onsets than HiHat band ({})",
            kick_onsets.len(),
            hihat_onsets.len()
        );
    }
}
