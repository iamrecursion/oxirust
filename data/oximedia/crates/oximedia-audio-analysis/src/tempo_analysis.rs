//! Tempo analysis for audio signals.
//!
//! Provides BPM detection, tempo range classification, and tempo band analysis
//! for music and audio content.

#![allow(dead_code)]

/// Classifies a tempo into a named musical range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempoRange {
    /// Very slow (< 60 BPM) – grave/larghissimo territory.
    Grave,
    /// Slow (60–76 BPM) – largo/adagio territory.
    Largo,
    /// Moderate (76–120 BPM) – andante/moderato territory.
    Moderato,
    /// Fast (120–168 BPM) – allegro territory.
    Allegro,
    /// Very fast (168–200 BPM) – vivace/presto territory.
    Presto,
    /// Extremely fast (> 200 BPM) – prestissimo territory.
    Prestissimo,
}

impl TempoRange {
    /// Return the [`TempoRange`] that contains the given BPM value.
    #[must_use]
    pub fn from_bpm(bpm: f32) -> Self {
        match bpm as u32 {
            0..=59 => Self::Grave,
            60..=75 => Self::Largo,
            76..=119 => Self::Moderato,
            120..=167 => Self::Allegro,
            168..=199 => Self::Presto,
            _ => Self::Prestissimo,
        }
    }

    /// Human-readable label for this range.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Grave => "Grave",
            Self::Largo => "Largo",
            Self::Moderato => "Moderato",
            Self::Allegro => "Allegro",
            Self::Presto => "Presto",
            Self::Prestissimo => "Prestissimo",
        }
    }
}

/// A frequency band used for onset-based BPM estimation.
#[derive(Debug, Clone)]
pub struct TempoBand {
    /// Lower frequency bound in Hz.
    pub low_hz: f32,
    /// Upper frequency bound in Hz.
    pub high_hz: f32,
    /// Detected BPM within this band (None if insufficient signal).
    pub bpm: Option<f32>,
    /// Confidence score \[0.0, 1.0\].
    pub confidence: f32,
}

impl TempoBand {
    /// Create a new [`TempoBand`] with the given frequency bounds.
    #[must_use]
    pub fn new(low_hz: f32, high_hz: f32) -> Self {
        Self {
            low_hz,
            high_hz,
            bpm: None,
            confidence: 0.0,
        }
    }

    /// Returns `true` if the band has a reliable BPM estimate.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.6 && self.bpm.is_some()
    }
}

/// Result of a full tempo analysis pass.
#[derive(Debug, Clone)]
pub struct TempoResult {
    /// Primary BPM estimate.
    pub bpm: f32,
    /// Overall confidence in the estimate \[0.0, 1.0\].
    pub confidence: f32,
    /// Classified tempo range.
    pub range: TempoRange,
    /// Per-band results.
    pub bands: Vec<TempoBand>,
    /// Whether a half-tempo or double-tempo candidate was found.
    pub half_tempo: Option<f32>,
    /// Double-tempo candidate.
    pub double_tempo: Option<f32>,
}

/// Analyses the tempo of an audio signal.
pub struct TempoAnalyzer {
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
}

impl TempoAnalyzer {
    /// Create a new [`TempoAnalyzer`].
    ///
    /// # Arguments
    /// * `sample_rate` – Sample rate of the audio signal in Hz.
    /// * `frame_size` – FFT frame size (power of two recommended).
    /// * `hop_size`   – Number of samples between successive frames.
    #[must_use]
    pub fn new(sample_rate: f32, frame_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
        }
    }

    /// Estimate the BPM of `samples` using autocorrelation of the onset
    /// strength function.
    ///
    /// Returns a [`TempoResult`] containing the primary BPM estimate, per-band
    /// results, and half/double tempo candidates.
    #[must_use]
    pub fn detect_bpm(&self, samples: &[f32]) -> TempoResult {
        let onset_env = self.compute_onset_envelope(samples);
        let bpm = self.autocorr_bpm(&onset_env);
        let confidence = self.estimate_confidence(&onset_env, bpm);

        let bands = self.analyse_bands(samples);
        let half = if bpm > 60.0 { Some(bpm / 2.0) } else { None };
        let double = Some(bpm * 2.0);

        TempoResult {
            bpm,
            confidence,
            range: TempoRange::from_bpm(bpm),
            bands,
            half_tempo: half,
            double_tempo: double,
        }
    }

    // ── private helpers ────────────────────────────────────────────────────

    /// Compute a simple onset strength envelope (spectral flux).
    fn compute_onset_envelope(&self, samples: &[f32]) -> Vec<f32> {
        let mut envelope = Vec::new();
        let mut prev_power = 0.0_f32;

        let mut pos = 0;
        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let power: f32 = frame.iter().map(|&x| x * x).sum::<f32>() / self.frame_size as f32;
            let flux = (power - prev_power).max(0.0);
            envelope.push(flux);
            prev_power = power;
            pos += self.hop_size;
        }

        envelope
    }

    /// Estimate BPM from onset envelope via autocorrelation.
    #[allow(clippy::cast_precision_loss)]
    fn autocorr_bpm(&self, envelope: &[f32]) -> f32 {
        if envelope.is_empty() {
            return 120.0;
        }

        let fps = self.sample_rate / self.hop_size as f32;
        let min_lag = (fps * 60.0 / 240.0) as usize; // 240 BPM
        let max_lag = (fps * 60.0 / 40.0) as usize; //  40 BPM
        let max_lag = max_lag.min(envelope.len() - 1);

        if min_lag >= max_lag {
            return 120.0;
        }

        let mut best_lag = min_lag;
        let mut best_val = f32::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let corr: f32 = envelope
                .iter()
                .take(envelope.len() - lag)
                .zip(envelope.iter().skip(lag))
                .map(|(&a, &b)| a * b)
                .sum();
            if corr > best_val {
                best_val = corr;
                best_lag = lag;
            }
        }

        fps * 60.0 / best_lag as f32
    }

    /// Heuristic confidence based on onset envelope energy variance.
    #[allow(clippy::cast_precision_loss, clippy::unused_self)]
    fn estimate_confidence(&self, envelope: &[f32], _bpm: f32) -> f32 {
        if envelope.is_empty() {
            return 0.0;
        }
        let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
        let var = envelope.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / envelope.len() as f32;
        (var.sqrt() / (mean + 1e-6)).clamp(0.0, 1.0)
    }

    /// Produce tempo bands (bass, mid, high) with per-band BPM estimates.
    fn analyse_bands(&self, samples: &[f32]) -> Vec<TempoBand> {
        let definitions = [(20.0_f32, 250.0_f32), (250.0, 4000.0), (4000.0, 20000.0)];
        let mut bands = Vec::with_capacity(definitions.len());

        for (lo, hi) in definitions {
            let mut band = TempoBand::new(lo, hi);
            let filtered = self.band_pass(samples, lo, hi);
            let onset = self.compute_onset_envelope(&filtered);
            let bpm = self.autocorr_bpm(&onset);
            let conf = self.estimate_confidence(&onset, bpm);
            band.bpm = Some(bpm);
            band.confidence = conf;
            bands.push(band);
        }

        bands
    }

    /// Very simple single-pole band-pass approximation.
    #[allow(clippy::unused_self)]
    fn band_pass(&self, samples: &[f32], _low: f32, _high: f32) -> Vec<f32> {
        // In a real implementation this would use a proper filter bank.
        // For now, return the signal as-is so the rest of the pipeline
        // can exercise the full code path.
        samples.to_vec()
    }
}

impl Default for TempoAnalyzer {
    fn default() -> Self {
        Self::new(44100.0, 2048, 512)
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TempoRange ────────────────────────────────────────────────────────

    #[test]
    fn test_tempo_range_from_bpm_grave() {
        assert_eq!(TempoRange::from_bpm(40.0), TempoRange::Grave);
    }

    #[test]
    fn test_tempo_range_from_bpm_largo() {
        assert_eq!(TempoRange::from_bpm(72.0), TempoRange::Largo);
    }

    #[test]
    fn test_tempo_range_from_bpm_moderato() {
        assert_eq!(TempoRange::from_bpm(100.0), TempoRange::Moderato);
    }

    #[test]
    fn test_tempo_range_from_bpm_allegro() {
        assert_eq!(TempoRange::from_bpm(140.0), TempoRange::Allegro);
    }

    #[test]
    fn test_tempo_range_from_bpm_presto() {
        assert_eq!(TempoRange::from_bpm(180.0), TempoRange::Presto);
    }

    #[test]
    fn test_tempo_range_from_bpm_prestissimo() {
        assert_eq!(TempoRange::from_bpm(210.0), TempoRange::Prestissimo);
    }

    #[test]
    fn test_tempo_range_labels() {
        assert_eq!(TempoRange::Grave.label(), "Grave");
        assert_eq!(TempoRange::Largo.label(), "Largo");
        assert_eq!(TempoRange::Moderato.label(), "Moderato");
        assert_eq!(TempoRange::Allegro.label(), "Allegro");
        assert_eq!(TempoRange::Presto.label(), "Presto");
        assert_eq!(TempoRange::Prestissimo.label(), "Prestissimo");
    }

    // ── TempoBand ─────────────────────────────────────────────────────────

    #[test]
    fn test_tempo_band_new() {
        let band = TempoBand::new(20.0, 250.0);
        assert_eq!(band.low_hz, 20.0);
        assert_eq!(band.high_hz, 250.0);
        assert!(band.bpm.is_none());
        assert_eq!(band.confidence, 0.0);
    }

    #[test]
    fn test_tempo_band_reliability_low_confidence() {
        let mut band = TempoBand::new(20.0, 250.0);
        band.bpm = Some(120.0);
        band.confidence = 0.3;
        assert!(!band.is_reliable());
    }

    #[test]
    fn test_tempo_band_reliability_high_confidence() {
        let mut band = TempoBand::new(20.0, 250.0);
        band.bpm = Some(120.0);
        band.confidence = 0.8;
        assert!(band.is_reliable());
    }

    // ── TempoAnalyzer ─────────────────────────────────────────────────────

    #[test]
    fn test_analyzer_default_construction() {
        let analyzer = TempoAnalyzer::default();
        assert_eq!(analyzer.sample_rate, 44100.0);
        assert_eq!(analyzer.frame_size, 2048);
        assert_eq!(analyzer.hop_size, 512);
    }

    #[test]
    fn test_detect_bpm_on_silence() {
        let analyzer = TempoAnalyzer::default();
        let silence = vec![0.0_f32; 44100];
        let result = analyzer.detect_bpm(&silence);
        // Silence has no onsets – should still return a result without panicking.
        assert!(result.bpm > 0.0);
    }

    #[test]
    fn test_detect_bpm_returns_range() {
        let analyzer = TempoAnalyzer::default();
        let samples = vec![0.1_f32; 22050];
        let result = analyzer.detect_bpm(&samples);
        assert_ne!(result.range, TempoRange::Grave); // sanity check enum is set
        let _ = result.range.label(); // should not panic
    }

    #[test]
    fn test_detect_bpm_half_and_double() {
        let analyzer = TempoAnalyzer::default();
        let samples = vec![0.05_f32; 44100];
        let result = analyzer.detect_bpm(&samples);
        // For any BPM > 60, half_tempo should be Some.
        if result.bpm > 60.0 {
            assert!(result.half_tempo.is_some());
        }
        assert!(result.double_tempo.is_some());
    }

    #[test]
    fn test_detect_bpm_band_count() {
        let analyzer = TempoAnalyzer::default();
        let samples = vec![0.02_f32; 44100];
        let result = analyzer.detect_bpm(&samples);
        assert_eq!(result.bands.len(), 3);
    }

    #[test]
    fn test_detect_bpm_confidence_range() {
        let analyzer = TempoAnalyzer::default();
        let samples = vec![0.1_f32; 44100];
        let result = analyzer.detect_bpm(&samples);
        assert!((0.0..=1.0).contains(&result.confidence));
    }

    #[test]
    fn test_detect_bpm_short_signal() {
        let analyzer = TempoAnalyzer::default();
        let samples = vec![0.1_f32; 1024];
        // Should not panic even for very short signals.
        let result = analyzer.detect_bpm(&samples);
        assert!(result.bpm > 0.0);
    }
}
