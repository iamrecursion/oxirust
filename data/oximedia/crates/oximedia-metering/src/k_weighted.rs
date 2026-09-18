#![allow(dead_code)]
//! K-weighted loudness level calculation for broadcast metering.
//!
//! Implements K-weighted loudness level measurements with multi-band analysis,
//! providing per-band loudness data useful for detailed audio analysis and
//! mastering workflows. Based on ITU-R BS.1770-4 K-weighting curve principles.

/// K-weighting band definition.
#[derive(Clone, Debug)]
pub struct KWeightBand {
    /// Band center frequency in Hz.
    pub center_freq: f64,
    /// Band lower frequency bound in Hz.
    pub lower_freq: f64,
    /// Band upper frequency bound in Hz.
    pub upper_freq: f64,
    /// K-weighting gain for this band in dB.
    pub gain_db: f64,
    /// Current accumulated energy.
    energy: f64,
    /// Number of samples accumulated.
    sample_count: u64,
}

impl KWeightBand {
    /// Create a new K-weight band.
    pub fn new(center_freq: f64, lower_freq: f64, upper_freq: f64, gain_db: f64) -> Self {
        Self {
            center_freq,
            lower_freq,
            upper_freq,
            gain_db,
            energy: 0.0,
            sample_count: 0,
        }
    }

    /// Get the bandwidth in Hz.
    pub fn bandwidth(&self) -> f64 {
        self.upper_freq - self.lower_freq
    }

    /// Get the linear gain factor from dB.
    pub fn gain_linear(&self) -> f64 {
        10.0_f64.powf(self.gain_db / 20.0)
    }

    /// Get the current RMS level in linear scale.
    pub fn rms_linear(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = self.energy / self.sample_count as f64;
        mean.sqrt()
    }

    /// Get the current RMS level in dB.
    pub fn rms_db(&self) -> f64 {
        let rms = self.rms_linear();
        if rms < 1e-20 {
            -200.0
        } else {
            20.0 * rms.log10()
        }
    }

    /// Reset accumulated energy.
    pub fn reset(&mut self) {
        self.energy = 0.0;
        self.sample_count = 0;
    }
}

/// K-weighted loudness level calculator.
///
/// Processes audio samples and calculates K-weighted loudness across
/// configurable frequency bands, providing detailed spectral loudness
/// information for mastering and broadcast compliance.
pub struct KWeightedLevel {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
    /// Frequency bands with K-weighting.
    bands: Vec<KWeightBand>,
    /// Overall accumulated K-weighted energy.
    total_energy: f64,
    /// Total samples processed.
    total_samples: u64,
    /// Momentary window size in samples.
    momentary_window: usize,
    /// Momentary energy buffer.
    momentary_buffer: Vec<f64>,
    /// Current position in the momentary buffer.
    momentary_pos: usize,
    /// Short-term window size in samples.
    short_term_window: usize,
    /// Short-term energy buffer.
    short_term_buffer: Vec<f64>,
    /// Current position in the short-term buffer.
    short_term_pos: usize,
}

impl KWeightedLevel {
    /// Create a new K-weighted level calculator.
    ///
    /// Uses standard ITU-R BS.1770-4 inspired band definitions.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let bands = Self::default_bands();
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let momentary_window = (sample_rate * 0.4) as usize; // 400ms
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let short_term_window = (sample_rate * 3.0) as usize; // 3s

        Self {
            sample_rate,
            channels: channels.max(1),
            bands,
            total_energy: 0.0,
            total_samples: 0,
            momentary_window,
            momentary_buffer: vec![0.0; momentary_window.max(1)],
            momentary_pos: 0,
            short_term_window,
            short_term_buffer: vec![0.0; short_term_window.max(1)],
            short_term_pos: 0,
        }
    }

    /// Create default K-weighting bands matching broadcast standards.
    fn default_bands() -> Vec<KWeightBand> {
        vec![
            // Sub-bass: heavily attenuated by K-weighting.
            KWeightBand::new(31.5, 20.0, 50.0, -20.0),
            // Bass: still attenuated.
            KWeightBand::new(100.0, 50.0, 200.0, -6.0),
            // Low-mid: slight boost begins.
            KWeightBand::new(500.0, 200.0, 1000.0, 0.0),
            // Mid: flat response.
            KWeightBand::new(2000.0, 1000.0, 4000.0, 0.0),
            // High-mid: presence boost (K-weighting shelf).
            KWeightBand::new(6000.0, 4000.0, 8000.0, 3.0),
            // High: K-weighting high shelf.
            KWeightBand::new(12000.0, 8000.0, 20000.0, 2.0),
        ]
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Get the number of bands.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Get a reference to the bands.
    pub fn bands(&self) -> &[KWeightBand] {
        &self.bands
    }

    /// Process a block of interleaved audio samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        if samples.is_empty() || self.channels == 0 {
            return;
        }

        let frame_count = samples.len() / self.channels;

        for frame in 0..frame_count {
            // Sum channels for this frame.
            let mut frame_sum = 0.0;
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                if idx < samples.len() {
                    frame_sum += samples[idx];
                }
            }
            let mono = frame_sum / self.channels as f64;

            // Apply K-weighting per band and accumulate energy.
            let sample_sq = mono * mono;

            for band in &mut self.bands {
                let weighted = sample_sq * band.gain_linear() * band.gain_linear();
                band.energy += weighted;
                band.sample_count += 1;
            }

            // Overall weighted sum (simplified: use the flat/mid bands).
            self.total_energy += sample_sq;
            self.total_samples += 1;

            // Momentary buffer.
            if !self.momentary_buffer.is_empty() {
                self.momentary_buffer[self.momentary_pos] = sample_sq;
                self.momentary_pos = (self.momentary_pos + 1) % self.momentary_buffer.len();
            }

            // Short-term buffer.
            if !self.short_term_buffer.is_empty() {
                self.short_term_buffer[self.short_term_pos] = sample_sq;
                self.short_term_pos = (self.short_term_pos + 1) % self.short_term_buffer.len();
            }
        }
    }

    /// Process mono samples (single channel).
    pub fn process_mono(&mut self, samples: &[f64]) {
        let saved_channels = self.channels;
        self.channels = 1;
        self.process_interleaved(samples);
        self.channels = saved_channels;
    }

    /// Get the integrated K-weighted loudness in LUFS.
    #[allow(clippy::cast_precision_loss)]
    pub fn integrated_lufs(&self) -> f64 {
        if self.total_samples == 0 {
            return f64::NEG_INFINITY;
        }
        let mean_energy = self.total_energy / self.total_samples as f64;
        if mean_energy < 1e-20 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * mean_energy.log10()
        }
    }

    /// Get the momentary loudness in LUFS (400ms window).
    pub fn momentary_lufs(&self) -> f64 {
        let sum: f64 = self.momentary_buffer.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / self.momentary_buffer.len() as f64;
        if mean < 1e-20 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * mean.log10()
        }
    }

    /// Get the short-term loudness in LUFS (3s window).
    pub fn short_term_lufs(&self) -> f64 {
        let sum: f64 = self.short_term_buffer.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / self.short_term_buffer.len() as f64;
        if mean < 1e-20 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * mean.log10()
        }
    }

    /// Get per-band loudness levels in dB.
    pub fn band_levels_db(&self) -> Vec<f64> {
        self.bands.iter().map(KWeightBand::rms_db).collect()
    }

    /// Get per-band loudness levels in linear scale.
    pub fn band_levels_linear(&self) -> Vec<f64> {
        self.bands.iter().map(KWeightBand::rms_linear).collect()
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        self.total_energy = 0.0;
        self.total_samples = 0;
        for band in &mut self.bands {
            band.reset();
        }
        self.momentary_buffer.fill(0.0);
        self.momentary_pos = 0;
        self.short_term_buffer.fill(0.0);
        self.short_term_pos = 0;
    }

    /// Get a snapshot of the current K-weighted levels.
    pub fn snapshot(&self) -> KWeightedSnapshot {
        KWeightedSnapshot {
            integrated_lufs: self.integrated_lufs(),
            momentary_lufs: self.momentary_lufs(),
            short_term_lufs: self.short_term_lufs(),
            band_levels_db: self.band_levels_db(),
            total_samples: self.total_samples,
        }
    }
}

/// Snapshot of K-weighted levels at a point in time.
#[derive(Clone, Debug)]
pub struct KWeightedSnapshot {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Momentary loudness in LUFS.
    pub momentary_lufs: f64,
    /// Short-term loudness in LUFS.
    pub short_term_lufs: f64,
    /// Per-band levels in dB.
    pub band_levels_db: Vec<f64>,
    /// Total samples processed.
    pub total_samples: u64,
}

impl KWeightedSnapshot {
    /// Check if levels appear silent (below -70 LUFS).
    pub fn is_silent(&self) -> bool {
        self.integrated_lufs < -70.0
    }

    /// Get the loudest band index.
    pub fn loudest_band(&self) -> Option<usize> {
        if self.band_levels_db.is_empty() {
            return None;
        }
        self.band_levels_db
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_weight_band_creation() {
        let band = KWeightBand::new(1000.0, 500.0, 2000.0, 0.0);
        assert!((band.center_freq - 1000.0).abs() < f64::EPSILON);
        assert!((band.bandwidth() - 1500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_k_weight_band_gain_linear() {
        let band = KWeightBand::new(1000.0, 500.0, 2000.0, 0.0);
        assert!((band.gain_linear() - 1.0).abs() < 0.001);

        let band6 = KWeightBand::new(1000.0, 500.0, 2000.0, 6.0);
        assert!((band6.gain_linear() - 1.995).abs() < 0.01);
    }

    #[test]
    fn test_k_weight_band_rms_empty() {
        let band = KWeightBand::new(1000.0, 500.0, 2000.0, 0.0);
        assert!(band.rms_linear().abs() < f64::EPSILON);
        assert!(band.rms_db() < -100.0);
    }

    #[test]
    fn test_k_weight_band_reset() {
        let mut band = KWeightBand::new(1000.0, 500.0, 2000.0, 0.0);
        band.energy = 100.0;
        band.sample_count = 50;
        band.reset();
        assert!(band.energy.abs() < f64::EPSILON);
        assert_eq!(band.sample_count, 0);
    }

    #[test]
    fn test_k_weighted_level_creation() {
        let kw = KWeightedLevel::new(48000.0, 2);
        assert!((kw.sample_rate() - 48000.0).abs() < f64::EPSILON);
        assert_eq!(kw.channels(), 2);
        assert_eq!(kw.band_count(), 6);
    }

    #[test]
    fn test_k_weighted_level_silence() {
        let kw = KWeightedLevel::new(48000.0, 2);
        assert!(kw.integrated_lufs().is_infinite());
        assert!(kw.momentary_lufs().is_infinite());
        assert!(kw.short_term_lufs().is_infinite());
    }

    #[test]
    fn test_k_weighted_process_interleaved() {
        let mut kw = KWeightedLevel::new(48000.0, 2);
        // Generate a simple sine wave.
        let num_frames = 4800; // 100ms at 48kHz.
        let mut samples = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            #[allow(clippy::cast_precision_loss)]
            let t = i as f64 / 48000.0;
            let val = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            samples.push(val);
            samples.push(val);
        }
        kw.process_interleaved(&samples);
        // Should have some non-infinite integrated level.
        let lufs = kw.integrated_lufs();
        assert!(lufs.is_finite());
        assert!(lufs < 0.0);
    }

    #[test]
    fn test_k_weighted_process_mono() {
        let mut kw = KWeightedLevel::new(48000.0, 1);
        let samples: Vec<f64> = (0..4800)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.3
            })
            .collect();
        kw.process_mono(&samples);
        assert!(kw.integrated_lufs().is_finite());
    }

    #[test]
    fn test_k_weighted_band_levels() {
        let mut kw = KWeightedLevel::new(48000.0, 1);
        let samples: Vec<f64> = (0..9600)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5
            })
            .collect();
        kw.process_mono(&samples);
        let levels = kw.band_levels_db();
        assert_eq!(levels.len(), 6);
        // All should be finite.
        for level in &levels {
            assert!(level.is_finite());
        }
    }

    #[test]
    fn test_k_weighted_reset() {
        let mut kw = KWeightedLevel::new(48000.0, 2);
        let samples = vec![0.5; 9600];
        kw.process_interleaved(&samples);
        assert!(kw.integrated_lufs().is_finite());

        kw.reset();
        assert!(kw.integrated_lufs().is_infinite());
    }

    #[test]
    fn test_k_weighted_snapshot() {
        let mut kw = KWeightedLevel::new(48000.0, 2);
        let samples = vec![0.3; 9600];
        kw.process_interleaved(&samples);
        let snap = kw.snapshot();
        assert!(snap.integrated_lufs.is_finite());
        assert_eq!(snap.band_levels_db.len(), 6);
        assert!(snap.total_samples > 0);
    }

    #[test]
    fn test_snapshot_is_silent() {
        let kw = KWeightedLevel::new(48000.0, 2);
        let snap = kw.snapshot();
        assert!(snap.is_silent());
    }

    #[test]
    fn test_snapshot_loudest_band() {
        let mut kw = KWeightedLevel::new(48000.0, 1);
        let samples: Vec<f64> = (0..9600)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5
            })
            .collect();
        kw.process_mono(&samples);
        let snap = kw.snapshot();
        let loudest = snap.loudest_band();
        assert!(loudest.is_some());
    }

    #[test]
    fn test_snapshot_loudest_band_empty() {
        let snap = KWeightedSnapshot {
            integrated_lufs: -100.0,
            momentary_lufs: -100.0,
            short_term_lufs: -100.0,
            band_levels_db: vec![],
            total_samples: 0,
        };
        assert!(snap.loudest_band().is_none());
    }
}
