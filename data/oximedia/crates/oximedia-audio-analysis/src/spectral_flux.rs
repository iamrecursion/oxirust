//! Spectral flux onset detection.
//!
//! Spectral flux measures the frame-to-frame change in the magnitude spectrum.
//! It is a widely used feature for onset detection, percussive event
//! identification, and music segmentation.

#![allow(dead_code)]

/// Sensitivity preset for the spectral flux detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxSensitivity {
    /// Low sensitivity -- only strong transients.
    Low,
    /// Medium sensitivity -- general-purpose default.
    Medium,
    /// High sensitivity -- picks up subtle spectral changes.
    High,
    /// Custom -- caller provides an explicit threshold.
    Custom,
}

impl FluxSensitivity {
    /// Default threshold value for this sensitivity preset.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        match self {
            Self::Low => 0.6,
            Self::Medium | Self::Custom => 0.35,
            Self::High => 0.15,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Custom => "Custom",
        }
    }
}

/// One frame of spectral-flux analysis.
#[derive(Debug, Clone)]
pub struct FluxFrame {
    /// Centre timestamp of this frame in seconds.
    pub time_s: f32,
    /// Raw spectral flux value (non-negative).
    pub flux: f32,
    /// Whether this frame exceeds the onset threshold.
    pub is_onset: bool,
}

impl FluxFrame {
    /// Create a new [`FluxFrame`].
    #[must_use]
    pub fn new(time_s: f32, flux: f32, is_onset: bool) -> Self {
        Self {
            time_s,
            flux,
            is_onset,
        }
    }
}

/// Result of a complete spectral-flux analysis pass.
#[derive(Debug, Clone)]
pub struct FluxResult {
    /// Per-frame spectral-flux values.
    pub frames: Vec<FluxFrame>,
    /// Detected onset times in seconds.
    pub onsets: Vec<f32>,
    /// Mean spectral flux across all frames.
    pub mean_flux: f32,
    /// Peak spectral flux.
    pub peak_flux: f32,
}

/// Spectral-flux-based onset detector.
pub struct SpectralFluxDetector {
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    sensitivity: FluxSensitivity,
    threshold: f32,
}

impl SpectralFluxDetector {
    /// Create a new [`SpectralFluxDetector`].
    ///
    /// # Arguments
    /// * `sample_rate` -- Sample rate in Hz.
    /// * `frame_size`  -- FFT frame size (power of two recommended).
    /// * `hop_size`    -- Hop between successive frames.
    /// * `sensitivity` -- Sensitivity preset.
    #[must_use]
    pub fn new(
        sample_rate: f32,
        frame_size: usize,
        hop_size: usize,
        sensitivity: FluxSensitivity,
    ) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            sensitivity,
            threshold: sensitivity.threshold(),
        }
    }

    /// Create a detector with a custom threshold value.
    #[must_use]
    pub fn with_threshold(
        sample_rate: f32,
        frame_size: usize,
        hop_size: usize,
        threshold: f32,
    ) -> Self {
        Self {
            sample_rate,
            frame_size,
            hop_size,
            sensitivity: FluxSensitivity::Custom,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Return the current onset threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Return the sensitivity setting.
    #[must_use]
    pub fn sensitivity(&self) -> FluxSensitivity {
        self.sensitivity
    }

    /// Analyse the full audio signal and produce a [`FluxResult`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32]) -> FluxResult {
        let magnitudes = self.compute_frame_magnitudes(samples);
        let flux_values = self.compute_flux(&magnitudes);

        let mean_flux = if flux_values.is_empty() {
            0.0
        } else {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        };
        let peak_flux = flux_values.iter().copied().fold(0.0_f32, f32::max);

        // Adaptive threshold: mean + threshold * (peak - mean)
        let adaptive = mean_flux + self.threshold * (peak_flux - mean_flux);

        let mut frames = Vec::with_capacity(flux_values.len());
        let mut onsets = Vec::new();

        for (i, &flux) in flux_values.iter().enumerate() {
            let time_s = (i * self.hop_size) as f32 / self.sample_rate;
            let is_onset = flux > adaptive;
            if is_onset {
                onsets.push(time_s);
            }
            frames.push(FluxFrame::new(time_s, flux, is_onset));
        }

        FluxResult {
            frames,
            onsets,
            mean_flux,
            peak_flux,
        }
    }

    /// Count the number of onsets detected in `samples`.
    #[must_use]
    pub fn count_onsets(&self, samples: &[f32]) -> usize {
        self.detect(samples).onsets.len()
    }

    // -- private helpers --

    /// Compute per-frame magnitude sums (cheap proxy for a full FFT magnitude
    /// spectrum).
    #[allow(clippy::cast_precision_loss)]
    fn compute_frame_magnitudes(&self, samples: &[f32]) -> Vec<f32> {
        let mut mags = Vec::new();
        let mut pos = 0;
        while pos + self.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.frame_size];
            let energy: f32 = frame.iter().map(|&x| x * x).sum::<f32>() / self.frame_size as f32;
            mags.push(energy.sqrt());
            pos += self.hop_size;
        }
        mags
    }

    /// Half-wave rectified difference (spectral flux).
    #[allow(clippy::unused_self)]
    fn compute_flux(&self, magnitudes: &[f32]) -> Vec<f32> {
        if magnitudes.len() < 2 {
            return vec![0.0; magnitudes.len()];
        }
        let mut flux = vec![0.0_f32];
        for i in 1..magnitudes.len() {
            let diff = (magnitudes[i] - magnitudes[i - 1]).max(0.0);
            flux.push(diff);
        }
        flux
    }
}

impl Default for SpectralFluxDetector {
    fn default() -> Self {
        Self::new(44100.0, 2048, 512, FluxSensitivity::Medium)
    }
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_thresholds() {
        assert!(FluxSensitivity::Low.threshold() > FluxSensitivity::Medium.threshold());
        assert!(FluxSensitivity::Medium.threshold() > FluxSensitivity::High.threshold());
    }

    #[test]
    fn test_sensitivity_labels() {
        assert_eq!(FluxSensitivity::Low.label(), "Low");
        assert_eq!(FluxSensitivity::Medium.label(), "Medium");
        assert_eq!(FluxSensitivity::High.label(), "High");
        assert_eq!(FluxSensitivity::Custom.label(), "Custom");
    }

    #[test]
    fn test_flux_frame_new() {
        let f = FluxFrame::new(1.5, 0.42, true);
        assert_eq!(f.time_s, 1.5);
        assert_eq!(f.flux, 0.42);
        assert!(f.is_onset);
    }

    #[test]
    fn test_default_detector() {
        let det = SpectralFluxDetector::default();
        assert_eq!(det.sample_rate, 44100.0);
        assert_eq!(det.frame_size, 2048);
        assert_eq!(det.hop_size, 512);
        assert_eq!(det.sensitivity(), FluxSensitivity::Medium);
    }

    #[test]
    fn test_custom_threshold_clamped() {
        let det = SpectralFluxDetector::with_threshold(44100.0, 2048, 512, 2.0);
        assert_eq!(det.threshold(), 1.0);
        let det2 = SpectralFluxDetector::with_threshold(44100.0, 2048, 512, -0.5);
        assert_eq!(det2.threshold(), 0.0);
    }

    #[test]
    fn test_detect_silence() {
        let det = SpectralFluxDetector::default();
        let silence = vec![0.0_f32; 44100];
        let result = det.detect(&silence);
        assert_eq!(result.mean_flux, 0.0);
        assert_eq!(result.peak_flux, 0.0);
        assert!(result.onsets.is_empty());
    }

    #[test]
    fn test_detect_constant_signal_no_onsets() {
        let det = SpectralFluxDetector::default();
        let constant = vec![0.5_f32; 44100];
        let result = det.detect(&constant);
        // Constant signal has no flux after the first frame.
        assert_eq!(result.peak_flux, 0.0);
    }

    #[test]
    fn test_detect_impulse_has_onset() {
        let det = SpectralFluxDetector::new(44100.0, 256, 128, FluxSensitivity::High);
        let mut samples = vec![0.0_f32; 44100];
        // Insert a loud impulse
        for s in samples[10000..10256].iter_mut() {
            *s = 0.9;
        }
        let result = det.detect(&samples);
        assert!(result.peak_flux > 0.0);
        assert!(!result.onsets.is_empty());
    }

    #[test]
    fn test_count_onsets_matches() {
        let det = SpectralFluxDetector::new(44100.0, 256, 128, FluxSensitivity::High);
        let mut samples = vec![0.0_f32; 44100];
        for s in samples[5000..5256].iter_mut() {
            *s = 0.8;
        }
        let count = det.count_onsets(&samples);
        let result = det.detect(&samples);
        assert_eq!(count, result.onsets.len());
    }

    #[test]
    fn test_flux_frames_length() {
        let det = SpectralFluxDetector::default();
        let samples = vec![0.1_f32; 44100];
        let result = det.detect(&samples);
        // Every frame should have a FluxFrame entry
        assert!(!result.frames.is_empty());
    }

    #[test]
    fn test_short_signal_no_panic() {
        let det = SpectralFluxDetector::default();
        let short = vec![0.1_f32; 100];
        let result = det.detect(&short);
        assert!(result.frames.is_empty());
        assert_eq!(result.mean_flux, 0.0);
    }

    #[test]
    fn test_mean_flux_non_negative() {
        let det = SpectralFluxDetector::default();
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let result = det.detect(&samples);
        assert!(result.mean_flux >= 0.0);
    }

    #[test]
    fn test_onset_times_monotonic() {
        let det = SpectralFluxDetector::new(44100.0, 256, 128, FluxSensitivity::High);
        let mut samples = vec![0.0_f32; 44100];
        for s in samples[4000..4256].iter_mut() {
            *s = 0.9;
        }
        for s in samples[20000..20256].iter_mut() {
            *s = 0.9;
        }
        let result = det.detect(&samples);
        for w in result.onsets.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }
}
