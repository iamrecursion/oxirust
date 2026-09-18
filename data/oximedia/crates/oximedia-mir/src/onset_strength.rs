//! Onset strength envelope computation for Music Information Retrieval.
//!
//! Provides spectral-flux, HFC-energy, and phase-deviation based onset
//! detection functions along with a streaming `OnsetDetector`.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Method used to compute the onset function value for each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnsetFunction {
    /// Spectral flux: half-wave rectified sum of magnitude increases.
    SpectralFlux,
    /// High-frequency content energy: weighted sum of spectral magnitudes.
    HfcEnergy,
    /// Phase deviation: deviation of unwrapped phase from expected trajectory.
    Phase,
}

impl OnsetFunction {
    /// Short human-readable description of the onset function.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::SpectralFlux => "Half-wave rectified spectral flux",
            Self::HfcEnergy => "High-frequency content energy",
            Self::Phase => "Phase-deviation based onset detection",
        }
    }
}

/// A single onset analysis frame produced by the detector.
#[derive(Debug, Clone)]
pub struct OnsetFrame {
    /// Frame index (hop-aligned).
    pub frame_index: usize,
    /// Computed onset function value for this frame.
    pub value: f32,
    /// Whether this frame has been labelled as an onset peak.
    pub onset_flag: bool,
    /// Timestamp in seconds.
    pub time_s: f32,
}

impl OnsetFrame {
    /// Create a new onset frame.
    #[must_use]
    pub fn new(frame_index: usize, value: f32, time_s: f32) -> Self {
        Self {
            frame_index,
            value,
            onset_flag: false,
            time_s,
        }
    }

    /// Returns `true` if this frame has been marked as an onset.
    #[must_use]
    pub fn is_onset(&self) -> bool {
        self.onset_flag
    }
}

/// Streaming onset detector that accumulates frames and exposes the
/// onset-strength envelope.
#[derive(Debug, Clone)]
pub struct OnsetDetector {
    function: OnsetFunction,
    sample_rate: f32,
    hop_size: usize,
    /// Threshold multiplier applied over the local mean to decide onset peaks.
    threshold_factor: f32,
    frames: Vec<OnsetFrame>,
    /// Previous magnitude spectrum, used for spectral-flux computation.
    prev_magnitudes: Vec<f32>,
    /// Previous phase spectrum, used for phase-deviation computation.
    prev_phase: Vec<f32>,
    prev_prev_phase: Vec<f32>,
}

impl OnsetDetector {
    /// Create a new onset detector.
    ///
    /// * `function`        – which onset function to use
    /// * `sample_rate`     – audio sample rate in Hz
    /// * `hop_size`        – analysis hop size in samples
    /// * `threshold_factor`– scalar multiplier over local mean for peak picking (e.g. 1.2)
    #[must_use]
    pub fn new(
        function: OnsetFunction,
        sample_rate: f32,
        hop_size: usize,
        threshold_factor: f32,
    ) -> Self {
        Self {
            function,
            sample_rate,
            hop_size,
            threshold_factor,
            frames: Vec::new(),
            prev_magnitudes: Vec::new(),
            prev_phase: Vec::new(),
            prev_prev_phase: Vec::new(),
        }
    }

    /// Add a spectral frame (magnitude + phase) and compute the onset value.
    ///
    /// Returns the computed onset function value for this frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_frame(&mut self, magnitudes: &[f32], phases: &[f32]) -> f32 {
        let frame_index = self.frames.len();
        let time_s = (frame_index * self.hop_size) as f32 / self.sample_rate;

        let value = match self.function {
            OnsetFunction::SpectralFlux => self.spectral_flux(magnitudes),
            OnsetFunction::HfcEnergy => Self::hfc_energy(magnitudes),
            OnsetFunction::Phase => self.phase_deviation(phases),
        };

        // Update history
        self.prev_magnitudes = magnitudes.to_vec();
        self.prev_prev_phase = self.prev_phase.clone();
        self.prev_phase = phases.to_vec();

        self.frames
            .push(OnsetFrame::new(frame_index, value, time_s));
        value
    }

    /// Compute the onset-strength envelope (all frame values, normalised 0–1).
    #[must_use]
    pub fn compute_envelope(&self) -> Vec<f32> {
        let values: Vec<f32> = self.frames.iter().map(|f| f.value).collect();
        let max = values.iter().copied().fold(0.0_f32, f32::max);
        if max > 0.0 {
            values.iter().map(|v| v / max).collect()
        } else {
            values
        }
    }

    /// Run peak-picking on the accumulated frames and mark onsets in-place.
    ///
    /// Uses a simple local-mean threshold: a frame is an onset if its value
    /// exceeds `threshold_factor * mean(surrounding window)`.
    pub fn pick_onsets(&mut self) {
        let n = self.frames.len();
        if n == 0 {
            return;
        }
        let window = 8_usize;
        let values: Vec<f32> = self.frames.iter().map(|f| f.value).collect();

        for i in 0..n {
            let lo = i.saturating_sub(window);
            let hi = (i + window + 1).min(n);
            let local_mean: f32 = values[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
            let threshold = local_mean * self.threshold_factor;
            if values[i] > threshold {
                // Simple local maximum check within ±2 frames
                let peak_lo = i.saturating_sub(2);
                let peak_hi = (i + 3).min(n);
                let is_local_max = values[peak_lo..peak_hi].iter().all(|&v| v <= values[i]);
                if is_local_max {
                    self.frames[i].onset_flag = true;
                }
            }
        }
    }

    /// Return an iterator over all frames currently stored.
    #[must_use]
    pub fn frames(&self) -> &[OnsetFrame] {
        &self.frames
    }

    /// Return only the frames marked as onsets.
    #[must_use]
    pub fn onset_frames(&self) -> Vec<&OnsetFrame> {
        self.frames.iter().filter(|f| f.onset_flag).collect()
    }

    // ---- internal helpers ----

    fn spectral_flux(&self, mags: &[f32]) -> f32 {
        if self.prev_magnitudes.is_empty() || self.prev_magnitudes.len() != mags.len() {
            return 0.0;
        }
        mags.iter()
            .zip(self.prev_magnitudes.iter())
            .map(|(cur, prev)| {
                let diff = cur - prev;
                if diff > 0.0 {
                    diff
                } else {
                    0.0
                }
            })
            .sum()
    }

    #[allow(clippy::cast_precision_loss)]
    fn hfc_energy(mags: &[f32]) -> f32 {
        mags.iter()
            .enumerate()
            .map(|(k, &m)| m * m * (k + 1) as f32)
            .sum()
    }

    fn phase_deviation(&self, phases: &[f32]) -> f32 {
        if self.prev_phase.is_empty()
            || self.prev_prev_phase.is_empty()
            || phases.len() != self.prev_phase.len()
        {
            return 0.0;
        }
        phases
            .iter()
            .zip(self.prev_phase.iter())
            .zip(self.prev_prev_phase.iter())
            .map(|((p, pp), ppp)| {
                // Predicted phase = 2 * prev - prev_prev
                let predicted = 2.0 * pp - ppp;
                let diff = (p - predicted + std::f32::consts::PI)
                    .rem_euclid(2.0 * std::f32::consts::PI)
                    - std::f32::consts::PI;
                diff.abs()
            })
            .sum::<f32>()
            / phases.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn linspace(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    // OnsetFunction tests

    #[test]
    fn test_onset_function_descriptions_non_empty() {
        for f in [
            OnsetFunction::SpectralFlux,
            OnsetFunction::HfcEnergy,
            OnsetFunction::Phase,
        ] {
            assert!(!f.description().is_empty());
        }
    }

    #[test]
    fn test_spectral_flux_description() {
        assert!(OnsetFunction::SpectralFlux.description().contains("flux"));
    }

    #[test]
    fn test_hfc_description() {
        assert!(OnsetFunction::HfcEnergy
            .description()
            .contains("High-frequency"));
    }

    #[test]
    fn test_phase_description() {
        assert!(OnsetFunction::Phase.description().contains("Phase"));
    }

    #[test]
    fn test_onset_function_equality() {
        assert_eq!(OnsetFunction::SpectralFlux, OnsetFunction::SpectralFlux);
        assert_ne!(OnsetFunction::SpectralFlux, OnsetFunction::Phase);
    }

    // OnsetFrame tests

    #[test]
    fn test_onset_frame_default_no_onset() {
        let frame = OnsetFrame::new(0, 0.5, 0.0);
        assert!(!frame.is_onset());
    }

    #[test]
    fn test_onset_frame_mark_onset() {
        let mut frame = OnsetFrame::new(1, 1.0, 0.023);
        frame.onset_flag = true;
        assert!(frame.is_onset());
    }

    #[test]
    fn test_onset_frame_fields() {
        let frame = OnsetFrame::new(5, 0.75, 0.116);
        assert_eq!(frame.frame_index, 5);
        assert!((frame.value - 0.75).abs() < 1e-6);
        assert!((frame.time_s - 0.116).abs() < 1e-4);
    }

    // OnsetDetector tests

    #[test]
    fn test_detector_no_frames_empty_envelope() {
        let det = OnsetDetector::new(OnsetFunction::SpectralFlux, 44100.0, 512, 1.2);
        let env = det.compute_envelope();
        assert!(env.is_empty());
    }

    #[test]
    fn test_detector_hfc_first_frame_nonzero() {
        let mut det = OnsetDetector::new(OnsetFunction::HfcEnergy, 44100.0, 512, 1.2);
        let mags: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let val = det.add_frame(&mags, &zeros(10));
        assert!(val > 0.0);
    }

    #[test]
    fn test_detector_spectral_flux_zero_on_first_frame() {
        let mut det = OnsetDetector::new(OnsetFunction::SpectralFlux, 44100.0, 512, 1.2);
        let mags = linspace(8);
        let val = det.add_frame(&mags, &zeros(8));
        // First frame has no previous, so flux == 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_detector_spectral_flux_increases_after_transient() {
        let mut det = OnsetDetector::new(OnsetFunction::SpectralFlux, 44100.0, 512, 1.2);
        let silence = zeros(16);
        let loud: Vec<f32> = vec![1.0; 16];
        det.add_frame(&silence, &zeros(16));
        let flux_val = det.add_frame(&loud, &zeros(16));
        assert!(flux_val > 0.0);
    }

    #[test]
    fn test_detector_envelope_normalised() {
        let mut det = OnsetDetector::new(OnsetFunction::HfcEnergy, 44100.0, 512, 1.2);
        for _ in 0..5 {
            let mags: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
            det.add_frame(&mags, &zeros(8));
        }
        let env = det.compute_envelope();
        let max = env.iter().cloned().fold(0.0_f32, f32::max);
        assert!((max - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pick_onsets_on_empty_does_not_panic() {
        let mut det = OnsetDetector::new(OnsetFunction::SpectralFlux, 44100.0, 512, 1.2);
        det.pick_onsets(); // should not panic
    }

    #[test]
    fn test_onset_frames_count() {
        let mut det = OnsetDetector::new(OnsetFunction::HfcEnergy, 44100.0, 512, 1.2);
        for i in 0..12 {
            let mags: Vec<f32> = vec![i as f32; 8];
            det.add_frame(&mags, &zeros(8));
        }
        assert_eq!(det.frames().len(), 12);
    }

    #[test]
    fn test_phase_deviation_zero_on_insufficient_history() {
        let mut det = OnsetDetector::new(OnsetFunction::Phase, 44100.0, 512, 1.2);
        let phases = vec![0.1_f32; 16];
        let v0 = det.add_frame(&zeros(16), &phases); // no history
        let v1 = det.add_frame(&zeros(16), &phases); // one previous, no prev_prev
        assert_eq!(v0, 0.0);
        assert_eq!(v1, 0.0);
    }

    #[test]
    fn test_pick_onsets_marks_at_least_one_on_impulse() {
        let mut det = OnsetDetector::new(OnsetFunction::SpectralFlux, 44100.0, 512, 1.05);
        // Feed silence then a loud frame then silence again
        let silence = zeros(16);
        let loud: Vec<f32> = vec![10.0; 16];
        for _ in 0..5 {
            det.add_frame(&silence, &zeros(16));
        }
        det.add_frame(&loud, &zeros(16));
        for _ in 0..5 {
            det.add_frame(&silence, &zeros(16));
        }
        det.pick_onsets();
        let onsets = det.onset_frames();
        assert!(!onsets.is_empty());
    }
}
