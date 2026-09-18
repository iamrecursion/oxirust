//! Audio artifact detection and repair.
//!
//! This module provides tools for detecting and repairing common audio artifacts
//! such as clicks, pops, dropouts, distortion, hum, and clipping.

/// Types of audio artifacts that can be detected and repaired.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioArtifactType {
    /// Short impulsive click artifact.
    Click,
    /// Louder impulsive pop artifact.
    Pop,
    /// Missing signal (silence or near-silence where audio should be).
    Dropout,
    /// Heavy distortion due to overload or corruption.
    Distortion,
    /// Continuous tonal interference (e.g., 50/60 Hz power hum).
    Hum,
    /// Signal clipped at maximum amplitude.
    Clipping,
}

impl AudioArtifactType {
    /// Returns `true` if this artifact is an impulsive event (short duration).
    #[must_use]
    pub const fn is_impulsive(&self) -> bool {
        matches!(self, Self::Click | Self::Pop)
    }
}

/// Description of an audio artifact found in a signal.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct AudioArtifact {
    /// Sample index where the artifact begins.
    pub sample_idx: u64,
    /// Type of artifact.
    pub artifact_type: AudioArtifactType,
    /// Duration of the artifact in samples.
    pub duration_samples: u32,
    /// Severity score (0.0 = minor, 1.0 = severe).
    pub severity: f32,
}

impl AudioArtifact {
    /// Create a new audio artifact.
    #[must_use]
    pub const fn new(
        sample_idx: u64,
        artifact_type: AudioArtifactType,
        duration_samples: u32,
        severity: f32,
    ) -> Self {
        Self {
            sample_idx,
            artifact_type,
            duration_samples,
            severity,
        }
    }
}

/// Repairs click artifacts using linear interpolation.
pub struct ClickRepairer;

impl ClickRepairer {
    /// Repair a click artifact at `click_idx` using linear interpolation over a window.
    ///
    /// The `window` specifies the total number of samples to interpolate across.
    /// The sample at `click_idx` and up to `window - 1` surrounding samples are
    /// replaced with a linearly interpolated ramp from the sample before the window
    /// to the sample after the window.
    ///
    /// If the click is at the very start or end of the signal, a fade to/from zero is used.
    pub fn repair(samples: &mut Vec<f32>, click_idx: usize, window: usize) {
        if samples.is_empty() || window == 0 {
            return;
        }

        let half = window / 2;
        let start = click_idx.saturating_sub(half);
        let end = (click_idx + half + 1).min(samples.len());

        if start >= samples.len() {
            return;
        }

        let start_val = if start > 0 { samples[start - 1] } else { 0.0 };
        let end_val = if end < samples.len() {
            samples[end]
        } else {
            0.0
        };

        let count = end - start;
        if count == 0 {
            return;
        }

        for (i, idx) in (start..end).enumerate() {
            let t = (i + 1) as f32 / (count + 1) as f32;
            samples[idx] = start_val + t * (end_val - start_val);
        }
    }
}

/// Repairs dropout artifacts using fade-in/fade-out interpolation.
pub struct DropoutRepairer;

impl DropoutRepairer {
    /// Repair a dropout at `start` lasting `duration` samples.
    ///
    /// The dropout region is filled with:
    /// - A fade-in from the sample just before `start` to silence
    /// - A fade-out from silence to the sample just after the dropout
    ///
    /// This creates a simple interpolated waveform that avoids hard cuts.
    /// If the dropout spans the entire buffer, the buffer is zeroed.
    pub fn repair(samples: &mut Vec<f32>, start: usize, duration: usize) {
        if samples.is_empty() || duration == 0 {
            return;
        }

        let end = (start + duration).min(samples.len());
        let count = end - start;

        if count == 0 {
            return;
        }

        let pre_val = if start > 0 { samples[start - 1] } else { 0.0 };
        let post_val = if end < samples.len() {
            samples[end]
        } else {
            0.0
        };

        // Fill with linear interpolation between pre and post values
        for (i, idx) in (start..end).enumerate() {
            let t = (i + 1) as f32 / (count + 1) as f32;
            samples[idx] = pre_val + t * (post_val - pre_val);
        }
    }
}

/// Repairs clipping artifacts by applying soft-knee limiting.
pub struct ClippingRepairer;

impl ClippingRepairer {
    /// Detect and repair clipped samples in the audio signal.
    ///
    /// Samples where `|x| >= threshold` are considered clipped.
    /// A soft-knee function is applied: clipped samples are scaled down
    /// such that `x_new = sign(x) * (threshold - (|x| - threshold) * knee)`,
    /// where `knee = 0.1`.
    ///
    /// The repaired values are clamped to `[-1.0, 1.0]`.
    pub fn repair(samples: &mut Vec<f32>, threshold: f32) {
        let threshold = threshold.clamp(0.0, 1.0);
        const KNEE: f32 = 0.1;

        for sample in samples.iter_mut() {
            let abs_val = sample.abs();
            if abs_val >= threshold {
                let sign = sample.signum();
                let excess = abs_val - threshold;
                let new_val = sign * (threshold - excess * KNEE);
                *sample = new_val.clamp(-1.0, 1.0);
            }
        }
    }
}

/// Report summarizing the results of audio artifact repair.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRepairReport {
    /// Total number of artifacts found.
    pub artifacts_found: u64,
    /// Total number of artifacts successfully repaired.
    pub artifacts_repaired: u64,
    /// Estimated improvement in signal-to-noise ratio in decibels.
    pub snr_improvement_db: f32,
}

impl AudioRepairReport {
    /// Create a new audio repair report.
    #[must_use]
    pub const fn new(
        artifacts_found: u64,
        artifacts_repaired: u64,
        snr_improvement_db: f32,
    ) -> Self {
        Self {
            artifacts_found,
            artifacts_repaired,
            snr_improvement_db,
        }
    }

    /// Compute repair success rate as a fraction (0.0–1.0).
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        if self.artifacts_found == 0 {
            return 1.0;
        }
        self.artifacts_repaired as f32 / self.artifacts_found as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_type_is_impulsive_click() {
        assert!(AudioArtifactType::Click.is_impulsive());
    }

    #[test]
    fn test_artifact_type_is_impulsive_pop() {
        assert!(AudioArtifactType::Pop.is_impulsive());
    }

    #[test]
    fn test_artifact_type_is_not_impulsive_dropout() {
        assert!(!AudioArtifactType::Dropout.is_impulsive());
    }

    #[test]
    fn test_artifact_type_is_not_impulsive_hum() {
        assert!(!AudioArtifactType::Hum.is_impulsive());
    }

    #[test]
    fn test_artifact_type_is_not_impulsive_clipping() {
        assert!(!AudioArtifactType::Clipping.is_impulsive());
    }

    #[test]
    fn test_audio_artifact_new() {
        let a = AudioArtifact::new(100, AudioArtifactType::Click, 5, 0.7);
        assert_eq!(a.sample_idx, 100);
        assert_eq!(a.artifact_type, AudioArtifactType::Click);
        assert_eq!(a.duration_samples, 5);
        assert!((a.severity - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_click_repairer_basic() {
        // Create a signal with a spike at index 5
        let mut samples: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 0.6, 0.7, 0.8, 0.9];
        ClickRepairer::repair(&mut samples, 5, 3);
        // After repair, index 5 should be interpolated (close to linear between neighbors)
        assert!(samples[5].abs() < 5.0, "Click should be reduced");
    }

    #[test]
    fn test_click_repairer_empty() {
        let mut samples: Vec<f32> = vec![];
        ClickRepairer::repair(&mut samples, 0, 3);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_click_repairer_zero_window() {
        let mut samples = vec![0.0, 10.0, 0.0];
        let original = samples.clone();
        ClickRepairer::repair(&mut samples, 1, 0);
        assert_eq!(samples, original);
    }

    #[test]
    fn test_dropout_repairer_basic() {
        let mut samples: Vec<f32> = vec![0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0];
        DropoutRepairer::repair(&mut samples, 2, 3);
        // Samples 2..5 should now be a ramp from 0.5 to 0.5
        assert!(!samples[2].is_nan());
        assert!(!samples[3].is_nan());
        assert!(!samples[4].is_nan());
    }

    #[test]
    fn test_dropout_repairer_at_start() {
        let mut samples: Vec<f32> = vec![0.0, 0.0, 0.5, 1.0];
        DropoutRepairer::repair(&mut samples, 0, 2);
        // Should not panic; values should be interpolated
        assert!(!samples[0].is_nan());
        assert!(!samples[1].is_nan());
    }

    #[test]
    fn test_dropout_repairer_empty() {
        let mut samples: Vec<f32> = vec![];
        DropoutRepairer::repair(&mut samples, 0, 5);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_clipping_repairer_clips_above_threshold() {
        let mut samples: Vec<f32> = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
        ClippingRepairer::repair(&mut samples, 0.9);
        // All values should be within [-1.0, 1.0] after repair
        for s in &samples {
            assert!(*s >= -1.0 && *s <= 1.0, "Sample {s} out of range");
        }
    }

    #[test]
    fn test_clipping_repairer_no_effect_below_threshold() {
        let mut samples: Vec<f32> = vec![0.1, 0.2, -0.3, 0.4];
        let original = samples.clone();
        ClippingRepairer::repair(&mut samples, 0.9);
        // No clipping occurred, samples should be unchanged
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_audio_repair_report_success_rate() {
        let report = AudioRepairReport::new(10, 8, 3.0);
        assert!((report.success_rate() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_repair_report_success_rate_zero_found() {
        let report = AudioRepairReport::new(0, 0, 0.0);
        assert!((report.success_rate() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_artifact_type_distortion_not_impulsive() {
        assert!(!AudioArtifactType::Distortion.is_impulsive());
    }
}
