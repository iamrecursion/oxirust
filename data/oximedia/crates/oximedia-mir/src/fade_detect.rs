#![allow(dead_code)]
//! Fade-in and fade-out detection in audio tracks.
//!
//! Analyzes the energy envelope of an audio signal to detect gradual volume
//! increases (fade-in) and decreases (fade-out). Supports configurable
//! thresholds, minimum fade durations, and reports detailed fade characteristics
//! including shape (linear, logarithmic, exponential).

/// A detected fade event.
#[derive(Debug, Clone, PartialEq)]
pub struct FadeEvent {
    /// Type of fade (in or out).
    pub fade_type: FadeType,
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
    /// Shape of the fade curve.
    pub shape: FadeShape,
    /// Confidence of detection (0.0 - 1.0).
    pub confidence: f32,
    /// Starting energy level (0.0 - 1.0).
    pub start_energy: f32,
    /// Ending energy level (0.0 - 1.0).
    pub end_energy: f32,
}

impl FadeEvent {
    /// Duration of the fade in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.end_secs - self.start_secs
    }

    /// Energy change (positive for fade-in, negative for fade-out).
    #[must_use]
    pub fn energy_delta(&self) -> f32 {
        self.end_energy - self.start_energy
    }

    /// Rate of change (energy units per second).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rate(&self) -> f32 {
        let dur = self.duration_secs();
        if dur <= 0.0 {
            return 0.0;
        }
        self.energy_delta() / dur as f32
    }
}

/// Type of fade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeType {
    /// Volume increasing from low to high.
    FadeIn,
    /// Volume decreasing from high to low.
    FadeOut,
}

/// Shape of the fade curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeShape {
    /// Linear ramp.
    Linear,
    /// Logarithmic curve (fast start, slow end).
    Logarithmic,
    /// Exponential curve (slow start, fast end).
    Exponential,
    /// S-curve (slow-fast-slow).
    SCurve,
    /// Unknown / could not classify.
    Unknown,
}

/// Configuration for fade detection.
#[derive(Debug, Clone)]
pub struct FadeDetectConfig {
    /// Minimum fade duration in seconds.
    pub min_fade_secs: f64,
    /// Maximum fade duration in seconds.
    pub max_fade_secs: f64,
    /// Energy threshold below which audio is considered silence.
    pub silence_threshold: f32,
    /// Minimum energy delta to qualify as a fade.
    pub min_energy_delta: f32,
    /// Window size for energy calculation in seconds.
    pub energy_window_secs: f64,
    /// Hop size for energy frames in seconds.
    pub hop_secs: f64,
}

impl Default for FadeDetectConfig {
    fn default() -> Self {
        Self {
            min_fade_secs: 0.5,
            max_fade_secs: 30.0,
            silence_threshold: 0.01,
            min_energy_delta: 0.2,
            energy_window_secs: 0.1,
            hop_secs: 0.05,
        }
    }
}

/// Full result of fade detection.
#[derive(Debug, Clone, PartialEq)]
pub struct FadeDetectResult {
    /// All detected fade events.
    pub fades: Vec<FadeEvent>,
    /// Whether a fade-in was detected at the start of the track.
    pub has_intro_fade: bool,
    /// Whether a fade-out was detected at the end of the track.
    pub has_outro_fade: bool,
    /// Total audio duration in seconds.
    pub total_duration_secs: f64,
}

impl FadeDetectResult {
    /// Count of fade-in events.
    #[must_use]
    pub fn fade_in_count(&self) -> usize {
        self.fades
            .iter()
            .filter(|f| f.fade_type == FadeType::FadeIn)
            .count()
    }

    /// Count of fade-out events.
    #[must_use]
    pub fn fade_out_count(&self) -> usize {
        self.fades
            .iter()
            .filter(|f| f.fade_type == FadeType::FadeOut)
            .count()
    }

    /// Total duration of all fades combined.
    #[must_use]
    pub fn total_fade_duration(&self) -> f64 {
        self.fades.iter().map(FadeEvent::duration_secs).sum()
    }
}

/// Compute per-frame RMS energy envelope.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn compute_energy_envelope(
    samples: &[f32],
    sample_rate: f32,
    window_secs: f64,
    hop_secs: f64,
) -> Vec<f32> {
    if samples.is_empty() || sample_rate <= 0.0 || window_secs <= 0.0 || hop_secs <= 0.0 {
        return Vec::new();
    }

    let window_samples = (window_secs * f64::from(sample_rate)) as usize;
    let hop_samples = (hop_secs * f64::from(sample_rate)) as usize;

    if window_samples == 0 || hop_samples == 0 {
        return Vec::new();
    }

    let mut envelope = Vec::new();
    let mut pos = 0;

    while pos + window_samples <= samples.len() {
        let window = &samples[pos..pos + window_samples];
        let rms = (window.iter().map(|&s| s * s).sum::<f32>() / window_samples as f32).sqrt();
        envelope.push(rms);
        pos += hop_samples;
    }

    envelope
}

/// Normalize an energy envelope to 0.0 - 1.0 range.
#[must_use]
pub fn normalize_envelope(envelope: &[f32]) -> Vec<f32> {
    if envelope.is_empty() {
        return Vec::new();
    }
    let max_val = envelope.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val <= 0.0 {
        return vec![0.0; envelope.len()];
    }
    envelope.iter().map(|&e| e / max_val).collect()
}

/// Classify fade shape by examining the energy curve between two points.
#[must_use]
pub fn classify_fade_shape(energy_segment: &[f32]) -> FadeShape {
    if energy_segment.len() < 3 {
        return FadeShape::Unknown;
    }

    let n = energy_segment.len();
    let mid = n / 2;

    let start = energy_segment[0];
    let middle = energy_segment[mid];
    let end = energy_segment[n - 1];

    let total_delta = (end - start).abs();
    if total_delta < f32::EPSILON {
        return FadeShape::Unknown;
    }

    // Linear interpolation at midpoint
    let expected_linear_mid = start + (end - start) * 0.5;
    let mid_deviation = (middle - expected_linear_mid) / total_delta;

    if mid_deviation.abs() < 0.1 {
        FadeShape::Linear
    } else if mid_deviation > 0.15 {
        // Middle is above the linear line -> logarithmic shape (fast start)
        FadeShape::Logarithmic
    } else if mid_deviation < -0.15 {
        // Middle is below the linear line -> exponential shape (slow start)
        FadeShape::Exponential
    } else {
        FadeShape::Unknown
    }
}

/// Detect monotonic increasing or decreasing runs in the envelope.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_fades_from_envelope(
    envelope: &[f32],
    hop_secs: f64,
    config: &FadeDetectConfig,
) -> Vec<FadeEvent> {
    if envelope.len() < 2 {
        return Vec::new();
    }

    let min_frames = (config.min_fade_secs / hop_secs).ceil() as usize;
    let max_frames = (config.max_fade_secs / hop_secs).ceil() as usize;

    let mut fades = Vec::new();

    // Simple approach: find monotonic runs using smoothed differences
    let smoothed = smooth_envelope(envelope, 3);

    let mut run_start = 0_usize;
    let mut direction: Option<FadeType> = None;

    for i in 1..smoothed.len() {
        let delta = smoothed[i] - smoothed[i - 1];
        let current_dir = if delta > 0.001 {
            Some(FadeType::FadeIn)
        } else if delta < -0.001 {
            Some(FadeType::FadeOut)
        } else {
            direction // maintain current direction through flat spots
        };

        if current_dir != direction {
            // Direction changed: check if previous run qualifies
            if let Some(dir) = direction {
                let run_len = i - run_start;
                if run_len >= min_frames && run_len <= max_frames {
                    let energy_delta = (smoothed[i - 1] - smoothed[run_start]).abs();
                    if energy_delta >= config.min_energy_delta {
                        let segment = &smoothed[run_start..i];
                        let shape = classify_fade_shape(segment);
                        let start_secs = run_start as f64 * hop_secs;
                        let end_secs = (i - 1) as f64 * hop_secs;
                        let confidence = (energy_delta / 1.0).min(1.0);

                        fades.push(FadeEvent {
                            fade_type: dir,
                            start_secs,
                            end_secs,
                            shape,
                            confidence,
                            start_energy: smoothed[run_start],
                            end_energy: smoothed[i - 1],
                        });
                    }
                }
            }
            run_start = i;
            direction = current_dir;
        }
    }

    // Check final run
    if let Some(dir) = direction {
        let run_len = smoothed.len() - run_start;
        if run_len >= min_frames && run_len <= max_frames {
            let energy_delta = (smoothed[smoothed.len() - 1] - smoothed[run_start]).abs();
            if energy_delta >= config.min_energy_delta {
                let segment = &smoothed[run_start..];
                let shape = classify_fade_shape(segment);
                let start_secs = run_start as f64 * hop_secs;
                let end_secs = (smoothed.len() - 1) as f64 * hop_secs;
                let confidence = (energy_delta / 1.0).min(1.0);

                fades.push(FadeEvent {
                    fade_type: dir,
                    start_secs,
                    end_secs,
                    shape,
                    confidence,
                    start_energy: smoothed[run_start],
                    end_energy: smoothed[smoothed.len() - 1],
                });
            }
        }
    }

    fades
}

/// Simple moving average smoothing.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn smooth_envelope(envelope: &[f32], radius: usize) -> Vec<f32> {
    if envelope.is_empty() {
        return Vec::new();
    }
    let mut smoothed = Vec::with_capacity(envelope.len());
    for i in 0..envelope.len() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(envelope.len());
        let sum: f32 = envelope[start..end].iter().sum();
        let count = (end - start) as f32;
        smoothed.push(sum / count);
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_event_duration() {
        let fade = FadeEvent {
            fade_type: FadeType::FadeIn,
            start_secs: 0.0,
            end_secs: 3.0,
            shape: FadeShape::Linear,
            confidence: 0.9,
            start_energy: 0.0,
            end_energy: 1.0,
        };
        assert!((fade.duration_secs() - 3.0).abs() < f64::EPSILON);
        assert!((fade.energy_delta() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fade_event_rate() {
        let fade = FadeEvent {
            fade_type: FadeType::FadeOut,
            start_secs: 10.0,
            end_secs: 15.0,
            shape: FadeShape::Linear,
            confidence: 0.8,
            start_energy: 1.0,
            end_energy: 0.0,
        };
        assert!((fade.rate() + 0.2).abs() < 0.01);
    }

    #[test]
    fn test_fade_event_rate_zero_duration() {
        let fade = FadeEvent {
            fade_type: FadeType::FadeIn,
            start_secs: 5.0,
            end_secs: 5.0,
            shape: FadeShape::Unknown,
            confidence: 0.5,
            start_energy: 0.0,
            end_energy: 1.0,
        };
        assert!(fade.rate().abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_energy_envelope() {
        // 1 second at 100 Hz sample rate, all ones
        let samples = vec![1.0_f32; 100];
        let env = compute_energy_envelope(&samples, 100.0, 0.1, 0.1);
        assert!(!env.is_empty());
        for &e in &env {
            assert!((e - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_compute_energy_envelope_empty() {
        assert!(compute_energy_envelope(&[], 44100.0, 0.1, 0.05).is_empty());
    }

    #[test]
    fn test_compute_energy_envelope_zero_params() {
        let samples = vec![1.0; 100];
        assert!(compute_energy_envelope(&samples, 0.0, 0.1, 0.05).is_empty());
        assert!(compute_energy_envelope(&samples, 100.0, 0.0, 0.05).is_empty());
    }

    #[test]
    fn test_normalize_envelope() {
        let env = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let norm = normalize_envelope(&env);
        assert!((norm[2] - 1.0).abs() < f32::EPSILON);
        assert!((norm[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_envelope_silent() {
        let env = vec![0.0, 0.0, 0.0];
        let norm = normalize_envelope(&env);
        assert!(norm.iter().all(|&v| v.abs() < f32::EPSILON));
    }

    #[test]
    fn test_normalize_envelope_empty() {
        assert!(normalize_envelope(&[]).is_empty());
    }

    #[test]
    fn test_classify_fade_shape_linear() {
        // Perfect linear ramp from 0 to 1
        let segment: Vec<f32> = (0..20).map(|i| i as f32 / 19.0).collect();
        let shape = classify_fade_shape(&segment);
        assert_eq!(shape, FadeShape::Linear);
    }

    #[test]
    fn test_classify_fade_shape_short() {
        assert_eq!(classify_fade_shape(&[0.0, 1.0]), FadeShape::Unknown);
    }

    #[test]
    fn test_fade_detect_result_counts() {
        let result = FadeDetectResult {
            fades: vec![
                FadeEvent {
                    fade_type: FadeType::FadeIn,
                    start_secs: 0.0,
                    end_secs: 2.0,
                    shape: FadeShape::Linear,
                    confidence: 0.9,
                    start_energy: 0.0,
                    end_energy: 1.0,
                },
                FadeEvent {
                    fade_type: FadeType::FadeOut,
                    start_secs: 58.0,
                    end_secs: 60.0,
                    shape: FadeShape::Linear,
                    confidence: 0.8,
                    start_energy: 1.0,
                    end_energy: 0.0,
                },
            ],
            has_intro_fade: true,
            has_outro_fade: true,
            total_duration_secs: 60.0,
        };
        assert_eq!(result.fade_in_count(), 1);
        assert_eq!(result.fade_out_count(), 1);
        assert!((result.total_fade_duration() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_fades_from_envelope_fade_in() {
        // Envelope that ramps up linearly
        let hop_secs = 0.05;
        let env: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let config = FadeDetectConfig {
            min_fade_secs: 0.2,
            max_fade_secs: 10.0,
            min_energy_delta: 0.2,
            ..FadeDetectConfig::default()
        };
        let fades = detect_fades_from_envelope(&env, hop_secs, &config);
        assert!(!fades.is_empty(), "Should detect a fade-in");
        assert_eq!(fades[0].fade_type, FadeType::FadeIn);
    }

    #[test]
    fn test_detect_fades_from_envelope_empty() {
        let config = FadeDetectConfig::default();
        assert!(detect_fades_from_envelope(&[], 0.05, &config).is_empty());
    }

    #[test]
    fn test_smooth_envelope() {
        let env = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let smoothed = smooth_envelope(&env, 1);
        // Middle value should be averaged with neighbors
        assert!(smoothed[2] < 1.0);
        assert!(smoothed[2] > 0.0);
    }

    #[test]
    fn test_fade_detect_config_default() {
        let cfg = FadeDetectConfig::default();
        assert!((cfg.min_fade_secs - 0.5).abs() < f64::EPSILON);
        assert!((cfg.max_fade_secs - 30.0).abs() < f64::EPSILON);
    }
}
