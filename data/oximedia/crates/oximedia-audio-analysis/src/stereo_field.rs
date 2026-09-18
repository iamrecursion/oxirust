//! Stereo field analysis.
//!
//! Analyses the spatial characteristics of a stereo audio signal: pan
//! position, width, mid/side balance, and phase correlation. These
//! metrics are essential for mastering, broadcast loudness compliance,
//! and immersive-audio down-mix verification.

#![allow(dead_code)]

/// Discrete stereo position bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StereoPosition {
    /// Hard left (pan < -0.75).
    HardLeft,
    /// Left (pan -0.75 .. -0.25).
    Left,
    /// Centre (pan -0.25 .. 0.25).
    Centre,
    /// Right (pan 0.25 .. 0.75).
    Right,
    /// Hard right (pan > 0.75).
    HardRight,
}

impl StereoPosition {
    /// Classify a continuous pan value in \[-1, 1\].
    #[must_use]
    pub fn from_pan(pan: f32) -> Self {
        if pan < -0.75 {
            Self::HardLeft
        } else if pan < -0.25 {
            Self::Left
        } else if pan <= 0.25 {
            Self::Centre
        } else if pan <= 0.75 {
            Self::Right
        } else {
            Self::HardRight
        }
    }

    /// Short label for display.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::HardLeft => "Hard L",
            Self::Left => "Left",
            Self::Centre => "Centre",
            Self::Right => "Right",
            Self::HardRight => "Hard R",
        }
    }
}

/// Frame-level stereo field measurement.
#[derive(Debug, Clone, Copy)]
pub struct StereoFieldFrame {
    /// Centre time of this frame in seconds.
    pub time_s: f32,
    /// Pan position \[-1, 1\] (negative = left).
    pub pan: f32,
    /// Stereo width \[0, 1\] (0 = mono, 1 = full width).
    pub width: f32,
    /// Phase correlation \[-1, 1\] (1 = perfectly in-phase, -1 = out-of-phase).
    pub correlation: f32,
    /// Mid-level RMS.
    pub mid_rms: f32,
    /// Side-level RMS.
    pub side_rms: f32,
}

/// Result of a complete stereo field analysis pass.
#[derive(Debug, Clone)]
pub struct StereoField {
    /// Per-frame measurements.
    pub frames: Vec<StereoFieldFrame>,
    /// Mean pan across the signal.
    pub mean_pan: f32,
    /// Mean stereo width.
    pub mean_width: f32,
    /// Mean phase correlation.
    pub mean_correlation: f32,
    /// Classified dominant position.
    pub dominant_position: StereoPosition,
}

impl StereoField {
    /// Returns `true` when the signal is essentially mono (width < 0.05).
    #[must_use]
    pub fn is_mono(&self) -> bool {
        self.mean_width < 0.05
    }

    /// Returns `true` when average phase correlation drops below zero,
    /// indicating potential phase cancellation issues.
    #[must_use]
    pub fn has_phase_issues(&self) -> bool {
        self.mean_correlation < 0.0
    }

    /// Fraction of frames where correlation is negative.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn negative_correlation_fraction(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let neg = self.frames.iter().filter(|f| f.correlation < 0.0).count();
        neg as f32 / self.frames.len() as f32
    }
}

/// Analyses the stereo field of an interleaved stereo signal.
pub struct StereoFieldAnalyzer {
    sample_rate: f32,
    /// Frame length in sample-pairs (i.e. stereo frames, not individual samples).
    frame_pairs: usize,
    /// Hop in sample-pairs.
    hop_pairs: usize,
}

impl StereoFieldAnalyzer {
    /// Create a new [`StereoFieldAnalyzer`].
    ///
    /// # Arguments
    /// * `sample_rate`  -- Sample rate in Hz.
    /// * `frame_ms`     -- Analysis window in milliseconds.
    /// * `hop_ms`       -- Hop in milliseconds.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn new(sample_rate: f32, frame_ms: f32, hop_ms: f32) -> Self {
        let frame_pairs = (sample_rate * frame_ms / 1000.0) as usize;
        let hop_pairs = (sample_rate * hop_ms / 1000.0).max(1.0) as usize;
        Self {
            sample_rate,
            frame_pairs,
            hop_pairs,
        }
    }

    /// Analyse interleaved stereo samples `[L0, R0, L1, R1, ...]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyse(&self, interleaved: &[f32]) -> StereoField {
        let total_pairs = interleaved.len() / 2;
        let mut frames = Vec::new();
        let mut pos = 0usize;

        while pos + self.frame_pairs <= total_pairs {
            let left: Vec<f32> = (pos..pos + self.frame_pairs)
                .map(|i| interleaved[i * 2])
                .collect();
            let right: Vec<f32> = (pos..pos + self.frame_pairs)
                .map(|i| interleaved[i * 2 + 1])
                .collect();

            let time_s = pos as f32 / self.sample_rate;
            let frame = self.analyse_frame(&left, &right, time_s);
            frames.push(frame);
            pos += self.hop_pairs;
        }

        let (mean_pan, mean_width, mean_corr) = if frames.is_empty() {
            (0.0, 0.0, 1.0)
        } else {
            let n = frames.len() as f32;
            let mp = frames.iter().map(|f| f.pan).sum::<f32>() / n;
            let mw = frames.iter().map(|f| f.width).sum::<f32>() / n;
            let mc = frames.iter().map(|f| f.correlation).sum::<f32>() / n;
            (mp, mw, mc)
        };

        StereoField {
            frames,
            mean_pan,
            mean_width,
            mean_correlation: mean_corr,
            dominant_position: StereoPosition::from_pan(mean_pan),
        }
    }

    /// Analyse a single frame from separate left / right buffers.
    #[allow(clippy::cast_precision_loss, clippy::unused_self)]
    fn analyse_frame(&self, left: &[f32], right: &[f32], time_s: f32) -> StereoFieldFrame {
        let n = left.len().min(right.len());
        if n == 0 {
            return StereoFieldFrame {
                time_s,
                pan: 0.0,
                width: 0.0,
                correlation: 1.0,
                mid_rms: 0.0,
                side_rms: 0.0,
            };
        }

        let mut sum_l2 = 0.0_f32;
        let mut sum_r2 = 0.0_f32;
        let mut sum_lr = 0.0_f32;
        let mut sum_mid2 = 0.0_f32;
        let mut sum_side2 = 0.0_f32;

        for i in 0..n {
            let l = left[i];
            let r = right[i];
            sum_l2 += l * l;
            sum_r2 += r * r;
            sum_lr += l * r;
            let mid = (l + r) * 0.5;
            let side = (l - r) * 0.5;
            sum_mid2 += mid * mid;
            sum_side2 += side * side;
        }

        let rms_l = (sum_l2 / n as f32).sqrt();
        let rms_r = (sum_r2 / n as f32).sqrt();
        let mid_rms = (sum_mid2 / n as f32).sqrt();
        let side_rms = (sum_side2 / n as f32).sqrt();

        // Pan: relative balance between L and R.
        let denom = rms_l + rms_r;
        let pan = if denom > 1e-12 {
            (rms_r - rms_l) / denom
        } else {
            0.0
        };

        // Width: ratio of side to mid energy.
        let width = if mid_rms > 1e-12 {
            (side_rms / mid_rms).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Normalised cross-correlation.
        let norm = (sum_l2 * sum_r2).sqrt();
        let correlation = if norm > 1e-12 {
            (sum_lr / norm).clamp(-1.0, 1.0)
        } else {
            1.0
        };

        StereoFieldFrame {
            time_s,
            pan,
            width,
            correlation,
            mid_rms,
            side_rms,
        }
    }
}

impl Default for StereoFieldAnalyzer {
    fn default() -> Self {
        Self::new(44100.0, 50.0, 25.0)
    }
}

// -- unit tests --

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_from_pan_hard_left() {
        assert_eq!(StereoPosition::from_pan(-0.9), StereoPosition::HardLeft);
    }

    #[test]
    fn test_position_from_pan_centre() {
        assert_eq!(StereoPosition::from_pan(0.0), StereoPosition::Centre);
    }

    #[test]
    fn test_position_from_pan_hard_right() {
        assert_eq!(StereoPosition::from_pan(0.9), StereoPosition::HardRight);
    }

    #[test]
    fn test_position_labels() {
        assert_eq!(StereoPosition::HardLeft.label(), "Hard L");
        assert_eq!(StereoPosition::Centre.label(), "Centre");
        assert_eq!(StereoPosition::HardRight.label(), "Hard R");
    }

    #[test]
    fn test_position_left_right() {
        assert_eq!(StereoPosition::from_pan(-0.5), StereoPosition::Left);
        assert_eq!(StereoPosition::from_pan(0.5), StereoPosition::Right);
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = StereoFieldAnalyzer::default();
        assert_eq!(analyzer.sample_rate, 44100.0);
        assert!(analyzer.frame_pairs > 0);
    }

    #[test]
    fn test_mono_signal() {
        let analyzer = StereoFieldAnalyzer::default();
        // Identical L and R -> mono
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            interleaved.push(s);
            interleaved.push(s);
        }
        let result = analyzer.analyse(&interleaved);
        assert!(result.is_mono());
        assert!(result.mean_correlation > 0.9);
        assert!(!result.has_phase_issues());
    }

    #[test]
    fn test_hard_left_signal() {
        let analyzer = StereoFieldAnalyzer::default();
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            interleaved.push(s);
            interleaved.push(0.0);
        }
        let result = analyzer.analyse(&interleaved);
        assert!(result.mean_pan < -0.5);
    }

    #[test]
    fn test_out_of_phase_detection() {
        let analyzer = StereoFieldAnalyzer::default();
        // L = +signal, R = -signal -> perfectly out-of-phase
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            interleaved.push(s);
            interleaved.push(-s);
        }
        let result = analyzer.analyse(&interleaved);
        assert!(result.has_phase_issues());
        assert!(result.mean_correlation < 0.0);
    }

    #[test]
    fn test_negative_correlation_fraction() {
        let analyzer = StereoFieldAnalyzer::default();
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            interleaved.push(s);
            interleaved.push(-s);
        }
        let result = analyzer.analyse(&interleaved);
        assert!(result.negative_correlation_fraction() > 0.5);
    }

    #[test]
    fn test_empty_signal() {
        let analyzer = StereoFieldAnalyzer::default();
        let result = analyzer.analyse(&[]);
        assert!(result.frames.is_empty());
        assert_eq!(result.mean_pan, 0.0);
        assert!(!result.has_phase_issues());
    }

    #[test]
    fn test_stereo_field_dominant_position() {
        let analyzer = StereoFieldAnalyzer::default();
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            interleaved.push(s);
            interleaved.push(s);
        }
        let result = analyzer.analyse(&interleaved);
        assert_eq!(result.dominant_position, StereoPosition::Centre);
    }

    #[test]
    fn test_frame_timestamps_monotonic() {
        let analyzer = StereoFieldAnalyzer::default();
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let s = (i as f32 * 0.01).sin();
            interleaved.push(s);
            interleaved.push(s * 0.5);
        }
        let result = analyzer.analyse(&interleaved);
        for w in result.frames.windows(2) {
            assert!(w[1].time_s > w[0].time_s);
        }
    }

    #[test]
    fn test_width_bounded() {
        let analyzer = StereoFieldAnalyzer::default();
        let mut interleaved = Vec::with_capacity(44100 * 2);
        for i in 0..44100 {
            let l = (i as f32 * 0.05).sin() * 0.5;
            let r = (i as f32 * 0.07).cos() * 0.5;
            interleaved.push(l);
            interleaved.push(r);
        }
        let result = analyzer.analyse(&interleaved);
        assert!(result.mean_width >= 0.0 && result.mean_width <= 1.0);
    }
}
