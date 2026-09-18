#![allow(dead_code)]
//! Short-term and integrated loudness metering (inspired by EBU R128 / ITU-R BS.1770).

/// A single loudness frame with its power level.
#[derive(Debug, Clone)]
pub struct LoudnessFrame {
    /// Mean square power of this frame.
    pub mean_square: f32,
    /// Duration of this frame in seconds.
    pub duration_secs: f32,
}

impl LoudnessFrame {
    /// Create a `LoudnessFrame` by computing the mean square of `samples`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_samples(samples: &[f32], sample_rate: f32) -> Self {
        let mean_square = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32
        };
        let duration_secs = if sample_rate > 0.0 {
            samples.len() as f32 / sample_rate
        } else {
            0.0
        };
        Self {
            mean_square,
            duration_secs,
        }
    }

    /// Returns `true` if this frame exceeds `threshold_lufs` (approximate).
    ///
    /// Converts mean square to LUFS-like value: `−0.691 + 10 log10(mean_square)`.
    #[must_use]
    pub fn is_loud(&self, threshold_lufs: f32) -> bool {
        if self.mean_square <= 0.0 {
            return false;
        }
        let lufs = -0.691 + 10.0 * self.mean_square.log10();
        lufs >= threshold_lufs
    }
}

// ---------------------------------------------------------------------------

/// Running loudness meter that accumulates samples and exposes momentary,
/// short-term, and integrated loudness.
///
/// This is a simplified implementation — not a full ITU-R BS.1770 gated
/// loudness meter, but suitable for non-broadcast monitoring use cases.
pub struct LoudnessMeter {
    sample_rate: f32,
    /// All sample power values accumulated (per block).
    integrated_power: Vec<f64>,
    /// Momentary window size in samples (400 ms).
    momentary_window: usize,
    /// Current sample buffer.
    buffer: Vec<f32>,
}

impl LoudnessMeter {
    /// Create a new `LoudnessMeter` for the given sample rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let momentary_window = (sample_rate * 0.4).round() as usize;
        Self {
            sample_rate,
            integrated_power: Vec::new(),
            momentary_window: momentary_window.max(1),
            buffer: Vec::new(),
        }
    }

    /// Push a single sample into the meter.
    pub fn push_sample(&mut self, sample: f32) {
        self.buffer.push(sample);
        // Flush once we have a full momentary window.
        if self.buffer.len() >= self.momentary_window {
            let power: f64 = self
                .buffer
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / self.buffer.len() as f64;
            self.integrated_power.push(power);
            self.buffer.clear();
        }
    }

    /// Momentary loudness (LUFS) over the most recent 400 ms window.
    ///
    /// Returns `None` if no data has been accumulated yet.
    #[must_use]
    pub fn momentary_loudness(&self) -> Option<f32> {
        let power = *self.integrated_power.last()?;
        if power <= 0.0 {
            return Some(-f32::INFINITY);
        }
        Some(-0.691 + 10.0 * (power as f32).log10())
    }

    /// Integrated loudness (LUFS) over the entire duration observed so far.
    ///
    /// Returns `None` if no data has been accumulated yet.
    #[must_use]
    pub fn integrated_loudness(&self) -> Option<f32> {
        if self.integrated_power.is_empty() {
            return None;
        }
        let mean_power: f64 =
            self.integrated_power.iter().sum::<f64>() / self.integrated_power.len() as f64;
        if mean_power <= 0.0 {
            return Some(-f32::INFINITY);
        }
        Some(-0.691 + 10.0 * (mean_power as f32).log10())
    }

    /// Reset the meter, clearing all accumulated data.
    pub fn reset(&mut self) {
        self.integrated_power.clear();
        self.buffer.clear();
    }

    /// Number of complete windows accumulated so far.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.integrated_power.len()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- LoudnessFrame ---

    #[test]
    fn test_frame_from_silence() {
        let frame = LoudnessFrame::from_samples(&[0.0, 0.0, 0.0], 44100.0);
        assert_eq!(frame.mean_square, 0.0);
        assert!(!frame.is_loud(-23.0));
    }

    #[test]
    fn test_frame_mean_square_unit_sine() {
        // Sine of amplitude 1.0 → mean square ≈ 0.5
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin())
            .collect();
        let frame = LoudnessFrame::from_samples(&samples, 44100.0);
        assert!(
            (frame.mean_square - 0.5).abs() < 0.01,
            "ms={}",
            frame.mean_square
        );
    }

    #[test]
    fn test_frame_is_loud_true() {
        // mean_square = 1.0 → LUFS ≈ -0.691
        let frame = LoudnessFrame {
            mean_square: 1.0,
            duration_secs: 0.4,
        };
        assert!(frame.is_loud(-1.0));
    }

    #[test]
    fn test_frame_is_loud_false() {
        let frame = LoudnessFrame {
            mean_square: 1e-6,
            duration_secs: 0.4,
        };
        assert!(!frame.is_loud(-23.0));
    }

    #[test]
    fn test_frame_duration() {
        let samples = vec![0.5_f32; 4410];
        let frame = LoudnessFrame::from_samples(&samples, 44100.0);
        assert!(
            (frame.duration_secs - 0.1).abs() < 1e-5,
            "dur={}",
            frame.duration_secs
        );
    }

    #[test]
    fn test_frame_empty_samples() {
        let frame = LoudnessFrame::from_samples(&[], 44100.0);
        assert_eq!(frame.mean_square, 0.0);
        assert_eq!(frame.duration_secs, 0.0);
    }

    // --- LoudnessMeter ---

    #[test]
    fn test_meter_no_data() {
        let meter = LoudnessMeter::new(44100.0);
        assert!(meter.momentary_loudness().is_none());
        assert!(meter.integrated_loudness().is_none());
    }

    #[test]
    fn test_meter_push_and_momentary() {
        let mut meter = LoudnessMeter::new(44100.0);
        // Fill one complete momentary window with 0.5 amplitude.
        let n = meter.momentary_window;
        for _ in 0..n {
            meter.push_sample(0.5);
        }
        assert!(meter.momentary_loudness().is_some());
        assert!(meter.window_count() >= 1);
    }

    #[test]
    fn test_meter_integrated_after_several_windows() {
        let mut meter = LoudnessMeter::new(44100.0);
        let n = meter.momentary_window * 3;
        for _ in 0..n {
            meter.push_sample(0.7);
        }
        let il = meter
            .integrated_loudness()
            .expect("should have integrated loudness");
        assert!(il.is_finite(), "integrated={il}");
    }

    #[test]
    fn test_meter_reset_clears_all() {
        let mut meter = LoudnessMeter::new(44100.0);
        let n = meter.momentary_window;
        for _ in 0..n {
            meter.push_sample(0.5);
        }
        meter.reset();
        assert!(meter.momentary_loudness().is_none());
        assert!(meter.integrated_loudness().is_none());
        assert_eq!(meter.window_count(), 0);
    }

    #[test]
    fn test_meter_silent_signal() {
        let mut meter = LoudnessMeter::new(44100.0);
        let n = meter.momentary_window;
        for _ in 0..n {
            meter.push_sample(0.0);
        }
        // Silence should give -infinity
        let ml = meter.momentary_loudness().expect("should return Some");
        assert!(ml == f32::NEG_INFINITY || ml < -100.0, "ml={ml}");
    }

    #[test]
    fn test_meter_window_count() {
        let mut meter = LoudnessMeter::new(44100.0);
        let n = meter.momentary_window * 5;
        for _ in 0..n {
            meter.push_sample(0.3);
        }
        assert_eq!(meter.window_count(), 5);
    }
}
