#![allow(dead_code)]
//! Per-clip gain management with fades, automation, and normalization.
//!
//! Manages gain for individual audio clips on a timeline, including static gain,
//! fade-in/fade-out curves, gain automation envelopes, and peak/loudness
//! normalization.

/// Fade curve type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FadeCurve {
    /// Linear ramp.
    Linear,
    /// Equal-power (sine-based) curve.
    EqualPower,
    /// Logarithmic (fast start, slow end).
    Logarithmic,
    /// Exponential (slow start, fast end).
    Exponential,
    /// S-curve (smooth sigmoid).
    SCurve,
}

impl FadeCurve {
    /// Evaluate the fade curve at position `t` (0.0 = start, 1.0 = end).
    /// Returns a value in 0.0..1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EqualPower => (t * std::f64::consts::FRAC_PI_2).sin(),
            Self::Logarithmic => {
                if t <= 0.0 {
                    0.0
                } else {
                    (1.0 + (t * 9.0 + 1.0).log10()) / 2.0
                }
            }
            Self::Exponential => t * t,
            Self::SCurve => {
                // Hermite S-curve: 3t^2 - 2t^3
                3.0 * t * t - 2.0 * t * t * t
            }
        }
    }
}

/// A fade specification (either fade-in or fade-out).
#[derive(Debug, Clone, PartialEq)]
pub struct Fade {
    /// Duration of the fade in seconds.
    pub duration: f64,
    /// Curve type.
    pub curve: FadeCurve,
}

impl Fade {
    /// Create a new fade with the given duration and curve.
    pub fn new(duration: f64, curve: FadeCurve) -> Self {
        Self {
            duration: duration.max(0.0),
            curve,
        }
    }

    /// Evaluate the fade-in gain at time `t` seconds from the clip start.
    pub fn fade_in_gain(&self, t: f64) -> f64 {
        if self.duration <= 0.0 || t >= self.duration {
            return 1.0;
        }
        if t <= 0.0 {
            return 0.0;
        }
        self.curve.evaluate(t / self.duration)
    }

    /// Evaluate the fade-out gain at time `t` seconds before the clip end.
    pub fn fade_out_gain(&self, time_before_end: f64) -> f64 {
        if self.duration <= 0.0 || time_before_end >= self.duration {
            return 1.0;
        }
        if time_before_end <= 0.0 {
            return 0.0;
        }
        self.curve.evaluate(time_before_end / self.duration)
    }
}

/// A single gain automation breakpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct GainBreakpoint {
    /// Time offset from clip start in seconds.
    pub time: f64,
    /// Gain value in dB.
    pub gain_db: f64,
}

impl GainBreakpoint {
    /// Create a new gain breakpoint.
    pub fn new(time: f64, gain_db: f64) -> Self {
        Self { time, gain_db }
    }
}

/// Gain automation envelope made of breakpoints.
#[derive(Debug, Clone)]
pub struct GainEnvelope {
    /// Breakpoints sorted by time.
    breakpoints: Vec<GainBreakpoint>,
}

impl GainEnvelope {
    /// Create an empty gain envelope.
    pub fn new() -> Self {
        Self {
            breakpoints: Vec::new(),
        }
    }

    /// Add a breakpoint (automatically sorted by time).
    pub fn add_breakpoint(&mut self, bp: GainBreakpoint) {
        self.breakpoints.push(bp);
        self.breakpoints.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return the number of breakpoints.
    pub fn len(&self) -> usize {
        self.breakpoints.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.breakpoints.is_empty()
    }

    /// Evaluate the envelope at a given time (in seconds from clip start).
    /// Returns gain in dB, linearly interpolating between breakpoints.
    pub fn evaluate_db(&self, time: f64) -> f64 {
        if self.breakpoints.is_empty() {
            return 0.0;
        }
        if self.breakpoints.len() == 1 {
            return self.breakpoints[0].gain_db;
        }

        // Before first breakpoint.
        if time <= self.breakpoints[0].time {
            return self.breakpoints[0].gain_db;
        }
        // After last breakpoint.
        if time >= self.breakpoints[self.breakpoints.len() - 1].time {
            return self.breakpoints[self.breakpoints.len() - 1].gain_db;
        }

        // Find surrounding breakpoints.
        for window in self.breakpoints.windows(2) {
            let (a, b) = (&window[0], &window[1]);
            if time >= a.time && time <= b.time {
                let dt = b.time - a.time;
                if dt <= 0.0 {
                    return a.gain_db;
                }
                let frac = (time - a.time) / dt;
                return a.gain_db + frac * (b.gain_db - a.gain_db);
            }
        }

        0.0
    }

    /// Evaluate the envelope at a given time, returning linear gain.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_linear(&self, time: f64) -> f64 {
        db_to_linear(self.evaluate_db(time))
    }
}

impl Default for GainEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalization mode for a clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMode {
    /// Normalize to a peak level in dBFS.
    Peak(f64),
    /// Normalize to an RMS level in dBFS.
    Rms(f64),
    /// No normalization.
    None,
}

/// Per-clip gain settings combining all gain controls.
#[derive(Debug, Clone)]
pub struct ClipGain {
    /// Static gain offset in dB.
    pub static_gain_db: f64,
    /// Optional fade-in.
    pub fade_in: Option<Fade>,
    /// Optional fade-out.
    pub fade_out: Option<Fade>,
    /// Gain automation envelope.
    pub envelope: GainEnvelope,
    /// Normalization mode.
    pub normalization: NormalizationMode,
    /// Clip duration in seconds (needed for fade-out computation).
    pub clip_duration: f64,
}

impl ClipGain {
    /// Create a new clip gain with default (unity) settings.
    pub fn new(clip_duration: f64) -> Self {
        Self {
            static_gain_db: 0.0,
            fade_in: None,
            fade_out: None,
            envelope: GainEnvelope::new(),
            normalization: NormalizationMode::None,
            clip_duration: clip_duration.max(0.0),
        }
    }

    /// Set the static gain.
    pub fn with_static_gain(mut self, db: f64) -> Self {
        self.static_gain_db = db;
        self
    }

    /// Set the fade-in.
    pub fn with_fade_in(mut self, fade: Fade) -> Self {
        self.fade_in = Some(fade);
        self
    }

    /// Set the fade-out.
    pub fn with_fade_out(mut self, fade: Fade) -> Self {
        self.fade_out = Some(fade);
        self
    }

    /// Set the normalization mode.
    pub fn with_normalization(mut self, mode: NormalizationMode) -> Self {
        self.normalization = mode;
        self
    }

    /// Compute the combined linear gain at a given time offset from clip start.
    #[allow(clippy::cast_precision_loss)]
    pub fn gain_at(&self, time: f64) -> f64 {
        let mut gain = db_to_linear(self.static_gain_db);

        // Fade-in.
        if let Some(ref fi) = self.fade_in {
            gain *= fi.fade_in_gain(time);
        }

        // Fade-out.
        if let Some(ref fo) = self.fade_out {
            let time_before_end = self.clip_duration - time;
            gain *= fo.fade_out_gain(time_before_end);
        }

        // Automation envelope.
        gain *= self.envelope.evaluate_linear(time);

        gain
    }

    /// Compute the normalization gain offset for a signal with the given peak and RMS.
    #[allow(clippy::cast_precision_loss)]
    pub fn normalization_gain_db(&self, peak_db: f64, rms_db: f64) -> f64 {
        match self.normalization {
            NormalizationMode::Peak(target) => target - peak_db,
            NormalizationMode::Rms(target) => target - rms_db,
            NormalizationMode::None => 0.0,
        }
    }
}

/// Convert dB to linear gain.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

/// Compute peak level in dBFS for a signal.
#[allow(clippy::cast_precision_loss)]
pub fn peak_dbfs(samples: &[f64]) -> f64 {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    linear_to_db(peak)
}

/// Compute RMS level in dBFS for a signal.
#[allow(clippy::cast_precision_loss)]
pub fn rms_dbfs(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt();
    linear_to_db(rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fade_curve() {
        let curve = FadeCurve::Linear;
        assert!((curve.evaluate(0.0)).abs() < 1e-10);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equal_power_fade_curve() {
        let curve = FadeCurve::EqualPower;
        assert!((curve.evaluate(0.0)).abs() < 1e-10);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 1e-10);
        // Mid-point should be > 0.5 for equal power.
        assert!(curve.evaluate(0.5) > 0.5);
    }

    #[test]
    fn test_s_curve() {
        let curve = FadeCurve::SCurve;
        assert!((curve.evaluate(0.0)).abs() < 1e-10);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((curve.evaluate(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_fade_curve() {
        let curve = FadeCurve::Exponential;
        assert!((curve.evaluate(0.0)).abs() < 1e-10);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 1e-10);
        // Mid-point should be < 0.5 for exponential.
        assert!(curve.evaluate(0.5) < 0.5);
    }

    #[test]
    fn test_fade_in_gain() {
        let fade = Fade::new(0.5, FadeCurve::Linear);
        assert!((fade.fade_in_gain(0.0)).abs() < 1e-10);
        assert!((fade.fade_in_gain(0.25) - 0.5).abs() < 1e-10);
        assert!((fade.fade_in_gain(0.5) - 1.0).abs() < 1e-10);
        assert!((fade.fade_in_gain(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fade_out_gain() {
        let fade = Fade::new(0.5, FadeCurve::Linear);
        assert!((fade.fade_out_gain(0.0)).abs() < 1e-10);
        assert!((fade.fade_out_gain(0.25) - 0.5).abs() < 1e-10);
        assert!((fade.fade_out_gain(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gain_envelope_single_breakpoint() {
        let mut env = GainEnvelope::new();
        env.add_breakpoint(GainBreakpoint::new(1.0, -6.0));
        assert_eq!(env.evaluate_db(0.0), -6.0);
        assert_eq!(env.evaluate_db(1.0), -6.0);
        assert_eq!(env.evaluate_db(5.0), -6.0);
    }

    #[test]
    fn test_gain_envelope_interpolation() {
        let mut env = GainEnvelope::new();
        env.add_breakpoint(GainBreakpoint::new(0.0, 0.0));
        env.add_breakpoint(GainBreakpoint::new(1.0, -12.0));
        let mid = env.evaluate_db(0.5);
        assert!((mid - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gain_envelope_empty() {
        let env = GainEnvelope::new();
        assert!(env.is_empty());
        assert_eq!(env.evaluate_db(0.5), 0.0);
    }

    #[test]
    fn test_clip_gain_static() {
        let cg = ClipGain::new(10.0).with_static_gain(-6.0);
        let gain = cg.gain_at(5.0);
        // -6dB ~ 0.5012
        assert!((gain - 0.5012).abs() < 0.01);
    }

    #[test]
    fn test_clip_gain_with_fades() {
        let cg = ClipGain::new(2.0)
            .with_fade_in(Fade::new(0.5, FadeCurve::Linear))
            .with_fade_out(Fade::new(0.5, FadeCurve::Linear));
        // At start: gain should be near 0.
        assert!(cg.gain_at(0.0) < 0.01);
        // At middle: gain should be near 1.
        assert!((cg.gain_at(1.0) - 1.0).abs() < 0.01);
        // At end: gain should be near 0.
        assert!(cg.gain_at(2.0) < 0.01);
    }

    #[test]
    fn test_normalization_gain_peak() {
        let cg = ClipGain::new(10.0).with_normalization(NormalizationMode::Peak(-1.0));
        let norm_gain = cg.normalization_gain_db(-6.0, -12.0);
        assert!((norm_gain - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_gain_rms() {
        let cg = ClipGain::new(10.0).with_normalization(NormalizationMode::Rms(-20.0));
        let norm_gain = cg.normalization_gain_db(-6.0, -24.0);
        assert!((norm_gain - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_dbfs() {
        let samples = vec![0.5, -0.3, 0.1];
        let peak = peak_dbfs(&samples);
        // peak of 0.5 -> ~-6.02 dBFS
        assert!((peak - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_rms_dbfs() {
        // Constant signal of 0.5 -> RMS = 0.5 -> ~-6.02 dBFS
        let samples = vec![0.5; 100];
        let rms = rms_dbfs(&samples);
        assert!((rms - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_rms_dbfs_empty() {
        assert_eq!(rms_dbfs(&[]), f64::NEG_INFINITY);
    }
}
