//! Clock discipline: error measurement, correction actions, and frequency estimation.

#![allow(dead_code)]

/// Represents a measured clock error in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockError {
    /// Signed error in nanoseconds (positive = clock is ahead).
    pub nanoseconds: i64,
}

impl ClockError {
    /// Create a new `ClockError` from a nanosecond value.
    #[must_use]
    pub fn new(nanoseconds: i64) -> Self {
        Self { nanoseconds }
    }

    /// Returns `true` when the absolute error exceeds 1 millisecond.
    #[must_use]
    pub fn is_large(&self) -> bool {
        self.nanoseconds.unsigned_abs() > 1_000_000
    }

    /// Returns the absolute error in microseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn abs_us(&self) -> f64 {
        self.nanoseconds.unsigned_abs() as f64 / 1_000.0
    }
}

/// Action the discipliner recommends to correct the local clock.
#[derive(Debug, Clone, PartialEq)]
pub enum DisciplineAction {
    /// Apply a step correction of the given nanoseconds.
    Step(i64),
    /// Slew (gradually adjust) the clock by the given nanoseconds per second.
    Slew(i64),
    /// No correction needed.
    Hold,
}

impl DisciplineAction {
    /// Human-readable description of the action.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Step(ns) => format!("Step clock by {} ns", ns),
            Self::Slew(pps) => format!("Slew clock at {} ns/s", pps),
            Self::Hold => "Hold – no correction required".to_owned(),
        }
    }
}

/// Estimated frequency offset of the local oscillator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyEstimate {
    /// Parts-per-million offset.  Positive means the clock runs fast.
    pub ppm: f64,
    /// Confidence in the estimate (0.0 – 1.0).
    pub confidence: f64,
}

impl FrequencyEstimate {
    /// Create a new frequency estimate.
    #[must_use]
    pub fn new(ppm: f64, confidence: f64) -> Self {
        Self {
            ppm,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return the parts-per-million offset.
    #[must_use]
    pub fn ppm_offset(&self) -> f64 {
        self.ppm
    }

    /// `true` when the confidence is high enough to trust the estimate.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.7
    }
}

/// Disciplines the local clock by collecting offset samples and computing corrections.
#[derive(Debug)]
pub struct ClockDiscipliner {
    samples: Vec<i64>,
    max_samples: usize,
    step_threshold_ns: i64,
}

impl ClockDiscipliner {
    /// Create a new discipliner.
    ///
    /// * `max_samples` – window size for the internal ring buffer.
    /// * `step_threshold_ns` – errors above this magnitude trigger a step instead of a slew.
    #[must_use]
    pub fn new(max_samples: usize, step_threshold_ns: i64) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
            step_threshold_ns,
        }
    }

    /// Add an offset sample (nanoseconds).
    pub fn add_sample(&mut self, offset_ns: i64) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(offset_ns);
    }

    /// Compute the recommended correction action.
    #[must_use]
    pub fn compute_correction(&self) -> DisciplineAction {
        if self.samples.is_empty() {
            return DisciplineAction::Hold;
        }
        let mean = self.mean_offset();
        if mean.abs() < 100 {
            DisciplineAction::Hold
        } else if mean.abs() > self.step_threshold_ns {
            DisciplineAction::Step(-mean)
        } else {
            DisciplineAction::Slew(-mean / 8)
        }
    }

    /// Estimate the frequency offset of the oscillator from accumulated samples.
    ///
    /// Requires at least 4 samples; returns `None` otherwise.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frequency_estimate(&self) -> Option<FrequencyEstimate> {
        if self.samples.len() < 4 {
            return None;
        }
        // Simple linear drift: compare first and last halves.
        let half = self.samples.len() / 2;
        let first_mean: f64 =
            self.samples[..half].iter().map(|&x| x as f64).sum::<f64>() / half as f64;
        let second_mean: f64 =
            self.samples[half..].iter().map(|&x| x as f64).sum::<f64>() / half as f64;
        let drift_ns = second_mean - first_mean;
        // Approximate: 1 ns/s drift ≈ 1 ppb = 0.001 ppm
        let ppm = drift_ns * 0.001;
        let confidence = (self.samples.len() as f64 / self.max_samples as f64).min(1.0);
        Some(FrequencyEstimate::new(ppm, confidence))
    }

    /// Mean offset of the current sample window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_offset(&self) -> i64 {
        if self.samples.is_empty() {
            return 0;
        }
        let sum: i64 = self.samples.iter().sum();
        sum / self.samples.len() as i64
    }

    /// Number of samples currently held.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

impl Default for ClockDiscipliner {
    fn default() -> Self {
        Self::new(64, 125_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_error_not_large() {
        let e = ClockError::new(500_000); // 0.5 ms
        assert!(!e.is_large());
    }

    #[test]
    fn test_clock_error_is_large() {
        let e = ClockError::new(2_000_000); // 2 ms
        assert!(e.is_large());
    }

    #[test]
    fn test_clock_error_negative_large() {
        let e = ClockError::new(-1_500_000);
        assert!(e.is_large());
    }

    #[test]
    fn test_clock_error_abs_us() {
        let e = ClockError::new(3_000);
        assert!((e.abs_us() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_discipline_action_hold_description() {
        assert_eq!(
            DisciplineAction::Hold.description(),
            "Hold – no correction required"
        );
    }

    #[test]
    fn test_discipline_action_step_description() {
        let s = DisciplineAction::Step(5000).description();
        assert!(s.contains("5000"));
    }

    #[test]
    fn test_discipline_action_slew_description() {
        let s = DisciplineAction::Slew(-200).description();
        assert!(s.contains("-200"));
    }

    #[test]
    fn test_frequency_estimate_ppm_offset() {
        let f = FrequencyEstimate::new(2.5, 0.9);
        assert!((f.ppm_offset() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_estimate_reliable() {
        let f = FrequencyEstimate::new(0.1, 0.8);
        assert!(f.is_reliable());
    }

    #[test]
    fn test_frequency_estimate_not_reliable() {
        let f = FrequencyEstimate::new(0.1, 0.5);
        assert!(!f.is_reliable());
    }

    #[test]
    fn test_discipliner_empty_hold() {
        let d = ClockDiscipliner::default();
        assert_eq!(d.compute_correction(), DisciplineAction::Hold);
    }

    #[test]
    fn test_discipliner_add_sample_and_count() {
        let mut d = ClockDiscipliner::default();
        d.add_sample(1000);
        d.add_sample(2000);
        assert_eq!(d.sample_count(), 2);
    }

    #[test]
    fn test_discipliner_mean_offset() {
        let mut d = ClockDiscipliner::default();
        d.add_sample(1000);
        d.add_sample(3000);
        assert_eq!(d.mean_offset(), 2000);
    }

    #[test]
    fn test_discipliner_hold_small_error() {
        let mut d = ClockDiscipliner::default();
        d.add_sample(50);
        assert_eq!(d.compute_correction(), DisciplineAction::Hold);
    }

    #[test]
    fn test_discipliner_slew_medium_error() {
        let mut d = ClockDiscipliner::default();
        for _ in 0..4 {
            d.add_sample(500_000); // 0.5 ms – above 100 ns, below step threshold
        }
        let action = d.compute_correction();
        assert!(matches!(action, DisciplineAction::Slew(_)));
    }

    #[test]
    fn test_discipliner_step_large_error() {
        let mut d = ClockDiscipliner::default();
        for _ in 0..4 {
            d.add_sample(200_000_000); // 200 ms
        }
        let action = d.compute_correction();
        assert!(matches!(action, DisciplineAction::Step(_)));
    }

    #[test]
    fn test_discipliner_ring_buffer_bounded() {
        let mut d = ClockDiscipliner::new(4, 1_000_000);
        for i in 0..10 {
            d.add_sample(i * 1000);
        }
        assert_eq!(d.sample_count(), 4);
    }

    #[test]
    fn test_frequency_estimate_none_few_samples() {
        let mut d = ClockDiscipliner::default();
        d.add_sample(100);
        d.add_sample(200);
        assert!(d.frequency_estimate().is_none());
    }

    #[test]
    fn test_frequency_estimate_some_many_samples() {
        let mut d = ClockDiscipliner::default();
        for i in 0..8 {
            d.add_sample(i * 100);
        }
        assert!(d.frequency_estimate().is_some());
    }
}
