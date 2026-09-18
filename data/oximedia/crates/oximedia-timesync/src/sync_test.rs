//! Synthetic clock drift/offset injection for testing sync algorithms.
//!
//! Provides a configurable synthetic clock that can inject known offsets,
//! frequency drifts, phase steps, and noise patterns so that sync algorithm
//! implementations can be unit-tested against ground-truth scenarios.
//!
//! # Design
//! - [`SyntheticClock`]: a deterministic virtual clock with configurable
//!   drift, offset, and noise injection.
//! - [`InjectionScenario`]: a sequence of [`ClockFault`] events applied at
//!   specific virtual times.
//! - [`ClockFault`]: an individual injection (step, ramp, noise, holdover).
//! - [`SyncTestHarness`]: orchestrates a scenario over a synthetic clock and
//!   records per-sample ground-truth for comparison.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Clock fault types
// ---------------------------------------------------------------------------

/// A single fault or perturbation injected into a synthetic clock.
#[derive(Debug, Clone)]
pub enum ClockFault {
    /// Instantaneous phase step: adds `delta_ns` nanoseconds to the offset.
    PhaseStep {
        /// Nanoseconds to add (positive = ahead, negative = behind).
        delta_ns: i64,
    },
    /// Frequency ramp: applies an additional `rate_ppb` ppb drift starting
    /// at the injection point.
    FrequencyRamp {
        /// Additional frequency offset in ppb (parts per billion).
        rate_ppb: f64,
    },
    /// White-noise injection: adds zero-mean Gaussian noise scaled by
    /// `std_dev_ns` using a deterministic PRNG seeded from the sample index.
    WhiteNoise {
        /// One-sigma noise amplitude in nanoseconds.
        std_dev_ns: f64,
    },
    /// Holdover: disables updates from the reference, letting the clock
    /// free-run with its current drift for `duration`.
    Holdover {
        /// How long the holdover lasts.
        duration: Duration,
    },
    /// Removes all active faults, resetting the clock to nominal operation.
    Reset,
}

// ---------------------------------------------------------------------------
// Injection scenario
// ---------------------------------------------------------------------------

/// A scheduled fault injection: a fault that activates at a particular
/// virtual sample index.
#[derive(Debug, Clone)]
pub struct ScheduledFault {
    /// Sample index (0-based) at which to apply the fault.
    pub at_sample: usize,
    /// The fault to inject.
    pub fault: ClockFault,
}

impl ScheduledFault {
    /// Convenience constructor.
    #[must_use]
    pub fn new(at_sample: usize, fault: ClockFault) -> Self {
        Self { at_sample, fault }
    }
}

/// An ordered sequence of scheduled faults forming a complete test scenario.
#[derive(Debug, Clone, Default)]
pub struct InjectionScenario {
    faults: Vec<ScheduledFault>,
}

impl InjectionScenario {
    /// Creates an empty scenario.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a scheduled fault, keeping the list sorted by `at_sample`.
    pub fn add(&mut self, fault: ScheduledFault) {
        self.faults.push(fault);
        self.faults.sort_by_key(|f| f.at_sample);
    }

    /// Returns all scheduled faults in order.
    #[must_use]
    pub fn faults(&self) -> &[ScheduledFault] {
        &self.faults
    }
}

// ---------------------------------------------------------------------------
// Synthetic clock
// ---------------------------------------------------------------------------

/// Internal state of the synthetic clock.
#[derive(Debug, Clone)]
struct SyntheticClockState {
    /// Accumulated phase offset in nanoseconds.
    phase_offset_ns: i64,
    /// Base frequency drift (ppb).
    base_drift_ppb: f64,
    /// Additional drift from active frequency ramps.
    ramp_drift_ppb: f64,
    /// Remaining holdover duration (in samples).
    holdover_samples_remaining: usize,
    /// Active white-noise std deviation (0 = no noise).
    noise_std_dev_ns: f64,
    /// Simple LCG PRNG state for deterministic noise.
    prng_state: u64,
}

impl SyntheticClockState {
    fn new(base_drift_ppb: f64) -> Self {
        Self {
            phase_offset_ns: 0,
            base_drift_ppb,
            ramp_drift_ppb: 0.0,
            holdover_samples_remaining: 0,
            noise_std_dev_ns: 0.0,
            prng_state: 0xDEAD_BEEF_1234_5678,
        }
    }

    /// Advances the state by one sample of `spacing` duration and returns the
    /// current observed offset in nanoseconds.
    fn advance(&mut self, spacing: Duration) -> i64 {
        let t_secs = spacing.as_secs_f64();
        let total_drift_ppb = self.base_drift_ppb + self.ramp_drift_ppb;

        // Accumulate drift (ppb × s = ns).
        self.phase_offset_ns += (total_drift_ppb * t_secs) as i64;

        // Apply holdover (no change to drift accumulation, just mark it).
        if self.holdover_samples_remaining > 0 {
            self.holdover_samples_remaining -= 1;
        }

        // Apply noise via a Box-Muller-free approximation (sum of 12 uniform
        // samples shifted by 6 gives ~N(0,1) by the CLT).
        let noise_ns = if self.noise_std_dev_ns > 0.0 {
            let mut u_sum: f64 = 0.0;
            for _ in 0..12 {
                self.prng_state = self
                    .prng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                u_sum += (self.prng_state >> 33) as f64 / (u32::MAX as f64);
            }
            (u_sum - 6.0) * self.noise_std_dev_ns
        } else {
            0.0
        };

        self.phase_offset_ns + noise_ns as i64
    }

    fn apply_fault(&mut self, fault: &ClockFault, sample_spacing: Duration) {
        match fault {
            ClockFault::PhaseStep { delta_ns } => {
                self.phase_offset_ns += delta_ns;
            }
            ClockFault::FrequencyRamp { rate_ppb } => {
                self.ramp_drift_ppb += rate_ppb;
            }
            ClockFault::WhiteNoise { std_dev_ns } => {
                self.noise_std_dev_ns = *std_dev_ns;
            }
            ClockFault::Holdover { duration } => {
                let samples =
                    (duration.as_secs_f64() / sample_spacing.as_secs_f64()).ceil() as usize;
                self.holdover_samples_remaining = samples;
                // During holdover the clock free-runs; set noise to 0.
                self.noise_std_dev_ns = 0.0;
            }
            ClockFault::Reset => {
                self.ramp_drift_ppb = 0.0;
                self.noise_std_dev_ns = 0.0;
                self.holdover_samples_remaining = 0;
            }
        }
    }
}

/// A configurable synthetic clock for testing synchronisation algorithms.
///
/// The clock produces a sequence of simulated offset measurements at a
/// configurable sample spacing.  Faults from an [`InjectionScenario`] are
/// applied at specified sample indices.
#[derive(Debug)]
pub struct SyntheticClock {
    state: SyntheticClockState,
    sample_spacing: Duration,
    current_sample: usize,
    scenario: InjectionScenario,
    /// Next index into `scenario.faults` to check.
    fault_cursor: usize,
}

impl SyntheticClock {
    /// Creates a new synthetic clock.
    ///
    /// # Arguments
    /// * `base_drift_ppb`  — nominal frequency offset in ppb.
    /// * `sample_spacing`  — interval between successive observations.
    /// * `scenario`        — fault injection scenario to apply.
    #[must_use]
    pub fn new(base_drift_ppb: f64, sample_spacing: Duration, scenario: InjectionScenario) -> Self {
        Self {
            state: SyntheticClockState::new(base_drift_ppb),
            sample_spacing,
            current_sample: 0,
            scenario,
            fault_cursor: 0,
        }
    }

    /// Creates a simple clock with no fault injections.
    #[must_use]
    pub fn simple(base_drift_ppb: f64, sample_spacing: Duration) -> Self {
        Self::new(base_drift_ppb, sample_spacing, InjectionScenario::new())
    }

    /// Advances the clock by one sample, applies any pending faults, and
    /// returns the simulated offset measurement in nanoseconds.
    pub fn tick(&mut self) -> i64 {
        // Apply any faults scheduled for this sample index.
        while self.fault_cursor < self.scenario.faults.len() {
            let at = self.scenario.faults[self.fault_cursor].at_sample;
            if at == self.current_sample {
                let fault = self.scenario.faults[self.fault_cursor].fault.clone();
                self.state.apply_fault(&fault, self.sample_spacing);
                self.fault_cursor += 1;
            } else {
                break;
            }
        }

        let offset = self.state.advance(self.sample_spacing);
        self.current_sample += 1;
        offset
    }

    /// Returns the current sample index (0-based).
    #[must_use]
    pub fn current_sample(&self) -> usize {
        self.current_sample
    }

    /// Returns the true (noiseless) phase offset in nanoseconds — useful for
    /// computing ground-truth error.
    #[must_use]
    pub fn true_phase_offset_ns(&self) -> i64 {
        self.state.phase_offset_ns
    }

    /// Returns whether the clock is currently in holdover mode.
    #[must_use]
    pub fn in_holdover(&self) -> bool {
        self.state.holdover_samples_remaining > 0
    }
}

// ---------------------------------------------------------------------------
// Sync test harness
// ---------------------------------------------------------------------------

/// A single recorded sample from the test harness.
#[derive(Debug, Clone, Copy)]
pub struct HarnessSample {
    /// Zero-based sample index.
    pub index: usize,
    /// Observed (possibly noisy) offset in nanoseconds.
    pub observed_ns: i64,
    /// True (noiseless) phase offset in nanoseconds.
    pub true_ns: i64,
    /// Error = observed − true (nanoseconds).
    pub error_ns: i64,
}

/// Orchestrates a test scenario over a synthetic clock and records all
/// samples for post-hoc analysis.
#[derive(Debug)]
pub struct SyncTestHarness {
    clock: SyntheticClock,
    samples: Vec<HarnessSample>,
}

impl SyncTestHarness {
    /// Creates a harness wrapping the given [`SyntheticClock`].
    #[must_use]
    pub fn new(clock: SyntheticClock) -> Self {
        Self {
            clock,
            samples: Vec::new(),
        }
    }

    /// Advances the harness by `count` samples, recording each one.
    pub fn run(&mut self, count: usize) {
        for _ in 0..count {
            let index = self.clock.current_sample();
            let observed_ns = self.clock.tick();
            let true_ns = self.clock.true_phase_offset_ns();
            self.samples.push(HarnessSample {
                index,
                observed_ns,
                true_ns,
                error_ns: observed_ns - true_ns,
            });
        }
    }

    /// Returns all recorded samples.
    #[must_use]
    pub fn samples(&self) -> &[HarnessSample] {
        &self.samples
    }

    /// Computes the RMS (root-mean-square) error across all recorded samples.
    ///
    /// Returns `None` if no samples have been recorded.
    #[must_use]
    pub fn rms_error_ns(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let sum_sq: f64 = self
            .samples
            .iter()
            .map(|s| (s.error_ns as f64).powi(2))
            .sum();
        Some((sum_sq / self.samples.len() as f64).sqrt())
    }

    /// Returns the maximum absolute error across all recorded samples.
    ///
    /// Returns `None` if no samples have been recorded.
    #[must_use]
    pub fn max_error_ns(&self) -> Option<i64> {
        self.samples.iter().map(|s| s.error_ns.abs()).max()
    }

    /// Returns the mean true phase offset (nanoseconds) across all samples.
    ///
    /// Returns `None` if no samples have been recorded.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_true_offset_ns(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let sum: f64 = self.samples.iter().map(|s| s.true_ns as f64).sum();
        Some(sum / self.samples.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SyntheticClock — basic drift
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_drift_gives_zero_offset() {
        let mut clock = SyntheticClock::simple(0.0, Duration::from_secs(1));
        // With no drift, phase offset stays at 0.
        for _ in 0..10 {
            let obs = clock.tick();
            assert_eq!(obs, 0, "zero drift → zero offset");
        }
    }

    #[test]
    fn test_constant_drift_accumulates_linearly() {
        // 1000 ppb = 1 µs/s.  After 10 s, offset should be 10_000 ns.
        let mut clock = SyntheticClock::simple(1000.0, Duration::from_secs(1));
        for _ in 0..9 {
            clock.tick();
        }
        let offset = clock.tick(); // 10th sample
        assert_eq!(offset, 10_000, "1000 ppb × 10 s = 10_000 ns");
    }

    #[test]
    fn test_sample_index_increments() {
        let mut clock = SyntheticClock::simple(0.0, Duration::from_millis(100));
        assert_eq!(clock.current_sample(), 0);
        clock.tick();
        assert_eq!(clock.current_sample(), 1);
        clock.tick();
        assert_eq!(clock.current_sample(), 2);
    }

    // -----------------------------------------------------------------------
    // Phase step injection
    // -----------------------------------------------------------------------

    #[test]
    fn test_phase_step_shifts_offset() {
        let mut scenario = InjectionScenario::new();
        scenario.add(ScheduledFault::new(
            5,
            ClockFault::PhaseStep { delta_ns: 100_000 },
        ));
        let mut clock = SyntheticClock::new(0.0, Duration::from_secs(1), scenario);

        for i in 0..5 {
            let obs = clock.tick();
            assert_eq!(obs, 0, "before step at sample {i}");
        }
        // Sample 5 — step is applied before advancing.
        let after_step = clock.tick();
        assert_eq!(
            after_step, 100_000,
            "phase step of 100_000 ns should appear immediately"
        );
    }

    // -----------------------------------------------------------------------
    // Frequency ramp injection
    // -----------------------------------------------------------------------

    #[test]
    fn test_frequency_ramp_adds_drift() {
        let mut scenario = InjectionScenario::new();
        // At sample 0, add an extra 1000 ppb ramp.
        scenario.add(ScheduledFault::new(
            0,
            ClockFault::FrequencyRamp { rate_ppb: 1000.0 },
        ));
        let mut clock = SyntheticClock::new(0.0, Duration::from_secs(1), scenario);

        for i in 1..=5 {
            let obs = clock.tick();
            // 1000 ppb × i s = i × 1000 ns.
            assert_eq!(obs, i as i64 * 1000, "sample {i}: expected {} ns", i * 1000);
        }
    }

    // -----------------------------------------------------------------------
    // White noise injection
    // -----------------------------------------------------------------------

    #[test]
    fn test_white_noise_stays_bounded() {
        let mut scenario = InjectionScenario::new();
        scenario.add(ScheduledFault::new(
            0,
            ClockFault::WhiteNoise { std_dev_ns: 100.0 },
        ));
        let mut clock = SyntheticClock::new(0.0, Duration::from_secs(1), scenario);

        // 10σ bound: extremely unlikely to exceed ±1000 ns for σ=100 ns.
        for _ in 0..100 {
            let obs = clock.tick();
            assert!(obs.abs() <= 1000, "noise sample out of 10σ bound: {obs} ns");
        }
    }

    // -----------------------------------------------------------------------
    // Reset fault
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_clears_ramp() {
        let mut scenario = InjectionScenario::new();
        scenario.add(ScheduledFault::new(
            0,
            ClockFault::FrequencyRamp { rate_ppb: 5000.0 },
        ));
        scenario.add(ScheduledFault::new(5, ClockFault::Reset));

        let mut clock = SyntheticClock::new(0.0, Duration::from_secs(1), scenario);
        for _ in 0..5 {
            clock.tick(); // build up offset
        }
        // Save offset before reset.
        let offset_before_reset = clock.true_phase_offset_ns();
        // Sample 5 triggers Reset then advances.
        let after_reset = clock.tick();
        // After reset no additional ramp is added; offset stays at the
        // accumulated value (Reset only clears future ramp, not accumulated phase).
        assert_eq!(
            after_reset, offset_before_reset,
            "reset should stop adding ramp"
        );
    }

    // -----------------------------------------------------------------------
    // SyncTestHarness
    // -----------------------------------------------------------------------

    #[test]
    fn test_harness_rms_error_zero_noise() {
        let clock = SyntheticClock::simple(0.0, Duration::from_secs(1));
        let mut harness = SyncTestHarness::new(clock);
        harness.run(20);
        let rms = harness.rms_error_ns().expect("should compute RMS");
        assert_eq!(rms, 0.0, "no noise → RMS error = 0");
    }

    #[test]
    fn test_harness_sample_count() {
        let clock = SyntheticClock::simple(0.0, Duration::from_millis(100));
        let mut harness = SyncTestHarness::new(clock);
        harness.run(50);
        assert_eq!(harness.samples().len(), 50);
    }

    #[test]
    fn test_harness_rms_none_on_empty() {
        let clock = SyntheticClock::simple(0.0, Duration::from_secs(1));
        let harness = SyncTestHarness::new(clock);
        assert!(harness.rms_error_ns().is_none());
        assert!(harness.max_error_ns().is_none());
    }

    #[test]
    fn test_harness_mean_true_offset() {
        // With 1000 ppb drift and 1 s spacing, after N ticks the true offset
        // grows linearly; the mean should match the middle-point offset.
        let clock = SyntheticClock::simple(1000.0, Duration::from_secs(1));
        let mut harness = SyncTestHarness::new(clock);
        harness.run(10);
        let mean = harness.mean_true_offset_ns().expect("should compute mean");
        // True offsets: 1000, 2000, ..., 10000 ns → mean = 5500 ns.
        assert!(
            (mean - 5500.0).abs() < 1.0,
            "expected mean ≈ 5500 ns, got {mean}"
        );
    }
}
