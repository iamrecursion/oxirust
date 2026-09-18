//! Clock discipline with PID controller.

use super::{ClockSource, ClockStats, SyncState};
use crate::error::TimeSyncResult;
use std::time::Instant;

/// PID controller for clock discipline.
#[derive(Debug, Clone)]
pub struct PidController {
    /// Proportional gain
    kp: f64,
    /// Integral gain
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// Integral accumulator
    integral: f64,
    /// Last error
    last_error: Option<f64>,
    /// Last update time
    last_update: Option<Instant>,
    /// Integral windup limit
    integral_limit: f64,
}

impl PidController {
    /// Create a new PID controller.
    #[must_use]
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            last_error: None,
            last_update: None,
            integral_limit: 1000.0, // Prevent windup
        }
    }

    /// Create PID controller with default gains for clock discipline.
    #[must_use]
    pub fn default_clock_discipline() -> Self {
        // Typical PID gains for clock discipline
        Self::new(0.1, 0.01, 0.001)
    }

    /// Update controller with new error measurement.
    pub fn update(&mut self, error: f64, now: Instant) -> f64 {
        let dt = if let Some(last_update) = self.last_update {
            now.duration_since(last_update).as_secs_f64()
        } else {
            1.0 // First update
        };

        // Proportional term
        let p_term = self.kp * error;

        // Integral term
        self.integral += error * dt;
        // Anti-windup
        if self.integral > self.integral_limit {
            self.integral = self.integral_limit;
        } else if self.integral < -self.integral_limit {
            self.integral = -self.integral_limit;
        }
        let i_term = self.ki * self.integral;

        // Derivative term
        let d_term = if let Some(last_error) = self.last_error {
            self.kd * (error - last_error) / dt
        } else {
            0.0
        };

        self.last_error = Some(error);
        self.last_update = Some(now);

        p_term + i_term + d_term
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = None;
        self.last_update = None;
    }

    /// Get current integral value.
    #[must_use]
    pub fn integral(&self) -> f64 {
        self.integral
    }
}

/// Clock discipline algorithm.
pub struct ClockDiscipline {
    /// PID controller
    pid: PidController,
    /// Current state
    state: SyncState,
    /// Clock source
    source: ClockSource,
    /// State start time
    state_start: Instant,
    /// Minimum offset for adjustment (nanoseconds)
    min_offset_threshold: i64,
    /// Maximum offset for slewing (nanoseconds)
    max_slew_offset: i64,
    /// Current frequency adjustment (ppb)
    freq_adjust_ppb: f64,
}

impl ClockDiscipline {
    /// Create a new clock discipline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pid: PidController::default_clock_discipline(),
            state: SyncState::Unsync,
            source: ClockSource::System,
            state_start: Instant::now(),
            min_offset_threshold: 1000,   // 1 microsecond
            max_slew_offset: 100_000_000, // 100 milliseconds
            freq_adjust_ppb: 0.0,
        }
    }

    /// Update discipline with new offset measurement.
    pub fn update(
        &mut self,
        offset_ns: i64,
        source: ClockSource,
    ) -> TimeSyncResult<ClockAdjustment> {
        let now = Instant::now();

        // Check if offset is below threshold
        if offset_ns.abs() < self.min_offset_threshold {
            // Already well synchronized
            if self.state != SyncState::Synced {
                self.transition_state(SyncState::Synced);
            }
            return Ok(ClockAdjustment::None);
        }

        // Determine adjustment strategy based on offset magnitude
        let adjustment = if offset_ns.abs() > self.max_slew_offset {
            // Step adjustment for large offsets
            if self.state != SyncState::Syncing {
                self.transition_state(SyncState::Syncing);
            }
            ClockAdjustment::Step(offset_ns)
        } else {
            // Slew adjustment using PID controller
            if self.state == SyncState::Unsync {
                self.transition_state(SyncState::Syncing);
            }

            let correction = self.pid.update(offset_ns as f64, now);
            self.freq_adjust_ppb = correction;

            ClockAdjustment::Slew {
                offset_ns,
                freq_adjust_ppb: correction,
            }
        };

        self.source = source;

        // Check if we should transition to synced state
        if self.state == SyncState::Syncing && offset_ns.abs() < self.min_offset_threshold * 10 {
            self.transition_state(SyncState::Synced);
        }

        Ok(adjustment)
    }

    /// Enter holdover mode (no external reference).
    pub fn enter_holdover(&mut self) {
        self.transition_state(SyncState::Holdover);
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> ClockStats {
        ClockStats {
            offset_ns: 0, // Would be tracked separately
            freq_offset_ppb: self.freq_adjust_ppb,
            jitter_ns: 0, // Would be calculated from measurements
            state_time: self.state_start.elapsed(),
            state: self.state,
            source: self.source,
        }
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> SyncState {
        self.state
    }

    /// Get current source.
    #[must_use]
    pub fn source(&self) -> ClockSource {
        self.source
    }

    /// Transition to a new state.
    fn transition_state(&mut self, new_state: SyncState) {
        if new_state != self.state {
            self.state = new_state;
            self.state_start = Instant::now();

            // Reset PID on state transitions
            if new_state == SyncState::Syncing {
                self.pid.reset();
            }
        }
    }
}

impl Default for ClockDiscipline {
    fn default() -> Self {
        Self::new()
    }
}

/// Clock adjustment recommendation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockAdjustment {
    /// No adjustment needed
    None,
    /// Step adjustment (set clock directly)
    Step(i64),
    /// Slew adjustment (gradual frequency correction)
    Slew {
        /// Offset to correct (nanoseconds)
        offset_ns: i64,
        /// Frequency adjustment (ppb)
        freq_adjust_ppb: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.01);
        let now = Instant::now();

        let output1 = pid.update(100.0, now);
        assert!(output1 > 0.0);

        let output2 = pid.update(50.0, now + Duration::from_secs(1));
        assert!(output2 < output1); // Error decreasing
    }

    #[test]
    fn test_clock_discipline() {
        let mut discipline = ClockDiscipline::new();
        assert_eq!(discipline.state(), SyncState::Unsync);

        // Small offset - should suggest no adjustment
        let adj = discipline
            .update(500, ClockSource::Ntp)
            .expect("should succeed in test");
        assert_eq!(adj, ClockAdjustment::None);

        // Large offset - should suggest step
        let adj = discipline
            .update(200_000_000, ClockSource::Ntp)
            .expect("should succeed in test");
        match adj {
            ClockAdjustment::Step(_) => {}
            _ => panic!("Expected step adjustment"),
        }

        // Medium offset - should suggest slew
        let adj = discipline
            .update(10_000, ClockSource::Ntp)
            .expect("should succeed in test");
        match adj {
            ClockAdjustment::Slew { .. } => {}
            _ => panic!("Expected slew adjustment"),
        }
    }

    #[test]
    fn test_holdover() {
        let mut discipline = ClockDiscipline::new();
        discipline
            .update(10_000, ClockSource::Ptp)
            .expect("should succeed in test");

        discipline.enter_holdover();
        assert_eq!(discipline.state(), SyncState::Holdover);
    }
}
