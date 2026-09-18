//! Speed ramping effect.

use crate::ParameterTrack;
use serde::{Deserialize, Serialize};

/// Speed curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeedCurve {
    /// Constant speed.
    Constant(f32),
    /// Variable speed with curve.
    Variable(ParameterTrack),
}

/// Speed ramp effect.
pub struct SpeedRamp {
    curve: SpeedCurve,
}

impl SpeedRamp {
    /// Create a new speed ramp with constant speed.
    #[must_use]
    pub fn new(speed: f32) -> Self {
        Self {
            curve: SpeedCurve::Constant(speed),
        }
    }

    /// Create with variable speed curve.
    #[must_use]
    pub fn with_curve(curve: ParameterTrack) -> Self {
        Self {
            curve: SpeedCurve::Variable(curve),
        }
    }

    /// Get speed at given time.
    #[must_use]
    pub fn get_speed(&self, time: f64) -> f32 {
        match &self.curve {
            SpeedCurve::Constant(speed) => *speed,
            SpeedCurve::Variable(track) => track.evaluate(time).unwrap_or(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_speed() {
        let ramp = SpeedRamp::new(2.0);
        assert_eq!(ramp.get_speed(0.0), 2.0);
        assert_eq!(ramp.get_speed(10.0), 2.0);
    }
}
