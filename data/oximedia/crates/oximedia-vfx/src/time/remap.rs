//! Time remapping with custom curves.

use crate::ParameterTrack;
use serde::{Deserialize, Serialize};

/// Time remapping curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemapCurve {
    /// Linear mapping.
    Linear,
    /// Custom curve with keyframes.
    Custom(ParameterTrack),
}

/// Time remapping effect.
pub struct TimeRemap {
    curve: RemapCurve,
    duration: f64,
}

impl TimeRemap {
    /// Create a new time remap effect.
    #[must_use]
    pub const fn new(duration: f64) -> Self {
        Self {
            curve: RemapCurve::Linear,
            duration,
        }
    }

    /// Set custom curve.
    #[must_use]
    pub fn with_curve(mut self, curve: ParameterTrack) -> Self {
        self.curve = RemapCurve::Custom(curve);
        self
    }

    /// Map input time to output time.
    #[must_use]
    pub fn map_time(&self, time: f64) -> f64 {
        match &self.curve {
            RemapCurve::Linear => time,
            RemapCurve::Custom(track) => {
                if let Some(value) = track.evaluate(time) {
                    f64::from(value).clamp(0.0, self.duration)
                } else {
                    time
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EasingFunction;

    #[test]
    fn test_time_remap_linear() {
        let remap = TimeRemap::new(10.0);
        assert_eq!(remap.map_time(5.0), 5.0);
    }

    #[test]
    fn test_time_remap_custom() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(0.0, 0.0, EasingFunction::Linear);
        track.add_keyframe(1.0, 2.0, EasingFunction::Linear);

        let remap = TimeRemap::new(2.0).with_curve(track);
        let mapped = remap.map_time(0.5);
        assert!((mapped - 1.0).abs() < 0.1);
    }
}
