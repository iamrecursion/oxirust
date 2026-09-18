//! Directional prediction angle optimization.

/// Directional prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectionalMode {
    /// Angle in degrees (0-180).
    pub angle: u8,
    /// Fine-tuned delta (-3 to +3).
    pub delta: i8,
}

impl DirectionalMode {
    /// Creates a new directional mode.
    #[must_use]
    pub const fn new(angle: u8, delta: i8) -> Self {
        Self { angle, delta }
    }

    /// Gets the effective angle.
    #[must_use]
    pub fn effective_angle(&self) -> i16 {
        i16::from(self.angle) + i16::from(self.delta)
    }
}

/// Angle optimizer for directional prediction.
pub struct AngleOptimizer {
    num_angles: usize,
    enable_delta: bool,
}

impl Default for AngleOptimizer {
    fn default() -> Self {
        Self::new(8, true)
    }
}

impl AngleOptimizer {
    /// Creates a new angle optimizer.
    #[must_use]
    pub fn new(num_angles: usize, enable_delta: bool) -> Self {
        Self {
            num_angles,
            enable_delta,
        }
    }

    /// Generates candidate directional modes.
    #[must_use]
    pub fn candidate_angles(&self) -> Vec<DirectionalMode> {
        let mut modes = Vec::new();
        let step = 180 / self.num_angles;

        for i in 0..self.num_angles {
            let base_angle = (i * step) as u8;

            if self.enable_delta {
                // Add modes with delta variations
                for delta in -3..=3 {
                    modes.push(DirectionalMode::new(base_angle, delta));
                }
            } else {
                modes.push(DirectionalMode::new(base_angle, 0));
            }
        }

        modes
    }

    /// Evaluates directional prediction quality.
    #[allow(dead_code)]
    #[must_use]
    pub fn evaluate_direction(&self, src: &[u8], neighbors: &[u8], mode: DirectionalMode) -> f64 {
        // Simplified evaluation
        // Would perform actual directional prediction in production
        let angle_cost = f64::from(mode.angle) / 180.0;
        let delta_cost = f64::from(mode.delta.abs()) * 0.1;

        let base_cost = src.iter().map(|&x| f64::from(x)).sum::<f64>() / src.len() as f64;
        let neighbor_cost =
            neighbors.iter().map(|&x| f64::from(x)).sum::<f64>() / neighbors.len().max(1) as f64;

        base_cost + angle_cost * 10.0 + delta_cost + neighbor_cost * 0.1
    }

    /// Converts angle to directional vector.
    #[must_use]
    pub fn angle_to_vector(&self, angle: i16) -> (f64, f64) {
        let radians = f64::from(angle) * std::f64::consts::PI / 180.0;
        (radians.cos(), radians.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directional_mode() {
        let mode = DirectionalMode::new(45, 2);
        assert_eq!(mode.angle, 45);
        assert_eq!(mode.delta, 2);
        assert_eq!(mode.effective_angle(), 47);
    }

    #[test]
    fn test_angle_optimizer_creation() {
        let optimizer = AngleOptimizer::default();
        assert_eq!(optimizer.num_angles, 8);
        assert!(optimizer.enable_delta);
    }

    #[test]
    fn test_candidate_angles_no_delta() {
        let optimizer = AngleOptimizer::new(4, false);
        let angles = optimizer.candidate_angles();
        assert_eq!(angles.len(), 4);
        assert!(angles.iter().all(|m| m.delta == 0));
    }

    #[test]
    fn test_candidate_angles_with_delta() {
        let optimizer = AngleOptimizer::new(4, true);
        let angles = optimizer.candidate_angles();
        assert_eq!(angles.len(), 4 * 7); // 4 base angles * 7 deltas (-3 to +3)
    }

    #[test]
    fn test_angle_to_vector() {
        let optimizer = AngleOptimizer::default();
        let (x, y) = optimizer.angle_to_vector(0);
        assert!((x - 1.0).abs() < 0.001);
        assert!(y.abs() < 0.001);

        let (x, y) = optimizer.angle_to_vector(90);
        assert!(x.abs() < 0.001);
        assert!((y - 1.0).abs() < 0.001);
    }
}
