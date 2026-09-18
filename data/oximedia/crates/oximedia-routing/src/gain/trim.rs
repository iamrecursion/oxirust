//! Trim controls for fine gain adjustment.

use serde::{Deserialize, Serialize};

/// Trim control for input/output gain adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrimControl {
    /// Trim value in dB
    pub trim_db: f32,
    /// Fine trim adjustment (smaller increments)
    pub fine_trim_db: f32,
    /// Trim range minimum
    pub min_trim_db: f32,
    /// Trim range maximum
    pub max_trim_db: f32,
}

impl Default for TrimControl {
    fn default() -> Self {
        Self::new()
    }
}

impl TrimControl {
    /// Create a new trim control
    #[must_use]
    pub fn new() -> Self {
        Self {
            trim_db: 0.0,
            fine_trim_db: 0.0,
            min_trim_db: -20.0,
            max_trim_db: 20.0,
        }
    }

    /// Create trim with custom range
    #[must_use]
    pub fn with_range(min_db: f32, max_db: f32) -> Self {
        Self {
            trim_db: 0.0,
            fine_trim_db: 0.0,
            min_trim_db: min_db,
            max_trim_db: max_db,
        }
    }

    /// Set trim value
    pub fn set_trim(&mut self, trim_db: f32) {
        self.trim_db = trim_db.clamp(self.min_trim_db, self.max_trim_db);
    }

    /// Adjust trim by delta
    pub fn adjust_trim(&mut self, delta_db: f32) {
        self.set_trim(self.trim_db + delta_db);
    }

    /// Set fine trim value
    pub fn set_fine_trim(&mut self, fine_db: f32) {
        self.fine_trim_db = fine_db.clamp(-1.0, 1.0);
    }

    /// Adjust fine trim by delta
    pub fn adjust_fine_trim(&mut self, delta_db: f32) {
        self.set_fine_trim(self.fine_trim_db + delta_db);
    }

    /// Get total trim (coarse + fine)
    #[must_use]
    pub fn total_trim_db(&self) -> f32 {
        self.trim_db + self.fine_trim_db
    }

    /// Reset trim to zero
    pub fn reset(&mut self) {
        self.trim_db = 0.0;
        self.fine_trim_db = 0.0;
    }

    /// Check if trim is at unity (0 dB)
    #[must_use]
    pub fn is_unity(&self) -> bool {
        self.total_trim_db().abs() < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_creation() {
        let trim = TrimControl::new();
        assert_eq!(trim.trim_db, 0.0);
        assert!(trim.is_unity());
    }

    #[test]
    fn test_set_trim() {
        let mut trim = TrimControl::new();
        trim.set_trim(5.0);
        assert!((trim.trim_db - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trim_clamping() {
        let mut trim = TrimControl::new();
        trim.set_trim(100.0);
        assert!(trim.trim_db <= trim.max_trim_db);

        trim.set_trim(-100.0);
        assert!(trim.trim_db >= trim.min_trim_db);
    }

    #[test]
    fn test_adjust_trim() {
        let mut trim = TrimControl::new();
        trim.adjust_trim(3.0);
        assert!((trim.trim_db - 3.0).abs() < f32::EPSILON);

        trim.adjust_trim(-1.0);
        assert!((trim.trim_db - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fine_trim() {
        let mut trim = TrimControl::new();
        trim.set_fine_trim(0.5);
        assert!((trim.fine_trim_db - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_total_trim() {
        let mut trim = TrimControl::new();
        trim.set_trim(5.0);
        trim.set_fine_trim(0.3);

        let total = trim.total_trim_db();
        assert!((total - 5.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut trim = TrimControl::new();
        trim.set_trim(5.0);
        trim.set_fine_trim(0.3);

        trim.reset();
        assert!(trim.is_unity());
    }

    #[test]
    fn test_custom_range() {
        let trim = TrimControl::with_range(-10.0, 10.0);
        assert!((trim.min_trim_db - (-10.0)).abs() < f32::EPSILON);
        assert!((trim.max_trim_db - 10.0).abs() < f32::EPSILON);
    }
}
