//! Automation system for parameter control over time.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Automation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomationMode {
    /// Read automation data
    Read,
    /// Write automation data (overwrite)
    Write,
    /// Touch mode (write when touching control)
    Touch,
    /// Latch mode (write after touch, continue until stop)
    Latch,
}

/// Automation curve type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveType {
    /// Linear interpolation
    Linear,
    /// Bezier curve
    Bezier,
    /// Stepped (no interpolation)
    Stepped,
    /// Exponential
    Exponential,
    /// Logarithmic
    Logarithmic,
}

/// Automation point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AutomationPoint {
    /// Time in seconds
    pub time: f64,
    /// Value
    pub value: f32,
    /// Curve type to next point
    pub curve_type: CurveType,
    /// Bezier control point (for bezier curves)
    pub bezier_control: Option<(f64, f32)>,
}

impl AutomationPoint {
    /// Create a new automation point
    #[must_use]
    pub fn new(time: f64, value: f32) -> Self {
        Self {
            time,
            value,
            curve_type: CurveType::Linear,
            bezier_control: None,
        }
    }

    /// Create a bezier automation point
    #[must_use]
    pub fn new_bezier(time: f64, value: f32, control_time: f64, control_value: f32) -> Self {
        Self {
            time,
            value,
            curve_type: CurveType::Bezier,
            bezier_control: Some((control_time, control_value)),
        }
    }
}

/// Automation lane for a single parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationLane {
    /// Lane name
    pub name: String,
    /// Automation mode
    pub mode: AutomationMode,
    /// Automation points (time -> point)
    points: BTreeMap<u64, AutomationPoint>, // Using u64 for time key (microseconds)
    /// Enabled flag
    pub enabled: bool,
}

impl AutomationLane {
    /// Create a new automation lane
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            mode: AutomationMode::Read,
            points: BTreeMap::new(),
            enabled: true,
        }
    }

    /// Add an automation point
    pub fn add_point(&mut self, point: AutomationPoint) {
        let time_key = (point.time * 1_000_000.0) as u64;
        self.points.insert(time_key, point);
    }

    /// Remove an automation point at time
    ///
    /// # Errors
    ///
    /// Returns an error if no point exists at the given time
    pub fn remove_point(&mut self, time: f64) -> AudioPostResult<()> {
        let time_key = (time * 1_000_000.0) as u64;
        self.points
            .remove(&time_key)
            .ok_or(AudioPostError::AutomationPointNotFound(time))?;
        Ok(())
    }

    /// Get value at a specific time
    #[must_use]
    pub fn get_value_at_time(&self, time: f64) -> f32 {
        if !self.enabled || self.points.is_empty() {
            return 0.0;
        }

        let time_key = (time * 1_000_000.0) as u64;

        // Find surrounding points
        let before = self.points.range(..=time_key).next_back();
        let after = self.points.range((time_key + 1)..).next();

        match (before, after) {
            (Some((_, p1)), Some((_, p2))) => {
                // Interpolate between points
                self.interpolate(p1, p2, time)
            }
            (Some((_, p)), None) => p.value, // After last point
            (None, Some((_, p))) => p.value, // Before first point
            (None, None) => 0.0,             // No points
        }
    }

    /// Interpolate between two points
    fn interpolate(&self, p1: &AutomationPoint, p2: &AutomationPoint, time: f64) -> f32 {
        if time <= p1.time {
            return p1.value;
        }
        if time >= p2.time {
            return p2.value;
        }

        let t = ((time - p1.time) / (p2.time - p1.time)) as f32;

        match p1.curve_type {
            CurveType::Linear => p1.value + (p2.value - p1.value) * t,
            CurveType::Stepped => p1.value,
            CurveType::Exponential => p1.value + (p2.value - p1.value) * t * t,
            CurveType::Logarithmic => p1.value + (p2.value - p1.value) * t.sqrt(),
            CurveType::Bezier => {
                if let Some((ct, cv)) = p1.bezier_control {
                    // Simplified bezier (quadratic)
                    let _ct_norm = ((ct - p1.time) / (p2.time - p1.time)) as f32;
                    let cv_norm = (cv - p1.value) / (p2.value - p1.value);

                    let u = 1.0 - t;
                    p1.value
                        + (p2.value - p1.value)
                            * (u * u * 0.0 + 2.0 * u * t * cv_norm + t * t * 1.0)
                } else {
                    p1.value + (p2.value - p1.value) * t
                }
            }
        }
    }

    /// Get all points
    #[must_use]
    pub fn get_points(&self) -> Vec<&AutomationPoint> {
        self.points.values().collect()
    }

    /// Get point count
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Automation manager for multiple parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationManager {
    /// Automation lanes
    lanes: BTreeMap<String, AutomationLane>,
}

impl AutomationManager {
    /// Create a new automation manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            lanes: BTreeMap::new(),
        }
    }

    /// Add an automation lane
    pub fn add_lane(&mut self, lane: AutomationLane) {
        self.lanes.insert(lane.name.clone(), lane);
    }

    /// Get an automation lane
    #[must_use]
    pub fn get_lane(&self, name: &str) -> Option<&AutomationLane> {
        self.lanes.get(name)
    }

    /// Get a mutable automation lane
    pub fn get_lane_mut(&mut self, name: &str) -> Option<&mut AutomationLane> {
        self.lanes.get_mut(name)
    }

    /// Remove an automation lane
    pub fn remove_lane(&mut self, name: &str) -> Option<AutomationLane> {
        self.lanes.remove(name)
    }

    /// Get all lane names
    #[must_use]
    pub fn get_lane_names(&self) -> Vec<&str> {
        self.lanes.keys().map(String::as_str).collect()
    }

    /// Get value for a parameter at time
    #[must_use]
    pub fn get_value(&self, parameter: &str, time: f64) -> Option<f32> {
        self.lanes
            .get(parameter)
            .map(|lane| lane.get_value_at_time(time))
    }
}

impl Default for AutomationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume automation
#[derive(Debug)]
pub struct VolumeAutomation {
    lane: AutomationLane,
}

impl VolumeAutomation {
    /// Create new volume automation
    #[must_use]
    pub fn new() -> Self {
        Self {
            lane: AutomationLane::new("Volume"),
        }
    }

    /// Add volume point (in dB)
    pub fn add_point(&mut self, time: f64, volume_db: f32) {
        self.lane.add_point(AutomationPoint::new(time, volume_db));
    }

    /// Get volume at time (in dB)
    #[must_use]
    pub fn get_volume_db(&self, time: f64) -> f32 {
        self.lane.get_value_at_time(time)
    }

    /// Get linear gain at time
    #[must_use]
    pub fn get_linear_gain(&self, time: f64) -> f32 {
        let db = self.get_volume_db(time);
        10.0_f32.powf(db / 20.0)
    }
}

impl Default for VolumeAutomation {
    fn default() -> Self {
        Self::new()
    }
}

/// Pan automation
#[derive(Debug)]
pub struct PanAutomation {
    lane: AutomationLane,
}

impl PanAutomation {
    /// Create new pan automation
    #[must_use]
    pub fn new() -> Self {
        Self {
            lane: AutomationLane::new("Pan"),
        }
    }

    /// Add pan point (-1.0 to 1.0)
    pub fn add_point(&mut self, time: f64, pan: f32) {
        self.lane.add_point(AutomationPoint::new(time, pan));
    }

    /// Get pan at time
    #[must_use]
    pub fn get_pan(&self, time: f64) -> f32 {
        self.lane.get_value_at_time(time).clamp(-1.0, 1.0)
    }
}

impl Default for PanAutomation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_point() {
        let point = AutomationPoint::new(1.0, 0.5);
        assert_eq!(point.time, 1.0);
        assert_eq!(point.value, 0.5);
    }

    #[test]
    fn test_bezier_point() {
        let point = AutomationPoint::new_bezier(1.0, 0.5, 1.5, 0.75);
        assert_eq!(point.curve_type, CurveType::Bezier);
        assert!(point.bezier_control.is_some());
    }

    #[test]
    fn test_automation_lane() {
        let mut lane = AutomationLane::new("Volume");
        lane.add_point(AutomationPoint::new(0.0, 0.0));
        lane.add_point(AutomationPoint::new(1.0, 1.0));
        assert_eq!(lane.point_count(), 2);
    }

    #[test]
    fn test_get_value_at_time() {
        let mut lane = AutomationLane::new("Volume");
        lane.add_point(AutomationPoint::new(0.0, 0.0));
        lane.add_point(AutomationPoint::new(1.0, 1.0));

        let value = lane.get_value_at_time(0.5);
        assert!((value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_remove_point() {
        let mut lane = AutomationLane::new("Volume");
        lane.add_point(AutomationPoint::new(1.0, 0.5));
        assert!(lane.remove_point(1.0).is_ok());
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_stepped_interpolation() {
        let mut lane = AutomationLane::new("Test");
        let mut p1 = AutomationPoint::new(0.0, 0.0);
        p1.curve_type = CurveType::Stepped;
        lane.add_point(p1);
        lane.add_point(AutomationPoint::new(1.0, 1.0));

        let value = lane.get_value_at_time(0.5);
        assert_eq!(value, 0.0); // Should be stepped
    }

    #[test]
    fn test_automation_manager() {
        let mut manager = AutomationManager::new();
        let lane = AutomationLane::new("Volume");
        manager.add_lane(lane);
        assert!(manager.get_lane("Volume").is_some());
    }

    #[test]
    fn test_manager_get_value() {
        let mut manager = AutomationManager::new();
        let mut lane = AutomationLane::new("Volume");
        lane.add_point(AutomationPoint::new(0.0, 0.0));
        lane.add_point(AutomationPoint::new(1.0, 1.0));
        manager.add_lane(lane);

        let value = manager.get_value("Volume", 0.5);
        assert!(value.is_some());
    }

    #[test]
    fn test_volume_automation() {
        let mut vol = VolumeAutomation::new();
        vol.add_point(0.0, -6.0);
        vol.add_point(1.0, 0.0);

        let db = vol.get_volume_db(0.5);
        assert!((db - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_volume_linear_gain() {
        let mut vol = VolumeAutomation::new();
        vol.add_point(0.0, 0.0);

        let gain = vol.get_linear_gain(0.0);
        assert!((gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pan_automation() {
        let mut pan = PanAutomation::new();
        pan.add_point(0.0, -1.0);
        pan.add_point(1.0, 1.0);

        let pan_value = pan.get_pan(0.5);
        assert!((pan_value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clear_lane() {
        let mut lane = AutomationLane::new("Test");
        lane.add_point(AutomationPoint::new(0.0, 0.0));
        lane.add_point(AutomationPoint::new(1.0, 1.0));
        lane.clear();
        assert_eq!(lane.point_count(), 0);
    }

    #[test]
    fn test_exponential_curve() {
        let mut lane = AutomationLane::new("Test");
        let mut p1 = AutomationPoint::new(0.0, 0.0);
        p1.curve_type = CurveType::Exponential;
        lane.add_point(p1);
        lane.add_point(AutomationPoint::new(1.0, 1.0));

        let value = lane.get_value_at_time(0.5);
        assert!(value < 0.5); // Exponential should be less than linear at midpoint
    }

    #[test]
    fn test_logarithmic_curve() {
        let mut lane = AutomationLane::new("Test");
        let mut p1 = AutomationPoint::new(0.0, 0.0);
        p1.curve_type = CurveType::Logarithmic;
        lane.add_point(p1);
        lane.add_point(AutomationPoint::new(1.0, 1.0));

        let value = lane.get_value_at_time(0.5);
        assert!(value > 0.5); // Logarithmic should be more than linear at midpoint
    }
}
