//! Depth-of-field computation for cinematographic scene analysis.
//!
//! Provides structures and algorithms for computing depth-of-field parameters,
//! hyperfocal distance, and focus zone analysis for camera/lens configurations.

#![allow(dead_code)]

/// A focus zone defined by near and far distances in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusZone {
    /// Near boundary of the in-focus region (metres).
    pub near_m: f64,
    /// Far boundary of the in-focus region (metres, may be `f64::INFINITY`).
    pub far_m: f64,
}

impl FocusZone {
    /// Creates a new `FocusZone`.
    ///
    /// # Panics
    ///
    /// Does not panic; `near_m` is clamped to 0 and `far_m` is set to `near_m` if less.
    #[must_use]
    pub fn new(near_m: f64, far_m: f64) -> Self {
        let near = near_m.max(0.0);
        let far = far_m.max(near);
        Self {
            near_m: near,
            far_m: far,
        }
    }

    /// Returns the depth (thickness) of the focus zone in metres.
    ///
    /// Returns `f64::INFINITY` when the far limit is infinite.
    #[must_use]
    pub fn depth_m(&self) -> f64 {
        self.far_m - self.near_m
    }

    /// Returns `true` when the focus zone extends to infinity.
    #[must_use]
    pub fn extends_to_infinity(&self) -> bool {
        self.far_m.is_infinite()
    }

    /// Returns `true` when `distance_m` falls within the focus zone.
    #[must_use]
    pub fn contains(&self, distance_m: f64) -> bool {
        distance_m >= self.near_m && distance_m <= self.far_m
    }
}

/// Depth-of-field parameters for a given lens/camera/focus configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthOfField {
    /// Focus zone (near and far distances in focus).
    pub focus_zone: FocusZone,
    /// Focal length of the lens in millimetres.
    pub focal_length_mm: f64,
    /// Aperture f-number.
    pub f_number: f64,
    /// Circle of confusion diameter in millimetres.
    pub coc_mm: f64,
    /// Focus distance in metres.
    pub focus_distance_m: f64,
}

impl DepthOfField {
    /// Creates a `DepthOfField` from standard photographic parameters.
    ///
    /// # Arguments
    ///
    /// * `focal_length_mm` - Focal length in mm (e.g. 50.0).
    /// * `f_number` - Aperture f-stop (e.g. 2.8).
    /// * `focus_distance_m` - Subject focus distance in metres.
    /// * `coc_mm` - Circle of confusion in mm (e.g. 0.029 for full-frame).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(
        focal_length_mm: f64,
        f_number: f64,
        focus_distance_m: f64,
        coc_mm: f64,
    ) -> Self {
        let f = focal_length_mm / 1000.0; // convert to metres
        let d = focus_distance_m;
        let c = coc_mm / 1000.0; // convert to metres

        // Depth of field formulas
        let denominator = f * f;
        let depth_factor = f_number * c * d * d;

        let near = if denominator > 0.0 {
            (d * denominator) / (denominator + depth_factor)
        } else {
            d
        };

        let far_denom = denominator - depth_factor;
        let far = if far_denom <= 0.0 {
            f64::INFINITY
        } else {
            (d * denominator) / far_denom
        };

        Self {
            focus_zone: FocusZone::new(near, far),
            focal_length_mm,
            f_number,
            coc_mm,
            focus_distance_m,
        }
    }

    /// Returns `true` if the given distance falls within the depth of field.
    #[must_use]
    pub fn is_in_focus(&self, distance_m: f64) -> bool {
        self.focus_zone.contains(distance_m)
    }

    /// Returns the hyperfocal distance in metres.
    ///
    /// Beyond the hyperfocal distance, everything from half the hyperfocal distance
    /// to infinity is acceptably sharp.
    #[must_use]
    pub fn hyperfocal_distance(&self) -> f64 {
        let f = self.focal_length_mm / 1000.0;
        let c = self.coc_mm / 1000.0;
        (f * f) / (self.f_number * c) + f
    }

    /// Returns the total depth (near to far) in metres.
    #[must_use]
    pub fn total_depth_m(&self) -> f64 {
        self.focus_zone.depth_m()
    }

    /// Returns the front depth (focus distance to near limit) in metres.
    #[must_use]
    pub fn front_depth_m(&self) -> f64 {
        (self.focus_distance_m - self.focus_zone.near_m).max(0.0)
    }

    /// Returns the rear depth (focus distance to far limit) in metres.
    #[must_use]
    pub fn rear_depth_m(&self) -> f64 {
        if self.focus_zone.far_m.is_infinite() {
            f64::INFINITY
        } else {
            (self.focus_zone.far_m - self.focus_distance_m).max(0.0)
        }
    }
}

/// Analyses depth-of-field for different camera and lens configurations.
#[derive(Debug, Clone)]
pub struct DofAnalyzer {
    /// Circle of confusion in mm (sensor-size dependent).
    pub coc_mm: f64,
}

impl DofAnalyzer {
    /// Creates a `DofAnalyzer` for a full-frame 35 mm sensor.
    #[must_use]
    pub fn full_frame() -> Self {
        Self { coc_mm: 0.029 }
    }

    /// Creates a `DofAnalyzer` for a Super 35 / APS-C sensor.
    #[must_use]
    pub fn super35() -> Self {
        Self { coc_mm: 0.019 }
    }

    /// Creates a `DofAnalyzer` with a custom circle of confusion.
    #[must_use]
    pub fn with_coc(coc_mm: f64) -> Self {
        Self {
            coc_mm: coc_mm.max(0.001),
        }
    }

    /// Computes the depth of field for the given lens/focus parameters.
    #[must_use]
    pub fn compute_dof(
        &self,
        focal_length_mm: f64,
        f_number: f64,
        focus_distance_m: f64,
    ) -> DepthOfField {
        DepthOfField::compute(focal_length_mm, f_number, focus_distance_m, self.coc_mm)
    }

    /// Returns the near and far distance (metres) of the sharp region.
    #[must_use]
    pub fn sharp_region_m(
        &self,
        focal_length_mm: f64,
        f_number: f64,
        focus_distance_m: f64,
    ) -> (f64, f64) {
        let dof = self.compute_dof(focal_length_mm, f_number, focus_distance_m);
        (dof.focus_zone.near_m, dof.focus_zone.far_m)
    }

    /// Returns `true` if the configuration yields a very shallow depth of field
    /// (total depth less than 0.5 m).
    #[must_use]
    pub fn is_shallow_dof(
        &self,
        focal_length_mm: f64,
        f_number: f64,
        focus_distance_m: f64,
    ) -> bool {
        let dof = self.compute_dof(focal_length_mm, f_number, focus_distance_m);
        dof.total_depth_m() < 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_zone_contains() {
        let zone = FocusZone::new(1.5, 3.0);
        assert!(zone.contains(2.0));
        assert!(!zone.contains(1.0));
        assert!(!zone.contains(4.0));
    }

    #[test]
    fn test_focus_zone_depth() {
        let zone = FocusZone::new(2.0, 5.0);
        assert!((zone.depth_m() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_focus_zone_infinity() {
        let zone = FocusZone::new(3.0, f64::INFINITY);
        assert!(zone.extends_to_infinity());
        assert!(zone.depth_m().is_infinite());
    }

    #[test]
    fn test_focus_zone_clamps_near() {
        let zone = FocusZone::new(-1.0, 5.0);
        assert_eq!(zone.near_m, 0.0);
    }

    #[test]
    fn test_dof_compute_focus_within_zone() {
        let dof = DepthOfField::compute(50.0, 2.8, 3.0, 0.029);
        assert!(dof.is_in_focus(3.0));
    }

    #[test]
    fn test_dof_near_less_than_focus_distance() {
        let dof = DepthOfField::compute(50.0, 2.8, 3.0, 0.029);
        assert!(dof.focus_zone.near_m < 3.0);
    }

    #[test]
    fn test_dof_far_greater_than_focus_distance() {
        let dof = DepthOfField::compute(50.0, 2.8, 3.0, 0.029);
        assert!(dof.focus_zone.far_m > 3.0);
    }

    #[test]
    fn test_hyperfocal_distance_positive() {
        let dof = DepthOfField::compute(50.0, 8.0, 10.0, 0.029);
        assert!(dof.hyperfocal_distance() > 0.0);
    }

    #[test]
    fn test_wide_aperture_shallower_dof() {
        let dof_wide = DepthOfField::compute(85.0, 1.4, 2.0, 0.029);
        let dof_narrow = DepthOfField::compute(85.0, 16.0, 2.0, 0.029);
        assert!(dof_wide.total_depth_m() < dof_narrow.total_depth_m());
    }

    #[test]
    fn test_longer_focal_shallower_dof() {
        let dof_wide = DepthOfField::compute(24.0, 5.6, 5.0, 0.029);
        let dof_tele = DepthOfField::compute(200.0, 5.6, 5.0, 0.029);
        assert!(dof_tele.total_depth_m() < dof_wide.total_depth_m());
    }

    #[test]
    fn test_dof_analyzer_full_frame_coc() {
        let analyzer = DofAnalyzer::full_frame();
        assert!((analyzer.coc_mm - 0.029).abs() < 1e-6);
    }

    #[test]
    fn test_dof_analyzer_super35_coc() {
        let analyzer = DofAnalyzer::super35();
        assert!((analyzer.coc_mm - 0.019).abs() < 1e-6);
    }

    #[test]
    fn test_sharp_region_near_far() {
        let analyzer = DofAnalyzer::full_frame();
        let (near, far) = analyzer.sharp_region_m(50.0, 5.6, 5.0);
        assert!(near < 5.0);
        assert!(far > 5.0);
    }

    #[test]
    fn test_is_shallow_dof() {
        let analyzer = DofAnalyzer::full_frame();
        // 85mm at f/1.4 focused at 1.5m should be very shallow
        assert!(analyzer.is_shallow_dof(85.0, 1.4, 1.5));
        // 24mm at f/16 focused at 5m should be deep
        assert!(!analyzer.is_shallow_dof(24.0, 16.0, 5.0));
    }

    #[test]
    fn test_front_rear_depth_sum() {
        let dof = DepthOfField::compute(50.0, 5.6, 5.0, 0.029);
        if !dof.focus_zone.extends_to_infinity() {
            let sum = dof.front_depth_m() + dof.rear_depth_m();
            assert!((sum - dof.total_depth_m()).abs() < 1e-6);
        }
    }
}
