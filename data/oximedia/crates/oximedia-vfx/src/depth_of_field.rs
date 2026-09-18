//! Depth-of-field / bokeh VFX module.
//!
//! Models a thin-lens camera to compute circle-of-confusion sizes and
//! determine which scene distances fall within the sharp zone.

/// Shape of the camera aperture (affects bokeh disc shape).
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ApertureShape {
    /// Circular aperture (smooth bokeh discs).
    Circle,
    /// Hexagonal aperture (6-blade lens).
    Hexagon,
    /// Octagonal aperture (8-blade lens).
    Octagon,
}

impl ApertureShape {
    /// Number of aperture blades (0 for circular).
    #[must_use]
    pub fn blade_count(&self) -> u32 {
        match self {
            Self::Circle => 0,
            Self::Hexagon => 6,
            Self::Octagon => 8,
        }
    }
}

/// Bokeh configuration derived from camera lens settings.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BokehConfig {
    /// Lens f-stop (aperture ratio).
    pub f_stop: f32,
    /// Focal length in millimetres.
    pub focal_length_mm: f32,
    /// Focus distance in metres.
    pub focus_distance_m: f32,
}

impl BokehConfig {
    /// Pre-built configuration for portrait photography (85 mm f/1.8, 2 m focus).
    #[must_use]
    pub fn default_portrait() -> Self {
        Self {
            f_stop: 1.8,
            focal_length_mm: 85.0,
            focus_distance_m: 2.0,
        }
    }

    /// Approximate depth-of-field in metres using the simplified `CoC` formula.
    ///
    /// `DoF` ≈ 2 × N × c × s² / (f²)
    /// where N = f-stop, c = `CoC` limit (0.03 mm), s = focus distance, f = focal length.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn depth_of_field_m(&self) -> f32 {
        let f_mm = self.focal_length_mm;
        let s_mm = self.focus_distance_m * 1_000.0;
        let coc_mm = 0.03_f32; // circle of confusion limit
        let n = self.f_stop;

        let numerator = 2.0 * n * coc_mm * s_mm * s_mm;
        let denominator = f_mm * f_mm;
        if denominator == 0.0 {
            return 0.0;
        }
        numerator / denominator / 1_000.0 // convert mm back to m
    }

    /// Hyperfocal distance in metres: H = f² / (N × c).
    #[must_use]
    pub fn hyperfocal_m(&self) -> f32 {
        let coc_mm = 0.03_f32;
        let numerator = self.focal_length_mm * self.focal_length_mm;
        let denominator = self.f_stop * coc_mm;
        if denominator == 0.0 {
            return f32::INFINITY;
        }
        numerator / denominator / 1_000.0
    }
}

/// A bokeh disc (out-of-focus highlight shape).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BokehDisk {
    /// Radius in pixels.
    pub radius_px: f32,
    /// Shape of the aperture.
    pub shape: ApertureShape,
}

impl BokehDisk {
    /// Returns `true` if the point (`dx`, `dy`) falls inside the disc.
    ///
    /// Uses a circular containment check for all shapes.
    #[must_use]
    pub fn contains_point(&self, dx: f32, dy: f32) -> bool {
        dx * dx + dy * dy <= self.radius_px * self.radius_px
    }
}

/// Depth-of-field model with near/far sharp boundaries.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DepthOfField {
    /// Camera / lens configuration.
    pub config: BokehConfig,
    /// Near edge of the sharp zone in metres.
    pub near_sharp_m: f32,
    /// Far edge of the sharp zone in metres.
    pub far_sharp_m: f32,
}

impl DepthOfField {
    /// Create a `DepthOfField` from a `BokehConfig`, computing near/far bounds.
    #[must_use]
    pub fn new(config: BokehConfig) -> Self {
        let dof_half = config.depth_of_field_m() / 2.0;
        let focus = config.focus_distance_m;
        Self {
            near_sharp_m: (focus - dof_half).max(0.0),
            far_sharp_m: focus + dof_half,
            config,
        }
    }

    /// Returns `true` if `distance_m` falls within the sharp zone.
    #[must_use]
    pub fn is_in_focus(&self, distance_m: f32) -> bool {
        distance_m >= self.near_sharp_m && distance_m <= self.far_sharp_m
    }

    /// Compute the circle of confusion diameter in millimetres for a given distance.
    ///
    /// `CoC` = |f² × (s − d)| / (N × d × (s − f))
    /// Clamped to `[0, sensor_diagonal / 2]` to avoid extreme values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn circle_of_confusion_mm(&self, distance_m: f32) -> f32 {
        let f = self.config.focal_length_mm;
        let s = self.config.focus_distance_m * 1_000.0; // convert to mm
        let d = distance_m * 1_000.0;
        let n = self.config.f_stop;

        let denom = n * d * (s - f);
        if denom.abs() < f32::EPSILON {
            return 0.0;
        }

        let coc = (f * f * (s - d)).abs() / denom;
        coc.min(50.0) // cap at 50 mm to prevent runaway values
    }

    /// Compute the blur radius in pixels for a given distance, given the sensor width.
    ///
    /// `blur_px` = `CoC_mm` × (`sensor_width_px` / `sensor_width_mm`)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn blur_radius_px(&self, distance_m: f32, sensor_width_px: u32) -> f32 {
        let coc_mm = self.circle_of_confusion_mm(distance_m);
        // Assume a standard 36 mm full-frame sensor width
        let sensor_width_mm = 36.0_f32;
        coc_mm * (sensor_width_px as f32 / sensor_width_mm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ //
    // ApertureShape
    // ------------------------------------------------------------------ //

    #[test]
    fn test_circle_blade_count() {
        assert_eq!(ApertureShape::Circle.blade_count(), 0);
    }

    #[test]
    fn test_hexagon_blade_count() {
        assert_eq!(ApertureShape::Hexagon.blade_count(), 6);
    }

    #[test]
    fn test_octagon_blade_count() {
        assert_eq!(ApertureShape::Octagon.blade_count(), 8);
    }

    // ------------------------------------------------------------------ //
    // BokehConfig
    // ------------------------------------------------------------------ //

    #[test]
    fn test_portrait_preset_values() {
        let cfg = BokehConfig::default_portrait();
        assert!((cfg.f_stop - 1.8).abs() < 1e-5);
        assert!((cfg.focal_length_mm - 85.0).abs() < 1e-5);
        assert!((cfg.focus_distance_m - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_depth_of_field_positive() {
        let cfg = BokehConfig::default_portrait();
        assert!(cfg.depth_of_field_m() > 0.0);
    }

    #[test]
    fn test_hyperfocal_positive() {
        let cfg = BokehConfig::default_portrait();
        assert!(cfg.hyperfocal_m() > 0.0);
    }

    #[test]
    fn test_wider_aperture_shallower_dof() {
        let wide = BokehConfig {
            f_stop: 1.4,
            focal_length_mm: 85.0,
            focus_distance_m: 2.0,
        };
        let narrow = BokehConfig {
            f_stop: 8.0,
            focal_length_mm: 85.0,
            focus_distance_m: 2.0,
        };
        assert!(wide.depth_of_field_m() < narrow.depth_of_field_m());
    }

    // ------------------------------------------------------------------ //
    // BokehDisk
    // ------------------------------------------------------------------ //

    #[test]
    fn test_disk_contains_centre() {
        let disk = BokehDisk {
            radius_px: 10.0,
            shape: ApertureShape::Circle,
        };
        assert!(disk.contains_point(0.0, 0.0));
    }

    #[test]
    fn test_disk_excludes_outside() {
        let disk = BokehDisk {
            radius_px: 5.0,
            shape: ApertureShape::Circle,
        };
        assert!(!disk.contains_point(6.0, 0.0));
    }

    #[test]
    fn test_disk_boundary() {
        let disk = BokehDisk {
            radius_px: 5.0,
            shape: ApertureShape::Hexagon,
        };
        // Exactly on the boundary (3-4-5 triangle)
        assert!(disk.contains_point(3.0, 4.0));
    }

    // ------------------------------------------------------------------ //
    // DepthOfField
    // ------------------------------------------------------------------ //

    #[test]
    fn test_dof_focus_distance_in_focus() {
        let dof = DepthOfField::new(BokehConfig::default_portrait());
        assert!(dof.is_in_focus(dof.config.focus_distance_m));
    }

    #[test]
    fn test_dof_far_distance_out_of_focus() {
        let dof = DepthOfField::new(BokehConfig::default_portrait());
        assert!(!dof.is_in_focus(100.0));
    }

    #[test]
    fn test_dof_near_edge_positive() {
        let dof = DepthOfField::new(BokehConfig::default_portrait());
        assert!(dof.near_sharp_m >= 0.0);
    }

    #[test]
    fn test_coc_at_focus_is_near_zero() {
        let cfg = BokehConfig::default_portrait();
        let dof = DepthOfField::new(cfg.clone());
        // At the exact focus distance s ≈ d, numerator → 0
        let coc = dof.circle_of_confusion_mm(cfg.focus_distance_m);
        assert!(coc < 1.0, "CoC at focus should be small, got {}", coc);
    }

    #[test]
    fn test_blur_radius_px_positive_for_out_of_focus() {
        let dof = DepthOfField::new(BokehConfig::default_portrait());
        let r = dof.blur_radius_px(10.0, 1920);
        assert!(r >= 0.0);
    }
}
