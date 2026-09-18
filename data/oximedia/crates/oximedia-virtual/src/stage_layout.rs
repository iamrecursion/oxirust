//! Virtual production stage layout management.
//!
//! Provides tools for defining and querying the physical layout of a virtual
//! production stage, including LED wall panels, acting areas, and technical zones.

#![allow(dead_code)]

/// Zone type for a stage area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneType {
    /// Performance acting area.
    ActingArea,
    /// Technical crew / equipment area.
    TechArea,
    /// Safety margin around obstacles.
    SafetyMargin,
    /// LED wall surface zone.
    LedWall,
    /// Camera dolly / crane position.
    CameraPosition,
}

impl ZoneType {
    /// Returns `true` if a person or equipment is typically present in this zone.
    #[must_use]
    pub fn is_occupied(&self) -> bool {
        matches!(
            self,
            ZoneType::ActingArea | ZoneType::TechArea | ZoneType::CameraPosition
        )
    }
}

/// A rectangular zone within the stage floor plan.
#[derive(Debug, Clone)]
pub struct StageZone {
    /// Unique zone identifier.
    pub id: u32,
    /// Human-readable zone name.
    pub name: String,
    /// Left edge X coordinate in metres.
    pub x_m: f32,
    /// Bottom edge Y coordinate in metres.
    pub y_m: f32,
    /// Zone width in metres.
    pub width_m: f32,
    /// Zone height (depth) in metres.
    pub height_m: f32,
    /// Functional type of this zone.
    pub zone_type: ZoneType,
}

impl StageZone {
    /// Compute the floor area of this zone in square metres.
    #[must_use]
    pub fn area_sqm(&self) -> f32 {
        self.width_m * self.height_m
    }

    /// Returns `true` if the point `(x, y)` falls inside this zone (inclusive bounds).
    #[must_use]
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.x_m
            && x <= self.x_m + self.width_m
            && y >= self.y_m
            && y <= self.y_m + self.height_m
    }
}

/// A single LED wall panel with physical and pixel dimensions.
#[derive(Debug, Clone)]
pub struct LedWallPanel {
    /// Panel identifier.
    pub id: u32,
    /// X position of the panel's left edge in metres.
    pub x_m: f32,
    /// Y position of the panel's bottom edge in metres.
    pub y_m: f32,
    /// Z (height from floor) of the panel's bottom edge in metres.
    pub z_m: f32,
    /// Physical panel width in metres.
    pub width_m: f32,
    /// Physical panel height in metres.
    pub height_m: f32,
    /// Pixel resolution `(width_px, height_px)`.
    pub resolution: (u32, u32),
}

impl LedWallPanel {
    /// Calculate pixel pitch in millimetres given the physical width in millimetres.
    ///
    /// Pixel pitch is the centre-to-centre distance between adjacent pixels.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pixel_pitch_mm(&self, physical_width_mm: f32) -> f32 {
        let px_wide = self.resolution.0 as f32;
        if px_wide == 0.0 {
            return 0.0;
        }
        physical_width_mm / px_wide
    }

    /// Aspect ratio of the panel (width / height).
    #[must_use]
    pub fn panel_aspect_ratio(&self) -> f32 {
        if self.height_m == 0.0 {
            return 0.0;
        }
        self.width_m / self.height_m
    }
}

/// Complete virtual production stage layout.
#[derive(Debug, Clone, Default)]
pub struct StageLayout {
    /// All defined zones on the stage floor plan.
    pub zones: Vec<StageZone>,
    /// All LED wall panels installed in the stage.
    pub led_panels: Vec<LedWallPanel>,
    /// Total stage width in metres.
    pub total_width_m: f32,
    /// Total stage depth in metres.
    pub total_depth_m: f32,
}

impl StageLayout {
    /// Total floor area covered by [`ZoneType::LedWall`] zones in square metres.
    #[must_use]
    pub fn led_zone_area(&self) -> f32 {
        self.zones
            .iter()
            .filter(|z| z.zone_type == ZoneType::LedWall)
            .map(StageZone::area_sqm)
            .sum()
    }

    /// Total floor area covered by [`ZoneType::ActingArea`] zones in square metres.
    #[must_use]
    pub fn acting_area(&self) -> f32 {
        self.zones
            .iter()
            .filter(|z| z.zone_type == ZoneType::ActingArea)
            .map(StageZone::area_sqm)
            .sum()
    }

    /// Number of LED wall panels installed.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.led_panels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_zone(id: u32, zone_type: ZoneType) -> StageZone {
        StageZone {
            id,
            name: format!("zone_{id}"),
            x_m: 0.0,
            y_m: 0.0,
            width_m: 4.0,
            height_m: 3.0,
            zone_type,
        }
    }

    fn make_panel(id: u32, w: f32, h: f32, res: (u32, u32)) -> LedWallPanel {
        LedWallPanel {
            id,
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
            width_m: w,
            height_m: h,
            resolution: res,
        }
    }

    #[test]
    fn test_zone_type_is_occupied_acting() {
        assert!(ZoneType::ActingArea.is_occupied());
    }

    #[test]
    fn test_zone_type_is_occupied_tech() {
        assert!(ZoneType::TechArea.is_occupied());
    }

    #[test]
    fn test_zone_type_is_occupied_camera() {
        assert!(ZoneType::CameraPosition.is_occupied());
    }

    #[test]
    fn test_zone_type_not_occupied_led_wall() {
        assert!(!ZoneType::LedWall.is_occupied());
    }

    #[test]
    fn test_zone_type_not_occupied_safety() {
        assert!(!ZoneType::SafetyMargin.is_occupied());
    }

    #[test]
    fn test_stage_zone_area() {
        let zone = make_zone(1, ZoneType::ActingArea);
        assert!((zone.area_sqm() - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_zone_contains_point_inside() {
        let zone = make_zone(1, ZoneType::ActingArea);
        assert!(zone.contains_point(2.0, 1.5));
    }

    #[test]
    fn test_stage_zone_contains_point_outside() {
        let zone = make_zone(1, ZoneType::ActingArea);
        assert!(!zone.contains_point(5.0, 1.5));
    }

    #[test]
    fn test_stage_zone_contains_point_boundary() {
        let zone = make_zone(1, ZoneType::ActingArea);
        // inclusive bounds
        assert!(zone.contains_point(0.0, 0.0));
        assert!(zone.contains_point(4.0, 3.0));
    }

    #[test]
    fn test_led_wall_panel_pixel_pitch() {
        // 500 mm wide, 500 pixels → 1 mm pitch
        let panel = make_panel(1, 0.5, 1.0, (500, 1000));
        let pitch = panel.pixel_pitch_mm(500.0);
        assert!((pitch - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_led_wall_panel_pixel_pitch_zero_res() {
        let panel = make_panel(1, 1.0, 1.0, (0, 0));
        assert_eq!(panel.pixel_pitch_mm(1000.0), 0.0);
    }

    #[test]
    fn test_led_wall_panel_aspect_ratio() {
        let panel = make_panel(1, 2.0, 1.0, (1920, 960));
        assert!((panel.panel_aspect_ratio() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_stage_layout_led_zone_area() {
        let mut layout = StageLayout {
            total_width_m: 20.0,
            total_depth_m: 15.0,
            ..Default::default()
        };
        layout.zones.push(make_zone(1, ZoneType::LedWall));
        layout.zones.push(make_zone(2, ZoneType::LedWall));
        layout.zones.push(make_zone(3, ZoneType::ActingArea));
        assert!((layout.led_zone_area() - 24.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_layout_acting_area() {
        let mut layout = StageLayout::default();
        layout.zones.push(make_zone(1, ZoneType::ActingArea));
        layout.zones.push(make_zone(2, ZoneType::TechArea));
        assert!((layout.acting_area() - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_layout_panel_count() {
        let mut layout = StageLayout::default();
        layout.led_panels.push(make_panel(1, 0.5, 1.0, (500, 1000)));
        layout.led_panels.push(make_panel(2, 0.5, 1.0, (500, 1000)));
        assert_eq!(layout.panel_count(), 2);
    }
}
