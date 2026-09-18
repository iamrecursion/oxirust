#![allow(dead_code)]
//! Stage zone management for virtual production environments.
//!
//! Defines zones within an LED stage (talent area, background wall, ceiling,
//! side panels, etc.) and provides a manager for coordinating them.

/// Zones that make up a virtual production stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageZone {
    /// Primary background LED wall.
    BackWall,
    /// Left side LED panel.
    SideLeft,
    /// Right side LED panel.
    SideRight,
    /// Ceiling LED canopy.
    Ceiling,
    /// Floor projection surface.
    Floor,
    /// Performance area for talent (no display surface).
    TalentArea,
}

impl StageZone {
    /// Returns true if this zone is an active display surface.
    #[must_use]
    pub fn is_display_surface(&self) -> bool {
        !matches!(self, StageZone::TalentArea)
    }

    /// Returns the nominal fill fraction relative to a full 360-degree
    /// stage enclosure (rough approximation, sums to ≤ 1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn coverage_fraction(&self) -> f32 {
        match self {
            StageZone::BackWall => 0.30,
            StageZone::SideLeft => 0.15,
            StageZone::SideRight => 0.15,
            StageZone::Ceiling => 0.20,
            StageZone::Floor => 0.10,
            StageZone::TalentArea => 0.10,
        }
    }
}

impl std::fmt::Display for StageZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageZone::BackWall => write!(f, "BackWall"),
            StageZone::SideLeft => write!(f, "SideLeft"),
            StageZone::SideRight => write!(f, "SideRight"),
            StageZone::Ceiling => write!(f, "Ceiling"),
            StageZone::Floor => write!(f, "Floor"),
            StageZone::TalentArea => write!(f, "TalentArea"),
        }
    }
}

/// Physical dimensions of a stage zone in metres.
#[derive(Debug, Clone, Copy)]
pub struct ZoneDimensions {
    /// Width in metres.
    pub width_m: f32,
    /// Height in metres.
    pub height_m: f32,
    /// Depth in metres (relevant for wrap-around surfaces).
    pub depth_m: f32,
}

impl ZoneDimensions {
    /// Create dimensions.
    #[must_use]
    pub fn new(width_m: f32, height_m: f32, depth_m: f32) -> Self {
        Self {
            width_m,
            height_m,
            depth_m,
        }
    }

    /// Surface area in square metres (width × height).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn surface_area_m2(&self) -> f32 {
        self.width_m * self.height_m
    }
}

/// Activation state of a stage zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneState {
    /// Zone is online and receiving content.
    Online,
    /// Zone is configured but not currently showing content.
    Standby,
    /// Zone has encountered an error.
    Faulted,
}

impl std::fmt::Display for ZoneState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZoneState::Online => write!(f, "Online"),
            ZoneState::Standby => write!(f, "Standby"),
            ZoneState::Faulted => write!(f, "Faulted"),
        }
    }
}

/// A stage layout entry binding a zone to its dimensions and state.
#[derive(Debug, Clone)]
pub struct StageLayout {
    /// Zone identifier.
    pub zone: StageZone,
    /// Physical dimensions.
    pub dimensions: ZoneDimensions,
    /// Current operational state.
    pub state: ZoneState,
    /// LED panel pixel pitch in millimetres (None if not a display surface).
    pub pixel_pitch_mm: Option<f32>,
}

impl StageLayout {
    /// Create a new stage layout entry in `Standby` state.
    #[must_use]
    pub fn new(zone: StageZone, dimensions: ZoneDimensions, pixel_pitch_mm: Option<f32>) -> Self {
        Self {
            zone,
            dimensions,
            state: ZoneState::Standby,
            pixel_pitch_mm,
        }
    }

    /// Bring the zone online.
    pub fn bring_online(&mut self) {
        if self.state != ZoneState::Faulted {
            self.state = ZoneState::Online;
        }
    }

    /// Put the zone into standby.
    pub fn put_standby(&mut self) {
        self.state = ZoneState::Standby;
    }

    /// Mark the zone as faulted.
    pub fn fault(&mut self) {
        self.state = ZoneState::Faulted;
    }

    /// Estimated horizontal pixel resolution given the width and pixel pitch.
    #[must_use]
    pub fn pixel_width(&self) -> Option<u32> {
        self.pixel_pitch_mm.map(|pitch_mm| {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let w = ((self.dimensions.width_m * 1000.0) / pitch_mm) as u32;
            w
        })
    }

    /// Estimated vertical pixel resolution given the height and pixel pitch.
    #[must_use]
    pub fn pixel_height(&self) -> Option<u32> {
        self.pixel_pitch_mm.map(|pitch_mm| {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let h = ((self.dimensions.height_m * 1000.0) / pitch_mm) as u32;
            h
        })
    }

    /// Returns true if this zone is online and showing content.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == ZoneState::Online
    }
}

/// Manages all stage zones for a virtual production set.
#[derive(Debug, Default)]
pub struct StageLayoutManager {
    zones: Vec<StageLayout>,
}

impl StageLayoutManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a zone layout.
    pub fn add_zone(&mut self, layout: StageLayout) {
        self.zones.push(layout);
    }

    /// Return an immutable slice of all zones.
    #[must_use]
    pub fn zones(&self) -> &[StageLayout] {
        &self.zones
    }

    /// Find a zone by type.
    #[must_use]
    pub fn find_zone(&self, zone: StageZone) -> Option<&StageLayout> {
        self.zones.iter().find(|z| z.zone == zone)
    }

    /// Find a mutable zone by type.
    pub fn find_zone_mut(&mut self, zone: StageZone) -> Option<&mut StageLayout> {
        self.zones.iter_mut().find(|z| z.zone == zone)
    }

    /// Bring all registered display surface zones online.
    pub fn bring_all_online(&mut self) {
        for z in &mut self.zones {
            if z.zone.is_display_surface() {
                z.bring_online();
            }
        }
    }

    /// Returns the count of zones currently online.
    #[must_use]
    pub fn online_count(&self) -> usize {
        self.zones.iter().filter(|z| z.is_active()).count()
    }

    /// Returns the total surface area of all online display zones in m².
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_online_area_m2(&self) -> f32 {
        self.zones
            .iter()
            .filter(|z| z.is_active() && z.zone.is_display_surface())
            .map(|z| z.dimensions.surface_area_m2())
            .sum()
    }

    /// Returns the count of faulted zones.
    #[must_use]
    pub fn faulted_count(&self) -> usize {
        self.zones
            .iter()
            .filter(|z| z.state == ZoneState::Faulted)
            .count()
    }

    /// Returns total number of registered zones.
    #[must_use]
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_back_wall() -> StageLayout {
        StageLayout::new(
            StageZone::BackWall,
            ZoneDimensions::new(12.0, 4.5, 0.1),
            Some(2.84),
        )
    }

    fn make_talent_area() -> StageLayout {
        StageLayout::new(
            StageZone::TalentArea,
            ZoneDimensions::new(8.0, 4.0, 6.0),
            None,
        )
    }

    #[test]
    fn test_stage_zone_is_display_surface() {
        assert!(StageZone::BackWall.is_display_surface());
        assert!(StageZone::Ceiling.is_display_surface());
        assert!(!StageZone::TalentArea.is_display_surface());
    }

    #[test]
    fn test_stage_zone_coverage_fractions_sum() {
        let zones = [
            StageZone::BackWall,
            StageZone::SideLeft,
            StageZone::SideRight,
            StageZone::Ceiling,
            StageZone::Floor,
            StageZone::TalentArea,
        ];
        let total: f32 = zones.iter().map(super::StageZone::coverage_fraction).sum();
        // Should sum to exactly 1.0
        assert!((total - 1.0).abs() < f32::EPSILON * 10.0);
    }

    #[test]
    fn test_stage_zone_display() {
        assert_eq!(StageZone::BackWall.to_string(), "BackWall");
        assert_eq!(StageZone::TalentArea.to_string(), "TalentArea");
        assert_eq!(StageZone::SideLeft.to_string(), "SideLeft");
    }

    #[test]
    fn test_zone_dimensions_surface_area() {
        let d = ZoneDimensions::new(10.0, 4.0, 0.5);
        assert!((d.surface_area_m2() - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_zone_state_display() {
        assert_eq!(ZoneState::Online.to_string(), "Online");
        assert_eq!(ZoneState::Standby.to_string(), "Standby");
        assert_eq!(ZoneState::Faulted.to_string(), "Faulted");
    }

    #[test]
    fn test_stage_layout_bring_online() {
        let mut layout = make_back_wall();
        assert_eq!(layout.state, ZoneState::Standby);
        layout.bring_online();
        assert_eq!(layout.state, ZoneState::Online);
        assert!(layout.is_active());
    }

    #[test]
    fn test_stage_layout_fault_prevents_online() {
        let mut layout = make_back_wall();
        layout.fault();
        layout.bring_online(); // should be blocked
        assert_eq!(layout.state, ZoneState::Faulted);
    }

    #[test]
    fn test_stage_layout_put_standby() {
        let mut layout = make_back_wall();
        layout.bring_online();
        layout.put_standby();
        assert_eq!(layout.state, ZoneState::Standby);
    }

    #[test]
    fn test_stage_layout_pixel_dimensions() {
        let layout = make_back_wall();
        // 12 000 mm / 2.84 mm = ~4225
        let pw = layout.pixel_width().expect("should succeed in test");
        assert!(pw > 4000 && pw < 5000);
        let ph = layout.pixel_height().expect("should succeed in test");
        assert!(ph > 1000 && ph < 2000);
    }

    #[test]
    fn test_stage_layout_no_pixel_pitch() {
        let layout = make_talent_area();
        assert!(layout.pixel_width().is_none());
        assert!(layout.pixel_height().is_none());
    }

    #[test]
    fn test_manager_bring_all_online() {
        let mut mgr = StageLayoutManager::new();
        mgr.add_zone(make_back_wall());
        mgr.add_zone(make_talent_area());
        mgr.bring_all_online();
        // Only display surfaces go online
        assert_eq!(mgr.online_count(), 1);
    }

    #[test]
    fn test_manager_total_online_area() {
        let mut mgr = StageLayoutManager::new();
        mgr.add_zone(make_back_wall());
        mgr.bring_all_online();
        let area = mgr.total_online_area_m2();
        // 12 × 4.5 = 54 m²
        assert!((area - 54.0).abs() < 0.001);
    }

    #[test]
    fn test_manager_faulted_count() {
        let mut mgr = StageLayoutManager::new();
        let mut zone = make_back_wall();
        zone.fault();
        mgr.add_zone(zone);
        assert_eq!(mgr.faulted_count(), 1);
    }

    #[test]
    fn test_manager_find_zone_mut() {
        let mut mgr = StageLayoutManager::new();
        mgr.add_zone(make_back_wall());
        let z = mgr
            .find_zone_mut(StageZone::BackWall)
            .expect("should succeed in test");
        z.bring_online();
        assert!(mgr
            .find_zone(StageZone::BackWall)
            .expect("should succeed in test")
            .is_active());
    }

    #[test]
    fn test_manager_zone_count() {
        let mut mgr = StageLayoutManager::new();
        assert_eq!(mgr.zone_count(), 0);
        mgr.add_zone(make_back_wall());
        assert_eq!(mgr.zone_count(), 1);
    }
}
