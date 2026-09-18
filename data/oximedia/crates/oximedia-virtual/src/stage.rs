//! Virtual stage / LED volume management.
//!
//! Data structures for describing physical LED stage volumes, their panel
//! configurations, and a library for managing multiple stage presets.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Panel facing
// ---------------------------------------------------------------------------

/// The facing direction of an LED panel within the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelFacing {
    /// Front wall panel (faces the subject / camera).
    Front,
    /// Left side wall panel.
    Left,
    /// Right side wall panel.
    Right,
    /// Ceiling panel (LED canopy / soffit).
    Ceiling,
    /// Floor panel (LED floor tile).
    Floor,
}

impl PanelFacing {
    /// Human-readable label for the panel facing.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            PanelFacing::Front => "Front",
            PanelFacing::Left => "Left",
            PanelFacing::Right => "Right",
            PanelFacing::Ceiling => "Ceiling",
            PanelFacing::Floor => "Floor",
        }
    }
}

// ---------------------------------------------------------------------------
// LED panel
// ---------------------------------------------------------------------------

/// A single LED panel tile within a virtual stage.
#[derive(Debug, Clone)]
pub struct LedPanel {
    /// Unique panel identifier within the stage.
    pub id: u32,
    /// Panel width in pixels.
    pub width_px: u32,
    /// Panel height in pixels.
    pub height_px: u32,
    /// X position in metres (stage coordinate system).
    pub x_pos: f64,
    /// Y position in metres.
    pub y_pos: f64,
    /// Z position in metres.
    pub z_pos: f64,
    /// The direction the panel faces.
    pub facing: PanelFacing,
}

impl LedPanel {
    /// Create a new LED panel.
    #[must_use]
    pub fn new(
        id: u32,
        width_px: u32,
        height_px: u32,
        x_pos: f64,
        y_pos: f64,
        z_pos: f64,
        facing: PanelFacing,
    ) -> Self {
        Self {
            id,
            width_px,
            height_px,
            x_pos,
            y_pos,
            z_pos,
            facing,
        }
    }

    /// Total pixel count of this panel.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width_px) * u64::from(self.height_px)
    }
}

// ---------------------------------------------------------------------------
// Virtual stage
// ---------------------------------------------------------------------------

/// A virtual production LED stage volume.
#[derive(Debug, Clone)]
pub struct VirtualStage {
    /// Unique stage identifier.
    pub id: u64,
    /// Human-readable stage name.
    pub name: String,
    /// Stage interior width in metres.
    pub width_m: f64,
    /// Stage interior height in metres.
    pub height_m: f64,
    /// Stage interior depth in metres.
    pub depth_m: f64,
    /// LED panels installed in this stage.
    pub led_panels: Vec<LedPanel>,
}

impl VirtualStage {
    /// Create a new, empty virtual stage.
    #[must_use]
    pub fn new(id: u64, name: &str, w: f64, h: f64, d: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            width_m: w,
            height_m: h,
            depth_m: d,
            led_panels: Vec::new(),
        }
    }

    /// Add an LED panel to the stage.
    pub fn add_panel(&mut self, panel: LedPanel) {
        self.led_panels.push(panel);
    }

    /// Total number of pixels across all panels.
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        self.led_panels.iter().map(LedPanel::pixel_count).sum()
    }

    /// Number of panels installed.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.led_panels.len()
    }

    /// Interior volume of the stage in cubic metres.
    #[must_use]
    pub fn stage_volume_m3(&self) -> f64 {
        self.width_m * self.height_m * self.depth_m
    }
}

// ---------------------------------------------------------------------------
// Stage library
// ---------------------------------------------------------------------------

/// A named collection of virtual stage definitions.
#[derive(Debug, Default)]
pub struct StageLibrary {
    /// All registered stages.
    pub stages: Vec<VirtualStage>,
}

impl StageLibrary {
    /// Create a new, empty stage library.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage to the library.
    pub fn add(&mut self, stage: VirtualStage) {
        self.stages.push(stage);
    }

    /// Find a stage by its exact name.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&VirtualStage> {
        self.stages.iter().find(|s| s.name == name)
    }

    /// Number of stages in the library.
    #[must_use]
    pub fn total_stages(&self) -> usize {
        self.stages.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_facing_label() {
        assert_eq!(PanelFacing::Front.label(), "Front");
        assert_eq!(PanelFacing::Left.label(), "Left");
        assert_eq!(PanelFacing::Right.label(), "Right");
        assert_eq!(PanelFacing::Ceiling.label(), "Ceiling");
        assert_eq!(PanelFacing::Floor.label(), "Floor");
    }

    #[test]
    fn test_led_panel_pixel_count() {
        let p = LedPanel::new(0, 1920, 1080, 0.0, 0.0, 0.0, PanelFacing::Front);
        assert_eq!(p.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_led_panel_zero_pixels() {
        let p = LedPanel::new(0, 0, 1080, 0.0, 0.0, 0.0, PanelFacing::Front);
        assert_eq!(p.pixel_count(), 0);
    }

    #[test]
    fn test_virtual_stage_new_empty() {
        let s = VirtualStage::new(1, "Stage A", 20.0, 8.0, 15.0);
        assert_eq!(s.id, 1);
        assert_eq!(s.name, "Stage A");
        assert_eq!(s.panel_count(), 0);
        assert_eq!(s.total_pixels(), 0);
    }

    #[test]
    fn test_virtual_stage_add_panel() {
        let mut s = VirtualStage::new(1, "Test", 20.0, 8.0, 15.0);
        s.add_panel(LedPanel::new(
            0,
            1920,
            1080,
            0.0,
            0.0,
            0.0,
            PanelFacing::Front,
        ));
        assert_eq!(s.panel_count(), 1);
    }

    #[test]
    fn test_virtual_stage_total_pixels_multi_panel() {
        let mut s = VirtualStage::new(1, "Test", 20.0, 8.0, 15.0);
        s.add_panel(LedPanel::new(
            0,
            1000,
            1000,
            0.0,
            0.0,
            0.0,
            PanelFacing::Front,
        ));
        s.add_panel(LedPanel::new(1, 500, 500, 5.0, 0.0, 0.0, PanelFacing::Left));
        // 1000*1000 + 500*500 = 1_250_000
        assert_eq!(s.total_pixels(), 1_250_000);
    }

    #[test]
    fn test_virtual_stage_volume() {
        let s = VirtualStage::new(1, "Test", 10.0, 5.0, 8.0);
        assert!((s.stage_volume_m3() - 400.0).abs() < 1e-10);
    }

    #[test]
    fn test_virtual_stage_volume_zero() {
        let s = VirtualStage::new(1, "Test", 0.0, 5.0, 8.0);
        assert_eq!(s.stage_volume_m3(), 0.0);
    }

    #[test]
    fn test_stage_library_new_empty() {
        let lib = StageLibrary::new();
        assert_eq!(lib.total_stages(), 0);
    }

    #[test]
    fn test_stage_library_add_and_find() {
        let mut lib = StageLibrary::new();
        lib.add(VirtualStage::new(1, "Hollywood Stage", 30.0, 10.0, 20.0));
        lib.add(VirtualStage::new(2, "Compact Stage", 10.0, 5.0, 8.0));
        assert_eq!(lib.total_stages(), 2);

        let found = lib.find_by_name("Compact Stage");
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").id, 2);
    }

    #[test]
    fn test_stage_library_find_missing() {
        let lib = StageLibrary::new();
        assert!(lib.find_by_name("Ghost Stage").is_none());
    }

    #[test]
    fn test_stage_library_total_stages() {
        let mut lib = StageLibrary::new();
        for i in 0..5 {
            lib.add(VirtualStage::new(i, &format!("Stage {i}"), 10.0, 5.0, 8.0));
        }
        assert_eq!(lib.total_stages(), 5);
    }
}
