//! LED panel topology management for virtual production.
//!
//! Models the physical arrangement of LED panels in a virtual production volume,
//! including panel addressing, adjacency, and coordinate mapping.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// The position of a panel in the topology grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PanelPosition {
    /// Column index (0-based, left to right).
    pub col: u32,
    /// Row index (0-based, top to bottom).
    pub row: u32,
}

impl PanelPosition {
    /// Create a new panel position.
    #[must_use]
    pub const fn new(col: u32, row: u32) -> Self {
        Self { col, row }
    }
}

/// Orientation of a panel relative to its nominal orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanelOrientation {
    /// No rotation applied.
    #[default]
    Normal,
    /// Rotated 90 degrees clockwise.
    Rotated90,
    /// Rotated 180 degrees (upside-down).
    Rotated180,
    /// Rotated 270 degrees clockwise (or 90 counter-clockwise).
    Rotated270,
}

/// A single addressable panel in the topology.
#[derive(Debug, Clone)]
pub struct TopologyPanel {
    /// Unique panel identifier.
    pub panel_id: u32,
    /// Position in the topology grid.
    pub position: PanelPosition,
    /// Orientation of this panel.
    pub orientation: PanelOrientation,
    /// Width of the panel in pixels.
    pub width_px: u32,
    /// Height of the panel in pixels.
    pub height_px: u32,
    /// Whether this panel is currently active.
    pub active: bool,
}

impl TopologyPanel {
    /// Create a new panel with default orientation and active state.
    #[must_use]
    pub fn new(panel_id: u32, position: PanelPosition, width_px: u32, height_px: u32) -> Self {
        Self {
            panel_id,
            position,
            orientation: PanelOrientation::Normal,
            width_px,
            height_px,
            active: true,
        }
    }

    /// Set the panel orientation.
    #[must_use]
    pub fn with_orientation(mut self, orientation: PanelOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Mark the panel as inactive.
    #[must_use]
    pub fn inactive(mut self) -> Self {
        self.active = false;
        self
    }

    /// Pixel count for this panel, accounting for orientation.
    /// (Orientation does not change pixel count, only mapping.)
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width_px) * u64::from(self.height_px)
    }

    /// Whether this panel has portrait orientation (rotated 90 or 270).
    #[must_use]
    pub fn is_portrait(&self) -> bool {
        matches!(
            self.orientation,
            PanelOrientation::Rotated90 | PanelOrientation::Rotated270
        )
    }
}

/// The complete topology of an LED wall, describing panel placement.
#[derive(Debug, Default)]
pub struct PanelTopology {
    /// All panels indexed by their position.
    panels: HashMap<PanelPosition, TopologyPanel>,
    /// Width of each panel tile in pixels (assumed uniform).
    tile_width_px: u32,
    /// Height of each panel tile in pixels (assumed uniform).
    tile_height_px: u32,
}

impl PanelTopology {
    /// Create a new empty topology with given tile dimensions.
    #[must_use]
    pub fn new(tile_width_px: u32, tile_height_px: u32) -> Self {
        Self {
            panels: HashMap::new(),
            tile_width_px,
            tile_height_px,
        }
    }

    /// Build a simple rectangular grid topology.
    #[must_use]
    pub fn rectangular(cols: u32, rows: u32, tile_width_px: u32, tile_height_px: u32) -> Self {
        let mut topology = Self::new(tile_width_px, tile_height_px);
        let mut id = 0u32;
        for row in 0..rows {
            for col in 0..cols {
                let pos = PanelPosition::new(col, row);
                topology.add_panel(TopologyPanel::new(id, pos, tile_width_px, tile_height_px));
                id += 1;
            }
        }
        topology
    }

    /// Add a panel to the topology.
    ///
    /// Returns `false` if a panel already exists at that position.
    pub fn add_panel(&mut self, panel: TopologyPanel) -> bool {
        if self.panels.contains_key(&panel.position) {
            return false;
        }
        self.panels.insert(panel.position, panel);
        true
    }

    /// Get a panel by position.
    #[must_use]
    pub fn get(&self, position: &PanelPosition) -> Option<&TopologyPanel> {
        self.panels.get(position)
    }

    /// Total number of panels in the topology.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Total number of active panels.
    #[must_use]
    pub fn active_panel_count(&self) -> usize {
        self.panels.values().filter(|p| p.active).count()
    }

    /// Total pixel count across all active panels.
    #[must_use]
    pub fn total_pixel_count(&self) -> u64 {
        self.panels
            .values()
            .filter(|p| p.active)
            .map(TopologyPanel::pixel_count)
            .sum()
    }

    /// Maximum column index in the topology.
    #[must_use]
    pub fn max_col(&self) -> Option<u32> {
        self.panels.keys().map(|p| p.col).max()
    }

    /// Maximum row index in the topology.
    #[must_use]
    pub fn max_row(&self) -> Option<u32> {
        self.panels.keys().map(|p| p.row).max()
    }

    /// Width of the full wall in pixels (active area).
    #[must_use]
    pub fn wall_width_px(&self) -> u32 {
        match self.max_col() {
            Some(c) => (c + 1) * self.tile_width_px,
            None => 0,
        }
    }

    /// Height of the full wall in pixels (active area).
    #[must_use]
    pub fn wall_height_px(&self) -> u32 {
        match self.max_row() {
            Some(r) => (r + 1) * self.tile_height_px,
            None => 0,
        }
    }

    /// Tile dimensions.
    #[must_use]
    pub fn tile_dimensions(&self) -> (u32, u32) {
        (self.tile_width_px, self.tile_height_px)
    }

    /// Convert a global wall pixel coordinate to a (`panel_position`, `local_x`, `local_y`) triple.
    #[must_use]
    pub fn global_to_local(
        &self,
        global_x: u32,
        global_y: u32,
    ) -> Option<(PanelPosition, u32, u32)> {
        if self.tile_width_px == 0 || self.tile_height_px == 0 {
            return None;
        }
        let col = global_x / self.tile_width_px;
        let row = global_y / self.tile_height_px;
        let local_x = global_x % self.tile_width_px;
        let local_y = global_y % self.tile_height_px;
        let pos = PanelPosition::new(col, row);
        if self.panels.contains_key(&pos) {
            Some((pos, local_x, local_y))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_position_eq() {
        let p1 = PanelPosition::new(2, 3);
        let p2 = PanelPosition::new(2, 3);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_topology_panel_pixel_count() {
        let panel = TopologyPanel::new(0, PanelPosition::new(0, 0), 256, 128);
        assert_eq!(panel.pixel_count(), 256 * 128);
    }

    #[test]
    fn test_topology_panel_is_portrait() {
        let panel = TopologyPanel::new(0, PanelPosition::new(0, 0), 256, 128)
            .with_orientation(PanelOrientation::Rotated90);
        assert!(panel.is_portrait());

        let panel2 = TopologyPanel::new(1, PanelPosition::new(1, 0), 256, 128);
        assert!(!panel2.is_portrait());
    }

    #[test]
    fn test_topology_panel_inactive() {
        let panel = TopologyPanel::new(0, PanelPosition::new(0, 0), 256, 128).inactive();
        assert!(!panel.active);
    }

    #[test]
    fn test_topology_rectangular_panel_count() {
        let topo = PanelTopology::rectangular(4, 3, 256, 128);
        assert_eq!(topo.panel_count(), 12);
    }

    #[test]
    fn test_topology_rectangular_wall_size() {
        let topo = PanelTopology::rectangular(4, 3, 256, 128);
        assert_eq!(topo.wall_width_px(), 4 * 256);
        assert_eq!(topo.wall_height_px(), 3 * 128);
    }

    #[test]
    fn test_topology_active_panel_count() {
        let mut topo = PanelTopology::new(256, 128);
        topo.add_panel(TopologyPanel::new(0, PanelPosition::new(0, 0), 256, 128));
        topo.add_panel(TopologyPanel::new(1, PanelPosition::new(1, 0), 256, 128).inactive());
        assert_eq!(topo.active_panel_count(), 1);
    }

    #[test]
    fn test_topology_total_pixel_count() {
        let topo = PanelTopology::rectangular(2, 2, 100, 100);
        // 4 active panels × 100×100 = 40000
        assert_eq!(topo.total_pixel_count(), 40000);
    }

    #[test]
    fn test_topology_add_panel_duplicate_position() {
        let mut topo = PanelTopology::new(256, 128);
        let p1 = TopologyPanel::new(0, PanelPosition::new(0, 0), 256, 128);
        let p2 = TopologyPanel::new(1, PanelPosition::new(0, 0), 256, 128);
        assert!(topo.add_panel(p1));
        assert!(!topo.add_panel(p2)); // duplicate position
    }

    #[test]
    fn test_topology_get_existing() {
        let mut topo = PanelTopology::new(256, 128);
        let pos = PanelPosition::new(2, 1);
        topo.add_panel(TopologyPanel::new(5, pos, 256, 128));
        assert!(topo.get(&pos).is_some());
        assert_eq!(topo.get(&pos).expect("should succeed in test").panel_id, 5);
    }

    #[test]
    fn test_topology_get_nonexistent() {
        let topo = PanelTopology::new(256, 128);
        assert!(topo.get(&PanelPosition::new(99, 99)).is_none());
    }

    #[test]
    fn test_topology_global_to_local() {
        let topo = PanelTopology::rectangular(4, 3, 256, 128);
        // Pixel (300, 150) should land in col=1 (300/256=1), row=1 (150/128=1)
        let result = topo.global_to_local(300, 150);
        assert!(result.is_some());
        let (pos, lx, ly) = result.expect("should succeed in test");
        assert_eq!(pos, PanelPosition::new(1, 1));
        assert_eq!(lx, 300 % 256);
        assert_eq!(ly, 150 % 128);
    }

    #[test]
    fn test_topology_global_to_local_out_of_bounds() {
        let topo = PanelTopology::rectangular(2, 2, 256, 128);
        // Panel at col=5 does not exist
        let result = topo.global_to_local(5 * 256, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_topology_empty() {
        let topo = PanelTopology::new(256, 128);
        assert_eq!(topo.panel_count(), 0);
        assert_eq!(topo.wall_width_px(), 0);
        assert_eq!(topo.wall_height_px(), 0);
    }

    #[test]
    fn test_panel_orientation_default() {
        assert_eq!(PanelOrientation::default(), PanelOrientation::Normal);
    }

    #[test]
    fn test_topology_tile_dimensions() {
        let topo = PanelTopology::new(320, 180);
        assert_eq!(topo.tile_dimensions(), (320, 180));
    }
}
