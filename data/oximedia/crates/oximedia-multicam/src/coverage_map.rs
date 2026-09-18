//! Camera coverage analysis for multicam production.

// ── CameraPosition ────────────────────────────────────────────────────────────

/// Position and field-of-view of a camera in 2-D space
#[derive(Debug, Clone)]
pub struct CameraPosition {
    /// Camera identifier
    pub id: u32,
    /// X coordinate (metres)
    pub x: f32,
    /// Y coordinate (metres)
    pub y: f32,
    /// Pan angle in degrees (0 = forward / +Y axis, positive = clockwise)
    pub angle_deg: f32,
    /// Horizontal field of view in degrees
    pub fov_deg: f32,
}

impl CameraPosition {
    /// Create a new camera position
    pub fn new(id: u32, x: f32, y: f32, angle_deg: f32, fov_deg: f32) -> Self {
        Self {
            id,
            x,
            y,
            angle_deg,
            fov_deg,
        }
    }

    /// Left boundary of the field of view (`angle_deg` − `fov_deg/2`)
    pub fn field_of_view_left(&self) -> f32 {
        self.angle_deg - self.fov_deg / 2.0
    }

    /// Right boundary of the field of view (`angle_deg` + `fov_deg/2`)
    pub fn field_of_view_right(&self) -> f32 {
        self.angle_deg + self.fov_deg / 2.0
    }

    /// Returns `true` when `target_angle` falls within the camera's FOV cone
    pub fn can_see(&self, target_angle: f32) -> bool {
        // Normalise target_angle relative to this camera's pointing direction
        let mut diff = (target_angle - self.angle_deg) % 360.0;
        if diff > 180.0 {
            diff -= 360.0;
        } else if diff < -180.0 {
            diff += 360.0;
        }
        diff.abs() <= self.fov_deg / 2.0
    }
}

// ── CoverageCell ──────────────────────────────────────────────────────────────

/// A single cell in the coverage grid
#[derive(Debug, Clone)]
pub struct CoverageCell {
    /// X coordinate of the cell centre
    pub x: f32,
    /// Y coordinate of the cell centre
    pub y: f32,
    /// IDs of cameras whose FOV covers this cell
    pub covering_cameras: Vec<u32>,
}

impl CoverageCell {
    /// Number of cameras that cover this cell
    pub fn coverage_count(&self) -> usize {
        self.covering_cameras.len()
    }

    /// Returns `true` if at least one camera covers this cell
    pub fn is_covered(&self) -> bool {
        !self.covering_cameras.is_empty()
    }
}

// ── CoverageAnalyzer ──────────────────────────────────────────────────────────

/// Analyses camera coverage over a square 2-D area
#[derive(Debug)]
pub struct CoverageAnalyzer {
    /// Registered cameras
    pub cameras: Vec<CameraPosition>,
    /// Side length of each grid cell (metres)
    pub grid_size: f32,
}

impl CoverageAnalyzer {
    /// Create a new analyser with the given grid cell size
    pub fn new(grid_size: f32) -> Self {
        Self {
            cameras: Vec::new(),
            grid_size,
        }
    }

    /// Register a camera
    pub fn add_camera(&mut self, cam: CameraPosition) {
        self.cameras.push(cam);
    }

    /// Compute which cameras cover each grid cell within a square `area × area` region.
    ///
    /// The grid origin is (0, 0).  Cell centres are offset by half a cell.
    /// The angle from (0, 0) to the cell centre is used as `target_angle`.
    pub fn compute_coverage_grid(&self, area: f32) -> Vec<CoverageCell> {
        let mut cells = Vec::new();
        let steps = (area / self.grid_size).ceil() as usize;
        for row in 0..steps {
            for col in 0..steps {
                let cx = col as f32 * self.grid_size + self.grid_size / 2.0;
                let cy = row as f32 * self.grid_size + self.grid_size / 2.0;
                // Angle from the origin to this cell centre (degrees)
                let target_angle = cy.atan2(cx).to_degrees();
                let covering_cameras: Vec<u32> = self
                    .cameras
                    .iter()
                    .filter(|cam| cam.can_see(target_angle))
                    .map(|cam| cam.id)
                    .collect();
                cells.push(CoverageCell {
                    x: cx,
                    y: cy,
                    covering_cameras,
                });
            }
        }
        cells
    }

    /// (x, y) coordinates of grid cells that have **no** camera coverage
    pub fn uncovered_areas(&self, area: f32) -> Vec<(f32, f32)> {
        self.compute_coverage_grid(area)
            .into_iter()
            .filter(|cell| !cell.is_covered())
            .map(|cell| (cell.x, cell.y))
            .collect()
    }

    /// Fraction of cells (0.0 – 100.0) that have at least one camera covering them
    pub fn coverage_pct(&self, area: f32) -> f64 {
        let grid = self.compute_coverage_grid(area);
        if grid.is_empty() {
            return 100.0;
        }
        let covered = grid.iter().filter(|c| c.is_covered()).count();
        covered as f64 / grid.len() as f64 * 100.0
    }
}

// ── CoverageMap (incremental) ─────────────────────────────────────────────────

use std::collections::HashSet;

/// An incremental camera-coverage map over a square 2-D area.
///
/// Unlike [`CoverageAnalyzer`] (which recomputes the entire grid on every
/// call), `CoverageMap` tracks which camera IDs have changed since the last
/// build via a *dirty set* and only recomputes the cells that overlap with
/// those cameras during [`update_incremental`].
///
/// # Typical workflow
///
/// 1. Create with [`CoverageMap::new`] supplying all cameras and the desired
///    grid parameters.
/// 2. Call [`full_rebuild`] once to populate the initial grid.
/// 3. When one or more cameras move / change FOV, call [`mark_dirty`] for each
///    affected camera and then [`update_incremental`].
///
/// [`update_incremental`]: CoverageMap::update_incremental
/// [`full_rebuild`]: CoverageMap::full_rebuild
/// [`mark_dirty`]: CoverageMap::mark_dirty
#[derive(Debug)]
pub struct CoverageMap {
    /// Side length of each grid cell (metres).
    pub grid_size: f32,
    /// Side length of the square area to cover (metres).
    pub area: f32,
    /// Computed coverage grid (populated by a build call).
    pub cells: Vec<CoverageCell>,
    /// Cameras registered with this map.
    pub cameras: Vec<CameraPosition>,
    /// Camera IDs that have changed since the last build.
    dirty_cameras: HashSet<u32>,
}

impl CoverageMap {
    /// Create a new, initially empty map with no cells computed.
    ///
    /// Call [`full_rebuild`] after construction to populate the grid.
    ///
    /// [`full_rebuild`]: CoverageMap::full_rebuild
    #[must_use]
    pub fn new(grid_size: f32, area: f32) -> Self {
        Self {
            grid_size,
            area,
            cells: Vec::new(),
            cameras: Vec::new(),
            dirty_cameras: HashSet::new(),
        }
    }

    /// Register a camera.
    pub fn add_camera(&mut self, cam: CameraPosition) {
        self.cameras.push(cam);
    }

    /// Mark a camera ID as having changed since the last build.
    ///
    /// The camera's cells will be recomputed on the next call to
    /// `update_incremental`.
    pub fn mark_dirty(&mut self, camera_id: u32) {
        self.dirty_cameras.insert(camera_id);
    }

    /// Recompute **only** the grid cells that may be affected by cameras in the
    /// dirty set, then clear the dirty set.
    ///
    /// Cells not overlapping any dirty camera retain their previous values.
    /// If the dirty set is empty this is a true no-op (no cell is touched).
    ///
    /// Note: "overlapping" is defined conservatively — any cell whose
    /// `covering_cameras` list contained a dirty camera ID, **or** that now
    /// falls within the FOV of a dirty camera, is recomputed.
    pub fn update_incremental(&mut self, cameras: &[CameraPosition]) {
        if self.dirty_cameras.is_empty() {
            return;
        }

        // Refresh our camera list from the provided slice.
        self.cameras = cameras.to_vec();

        let dirty = &self.dirty_cameras;
        let w = self.grid_size;
        let steps = (self.area / w).ceil() as usize;

        for cell in &mut self.cells {
            // Recompute this cell only if it was previously covered by a dirty
            // camera OR it might now be covered by one.
            let touches_dirty = cell.covering_cameras.iter().any(|id| dirty.contains(id))
                || self.cameras.iter().any(|cam| {
                    if !dirty.contains(&cam.id) {
                        return false;
                    }
                    let target_angle = cell.y.atan2(cell.x).to_degrees();
                    cam.can_see(target_angle)
                });

            if !touches_dirty {
                continue;
            }

            let target_angle = cell.y.atan2(cell.x).to_degrees();
            cell.covering_cameras = self
                .cameras
                .iter()
                .filter(|cam| cam.can_see(target_angle))
                .map(|cam| cam.id)
                .collect();
        }

        // If the grid is empty (first call without a prior full_rebuild) fall
        // back to generating it from scratch.
        if self.cells.is_empty() {
            for row in 0..steps {
                for col in 0..steps {
                    let cx = col as f32 * w + w / 2.0;
                    let cy = row as f32 * w + w / 2.0;
                    let target_angle = cy.atan2(cx).to_degrees();
                    let covering_cameras: Vec<u32> = self
                        .cameras
                        .iter()
                        .filter(|cam| cam.can_see(target_angle))
                        .map(|cam| cam.id)
                        .collect();
                    self.cells.push(CoverageCell {
                        x: cx,
                        y: cy,
                        covering_cameras,
                    });
                }
            }
        }

        self.dirty_cameras.clear();
    }

    /// Recompute the **entire** coverage grid from scratch.
    ///
    /// This is the correct starting point before the first incremental update
    /// and is also useful when many cameras change simultaneously.
    pub fn full_rebuild(&mut self, cameras: &[CameraPosition]) {
        self.cameras = cameras.to_vec();
        self.cells.clear();
        let w = self.grid_size;
        let steps = (self.area / w).ceil() as usize;
        for row in 0..steps {
            for col in 0..steps {
                let cx = col as f32 * w + w / 2.0;
                let cy = row as f32 * w + w / 2.0;
                let target_angle = cy.atan2(cx).to_degrees();
                let covering_cameras: Vec<u32> = self
                    .cameras
                    .iter()
                    .filter(|cam| cam.can_see(target_angle))
                    .map(|cam| cam.id)
                    .collect();
                self.cells.push(CoverageCell {
                    x: cx,
                    y: cy,
                    covering_cameras,
                });
            }
        }
        self.dirty_cameras.clear();
    }

    /// Return the number of cameras currently in the dirty set.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.dirty_cameras.len()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn wide_cam(id: u32) -> CameraPosition {
        // Points at 45° with a 180° FOV – covers a full half-plane
        CameraPosition::new(id, 0.0, 0.0, 45.0, 180.0)
    }

    // ── CameraPosition ───────────────────────────────────────────────────────

    #[test]
    fn test_field_of_view_boundaries() {
        let cam = CameraPosition::new(1, 0.0, 0.0, 90.0, 60.0);
        assert!((cam.field_of_view_left() - 60.0).abs() < f32::EPSILON);
        assert!((cam.field_of_view_right() - 120.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_can_see_target_within_fov() {
        let cam = CameraPosition::new(1, 0.0, 0.0, 0.0, 90.0);
        // Target exactly on the pointing direction
        assert!(cam.can_see(0.0));
        // Target 44° to the side – still inside 45° half-angle
        assert!(cam.can_see(44.0));
    }

    #[test]
    fn test_cannot_see_target_outside_fov() {
        let cam = CameraPosition::new(1, 0.0, 0.0, 0.0, 90.0);
        // 46° is outside the 45° half-angle
        assert!(!cam.can_see(46.0));
    }

    #[test]
    fn test_can_see_negative_angle() {
        let cam = CameraPosition::new(1, 0.0, 0.0, 0.0, 90.0);
        assert!(cam.can_see(-44.0));
        assert!(!cam.can_see(-46.0));
    }

    #[test]
    fn test_camera_position_fields_stored() {
        let cam = CameraPosition::new(7, 1.5, -2.0, 30.0, 60.0);
        assert_eq!(cam.id, 7);
        assert!((cam.x - 1.5).abs() < f32::EPSILON);
        assert!((cam.y - -2.0).abs() < f32::EPSILON);
        assert!((cam.angle_deg - 30.0).abs() < f32::EPSILON);
        assert!((cam.fov_deg - 60.0).abs() < f32::EPSILON);
    }

    // ── CoverageCell ─────────────────────────────────────────────────────────

    #[test]
    fn test_coverage_cell_is_covered_with_cameras() {
        let cell = CoverageCell {
            x: 1.0,
            y: 1.0,
            covering_cameras: vec![1, 2],
        };
        assert!(cell.is_covered());
        assert_eq!(cell.coverage_count(), 2);
    }

    #[test]
    fn test_coverage_cell_not_covered_when_empty() {
        let cell = CoverageCell {
            x: 1.0,
            y: 1.0,
            covering_cameras: vec![],
        };
        assert!(!cell.is_covered());
        assert_eq!(cell.coverage_count(), 0);
    }

    // ── CoverageAnalyzer ─────────────────────────────────────────────────────

    #[test]
    fn test_analyzer_new_no_cameras() {
        let az = CoverageAnalyzer::new(1.0);
        assert!(az.cameras.is_empty());
    }

    #[test]
    fn test_analyzer_add_camera() {
        let mut az = CoverageAnalyzer::new(1.0);
        az.add_camera(wide_cam(1));
        assert_eq!(az.cameras.len(), 1);
    }

    #[test]
    fn test_compute_coverage_grid_has_correct_cell_count() {
        let az = CoverageAnalyzer::new(1.0);
        let grid = az.compute_coverage_grid(3.0);
        // 3×3 grid = 9 cells
        assert_eq!(grid.len(), 9);
    }

    #[test]
    fn test_coverage_pct_no_cameras_zero() {
        let az = CoverageAnalyzer::new(1.0);
        let pct = az.coverage_pct(4.0);
        // No cameras → no coverage
        assert!((pct - 0.0).abs() < f64::EPSILON || pct == 0.0);
    }

    #[test]
    fn test_coverage_pct_wide_camera_covers_some_area() {
        let mut az = CoverageAnalyzer::new(1.0);
        az.add_camera(wide_cam(1));
        let pct = az.coverage_pct(4.0);
        assert!(pct > 0.0);
    }

    #[test]
    fn test_uncovered_areas_no_cameras_returns_all_cells() {
        let az = CoverageAnalyzer::new(1.0);
        let uncovered = az.uncovered_areas(2.0);
        // No cameras → all 4 cells are uncovered
        assert_eq!(uncovered.len(), 4);
    }

    // ── CoverageMap (incremental) tests ──────────────────────────────────────

    /// After a single-angle change, incremental update must produce the same
    /// cell coverage vectors as a full rebuild.
    #[test]
    fn test_incremental_coverage_matches_full_rebuild() {
        // Camera 0 points straight (0°, 180° FOV) — covers everything in
        // front.  Camera 1 points at 90° with a narrow FOV.
        let cam0 = CameraPosition::new(0, 0.0, 0.0, 0.0, 180.0);
        let cam1_initial = CameraPosition::new(1, 0.0, 0.0, 90.0, 30.0);

        let cameras = vec![cam0.clone(), cam1_initial.clone()];

        // Build baseline with full rebuild.
        let mut base_map = CoverageMap::new(1.0, 4.0);
        base_map.full_rebuild(&cameras);

        // Now change camera 1.
        let cam1_updated = CameraPosition::new(1, 0.0, 0.0, 45.0, 60.0);
        let updated_cameras = vec![cam0.clone(), cam1_updated.clone()];

        // Reference: full rebuild with updated cameras.
        let mut ref_map = CoverageMap::new(1.0, 4.0);
        ref_map.full_rebuild(&updated_cameras);

        // Incremental: mark camera 1 dirty and update.
        base_map.mark_dirty(1);
        base_map.update_incremental(&updated_cameras);

        // Both maps must have the same number of cells.
        assert_eq!(
            base_map.cells.len(),
            ref_map.cells.len(),
            "cell counts must match"
        );

        // Every cell must have the same covering_cameras set (order may differ).
        for (idx, (inc_cell, ref_cell)) in
            base_map.cells.iter().zip(ref_map.cells.iter()).enumerate()
        {
            let mut inc_sorted = inc_cell.covering_cameras.clone();
            let mut ref_sorted = ref_cell.covering_cameras.clone();
            inc_sorted.sort_unstable();
            ref_sorted.sort_unstable();
            assert_eq!(
                inc_sorted, ref_sorted,
                "cell {idx} mismatch: incremental {:?} vs full {:?}",
                inc_sorted, ref_sorted
            );
        }
    }

    /// When the dirty set is empty, `update_incremental` must be a true no-op:
    /// cell data must not change.
    #[test]
    fn test_no_dirty_no_recompute() {
        let cam = CameraPosition::new(0, 0.0, 0.0, 45.0, 180.0);
        let cameras = vec![cam.clone()];

        let mut map = CoverageMap::new(1.0, 3.0);
        map.full_rebuild(&cameras);

        // Snapshot the covering_cameras for each cell.
        let snapshot: Vec<Vec<u32>> = map
            .cells
            .iter()
            .map(|c| c.covering_cameras.clone())
            .collect();

        // Calling update_incremental with empty dirty set — no change.
        assert_eq!(map.dirty_count(), 0);
        map.update_incremental(&cameras);

        // Cells must be identical.
        for (idx, (cell, snap)) in map.cells.iter().zip(snapshot.iter()).enumerate() {
            assert_eq!(
                cell.covering_cameras, *snap,
                "cell {idx} changed even though dirty set was empty"
            );
        }
    }

    /// Verifying that `mark_dirty` increments the dirty set and that
    /// `update_incremental` clears it.
    #[test]
    fn test_dirty_set_cleared_after_update() {
        let cam = CameraPosition::new(5, 0.0, 0.0, 0.0, 90.0);
        let cameras = vec![cam];

        let mut map = CoverageMap::new(1.0, 2.0);
        map.full_rebuild(&cameras);

        map.mark_dirty(5);
        assert_eq!(map.dirty_count(), 1);
        map.update_incremental(&cameras);
        assert_eq!(
            map.dirty_count(),
            0,
            "dirty set should be empty after update"
        );
    }
}
