//! Shot coverage map utilities.
//!
//! Provides a grid-based map for tracking which regions of a scene have been
//! covered by shots, and utilities for detecting 180-degree axis violations.

#![allow(dead_code)]

/// Shot size classification, locally defined for use in coverage tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotSize {
    /// Extreme Wide Shot.
    ExtremeWide,
    /// Wide Shot.
    Wide,
    /// Medium Wide Shot.
    MediumWide,
    /// Medium Shot.
    Medium,
    /// Medium Close-Up.
    MediumClose,
    /// Close-Up.
    CloseUp,
    /// Extreme Close-Up.
    ExtremeCloseUp,
}

/// A single cell in the coverage grid.
#[derive(Debug, Clone)]
pub struct CoverageCell {
    /// Row index of this cell (0-based).
    pub row: u32,
    /// Column index of this cell (0-based).
    pub col: u32,
    /// Number of shots recorded for this cell.
    pub shot_count: u32,
    /// The most frequently used shot size in this cell, if any.
    pub dominant_size: Option<ShotSize>,
}

impl CoverageCell {
    /// Create a new, empty coverage cell.
    #[must_use]
    pub fn new(row: u32, col: u32) -> Self {
        Self {
            row,
            col,
            shot_count: 0,
            dominant_size: None,
        }
    }

    /// Return `true` when this cell has been covered by at least 2 shots.
    #[must_use]
    pub fn is_well_covered(&self) -> bool {
        self.shot_count >= 2
    }
}

/// A two-dimensional grid of [`CoverageCell`]s for tracking shot coverage.
#[derive(Debug, Clone)]
pub struct CoverageGrid {
    /// Row-major storage: `cells[row][col]`.
    pub cells: Vec<Vec<CoverageCell>>,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
}

impl CoverageGrid {
    /// Create a new grid with the given dimensions, all cells initialised empty.
    #[must_use]
    pub fn new(rows: u32, cols: u32) -> Self {
        let cells = (0..rows)
            .map(|r| (0..cols).map(|c| CoverageCell::new(r, c)).collect())
            .collect();
        Self { cells, rows, cols }
    }

    /// Record a shot of `size` for the cell at `(row, col)`.
    ///
    /// Out-of-bounds coordinates are silently ignored.
    pub fn record_shot(&mut self, row: u32, col: u32, size: ShotSize) {
        if row >= self.rows || col >= self.cols {
            return;
        }
        let cell = &mut self.cells[row as usize][col as usize];
        cell.shot_count += 1;
        // Keep the most recently recorded size as dominant (simple heuristic).
        cell.dominant_size = Some(size);
    }

    /// Return the fraction of cells that are well-covered (shot_count >= 2).
    ///
    /// Returns `0.0` if the grid has no cells.
    #[must_use]
    pub fn coverage_pct(&self) -> f64 {
        let total = (self.rows as usize) * (self.cols as usize);
        if total == 0 {
            return 0.0;
        }
        let covered = self
            .cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.is_well_covered())
            .count();
        covered as f64 / total as f64
    }

    /// Return the `(row, col)` indices of all cells that are not well-covered
    /// (i.e., `shot_count < 2`).
    #[must_use]
    pub fn poorly_covered(&self) -> Vec<(u32, u32)> {
        self.cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| !c.is_well_covered())
            .map(|c| (c.row, c.col))
            .collect()
    }
}

/// A directional line, typically an imaginary axis running through a scene.
#[derive(Debug, Clone, Copy)]
pub struct AxisLine {
    /// Angle of the line in degrees (0 = horizontal, 90 = vertical).
    pub angle_deg: f32,
    /// Start point in normalised [0, 1] frame coordinates.
    pub start: (f32, f32),
    /// End point in normalised [0, 1] frame coordinates.
    pub end: (f32, f32),
}

impl AxisLine {
    /// Create a new axis line.
    #[must_use]
    pub fn new(angle_deg: f32, start: (f32, f32), end: (f32, f32)) -> Self {
        Self {
            angle_deg,
            start,
            end,
        }
    }

    /// Return `true` if this line is approximately horizontal (within ±15°).
    #[must_use]
    pub fn is_horizontal(&self) -> bool {
        let a = self.angle_deg % 180.0;
        a.abs() <= 15.0 || a >= 165.0
    }
}

/// Detects violations of the 180-degree rule (crossing the action axis).
pub struct AxisViolationDetector;

impl AxisViolationDetector {
    /// Return `true` if the two consecutive shots have crossed the action axis.
    ///
    /// `shot_a_side` and `shot_b_side` represent which side of the axis each
    /// camera was placed on (`true` = left/front, `false` = right/back).
    /// A violation occurs when the camera side changes between shots.
    #[must_use]
    pub fn check(shot_a_side: bool, shot_b_side: bool) -> bool {
        shot_a_side != shot_b_side
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CoverageCell ---

    #[test]
    fn test_cell_not_well_covered_at_zero() {
        let cell = CoverageCell::new(0, 0);
        assert!(!cell.is_well_covered());
    }

    #[test]
    fn test_cell_not_well_covered_at_one() {
        let mut cell = CoverageCell::new(0, 0);
        cell.shot_count = 1;
        assert!(!cell.is_well_covered());
    }

    #[test]
    fn test_cell_well_covered_at_two() {
        let mut cell = CoverageCell::new(1, 2);
        cell.shot_count = 2;
        assert!(cell.is_well_covered());
    }

    #[test]
    fn test_cell_well_covered_above_two() {
        let mut cell = CoverageCell::new(0, 0);
        cell.shot_count = 10;
        assert!(cell.is_well_covered());
    }

    // --- CoverageGrid ---

    #[test]
    fn test_grid_dimensions() {
        let grid = CoverageGrid::new(3, 4);
        assert_eq!(grid.rows, 3);
        assert_eq!(grid.cols, 4);
        assert_eq!(grid.cells.len(), 3);
        assert_eq!(grid.cells[0].len(), 4);
    }

    #[test]
    fn test_grid_initial_coverage_zero() {
        let grid = CoverageGrid::new(2, 2);
        assert_eq!(grid.coverage_pct(), 0.0);
    }

    #[test]
    fn test_record_shot_increments_count() {
        let mut grid = CoverageGrid::new(3, 3);
        grid.record_shot(1, 1, ShotSize::Medium);
        grid.record_shot(1, 1, ShotSize::CloseUp);
        assert_eq!(grid.cells[1][1].shot_count, 2);
        assert!(grid.cells[1][1].is_well_covered());
    }

    #[test]
    fn test_record_shot_out_of_bounds_ignored() {
        let mut grid = CoverageGrid::new(2, 2);
        grid.record_shot(5, 5, ShotSize::Wide); // out of bounds
        assert_eq!(grid.coverage_pct(), 0.0);
    }

    #[test]
    fn test_coverage_pct_full() {
        let mut grid = CoverageGrid::new(2, 2);
        for r in 0..2u32 {
            for c in 0..2u32 {
                grid.record_shot(r, c, ShotSize::Wide);
                grid.record_shot(r, c, ShotSize::Wide);
            }
        }
        assert!((grid.coverage_pct() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coverage_pct_half() {
        let mut grid = CoverageGrid::new(1, 2);
        grid.record_shot(0, 0, ShotSize::Wide);
        grid.record_shot(0, 0, ShotSize::Wide);
        // cell (0,0) well covered, (0,1) not → 50 %
        assert!((grid.coverage_pct() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_poorly_covered_all() {
        let grid = CoverageGrid::new(2, 3);
        let poorly = grid.poorly_covered();
        assert_eq!(poorly.len(), 6); // all 6 cells are poorly covered
    }

    #[test]
    fn test_poorly_covered_none_after_coverage() {
        let mut grid = CoverageGrid::new(1, 1);
        grid.record_shot(0, 0, ShotSize::Medium);
        grid.record_shot(0, 0, ShotSize::Medium);
        assert!(grid.poorly_covered().is_empty());
    }

    // --- AxisLine ---

    #[test]
    fn test_is_horizontal_zero_degrees() {
        let line = AxisLine::new(0.0, (0.0, 0.5), (1.0, 0.5));
        assert!(line.is_horizontal());
    }

    #[test]
    fn test_is_not_horizontal_vertical() {
        let line = AxisLine::new(90.0, (0.5, 0.0), (0.5, 1.0));
        assert!(!line.is_horizontal());
    }

    // --- AxisViolationDetector ---

    #[test]
    fn test_no_violation_same_side() {
        assert!(!AxisViolationDetector::check(true, true));
        assert!(!AxisViolationDetector::check(false, false));
    }

    #[test]
    fn test_violation_different_sides() {
        assert!(AxisViolationDetector::check(true, false));
        assert!(AxisViolationDetector::check(false, true));
    }
}
