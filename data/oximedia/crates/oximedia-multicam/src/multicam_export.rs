#![allow(dead_code)]
//! Multi-camera export layout definitions.
//!
//! Provides `ExportLayout`, `ExportAngle`, and `MulticamExport` for
//! constructing and validating multi-angle output configurations.

/// Layout mode for a multi-camera export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportLayout {
    /// Only the primary angle is exported.
    SingleAngle,
    /// Two angles displayed side by side.
    SideBySide,
    /// All angles arranged in a regular grid.
    Grid,
    /// One angle fills the frame; others appear as insets.
    PiP,
}

impl ExportLayout {
    /// Returns how many tracks are typically expected for this layout.
    #[must_use]
    pub fn track_count(&self) -> usize {
        match self {
            Self::SingleAngle => 1,
            Self::SideBySide => 2,
            Self::Grid => 4,
            Self::PiP => 2,
        }
    }
}

/// A single angle included in a multi-camera export.
#[derive(Debug, Clone)]
pub struct ExportAngle {
    /// Angle index.
    pub index: usize,
    /// Whether this angle is the primary / featured angle.
    pub primary: bool,
    /// Optional label for the angle.
    pub label: Option<String>,
}

impl ExportAngle {
    /// Create a new `ExportAngle`.
    #[must_use]
    pub fn new(index: usize, primary: bool) -> Self {
        Self {
            index,
            primary,
            label: None,
        }
    }

    /// Returns `true` if this is the primary angle.
    #[must_use]
    pub fn is_primary(&self) -> bool {
        self.primary
    }

    /// Attach a label to this angle.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Builder and container for a complete multi-camera export.
#[derive(Debug, Clone)]
pub struct MulticamExport {
    layout: ExportLayout,
    angles: Vec<ExportAngle>,
}

impl MulticamExport {
    /// Create a new `MulticamExport` for the given layout.
    #[must_use]
    pub fn new(layout: ExportLayout) -> Self {
        Self {
            layout,
            angles: Vec::new(),
        }
    }

    /// Add an angle to the export.
    pub fn add_angle(&mut self, angle: ExportAngle) {
        self.angles.push(angle);
    }

    /// Return the configured layout.
    #[must_use]
    pub fn build_layout(&self) -> ExportLayout {
        self.layout
    }

    /// Validate that the export is consistent.
    ///
    /// Returns `Ok(())` when the angle count matches the layout expectation.
    /// Returns `Err` with a descriptive message otherwise.
    pub fn validate(&self) -> Result<(), String> {
        let expected = self.layout.track_count();
        let actual = self.angles.len();
        if actual < 1 {
            return Err("Export must contain at least one angle".to_string());
        }
        if self.layout != ExportLayout::SingleAngle && actual < expected {
            return Err(format!(
                "Layout {:?} expects {} angles, only {} provided",
                self.layout, expected, actual
            ));
        }
        let primary_count = self.angles.iter().filter(|a| a.primary).count();
        if primary_count != 1 {
            return Err(format!(
                "Exactly one primary angle required, found {primary_count}"
            ));
        }
        Ok(())
    }

    /// Return a reference to all configured angles.
    #[must_use]
    pub fn angles(&self) -> &[ExportAngle] {
        &self.angles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_angle_track_count() {
        assert_eq!(ExportLayout::SingleAngle.track_count(), 1);
    }

    #[test]
    fn test_side_by_side_track_count() {
        assert_eq!(ExportLayout::SideBySide.track_count(), 2);
    }

    #[test]
    fn test_grid_track_count() {
        assert_eq!(ExportLayout::Grid.track_count(), 4);
    }

    #[test]
    fn test_pip_track_count() {
        assert_eq!(ExportLayout::PiP.track_count(), 2);
    }

    #[test]
    fn test_export_angle_is_primary() {
        let a = ExportAngle::new(0, true);
        assert!(a.is_primary());
        let b = ExportAngle::new(1, false);
        assert!(!b.is_primary());
    }

    #[test]
    fn test_export_angle_label() {
        let a = ExportAngle::new(0, true).with_label("Camera A");
        assert_eq!(a.label.as_deref(), Some("Camera A"));
    }

    #[test]
    fn test_build_layout_returns_configured_layout() {
        let export = MulticamExport::new(ExportLayout::Grid);
        assert_eq!(export.build_layout(), ExportLayout::Grid);
    }

    #[test]
    fn test_validate_no_angles_fails() {
        let export = MulticamExport::new(ExportLayout::SingleAngle);
        assert!(export.validate().is_err());
    }

    #[test]
    fn test_validate_no_primary_fails() {
        let mut export = MulticamExport::new(ExportLayout::SingleAngle);
        export.add_angle(ExportAngle::new(0, false));
        assert!(export.validate().is_err());
    }

    #[test]
    fn test_validate_single_angle_ok() {
        let mut export = MulticamExport::new(ExportLayout::SingleAngle);
        export.add_angle(ExportAngle::new(0, true));
        assert!(export.validate().is_ok());
    }

    #[test]
    fn test_validate_side_by_side_missing_angle() {
        let mut export = MulticamExport::new(ExportLayout::SideBySide);
        export.add_angle(ExportAngle::new(0, true));
        assert!(export.validate().is_err());
    }

    #[test]
    fn test_validate_side_by_side_ok() {
        let mut export = MulticamExport::new(ExportLayout::SideBySide);
        export.add_angle(ExportAngle::new(0, true));
        export.add_angle(ExportAngle::new(1, false));
        assert!(export.validate().is_ok());
    }

    #[test]
    fn test_validate_multiple_primary_fails() {
        let mut export = MulticamExport::new(ExportLayout::SideBySide);
        export.add_angle(ExportAngle::new(0, true));
        export.add_angle(ExportAngle::new(1, true));
        assert!(export.validate().is_err());
    }

    #[test]
    fn test_angles_accessor() {
        let mut export = MulticamExport::new(ExportLayout::SingleAngle);
        export.add_angle(ExportAngle::new(0, true));
        assert_eq!(export.angles().len(), 1);
    }
}
