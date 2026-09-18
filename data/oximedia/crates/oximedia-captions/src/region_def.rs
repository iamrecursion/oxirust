#![allow(dead_code)]
//! Caption region definitions and spatial layout management.
//!
//! This module provides comprehensive region definitions for caption positioning,
//! supporting TTML/IMSC regions, `WebVTT` cue positioning, CEA-608/708 safe areas,
//! and custom region layouts for broadcast and streaming workflows.

use std::collections::HashMap;
use std::fmt;

/// Represents a unit of measurement for region coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegionUnit {
    /// Percentage of the video frame (0.0 - 100.0).
    Percent,
    /// Pixel coordinates.
    Pixels,
    /// Character cells (used by CEA-608).
    Cells,
    /// Lines (used by `WebVTT`).
    Lines,
}

impl fmt::Display for RegionUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Percent => write!(f, "%"),
            Self::Pixels => write!(f, "px"),
            Self::Cells => write!(f, "c"),
            Self::Lines => write!(f, "ln"),
        }
    }
}

/// A coordinate value with an associated unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionCoord {
    /// The numeric value.
    pub value: f64,
    /// The unit of measurement.
    pub unit: RegionUnit,
}

impl RegionCoord {
    /// Create a new coordinate.
    #[must_use]
    pub fn new(value: f64, unit: RegionUnit) -> Self {
        Self { value, unit }
    }

    /// Create a percentage coordinate.
    #[must_use]
    pub fn percent(value: f64) -> Self {
        Self::new(value, RegionUnit::Percent)
    }

    /// Create a pixel coordinate.
    #[must_use]
    pub fn pixels(value: f64) -> Self {
        Self::new(value, RegionUnit::Pixels)
    }

    /// Convert this coordinate to a percentage given a reference dimension.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_percent(&self, reference_dimension: u32) -> f64 {
        match self.unit {
            RegionUnit::Percent => self.value,
            RegionUnit::Pixels => {
                if reference_dimension == 0 {
                    0.0
                } else {
                    (self.value / f64::from(reference_dimension)) * 100.0
                }
            }
            RegionUnit::Cells => {
                // CEA-608: 32 columns, approximate percentage
                (self.value / 32.0) * 100.0
            }
            RegionUnit::Lines => {
                // Approximate: 15 lines for standard video
                (self.value / 15.0) * 100.0
            }
        }
    }
}

/// Anchor point for region positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionAnchor {
    /// Top-left corner.
    TopLeft,
    /// Top-center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Middle-left.
    MiddleLeft,
    /// Center.
    Center,
    /// Middle-right.
    MiddleRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-center.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl RegionAnchor {
    /// Get the anchor offset as (`x_fraction`, `y_fraction`) in 0.0..1.0 range.
    #[must_use]
    pub fn offset(&self) -> (f64, f64) {
        match self {
            Self::TopLeft => (0.0, 0.0),
            Self::TopCenter => (0.5, 0.0),
            Self::TopRight => (1.0, 0.0),
            Self::MiddleLeft => (0.0, 0.5),
            Self::Center => (0.5, 0.5),
            Self::MiddleRight => (1.0, 0.5),
            Self::BottomLeft => (0.0, 1.0),
            Self::BottomCenter => (0.5, 1.0),
            Self::BottomRight => (1.0, 1.0),
        }
    }
}

/// Overflow behavior when text exceeds the region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowMode {
    /// Clip text at the region boundary.
    Hidden,
    /// Allow text to overflow the region.
    Visible,
    /// Scroll text within the region.
    Scroll,
    /// Dynamically resize the region to fit.
    Dynamic,
}

/// Writing mode for the text in the region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritingMode {
    /// Left-to-right, top-to-bottom.
    LeftToRight,
    /// Right-to-left, top-to-bottom.
    RightToLeft,
    /// Top-to-bottom, right-to-left (CJK vertical).
    TopToBottomRtl,
    /// Top-to-bottom, left-to-right.
    TopToBottomLtr,
}

/// A defined region for caption placement.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionRegion {
    /// Unique identifier for this region.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// X coordinate of the origin.
    pub origin_x: RegionCoord,
    /// Y coordinate of the origin.
    pub origin_y: RegionCoord,
    /// Width of the region.
    pub width: RegionCoord,
    /// Height of the region.
    pub height: RegionCoord,
    /// Anchor point.
    pub anchor: RegionAnchor,
    /// Overflow behavior.
    pub overflow: OverflowMode,
    /// Writing mode.
    pub writing_mode: WritingMode,
    /// Z-index for stacking order.
    pub z_index: i32,
    /// Background color as RGBA hex string.
    pub background_color: Option<String>,
    /// Background opacity (0.0 - 1.0).
    pub background_opacity: f64,
    /// Padding in pixels (top, right, bottom, left).
    pub padding: [f64; 4],
}

impl CaptionRegion {
    /// Create a new region with default settings.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            origin_x: RegionCoord::percent(10.0),
            origin_y: RegionCoord::percent(80.0),
            width: RegionCoord::percent(80.0),
            height: RegionCoord::percent(15.0),
            anchor: RegionAnchor::BottomCenter,
            overflow: OverflowMode::Hidden,
            writing_mode: WritingMode::LeftToRight,
            z_index: 0,
            background_color: None,
            background_opacity: 0.0,
            padding: [0.0; 4],
        }
    }

    /// Create a standard bottom region (most common caption placement).
    #[must_use]
    pub fn standard_bottom() -> Self {
        Self::new("bottom", "Standard Bottom Region")
    }

    /// Create a standard top region (pop-on captions).
    #[must_use]
    pub fn standard_top() -> Self {
        let mut region = Self::new("top", "Standard Top Region");
        region.origin_y = RegionCoord::percent(5.0);
        region
    }

    /// Compute the bounding box as (x, y, w, h) in percentages.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bounding_box_percent(
        &self,
        frame_width: u32,
        frame_height: u32,
    ) -> (f64, f64, f64, f64) {
        let x = self.origin_x.to_percent(frame_width);
        let y = self.origin_y.to_percent(frame_height);
        let w = self.width.to_percent(frame_width);
        let h = self.height.to_percent(frame_height);
        (x, y, w, h)
    }

    /// Check if a point (in percent) falls within this region.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn contains_point(&self, px: f64, py: f64, frame_width: u32, frame_height: u32) -> bool {
        let (x, y, w, h) = self.bounding_box_percent(frame_width, frame_height);
        px >= x && px <= x + w && py >= y && py <= y + h
    }

    /// Calculate the area of this region in percentage-squared.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn area_percent(&self, frame_width: u32, frame_height: u32) -> f64 {
        let (_, _, w, h) = self.bounding_box_percent(frame_width, frame_height);
        w * h
    }

    /// Check if this region overlaps with another.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn overlaps(&self, other: &Self, frame_width: u32, frame_height: u32) -> bool {
        let (ax, ay, aw, ah) = self.bounding_box_percent(frame_width, frame_height);
        let (bx, by, bw, bh) = other.bounding_box_percent(frame_width, frame_height);

        ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by
    }

    /// Set the region to use a semi-transparent black background.
    #[must_use]
    pub fn with_black_background(mut self, opacity: f64) -> Self {
        self.background_color = Some("#000000".to_string());
        self.background_opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set padding uniformly.
    #[must_use]
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.padding = [padding; 4];
        self
    }
}

/// A set of named regions forming a layout scheme.
#[derive(Debug, Clone)]
pub struct RegionLayout {
    /// Layout identifier.
    pub id: String,
    /// Layout description.
    pub description: String,
    /// Regions in this layout, keyed by region ID.
    pub regions: HashMap<String, CaptionRegion>,
}

impl RegionLayout {
    /// Create a new empty layout.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            regions: HashMap::new(),
        }
    }

    /// Add a region to this layout.
    pub fn add_region(&mut self, region: CaptionRegion) {
        self.regions.insert(region.id.clone(), region);
    }

    /// Get a region by its ID.
    #[must_use]
    pub fn get_region(&self, id: &str) -> Option<&CaptionRegion> {
        self.regions.get(id)
    }

    /// Remove a region by ID, returning it if it existed.
    #[must_use]
    pub fn remove_region(&self, id: &str) -> Option<CaptionRegion> {
        let mut regions = self.regions.clone();
        regions.remove(id)
    }

    /// Count the number of regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Check if any regions in this layout overlap with each other.
    #[must_use]
    pub fn has_overlapping_regions(&self, frame_width: u32, frame_height: u32) -> bool {
        let regions: Vec<&CaptionRegion> = self.regions.values().collect();
        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                if regions[i].overlaps(regions[j], frame_width, frame_height) {
                    return true;
                }
            }
        }
        false
    }

    /// Create a standard two-region layout (top + bottom).
    #[must_use]
    pub fn standard_two_region() -> Self {
        let mut layout = Self::new("standard-2", "Standard top and bottom regions");
        layout.add_region(CaptionRegion::standard_bottom());
        layout.add_region(CaptionRegion::standard_top());
        layout
    }

    /// Create a CEA-608 compatible layout with safe area constraints.
    #[must_use]
    pub fn cea608_safe_area() -> Self {
        let mut layout = Self::new("cea608", "CEA-608 safe area layout");
        let mut bottom = CaptionRegion::new("cea608-bottom", "CEA-608 Bottom");
        bottom.origin_x = RegionCoord::percent(10.0);
        bottom.origin_y = RegionCoord::percent(70.0);
        bottom.width = RegionCoord::percent(80.0);
        bottom.height = RegionCoord::percent(20.0);
        layout.add_region(bottom);

        let mut top = CaptionRegion::new("cea608-top", "CEA-608 Top");
        top.origin_x = RegionCoord::percent(10.0);
        top.origin_y = RegionCoord::percent(5.0);
        top.width = RegionCoord::percent(80.0);
        top.height = RegionCoord::percent(20.0);
        layout.add_region(top);

        layout
    }
}

/// Validates a region against broadcast safe area constraints.
#[derive(Debug, Clone)]
pub struct RegionValidator {
    /// Minimum margin from the left edge (percent).
    pub min_margin_left: f64,
    /// Minimum margin from the right edge (percent).
    pub min_margin_right: f64,
    /// Minimum margin from the top edge (percent).
    pub min_margin_top: f64,
    /// Minimum margin from the bottom edge (percent).
    pub min_margin_bottom: f64,
    /// Maximum allowed region area (percent of frame area).
    pub max_area_percent: f64,
}

impl Default for RegionValidator {
    fn default() -> Self {
        Self {
            min_margin_left: 5.0,
            min_margin_right: 5.0,
            min_margin_top: 5.0,
            min_margin_bottom: 5.0,
            max_area_percent: 50.0,
        }
    }
}

/// Result of a region validation check.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Whether the region passed all checks.
    pub passed: bool,
    /// List of issues found.
    pub issues: Vec<String>,
}

impl RegionValidator {
    /// Validate a caption region.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn validate(
        &self,
        region: &CaptionRegion,
        frame_width: u32,
        frame_height: u32,
    ) -> ValidationResult {
        let mut issues = Vec::new();
        let (x, y, w, h) = region.bounding_box_percent(frame_width, frame_height);

        if x < self.min_margin_left {
            issues.push(format!(
                "Region left edge ({x:.1}%) is within the left margin ({:.1}%)",
                self.min_margin_left
            ));
        }
        if x + w > 100.0 - self.min_margin_right {
            issues.push(format!(
                "Region right edge ({:.1}%) exceeds right margin ({:.1}%)",
                x + w,
                100.0 - self.min_margin_right
            ));
        }
        if y < self.min_margin_top {
            issues.push(format!(
                "Region top edge ({y:.1}%) is within top margin ({:.1}%)",
                self.min_margin_top
            ));
        }
        if y + h > 100.0 - self.min_margin_bottom {
            issues.push(format!(
                "Region bottom edge ({:.1}%) exceeds bottom margin ({:.1}%)",
                y + h,
                100.0 - self.min_margin_bottom
            ));
        }

        let area = w * h;
        let frame_area = 100.0 * 100.0;
        let area_pct = (area / frame_area) * 100.0;
        if area_pct > self.max_area_percent {
            issues.push(format!(
                "Region area ({area_pct:.1}%) exceeds maximum ({:.1}%)",
                self.max_area_percent
            ));
        }

        ValidationResult {
            passed: issues.is_empty(),
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_coord_percent() {
        let coord = RegionCoord::percent(50.0);
        assert_eq!(coord.unit, RegionUnit::Percent);
        let pct = coord.to_percent(1920);
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_coord_pixels_to_percent() {
        let coord = RegionCoord::pixels(960.0);
        let pct = coord.to_percent(1920);
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_coord_zero_reference() {
        let coord = RegionCoord::pixels(100.0);
        let pct = coord.to_percent(0);
        assert!((pct - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_anchor_offset() {
        let (x, y) = RegionAnchor::Center.offset();
        assert!((x - 0.5).abs() < f64::EPSILON);
        assert!((y - 0.5).abs() < f64::EPSILON);

        let (x, y) = RegionAnchor::TopLeft.offset();
        assert!((x - 0.0).abs() < f64::EPSILON);
        assert!((y - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_standard_bottom() {
        let region = CaptionRegion::standard_bottom();
        assert_eq!(region.id, "bottom");
        assert_eq!(region.anchor, RegionAnchor::BottomCenter);
    }

    #[test]
    fn test_region_standard_top() {
        let region = CaptionRegion::standard_top();
        assert_eq!(region.id, "top");
        assert!((region.origin_y.value - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_bounding_box() {
        let region = CaptionRegion::standard_bottom();
        let (x, y, w, h) = region.bounding_box_percent(1920, 1080);
        assert!((x - 10.0).abs() < f64::EPSILON);
        assert!((y - 80.0).abs() < f64::EPSILON);
        assert!((w - 80.0).abs() < f64::EPSILON);
        assert!((h - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_contains_point() {
        let region = CaptionRegion::standard_bottom();
        // Point inside the region
        assert!(region.contains_point(50.0, 85.0, 1920, 1080));
        // Point outside the region
        assert!(!region.contains_point(5.0, 50.0, 1920, 1080));
    }

    #[test]
    fn test_region_overlap() {
        let r1 = CaptionRegion::standard_bottom();
        let r2 = CaptionRegion::standard_top();
        // Standard top and bottom should not overlap
        assert!(!r1.overlaps(&r2, 1920, 1080));
    }

    #[test]
    fn test_region_with_black_background() {
        let region = CaptionRegion::standard_bottom().with_black_background(0.8);
        assert_eq!(region.background_color.as_deref(), Some("#000000"));
        assert!((region.background_opacity - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_layout_add_and_get() {
        let mut layout = RegionLayout::new("test", "Test layout");
        layout.add_region(CaptionRegion::standard_bottom());
        assert_eq!(layout.region_count(), 1);
        assert!(layout.get_region("bottom").is_some());
        assert!(layout.get_region("nonexistent").is_none());
    }

    #[test]
    fn test_region_layout_standard_two() {
        let layout = RegionLayout::standard_two_region();
        assert_eq!(layout.region_count(), 2);
        assert!(layout.get_region("bottom").is_some());
        assert!(layout.get_region("top").is_some());
    }

    #[test]
    fn test_region_validator_default_pass() {
        let validator = RegionValidator::default();
        let region = CaptionRegion::standard_bottom();
        let result = validator.validate(&region, 1920, 1080);
        assert!(
            result.passed,
            "Standard bottom region should pass default validation: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_region_validator_fails_on_margin_violation() {
        let validator = RegionValidator::default();
        let mut region = CaptionRegion::new("edge", "Edge region");
        region.origin_x = RegionCoord::percent(0.0); // Violates 5% left margin
        region.origin_y = RegionCoord::percent(50.0);
        region.width = RegionCoord::percent(20.0);
        region.height = RegionCoord::percent(10.0);
        let result = validator.validate(&region, 1920, 1080);
        assert!(!result.passed);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_region_unit_display() {
        assert_eq!(format!("{}", RegionUnit::Percent), "%");
        assert_eq!(format!("{}", RegionUnit::Pixels), "px");
        assert_eq!(format!("{}", RegionUnit::Cells), "c");
        assert_eq!(format!("{}", RegionUnit::Lines), "ln");
    }

    #[test]
    fn test_region_with_padding() {
        let region = CaptionRegion::standard_bottom().with_padding(8.0);
        assert!((region.padding[0] - 8.0).abs() < f64::EPSILON);
        assert!((region.padding[3] - 8.0).abs() < f64::EPSILON);
    }
}
