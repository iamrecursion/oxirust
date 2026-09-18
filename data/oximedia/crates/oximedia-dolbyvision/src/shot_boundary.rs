//! Dolby Vision shot boundary detection and classification.
//!
//! In Dolby Vision workflows, each scene change or shot cut requires its own
//! RPU trim metadata. This module provides types for detecting, classifying,
//! and collecting shot boundaries so that downstream tools can apply per-shot
//! processing.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// DvShotType
// ---------------------------------------------------------------------------

/// Classifies the type of visual transition at a shot boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DvShotType {
    /// Hard cut - instantaneous transition between shots.
    HardCut,
    /// Dissolve - gradual cross-fade between two shots.
    Dissolve,
    /// Fade in from black.
    FadeIn,
    /// Fade out to black.
    FadeOut,
    /// Wipe transition.
    Wipe,
    /// Flash or white-out frame.
    Flash,
    /// Scene change detected but transition type is unknown.
    Unknown,
}

impl DvShotType {
    /// Returns `true` if this is an instantaneous (non-gradual) transition.
    #[must_use]
    pub const fn is_instantaneous(self) -> bool {
        matches!(self, Self::HardCut | Self::Flash)
    }

    /// Returns `true` if this is a gradual transition spanning multiple frames.
    #[must_use]
    pub const fn is_gradual(self) -> bool {
        matches!(
            self,
            Self::Dissolve | Self::FadeIn | Self::FadeOut | Self::Wipe
        )
    }

    /// Short label for logging / reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HardCut => "hard-cut",
            Self::Dissolve => "dissolve",
            Self::FadeIn => "fade-in",
            Self::FadeOut => "fade-out",
            Self::Wipe => "wipe",
            Self::Flash => "flash",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for DvShotType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// DvShotBoundary
// ---------------------------------------------------------------------------

/// A single detected shot boundary with frame position and confidence.
#[derive(Debug, Clone)]
pub struct DvShotBoundary {
    /// Frame index where the boundary occurs (0-based).
    pub frame_index: u64,
    /// Type of shot transition at this boundary.
    pub shot_type: DvShotType,
    /// Detection confidence in the range `[0.0, 1.0]`.
    pub confidence: f64,
    /// For gradual transitions, the number of frames the transition spans.
    /// For instantaneous transitions this is 1.
    pub transition_length: u32,
    /// Optional label or note attached to this boundary.
    pub label: Option<String>,
}

impl DvShotBoundary {
    /// Create a new shot boundary.
    #[must_use]
    pub fn new(frame_index: u64, shot_type: DvShotType, confidence: f64) -> Self {
        let transition_length = if shot_type.is_instantaneous() { 1 } else { 0 };
        Self {
            frame_index,
            shot_type,
            confidence: confidence.clamp(0.0, 1.0),
            transition_length,
            label: None,
        }
    }

    /// Create a gradual transition boundary with explicit transition length.
    #[must_use]
    pub fn gradual(
        frame_index: u64,
        shot_type: DvShotType,
        confidence: f64,
        transition_length: u32,
    ) -> Self {
        Self {
            frame_index,
            shot_type,
            confidence: confidence.clamp(0.0, 1.0),
            transition_length: transition_length.max(1),
            label: None,
        }
    }

    /// Attach a label to this boundary.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns `true` if the confidence exceeds the given threshold.
    #[must_use]
    pub fn above_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Returns the last frame involved in this transition (inclusive).
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.frame_index + u64::from(self.transition_length.saturating_sub(1))
    }

    /// Returns `true` if the given frame falls within the transition range.
    #[must_use]
    pub fn overlaps_frame(&self, frame: u64) -> bool {
        frame >= self.frame_index && frame <= self.end_frame()
    }
}

impl fmt::Display for DvShotBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "boundary@{} type={} confidence={:.2}",
            self.frame_index, self.shot_type, self.confidence
        )
    }
}

// ---------------------------------------------------------------------------
// ShotBoundaryList
// ---------------------------------------------------------------------------

/// An ordered collection of [`DvShotBoundary`] entries for a video stream.
#[derive(Debug, Clone, Default)]
pub struct ShotBoundaryList {
    boundaries: Vec<DvShotBoundary>,
}

impl ShotBoundaryList {
    /// Create an empty boundary list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a boundary. The caller is responsible for maintaining order.
    pub fn push(&mut self, boundary: DvShotBoundary) {
        self.boundaries.push(boundary);
    }

    /// Number of boundaries in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.boundaries.len()
    }

    /// Returns `true` if the list contains no boundaries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }

    /// Sort boundaries by frame index.
    pub fn sort_by_frame(&mut self) {
        self.boundaries.sort_by_key(|b| b.frame_index);
    }

    /// Return all boundaries whose confidence is at or above `threshold`.
    #[must_use]
    pub fn above_threshold(&self, threshold: f64) -> Vec<&DvShotBoundary> {
        self.boundaries
            .iter()
            .filter(|b| b.above_threshold(threshold))
            .collect()
    }

    /// Return all boundaries of a specific shot type.
    #[must_use]
    pub fn by_type(&self, shot_type: DvShotType) -> Vec<&DvShotBoundary> {
        self.boundaries
            .iter()
            .filter(|b| b.shot_type == shot_type)
            .collect()
    }

    /// Return boundaries within the frame range `[from, to]` (inclusive).
    #[must_use]
    pub fn in_range(&self, from: u64, to: u64) -> Vec<&DvShotBoundary> {
        self.boundaries
            .iter()
            .filter(|b| b.frame_index >= from && b.frame_index <= to)
            .collect()
    }

    /// Iterator over all boundaries.
    pub fn iter(&self) -> impl Iterator<Item = &DvShotBoundary> {
        self.boundaries.iter()
    }

    /// Average confidence across all boundaries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_confidence(&self) -> f64 {
        if self.boundaries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.boundaries.iter().map(|b| b.confidence).sum();
        sum / self.boundaries.len() as f64
    }

    /// Count of boundaries grouped by shot type.
    #[must_use]
    pub fn type_counts(&self) -> std::collections::HashMap<DvShotType, usize> {
        let mut map = std::collections::HashMap::new();
        for b in &self.boundaries {
            *map.entry(b.shot_type).or_insert(0) += 1;
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary(frame: u64, st: DvShotType, conf: f64) -> DvShotBoundary {
        DvShotBoundary::new(frame, st, conf)
    }

    #[test]
    fn test_shot_type_is_instantaneous() {
        assert!(DvShotType::HardCut.is_instantaneous());
        assert!(DvShotType::Flash.is_instantaneous());
        assert!(!DvShotType::Dissolve.is_instantaneous());
        assert!(!DvShotType::FadeIn.is_instantaneous());
    }

    #[test]
    fn test_shot_type_is_gradual() {
        assert!(DvShotType::Dissolve.is_gradual());
        assert!(DvShotType::FadeIn.is_gradual());
        assert!(DvShotType::FadeOut.is_gradual());
        assert!(DvShotType::Wipe.is_gradual());
        assert!(!DvShotType::HardCut.is_gradual());
    }

    #[test]
    fn test_shot_type_label() {
        assert_eq!(DvShotType::HardCut.label(), "hard-cut");
        assert_eq!(DvShotType::Dissolve.label(), "dissolve");
        assert_eq!(DvShotType::Unknown.label(), "unknown");
    }

    #[test]
    fn test_shot_type_display() {
        let s = format!("{}", DvShotType::Flash);
        assert_eq!(s, "flash");
    }

    #[test]
    fn test_boundary_new_clamps_confidence() {
        let b = DvShotBoundary::new(10, DvShotType::HardCut, 1.5);
        assert!((b.confidence - 1.0).abs() < f64::EPSILON);
        let b2 = DvShotBoundary::new(10, DvShotType::HardCut, -0.3);
        assert!(b2.confidence.abs() < f64::EPSILON);
    }

    #[test]
    fn test_boundary_instantaneous_transition_length() {
        let b = DvShotBoundary::new(5, DvShotType::HardCut, 0.9);
        assert_eq!(b.transition_length, 1);
    }

    #[test]
    fn test_boundary_gradual() {
        let b = DvShotBoundary::gradual(100, DvShotType::Dissolve, 0.8, 15);
        assert_eq!(b.transition_length, 15);
        assert_eq!(b.end_frame(), 114);
    }

    #[test]
    fn test_boundary_with_label() {
        let b = DvShotBoundary::new(0, DvShotType::FadeIn, 0.95).with_label("opening fade");
        assert_eq!(b.label.as_deref(), Some("opening fade"));
    }

    #[test]
    fn test_boundary_above_threshold() {
        let b = boundary(0, DvShotType::HardCut, 0.7);
        assert!(b.above_threshold(0.5));
        assert!(!b.above_threshold(0.8));
    }

    #[test]
    fn test_boundary_overlaps_frame() {
        let b = DvShotBoundary::gradual(10, DvShotType::Dissolve, 0.9, 5);
        assert!(b.overlaps_frame(10));
        assert!(b.overlaps_frame(14));
        assert!(!b.overlaps_frame(9));
        assert!(!b.overlaps_frame(15));
    }

    #[test]
    fn test_boundary_display() {
        let b = boundary(42, DvShotType::HardCut, 0.85);
        let s = format!("{b}");
        assert!(s.contains("42"));
        assert!(s.contains("hard-cut"));
    }

    #[test]
    fn test_list_push_and_len() {
        let mut list = ShotBoundaryList::new();
        assert!(list.is_empty());
        list.push(boundary(0, DvShotType::HardCut, 0.9));
        list.push(boundary(100, DvShotType::Dissolve, 0.7));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_list_sort_by_frame() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(200, DvShotType::HardCut, 0.9));
        list.push(boundary(50, DvShotType::FadeIn, 0.8));
        list.push(boundary(100, DvShotType::Flash, 0.7));
        list.sort_by_frame();
        let frames: Vec<u64> = list.iter().map(|b| b.frame_index).collect();
        assert_eq!(frames, vec![50, 100, 200]);
    }

    #[test]
    fn test_list_above_threshold() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(0, DvShotType::HardCut, 0.9));
        list.push(boundary(10, DvShotType::Dissolve, 0.3));
        list.push(boundary(20, DvShotType::FadeOut, 0.6));
        let high = list.above_threshold(0.5);
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_list_by_type() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(0, DvShotType::HardCut, 0.9));
        list.push(boundary(100, DvShotType::Dissolve, 0.7));
        list.push(boundary(200, DvShotType::HardCut, 0.8));
        let cuts = list.by_type(DvShotType::HardCut);
        assert_eq!(cuts.len(), 2);
    }

    #[test]
    fn test_list_in_range() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(10, DvShotType::HardCut, 0.9));
        list.push(boundary(50, DvShotType::Flash, 0.8));
        list.push(boundary(200, DvShotType::FadeIn, 0.7));
        let range = list.in_range(0, 100);
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_list_average_confidence() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(0, DvShotType::HardCut, 0.8));
        list.push(boundary(10, DvShotType::HardCut, 0.6));
        let avg = list.average_confidence();
        assert!((avg - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_list_average_confidence_empty() {
        let list = ShotBoundaryList::new();
        assert!(list.average_confidence().abs() < f64::EPSILON);
    }

    #[test]
    fn test_list_type_counts() {
        let mut list = ShotBoundaryList::new();
        list.push(boundary(0, DvShotType::HardCut, 0.9));
        list.push(boundary(10, DvShotType::Dissolve, 0.8));
        list.push(boundary(20, DvShotType::HardCut, 0.7));
        let counts = list.type_counts();
        assert_eq!(counts[&DvShotType::HardCut], 2);
        assert_eq!(counts[&DvShotType::Dissolve], 1);
    }
}
