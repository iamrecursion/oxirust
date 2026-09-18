//! Subclip management - create and organize subclips from parent clips.

pub mod inherit;

use std::collections::HashMap;

/// Unique identifier for a subclip.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubclipId(pub u64);

impl std::fmt::Display for SubclipId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "subclip:{}", self.0)
    }
}

/// A subclip references a range within a parent clip.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct Subclip {
    /// Unique subclip identifier.
    pub id: SubclipId,
    /// ID of the parent clip.
    pub parent_id: u64,
    /// In point (frame number within parent).
    pub in_point: u64,
    /// Out point (frame number within parent, exclusive).
    pub out_point: u64,
    /// Human-readable label.
    pub label: String,
    /// Display color `[R, G, B]`.
    pub color: [u8; 3],
}

impl Subclip {
    /// Create a new subclip.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(
        id: SubclipId,
        parent_id: u64,
        in_point: u64,
        out_point: u64,
        label: impl Into<String>,
        color: [u8; 3],
    ) -> Self {
        Self {
            id,
            parent_id,
            in_point,
            out_point,
            label: label.into(),
            color,
        }
    }

    /// Duration in frames (`out_point - in_point`).
    #[allow(dead_code)]
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.out_point.saturating_sub(self.in_point)
    }
}

/// Hierarchical subclip tree: maps parent clip IDs to their subclips.
#[allow(dead_code)]
pub struct SubclipTree {
    children: HashMap<u64, Vec<Subclip>>,
}

impl SubclipTree {
    /// Create a new empty tree.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
        }
    }

    /// Add a subclip under the given parent.
    #[allow(dead_code)]
    pub fn add(&mut self, parent: u64, subclip: Subclip) {
        self.children.entry(parent).or_default().push(subclip);
    }

    /// Get all subclips for a given parent.
    #[allow(dead_code)]
    #[must_use]
    pub fn children_of(&self, parent: u64) -> Vec<&Subclip> {
        self.children
            .get(&parent)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Flatten all subclips across all parents.
    #[allow(dead_code)]
    #[must_use]
    pub fn flatten(&self) -> Vec<&Subclip> {
        self.children.values().flatten().collect()
    }

    /// Total number of subclips.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.children.values().map(|v| v.len()).sum()
    }

    /// Returns true if the tree has no subclips.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.values().all(|v| v.is_empty())
    }
}

impl Default for SubclipTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates subclips against their parent constraints.
#[allow(dead_code)]
pub struct SubclipValidator;

impl SubclipValidator {
    /// Validate a subclip against its parent clip's total duration.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if the subclip is invalid.
    #[allow(dead_code)]
    pub fn validate(subclip: &Subclip, parent_duration: u64) -> Result<(), String> {
        if subclip.in_point >= subclip.out_point {
            return Err(format!(
                "in_point ({}) must be less than out_point ({})",
                subclip.in_point, subclip.out_point
            ));
        }
        if subclip.out_point > parent_duration {
            return Err(format!(
                "out_point ({}) exceeds parent duration ({})",
                subclip.out_point, parent_duration
            ));
        }
        if subclip.label.is_empty() {
            return Err("label must not be empty".to_string());
        }
        Ok(())
    }
}

/// Exports subclips to EDL format.
#[allow(dead_code)]
pub struct SubclipExporter;

impl SubclipExporter {
    /// Convert a subclip to an EDL event line.
    ///
    /// Format: `NNN  AX      V     C        HH:MM:SS:FF HH:MM:SS:FF HH:MM:SS:FF HH:MM:SS:FF`
    /// where the first timecode pair is source in/out and the second is record in/out.
    #[allow(dead_code)]
    #[must_use]
    pub fn to_edl_event(subclip: &Subclip, fps: f64) -> String {
        let in_tc = frames_to_timecode(subclip.in_point, fps);
        let out_tc = frames_to_timecode(subclip.out_point, fps);
        // Record in = same as source in; record out = same as source out (simplified)
        format!(
            "001  AX      V     C        {} {} {} {}",
            in_tc, out_tc, in_tc, out_tc
        )
    }
}

/// Convert a frame index to a `HH:MM:SS:FF` timecode string.
fn frames_to_timecode(frame: u64, fps: f64) -> String {
    let fps_int = fps.round() as u64;
    let fps_safe = fps_int.max(1);
    let total_seconds = frame / fps_safe;
    let ff = frame % fps_safe;
    let ss = total_seconds % 60;
    let mm = (total_seconds / 60) % 60;
    let hh = total_seconds / 3600;
    format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_subclip(id: u64, parent: u64, in_pt: u64, out_pt: u64, label: &str) -> Subclip {
        Subclip::new(SubclipId(id), parent, in_pt, out_pt, label, [255, 0, 0])
    }

    #[test]
    fn test_subclip_duration() {
        let sc = make_subclip(1, 10, 50, 150, "Scene A");
        assert_eq!(sc.duration(), 100);
    }

    #[test]
    fn test_subclip_duration_zero() {
        let sc = make_subclip(1, 10, 100, 100, "Empty");
        assert_eq!(sc.duration(), 0);
    }

    #[test]
    fn test_subclip_id_display() {
        let id = SubclipId(42);
        assert_eq!(id.to_string(), "subclip:42");
    }

    #[test]
    fn test_subclip_tree_add_and_children() {
        let mut tree = SubclipTree::new();
        tree.add(1, make_subclip(1, 1, 0, 100, "A"));
        tree.add(1, make_subclip(2, 1, 100, 200, "B"));
        tree.add(2, make_subclip(3, 2, 0, 50, "C"));

        let children = tree.children_of(1);
        assert_eq!(children.len(), 2);

        let children2 = tree.children_of(2);
        assert_eq!(children2.len(), 1);
    }

    #[test]
    fn test_subclip_tree_flatten() {
        let mut tree = SubclipTree::new();
        tree.add(1, make_subclip(1, 1, 0, 100, "A"));
        tree.add(2, make_subclip(2, 2, 0, 50, "B"));

        let flat = tree.flatten();
        assert_eq!(flat.len(), 2);
    }

    #[test]
    fn test_subclip_tree_empty() {
        let tree = SubclipTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.children_of(99).len(), 0);
    }

    #[test]
    fn test_subclip_tree_len() {
        let mut tree = SubclipTree::new();
        tree.add(1, make_subclip(1, 1, 0, 100, "A"));
        tree.add(1, make_subclip(2, 1, 100, 200, "B"));
        assert_eq!(tree.len(), 2);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_validator_valid() {
        let sc = make_subclip(1, 10, 10, 50, "Good");
        assert!(SubclipValidator::validate(&sc, 100).is_ok());
    }

    #[test]
    fn test_validator_in_out_equal() {
        let sc = make_subclip(1, 10, 50, 50, "Bad");
        assert!(SubclipValidator::validate(&sc, 100).is_err());
    }

    #[test]
    fn test_validator_out_exceeds_parent() {
        let sc = make_subclip(1, 10, 10, 150, "Bad");
        assert!(SubclipValidator::validate(&sc, 100).is_err());
    }

    #[test]
    fn test_validator_empty_label() {
        let sc = Subclip::new(SubclipId(1), 10, 10, 50, "", [0, 0, 0]);
        assert!(SubclipValidator::validate(&sc, 100).is_err());
    }

    #[test]
    fn test_edl_event_format() {
        let sc = make_subclip(1, 10, 0, 25, "Test");
        let edl = SubclipExporter::to_edl_event(&sc, 25.0);
        assert!(edl.starts_with("001  AX      V     C        "));
        // At 25fps, 0 frames = 00:00:00:00, 25 frames = 00:00:01:00
        assert!(edl.contains("00:00:00:00"));
        assert!(edl.contains("00:00:01:00"));
    }

    #[test]
    fn test_frames_to_timecode() {
        // 30fps, frame 90 = 3 seconds = 00:00:03:00
        assert_eq!(frames_to_timecode(90, 30.0), "00:00:03:00");
        // 25fps, frame 25 = 1 second = 00:00:01:00
        assert_eq!(frames_to_timecode(25, 25.0), "00:00:01:00");
        // 24fps, frame 48 = 2 seconds = 00:00:02:00
        assert_eq!(frames_to_timecode(48, 24.0), "00:00:02:00");
    }
}
