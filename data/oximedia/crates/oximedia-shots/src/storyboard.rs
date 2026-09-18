//! Storyboard / visual planning for shots.
//!
//! Provides lightweight data structures for describing storyboard panels,
//! shot sequences, and visual notes used in pre-production planning.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// Panel geometry
// ──────────────────────────────────────────────────────────────────────────────

/// Normalised bounding box in \[0, 1\] frame space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Left edge (0 = frame left).
    pub x: f32,
    /// Top edge (0 = frame top).
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl BoundingBox {
    /// Create a bounding box from `(x, y, width, height)`.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the centre of the bounding box.
    #[must_use]
    pub fn centre(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Return the area of the bounding box.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Return `true` if the point `(px, py)` is inside the box.
    #[must_use]
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    /// Return the intersection area with another bounding box.
    #[must_use]
    pub fn intersection_area(&self, other: &Self) -> f32 {
        let ix = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let iy = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        if ix > 0.0 && iy > 0.0 {
            ix * iy
        } else {
            0.0
        }
    }

    /// Return the IoU (Intersection-over-Union) with another bounding box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let inter = self.intersection_area(other);
        if inter == 0.0 {
            return 0.0;
        }
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Subject annotation
// ──────────────────────────────────────────────────────────────────────────────

/// A labelled subject region within a storyboard panel.
#[derive(Debug, Clone)]
pub struct SubjectAnnotation {
    /// Label (e.g. "Hero", "Villain", "Camera A").
    pub label: String,
    /// Bounding region.
    pub region: BoundingBox,
    /// Whether this subject is the primary focus.
    pub is_primary: bool,
}

impl SubjectAnnotation {
    /// Create a new subject annotation.
    #[must_use]
    pub fn new(label: impl Into<String>, region: BoundingBox, is_primary: bool) -> Self {
        Self {
            label: label.into(),
            region,
            is_primary,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Camera action
// ──────────────────────────────────────────────────────────────────────────────

/// Intended camera action for this panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraAction {
    /// Static / locked-off camera.
    Static,
    /// Pan left.
    PanLeft,
    /// Pan right.
    PanRight,
    /// Tilt up.
    TiltUp,
    /// Tilt down.
    TiltDown,
    /// Zoom in.
    ZoomIn,
    /// Zoom out.
    ZoomOut,
    /// Dolly (push) toward subject.
    DollyIn,
    /// Dolly (pull) away from subject.
    DollyOut,
    /// Handheld / free movement.
    Handheld,
}

impl fmt::Display for CameraAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Static => "Static",
            Self::PanLeft => "Pan Left",
            Self::PanRight => "Pan Right",
            Self::TiltUp => "Tilt Up",
            Self::TiltDown => "Tilt Down",
            Self::ZoomIn => "Zoom In",
            Self::ZoomOut => "Zoom Out",
            Self::DollyIn => "Dolly In",
            Self::DollyOut => "Dolly Out",
            Self::Handheld => "Handheld",
        };
        write!(f, "{s}")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Storyboard panel
// ──────────────────────────────────────────────────────────────────────────────

/// A single storyboard panel representing one planned shot.
#[derive(Debug, Clone)]
pub struct StoryboardPanel {
    /// Sequential panel index (0-based).
    pub index: usize,
    /// Scene identifier.
    pub scene: String,
    /// Shot number within the scene.
    pub shot_number: u32,
    /// Brief description of the action in this panel.
    pub action: String,
    /// Dialogue or sound cue (if any).
    pub dialogue: Option<String>,
    /// Intended camera action.
    pub camera_action: CameraAction,
    /// Annotated subjects.
    pub subjects: Vec<SubjectAnnotation>,
    /// Estimated duration in seconds.
    pub estimated_duration_secs: f32,
    /// Whether this panel is a key / establishing panel.
    pub is_key_panel: bool,
}

impl StoryboardPanel {
    /// Create a minimal panel.
    #[must_use]
    pub fn new(
        index: usize,
        scene: impl Into<String>,
        shot_number: u32,
        action: impl Into<String>,
    ) -> Self {
        Self {
            index,
            scene: scene.into(),
            shot_number,
            action: action.into(),
            dialogue: None,
            camera_action: CameraAction::Static,
            subjects: Vec::new(),
            estimated_duration_secs: 3.0,
            is_key_panel: false,
        }
    }

    /// Attach a dialogue / sound cue.
    #[must_use]
    pub fn with_dialogue(mut self, text: impl Into<String>) -> Self {
        self.dialogue = Some(text.into());
        self
    }

    /// Set the camera action.
    #[must_use]
    pub fn with_camera_action(mut self, action: CameraAction) -> Self {
        self.camera_action = action;
        self
    }

    /// Set the estimated duration.
    #[must_use]
    pub fn with_duration(mut self, secs: f32) -> Self {
        self.estimated_duration_secs = secs;
        self
    }

    /// Mark this panel as a key / establishing panel.
    #[must_use]
    pub fn key(mut self) -> Self {
        self.is_key_panel = true;
        self
    }

    /// Add a subject annotation.
    pub fn add_subject(&mut self, subject: SubjectAnnotation) {
        self.subjects.push(subject);
    }

    /// Return the primary subject, if any.
    #[must_use]
    pub fn primary_subject(&self) -> Option<&SubjectAnnotation> {
        self.subjects.iter().find(|s| s.is_primary)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Storyboard
// ──────────────────────────────────────────────────────────────────────────────

/// A complete storyboard composed of ordered panels.
#[derive(Debug, Clone, Default)]
pub struct Storyboard {
    panels: Vec<StoryboardPanel>,
}

impl Storyboard {
    /// Create an empty storyboard.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a panel.
    pub fn push(&mut self, panel: StoryboardPanel) {
        self.panels.push(panel);
    }

    /// Return the number of panels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.panels.len()
    }

    /// Return `true` if there are no panels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }

    /// Return all key panels.
    #[must_use]
    pub fn key_panels(&self) -> Vec<&StoryboardPanel> {
        self.panels.iter().filter(|p| p.is_key_panel).collect()
    }

    /// Return estimated total duration across all panels in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f32 {
        self.panels.iter().map(|p| p.estimated_duration_secs).sum()
    }

    /// Return panels for a specific scene.
    #[must_use]
    pub fn panels_for_scene(&self, scene: &str) -> Vec<&StoryboardPanel> {
        self.panels.iter().filter(|p| p.scene == scene).collect()
    }

    /// Return panels that include a given camera action.
    #[must_use]
    pub fn panels_with_action(&self, action: CameraAction) -> Vec<&StoryboardPanel> {
        self.panels
            .iter()
            .filter(|p| p.camera_action == action)
            .collect()
    }

    /// Return an iterator over all panels.
    pub fn iter(&self) -> impl Iterator<Item = &StoryboardPanel> {
        self.panels.iter()
    }

    /// Return the panel at the given index, or `None`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&StoryboardPanel> {
        self.panels.get(index)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_centre() {
        let bb = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        assert_eq!(bb.centre(), (0.25, 0.25));
    }

    #[test]
    fn test_bounding_box_area() {
        let bb = BoundingBox::new(0.0, 0.0, 0.4, 0.5);
        assert!((bb.area() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox::new(0.1, 0.1, 0.5, 0.5);
        assert!(bb.contains(0.3, 0.3));
        assert!(!bb.contains(0.0, 0.0));
    }

    #[test]
    fn test_bounding_box_iou_identical() {
        let bb = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        assert!((bb.iou(&bb) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 0.2, 0.2);
        let b = BoundingBox::new(0.5, 0.5, 0.2, 0.2);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_camera_action_display() {
        assert_eq!(CameraAction::PanLeft.to_string(), "Pan Left");
        assert_eq!(CameraAction::ZoomIn.to_string(), "Zoom In");
        assert_eq!(CameraAction::Static.to_string(), "Static");
    }

    #[test]
    fn test_storyboard_panel_new() {
        let p = StoryboardPanel::new(0, "1", 1, "Hero enters");
        assert_eq!(p.index, 0);
        assert_eq!(p.scene, "1");
        assert_eq!(p.shot_number, 1);
        assert!(!p.is_key_panel);
    }

    #[test]
    fn test_storyboard_panel_with_dialogue() {
        let p = StoryboardPanel::new(0, "1", 1, "Talk").with_dialogue("Hello world");
        assert_eq!(p.dialogue.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_storyboard_panel_key() {
        let p = StoryboardPanel::new(0, "1", 1, "Establishing").key();
        assert!(p.is_key_panel);
    }

    #[test]
    fn test_storyboard_panel_primary_subject() {
        let mut p = StoryboardPanel::new(0, "1", 1, "Two shot");
        p.add_subject(SubjectAnnotation::new(
            "A",
            BoundingBox::new(0.0, 0.0, 0.3, 0.5),
            false,
        ));
        p.add_subject(SubjectAnnotation::new(
            "B",
            BoundingBox::new(0.5, 0.0, 0.3, 0.5),
            true,
        ));
        assert_eq!(
            p.primary_subject().expect("should succeed in test").label,
            "B"
        );
    }

    #[test]
    fn test_storyboard_empty() {
        let sb = Storyboard::new();
        assert!(sb.is_empty());
        assert_eq!(sb.len(), 0);
        assert_eq!(sb.total_duration_secs(), 0.0);
    }

    #[test]
    fn test_storyboard_total_duration() {
        let mut sb = Storyboard::new();
        sb.push(StoryboardPanel::new(0, "1", 1, "A").with_duration(2.5));
        sb.push(StoryboardPanel::new(1, "1", 2, "B").with_duration(4.0));
        assert!((sb.total_duration_secs() - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_storyboard_key_panels() {
        let mut sb = Storyboard::new();
        sb.push(StoryboardPanel::new(0, "1", 1, "A"));
        sb.push(StoryboardPanel::new(1, "1", 2, "B").key());
        assert_eq!(sb.key_panels().len(), 1);
    }

    #[test]
    fn test_storyboard_panels_for_scene() {
        let mut sb = Storyboard::new();
        sb.push(StoryboardPanel::new(0, "1", 1, "A"));
        sb.push(StoryboardPanel::new(1, "2", 1, "B"));
        sb.push(StoryboardPanel::new(2, "1", 2, "C"));
        assert_eq!(sb.panels_for_scene("1").len(), 2);
        assert_eq!(sb.panels_for_scene("2").len(), 1);
    }

    #[test]
    fn test_storyboard_panels_with_action() {
        let mut sb = Storyboard::new();
        sb.push(StoryboardPanel::new(0, "1", 1, "A").with_camera_action(CameraAction::PanLeft));
        sb.push(StoryboardPanel::new(1, "1", 2, "B"));
        sb.push(StoryboardPanel::new(2, "1", 3, "C").with_camera_action(CameraAction::PanLeft));
        assert_eq!(sb.panels_with_action(CameraAction::PanLeft).len(), 2);
        assert_eq!(sb.panels_with_action(CameraAction::Static).len(), 1);
    }

    #[test]
    fn test_storyboard_get() {
        let mut sb = Storyboard::new();
        sb.push(StoryboardPanel::new(0, "1", 1, "Only panel"));
        assert!(sb.get(0).is_some());
        assert!(sb.get(1).is_none());
    }
}
