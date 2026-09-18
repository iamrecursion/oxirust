#![allow(dead_code)]
//! Virtual studio environment management.
//!
//! Provides `StudioElement`, `VirtualStudio` for modelling virtual production
//! stage geometry and tracking state.

/// A physical or virtual element within the studio environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StudioElement {
    /// The studio floor surface.
    Floor,
    /// A studio wall.
    Wall,
    /// The ceiling of the studio.
    Ceiling,
    /// A prop or set piece.
    Prop,
}

impl StudioElement {
    /// Returns `true` if this element represents a physical structure
    /// that a person could walk on or touch.
    #[must_use]
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::Floor | Self::Wall | Self::Ceiling)
    }
}

/// A placed instance of a `StudioElement` with a name and optional position.
#[derive(Debug, Clone)]
pub struct PlacedElement {
    /// Element type.
    pub element: StudioElement,
    /// Descriptive name (e.g., "Back wall", "Coffee table").
    pub name: String,
    /// Optional 3-D centre position in metres (x, y, z).
    pub position: Option<(f32, f32, f32)>,
}

impl PlacedElement {
    /// Create a new `PlacedElement`.
    #[must_use]
    pub fn new(element: StudioElement, name: impl Into<String>) -> Self {
        Self {
            element,
            name: name.into(),
            position: None,
        }
    }

    /// Attach a 3-D position.
    #[must_use]
    pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = Some((x, y, z));
        self
    }
}

/// A virtual studio containing a collection of placed elements and a tracking flag.
#[derive(Debug, Default)]
pub struct VirtualStudio {
    elements: Vec<PlacedElement>,
    tracking_enabled: bool,
}

impl VirtualStudio {
    /// Create an empty studio with tracking disabled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a studio with tracking explicitly configured.
    #[must_use]
    pub fn with_tracking(tracking_enabled: bool) -> Self {
        Self {
            elements: Vec::new(),
            tracking_enabled,
        }
    }

    /// Add a `PlacedElement` to the studio.
    pub fn add_element(&mut self, element: PlacedElement) {
        self.elements.push(element);
    }

    /// Returns the number of elements currently in the studio.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if camera tracking is enabled for this studio.
    #[must_use]
    pub fn has_tracking(&self) -> bool {
        self.tracking_enabled
    }

    /// Enable or disable camera tracking.
    pub fn set_tracking(&mut self, enabled: bool) {
        self.tracking_enabled = enabled;
    }

    /// Return all placed elements.
    #[must_use]
    pub fn elements(&self) -> &[PlacedElement] {
        &self.elements
    }

    /// Return elements filtered by type.
    #[must_use]
    pub fn elements_of_type(&self, element_type: StudioElement) -> Vec<&PlacedElement> {
        self.elements
            .iter()
            .filter(|e| e.element == element_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_is_physical() {
        assert!(StudioElement::Floor.is_physical());
    }

    #[test]
    fn test_wall_is_physical() {
        assert!(StudioElement::Wall.is_physical());
    }

    #[test]
    fn test_ceiling_is_physical() {
        assert!(StudioElement::Ceiling.is_physical());
    }

    #[test]
    fn test_prop_is_not_physical() {
        assert!(!StudioElement::Prop.is_physical());
    }

    #[test]
    fn test_placed_element_no_position() {
        let el = PlacedElement::new(StudioElement::Floor, "Main floor");
        assert!(el.position.is_none());
        assert_eq!(el.name, "Main floor");
    }

    #[test]
    fn test_placed_element_with_position() {
        let el = PlacedElement::new(StudioElement::Wall, "Back wall").at(0.0, 2.0, -5.0);
        assert_eq!(el.position, Some((0.0, 2.0, -5.0)));
    }

    #[test]
    fn test_studio_default_no_tracking() {
        let studio = VirtualStudio::new();
        assert!(!studio.has_tracking());
    }

    #[test]
    fn test_studio_with_tracking() {
        let studio = VirtualStudio::with_tracking(true);
        assert!(studio.has_tracking());
    }

    #[test]
    fn test_studio_set_tracking() {
        let mut studio = VirtualStudio::new();
        studio.set_tracking(true);
        assert!(studio.has_tracking());
    }

    #[test]
    fn test_studio_add_and_count() {
        let mut studio = VirtualStudio::new();
        assert_eq!(studio.element_count(), 0);
        studio.add_element(PlacedElement::new(StudioElement::Floor, "Floor"));
        studio.add_element(PlacedElement::new(StudioElement::Wall, "Wall A"));
        assert_eq!(studio.element_count(), 2);
    }

    #[test]
    fn test_elements_of_type_filter() {
        let mut studio = VirtualStudio::new();
        studio.add_element(PlacedElement::new(StudioElement::Floor, "Floor"));
        studio.add_element(PlacedElement::new(StudioElement::Prop, "Chair"));
        studio.add_element(PlacedElement::new(StudioElement::Prop, "Desk"));
        let props = studio.elements_of_type(StudioElement::Prop);
        assert_eq!(props.len(), 2);
    }

    #[test]
    fn test_elements_of_type_empty_result() {
        let mut studio = VirtualStudio::new();
        studio.add_element(PlacedElement::new(StudioElement::Floor, "Floor"));
        let ceilings = studio.elements_of_type(StudioElement::Ceiling);
        assert!(ceilings.is_empty());
    }

    #[test]
    fn test_elements_accessor_length() {
        let mut studio = VirtualStudio::new();
        studio.add_element(PlacedElement::new(StudioElement::Ceiling, "Ceiling"));
        assert_eq!(studio.elements().len(), 1);
    }
}
