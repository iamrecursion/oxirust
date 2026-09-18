//! Input source manager for professional video switchers.
//!
//! Provides a high-level interface for registering, querying, and activating
//! input sources. Works on top of the lower-level `input` router and supports
//! labelling, grouping, and active-source queries.

#![allow(dead_code)]

use std::collections::HashMap;

/// Broad category of a switcher input source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwitcherInputKind {
    /// Serial Digital Interface (SDI) input.
    Sdi,
    /// Network Device Interface (NDI) stream.
    Ndi,
    /// HDMI input.
    Hdmi,
    /// IP video stream (SRT, RTP, etc.).
    IpStream,
    /// Internal test/color generator.
    TestGenerator,
    /// Still frame from media pool.
    MediaPool,
    /// M/E output fed back as an input.
    MixEffectReturn,
}

impl SwitcherInputKind {
    /// Returns `true` if the kind represents a live physical input.
    #[must_use]
    pub fn is_physical(self) -> bool {
        matches!(self, Self::Sdi | Self::Hdmi | Self::Ndi | Self::IpStream)
    }
}

impl std::fmt::Display for SwitcherInputKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Sdi => "SDI",
            Self::Ndi => "NDI",
            Self::Hdmi => "HDMI",
            Self::IpStream => "IP Stream",
            Self::TestGenerator => "Test Generator",
            Self::MediaPool => "Media Pool",
            Self::MixEffectReturn => "M/E Return",
        };
        write!(f, "{name}")
    }
}

/// Represents a single input registered with the [`InputManager`].
#[derive(Debug, Clone)]
pub struct ManagedInput {
    /// Unique numeric ID (1-based channel number).
    pub id: u32,
    /// Human-readable label (e.g. "Camera 1").
    pub label: String,
    /// Source type.
    pub kind: SwitcherInputKind,
    /// Whether the input is currently enabled/active.
    pub active: bool,
    /// Optional group tag for logical grouping.
    pub group: Option<String>,
}

impl ManagedInput {
    /// Create a new managed input.
    #[must_use]
    pub fn new(id: u32, label: &str, kind: SwitcherInputKind) -> Self {
        Self {
            id,
            label: label.to_string(),
            kind,
            active: false,
            group: None,
        }
    }

    /// Attach this input to a named group.
    #[must_use]
    pub fn with_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }
}

/// Manages the full set of inputs available to a switcher.
///
/// Inputs can be registered, labelled, grouped, activated/deactivated, and
/// looked up by ID or label.
#[derive(Debug, Default)]
pub struct InputManager {
    inputs: HashMap<u32, ManagedInput>,
    /// Currently selected (active) input ID, if any.
    active_id: Option<u32>,
}

impl InputManager {
    /// Create a new, empty input manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new input.
    ///
    /// If an input with the same ID already exists it is replaced.
    pub fn register(&mut self, input: ManagedInput) {
        self.inputs.insert(input.id, input);
    }

    /// Deregister (remove) an input by ID.
    pub fn deregister(&mut self, id: u32) {
        self.inputs.remove(&id);
        if self.active_id == Some(id) {
            self.active_id = None;
        }
    }

    /// Set the active (selected) input.
    ///
    /// Returns `false` if the ID is not registered.
    pub fn set_active(&mut self, id: u32) -> bool {
        if self.inputs.contains_key(&id) {
            self.active_id = Some(id);
            if let Some(inp) = self.inputs.get_mut(&id) {
                inp.active = true;
            }
            true
        } else {
            false
        }
    }

    /// Get the currently active input, if one is selected.
    #[must_use]
    pub fn get_active(&self) -> Option<&ManagedInput> {
        self.active_id.and_then(|id| self.inputs.get(&id))
    }

    /// Look up an input by its numeric ID.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&ManagedInput> {
        self.inputs.get(&id)
    }

    /// Find inputs by label (case-insensitive partial match).
    #[must_use]
    pub fn find_by_label(&self, query: &str) -> Vec<&ManagedInput> {
        let lower = query.to_lowercase();
        self.inputs
            .values()
            .filter(|inp| inp.label.to_lowercase().contains(&lower))
            .collect()
    }

    /// Return all inputs of a given kind.
    #[must_use]
    pub fn inputs_of_kind(&self, kind: SwitcherInputKind) -> Vec<&ManagedInput> {
        self.inputs
            .values()
            .filter(|inp| inp.kind == kind)
            .collect()
    }

    /// Return all inputs belonging to a named group.
    #[must_use]
    pub fn inputs_in_group(&self, group: &str) -> Vec<&ManagedInput> {
        self.inputs
            .values()
            .filter(|inp| inp.group.as_deref() == Some(group))
            .collect()
    }

    /// Return all registered inputs sorted by ID.
    #[must_use]
    pub fn all_sorted(&self) -> Vec<&ManagedInput> {
        let mut v: Vec<&ManagedInput> = self.inputs.values().collect();
        v.sort_by_key(|inp| inp.id);
        v
    }

    /// Total number of registered inputs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.inputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> InputManager {
        let mut m = InputManager::new();
        m.register(ManagedInput::new(1, "Camera 1", SwitcherInputKind::Sdi));
        m.register(ManagedInput::new(2, "Camera 2", SwitcherInputKind::Sdi));
        m.register(ManagedInput::new(3, "Laptop", SwitcherInputKind::Hdmi));
        m.register(
            ManagedInput::new(4, "Stream Feed", SwitcherInputKind::IpStream).with_group("remote"),
        );
        m
    }

    #[test]
    fn test_new_manager_is_empty() {
        let m = InputManager::new();
        assert_eq!(m.count(), 0);
        assert!(m.get_active().is_none());
    }

    #[test]
    fn test_register_input() {
        let m = make_manager();
        assert_eq!(m.count(), 4);
    }

    #[test]
    fn test_get_by_id() {
        let m = make_manager();
        let inp = m.get(1).expect("should succeed in test");
        assert_eq!(inp.label, "Camera 1");
        assert_eq!(inp.kind, SwitcherInputKind::Sdi);
    }

    #[test]
    fn test_get_missing_id() {
        let m = make_manager();
        assert!(m.get(99).is_none());
    }

    #[test]
    fn test_set_active_and_get_active() {
        let mut m = make_manager();
        let ok = m.set_active(2);
        assert!(ok);
        let active = m.get_active().expect("should succeed in test");
        assert_eq!(active.id, 2);
        assert_eq!(active.label, "Camera 2");
    }

    #[test]
    fn test_set_active_nonexistent_returns_false() {
        let mut m = make_manager();
        let ok = m.set_active(99);
        assert!(!ok);
        assert!(m.get_active().is_none());
    }

    #[test]
    fn test_deregister_clears_active() {
        let mut m = make_manager();
        m.set_active(1);
        m.deregister(1);
        assert!(m.get_active().is_none());
        assert_eq!(m.count(), 3);
    }

    #[test]
    fn test_find_by_label() {
        let m = make_manager();
        let results = m.find_by_label("camera");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inputs_of_kind() {
        let m = make_manager();
        let sdi = m.inputs_of_kind(SwitcherInputKind::Sdi);
        assert_eq!(sdi.len(), 2);
        let hdmi = m.inputs_of_kind(SwitcherInputKind::Hdmi);
        assert_eq!(hdmi.len(), 1);
    }

    #[test]
    fn test_inputs_in_group() {
        let m = make_manager();
        let remote = m.inputs_in_group("remote");
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].id, 4);
    }

    #[test]
    fn test_all_sorted_order() {
        let m = make_manager();
        let sorted = m.all_sorted();
        assert_eq!(sorted.len(), 4);
        for pair in sorted.windows(2) {
            assert!(pair[0].id < pair[1].id);
        }
    }

    #[test]
    fn test_switcher_input_kind_is_physical() {
        assert!(SwitcherInputKind::Sdi.is_physical());
        assert!(SwitcherInputKind::Hdmi.is_physical());
        assert!(!SwitcherInputKind::TestGenerator.is_physical());
        assert!(!SwitcherInputKind::MediaPool.is_physical());
    }

    #[test]
    fn test_switcher_input_kind_display() {
        assert_eq!(SwitcherInputKind::Sdi.to_string(), "SDI");
        assert_eq!(SwitcherInputKind::MixEffectReturn.to_string(), "M/E Return");
    }

    #[test]
    fn test_replace_existing_input() {
        let mut m = InputManager::new();
        m.register(ManagedInput::new(1, "Old Label", SwitcherInputKind::Sdi));
        m.register(ManagedInput::new(1, "New Label", SwitcherInputKind::Hdmi));
        assert_eq!(m.count(), 1);
        assert_eq!(m.get(1).expect("should succeed in test").label, "New Label");
    }
}
