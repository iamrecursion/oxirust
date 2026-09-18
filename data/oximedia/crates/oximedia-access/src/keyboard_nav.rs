#![allow(dead_code)]
//! Keyboard navigation accessibility support for media player UIs.
//!
//! This module defines keyboard shortcuts, focus management, and navigation
//! models to ensure media player interfaces can be fully operated with a
//! keyboard alone, meeting WCAG 2.1 success criterion 2.1.1.

use std::collections::HashMap;
use std::fmt;

/// Modifier keys that can be combined with a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modifier {
    /// Control key.
    Ctrl,
    /// Shift key.
    Shift,
    /// Alt/Option key.
    Alt,
    /// Meta/Command/Windows key.
    Meta,
}

impl fmt::Display for Modifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ctrl => write!(f, "Ctrl"),
            Self::Shift => write!(f, "Shift"),
            Self::Alt => write!(f, "Alt"),
            Self::Meta => write!(f, "Meta"),
        }
    }
}

/// A keyboard shortcut consisting of modifiers and a key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyBinding {
    /// Set of modifier keys.
    pub modifiers: Vec<Modifier>,
    /// The primary key (e.g. "Space", "Enter", "`ArrowLeft`").
    pub key: String,
}

impl fmt::Display for KeyBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = self
            .modifiers
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        parts.push(self.key.clone());
        write!(f, "{}", parts.join("+"))
    }
}

impl KeyBinding {
    /// Create a new key binding with no modifiers.
    #[must_use]
    pub fn key(key: impl Into<String>) -> Self {
        Self {
            modifiers: Vec::new(),
            key: key.into(),
        }
    }

    /// Create a key binding with a single modifier.
    #[must_use]
    pub fn with_modifier(mut self, modifier: Modifier) -> Self {
        if !self.modifiers.contains(&modifier) {
            self.modifiers.push(modifier);
        }
        self
    }

    /// Create a Ctrl+key binding.
    #[must_use]
    pub fn ctrl(key: impl Into<String>) -> Self {
        Self::key(key).with_modifier(Modifier::Ctrl)
    }

    /// Create a Shift+key binding.
    #[must_use]
    pub fn shift(key: impl Into<String>) -> Self {
        Self::key(key).with_modifier(Modifier::Shift)
    }

    /// Check whether this binding uses a specific modifier.
    #[must_use]
    pub fn has_modifier(&self, modifier: Modifier) -> bool {
        self.modifiers.contains(&modifier)
    }

    /// Get the number of modifiers.
    #[must_use]
    pub fn modifier_count(&self) -> usize {
        self.modifiers.len()
    }
}

/// Standard media player actions that can be bound to keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaAction {
    /// Play/pause toggle.
    PlayPause,
    /// Stop playback.
    Stop,
    /// Skip forward by a configured amount.
    SkipForward,
    /// Skip backward by a configured amount.
    SkipBackward,
    /// Go to next frame (when paused).
    NextFrame,
    /// Go to previous frame (when paused).
    PreviousFrame,
    /// Increase volume.
    VolumeUp,
    /// Decrease volume.
    VolumeDown,
    /// Mute/unmute.
    MuteToggle,
    /// Toggle fullscreen.
    FullscreenToggle,
    /// Toggle captions.
    CaptionToggle,
    /// Speed up playback.
    SpeedUp,
    /// Slow down playback.
    SpeedDown,
    /// Toggle loop.
    LoopToggle,
    /// Focus next UI element.
    FocusNext,
    /// Focus previous UI element.
    FocusPrevious,
}

impl fmt::Display for MediaAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayPause => write!(f, "Play/Pause"),
            Self::Stop => write!(f, "Stop"),
            Self::SkipForward => write!(f, "Skip Forward"),
            Self::SkipBackward => write!(f, "Skip Backward"),
            Self::NextFrame => write!(f, "Next Frame"),
            Self::PreviousFrame => write!(f, "Previous Frame"),
            Self::VolumeUp => write!(f, "Volume Up"),
            Self::VolumeDown => write!(f, "Volume Down"),
            Self::MuteToggle => write!(f, "Mute Toggle"),
            Self::FullscreenToggle => write!(f, "Fullscreen Toggle"),
            Self::CaptionToggle => write!(f, "Caption Toggle"),
            Self::SpeedUp => write!(f, "Speed Up"),
            Self::SpeedDown => write!(f, "Speed Down"),
            Self::LoopToggle => write!(f, "Loop Toggle"),
            Self::FocusNext => write!(f, "Focus Next"),
            Self::FocusPrevious => write!(f, "Focus Previous"),
        }
    }
}

/// A keyboard shortcut map that binds keys to actions.
#[derive(Debug, Clone)]
pub struct KeyboardShortcutMap {
    /// Bindings from key to action.
    bindings: HashMap<KeyBinding, MediaAction>,
    /// Description of each binding for help text.
    descriptions: HashMap<KeyBinding, String>,
}

impl KeyboardShortcutMap {
    /// Create an empty shortcut map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            descriptions: HashMap::new(),
        }
    }

    /// Create a shortcut map with default media player bindings.
    #[must_use]
    pub fn defaults() -> Self {
        let mut map = Self::new();
        map.bind(
            KeyBinding::key("Space"),
            MediaAction::PlayPause,
            "Toggle play/pause",
        );
        map.bind(
            KeyBinding::key("ArrowRight"),
            MediaAction::SkipForward,
            "Skip forward 5s",
        );
        map.bind(
            KeyBinding::key("ArrowLeft"),
            MediaAction::SkipBackward,
            "Skip backward 5s",
        );
        map.bind(
            KeyBinding::key("ArrowUp"),
            MediaAction::VolumeUp,
            "Increase volume",
        );
        map.bind(
            KeyBinding::key("ArrowDown"),
            MediaAction::VolumeDown,
            "Decrease volume",
        );
        map.bind(KeyBinding::key("m"), MediaAction::MuteToggle, "Toggle mute");
        map.bind(
            KeyBinding::key("f"),
            MediaAction::FullscreenToggle,
            "Toggle fullscreen",
        );
        map.bind(
            KeyBinding::key("c"),
            MediaAction::CaptionToggle,
            "Toggle captions",
        );
        map.bind(
            KeyBinding::key("Tab"),
            MediaAction::FocusNext,
            "Focus next element",
        );
        map.bind(
            KeyBinding::shift("Tab"),
            MediaAction::FocusPrevious,
            "Focus previous element",
        );
        map.bind(KeyBinding::key("."), MediaAction::NextFrame, "Next frame");
        map.bind(
            KeyBinding::key(","),
            MediaAction::PreviousFrame,
            "Previous frame",
        );
        map
    }

    /// Bind a key to an action.
    pub fn bind(&mut self, key: KeyBinding, action: MediaAction, description: &str) {
        self.descriptions
            .insert(key.clone(), description.to_string());
        self.bindings.insert(key, action);
    }

    /// Remove a binding.
    pub fn unbind(&mut self, key: &KeyBinding) {
        self.bindings.remove(key);
        self.descriptions.remove(key);
    }

    /// Look up the action for a key.
    #[must_use]
    pub fn get_action(&self, key: &KeyBinding) -> Option<MediaAction> {
        self.bindings.get(key).copied()
    }

    /// Get the description for a key binding.
    #[must_use]
    pub fn get_description(&self, key: &KeyBinding) -> Option<&str> {
        self.descriptions.get(key).map(String::as_str)
    }

    /// Get all bindings for a given action.
    #[must_use]
    pub fn bindings_for_action(&self, action: MediaAction) -> Vec<&KeyBinding> {
        self.bindings
            .iter()
            .filter(|(_, a)| **a == action)
            .map(|(k, _)| k)
            .collect()
    }

    /// Get the total number of bindings.
    #[must_use]
    pub fn count(&self) -> usize {
        self.bindings.len()
    }

    /// Check whether a key has a binding.
    #[must_use]
    pub fn is_bound(&self, key: &KeyBinding) -> bool {
        self.bindings.contains_key(key)
    }

    /// Generate help text listing all shortcuts.
    #[must_use]
    pub fn help_text(&self) -> String {
        let mut lines: Vec<String> = self
            .bindings
            .iter()
            .map(|(key, action)| {
                let desc = self
                    .descriptions
                    .get(key)
                    .map_or(String::new(), std::clone::Clone::clone);
                format!("{key}: {action} - {desc}")
            })
            .collect();
        lines.sort();
        lines.join("\n")
    }
}

impl Default for KeyboardShortcutMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Focus order for UI elements to ensure logical Tab navigation.
#[derive(Debug, Clone)]
pub struct FocusOrder {
    /// Ordered list of focusable element IDs.
    elements: Vec<String>,
    /// Current focus index.
    current_index: Option<usize>,
}

impl FocusOrder {
    /// Create a new focus order.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            current_index: None,
        }
    }

    /// Add an element to the focus order.
    pub fn add_element(&mut self, id: impl Into<String>) {
        self.elements.push(id.into());
    }

    /// Get the current focused element ID.
    #[must_use]
    pub fn current(&self) -> Option<&str> {
        self.current_index
            .and_then(|i| self.elements.get(i))
            .map(String::as_str)
    }

    /// Move focus to the next element. Wraps around at the end.
    pub fn focus_next(&mut self) {
        if self.elements.is_empty() {
            return;
        }
        self.current_index = Some(match self.current_index {
            Some(i) => (i + 1) % self.elements.len(),
            None => 0,
        });
    }

    /// Move focus to the previous element. Wraps around at the beginning.
    pub fn focus_previous(&mut self) {
        if self.elements.is_empty() {
            return;
        }
        self.current_index = Some(match self.current_index {
            Some(0) => self.elements.len() - 1,
            Some(i) => i - 1,
            None => self.elements.len() - 1,
        });
    }

    /// Set focus to a specific element by ID.
    pub fn focus_element(&mut self, id: &str) -> bool {
        if let Some(idx) = self.elements.iter().position(|e| e == id) {
            self.current_index = Some(idx);
            true
        } else {
            false
        }
    }

    /// Get the total number of focusable elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check whether the focus order is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Reset focus to no element.
    pub fn reset(&mut self) {
        self.current_index = None;
    }
}

impl Default for FocusOrder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modifier_display() {
        assert_eq!(Modifier::Ctrl.to_string(), "Ctrl");
        assert_eq!(Modifier::Shift.to_string(), "Shift");
        assert_eq!(Modifier::Alt.to_string(), "Alt");
        assert_eq!(Modifier::Meta.to_string(), "Meta");
    }

    #[test]
    fn test_key_binding_simple() {
        let kb = KeyBinding::key("Space");
        assert_eq!(kb.key, "Space");
        assert_eq!(kb.modifier_count(), 0);
        assert_eq!(kb.to_string(), "Space");
    }

    #[test]
    fn test_key_binding_with_modifier() {
        let kb = KeyBinding::ctrl("s");
        assert!(kb.has_modifier(Modifier::Ctrl));
        assert!(!kb.has_modifier(Modifier::Shift));
        assert_eq!(kb.to_string(), "Ctrl+s");
    }

    #[test]
    fn test_key_binding_shift() {
        let kb = KeyBinding::shift("Tab");
        assert!(kb.has_modifier(Modifier::Shift));
        assert_eq!(kb.to_string(), "Shift+Tab");
    }

    #[test]
    fn test_media_action_display() {
        assert_eq!(MediaAction::PlayPause.to_string(), "Play/Pause");
        assert_eq!(MediaAction::VolumeUp.to_string(), "Volume Up");
        assert_eq!(MediaAction::CaptionToggle.to_string(), "Caption Toggle");
    }

    #[test]
    fn test_shortcut_map_defaults() {
        let map = KeyboardShortcutMap::defaults();
        assert!(map.count() >= 10);
        assert_eq!(
            map.get_action(&KeyBinding::key("Space")),
            Some(MediaAction::PlayPause)
        );
        assert_eq!(
            map.get_action(&KeyBinding::key("m")),
            Some(MediaAction::MuteToggle)
        );
    }

    #[test]
    fn test_shortcut_map_custom_bind() {
        let mut map = KeyboardShortcutMap::new();
        map.bind(KeyBinding::ctrl("p"), MediaAction::PlayPause, "Play/Pause");
        assert_eq!(
            map.get_action(&KeyBinding::ctrl("p")),
            Some(MediaAction::PlayPause)
        );
        assert_eq!(map.count(), 1);
    }

    #[test]
    fn test_shortcut_map_unbind() {
        let mut map = KeyboardShortcutMap::defaults();
        let space = KeyBinding::key("Space");
        assert!(map.is_bound(&space));
        map.unbind(&space);
        assert!(!map.is_bound(&space));
    }

    #[test]
    fn test_shortcut_map_description() {
        let map = KeyboardShortcutMap::defaults();
        let space = KeyBinding::key("Space");
        assert_eq!(map.get_description(&space), Some("Toggle play/pause"));
    }

    #[test]
    fn test_bindings_for_action() {
        let map = KeyboardShortcutMap::defaults();
        let bindings = map.bindings_for_action(MediaAction::PlayPause);
        assert!(!bindings.is_empty());
    }

    #[test]
    fn test_focus_order_navigation() {
        let mut fo = FocusOrder::new();
        fo.add_element("play_btn");
        fo.add_element("volume_slider");
        fo.add_element("timeline");

        assert!(fo.current().is_none());
        fo.focus_next();
        assert_eq!(fo.current(), Some("play_btn"));
        fo.focus_next();
        assert_eq!(fo.current(), Some("volume_slider"));
        fo.focus_next();
        assert_eq!(fo.current(), Some("timeline"));
        fo.focus_next();
        assert_eq!(fo.current(), Some("play_btn")); // wraps
    }

    #[test]
    fn test_focus_order_previous() {
        let mut fo = FocusOrder::new();
        fo.add_element("a");
        fo.add_element("b");
        fo.add_element("c");

        fo.focus_previous(); // wraps to last
        assert_eq!(fo.current(), Some("c"));
        fo.focus_previous();
        assert_eq!(fo.current(), Some("b"));
        fo.focus_previous();
        assert_eq!(fo.current(), Some("a"));
        fo.focus_previous(); // wraps
        assert_eq!(fo.current(), Some("c"));
    }

    #[test]
    fn test_focus_order_by_id() {
        let mut fo = FocusOrder::new();
        fo.add_element("a");
        fo.add_element("b");
        fo.add_element("c");

        assert!(fo.focus_element("b"));
        assert_eq!(fo.current(), Some("b"));
        assert!(!fo.focus_element("nonexistent"));
    }

    #[test]
    fn test_focus_order_reset() {
        let mut fo = FocusOrder::new();
        fo.add_element("x");
        fo.focus_next();
        assert!(fo.current().is_some());
        fo.reset();
        assert!(fo.current().is_none());
    }

    #[test]
    fn test_help_text_contains_bindings() {
        let map = KeyboardShortcutMap::defaults();
        let help = map.help_text();
        assert!(!help.is_empty());
        assert!(help.contains("Play/Pause"));
    }
}
