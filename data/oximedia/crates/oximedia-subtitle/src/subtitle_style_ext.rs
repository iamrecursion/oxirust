//! Extended subtitle style management for OxiMedia.
//!
//! Provides a named attribute system, a style set supporting get/set/merge,
//! and a preset library for common styling templates.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single stylistic attribute that can be applied to a subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleAttribute {
    /// Font family name.
    FontFamily(String),
    /// Font size in points.
    FontSize(f32),
    /// Bold weight.
    Bold(bool),
    /// Italic style.
    Italic(bool),
    /// Underline decoration.
    Underline(bool),
    /// Foreground colour as RGBA (0–255 each).
    Color(u8, u8, u8, u8),
    /// Background colour as RGBA.
    BackgroundColor(u8, u8, u8, u8),
    /// Outline width in pixels.
    OutlineWidth(f32),
    /// Shadow offset in pixels.
    ShadowOffset(f32, f32),
    /// Letter spacing in points.
    LetterSpacing(f32),
    /// Line height multiplier.
    LineHeight(f32),
    /// Text alignment: "left", "center", "right".
    TextAlign(String),
    /// Opacity (0.0–1.0).
    Opacity(f32),
}

impl StyleAttribute {
    /// Returns `true` when this attribute affects visual rendering.
    #[must_use]
    pub fn is_visual(&self) -> bool {
        !matches!(self, Self::FontFamily(_) | Self::TextAlign(_))
    }

    /// Returns the key name used to identify this attribute type.
    #[must_use]
    pub fn key(&self) -> &'static str {
        match self {
            Self::FontFamily(_) => "font-family",
            Self::FontSize(_) => "font-size",
            Self::Bold(_) => "bold",
            Self::Italic(_) => "italic",
            Self::Underline(_) => "underline",
            Self::Color(_, _, _, _) => "color",
            Self::BackgroundColor(_, _, _, _) => "background-color",
            Self::OutlineWidth(_) => "outline-width",
            Self::ShadowOffset(_, _) => "shadow-offset",
            Self::LetterSpacing(_) => "letter-spacing",
            Self::LineHeight(_) => "line-height",
            Self::TextAlign(_) => "text-align",
            Self::Opacity(_) => "opacity",
        }
    }
}

/// A named collection of style attributes for a subtitle track or cue.
#[derive(Debug, Clone, Default)]
pub struct SubtitleStyleSet {
    name: String,
    attributes: HashMap<String, StyleAttribute>,
}

impl SubtitleStyleSet {
    /// Creates a new empty style set with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: HashMap::new(),
        }
    }

    /// Returns the name of this style set.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets an attribute, replacing any existing attribute with the same key.
    pub fn set(&mut self, attr: StyleAttribute) {
        self.attributes.insert(attr.key().to_owned(), attr);
    }

    /// Returns the attribute for the given key, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&StyleAttribute> {
        self.attributes.get(key)
    }

    /// Removes an attribute by key. Returns the removed attribute if it existed.
    pub fn remove(&mut self, key: &str) -> Option<StyleAttribute> {
        self.attributes.remove(key)
    }

    /// Returns the number of attributes currently set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    /// Returns `true` when no attributes are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Merges attributes from `other` into this set.
    /// Attributes in `other` take precedence over existing ones.
    pub fn merge(&mut self, other: &SubtitleStyleSet) {
        for (key, attr) in &other.attributes {
            self.attributes.insert(key.clone(), attr.clone());
        }
    }

    /// Returns all visual attributes.
    #[must_use]
    pub fn visual_attributes(&self) -> Vec<&StyleAttribute> {
        self.attributes.values().filter(|a| a.is_visual()).collect()
    }
}

/// A preset-based library of named `SubtitleStyleSet` entries.
#[derive(Debug, Default)]
pub struct StylePresetLibrary {
    presets: HashMap<String, SubtitleStyleSet>,
}

impl StylePresetLibrary {
    /// Creates a new empty preset library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a library pre-populated with sensible broadcast defaults.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut lib = Self::new();

        let mut default_style = SubtitleStyleSet::new("default");
        default_style.set(StyleAttribute::FontFamily("Arial".to_owned()));
        default_style.set(StyleAttribute::FontSize(48.0));
        default_style.set(StyleAttribute::Color(255, 255, 255, 255));
        default_style.set(StyleAttribute::Bold(false));
        lib.add(default_style);

        let mut hearing_impaired = SubtitleStyleSet::new("hearing-impaired");
        hearing_impaired.set(StyleAttribute::FontSize(52.0));
        hearing_impaired.set(StyleAttribute::Color(255, 255, 0, 255));
        hearing_impaired.set(StyleAttribute::BackgroundColor(0, 0, 0, 200));
        lib.add(hearing_impaired);

        lib
    }

    /// Adds a style set to the library, keyed by its name.
    pub fn add(&mut self, style_set: SubtitleStyleSet) {
        self.presets.insert(style_set.name().to_owned(), style_set);
    }

    /// Finds a preset by name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&SubtitleStyleSet> {
        self.presets.get(name)
    }

    /// Returns `true` when a preset with the given name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.presets.contains_key(name)
    }

    /// Returns the number of presets stored.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Returns the names of all presets in arbitrary order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.presets.keys().map(String::as_str).collect()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_attribute_is_visual_font_size() {
        assert!(StyleAttribute::FontSize(48.0).is_visual());
    }

    #[test]
    fn test_style_attribute_is_visual_font_family_false() {
        assert!(!StyleAttribute::FontFamily("Arial".into()).is_visual());
    }

    #[test]
    fn test_style_attribute_key() {
        assert_eq!(StyleAttribute::Bold(true).key(), "bold");
        assert_eq!(StyleAttribute::Color(0, 0, 0, 255).key(), "color");
        assert_eq!(StyleAttribute::Opacity(0.8).key(), "opacity");
    }

    #[test]
    fn test_style_set_set_and_get() {
        let mut set = SubtitleStyleSet::new("test");
        set.set(StyleAttribute::FontSize(36.0));
        let attr = set.get("font-size");
        assert!(attr.is_some());
        assert!(
            matches!(attr.expect("should succeed in test"), StyleAttribute::FontSize(v) if (*v - 36.0).abs() < 0.01)
        );
    }

    #[test]
    fn test_style_set_remove() {
        let mut set = SubtitleStyleSet::new("test");
        set.set(StyleAttribute::Bold(true));
        let removed = set.remove("bold");
        assert!(removed.is_some());
        assert!(set.get("bold").is_none());
    }

    #[test]
    fn test_style_set_is_empty() {
        let set = SubtitleStyleSet::new("empty");
        assert!(set.is_empty());
    }

    #[test]
    fn test_style_set_len() {
        let mut set = SubtitleStyleSet::new("test");
        set.set(StyleAttribute::Bold(false));
        set.set(StyleAttribute::Italic(true));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_style_set_merge_overrides() {
        let mut base = SubtitleStyleSet::new("base");
        base.set(StyleAttribute::FontSize(36.0));

        let mut overlay = SubtitleStyleSet::new("overlay");
        overlay.set(StyleAttribute::FontSize(48.0));

        base.merge(&overlay);
        let attr = base.get("font-size").expect("should succeed in test");
        assert!(matches!(attr, StyleAttribute::FontSize(v) if (*v - 48.0).abs() < 0.01));
    }

    #[test]
    fn test_style_set_merge_adds_new_keys() {
        let mut base = SubtitleStyleSet::new("base");
        base.set(StyleAttribute::FontSize(36.0));

        let mut extra = SubtitleStyleSet::new("extra");
        extra.set(StyleAttribute::Bold(true));

        base.merge(&extra);
        assert_eq!(base.len(), 2);
    }

    #[test]
    fn test_style_set_visual_attributes() {
        let mut set = SubtitleStyleSet::new("test");
        set.set(StyleAttribute::FontFamily("Arial".into()));
        set.set(StyleAttribute::FontSize(48.0));
        set.set(StyleAttribute::Color(255, 255, 255, 255));
        // FontFamily and TextAlign are not visual; FontSize and Color are
        let visual = set.visual_attributes();
        assert_eq!(visual.len(), 2);
    }

    #[test]
    fn test_preset_library_add_and_find() {
        let mut lib = StylePresetLibrary::new();
        let style = SubtitleStyleSet::new("my-style");
        lib.add(style);
        assert!(lib.find("my-style").is_some());
        assert!(lib.find("nonexistent").is_none());
    }

    #[test]
    fn test_preset_library_with_defaults() {
        let lib = StylePresetLibrary::with_defaults();
        assert!(lib.contains("default"));
        assert!(lib.contains("hearing-impaired"));
        assert_eq!(lib.count(), 2);
    }

    #[test]
    fn test_preset_library_names() {
        let lib = StylePresetLibrary::with_defaults();
        let names = lib.names();
        assert!(names.contains(&"default"));
        assert!(names.contains(&"hearing-impaired"));
    }
}
