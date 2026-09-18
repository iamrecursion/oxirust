//! Edit presets and templates for common editing patterns.
//!
//! Provides reusable editing templates such as montage, interview cut,
//! picture-in-picture, and split-screen layouts.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

/// Category of editing preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetCategory {
    /// Montage / highlight reel style.
    Montage,
    /// Interview / talking-head cut.
    Interview,
    /// Picture-in-picture layout.
    Pip,
    /// Split-screen layout.
    SplitScreen,
    /// Social media format (vertical, square, etc.).
    Social,
    /// Trailer / promo style.
    Trailer,
    /// Custom / user-defined.
    Custom,
}

/// A single track layout entry within a preset.
#[derive(Debug, Clone)]
pub struct TrackLayout {
    /// Track index.
    pub track_index: u32,
    /// Track label (e.g., "A-Roll", "B-Roll", "Music").
    pub label: String,
    /// Whether this is a video track (false = audio).
    pub is_video: bool,
    /// Default opacity (0.0 to 1.0, video only).
    pub opacity: f64,
    /// Position X offset (normalised 0.0..1.0, video only).
    pub x_offset: f64,
    /// Position Y offset (normalised 0.0..1.0, video only).
    pub y_offset: f64,
    /// Scale factor (1.0 = full size).
    pub scale: f64,
}

impl TrackLayout {
    /// Create a full-screen video track layout.
    #[must_use]
    pub fn video(track_index: u32, label: &str) -> Self {
        Self {
            track_index,
            label: label.to_string(),
            is_video: true,
            opacity: 1.0,
            x_offset: 0.0,
            y_offset: 0.0,
            scale: 1.0,
        }
    }

    /// Create an audio track layout.
    #[must_use]
    pub fn audio(track_index: u32, label: &str) -> Self {
        Self {
            track_index,
            label: label.to_string(),
            is_video: false,
            opacity: 1.0,
            x_offset: 0.0,
            y_offset: 0.0,
            scale: 1.0,
        }
    }

    /// Set position and scale for PIP / split-screen.
    #[must_use]
    pub fn with_transform(mut self, x: f64, y: f64, scale: f64) -> Self {
        self.x_offset = x;
        self.y_offset = y;
        self.scale = scale;
        self
    }

    /// Set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

/// Transition style to apply between clips in a preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetTransition {
    /// Hard cut (no transition).
    Cut,
    /// Cross-dissolve.
    Dissolve,
    /// Dip to black.
    DipToBlack,
    /// Wipe left-to-right.
    WipeLeft,
    /// Wipe right-to-left.
    WipeRight,
}

/// An editing preset / template.
#[derive(Debug, Clone)]
pub struct EditPreset {
    /// Unique name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Category.
    pub category: PresetCategory,
    /// Track layouts.
    pub tracks: Vec<TrackLayout>,
    /// Default transition between clips.
    pub default_transition: PresetTransition,
    /// Default transition duration in timebase units.
    pub transition_duration: u64,
    /// Custom metadata (key-value pairs).
    pub metadata: HashMap<String, String>,
}

impl EditPreset {
    /// Create a new preset.
    #[must_use]
    pub fn new(name: &str, category: PresetCategory) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            category,
            tracks: Vec::new(),
            default_transition: PresetTransition::Cut,
            transition_duration: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a track layout.
    #[must_use]
    pub fn with_track(mut self, layout: TrackLayout) -> Self {
        self.tracks.push(layout);
        self
    }

    /// Set default transition.
    #[must_use]
    pub fn with_transition(mut self, transition: PresetTransition, duration: u64) -> Self {
        self.default_transition = transition;
        self.transition_duration = duration;
        self
    }

    /// Add a metadata entry.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata entry.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Number of video tracks.
    #[must_use]
    pub fn video_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.is_video).count()
    }

    /// Number of audio tracks.
    #[must_use]
    pub fn audio_track_count(&self) -> usize {
        self.tracks.iter().filter(|t| !t.is_video).count()
    }
}

/// Library of editing presets.
#[derive(Debug, Clone, Default)]
pub struct PresetLibrary {
    /// Presets keyed by name.
    presets: HashMap<String, EditPreset>,
}

impl PresetLibrary {
    /// Create a new empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a preset.
    pub fn register(&mut self, preset: EditPreset) {
        self.presets.insert(preset.name.clone(), preset);
    }

    /// Get a preset by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&EditPreset> {
        self.presets.get(name)
    }

    /// Remove a preset.
    pub fn remove(&mut self, name: &str) -> Option<EditPreset> {
        self.presets.remove(name)
    }

    /// List all preset names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.presets.keys().map(String::as_str).collect()
    }

    /// List presets filtered by category.
    #[must_use]
    pub fn by_category(&self, category: PresetCategory) -> Vec<&EditPreset> {
        self.presets
            .values()
            .filter(|p| p.category == category)
            .collect()
    }

    /// Number of presets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Whether the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    /// Create a library pre-loaded with built-in presets.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut lib = Self::new();
        lib.register(builtin_montage());
        lib.register(builtin_interview());
        lib.register(builtin_pip());
        lib.register(builtin_split_screen());
        lib
    }
}

/// Built-in montage preset.
#[must_use]
fn builtin_montage() -> EditPreset {
    EditPreset::new("montage", PresetCategory::Montage)
        .with_description("Fast-paced montage with dissolves")
        .with_track(TrackLayout::video(0, "B-Roll"))
        .with_track(TrackLayout::audio(1, "Music"))
        .with_transition(PresetTransition::Dissolve, 500)
}

/// Built-in interview preset.
#[must_use]
fn builtin_interview() -> EditPreset {
    EditPreset::new("interview", PresetCategory::Interview)
        .with_description("Two-camera interview with hard cuts")
        .with_track(TrackLayout::video(0, "Cam A"))
        .with_track(TrackLayout::video(1, "Cam B"))
        .with_track(TrackLayout::audio(2, "Lav Mic"))
        .with_transition(PresetTransition::Cut, 0)
}

/// Built-in picture-in-picture preset.
#[must_use]
fn builtin_pip() -> EditPreset {
    EditPreset::new("pip", PresetCategory::Pip)
        .with_description("Full-screen main with small overlay")
        .with_track(TrackLayout::video(0, "Main"))
        .with_track(
            TrackLayout::video(1, "PIP")
                .with_transform(0.7, 0.7, 0.25)
                .with_opacity(0.95),
        )
        .with_track(TrackLayout::audio(2, "Audio"))
}

/// Built-in split-screen preset.
#[must_use]
fn builtin_split_screen() -> EditPreset {
    EditPreset::new("split_screen", PresetCategory::SplitScreen)
        .with_description("Side-by-side 50/50 split")
        .with_track(TrackLayout::video(0, "Left").with_transform(0.0, 0.0, 0.5))
        .with_track(TrackLayout::video(1, "Right").with_transform(0.5, 0.0, 0.5))
        .with_track(TrackLayout::audio(2, "Audio"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_layout_video() {
        let t = TrackLayout::video(0, "Main");
        assert!(t.is_video);
        assert!((t.opacity - 1.0).abs() < f64::EPSILON);
        assert!((t.scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_track_layout_audio() {
        let t = TrackLayout::audio(1, "Music");
        assert!(!t.is_video);
        assert_eq!(t.label, "Music");
    }

    #[test]
    fn test_track_layout_transform() {
        let t = TrackLayout::video(0, "PIP").with_transform(0.7, 0.7, 0.25);
        assert!((t.x_offset - 0.7).abs() < f64::EPSILON);
        assert!((t.scale - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_track_layout_opacity_clamped() {
        let t = TrackLayout::video(0, "V").with_opacity(1.5);
        assert!((t.opacity - 1.0).abs() < f64::EPSILON);
        let t2 = TrackLayout::video(0, "V").with_opacity(-0.5);
        assert!(t2.opacity.abs() < f64::EPSILON);
    }

    #[test]
    fn test_preset_new() {
        let p = EditPreset::new("test", PresetCategory::Custom);
        assert_eq!(p.name, "test");
        assert_eq!(p.category, PresetCategory::Custom);
        assert!(p.tracks.is_empty());
    }

    #[test]
    fn test_preset_builder() {
        let p = EditPreset::new("p", PresetCategory::Montage)
            .with_description("desc")
            .with_track(TrackLayout::video(0, "V"))
            .with_track(TrackLayout::audio(1, "A"))
            .with_transition(PresetTransition::Dissolve, 1000);
        assert_eq!(p.description, "desc");
        assert_eq!(p.video_track_count(), 1);
        assert_eq!(p.audio_track_count(), 1);
        assert_eq!(p.default_transition, PresetTransition::Dissolve);
        assert_eq!(p.transition_duration, 1000);
    }

    #[test]
    fn test_preset_metadata() {
        let mut p = EditPreset::new("p", PresetCategory::Social);
        p.set_metadata("platform", "instagram");
        assert_eq!(p.get_metadata("platform"), Some("instagram"));
        assert_eq!(p.get_metadata("missing"), None);
    }

    #[test]
    fn test_library_empty() {
        let lib = PresetLibrary::new();
        assert!(lib.is_empty());
        assert_eq!(lib.len(), 0);
    }

    #[test]
    fn test_library_register_get() {
        let mut lib = PresetLibrary::new();
        lib.register(EditPreset::new("my_preset", PresetCategory::Custom));
        assert_eq!(lib.len(), 1);
        assert!(lib.get("my_preset").is_some());
        assert!(lib.get("nonexistent").is_none());
    }

    #[test]
    fn test_library_remove() {
        let mut lib = PresetLibrary::new();
        lib.register(EditPreset::new("x", PresetCategory::Trailer));
        assert!(lib.remove("x").is_some());
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_by_category() {
        let mut lib = PresetLibrary::new();
        lib.register(EditPreset::new("a", PresetCategory::Montage));
        lib.register(EditPreset::new("b", PresetCategory::Interview));
        lib.register(EditPreset::new("c", PresetCategory::Montage));
        let montages = lib.by_category(PresetCategory::Montage);
        assert_eq!(montages.len(), 2);
    }

    #[test]
    fn test_library_builtins() {
        let lib = PresetLibrary::with_builtins();
        assert_eq!(lib.len(), 4);
        assert!(lib.get("montage").is_some());
        assert!(lib.get("interview").is_some());
        assert!(lib.get("pip").is_some());
        assert!(lib.get("split_screen").is_some());
    }

    #[test]
    fn test_builtin_montage() {
        let p = builtin_montage();
        assert_eq!(p.category, PresetCategory::Montage);
        assert_eq!(p.default_transition, PresetTransition::Dissolve);
        assert_eq!(p.video_track_count(), 1);
    }

    #[test]
    fn test_builtin_pip() {
        let p = builtin_pip();
        assert_eq!(p.video_track_count(), 2);
        let pip_track = &p.tracks[1];
        assert!((pip_track.scale - 0.25).abs() < f64::EPSILON);
    }
}
