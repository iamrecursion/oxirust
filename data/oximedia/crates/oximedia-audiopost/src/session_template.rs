#![allow(dead_code)]
//! Session templates for audio post-production projects.
//!
//! A [`SessionTemplate`] defines a reusable track layout that can be
//! instantiated when creating a new mix session.  The
//! [`SessionTemplateLibrary`] acts as an in-memory registry.

// ---------------------------------------------------------------------------
// TemplateTrackType
// ---------------------------------------------------------------------------

/// Classifies the audio track type within a session template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateTrackType {
    /// Mono or stereo dialogue track.
    Dialogue,
    /// Mono or stereo music track.
    Music,
    /// Mono or stereo sound effects track.
    SoundEffects,
    /// Foley performance track.
    Foley,
    /// Ambience / room-tone track.
    Ambience,
    /// Narration / voice-over track.
    Narration,
    /// ADR (automated dialogue replacement) track.
    Adr,
    /// Multi-channel surround bus.
    SurroundBus,
    /// Stems mix-down bus.
    StemBus,
    /// Generic aux return track.
    AuxReturn,
}

impl TemplateTrackType {
    /// Returns `true` for track types that carry audio content (all types in
    /// this enum are audio, so this always returns `true`, but the method
    /// exists as a semantic predicate for forward-compatibility).
    #[must_use]
    pub fn is_audio(self) -> bool {
        true
    }

    /// Returns `true` for bus/routing tracks rather than content tracks.
    #[must_use]
    pub fn is_bus(self) -> bool {
        matches!(self, Self::SurroundBus | Self::StemBus | Self::AuxReturn)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Dialogue => "dialogue",
            Self::Music => "music",
            Self::SoundEffects => "sound_effects",
            Self::Foley => "foley",
            Self::Ambience => "ambience",
            Self::Narration => "narration",
            Self::Adr => "adr",
            Self::SurroundBus => "surround_bus",
            Self::StemBus => "stem_bus",
            Self::AuxReturn => "aux_return",
        }
    }
}

// ---------------------------------------------------------------------------
// TemplateTrack
// ---------------------------------------------------------------------------

/// A single track definition inside a [`SessionTemplate`].
#[derive(Debug, Clone)]
pub struct TemplateTrack {
    /// Display name of the track.
    pub name: String,
    /// Track type.
    pub track_type: TemplateTrackType,
    /// Number of audio channels (1 = mono, 2 = stereo, 6 = 5.1, etc.).
    pub channels: u8,
}

impl TemplateTrack {
    /// Create a new template track.
    #[must_use]
    pub fn new(name: impl Into<String>, track_type: TemplateTrackType, channels: u8) -> Self {
        Self {
            name: name.into(),
            track_type,
            channels: channels.max(1),
        }
    }
}

// ---------------------------------------------------------------------------
// SessionTemplate
// ---------------------------------------------------------------------------

/// A reusable layout of tracks that seeds a new mix session.
#[derive(Debug, Clone)]
pub struct SessionTemplate {
    /// Unique identifier.
    pub id: u64,
    /// Display name of the template.
    pub name: String,
    /// Sample rate in Hz this template was designed for.
    pub sample_rate: u32,
    /// Ordered list of track definitions.
    tracks: Vec<TemplateTrack>,
}

impl SessionTemplate {
    /// Create a new, empty template.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            id,
            name: name.into(),
            sample_rate,
            tracks: Vec::new(),
        }
    }

    /// Add a track definition.
    pub fn add_track(&mut self, track: TemplateTrack) {
        self.tracks.push(track);
    }

    /// Number of track definitions.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Iterate over track definitions.
    pub fn tracks(&self) -> impl Iterator<Item = &TemplateTrack> {
        self.tracks.iter()
    }

    /// Count tracks of a specific type.
    #[must_use]
    pub fn count_of_type(&self, kind: TemplateTrackType) -> usize {
        self.tracks.iter().filter(|t| t.track_type == kind).count()
    }
}

// ---------------------------------------------------------------------------
// SessionTemplateLibrary
// ---------------------------------------------------------------------------

/// In-memory registry of [`SessionTemplate`]s.
#[derive(Debug, Default)]
pub struct SessionTemplateLibrary {
    templates: Vec<SessionTemplate>,
    next_id: u64,
}

impl SessionTemplateLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pre-built template to the library.  The template's `id` field is
    /// replaced with a library-assigned ID.
    pub fn add_template(&mut self, mut template: SessionTemplate) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        template.id = id;
        self.templates.push(template);
        id
    }

    /// Find the first template whose name exactly matches `name`
    /// (case-insensitive).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&SessionTemplate> {
        let lower = name.to_lowercase();
        self.templates
            .iter()
            .find(|t| t.name.to_lowercase() == lower)
    }

    /// Find all templates whose name contains `substr` (case-insensitive).
    #[must_use]
    pub fn find_containing(&self, substr: &str) -> Vec<&SessionTemplate> {
        let lower = substr.to_lowercase();
        self.templates
            .iter()
            .filter(|t| t.name.to_lowercase().contains(&lower))
            .collect()
    }

    /// Total number of templates in the library.
    #[must_use]
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Iterate over all templates.
    pub fn iter(&self) -> impl Iterator<Item = &SessionTemplate> {
        self.templates.iter()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_basic_template(id: u64) -> SessionTemplate {
        let mut t = SessionTemplate::new(id, "Feature Film", 48_000);
        t.add_track(TemplateTrack::new("DX 1", TemplateTrackType::Dialogue, 1));
        t.add_track(TemplateTrack::new("DX 2", TemplateTrackType::Dialogue, 1));
        t.add_track(TemplateTrack::new("MX", TemplateTrackType::Music, 2));
        t.add_track(TemplateTrack::new("FX", TemplateTrackType::SoundEffects, 2));
        t
    }

    #[test]
    fn test_template_track_type_is_audio() {
        assert!(TemplateTrackType::Dialogue.is_audio());
        assert!(TemplateTrackType::SurroundBus.is_audio());
    }

    #[test]
    fn test_template_track_type_is_bus() {
        assert!(TemplateTrackType::SurroundBus.is_bus());
        assert!(TemplateTrackType::StemBus.is_bus());
        assert!(TemplateTrackType::AuxReturn.is_bus());
        assert!(!TemplateTrackType::Dialogue.is_bus());
    }

    #[test]
    fn test_template_track_type_labels() {
        assert_eq!(TemplateTrackType::Dialogue.label(), "dialogue");
        assert_eq!(TemplateTrackType::StemBus.label(), "stem_bus");
        assert_eq!(TemplateTrackType::Adr.label(), "adr");
    }

    #[test]
    fn test_template_track_count() {
        let t = make_basic_template(0);
        assert_eq!(t.track_count(), 4);
    }

    #[test]
    fn test_template_count_of_type() {
        let t = make_basic_template(0);
        assert_eq!(t.count_of_type(TemplateTrackType::Dialogue), 2);
        assert_eq!(t.count_of_type(TemplateTrackType::Music), 1);
        assert_eq!(t.count_of_type(TemplateTrackType::Foley), 0);
    }

    #[test]
    fn test_template_sample_rate() {
        let t = SessionTemplate::new(1, "Test", 44_100);
        assert_eq!(t.sample_rate, 44_100);
    }

    #[test]
    fn test_template_track_min_channel_one() {
        let track = TemplateTrack::new("mono", TemplateTrackType::Dialogue, 0);
        assert_eq!(track.channels, 1); // clamped to 1
    }

    #[test]
    fn test_library_add_and_count() {
        let mut lib = SessionTemplateLibrary::new();
        assert_eq!(lib.count(), 0);
        lib.add_template(make_basic_template(0));
        lib.add_template(make_basic_template(0));
        assert_eq!(lib.count(), 2);
    }

    #[test]
    fn test_library_find_by_name_exact() {
        let mut lib = SessionTemplateLibrary::new();
        lib.add_template(make_basic_template(0));
        let found = lib.find_by_name("Feature Film");
        assert!(found.is_some());
    }

    #[test]
    fn test_library_find_by_name_case_insensitive() {
        let mut lib = SessionTemplateLibrary::new();
        lib.add_template(make_basic_template(0));
        let found = lib.find_by_name("feature film");
        assert!(found.is_some());
    }

    #[test]
    fn test_library_find_by_name_not_found() {
        let mut lib = SessionTemplateLibrary::new();
        lib.add_template(make_basic_template(0));
        assert!(lib.find_by_name("Unknown Template").is_none());
    }

    #[test]
    fn test_library_find_containing() {
        let mut lib = SessionTemplateLibrary::new();
        lib.add_template(make_basic_template(0)); // "Feature Film"
        let mut advert = SessionTemplate::new(0, "TV Advert", 48_000);
        advert.add_track(TemplateTrack::new("DX", TemplateTrackType::Dialogue, 1));
        lib.add_template(advert);
        let results = lib.find_containing("feature");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_library_id_assignment() {
        let mut lib = SessionTemplateLibrary::new();
        let id0 = lib.add_template(make_basic_template(99));
        let id1 = lib.add_template(make_basic_template(99));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }
}
