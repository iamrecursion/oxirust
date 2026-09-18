//! QC profile definitions for different delivery contexts.
//!
//! Provides `QcProfileType`, `QcProfile`, and `QcProfileLibrary` for
//! configuring quality control checks based on delivery target.

#![allow(dead_code)]

use std::collections::HashMap;

/// Type of QC profile, representing the delivery target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QcProfileType {
    /// Broadcast television delivery.
    Broadcast,
    /// Online streaming platform delivery.
    Streaming,
    /// Long-term archive preservation.
    Archive,
    /// Theatrical/cinema release.
    Theatrical,
}

impl QcProfileType {
    /// Returns a numeric strictness level (1 = lenient, 4 = strictest).
    pub fn strictness_level(self) -> u8 {
        match self {
            QcProfileType::Streaming => 1,
            QcProfileType::Archive => 2,
            QcProfileType::Broadcast => 3,
            QcProfileType::Theatrical => 4,
        }
    }

    /// Returns a human-readable name for the profile type.
    pub fn name(self) -> &'static str {
        match self {
            QcProfileType::Broadcast => "Broadcast",
            QcProfileType::Streaming => "Streaming",
            QcProfileType::Archive => "Archive",
            QcProfileType::Theatrical => "Theatrical",
        }
    }
}

/// Tolerances used when evaluating a particular QC check.
#[derive(Debug, Clone)]
pub struct QcTolerance {
    /// Maximum allowed deviation (e.g. loudness in LUFS).
    pub max_deviation: f64,
    /// Whether exceeding the tolerance causes a fatal failure.
    pub is_fatal: bool,
}

impl QcTolerance {
    /// Creates a new tolerance specification.
    pub fn new(max_deviation: f64, is_fatal: bool) -> Self {
        Self {
            max_deviation,
            is_fatal,
        }
    }
}

/// A QC profile describing tolerances for a specific delivery target.
#[derive(Debug, Clone)]
pub struct QcProfile {
    /// Type of this profile.
    pub profile_type: QcProfileType,
    /// Name of the profile.
    pub name: String,
    /// Per-check tolerances, keyed by check name.
    tolerances: HashMap<String, QcTolerance>,
}

impl QcProfile {
    /// Creates a new `QcProfile` for the given type.
    pub fn new(profile_type: QcProfileType, name: impl Into<String>) -> Self {
        Self {
            profile_type,
            name: name.into(),
            tolerances: HashMap::new(),
        }
    }

    /// Adds a tolerance entry for a named check.
    pub fn set_tolerance(&mut self, check: impl Into<String>, tolerance: QcTolerance) {
        self.tolerances.insert(check.into(), tolerance);
    }

    /// Returns the tolerance for a specific check, if defined.
    pub fn tolerance_for(&self, check: &str) -> Option<&QcTolerance> {
        self.tolerances.get(check)
    }

    /// Returns the strictness level of this profile.
    pub fn strictness_level(&self) -> u8 {
        self.profile_type.strictness_level()
    }

    /// Returns `true` if this profile is stricter than another.
    pub fn is_stricter_than(&self, other: &QcProfile) -> bool {
        self.strictness_level() > other.strictness_level()
    }
}

/// A library of named QC profiles.
#[derive(Debug, Default)]
pub struct QcProfileLibrary {
    profiles: HashMap<String, QcProfile>,
}

impl QcProfileLibrary {
    /// Creates an empty profile library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a library pre-loaded with default profiles.
    pub fn with_defaults() -> Self {
        let mut lib = Self::new();

        let mut broadcast = QcProfile::new(QcProfileType::Broadcast, "Default Broadcast");
        broadcast.set_tolerance("loudness", QcTolerance::new(1.0, true));
        broadcast.set_tolerance("black_frames", QcTolerance::new(0.5, false));
        lib.add_profile(broadcast);

        let mut streaming = QcProfile::new(QcProfileType::Streaming, "Default Streaming");
        streaming.set_tolerance("loudness", QcTolerance::new(2.0, false));
        streaming.set_tolerance("black_frames", QcTolerance::new(2.0, false));
        lib.add_profile(streaming);

        let mut archive = QcProfile::new(QcProfileType::Archive, "Default Archive");
        archive.set_tolerance("loudness", QcTolerance::new(1.5, false));
        lib.add_profile(archive);

        let mut theatrical = QcProfile::new(QcProfileType::Theatrical, "Default Theatrical");
        theatrical.set_tolerance("loudness", QcTolerance::new(0.5, true));
        theatrical.set_tolerance("black_frames", QcTolerance::new(0.1, true));
        lib.add_profile(theatrical);

        lib
    }

    /// Adds a profile to the library. Key is the profile's name.
    pub fn add_profile(&mut self, profile: QcProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Retrieves a profile by name.
    pub fn get_profile(&self, name: &str) -> Option<&QcProfile> {
        self.profiles.get(name)
    }

    /// Returns the number of profiles in the library.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns `true` if the library contains no profiles.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Returns all profiles sorted by strictness level (ascending).
    pub fn profiles_by_strictness(&self) -> Vec<&QcProfile> {
        let mut v: Vec<&QcProfile> = self.profiles.values().collect();
        v.sort_by_key(|p| p.strictness_level());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strictness_levels_ordered() {
        assert!(
            QcProfileType::Streaming.strictness_level() < QcProfileType::Archive.strictness_level()
        );
        assert!(
            QcProfileType::Archive.strictness_level() < QcProfileType::Broadcast.strictness_level()
        );
        assert!(
            QcProfileType::Broadcast.strictness_level()
                < QcProfileType::Theatrical.strictness_level()
        );
    }

    #[test]
    fn test_theatrical_is_strictest() {
        assert_eq!(QcProfileType::Theatrical.strictness_level(), 4);
    }

    #[test]
    fn test_streaming_is_most_lenient() {
        assert_eq!(QcProfileType::Streaming.strictness_level(), 1);
    }

    #[test]
    fn test_profile_type_names() {
        assert_eq!(QcProfileType::Broadcast.name(), "Broadcast");
        assert_eq!(QcProfileType::Streaming.name(), "Streaming");
        assert_eq!(QcProfileType::Archive.name(), "Archive");
        assert_eq!(QcProfileType::Theatrical.name(), "Theatrical");
    }

    #[test]
    fn test_tolerance_creation() {
        let t = QcTolerance::new(1.5, true);
        assert!((t.max_deviation - 1.5).abs() < f64::EPSILON);
        assert!(t.is_fatal);
    }

    #[test]
    fn test_profile_set_and_get_tolerance() {
        let mut profile = QcProfile::new(QcProfileType::Broadcast, "Test");
        profile.set_tolerance("loudness", QcTolerance::new(1.0, true));
        let t = profile
            .tolerance_for("loudness")
            .expect("should succeed in test");
        assert!((t.max_deviation - 1.0).abs() < f64::EPSILON);
        assert!(t.is_fatal);
    }

    #[test]
    fn test_profile_missing_tolerance_returns_none() {
        let profile = QcProfile::new(QcProfileType::Streaming, "Test");
        assert!(profile.tolerance_for("nonexistent").is_none());
    }

    #[test]
    fn test_profile_strictness_delegates_to_type() {
        let p = QcProfile::new(QcProfileType::Theatrical, "T");
        assert_eq!(p.strictness_level(), 4);
    }

    #[test]
    fn test_is_stricter_than() {
        let broadcast = QcProfile::new(QcProfileType::Broadcast, "B");
        let streaming = QcProfile::new(QcProfileType::Streaming, "S");
        assert!(broadcast.is_stricter_than(&streaming));
        assert!(!streaming.is_stricter_than(&broadcast));
    }

    #[test]
    fn test_library_add_and_get() {
        let mut lib = QcProfileLibrary::new();
        let profile = QcProfile::new(QcProfileType::Broadcast, "MyBroadcast");
        lib.add_profile(profile);
        assert!(lib.get_profile("MyBroadcast").is_some());
        assert!(lib.get_profile("Unknown").is_none());
    }

    #[test]
    fn test_library_len() {
        let lib = QcProfileLibrary::with_defaults();
        assert_eq!(lib.len(), 4);
        assert!(!lib.is_empty());
    }

    #[test]
    fn test_library_empty() {
        let lib = QcProfileLibrary::new();
        assert!(lib.is_empty());
        assert_eq!(lib.len(), 0);
    }

    #[test]
    fn test_profiles_by_strictness_sorted() {
        let lib = QcProfileLibrary::with_defaults();
        let sorted = lib.profiles_by_strictness();
        for w in sorted.windows(2) {
            assert!(w[0].strictness_level() <= w[1].strictness_level());
        }
    }

    #[test]
    fn test_default_broadcast_profile_has_loudness() {
        let lib = QcProfileLibrary::with_defaults();
        let p = lib
            .get_profile("Default Broadcast")
            .expect("should succeed in test");
        assert!(p.tolerance_for("loudness").is_some());
    }

    #[test]
    fn test_default_theatrical_loudness_is_fatal() {
        let lib = QcProfileLibrary::with_defaults();
        let p = lib
            .get_profile("Default Theatrical")
            .expect("should succeed in test");
        let t = p.tolerance_for("loudness").expect("should succeed in test");
        assert!(t.is_fatal);
    }
}
