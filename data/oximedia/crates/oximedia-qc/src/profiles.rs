//! QC profile management and configuration.
//!
//! Provides reusable QC profiles for different use cases,
//! with support for custom profiles and profile persistence.

use crate::rules::Thresholds;
use std::collections::HashMap;

/// QC profile definition.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct QcProfile {
    /// Profile name.
    pub name: String,

    /// Profile description.
    pub description: String,

    /// Rule names to include.
    pub rules: Vec<String>,

    /// Threshold configuration.
    pub thresholds: Thresholds,

    /// Custom parameters.
    #[cfg_attr(feature = "json", serde(default))]
    pub parameters: HashMap<String, ProfileParameter>,
}

/// Profile parameter value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "json", serde(untagged))]
pub enum ProfileParameter {
    /// String value.
    String(String),
    /// Integer value.
    Integer(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Boolean(bool),
}

impl QcProfile {
    /// Creates a new QC profile.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            rules: Vec::new(),
            thresholds: Thresholds::default(),
            parameters: HashMap::new(),
        }
    }

    /// Adds a rule to the profile.
    #[must_use]
    pub fn with_rule(mut self, rule_name: impl Into<String>) -> Self {
        self.rules.push(rule_name.into());
        self
    }

    /// Sets the thresholds for the profile.
    #[must_use]
    pub const fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Adds a parameter to the profile.
    #[must_use]
    pub fn with_parameter(mut self, key: impl Into<String>, value: ProfileParameter) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Exports the profile as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "json")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Imports a profile from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    #[cfg(feature = "json")]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Profile manager for storing and retrieving QC profiles.
pub struct ProfileManager {
    profiles: HashMap<String, QcProfile>,
}

impl ProfileManager {
    /// Creates a new profile manager with built-in profiles.
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            profiles: HashMap::new(),
        };

        manager.load_builtin_profiles();
        manager
    }

    /// Loads built-in profiles.
    fn load_builtin_profiles(&mut self) {
        // Netflix delivery profile
        let netflix = QcProfile::new("netflix", "Netflix delivery specifications")
            .with_rule("video_codec_validation")
            .with_rule("resolution_validation")
            .with_rule("frame_rate_validation")
            .with_rule("bitrate_analysis")
            .with_rule("audio_codec_validation")
            .with_rule("loudness_compliance")
            .with_thresholds(
                Thresholds::new()
                    .with_min_video_bitrate(5_000_000)
                    .with_loudness_target(-27.0),
            );

        // Amazon Prime Video profile
        let amazon = QcProfile::new("amazon", "Amazon Prime Video specifications")
            .with_rule("video_codec_validation")
            .with_rule("resolution_validation")
            .with_rule("audio_codec_validation")
            .with_rule("loudness_compliance")
            .with_thresholds(Thresholds::new().with_loudness_target(-24.0));

        // Apple TV+ profile
        let apple = QcProfile::new("apple", "Apple TV+ delivery specifications")
            .with_rule("video_codec_validation")
            .with_rule("resolution_validation")
            .with_rule("frame_rate_validation")
            .with_rule("audio_codec_validation")
            .with_rule("loudness_compliance")
            .with_thresholds(Thresholds::new().with_loudness_target(-16.0));

        // BBC iPlayer profile
        let bbc = QcProfile::new("bbc", "BBC iPlayer specifications")
            .with_rule("video_codec_validation")
            .with_rule("resolution_validation")
            .with_rule("audio_codec_validation")
            .with_rule("loudness_compliance")
            .with_rule("broadcast_compliance")
            .with_thresholds(
                Thresholds::new().with_loudness_target(-23.0), // EBU R128
            );

        // DPP (Digital Production Partnership) profile
        let dpp = QcProfile::new("dpp", "UK DPP delivery specifications")
            .with_rule("video_codec_validation")
            .with_rule("resolution_validation")
            .with_rule("frame_rate_validation")
            .with_rule("interlacing_detection")
            .with_rule("audio_codec_validation")
            .with_rule("loudness_compliance")
            .with_rule("broadcast_compliance")
            .with_thresholds(Thresholds::new().with_loudness_target(-23.0));

        // Archive/Preservation profile
        let archive = QcProfile::new("archive", "Digital preservation and archive quality")
            .with_rule("video_codec_validation")
            .with_rule("audio_codec_validation")
            .with_rule("container_validation")
            .with_rule("metadata_validation")
            .with_rule("checksum_validation")
            .with_thresholds(Thresholds::new());

        self.profiles.insert("netflix".to_string(), netflix);
        self.profiles.insert("amazon".to_string(), amazon);
        self.profiles.insert("apple".to_string(), apple);
        self.profiles.insert("bbc".to_string(), bbc);
        self.profiles.insert("dpp".to_string(), dpp);
        self.profiles.insert("archive".to_string(), archive);
    }

    /// Adds a profile to the manager.
    pub fn add_profile(&mut self, profile: QcProfile) {
        let name = profile.name.clone();
        self.profiles.insert(name, profile);
    }

    /// Retrieves a profile by name.
    #[must_use]
    pub fn get_profile(&self, name: &str) -> Option<&QcProfile> {
        self.profiles.get(name)
    }

    /// Lists all available profile names.
    #[must_use]
    pub fn list_profiles(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }

    /// Removes a profile from the manager.
    pub fn remove_profile(&mut self, name: &str) -> Option<QcProfile> {
        self.profiles.remove(name)
    }

    /// Exports all profiles as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    #[cfg(feature = "json")]
    pub fn export_all(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.profiles)
    }

    /// Imports profiles from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    #[cfg(feature = "json")]
    pub fn import_profiles(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let profiles: HashMap<String, QcProfile> = serde_json::from_str(json)?;
        self.profiles.extend(profiles);
        Ok(())
    }
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_creation() {
        let profile = QcProfile::new("test", "Test profile")
            .with_rule("video_codec_validation")
            .with_parameter("max_bitrate", ProfileParameter::Integer(10_000_000));

        assert_eq!(profile.name, "test");
        assert_eq!(profile.rules.len(), 1);
        assert_eq!(profile.parameters.len(), 1);
    }

    #[test]
    fn test_profile_manager() {
        let manager = ProfileManager::new();
        let profiles = manager.list_profiles();
        assert!(!profiles.is_empty());
        assert!(manager.get_profile("netflix").is_some());
    }

    #[test]
    fn test_add_custom_profile() {
        let mut manager = ProfileManager::new();
        let profile = QcProfile::new("custom", "Custom profile");
        manager.add_profile(profile);

        assert!(manager.get_profile("custom").is_some());
    }

    #[test]
    fn test_remove_profile() {
        let mut manager = ProfileManager::new();
        let profile = QcProfile::new("temp", "Temporary profile");
        manager.add_profile(profile);

        let removed = manager.remove_profile("temp");
        assert!(removed.is_some());
        assert!(manager.get_profile("temp").is_none());
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_profile_json_export_import() {
        let profile = QcProfile::new("test", "Test profile").with_rule("test_rule");

        let json = profile.to_json().expect("should succeed in test");
        let imported = QcProfile::from_json(&json).expect("should succeed in test");

        assert_eq!(imported.name, profile.name);
        assert_eq!(imported.rules.len(), profile.rules.len());
    }
}
