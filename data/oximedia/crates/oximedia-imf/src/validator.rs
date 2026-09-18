//! IMF Package validation
//!
//! This module provides comprehensive validation for IMF packages,
//! checking SMPTE conformance, file integrity, and timeline consistency.

use crate::{
    AssetMap, CompositionPlaylist, ImfPackage, ImfResult, PackingList, Sequence, SequenceType,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Conformance level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConformanceLevel {
    /// SMPTE ST 2067-2 (IMF Core Constraints)
    ImfCore,
    /// SMPTE ST 2067-20 (Application #2)
    App2,
    /// SMPTE ST 2067-21 (Application #2 Extended)
    App2Extended,
    /// SMPTE ST 2067-30 (Application #3)
    App3,
    /// SMPTE ST 2067-40 (Application #4)
    App4,
    /// SMPTE ST 2067-50 (Application #5)
    App5,
    /// Custom/Unknown conformance
    Custom,
}

impl ConformanceLevel {
    /// Get conformance level name
    pub fn as_str(&self) -> &str {
        match self {
            Self::ImfCore => "IMF Core",
            Self::App2 => "Application #2",
            Self::App2Extended => "Application #2 Extended",
            Self::App3 => "Application #3",
            Self::App4 => "Application #4",
            Self::App5 => "Application #5",
            Self::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for ConformanceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Validation error severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Fatal error (package cannot be used)
    Error,
    /// Warning (package may work but has issues)
    Warning,
    /// Informational message
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    severity: Severity,
    category: String,
    message: String,
    location: Option<String>,
    suggestion: Option<String>,
}

impl ValidationError {
    /// Create a new validation error
    pub fn new(severity: Severity, category: String, message: String) -> Self {
        Self {
            severity,
            category,
            message,
            location: None,
            suggestion: None,
        }
    }

    /// Create an error
    pub fn error(category: String, message: String) -> Self {
        Self::new(Severity::Error, category, message)
    }

    /// Create a warning
    pub fn warning(category: String, message: String) -> Self {
        Self::new(Severity::Warning, category, message)
    }

    /// Create an info message
    pub fn info(category: String, message: String) -> Self {
        Self::new(Severity::Info, category, message)
    }

    /// Get severity
    pub fn severity(&self) -> Severity {
        self.severity
    }

    /// Get category
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Get message
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Get location
    pub fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }

    /// Set location
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    /// Get suggestion
    pub fn suggestion(&self) -> Option<&str> {
        self.suggestion.as_deref()
    }

    /// Set suggestion
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        let mut s = format!("[{}] {}: {}", self.severity, self.category, self.message);

        if let Some(ref location) = self.location {
            s.push_str(&format!(" (at {location})"));
        }

        if let Some(ref suggestion) = self.suggestion {
            s.push_str(&format!("\n  Suggestion: {suggestion}"));
        }

        s
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    errors: Vec<ValidationError>,
    conformance_level: Option<ConformanceLevel>,
    validated_at: chrono::DateTime<chrono::Utc>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            conformance_level: None,
            validated_at: chrono::Utc::now(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// Add multiple errors
    pub fn add_errors(&mut self, errors: Vec<ValidationError>) {
        self.errors.extend(errors);
    }

    /// Get errors
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == Severity::Warning)
            .count()
    }

    /// Get info count
    pub fn info_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == Severity::Info)
            .count()
    }

    /// Check if valid (no errors)
    pub fn is_valid(&self) -> bool {
        self.error_count() == 0
    }

    /// Get conformance level
    pub fn conformance_level(&self) -> Option<ConformanceLevel> {
        self.conformance_level
    }

    /// Set conformance level
    pub fn set_conformance_level(&mut self, level: ConformanceLevel) {
        self.conformance_level = Some(level);
    }

    /// Get validation timestamp
    pub fn validated_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.validated_at
    }

    /// Format report as string
    pub fn format(&self) -> String {
        let mut s = String::new();

        s.push_str("=== IMF Package Validation Report ===\n");
        s.push_str(&format!("Validated at: {}\n", self.validated_at));

        if let Some(level) = self.conformance_level {
            s.push_str(&format!("Conformance level: {level}\n"));
        }

        s.push('\n');
        s.push_str(&format!("Errors: {}\n", self.error_count()));
        s.push_str(&format!("Warnings: {}\n", self.warning_count()));
        s.push_str(&format!("Info: {}\n", self.info_count()));
        s.push('\n');

        if self.errors.is_empty() {
            s.push_str("No issues found.\n");
        } else {
            s.push_str("Issues:\n");
            for error in &self.errors {
                s.push_str(&format!("  {}\n", error.format()));
            }
        }

        s
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// IMF package validator
pub struct Validator {
    strict_mode: bool,
    check_hashes: bool,
    conformance_level: ConformanceLevel,
}

impl Validator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            check_hashes: true,
            conformance_level: ConformanceLevel::ImfCore,
        }
    }

    /// Enable strict mode
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Enable hash checking
    pub fn with_hash_checking(mut self, check: bool) -> Self {
        self.check_hashes = check;
        self
    }

    /// Set conformance level
    pub fn with_conformance_level(mut self, level: ConformanceLevel) -> Self {
        self.conformance_level = level;
        self
    }

    /// Validate a package
    pub fn validate(&self, package: &ImfPackage) -> ImfResult<ValidationReport> {
        let mut report = ValidationReport::new();
        report.set_conformance_level(self.conformance_level);

        // Validate ASSETMAP
        let assetmap_errors = self.validate_assetmap(package.asset_map(), package.root_path())?;
        report.add_errors(assetmap_errors);

        // Validate PKLs
        for pkl in package.packing_lists() {
            let pkl_errors = self.validate_pkl(pkl, package)?;
            report.add_errors(pkl_errors);
        }

        // Validate CPLs
        for cpl in package.composition_playlists() {
            let cpl_errors = self.validate_cpl(cpl, package)?;
            report.add_errors(cpl_errors);
        }

        // Validate PKL/ASSETMAP consistency
        let consistency_errors = self.validate_consistency(package)?;
        report.add_errors(consistency_errors);

        // Validate timelines
        for cpl in package.composition_playlists() {
            let timeline_errors = self.validate_timeline(cpl)?;
            report.add_errors(timeline_errors);
        }

        Ok(report)
    }

    /// Validate ASSETMAP
    fn validate_assetmap(
        &self,
        assetmap: &AssetMap,
        root_path: &Path,
    ) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check for required fields
        if assetmap.creator().is_none() && self.strict_mode {
            errors.push(ValidationError::warning(
                "ASSETMAP".to_string(),
                "Creator field is empty".to_string(),
            ));
        }

        // Check volume count
        if assetmap.volume_count() == 0 {
            errors.push(ValidationError::error(
                "ASSETMAP".to_string(),
                "VolumeCount must be at least 1".to_string(),
            ));
        }

        // Check assets
        if assetmap.assets().is_empty() {
            errors.push(ValidationError::error(
                "ASSETMAP".to_string(),
                "No assets found in ASSETMAP".to_string(),
            ));
        }

        // Check for at least one PKL
        if assetmap.packing_lists().is_empty() {
            errors.push(ValidationError::error(
                "ASSETMAP".to_string(),
                "No packing lists found in ASSETMAP".to_string(),
            ));
        }

        // Validate file paths
        for asset in assetmap.assets() {
            if let Some(path) = asset.primary_path() {
                let full_path = root_path.join(path);
                if !full_path.exists() {
                    errors.push(
                        ValidationError::error(
                            "ASSETMAP".to_string(),
                            format!("Asset file not found: {}", path.display()),
                        )
                        .with_location(format!("Asset ID: {}", asset.id())),
                    );
                }
            } else {
                errors.push(
                    ValidationError::error(
                        "ASSETMAP".to_string(),
                        "Asset has no file path".to_string(),
                    )
                    .with_location(format!("Asset ID: {}", asset.id())),
                );
            }
        }

        Ok(errors)
    }

    /// Validate PKL
    fn validate_pkl(
        &self,
        pkl: &PackingList,
        package: &ImfPackage,
    ) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check for required fields
        if pkl.creator().is_none() && self.strict_mode {
            errors.push(ValidationError::warning(
                "PKL".to_string(),
                "Creator field is empty".to_string(),
            ));
        }

        // Check assets
        if pkl.assets().is_empty() {
            errors.push(ValidationError::error(
                "PKL".to_string(),
                "No assets found in PKL".to_string(),
            ));
        }

        // Validate hashes
        if self.check_hashes {
            for asset in pkl.assets() {
                if let Some(am_asset) = package.asset_map().find_asset(asset.id()) {
                    if let Some(path) = am_asset.primary_path() {
                        let full_path = package.root_path().join(path);
                        if full_path.exists() {
                            match asset.verify(&full_path) {
                                Ok(true) => {}
                                Ok(false) => {
                                    errors.push(
                                        ValidationError::error(
                                            "PKL".to_string(),
                                            format!(
                                                "Hash verification failed for {}",
                                                path.display()
                                            ),
                                        )
                                        .with_location(format!("Asset ID: {}", asset.id())),
                                    );
                                }
                                Err(e) => {
                                    errors.push(
                                        ValidationError::error(
                                            "PKL".to_string(),
                                            format!("Hash verification error: {e}"),
                                        )
                                        .with_location(format!("Asset ID: {}", asset.id())),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(errors)
    }

    /// Validate CPL
    fn validate_cpl(
        &self,
        cpl: &CompositionPlaylist,
        _package: &ImfPackage,
    ) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check for required fields
        if cpl.content_title().is_empty() {
            errors.push(ValidationError::error(
                "CPL".to_string(),
                "ContentTitle is empty".to_string(),
            ));
        }

        if cpl.creator().is_none() && self.strict_mode {
            errors.push(ValidationError::warning(
                "CPL".to_string(),
                "Creator field is empty".to_string(),
            ));
        }

        // Check segments
        if cpl.segments().is_empty() {
            errors.push(ValidationError::error(
                "CPL".to_string(),
                "No segments found in CPL".to_string(),
            ));
        }

        // Check edit rate
        let edit_rate = cpl.edit_rate();
        if edit_rate.numerator() == 0 || edit_rate.denominator() == 0 {
            errors.push(ValidationError::error(
                "CPL".to_string(),
                "Invalid edit rate".to_string(),
            ));
        }

        // Validate sequences
        for segment in cpl.segments() {
            for sequence in segment.sequences() {
                let seq_errors = self.validate_sequence(sequence, cpl)?;
                errors.extend(seq_errors);
            }
        }

        Ok(errors)
    }

    /// Validate sequence
    fn validate_sequence(
        &self,
        sequence: &Sequence,
        cpl: &CompositionPlaylist,
    ) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check resources
        if sequence.resources().is_empty() && sequence.sequence_type() != SequenceType::Marker {
            errors.push(
                ValidationError::warning(
                    "CPL".to_string(),
                    format!("Sequence {} has no resources", sequence.id()),
                )
                .with_location(format!("Sequence type: {:?}", sequence.sequence_type())),
            );
        }

        // Validate resource edit rates match CPL edit rate
        for resource in sequence.resources() {
            if resource.edit_rate() != cpl.edit_rate() && self.strict_mode {
                errors.push(
                    ValidationError::warning(
                        "CPL".to_string(),
                        format!(
                            "Resource edit rate ({}) differs from CPL edit rate ({})",
                            resource.edit_rate(),
                            cpl.edit_rate()
                        ),
                    )
                    .with_location(format!("Resource ID: {}", resource.id())),
                );
            }
        }

        Ok(errors)
    }

    /// Validate PKL/ASSETMAP consistency
    fn validate_consistency(&self, package: &ImfPackage) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        let assetmap = package.asset_map();
        let mut all_pkl_assets = HashSet::new();

        // Collect all PKL assets
        for pkl in package.packing_lists() {
            for asset in pkl.assets() {
                all_pkl_assets.insert(asset.id());
            }
        }

        // Check that all PKL assets are in ASSETMAP
        for asset_id in &all_pkl_assets {
            if assetmap.find_asset(*asset_id).is_none() {
                errors.push(
                    ValidationError::error(
                        "Consistency".to_string(),
                        "Asset in PKL not found in ASSETMAP".to_string(),
                    )
                    .with_location(format!("Asset ID: {asset_id}")),
                );
            }
        }

        // Check that all ASSETMAP assets (except PKL itself) are in some PKL
        for am_asset in assetmap.assets() {
            if !am_asset.is_packing_list() && !all_pkl_assets.contains(&am_asset.id()) {
                errors.push(
                    ValidationError::warning(
                        "Consistency".to_string(),
                        "Asset in ASSETMAP not found in any PKL".to_string(),
                    )
                    .with_location(format!("Asset ID: {}", am_asset.id())),
                );
            }
        }

        Ok(errors)
    }

    /// Validate timeline continuity
    fn validate_timeline(&self, cpl: &CompositionPlaylist) -> ImfResult<Vec<ValidationError>> {
        let mut errors = Vec::new();

        for segment in cpl.segments() {
            // Group sequences by type
            let mut sequences_by_type: HashMap<SequenceType, Vec<&Sequence>> = HashMap::new();

            for sequence in segment.sequences() {
                sequences_by_type
                    .entry(sequence.sequence_type())
                    .or_default()
                    .push(sequence);
            }

            // Validate that all video sequences have the same duration
            if let Some(video_sequences) = sequences_by_type.get(&SequenceType::MainImage) {
                let durations: Vec<u64> =
                    video_sequences.iter().map(|s| s.total_duration()).collect();

                if durations.len() > 1 {
                    let first_duration = durations[0];
                    for (i, &duration) in durations.iter().enumerate().skip(1) {
                        if duration != first_duration {
                            errors.push(
                                ValidationError::error(
                                    "Timeline".to_string(),
                                    format!(
                                        "Video sequence duration mismatch: {first_duration} vs {duration}"
                                    ),
                                )
                                .with_location(format!(
                                    "Segment {}, Sequence {}",
                                    segment.id(),
                                    video_sequences[i].id()
                                )),
                            );
                        }
                    }
                }
            }

            // Validate that audio sequences match video duration
            if let (Some(video_sequences), Some(audio_sequences)) = (
                sequences_by_type.get(&SequenceType::MainImage),
                sequences_by_type.get(&SequenceType::MainAudio),
            ) {
                if let Some(video_seq) = video_sequences.first() {
                    let video_duration = video_seq.total_duration();

                    for audio_seq in audio_sequences {
                        let audio_duration = audio_seq.total_duration();
                        if audio_duration != video_duration && self.strict_mode {
                            errors.push(
                                ValidationError::warning(
                                    "Timeline".to_string(),
                                    format!(
                                        "Audio duration ({audio_duration}) differs from video duration ({video_duration})"
                                    ),
                                )
                                .with_location(format!(
                                    "Segment {}, Audio Sequence {}",
                                    segment.id(),
                                    audio_seq.id()
                                )),
                            );
                        }
                    }
                }
            }
        }

        Ok(errors)
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error() {
        let error = ValidationError::error("Test".to_string(), "Test message".to_string())
            .with_location("test.xml".to_string())
            .with_suggestion("Fix the issue".to_string());

        assert_eq!(error.severity(), Severity::Error);
        assert_eq!(error.category(), "Test");
        assert_eq!(error.message(), "Test message");
        assert_eq!(error.location(), Some("test.xml"));
        assert_eq!(error.suggestion(), Some("Fix the issue"));
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();

        report.add_error(ValidationError::error(
            "Test".to_string(),
            "Error 1".to_string(),
        ));
        report.add_error(ValidationError::warning(
            "Test".to_string(),
            "Warning 1".to_string(),
        ));
        report.add_error(ValidationError::info(
            "Test".to_string(),
            "Info 1".to_string(),
        ));

        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);
        assert_eq!(report.info_count(), 1);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_conformance_level() {
        assert_eq!(ConformanceLevel::ImfCore.as_str(), "IMF Core");
        assert_eq!(ConformanceLevel::App2.as_str(), "Application #2");
    }

    #[test]
    fn test_validator_creation() {
        let validator = Validator::new()
            .with_strict_mode(true)
            .with_hash_checking(true)
            .with_conformance_level(ConformanceLevel::App2);

        assert!(validator.strict_mode);
        assert!(validator.check_hashes);
        assert_eq!(validator.conformance_level, ConformanceLevel::App2);
    }
}
