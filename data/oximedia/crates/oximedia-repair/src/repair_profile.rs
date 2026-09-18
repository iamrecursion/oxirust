//! Repair configuration presets per container format.
//!
//! A `RepairProfile` bundles all `RepairOptions` configuration together with a
//! human-readable name and an optional container format tag.  Profiles can be
//! serialised to JSON and loaded back, making it easy to share settings across
//! runs and tools.
//!
//! Built-in profiles:
//! - `default` — balanced settings suitable for most files
//! - `mp4_safe` — conservative settings for MP4/MOV files
//! - `mkv_aggressive` — aggressive settings for Matroska/WebM
//! - `mpeg_extract` — extract mode for MPEG-TS / partial recordings

#![allow(dead_code)]

use crate::{IssueType, RepairMode};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Container format hint for a profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerFormat {
    /// ISO Base Media (MP4, M4V, MOV).
    Mp4,
    /// Matroska / WebM container.
    Matroska,
    /// AVI (Audio Video Interleave).
    Avi,
    /// MPEG Transport Stream.
    MpegTs,
    /// Any / unspecified format.
    Any,
}

impl std::fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mp4 => write!(f, "MP4"),
            Self::Matroska => write!(f, "Matroska/WebM"),
            Self::Avi => write!(f, "AVI"),
            Self::MpegTs => write!(f, "MPEG-TS"),
            Self::Any => write!(f, "Any"),
        }
    }
}

/// Serialisable version of `RepairMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerialRepairMode {
    /// Only fix obvious issues.
    Safe,
    /// Fix most issues.
    Balanced,
    /// Maximum recovery.
    Aggressive,
    /// Extract only playable portions.
    Extract,
    /// Custom per-issue-type aggressiveness.
    Custom,
}

impl From<RepairMode> for SerialRepairMode {
    fn from(m: RepairMode) -> Self {
        match m {
            RepairMode::Safe => Self::Safe,
            RepairMode::Balanced => Self::Balanced,
            RepairMode::Aggressive => Self::Aggressive,
            RepairMode::Extract => Self::Extract,
            RepairMode::Custom => Self::Custom,
        }
    }
}

impl From<SerialRepairMode> for RepairMode {
    fn from(m: SerialRepairMode) -> Self {
        match m {
            SerialRepairMode::Safe => Self::Safe,
            SerialRepairMode::Balanced => Self::Balanced,
            SerialRepairMode::Aggressive => Self::Aggressive,
            SerialRepairMode::Extract => Self::Extract,
            SerialRepairMode::Custom => Self::Custom,
        }
    }
}

/// Serialisable `IssueType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerialIssueType {
    /// Corrupted file header.
    CorruptedHeader,
    /// Missing container index/moov atom.
    MissingIndex,
    /// Invalid or out-of-order timestamps.
    InvalidTimestamps,
    /// Audio/video stream desynchronisation.
    AVDesync,
    /// Unexpectedly truncated file.
    Truncated,
    /// Corrupt packet data.
    CorruptPackets,
    /// Corrupt metadata fields.
    CorruptMetadata,
    /// Missing keyframes in video stream.
    MissingKeyframes,
    /// Invalid frame display order.
    InvalidFrameOrder,
    /// Conversion error during repair.
    ConversionError,
}

impl From<IssueType> for SerialIssueType {
    fn from(t: IssueType) -> Self {
        match t {
            IssueType::CorruptedHeader => Self::CorruptedHeader,
            IssueType::MissingIndex => Self::MissingIndex,
            IssueType::InvalidTimestamps => Self::InvalidTimestamps,
            IssueType::AVDesync => Self::AVDesync,
            IssueType::Truncated => Self::Truncated,
            IssueType::CorruptPackets => Self::CorruptPackets,
            IssueType::CorruptMetadata => Self::CorruptMetadata,
            IssueType::MissingKeyframes => Self::MissingKeyframes,
            IssueType::InvalidFrameOrder => Self::InvalidFrameOrder,
            IssueType::ConversionError => Self::ConversionError,
        }
    }
}

impl From<SerialIssueType> for IssueType {
    fn from(t: SerialIssueType) -> Self {
        match t {
            SerialIssueType::CorruptedHeader => Self::CorruptedHeader,
            SerialIssueType::MissingIndex => Self::MissingIndex,
            SerialIssueType::InvalidTimestamps => Self::InvalidTimestamps,
            SerialIssueType::AVDesync => Self::AVDesync,
            SerialIssueType::Truncated => Self::Truncated,
            SerialIssueType::CorruptPackets => Self::CorruptPackets,
            SerialIssueType::CorruptMetadata => Self::CorruptMetadata,
            SerialIssueType::MissingKeyframes => Self::MissingKeyframes,
            SerialIssueType::InvalidFrameOrder => Self::InvalidFrameOrder,
            SerialIssueType::ConversionError => Self::ConversionError,
        }
    }
}

/// A reusable repair configuration preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairProfile {
    /// Human-readable profile name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Target container format hint.
    pub container: ContainerFormat,
    /// Repair mode.
    pub mode: SerialRepairMode,
    /// Whether to create a backup before repairing.
    pub create_backup: bool,
    /// Whether to verify the output after repair.
    pub verify_after_repair: bool,
    /// Maximum file size in bytes to attempt repair (None = unlimited).
    pub max_file_size: Option<u64>,
    /// Only attempt to fix these issue types (empty = all issues).
    pub fix_issues: Vec<SerialIssueType>,
    /// Skip backup when file is larger than this threshold (bytes).
    pub skip_backup_threshold: Option<u64>,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl RepairProfile {
    /// Create a new profile with the given name and container format.
    pub fn new(name: impl Into<String>, container: ContainerFormat) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            container,
            mode: SerialRepairMode::Balanced,
            create_backup: true,
            verify_after_repair: true,
            max_file_size: None,
            fix_issues: Vec::new(),
            skip_backup_threshold: None,
            verbose: false,
        }
    }

    /// Convert this profile into `RepairOptions`.
    pub fn to_options(&self) -> crate::RepairOptions {
        crate::RepairOptions {
            mode: self.mode.into(),
            custom_config: None,
            create_backup: self.create_backup,
            verify_after_repair: self.verify_after_repair,
            output_dir: None,
            max_file_size: self.max_file_size,
            verbose: self.verbose,
            fix_issues: self
                .fix_issues
                .iter()
                .copied()
                .map(IssueType::from)
                .collect(),
            skip_backup_threshold: self.skip_backup_threshold,
            progress_callback: None,
        }
    }

    /// Serialise to JSON string.
    pub fn to_json(&self) -> crate::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::RepairError::RepairFailed(format!("JSON serialise error: {e}")))
    }

    /// Deserialise from JSON string.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| crate::RepairError::RepairFailed(format!("JSON deserialise error: {e}")))
    }

    /// Save profile to a JSON file.
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(crate::RepairError::Io)
    }

    /// Load a profile from a JSON file.
    pub fn load(path: &Path) -> crate::Result<Self> {
        let json = std::fs::read_to_string(path).map_err(crate::RepairError::Io)?;
        Self::from_json(&json)
    }
}

// ---------------------------------------------------------------------------
// Built-in profiles
// ---------------------------------------------------------------------------

/// Return the default balanced profile (works for any container).
pub fn default_profile() -> RepairProfile {
    let mut p = RepairProfile::new("default", ContainerFormat::Any);
    p.description = "Balanced settings suitable for most media files.".to_string();
    p
}

/// Conservative profile for MP4/MOV files.
///
/// Uses Safe mode to minimise the risk of introducing playback artefacts.
pub fn mp4_safe_profile() -> RepairProfile {
    let mut p = RepairProfile::new("mp4_safe", ContainerFormat::Mp4);
    p.description = "Conservative repair for MP4/MOV. Minimal data alteration.".to_string();
    p.mode = SerialRepairMode::Safe;
    p.create_backup = true;
    p.verify_after_repair = true;
    p
}

/// Aggressive profile for Matroska/WebM files.
///
/// Enables Aggressive mode and turns off backup for large files.
pub fn mkv_aggressive_profile() -> RepairProfile {
    let mut p = RepairProfile::new("mkv_aggressive", ContainerFormat::Matroska);
    p.description =
        "Aggressive repair for Matroska/WebM. May introduce minor artefacts.".to_string();
    p.mode = SerialRepairMode::Aggressive;
    p.create_backup = true;
    p.skip_backup_threshold = Some(2 * 1024 * 1024 * 1024); // skip backup for >2 GiB
    p
}

/// Extract mode profile for MPEG-TS (e.g. partial recordings from capture cards).
///
/// Only extracts playable segments; does not attempt structural repair.
pub fn mpeg_extract_profile() -> RepairProfile {
    let mut p = RepairProfile::new("mpeg_extract", ContainerFormat::MpegTs);
    p.description = "Extract playable segments from partial MPEG-TS recordings.".to_string();
    p.mode = SerialRepairMode::Extract;
    p.create_backup = false;
    p.verify_after_repair = false;
    p
}

/// Return all built-in profiles.
pub fn builtin_profiles() -> Vec<RepairProfile> {
    vec![
        default_profile(),
        mp4_safe_profile(),
        mkv_aggressive_profile(),
        mpeg_extract_profile(),
    ]
}

/// Look up a built-in profile by name.
pub fn find_builtin(name: &str) -> Option<RepairProfile> {
    builtin_profiles().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile_mode() {
        let p = default_profile();
        assert_eq!(p.mode, SerialRepairMode::Balanced);
        assert_eq!(p.container, ContainerFormat::Any);
    }

    #[test]
    fn test_mp4_safe_profile() {
        let p = mp4_safe_profile();
        assert_eq!(p.mode, SerialRepairMode::Safe);
        assert_eq!(p.container, ContainerFormat::Mp4);
        assert!(p.create_backup);
    }

    #[test]
    fn test_mkv_aggressive_profile() {
        let p = mkv_aggressive_profile();
        assert_eq!(p.mode, SerialRepairMode::Aggressive);
        assert_eq!(p.container, ContainerFormat::Matroska);
        assert!(p.skip_backup_threshold.is_some());
    }

    #[test]
    fn test_mpeg_extract_profile() {
        let p = mpeg_extract_profile();
        assert_eq!(p.mode, SerialRepairMode::Extract);
        assert!(!p.create_backup);
    }

    #[test]
    fn test_builtin_profiles_count() {
        assert_eq!(builtin_profiles().len(), 4);
    }

    #[test]
    fn test_find_builtin_found() {
        let p = find_builtin("default");
        assert!(p.is_some());
        assert_eq!(p.expect("default profile should exist").name, "default");
    }

    #[test]
    fn test_find_builtin_not_found() {
        let p = find_builtin("nonexistent_profile");
        assert!(p.is_none());
    }

    #[test]
    fn test_to_json_and_back() {
        let original = mp4_safe_profile();
        let json = original.to_json().expect("serialise");
        assert!(json.contains("mp4_safe"));
        assert!(json.contains("Safe"));

        let loaded = RepairProfile::from_json(&json).expect("deserialise");
        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.mode, original.mode);
        assert_eq!(loaded.container, original.container);
    }

    #[test]
    fn test_save_and_load_profile() {
        let path = std::env::temp_dir().join("oximedia_repair_profile_test.json");
        let profile = mkv_aggressive_profile();
        profile.save(&path).expect("save");

        let loaded = RepairProfile::load(&path).expect("load");
        assert_eq!(loaded.name, profile.name);
        assert_eq!(loaded.mode, profile.mode);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_to_options_mode_conversion() {
        let p = mkv_aggressive_profile();
        let opts = p.to_options();
        assert_eq!(opts.mode, RepairMode::Aggressive);
    }

    #[test]
    fn test_to_options_fix_issues_conversion() {
        let mut p = default_profile();
        p.fix_issues = vec![SerialIssueType::CorruptedHeader, SerialIssueType::Truncated];
        let opts = p.to_options();
        assert_eq!(opts.fix_issues.len(), 2);
        assert!(opts.fix_issues.contains(&IssueType::CorruptedHeader));
    }

    #[test]
    fn test_container_format_display() {
        assert_eq!(format!("{}", ContainerFormat::Mp4), "MP4");
        assert_eq!(format!("{}", ContainerFormat::Matroska), "Matroska/WebM");
        assert_eq!(format!("{}", ContainerFormat::Avi), "AVI");
        assert_eq!(format!("{}", ContainerFormat::MpegTs), "MPEG-TS");
        assert_eq!(format!("{}", ContainerFormat::Any), "Any");
    }

    #[test]
    fn test_profile_new_defaults() {
        let p = RepairProfile::new("test", ContainerFormat::Avi);
        assert!(p.fix_issues.is_empty());
        assert!(!p.verbose);
        assert!(p.create_backup);
    }

    #[test]
    fn test_issue_type_round_trip() {
        let types = [
            SerialIssueType::CorruptedHeader,
            SerialIssueType::MissingIndex,
            SerialIssueType::InvalidTimestamps,
            SerialIssueType::AVDesync,
            SerialIssueType::Truncated,
            SerialIssueType::CorruptPackets,
            SerialIssueType::CorruptMetadata,
            SerialIssueType::MissingKeyframes,
            SerialIssueType::InvalidFrameOrder,
            SerialIssueType::ConversionError,
        ];
        for t in types {
            let rt: SerialIssueType = IssueType::from(t).into();
            assert_eq!(rt, t);
        }
    }

    #[test]
    fn test_repair_mode_round_trip() {
        let modes = [
            SerialRepairMode::Safe,
            SerialRepairMode::Balanced,
            SerialRepairMode::Aggressive,
            SerialRepairMode::Extract,
            SerialRepairMode::Custom,
        ];
        for m in modes {
            let rt: SerialRepairMode = RepairMode::from(m).into();
            assert_eq!(rt, m);
        }
    }
}
