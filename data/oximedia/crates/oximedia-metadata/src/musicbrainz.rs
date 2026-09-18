//! MusicBrainz metadata tag mappings and MBID validation.
//!
//! [MusicBrainz](https://musicbrainz.org/) is an open music encyclopedia that
//! uses UUIDs (MBIDs - MusicBrainz Identifiers) to uniquely identify musical
//! entities: recordings, releases, artists, release groups, works, etc.
//!
//! This module provides:
//!
//! - **MBID validation** (UUID format checking per MusicBrainz spec)
//! - **Tag key mappings** across ID3v2, Vorbis Comments, APEv2, and iTunes
//! - **Typed containers** for MusicBrainz metadata with builder pattern
//! - **Metadata integration** (read/write from/to `Metadata` containers)
//!
//! # Tag Mappings
//!
//! | Entity             | Vorbis Comment                     | ID3v2 TXXX                        |
//! |--------------------|------------------------------------|-----------------------------------|
//! | Recording          | `MUSICBRAINZ_TRACKID`              | `MusicBrainz Release Track Id`    |
//! | Release            | `MUSICBRAINZ_ALBUMID`              | `MusicBrainz Album Id`            |
//! | Artist             | `MUSICBRAINZ_ARTISTID`             | `MusicBrainz Artist Id`           |
//! | Release Group      | `MUSICBRAINZ_RELEASEGROUPID`       | `MusicBrainz Release Group Id`    |
//! | Work               | `MUSICBRAINZ_WORKID`               | `MusicBrainz Work Id`             |
//! | Album Artist       | `MUSICBRAINZ_ALBUMARTISTID`        | `MusicBrainz Album Artist Id`     |
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::musicbrainz::{MusicBrainzTags, validate_mbid};
//!
//! // Validate an MBID
//! assert!(validate_mbid("f27ec8db-af05-4f36-916e-3c57f9e4e1b3"));
//! assert!(!validate_mbid("not-a-valid-uuid"));
//!
//! // Build MusicBrainz tags
//! let tags = MusicBrainzTags::new()
//!     .with_recording_id("f27ec8db-af05-4f36-916e-3c57f9e4e1b3")
//!     .with_release_id("a1b2c3d4-e5f6-7890-abcd-ef1234567890");
//!
//! assert!(tags.recording_id().is_some());
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};

/// Validate a MusicBrainz Identifier (MBID).
///
/// MBIDs are standard UUID v4 strings in the format:
/// `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` where x is a hex digit.
///
/// Returns true if the string is a valid UUID format.
pub fn validate_mbid(mbid: &str) -> bool {
    let trimmed = mbid.trim();
    if trimmed.len() != 36 {
        return false;
    }

    let parts: Vec<&str> = trimmed.split('-').collect();
    if parts.len() != 5 {
        return false;
    }

    let expected_lengths = [8, 4, 4, 4, 12];
    for (part, expected_len) in parts.iter().zip(expected_lengths.iter()) {
        if part.len() != *expected_len {
            return false;
        }
        if !part.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
    }

    true
}

/// Normalize an MBID to lowercase.
pub fn normalize_mbid(mbid: &str) -> Option<String> {
    let trimmed = mbid.trim();
    if validate_mbid(trimmed) {
        Some(trimmed.to_lowercase())
    } else {
        None
    }
}

/// Construct a MusicBrainz URL for an entity.
///
/// # Errors
///
/// Returns an error if the MBID is invalid.
pub fn mbid_url(entity_type: MbEntityType, mbid: &str) -> Result<String, Error> {
    if !validate_mbid(mbid) {
        return Err(Error::ParseError(format!("Invalid MBID: '{mbid}'")));
    }
    let normalized = mbid.trim().to_lowercase();
    let type_str = entity_type.url_path();
    Ok(format!("https://musicbrainz.org/{type_str}/{normalized}"))
}

/// MusicBrainz entity types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MbEntityType {
    /// A specific audio recording.
    Recording,
    /// A release (album, single, EP, etc.).
    Release,
    /// A release group (collection of releases).
    ReleaseGroup,
    /// An artist (person or group).
    Artist,
    /// A musical work (composition).
    Work,
    /// A label (record label).
    Label,
}

impl MbEntityType {
    /// URL path component for this entity type.
    pub fn url_path(self) -> &'static str {
        match self {
            Self::Recording => "recording",
            Self::Release => "release",
            Self::ReleaseGroup => "release-group",
            Self::Artist => "artist",
            Self::Work => "work",
            Self::Label => "label",
        }
    }
}

impl std::fmt::Display for MbEntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recording => write!(f, "Recording"),
            Self::Release => write!(f, "Release"),
            Self::ReleaseGroup => write!(f, "Release Group"),
            Self::Artist => write!(f, "Artist"),
            Self::Work => write!(f, "Work"),
            Self::Label => write!(f, "Label"),
        }
    }
}

/// Tag key mapping for a specific metadata format.
#[derive(Debug, Clone)]
pub struct MbTagKeys {
    /// Recording MBID key.
    pub recording: &'static str,
    /// Release MBID key.
    pub release: &'static str,
    /// Release group MBID key.
    pub release_group: &'static str,
    /// Artist MBID key.
    pub artist: &'static str,
    /// Album artist MBID key.
    pub album_artist: &'static str,
    /// Work MBID key.
    pub work: &'static str,
    /// Label MBID key (if supported).
    pub label: Option<&'static str>,
    /// Track ID (disc-specific track, different from recording).
    pub track: Option<&'static str>,
}

/// Get MusicBrainz tag keys for Vorbis Comments.
pub fn vorbis_tag_keys() -> MbTagKeys {
    MbTagKeys {
        recording: "MUSICBRAINZ_TRACKID",
        release: "MUSICBRAINZ_ALBUMID",
        release_group: "MUSICBRAINZ_RELEASEGROUPID",
        artist: "MUSICBRAINZ_ARTISTID",
        album_artist: "MUSICBRAINZ_ALBUMARTISTID",
        work: "MUSICBRAINZ_WORKID",
        label: Some("MUSICBRAINZ_LABELID"),
        track: Some("MUSICBRAINZ_RELEASETRACKID"),
    }
}

/// Get MusicBrainz tag keys for ID3v2 TXXX frames.
pub fn id3v2_tag_keys() -> MbTagKeys {
    MbTagKeys {
        recording: "MusicBrainz Release Track Id",
        release: "MusicBrainz Album Id",
        release_group: "MusicBrainz Release Group Id",
        artist: "MusicBrainz Artist Id",
        album_artist: "MusicBrainz Album Artist Id",
        work: "MusicBrainz Work Id",
        label: Some("MusicBrainz Label Id"),
        track: None,
    }
}

/// Get MusicBrainz tag keys for APEv2.
pub fn apev2_tag_keys() -> MbTagKeys {
    MbTagKeys {
        recording: "MUSICBRAINZ_TRACKID",
        release: "MUSICBRAINZ_ALBUMID",
        release_group: "MUSICBRAINZ_RELEASEGROUPID",
        artist: "MUSICBRAINZ_ARTISTID",
        album_artist: "MUSICBRAINZ_ALBUMARTISTID",
        work: "MUSICBRAINZ_WORKID",
        label: None,
        track: None,
    }
}

/// Get the appropriate tag keys for a metadata format.
pub fn tag_keys_for_format(format: MetadataFormat) -> MbTagKeys {
    match format {
        MetadataFormat::Id3v2 => id3v2_tag_keys(),
        MetadataFormat::Apev2 => apev2_tag_keys(),
        // Vorbis-style keys are used for most other formats
        _ => vorbis_tag_keys(),
    }
}

/// Container for MusicBrainz tag values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MusicBrainzTags {
    /// Recording MBID.
    recording_id: Option<String>,
    /// Release (album) MBID.
    release_id: Option<String>,
    /// Release group MBID.
    release_group_id: Option<String>,
    /// Artist MBID (may be multiple, semicolon-separated).
    artist_id: Option<String>,
    /// Album artist MBID.
    album_artist_id: Option<String>,
    /// Work MBID.
    work_id: Option<String>,
    /// Label MBID.
    label_id: Option<String>,
    /// Release track MBID (disc-specific track).
    track_id: Option<String>,
}

impl MusicBrainzTags {
    /// Create an empty tag container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the recording MBID.
    pub fn with_recording_id(mut self, mbid: impl Into<String>) -> Self {
        self.recording_id = Some(mbid.into());
        self
    }

    /// Set the release MBID.
    pub fn with_release_id(mut self, mbid: impl Into<String>) -> Self {
        self.release_id = Some(mbid.into());
        self
    }

    /// Set the release group MBID.
    pub fn with_release_group_id(mut self, mbid: impl Into<String>) -> Self {
        self.release_group_id = Some(mbid.into());
        self
    }

    /// Set the artist MBID.
    pub fn with_artist_id(mut self, mbid: impl Into<String>) -> Self {
        self.artist_id = Some(mbid.into());
        self
    }

    /// Set the album artist MBID.
    pub fn with_album_artist_id(mut self, mbid: impl Into<String>) -> Self {
        self.album_artist_id = Some(mbid.into());
        self
    }

    /// Set the work MBID.
    pub fn with_work_id(mut self, mbid: impl Into<String>) -> Self {
        self.work_id = Some(mbid.into());
        self
    }

    /// Set the label MBID.
    pub fn with_label_id(mut self, mbid: impl Into<String>) -> Self {
        self.label_id = Some(mbid.into());
        self
    }

    /// Set the release track MBID.
    pub fn with_track_id(mut self, mbid: impl Into<String>) -> Self {
        self.track_id = Some(mbid.into());
        self
    }

    // ---- Getters ----

    /// Recording MBID.
    pub fn recording_id(&self) -> Option<&str> {
        self.recording_id.as_deref()
    }

    /// Release MBID.
    pub fn release_id(&self) -> Option<&str> {
        self.release_id.as_deref()
    }

    /// Release group MBID.
    pub fn release_group_id(&self) -> Option<&str> {
        self.release_group_id.as_deref()
    }

    /// Artist MBID.
    pub fn artist_id(&self) -> Option<&str> {
        self.artist_id.as_deref()
    }

    /// Album artist MBID.
    pub fn album_artist_id(&self) -> Option<&str> {
        self.album_artist_id.as_deref()
    }

    /// Work MBID.
    pub fn work_id(&self) -> Option<&str> {
        self.work_id.as_deref()
    }

    /// Label MBID.
    pub fn label_id(&self) -> Option<&str> {
        self.label_id.as_deref()
    }

    /// Release track MBID.
    pub fn track_id(&self) -> Option<&str> {
        self.track_id.as_deref()
    }

    /// Returns true if any MusicBrainz ID is present.
    pub fn has_data(&self) -> bool {
        self.recording_id.is_some()
            || self.release_id.is_some()
            || self.release_group_id.is_some()
            || self.artist_id.is_some()
            || self.album_artist_id.is_some()
            || self.work_id.is_some()
            || self.label_id.is_some()
            || self.track_id.is_some()
    }

    /// Count how many MBIDs are set.
    pub fn id_count(&self) -> usize {
        let fields: [&Option<String>; 8] = [
            &self.recording_id,
            &self.release_id,
            &self.release_group_id,
            &self.artist_id,
            &self.album_artist_id,
            &self.work_id,
            &self.label_id,
            &self.track_id,
        ];
        fields.iter().filter(|f| f.is_some()).count()
    }

    /// Validate all set MBIDs.
    ///
    /// Returns a list of validation errors (empty = all valid).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        let fields: [(&str, &Option<String>); 8] = [
            ("recording_id", &self.recording_id),
            ("release_id", &self.release_id),
            ("release_group_id", &self.release_group_id),
            ("artist_id", &self.artist_id),
            ("album_artist_id", &self.album_artist_id),
            ("work_id", &self.work_id),
            ("label_id", &self.label_id),
            ("track_id", &self.track_id),
        ];

        for (name, value) in &fields {
            if let Some(mbid) = value {
                // Support semicolon-separated multiple MBIDs (artist IDs)
                for single_mbid in mbid.split(';') {
                    let trimmed = single_mbid.trim();
                    if !trimmed.is_empty() && !validate_mbid(trimmed) {
                        errors.push(format!("Invalid MBID for {name}: '{trimmed}'"));
                    }
                }
            }
        }

        errors
    }

    /// Get the MusicBrainz URL for the recording.
    pub fn recording_url(&self) -> Option<Result<String, Error>> {
        self.recording_id
            .as_ref()
            .map(|id| mbid_url(MbEntityType::Recording, id))
    }

    /// Get the MusicBrainz URL for the release.
    pub fn release_url(&self) -> Option<Result<String, Error>> {
        self.release_id
            .as_ref()
            .map(|id| mbid_url(MbEntityType::Release, id))
    }

    /// Get the MusicBrainz URL for the artist.
    pub fn artist_url(&self) -> Option<Result<String, Error>> {
        self.artist_id
            .as_ref()
            .map(|id| mbid_url(MbEntityType::Artist, id))
    }

    /// Write MusicBrainz tags to a `Metadata` container.
    ///
    /// Uses the appropriate tag key names for the container's format.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        let keys = tag_keys_for_format(metadata.format());

        if let Some(ref id) = self.recording_id {
            metadata.insert(keys.recording.to_string(), MetadataValue::Text(id.clone()));
        }
        if let Some(ref id) = self.release_id {
            metadata.insert(keys.release.to_string(), MetadataValue::Text(id.clone()));
        }
        if let Some(ref id) = self.release_group_id {
            metadata.insert(
                keys.release_group.to_string(),
                MetadataValue::Text(id.clone()),
            );
        }
        if let Some(ref id) = self.artist_id {
            metadata.insert(keys.artist.to_string(), MetadataValue::Text(id.clone()));
        }
        if let Some(ref id) = self.album_artist_id {
            metadata.insert(
                keys.album_artist.to_string(),
                MetadataValue::Text(id.clone()),
            );
        }
        if let Some(ref id) = self.work_id {
            metadata.insert(keys.work.to_string(), MetadataValue::Text(id.clone()));
        }
        if let (Some(ref id), Some(key)) = (&self.label_id, keys.label) {
            metadata.insert(key.to_string(), MetadataValue::Text(id.clone()));
        }
        if let (Some(ref id), Some(key)) = (&self.track_id, keys.track) {
            metadata.insert(key.to_string(), MetadataValue::Text(id.clone()));
        }
    }

    /// Extract MusicBrainz tags from a `Metadata` container.
    pub fn from_metadata(metadata: &Metadata) -> Self {
        let keys = tag_keys_for_format(metadata.format());
        let mut tags = MusicBrainzTags::new();

        tags.recording_id = get_text(metadata, keys.recording);
        tags.release_id = get_text(metadata, keys.release);
        tags.release_group_id = get_text(metadata, keys.release_group);
        tags.artist_id = get_text(metadata, keys.artist);
        tags.album_artist_id = get_text(metadata, keys.album_artist);
        tags.work_id = get_text(metadata, keys.work);
        if let Some(key) = keys.label {
            tags.label_id = get_text(metadata, key);
        }
        if let Some(key) = keys.track {
            tags.track_id = get_text(metadata, key);
        }

        tags
    }
}

/// Helper to extract text from metadata.
fn get_text(metadata: &Metadata, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_MBID: &str = "f27ec8db-af05-4f36-916e-3c57f9e4e1b3";
    const VALID_MBID_2: &str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";

    // ---- MBID validation tests ----

    #[test]
    fn test_validate_mbid_valid() {
        assert!(validate_mbid(VALID_MBID));
        assert!(validate_mbid(VALID_MBID_2));
        assert!(validate_mbid("00000000-0000-0000-0000-000000000000"));
        assert!(validate_mbid("FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"));
    }

    #[test]
    fn test_validate_mbid_valid_with_whitespace() {
        assert!(validate_mbid(&format!("  {VALID_MBID}  ")));
    }

    #[test]
    fn test_validate_mbid_invalid() {
        assert!(!validate_mbid("not-a-valid-uuid"));
        assert!(!validate_mbid(""));
        assert!(!validate_mbid("f27ec8db-af05-4f36-916e")); // too short
        assert!(!validate_mbid("f27ec8db-af05-4f36-916e-3c57f9e4e1b3-extra")); // too long
        assert!(!validate_mbid("f27ec8db_af05_4f36_916e_3c57f9e4e1b3")); // underscores
        assert!(!validate_mbid("g27ec8db-af05-4f36-916e-3c57f9e4e1b3")); // 'g' not hex
    }

    #[test]
    fn test_validate_mbid_wrong_section_lengths() {
        assert!(!validate_mbid("f27ec8d-baf05-4f36-916e-3c57f9e4e1b3")); // first section 7 chars
    }

    #[test]
    fn test_normalize_mbid() {
        let upper = "F27EC8DB-AF05-4F36-916E-3C57F9E4E1B3";
        let normalized = normalize_mbid(upper);
        assert_eq!(normalized.as_deref(), Some(VALID_MBID));
    }

    #[test]
    fn test_normalize_mbid_invalid() {
        assert!(normalize_mbid("invalid").is_none());
    }

    // ---- MBID URL tests ----

    #[test]
    fn test_mbid_url_recording() {
        let url = mbid_url(MbEntityType::Recording, VALID_MBID).expect("should succeed");
        assert_eq!(
            url,
            format!("https://musicbrainz.org/recording/{VALID_MBID}")
        );
    }

    #[test]
    fn test_mbid_url_release() {
        let url = mbid_url(MbEntityType::Release, VALID_MBID).expect("should succeed");
        assert!(url.contains("/release/"));
    }

    #[test]
    fn test_mbid_url_artist() {
        let url = mbid_url(MbEntityType::Artist, VALID_MBID).expect("should succeed");
        assert!(url.contains("/artist/"));
    }

    #[test]
    fn test_mbid_url_invalid() {
        let result = mbid_url(MbEntityType::Recording, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_mbid_url_normalizes_case() {
        let upper = "F27EC8DB-AF05-4F36-916E-3C57F9E4E1B3";
        let url = mbid_url(MbEntityType::Recording, upper).expect("should succeed");
        assert!(url.contains(VALID_MBID));
    }

    // ---- Entity type tests ----

    #[test]
    fn test_entity_type_url_path() {
        assert_eq!(MbEntityType::Recording.url_path(), "recording");
        assert_eq!(MbEntityType::Release.url_path(), "release");
        assert_eq!(MbEntityType::ReleaseGroup.url_path(), "release-group");
        assert_eq!(MbEntityType::Artist.url_path(), "artist");
        assert_eq!(MbEntityType::Work.url_path(), "work");
        assert_eq!(MbEntityType::Label.url_path(), "label");
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(MbEntityType::Recording.to_string(), "Recording");
        assert_eq!(MbEntityType::ReleaseGroup.to_string(), "Release Group");
    }

    // ---- Tag keys tests ----

    #[test]
    fn test_vorbis_tag_keys() {
        let keys = vorbis_tag_keys();
        assert_eq!(keys.recording, "MUSICBRAINZ_TRACKID");
        assert_eq!(keys.release, "MUSICBRAINZ_ALBUMID");
        assert_eq!(keys.artist, "MUSICBRAINZ_ARTISTID");
        assert!(keys.label.is_some());
        assert!(keys.track.is_some());
    }

    #[test]
    fn test_id3v2_tag_keys() {
        let keys = id3v2_tag_keys();
        assert_eq!(keys.recording, "MusicBrainz Release Track Id");
        assert_eq!(keys.release, "MusicBrainz Album Id");
        assert!(keys.label.is_some());
    }

    #[test]
    fn test_apev2_tag_keys() {
        let keys = apev2_tag_keys();
        assert_eq!(keys.recording, "MUSICBRAINZ_TRACKID");
        assert!(keys.label.is_none());
        assert!(keys.track.is_none());
    }

    #[test]
    fn test_tag_keys_for_format() {
        let id3_keys = tag_keys_for_format(MetadataFormat::Id3v2);
        assert_eq!(id3_keys.recording, "MusicBrainz Release Track Id");

        let vorbis_keys = tag_keys_for_format(MetadataFormat::VorbisComments);
        assert_eq!(vorbis_keys.recording, "MUSICBRAINZ_TRACKID");

        let ape_keys = tag_keys_for_format(MetadataFormat::Apev2);
        assert_eq!(ape_keys.recording, "MUSICBRAINZ_TRACKID");
    }

    // ---- MusicBrainzTags tests ----

    #[test]
    fn test_tags_new_empty() {
        let tags = MusicBrainzTags::new();
        assert!(!tags.has_data());
        assert_eq!(tags.id_count(), 0);
    }

    #[test]
    fn test_tags_with_builders() {
        let tags = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID_2)
            .with_artist_id(VALID_MBID);

        assert!(tags.has_data());
        assert_eq!(tags.id_count(), 3);
        assert_eq!(tags.recording_id(), Some(VALID_MBID));
        assert_eq!(tags.release_id(), Some(VALID_MBID_2));
        assert_eq!(tags.artist_id(), Some(VALID_MBID));
    }

    #[test]
    fn test_tags_all_fields() {
        let tags = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID)
            .with_release_group_id(VALID_MBID)
            .with_artist_id(VALID_MBID)
            .with_album_artist_id(VALID_MBID)
            .with_work_id(VALID_MBID)
            .with_label_id(VALID_MBID)
            .with_track_id(VALID_MBID);

        assert_eq!(tags.id_count(), 8);
    }

    #[test]
    fn test_tags_validate_valid() {
        let tags = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID_2);

        assert!(tags.validate().is_empty());
    }

    #[test]
    fn test_tags_validate_invalid() {
        let tags = MusicBrainzTags::new()
            .with_recording_id("not-valid")
            .with_release_id(VALID_MBID);

        let errors = tags.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("recording_id"));
    }

    #[test]
    fn test_tags_validate_multiple_artist_ids() {
        let multi = format!("{VALID_MBID}; {VALID_MBID_2}");
        let tags = MusicBrainzTags::new().with_artist_id(multi);
        assert!(tags.validate().is_empty());
    }

    #[test]
    fn test_tags_validate_invalid_in_multi() {
        let multi = format!("{VALID_MBID}; not-valid");
        let tags = MusicBrainzTags::new().with_artist_id(multi);
        let errors = tags.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_tags_urls() {
        let tags = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID_2)
            .with_artist_id(VALID_MBID);

        let rec_url = tags
            .recording_url()
            .expect("should have recording")
            .expect("should be valid");
        assert!(rec_url.contains("/recording/"));

        let rel_url = tags
            .release_url()
            .expect("should have release")
            .expect("should be valid");
        assert!(rel_url.contains("/release/"));

        let art_url = tags
            .artist_url()
            .expect("should have artist")
            .expect("should be valid");
        assert!(art_url.contains("/artist/"));
    }

    #[test]
    fn test_tags_urls_none() {
        let tags = MusicBrainzTags::new();
        assert!(tags.recording_url().is_none());
        assert!(tags.release_url().is_none());
        assert!(tags.artist_url().is_none());
    }

    // ---- Metadata integration tests ----

    #[test]
    fn test_tags_vorbis_metadata_round_trip() {
        let original = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID_2)
            .with_artist_id(VALID_MBID)
            .with_work_id(VALID_MBID_2);

        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        original.to_metadata(&mut metadata);

        // Verify tag keys
        assert!(metadata.contains("MUSICBRAINZ_TRACKID"));
        assert!(metadata.contains("MUSICBRAINZ_ALBUMID"));
        assert!(metadata.contains("MUSICBRAINZ_ARTISTID"));
        assert!(metadata.contains("MUSICBRAINZ_WORKID"));

        let restored = MusicBrainzTags::from_metadata(&metadata);
        assert_eq!(restored.recording_id(), Some(VALID_MBID));
        assert_eq!(restored.release_id(), Some(VALID_MBID_2));
        assert_eq!(restored.artist_id(), Some(VALID_MBID));
        assert_eq!(restored.work_id(), Some(VALID_MBID_2));
    }

    #[test]
    fn test_tags_id3v2_metadata_round_trip() {
        let original = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_release_id(VALID_MBID_2);

        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        original.to_metadata(&mut metadata);

        // Verify ID3v2 tag keys
        assert!(metadata.contains("MusicBrainz Release Track Id"));
        assert!(metadata.contains("MusicBrainz Album Id"));

        let restored = MusicBrainzTags::from_metadata(&metadata);
        assert_eq!(restored.recording_id(), Some(VALID_MBID));
        assert_eq!(restored.release_id(), Some(VALID_MBID_2));
    }

    #[test]
    fn test_tags_from_empty_metadata() {
        let metadata = Metadata::new(MetadataFormat::VorbisComments);
        let tags = MusicBrainzTags::from_metadata(&metadata);
        assert!(!tags.has_data());
        assert_eq!(tags.id_count(), 0);
    }

    #[test]
    fn test_tags_vorbis_label_and_track() {
        let original = MusicBrainzTags::new()
            .with_label_id(VALID_MBID)
            .with_track_id(VALID_MBID_2);

        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        original.to_metadata(&mut metadata);

        assert!(metadata.contains("MUSICBRAINZ_LABELID"));
        assert!(metadata.contains("MUSICBRAINZ_RELEASETRACKID"));

        let restored = MusicBrainzTags::from_metadata(&metadata);
        assert_eq!(restored.label_id(), Some(VALID_MBID));
        assert_eq!(restored.track_id(), Some(VALID_MBID_2));
    }

    #[test]
    fn test_tags_apev2_no_label_track() {
        let original = MusicBrainzTags::new()
            .with_recording_id(VALID_MBID)
            .with_label_id(VALID_MBID_2) // not supported in APEv2
            .with_track_id(VALID_MBID_2); // not supported in APEv2

        let mut metadata = Metadata::new(MetadataFormat::Apev2);
        original.to_metadata(&mut metadata);

        // Recording should be written
        assert!(metadata.contains("MUSICBRAINZ_TRACKID"));
        // Label and track should NOT be written (APEv2 doesn't support them)
        assert!(!metadata.contains("MUSICBRAINZ_LABELID"));
        assert!(!metadata.contains("MUSICBRAINZ_RELEASETRACKID"));
    }

    #[test]
    fn test_tags_equality() {
        let t1 = MusicBrainzTags::new().with_recording_id(VALID_MBID);
        let t2 = MusicBrainzTags::new().with_recording_id(VALID_MBID);
        let t3 = MusicBrainzTags::new().with_recording_id(VALID_MBID_2);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }
}
