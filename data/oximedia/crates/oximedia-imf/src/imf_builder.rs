//! High-level fluent builder API for creating IMF packages from scratch.
//!
//! Provides [`ImfPackageBuilder`] which auto-generates a CPL, PKL, and AssetMap
//! from a set of video, audio, and subtitle track descriptions.
//!
//! # Example
//! ```ignore
//! use oximedia_imf::imf_builder::{ImfPackageBuilder, EditRate};
//! use oximedia_imf::application_profile::ApplicationProfile;
//!
//! let pkg = ImfPackageBuilder::new("My Feature")
//!     .add_video_track("/essence/video.mxf", EditRate::fps_24())
//!     .add_audio_track("/essence/audio.mxf", 8, 48000)
//!     .with_application_profile(ApplicationProfile::App2)
//!     .build()
//!     .unwrap();
//! ```

use crate::application_profile::ApplicationProfile;
use crate::{ImfError, ImfResult};

use std::fmt;
use std::path::PathBuf;

use uuid::Uuid;

// ---------------------------------------------------------------------------
// EditRate
// ---------------------------------------------------------------------------

/// Frame/edit rate expressed as a rational number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditRate {
    /// Numerator (e.g. 24, 25, 30000 …)
    pub numerator: u32,
    /// Denominator (e.g. 1, 1001 …)
    pub denominator: u32,
}

impl EditRate {
    /// Construct an [`EditRate`].
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// 24 fps progressive.
    pub fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// 25 fps (PAL/European broadcast).
    pub fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// 30 fps progressive.
    pub fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// 23.976 fps (23000/1001 NTSC film).
    pub fn fps_23_976() -> Self {
        Self::new(24000, 1001)
    }

    /// 29.97 fps (30000/1001 NTSC broadcast).
    pub fn fps_29_97() -> Self {
        Self::new(30000, 1001)
    }

    /// 48 fps (high-frame-rate cinema).
    pub fn fps_48() -> Self {
        Self::new(48, 1)
    }

    /// 60 fps.
    pub fn fps_60() -> Self {
        Self::new(60, 1)
    }

    /// Rate as a floating-point value (for informational display only).
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> f64 {
        if self.denominator == 0 {
            return 0.0;
        }
        self.numerator as f64 / self.denominator as f64
    }
}

impl fmt::Display for EditRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

// ---------------------------------------------------------------------------
// TrackDescriptor — internal description of each requested track
// ---------------------------------------------------------------------------

/// Kind of essence track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrackKind {
    /// MXF-wrapped video essence.
    Video,
    /// MXF-wrapped audio essence.
    Audio {
        /// Number of audio channels carried by this track file.
        channels: u32,
        /// Sample rate in Hz (e.g. 48000).
        sample_rate: u32,
    },
    /// XML/TTML subtitle track.
    Subtitle,
}

impl fmt::Display for TrackKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio {
                channels,
                sample_rate,
            } => {
                write!(f, "Audio({channels}ch/{sample_rate}Hz)")
            }
            Self::Subtitle => write!(f, "Subtitle"),
        }
    }
}

/// Description of a single track to be included in the package.
#[derive(Clone, Debug)]
pub struct TrackDescriptor {
    /// Unique ID that will be used in CPL/PKL/AssetMap.
    pub id: Uuid,
    /// Absolute path to the source essence file.
    pub path: PathBuf,
    /// Kind of track.
    pub kind: TrackKind,
    /// Edit rate for this track.
    pub edit_rate: EditRate,
    /// Optional annotation label.
    pub label: Option<String>,
}

impl TrackDescriptor {
    /// MIME type string for PKL `Type` element.
    pub fn mime_type(&self) -> &'static str {
        match &self.kind {
            TrackKind::Video => "video/mxf",
            TrackKind::Audio { .. } => "audio/mxf",
            TrackKind::Subtitle => "application/ttml+xml",
        }
    }
}

// ---------------------------------------------------------------------------
// Built package components
// ---------------------------------------------------------------------------

/// A minimal asset entry included in the generated AssetMap XML.
#[derive(Clone, Debug)]
pub struct AssetEntry {
    /// Asset UUID.
    pub id: Uuid,
    /// Relative filename within the package.
    pub filename: String,
    /// Whether this asset is a packing list.
    pub is_packing_list: bool,
}

/// A minimal PKL asset entry.
#[derive(Clone, Debug)]
pub struct PklEntry {
    /// Asset UUID.
    pub id: Uuid,
    /// Relative filename.
    pub filename: String,
    /// File size in bytes (0 = not yet computed / virtual).
    pub size: u64,
    /// SHA-1 hex hash (simplified; real production code would use sha2).
    pub hash: String,
    /// MIME type.
    pub mime_type: String,
}

/// A virtual CPL resource referencing one track file.
#[derive(Clone, Debug)]
pub struct CplResource {
    /// Resource UUID.
    pub id: Uuid,
    /// Track UUID.
    pub track_file_id: Uuid,
    /// Entry point in edit units.
    pub entry_point: u64,
    /// Duration in edit units.  0 = "use full file".
    pub duration: u64,
    /// Edit rate for this resource.
    pub edit_rate: EditRate,
}

/// A CPL segment containing a parallel set of resources.
#[derive(Clone, Debug)]
pub struct CplSegment {
    /// Segment UUID.
    pub id: Uuid,
    /// Resources in this segment (one per track).
    pub resources: Vec<CplResource>,
}

/// The fully built IMF package description.
///
/// This is an in-memory representation — no files are written to disk.
/// Callers can inspect the structure and serialize it as needed.
#[derive(Debug)]
pub struct BuiltImfPackage {
    /// Package ID (AssetMap ID).
    pub package_id: Uuid,
    /// PKL ID.
    pub pkl_id: Uuid,
    /// CPL ID.
    pub cpl_id: Uuid,
    /// Content title supplied to the builder.
    pub title: String,
    /// Edit rate of the primary composition.
    pub edit_rate: EditRate,
    /// Application profile if set.
    pub application_profile: Option<ApplicationProfile>,
    /// Track descriptors (one per `add_*_track` call).
    pub tracks: Vec<TrackDescriptor>,
    /// CPL segments.
    pub segments: Vec<CplSegment>,
    /// AssetMap entries.
    pub asset_map_entries: Vec<AssetEntry>,
    /// PKL entries.
    pub pkl_entries: Vec<PklEntry>,
    /// Creator string.
    pub creator: Option<String>,
    /// Issuer string.
    pub issuer: Option<String>,
}

impl BuiltImfPackage {
    /// Returns all video tracks.
    pub fn video_tracks(&self) -> impl Iterator<Item = &TrackDescriptor> {
        self.tracks.iter().filter(|t| t.kind == TrackKind::Video)
    }

    /// Returns all audio tracks.
    pub fn audio_tracks(&self) -> impl Iterator<Item = &TrackDescriptor> {
        self.tracks
            .iter()
            .filter(|t| matches!(t.kind, TrackKind::Audio { .. }))
    }

    /// Returns all subtitle tracks.
    pub fn subtitle_tracks(&self) -> impl Iterator<Item = &TrackDescriptor> {
        self.tracks.iter().filter(|t| t.kind == TrackKind::Subtitle)
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("IMF Package: {}\n", self.title));
        s.push_str(&format!("  Package ID : {}\n", self.package_id));
        s.push_str(&format!("  CPL ID     : {}\n", self.cpl_id));
        s.push_str(&format!("  PKL ID     : {}\n", self.pkl_id));
        s.push_str(&format!("  Edit rate  : {}\n", self.edit_rate));
        if let Some(ref profile) = self.application_profile {
            s.push_str(&format!("  Profile    : {}\n", profile.urn()));
        }
        s.push_str(&format!("  Tracks     : {}\n", self.tracks.len()));
        for t in &self.tracks {
            s.push_str(&format!(
                "    [{}] {} @ {}\n",
                t.kind,
                t.path.display(),
                t.edit_rate
            ));
        }
        s.push_str(&format!("  Segments   : {}\n", self.segments.len()));
        s
    }
}

// ---------------------------------------------------------------------------
// ImfPackageBuilder
// ---------------------------------------------------------------------------

/// High-level fluent builder for creating complete IMF package metadata.
///
/// Generates CPL, PKL, and AssetMap data structures in a single [`build`] call.
///
/// [`build`]: ImfPackageBuilder::build
#[derive(Debug, Default)]
pub struct ImfPackageBuilder {
    title: String,
    creator: Option<String>,
    issuer: Option<String>,
    edit_rate: Option<EditRate>,
    application_profile: Option<ApplicationProfile>,
    tracks: Vec<TrackDescriptor>,
}

impl ImfPackageBuilder {
    /// Create a new builder with the given content title.
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            ..Self::default()
        }
    }

    /// Set the creator metadata field (e.g. NLE software name).
    pub fn with_creator(mut self, creator: &str) -> Self {
        self.creator = Some(creator.to_string());
        self
    }

    /// Set the issuer metadata field (e.g. studio name).
    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.issuer = Some(issuer.to_string());
        self
    }

    /// Override the default CPL edit rate (24 fps is used if not set).
    pub fn with_edit_rate(mut self, rate: EditRate) -> Self {
        self.edit_rate = Some(rate);
        self
    }

    /// Specify the SMPTE application profile this package targets.
    pub fn with_application_profile(mut self, profile: ApplicationProfile) -> Self {
        self.application_profile = Some(profile);
        self
    }

    /// Add a video track.
    ///
    /// `path` is the path to the MXF track file.
    /// `edit_rate` is the track's frame rate.
    pub fn add_video_track(mut self, path: impl Into<PathBuf>, edit_rate: EditRate) -> Self {
        self.tracks.push(TrackDescriptor {
            id: Uuid::new_v4(),
            path: path.into(),
            kind: TrackKind::Video,
            edit_rate,
            label: None,
        });
        self
    }

    /// Add an audio track.
    ///
    /// `path` is the path to the MXF track file.
    /// `channels` is the channel count (e.g. 2 for stereo, 8 for 7.1).
    /// `sample_rate` is the sample rate in Hz (e.g. 48000).
    pub fn add_audio_track(
        mut self,
        path: impl Into<PathBuf>,
        channels: u32,
        sample_rate: u32,
    ) -> Self {
        self.tracks.push(TrackDescriptor {
            id: Uuid::new_v4(),
            path: path.into(),
            kind: TrackKind::Audio {
                channels,
                sample_rate,
            },
            edit_rate: self.edit_rate.unwrap_or_else(EditRate::fps_24),
            label: None,
        });
        self
    }

    /// Add a subtitle/caption track (TTML/IMSC1 XML file).
    pub fn add_subtitle_track(mut self, path: impl Into<PathBuf>) -> Self {
        self.tracks.push(TrackDescriptor {
            id: Uuid::new_v4(),
            path: path.into(),
            kind: TrackKind::Subtitle,
            edit_rate: self.edit_rate.unwrap_or_else(EditRate::fps_24),
            label: None,
        });
        self
    }

    /// Add an annotation label to the most recently added track.
    pub fn with_track_label(mut self, label: &str) -> ImfResult<Self> {
        let last = self.tracks.last_mut().ok_or_else(|| {
            ImfError::InvalidStructure("No track to label — call add_*_track first".to_string())
        })?;
        last.label = Some(label.to_string());
        Ok(self)
    }

    /// Validate builder state before building.
    fn validate(&self) -> ImfResult<()> {
        if self.title.trim().is_empty() {
            return Err(ImfError::InvalidStructure(
                "Package title must not be empty".to_string(),
            ));
        }
        if self.tracks.is_empty() {
            return Err(ImfError::MissingElement(
                "At least one track is required".to_string(),
            ));
        }
        // Verify all video tracks have the same edit rate
        let mut video_rates = self
            .tracks
            .iter()
            .filter(|t| t.kind == TrackKind::Video)
            .map(|t| t.edit_rate);
        if let Some(first_rate) = video_rates.next() {
            for rate in video_rates {
                if rate != first_rate {
                    return Err(ImfError::InvalidEditRate(
                        "All video tracks must share the same edit rate".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Build the [`BuiltImfPackage`].
    ///
    /// This assembles CPL segments, PKL entries, and AssetMap entries from
    /// the registered tracks.  No file I/O is performed.
    pub fn build(self) -> ImfResult<BuiltImfPackage> {
        self.validate()?;

        let cpl_edit_rate = self
            .edit_rate
            .or_else(|| {
                self.tracks
                    .iter()
                    .find(|t| t.kind == TrackKind::Video)
                    .map(|t| t.edit_rate)
            })
            .unwrap_or_else(EditRate::fps_24);

        let package_id = Uuid::new_v4();
        let pkl_id = Uuid::new_v4();
        let cpl_id = Uuid::new_v4();

        // ---- Build CPL: one segment containing all tracks -----------------
        let segment_id = Uuid::new_v4();
        let mut cpl_resources = Vec::new();
        for track in &self.tracks {
            cpl_resources.push(CplResource {
                id: Uuid::new_v4(),
                track_file_id: track.id,
                entry_point: 0,
                duration: 0, // full track
                edit_rate: track.edit_rate,
            });
        }
        let segments = vec![CplSegment {
            id: segment_id,
            resources: cpl_resources,
        }];

        // ---- Build AssetMap entries ----------------------------------------
        let mut asset_map_entries: Vec<AssetEntry> = self
            .tracks
            .iter()
            .map(|t| {
                let filename = t
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("{}.mxf", t.id));
                AssetEntry {
                    id: t.id,
                    filename,
                    is_packing_list: false,
                }
            })
            .collect();

        // CPL as asset
        asset_map_entries.push(AssetEntry {
            id: cpl_id,
            filename: format!("CPL_{cpl_id}.xml"),
            is_packing_list: false,
        });

        // PKL as asset
        asset_map_entries.push(AssetEntry {
            id: pkl_id,
            filename: format!("PKL_{pkl_id}.xml"),
            is_packing_list: true,
        });

        // ---- Build PKL entries ---------------------------------------------
        let mut pkl_entries: Vec<PklEntry> = self
            .tracks
            .iter()
            .map(|t| {
                let filename = t
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("{}.mxf", t.id));
                PklEntry {
                    id: t.id,
                    filename,
                    size: 0, // not computed in-memory builder
                    hash: String::new(),
                    mime_type: t.mime_type().to_string(),
                }
            })
            .collect();

        // PKL also includes the CPL itself
        pkl_entries.push(PklEntry {
            id: cpl_id,
            filename: format!("CPL_{cpl_id}.xml"),
            size: 0,
            hash: String::new(),
            mime_type: "application/xml".to_string(),
        });

        Ok(BuiltImfPackage {
            package_id,
            pkl_id,
            cpl_id,
            title: self.title,
            edit_rate: cpl_edit_rate,
            application_profile: self.application_profile,
            tracks: self.tracks,
            segments,
            asset_map_entries,
            pkl_entries,
            creator: self.creator,
            issuer: self.issuer,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application_profile::ApplicationProfile;

    fn video_path() -> PathBuf {
        PathBuf::from("/virtual/video_001.mxf")
    }

    fn audio_path() -> PathBuf {
        PathBuf::from("/virtual/audio_001.mxf")
    }

    fn subtitle_path() -> PathBuf {
        PathBuf::from("/virtual/subtitle.xml")
    }

    // --- EditRate tests ---

    #[test]
    fn test_edit_rate_fps_24_display() {
        assert_eq!(EditRate::fps_24().to_string(), "24");
    }

    #[test]
    fn test_edit_rate_fps_23_976_display() {
        assert_eq!(EditRate::fps_23_976().to_string(), "24000/1001");
    }

    #[test]
    fn test_edit_rate_as_f64() {
        let rate = EditRate::fps_25();
        assert!((rate.as_f64() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_edit_rate_zero_denominator() {
        let rate = EditRate::new(24, 0);
        assert_eq!(rate.as_f64(), 0.0);
    }

    // --- Builder happy path ---

    #[test]
    fn test_build_video_only() {
        let pkg = ImfPackageBuilder::new("Test Package")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build should succeed");

        assert_eq!(pkg.title, "Test Package");
        assert_eq!(pkg.tracks.len(), 1);
        assert_eq!(pkg.video_tracks().count(), 1);
        assert_eq!(pkg.audio_tracks().count(), 0);
    }

    #[test]
    fn test_build_video_and_audio() {
        let pkg = ImfPackageBuilder::new("Film")
            .add_video_track(video_path(), EditRate::fps_24())
            .add_audio_track(audio_path(), 8, 48000)
            .build()
            .expect("build should succeed");

        assert_eq!(pkg.tracks.len(), 2);
        assert_eq!(pkg.audio_tracks().count(), 1);

        let audio = pkg.audio_tracks().next().expect("audio track");
        match audio.kind {
            TrackKind::Audio {
                channels,
                sample_rate,
            } => {
                assert_eq!(channels, 8);
                assert_eq!(sample_rate, 48000);
            }
            _ => panic!("expected audio track kind"),
        }
    }

    #[test]
    fn test_build_with_subtitle_track() {
        let pkg = ImfPackageBuilder::new("Series")
            .add_video_track(video_path(), EditRate::fps_25())
            .add_audio_track(audio_path(), 2, 48000)
            .add_subtitle_track(subtitle_path())
            .build()
            .expect("build should succeed");

        assert_eq!(pkg.subtitle_tracks().count(), 1);
        assert_eq!(pkg.tracks.len(), 3);
    }

    #[test]
    fn test_build_generates_unique_ids() {
        let pkg1 = ImfPackageBuilder::new("A")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");
        let pkg2 = ImfPackageBuilder::new("A")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");

        // Each build call should yield fresh UUIDs
        assert_ne!(pkg1.package_id, pkg2.package_id);
        assert_ne!(pkg1.cpl_id, pkg2.cpl_id);
        assert_ne!(pkg1.pkl_id, pkg2.pkl_id);
    }

    #[test]
    fn test_build_with_application_profile() {
        let pkg = ImfPackageBuilder::new("Streaming")
            .add_video_track(video_path(), EditRate::fps_24())
            .with_application_profile(ApplicationProfile::App2)
            .build()
            .expect("build");

        assert!(pkg.application_profile.is_some());
        let profile = pkg.application_profile.as_ref().expect("profile");
        assert_eq!(*profile, ApplicationProfile::App2);
    }

    #[test]
    fn test_build_edit_rate_defaults_to_video_rate() {
        let pkg = ImfPackageBuilder::new("Hi-FPS")
            .add_video_track(video_path(), EditRate::fps_48())
            .add_audio_track(audio_path(), 2, 48000)
            .build()
            .expect("build");

        assert_eq!(pkg.edit_rate, EditRate::fps_48());
    }

    #[test]
    fn test_build_explicit_edit_rate_overrides() {
        let pkg = ImfPackageBuilder::new("Broadcast")
            .with_edit_rate(EditRate::fps_25())
            .add_video_track(video_path(), EditRate::fps_25())
            .build()
            .expect("build");

        assert_eq!(pkg.edit_rate, EditRate::fps_25());
    }

    #[test]
    fn test_build_cpl_segment_has_all_resources() {
        let pkg = ImfPackageBuilder::new("Multi")
            .add_video_track(video_path(), EditRate::fps_24())
            .add_audio_track(audio_path(), 2, 48000)
            .add_subtitle_track(subtitle_path())
            .build()
            .expect("build");

        assert_eq!(pkg.segments.len(), 1);
        assert_eq!(pkg.segments[0].resources.len(), 3);
    }

    #[test]
    fn test_build_asset_map_includes_pkl() {
        let pkg = ImfPackageBuilder::new("Pack")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");

        let pkl_entries: Vec<_> = pkg
            .asset_map_entries
            .iter()
            .filter(|e| e.is_packing_list)
            .collect();
        assert_eq!(pkl_entries.len(), 1);
    }

    #[test]
    fn test_build_pkl_includes_cpl() {
        let pkg = ImfPackageBuilder::new("X")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");

        let cpl_in_pkl = pkg
            .pkl_entries
            .iter()
            .any(|e| e.id == pkg.cpl_id && e.mime_type == "application/xml");
        assert!(cpl_in_pkl, "CPL must appear in PKL entries");
    }

    #[test]
    fn test_build_error_empty_title() {
        let result = ImfPackageBuilder::new("   ")
            .add_video_track(video_path(), EditRate::fps_24())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_error_no_tracks() {
        let result = ImfPackageBuilder::new("No Tracks").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_creator_issuer() {
        let pkg = ImfPackageBuilder::new("Studio Film")
            .with_creator("OxiMedia Encoder 0.1.2")
            .with_issuer("CoolJapan Studios")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");

        assert_eq!(pkg.creator.as_deref(), Some("OxiMedia Encoder 0.1.2"));
        assert_eq!(pkg.issuer.as_deref(), Some("CoolJapan Studios"));
    }

    #[test]
    fn test_build_summary_contains_title() {
        let pkg = ImfPackageBuilder::new("Summary Test")
            .add_video_track(video_path(), EditRate::fps_24())
            .build()
            .expect("build");
        let summary = pkg.summary();
        assert!(summary.contains("Summary Test"));
    }

    #[test]
    fn test_track_mime_types() {
        let pkg = ImfPackageBuilder::new("MIME")
            .add_video_track(video_path(), EditRate::fps_24())
            .add_audio_track(audio_path(), 2, 48000)
            .add_subtitle_track(subtitle_path())
            .build()
            .expect("build");

        let kinds: Vec<_> = pkg.tracks.iter().map(|t| t.mime_type()).collect();
        assert!(kinds.contains(&"video/mxf"));
        assert!(kinds.contains(&"audio/mxf"));
        assert!(kinds.contains(&"application/ttml+xml"));
    }

    #[test]
    fn test_track_label() {
        let pkg = ImfPackageBuilder::new("Labelled")
            .add_audio_track(audio_path(), 2, 48000)
            .with_track_label("English Stereo")
            .expect("with_track_label")
            .build()
            .expect("build");

        let label = pkg.tracks[0].label.as_deref();
        assert_eq!(label, Some("English Stereo"));
    }

    #[test]
    fn test_track_label_no_track_error() {
        let result = ImfPackageBuilder::new("LabelFirst").with_track_label("should fail");
        assert!(result.is_err());
    }

    #[test]
    fn test_video_tracks_iterator() {
        let pkg = ImfPackageBuilder::new("Multi-Vid")
            .add_video_track(PathBuf::from("/a.mxf"), EditRate::fps_24())
            .add_video_track(PathBuf::from("/b.mxf"), EditRate::fps_24())
            .add_audio_track(audio_path(), 2, 48000)
            .build()
            .expect("build");

        assert_eq!(pkg.video_tracks().count(), 2);
        assert_eq!(pkg.audio_tracks().count(), 1);
    }
}
