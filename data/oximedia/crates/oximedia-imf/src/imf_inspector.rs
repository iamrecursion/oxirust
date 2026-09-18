//! IMF package inspector — produces detailed human-readable and JSON inspection reports.
//!
//! Introspects an [`InspectablePackage`] (a lightweight description) and
//! generates an [`InspectionReport`] that summarises:
//! - Essence track inventory (count per kind)
//! - Total timeline duration
//! - Video resolution / codec
//! - Audio channel count and sample format
//! - Subtitle / caption track presence
//! - PKL asset count and total size
//! - Hash algorithm in use

use std::fmt;

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Sample format used for audio tracks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioSampleFormat {
    /// 16-bit signed PCM.
    Pcm16,
    /// 24-bit signed PCM.
    Pcm24,
    /// 32-bit signed PCM.
    Pcm32,
    /// 32-bit IEEE 754 float.
    Float32,
    /// Immersive Audio Bitstream.
    Iab,
    /// Unknown/unspecified.
    Unknown(String),
}

impl fmt::Display for AudioSampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pcm16 => write!(f, "PCM-16"),
            Self::Pcm24 => write!(f, "PCM-24"),
            Self::Pcm32 => write!(f, "PCM-32"),
            Self::Float32 => write!(f, "Float32"),
            Self::Iab => write!(f, "IAB"),
            Self::Unknown(s) => write!(f, "Unknown({s})"),
        }
    }
}

impl AudioSampleFormat {
    /// Parse from a string description.
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "PCM16" | "PCM-16" | "PCM_16" => Self::Pcm16,
            "PCM24" | "PCM-24" | "PCM_24" => Self::Pcm24,
            "PCM32" | "PCM-32" | "PCM_32" => Self::Pcm32,
            "FLOAT32" | "FLOAT-32" => Self::Float32,
            "IAB" => Self::Iab,
            other => Self::Unknown(other.to_string()),
        }
    }
}

/// Colour space / transfer function descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// ITU-R BT.709 (HD SDR).
    Rec709,
    /// ITU-R BT.2020 SDR.
    Rec2020,
    /// ITU-R BT.2020 with PQ transfer (HDR10).
    Rec2020Pq,
    /// ITU-R BT.2020 with HLG transfer.
    Rec2020Hlg,
    /// Academy Color Encoding System.
    Aces,
    /// Unknown / not specified.
    #[default]
    Unknown,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rec709 => write!(f, "Rec.709"),
            Self::Rec2020 => write!(f, "Rec.2020"),
            Self::Rec2020Pq => write!(f, "Rec.2020/PQ (HDR10)"),
            Self::Rec2020Hlg => write!(f, "Rec.2020/HLG"),
            Self::Aces => write!(f, "ACES"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Description of a single video track found in the package.
#[derive(Clone, Debug)]
pub struct VideoTrackInfo {
    /// Track UUID.
    pub id: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Edit rate numerator.
    pub frame_rate_num: u32,
    /// Edit rate denominator.
    pub frame_rate_den: u32,
    /// Codec / essence type string (e.g. "JPEG2000", "AVC", "ProRes").
    pub codec: String,
    /// Color space.
    pub color_space: ColorSpace,
    /// Bit depth.
    pub bit_depth: u32,
}

impl VideoTrackInfo {
    /// Frame rate as f64.
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_rate_f64(&self) -> f64 {
        if self.frame_rate_den == 0 {
            return 0.0;
        }
        self.frame_rate_num as f64 / self.frame_rate_den as f64
    }
}

impl fmt::Display for VideoTrackInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}x{} @ {}/{} fps codec={} cs={} {}bit",
            self.id,
            self.width,
            self.height,
            self.frame_rate_num,
            self.frame_rate_den,
            self.codec,
            self.color_space,
            self.bit_depth
        )
    }
}

/// Description of a single audio track found in the package.
#[derive(Clone, Debug)]
pub struct AudioTrackInfo {
    /// Track UUID.
    pub id: String,
    /// Number of channels.
    pub channels: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Sample format.
    pub format: AudioSampleFormat,
    /// Language tag (e.g. `"en"`, `"fr"`).
    pub language: Option<String>,
}

impl fmt::Display for AudioTrackInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lang = self
            .language
            .as_deref()
            .map(|l| format!(" lang={l}"))
            .unwrap_or_default();
        write!(
            f,
            "[{}] {}ch @ {}Hz format={}{}",
            self.id, self.channels, self.sample_rate, self.format, lang
        )
    }
}

/// Description of a subtitle/caption track.
#[derive(Clone, Debug)]
pub struct SubtitleTrackInfo {
    /// Track UUID.
    pub id: String,
    /// Format (e.g. "TTML", "SRT", "IMSC1").
    pub format: String,
    /// Language tag.
    pub language: Option<String>,
}

impl fmt::Display for SubtitleTrackInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lang = self
            .language
            .as_deref()
            .map(|l| format!(" lang={l}"))
            .unwrap_or_default();
        write!(f, "[{}] format={}{}", self.id, self.format, lang)
    }
}

// ---------------------------------------------------------------------------
// InspectablePackage — lightweight input DTO
// ---------------------------------------------------------------------------

/// A lightweight description of an IMF package suitable for inspection.
///
/// Build this from your actual [`crate::ImfPackage`] or synthesise it for
/// testing.
#[derive(Debug, Clone, Default)]
pub struct InspectablePackage {
    /// Package UUID.
    pub package_id: String,
    /// Content title from the primary CPL.
    pub title: String,
    /// Primary edit rate numerator.
    pub edit_rate_num: u32,
    /// Primary edit rate denominator.
    pub edit_rate_den: u32,
    /// Total duration of the primary CPL in edit units.
    pub total_duration_eu: u64,
    /// Video tracks.
    pub video_tracks: Vec<VideoTrackInfo>,
    /// Audio tracks.
    pub audio_tracks: Vec<AudioTrackInfo>,
    /// Subtitle tracks.
    pub subtitle_tracks: Vec<SubtitleTrackInfo>,
    /// Number of PKL assets.
    pub pkl_asset_count: usize,
    /// Total size of all PKL assets in bytes.
    pub pkl_total_size_bytes: u64,
    /// Hash algorithm from the PKL.
    pub pkl_hash_algorithm: String,
    /// Number of CPLs.
    pub cpl_count: usize,
    /// Whether the package has supplemental packages.
    pub has_supplemental: bool,
    /// Application profile URN (if known).
    pub application_profile_urn: Option<String>,
}

impl InspectablePackage {
    /// Compute total duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_duration_seconds(&self) -> f64 {
        if self.edit_rate_num == 0 || self.edit_rate_den == 0 {
            return 0.0;
        }
        self.total_duration_eu as f64 * self.edit_rate_den as f64 / self.edit_rate_num as f64
    }
}

// ---------------------------------------------------------------------------
// InspectionReport
// ---------------------------------------------------------------------------

/// Full inspection report produced by [`ImfInspector::inspect`].
#[derive(Debug, Clone)]
pub struct InspectionReport {
    /// Package identifier.
    pub package_id: String,
    /// Content title.
    pub title: String,
    /// Number of video tracks.
    pub video_track_count: usize,
    /// Number of audio tracks.
    pub audio_track_count: usize,
    /// Number of subtitle tracks.
    pub subtitle_track_count: usize,
    /// Total duration in seconds.
    pub total_duration_seconds: f64,
    /// Total duration in edit units.
    pub total_duration_eu: u64,
    /// Primary edit rate as `"num/den"` string.
    pub edit_rate: String,
    /// Per-track video information.
    pub video_tracks: Vec<VideoTrackInfo>,
    /// Per-track audio information.
    pub audio_tracks: Vec<AudioTrackInfo>,
    /// Per-track subtitle information.
    pub subtitle_tracks: Vec<SubtitleTrackInfo>,
    /// PKL asset count.
    pub pkl_asset_count: usize,
    /// PKL total size in bytes.
    pub pkl_total_size_bytes: u64,
    /// Hash algorithm in use.
    pub hash_algorithm: String,
    /// Number of CPLs.
    pub cpl_count: usize,
    /// Whether supplemental packages are present.
    pub has_supplemental: bool,
    /// Application profile URN if known.
    pub application_profile_urn: Option<String>,
}

impl InspectionReport {
    /// Generate a human-readable multi-line text report.
    pub fn to_text(&self) -> String {
        let mut s = String::new();

        s.push_str("=== IMF Package Inspection Report ===\n");
        s.push_str(&format!("Package ID   : {}\n", self.package_id));
        s.push_str(&format!("Title        : {}\n", self.title));
        s.push_str(&format!("CPLs         : {}\n", self.cpl_count));
        s.push_str(&format!("Edit rate    : {}\n", self.edit_rate));
        s.push_str(&format!(
            "Duration     : {:.3}s ({} edit units)\n",
            self.total_duration_seconds, self.total_duration_eu
        ));

        if let Some(ref urn) = self.application_profile_urn {
            s.push_str(&format!("Profile      : {urn}\n"));
        }

        s.push_str(&format!("Hash algo    : {}\n", self.hash_algorithm));
        s.push_str(&format!("PKL assets   : {}\n", self.pkl_asset_count));
        s.push_str(&format!(
            "PKL size     : {} bytes ({:.2} GiB)\n",
            self.pkl_total_size_bytes,
            self.pkl_total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));

        if self.has_supplemental {
            s.push_str("Supplemental : yes\n");
        }

        s.push('\n');

        // Video
        s.push_str(&format!("Video tracks ({}):\n", self.video_track_count));
        if self.video_tracks.is_empty() {
            s.push_str("  (none)\n");
        } else {
            for vt in &self.video_tracks {
                s.push_str(&format!("  {vt}\n"));
            }
        }

        // Audio
        s.push_str(&format!("\nAudio tracks ({}):\n", self.audio_track_count));
        if self.audio_tracks.is_empty() {
            s.push_str("  (none)\n");
        } else {
            for at in &self.audio_tracks {
                s.push_str(&format!("  {at}\n"));
            }
        }

        // Subtitles
        s.push_str(&format!(
            "\nSubtitle tracks ({}):\n",
            self.subtitle_track_count
        ));
        if self.subtitle_tracks.is_empty() {
            s.push_str("  (none)\n");
        } else {
            for st in &self.subtitle_tracks {
                s.push_str(&format!("  {st}\n"));
            }
        }

        s
    }

    /// Serialize to a JSON string.
    ///
    /// Uses hand-rolled serialization to avoid pulling in `serde_json` just
    /// for this module.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!(
            "  \"package_id\": \"{}\",\n",
            escape_json(&self.package_id)
        ));
        s.push_str(&format!("  \"title\": \"{}\",\n", escape_json(&self.title)));
        s.push_str(&format!("  \"cpl_count\": {},\n", self.cpl_count));
        s.push_str(&format!(
            "  \"edit_rate\": \"{}\",\n",
            escape_json(&self.edit_rate)
        ));
        s.push_str(&format!(
            "  \"total_duration_seconds\": {:.6},\n",
            self.total_duration_seconds
        ));
        s.push_str(&format!(
            "  \"total_duration_eu\": {},\n",
            self.total_duration_eu
        ));
        s.push_str(&format!(
            "  \"video_track_count\": {},\n",
            self.video_track_count
        ));
        s.push_str(&format!(
            "  \"audio_track_count\": {},\n",
            self.audio_track_count
        ));
        s.push_str(&format!(
            "  \"subtitle_track_count\": {},\n",
            self.subtitle_track_count
        ));
        s.push_str(&format!(
            "  \"pkl_asset_count\": {},\n",
            self.pkl_asset_count
        ));
        s.push_str(&format!(
            "  \"pkl_total_size_bytes\": {},\n",
            self.pkl_total_size_bytes
        ));
        s.push_str(&format!(
            "  \"hash_algorithm\": \"{}\",\n",
            escape_json(&self.hash_algorithm)
        ));
        s.push_str(&format!(
            "  \"has_supplemental\": {},\n",
            self.has_supplemental
        ));

        // application profile
        match &self.application_profile_urn {
            Some(urn) => s.push_str(&format!(
                "  \"application_profile_urn\": \"{}\",\n",
                escape_json(urn)
            )),
            None => s.push_str("  \"application_profile_urn\": null,\n"),
        }

        // Video tracks array
        s.push_str("  \"video_tracks\": [\n");
        for (i, vt) in self.video_tracks.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!("      \"id\": \"{}\",\n", escape_json(&vt.id)));
            s.push_str(&format!("      \"width\": {},\n", vt.width));
            s.push_str(&format!("      \"height\": {},\n", vt.height));
            s.push_str(&format!(
                "      \"frame_rate_num\": {},\n",
                vt.frame_rate_num
            ));
            s.push_str(&format!(
                "      \"frame_rate_den\": {},\n",
                vt.frame_rate_den
            ));
            s.push_str(&format!(
                "      \"codec\": \"{}\",\n",
                escape_json(&vt.codec)
            ));
            s.push_str(&format!("      \"bit_depth\": {}\n", vt.bit_depth));
            if i + 1 < self.video_tracks.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ],\n");

        // Audio tracks array
        s.push_str("  \"audio_tracks\": [\n");
        for (i, at) in self.audio_tracks.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!("      \"id\": \"{}\",\n", escape_json(&at.id)));
            s.push_str(&format!("      \"channels\": {},\n", at.channels));
            s.push_str(&format!("      \"sample_rate\": {},\n", at.sample_rate));
            s.push_str(&format!(
                "      \"format\": \"{}\"\n",
                escape_json(&at.format.to_string())
            ));
            if i + 1 < self.audio_tracks.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ],\n");

        // Subtitle tracks array
        s.push_str("  \"subtitle_tracks\": [\n");
        for (i, st) in self.subtitle_tracks.iter().enumerate() {
            s.push_str("    {\n");
            s.push_str(&format!("      \"id\": \"{}\",\n", escape_json(&st.id)));
            s.push_str(&format!(
                "      \"format\": \"{}\"\n",
                escape_json(&st.format)
            ));
            if i + 1 < self.subtitle_tracks.len() {
                s.push_str("    },\n");
            } else {
                s.push_str("    }\n");
            }
        }
        s.push_str("  ]\n");

        s.push('}');
        s
    }
}

/// Minimal JSON string escaping.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ImfInspector
// ---------------------------------------------------------------------------

/// Inspects an [`InspectablePackage`] and produces a detailed [`InspectionReport`].
pub struct ImfInspector;

impl ImfInspector {
    /// Inspect `package` and produce a full [`InspectionReport`].
    pub fn inspect(package: &InspectablePackage) -> InspectionReport {
        let edit_rate = if package.edit_rate_den == 1 {
            format!("{}", package.edit_rate_num)
        } else {
            format!("{}/{}", package.edit_rate_num, package.edit_rate_den)
        };

        InspectionReport {
            package_id: package.package_id.clone(),
            title: package.title.clone(),
            video_track_count: package.video_tracks.len(),
            audio_track_count: package.audio_tracks.len(),
            subtitle_track_count: package.subtitle_tracks.len(),
            total_duration_seconds: package.total_duration_seconds(),
            total_duration_eu: package.total_duration_eu,
            edit_rate,
            video_tracks: package.video_tracks.clone(),
            audio_tracks: package.audio_tracks.clone(),
            subtitle_tracks: package.subtitle_tracks.clone(),
            pkl_asset_count: package.pkl_asset_count,
            pkl_total_size_bytes: package.pkl_total_size_bytes,
            hash_algorithm: package.pkl_hash_algorithm.clone(),
            cpl_count: package.cpl_count,
            has_supplemental: package.has_supplemental,
            application_profile_urn: package.application_profile_urn.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_video_track(id: &str) -> VideoTrackInfo {
        VideoTrackInfo {
            id: id.to_string(),
            width: 1920,
            height: 1080,
            frame_rate_num: 24,
            frame_rate_den: 1,
            codec: "JPEG2000".to_string(),
            color_space: ColorSpace::Rec709,
            bit_depth: 12,
        }
    }

    fn sample_audio_track(id: &str) -> AudioTrackInfo {
        AudioTrackInfo {
            id: id.to_string(),
            channels: 8,
            sample_rate: 48000,
            format: AudioSampleFormat::Pcm24,
            language: Some("en".to_string()),
        }
    }

    fn sample_subtitle_track(id: &str) -> SubtitleTrackInfo {
        SubtitleTrackInfo {
            id: id.to_string(),
            format: "TTML".to_string(),
            language: Some("en".to_string()),
        }
    }

    fn sample_package() -> InspectablePackage {
        InspectablePackage {
            package_id: "urn:uuid:test-0001".to_string(),
            title: "Test Feature Film".to_string(),
            edit_rate_num: 24,
            edit_rate_den: 1,
            total_duration_eu: 172800, // 2 hours at 24fps
            video_tracks: vec![sample_video_track("vid-001")],
            audio_tracks: vec![sample_audio_track("aud-001"), sample_audio_track("aud-002")],
            subtitle_tracks: vec![sample_subtitle_track("sub-001")],
            pkl_asset_count: 5,
            pkl_total_size_bytes: 2_147_483_648, // 2 GiB
            pkl_hash_algorithm: "SHA-256".to_string(),
            cpl_count: 1,
            has_supplemental: false,
            application_profile_urn: Some(
                "urn:smpte:ul:060E2B34.04010105.0E090604.00000000".to_string(),
            ),
        }
    }

    #[test]
    fn test_inspect_track_counts() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        assert_eq!(report.video_track_count, 1);
        assert_eq!(report.audio_track_count, 2);
        assert_eq!(report.subtitle_track_count, 1);
    }

    #[test]
    fn test_inspect_duration_seconds() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        // 172800 edit units / 24 fps = 7200 seconds = 2 hours
        assert!((report.total_duration_seconds - 7200.0).abs() < 0.001);
    }

    #[test]
    fn test_inspect_edit_rate_format() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        assert_eq!(report.edit_rate, "24");

        let mut ntsc = sample_package();
        ntsc.edit_rate_num = 30000;
        ntsc.edit_rate_den = 1001;
        let r = ImfInspector::inspect(&ntsc);
        assert_eq!(r.edit_rate, "30000/1001");
    }

    #[test]
    fn test_inspect_video_resolution() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        let vt = &report.video_tracks[0];
        assert_eq!(vt.width, 1920);
        assert_eq!(vt.height, 1080);
        assert_eq!(vt.codec, "JPEG2000");
    }

    #[test]
    fn test_inspect_audio_channels() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        let at = &report.audio_tracks[0];
        assert_eq!(at.channels, 8);
        assert_eq!(at.sample_rate, 48000);
        assert_eq!(at.format, AudioSampleFormat::Pcm24);
    }

    #[test]
    fn test_inspect_hash_algorithm() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        assert_eq!(report.hash_algorithm, "SHA-256");
    }

    #[test]
    fn test_text_report_contains_title() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        let text = report.to_text();
        assert!(text.contains("Test Feature Film"));
        assert!(text.contains("Video tracks (1)"));
        assert!(text.contains("Audio tracks (2)"));
        assert!(text.contains("Subtitle tracks (1)"));
    }

    #[test]
    fn test_json_report_is_valid_structure() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        let json = report.to_json();
        // Basic structural checks
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"title\""));
        assert!(json.contains("\"video_tracks\""));
        assert!(json.contains("\"audio_tracks\""));
        assert!(json.contains("\"subtitle_tracks\""));
    }

    #[test]
    fn test_json_report_contains_track_count() {
        let pkg = sample_package();
        let report = ImfInspector::inspect(&pkg);
        let json = report.to_json();
        assert!(json.contains("\"video_track_count\": 1"));
        assert!(json.contains("\"audio_track_count\": 2"));
    }

    #[test]
    fn test_empty_package_inspection() {
        let pkg = InspectablePackage {
            package_id: "urn:uuid:empty-001".to_string(),
            title: "Empty Package".to_string(),
            edit_rate_num: 25,
            edit_rate_den: 1,
            ..InspectablePackage::default()
        };
        let report = ImfInspector::inspect(&pkg);
        assert_eq!(report.video_track_count, 0);
        assert_eq!(report.audio_track_count, 0);
        assert_eq!(report.total_duration_seconds, 0.0);
        let text = report.to_text();
        assert!(text.contains("Empty Package"));
    }
}
