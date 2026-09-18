//! Clip-level metadata attached to timeline clips.
//!
//! In addition to the free-form key/value `TimelineClipMetadata` / `ClipMetadataTemplate`
//! types, this module also provides richly-typed structures for the two most
//! common classes of embedded media metadata:
//!
//! * [`CodecInfo`]  — video/audio codec parameters derived from the container.
//! * [`ColorSpaceInfo`] — colour space, transfer function, and primaries metadata.

// ─────────────────────────────────────────────────────────────────────────────
// Typed media metadata
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the codec family used by a media stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodecKind {
    /// H.264 / AVC video codec.
    H264,
    /// H.265 / HEVC video codec.
    H265,
    /// AV1 video codec.
    Av1,
    /// VP9 video codec.
    Vp9,
    /// VP8 video codec.
    Vp8,
    /// Apple ProRes (all variants).
    ProRes,
    /// Avid DNxHD / DNxHR.
    DnxHd,
    /// JPEG 2000.
    Jpeg2000,
    /// AAC audio.
    Aac,
    /// Opus audio.
    Opus,
    /// PCM (uncompressed) audio.
    Pcm,
    /// FLAC lossless audio.
    Flac,
    /// Any other codec identified by its short name string.
    Other(String),
}

impl CodecKind {
    /// Returns a human-readable short name for the codec.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "h265",
            Self::Av1 => "av1",
            Self::Vp9 => "vp9",
            Self::Vp8 => "vp8",
            Self::ProRes => "prores",
            Self::DnxHd => "dnxhd",
            Self::Jpeg2000 => "jpeg2000",
            Self::Aac => "aac",
            Self::Opus => "opus",
            Self::Pcm => "pcm",
            Self::Flac => "flac",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Whether this codec is video.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::H264
                | Self::H265
                | Self::Av1
                | Self::Vp9
                | Self::Vp8
                | Self::ProRes
                | Self::DnxHd
                | Self::Jpeg2000
        )
    }

    /// Whether this codec is audio.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Aac | Self::Opus | Self::Pcm | Self::Flac)
    }
}

impl std::fmt::Display for CodecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Codec and stream parameters extracted from the container.
///
/// These fields are read-only once ingest is complete; they describe the
/// original encoded media and must not be modified by the editor.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecInfo {
    /// Codec family.
    pub codec: CodecKind,
    /// Codec profile string (e.g. `"High"`, `"4444"`, `"dnxhd_1080p_i_185x"`).
    pub profile: Option<String>,
    /// Codec level string (e.g. `"4.1"`, `"5.2"`).
    pub level: Option<String>,
    /// Bit-rate in bits per second.
    pub bit_rate: Option<u64>,
    /// Frame-rate numerator (video only).
    pub frame_rate_num: Option<u32>,
    /// Frame-rate denominator (video only).
    pub frame_rate_den: Option<u32>,
    /// Frame width in pixels (video only).
    pub width: Option<u32>,
    /// Frame height in pixels (video only).
    pub height: Option<u32>,
    /// Audio sample-rate in Hz (audio only).
    pub sample_rate: Option<u32>,
    /// Number of audio channels (audio only).
    pub channels: Option<u8>,
    /// Bit depth (bits per sample / pixel component).
    pub bit_depth: Option<u8>,
}

impl CodecInfo {
    /// Creates a minimal `CodecInfo` for the given codec.
    #[must_use]
    pub fn new(codec: CodecKind) -> Self {
        Self {
            codec,
            profile: None,
            level: None,
            bit_rate: None,
            frame_rate_num: None,
            frame_rate_den: None,
            width: None,
            height: None,
            sample_rate: None,
            channels: None,
            bit_depth: None,
        }
    }

    /// Convenience builder: set profile.
    #[must_use]
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Convenience builder: set video resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Convenience builder: set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, num: u32, den: u32) -> Self {
        self.frame_rate_num = Some(num);
        self.frame_rate_den = Some(den);
        self
    }

    /// Convenience builder: set audio parameters.
    #[must_use]
    pub fn with_audio(mut self, sample_rate: u32, channels: u8) -> Self {
        self.sample_rate = Some(sample_rate);
        self.channels = Some(channels);
        self
    }

    /// Returns the frame rate as a floating-point value, if both parts are set.
    #[must_use]
    pub fn frame_rate_f64(&self) -> Option<f64> {
        match (self.frame_rate_num, self.frame_rate_den) {
            (Some(n), Some(d)) if d != 0 => Some(f64::from(n) / f64::from(d)),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// ITU-R / SMPTE colour primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorPrimaries {
    /// BT.709 (HD, most SDR web video).
    Bt709,
    /// BT.2020 (UHD / HDR).
    Bt2020,
    /// DCI-P3 (digital cinema).
    DciP3,
    /// Display P3 (Apple devices).
    DisplayP3,
    /// BT.601 PAL.
    Bt601Pal,
    /// BT.601 NTSC.
    Bt601Ntsc,
    /// Unknown / unspecified.
    Unspecified,
}

impl ColorPrimaries {
    /// Returns the ITU-R short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt709 => "bt709",
            Self::Bt2020 => "bt2020",
            Self::DciP3 => "dci_p3",
            Self::DisplayP3 => "display_p3",
            Self::Bt601Pal => "bt601_pal",
            Self::Bt601Ntsc => "bt601_ntsc",
            Self::Unspecified => "unspecified",
        }
    }
}

impl std::fmt::Display for ColorPrimaries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// OETF / EOTF transfer characteristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferCharacteristic {
    /// BT.709 (gamma ~2.4).
    Bt709,
    /// sRGB (IEC 61966-2-1 piecewise gamma).
    Srgb,
    /// ST 2084 / PQ (Perceptual Quantizer, HDR).
    St2084Pq,
    /// ARIB STD-B67 / HLG (Hybrid Log-Gamma).
    Hlg,
    /// Linear (no gamma applied).
    Linear,
    /// Unknown / unspecified.
    Unspecified,
}

impl TransferCharacteristic {
    /// Returns the short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt709 => "bt709",
            Self::Srgb => "srgb",
            Self::St2084Pq => "st2084",
            Self::Hlg => "hlg",
            Self::Linear => "linear",
            Self::Unspecified => "unspecified",
        }
    }

    /// Whether this is a high dynamic range transfer function.
    #[must_use]
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::St2084Pq | Self::Hlg)
    }
}

impl std::fmt::Display for TransferCharacteristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// YCbCr / RGB matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixCoefficients {
    /// BT.709.
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl,
    /// BT.2020 constant luminance.
    Bt2020Cl,
    /// Identity / RGB (no matrix).
    Identity,
    /// BT.601 (625-line / PAL).
    Bt601,
    /// Unspecified.
    Unspecified,
}

impl MatrixCoefficients {
    /// Returns the short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt709 => "bt709",
            Self::Bt2020Ncl => "bt2020_ncl",
            Self::Bt2020Cl => "bt2020_cl",
            Self::Identity => "identity",
            Self::Bt601 => "bt601",
            Self::Unspecified => "unspecified",
        }
    }
}

impl std::fmt::Display for MatrixCoefficients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Colour range: full-swing vs. limited (studio-swing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorRange {
    /// Limited range (Y: 16–235, UV: 16–240 for 8-bit).
    Limited,
    /// Full range (0–255 for 8-bit).
    Full,
    /// Unspecified.
    Unspecified,
}

impl ColorRange {
    /// Returns the short name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Limited => "limited",
            Self::Full => "full",
            Self::Unspecified => "unspecified",
        }
    }
}

impl std::fmt::Display for ColorRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Complete colour-space descriptor for a video clip.
///
/// All fields are optional because any of them may be absent from a given
/// container or codec.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorSpaceInfo {
    /// Colour primaries (chromaticities of R, G, B).
    pub primaries: ColorPrimaries,
    /// Opto-electrical transfer characteristic (gamma / EOTF).
    pub transfer: TransferCharacteristic,
    /// YCbCr matrix coefficients (or `Identity` for RGB sources).
    pub matrix: MatrixCoefficients,
    /// Colour range.
    pub range: ColorRange,
    /// Maximum content light level in nits (HDR10 only).
    pub max_cll: Option<u32>,
    /// Maximum frame average light level in nits (HDR10 only).
    pub max_fall: Option<u32>,
}

impl ColorSpaceInfo {
    /// Creates a colour-space descriptor for a standard BT.709 SDR clip.
    #[must_use]
    pub fn bt709_sdr() -> Self {
        Self {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            matrix: MatrixCoefficients::Bt709,
            range: ColorRange::Limited,
            max_cll: None,
            max_fall: None,
        }
    }

    /// Creates a colour-space descriptor for an HDR10 (BT.2020 / PQ) clip.
    #[must_use]
    pub fn hdr10(max_cll: u32, max_fall: u32) -> Self {
        Self {
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::St2084Pq,
            matrix: MatrixCoefficients::Bt2020Ncl,
            range: ColorRange::Limited,
            max_cll: Some(max_cll),
            max_fall: Some(max_fall),
        }
    }

    /// Creates a colour-space descriptor for an HLG clip.
    #[must_use]
    pub fn hlg() -> Self {
        Self {
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Hlg,
            matrix: MatrixCoefficients::Bt2020Ncl,
            range: ColorRange::Limited,
            max_cll: None,
            max_fall: None,
        }
    }

    /// Whether this clip contains HDR content.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.transfer.is_hdr()
    }
}

/// Combined embedded media metadata: codec parameters plus colour space.
///
/// Attach an instance of this to a `TimelineClipMetadata` via
/// [`EmbeddedMediaMetadata`] rather than storing it inside free-form
/// key/value fields.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedMediaMetadata {
    /// Video stream codec info (absent for audio-only clips).
    pub video_codec: Option<CodecInfo>,
    /// Audio stream codec info (absent for video-only clips).
    pub audio_codec: Option<CodecInfo>,
    /// Colour-space information for the video stream.
    pub color_space: Option<ColorSpaceInfo>,
    /// Container format short name (e.g. `"mov"`, `"mkv"`, `"mp4"`).
    pub container: Option<String>,
    /// Duration of the source media in frames (may differ from clip duration).
    pub total_frames: Option<u64>,
    /// Timecode at the first frame of the source file (`HH:MM:SS:FF`).
    pub start_timecode: Option<String>,
}

impl EmbeddedMediaMetadata {
    /// Creates an empty embedded-media descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            video_codec: None,
            audio_codec: None,
            color_space: None,
            container: None,
            total_frames: None,
            start_timecode: None,
        }
    }

    /// Convenience builder: attach a video codec descriptor.
    #[must_use]
    pub fn with_video_codec(mut self, codec: CodecInfo) -> Self {
        self.video_codec = Some(codec);
        self
    }

    /// Convenience builder: attach an audio codec descriptor.
    #[must_use]
    pub fn with_audio_codec(mut self, codec: CodecInfo) -> Self {
        self.audio_codec = Some(codec);
        self
    }

    /// Convenience builder: attach colour-space information.
    #[must_use]
    pub fn with_color_space(mut self, cs: ColorSpaceInfo) -> Self {
        self.color_space = Some(cs);
        self
    }

    /// Convenience builder: set the container name.
    #[must_use]
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = Some(container.into());
        self
    }

    /// Whether the clip contains HDR video content.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.color_space
            .as_ref()
            .map_or(false, ColorSpaceInfo::is_hdr)
    }
}

impl Default for EmbeddedMediaMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// A single metadata field attached to a clip.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ClipMetadataField {
    /// Metadata key.
    pub key: String,
    /// Metadata value.
    pub value: String,
    /// Whether this field may be changed by the user.
    pub editable: bool,
}

impl ClipMetadataField {
    /// Create a new metadata field.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>, editable: bool) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            editable,
        }
    }

    /// Returns `true` when this field cannot be modified.
    #[must_use]
    pub fn is_readonly(&self) -> bool {
        !self.editable
    }
}

/// All metadata fields associated with one timeline clip.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TimelineClipMetadata {
    /// Identifier of the clip this metadata belongs to.
    pub clip_id: u64,
    /// Ordered list of metadata fields.
    pub fields: Vec<ClipMetadataField>,
}

impl TimelineClipMetadata {
    /// Create empty metadata for the specified clip.
    #[must_use]
    pub fn new(clip_id: u64) -> Self {
        Self {
            clip_id,
            fields: Vec::new(),
        }
    }

    /// Retrieve the value of a field by key.
    ///
    /// Returns `None` when the key is not present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.key == key)
            .map(|f| f.value.as_str())
    }

    /// Update the value of an existing field.
    ///
    /// Returns `true` on success, `false` when the key is not found or the
    /// field is read-only.
    pub fn set(&mut self, key: &str, value: &str) -> bool {
        if let Some(field) = self.fields.iter_mut().find(|f| f.key == key) {
            if field.editable {
                field.value = value.to_string();
                return true;
            }
        }
        false
    }

    /// Append a new field.
    pub fn add_field(&mut self, key: impl Into<String>, value: impl Into<String>, editable: bool) {
        self.fields
            .push(ClipMetadataField::new(key, value, editable));
    }

    /// Number of fields currently stored.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Iterate over all editable fields.
    pub fn editable_fields(&self) -> impl Iterator<Item = &ClipMetadataField> {
        self.fields.iter().filter(|f| f.editable)
    }
}

/// A reusable template that produces pre-populated [`TimelineClipMetadata`].
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ClipMetadataTemplate {
    /// Human-readable name for this template.
    pub name: String,
    /// Field definitions: (key, `default_value`, editable).
    pub fields: Vec<(String, String, bool)>,
}

impl ClipMetadataTemplate {
    /// Create a new template with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field definition to this template.
    pub fn add(
        &mut self,
        key: impl Into<String>,
        default_value: impl Into<String>,
        editable: bool,
    ) {
        self.fields
            .push((key.into(), default_value.into(), editable));
    }

    /// Instantiate metadata for `clip_id` using the template defaults.
    #[must_use]
    pub fn apply(&self, clip_id: u64) -> TimelineClipMetadata {
        let mut meta = TimelineClipMetadata::new(clip_id);
        for (key, value, editable) in &self.fields {
            meta.add_field(key.clone(), value.clone(), *editable);
        }
        meta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_readonly() {
        let f = ClipMetadataField::new("source", "camera_a", false);
        assert!(f.is_readonly());
    }

    #[test]
    fn test_field_editable() {
        let f = ClipMetadataField::new("comment", "review later", true);
        assert!(!f.is_readonly());
    }

    #[test]
    fn test_metadata_new_empty() {
        let m = TimelineClipMetadata::new(42);
        assert_eq!(m.clip_id, 42);
        assert_eq!(m.field_count(), 0);
    }

    #[test]
    fn test_add_and_get_field() {
        let mut m = TimelineClipMetadata::new(1);
        m.add_field("scene", "int_office_day", true);
        assert_eq!(m.get("scene"), Some("int_office_day"));
    }

    #[test]
    fn test_get_missing_key() {
        let m = TimelineClipMetadata::new(1);
        assert_eq!(m.get("nothing"), None);
    }

    #[test]
    fn test_set_editable_field() {
        let mut m = TimelineClipMetadata::new(5);
        m.add_field("note", "original", true);
        let ok = m.set("note", "updated");
        assert!(ok);
        assert_eq!(m.get("note"), Some("updated"));
    }

    #[test]
    fn test_set_readonly_field_fails() {
        let mut m = TimelineClipMetadata::new(5);
        m.add_field("tc_in", "01:00:00:00", false);
        let ok = m.set("tc_in", "01:00:01:00");
        assert!(!ok);
        assert_eq!(m.get("tc_in"), Some("01:00:00:00"));
    }

    #[test]
    fn test_set_missing_key_fails() {
        let mut m = TimelineClipMetadata::new(5);
        let ok = m.set("does_not_exist", "value");
        assert!(!ok);
    }

    #[test]
    fn test_field_count() {
        let mut m = TimelineClipMetadata::new(7);
        m.add_field("a", "1", true);
        m.add_field("b", "2", false);
        m.add_field("c", "3", true);
        assert_eq!(m.field_count(), 3);
    }

    #[test]
    fn test_editable_fields_iterator() {
        let mut m = TimelineClipMetadata::new(9);
        m.add_field("ro", "x", false);
        m.add_field("rw1", "y", true);
        m.add_field("rw2", "z", true);
        let editable: Vec<_> = m.editable_fields().collect();
        assert_eq!(editable.len(), 2);
    }

    #[test]
    fn test_template_apply() {
        let mut tmpl = ClipMetadataTemplate::new("broadcast");
        tmpl.add("channel", "BBC1", false);
        tmpl.add("rating", "PG", true);
        let meta = tmpl.apply(99);
        assert_eq!(meta.clip_id, 99);
        assert_eq!(meta.field_count(), 2);
        assert_eq!(meta.get("channel"), Some("BBC1"));
        assert_eq!(meta.get("rating"), Some("PG"));
    }

    #[test]
    fn test_template_readonly_field_preserved() {
        let mut tmpl = ClipMetadataTemplate::new("archive");
        tmpl.add("ingest_date", "2026-01-01", false);
        let meta = tmpl.apply(1);
        let field = meta
            .fields
            .iter()
            .find(|f| f.key == "ingest_date")
            .expect("should succeed in test");
        assert!(field.is_readonly());
    }

    #[test]
    fn test_template_empty_apply() {
        let tmpl = ClipMetadataTemplate::new("empty");
        let meta = tmpl.apply(0);
        assert_eq!(meta.field_count(), 0);
    }
}
