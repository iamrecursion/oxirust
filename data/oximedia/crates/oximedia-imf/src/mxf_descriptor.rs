//! MXF essence container label management and descriptor information.
//!
//! IMF packages wrap media essence in MXF (Material eXchange Format)
//! containers. Each essence track is identified by a SMPTE-registered
//! container label (UL). This module provides types for representing those
//! labels, querying descriptor properties, and collecting essence tracks.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// EssenceContainerLabel
// ---------------------------------------------------------------------------

/// SMPTE essence container label identifying the codec / wrapping of an MXF
/// track file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EssenceContainerLabel {
    /// JPEG 2000 codestream (SMPTE ST 422).
    Jpeg2000,
    /// MPEG-H HEVC / H.265 (SMPTE ST 2067-21).
    Hevc,
    /// ProRes (SMPTE RDD 44).
    ProRes,
    /// Uncompressed video (SMPTE ST 384).
    Uncompressed,
    /// PCM audio (SMPTE ST 382).
    PcmAudio,
    /// Timed text / IMSC1 (SMPTE ST 2067-5).
    TimedText,
    /// IAB immersive audio (SMPTE ST 2067-201).
    ImmersiveAudio,
    /// Ancillary data essence.
    Ancillary,
    /// Unknown / custom label with raw UL bytes.
    Custom([u8; 16]),
}

impl EssenceContainerLabel {
    /// Returns `true` if this label represents a video codec.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(
            self,
            Self::Jpeg2000 | Self::Hevc | Self::ProRes | Self::Uncompressed
        )
    }

    /// Returns `true` if this label represents an audio codec.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::PcmAudio | Self::ImmersiveAudio)
    }

    /// Returns `true` if this label represents a text/subtitle track.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::TimedText)
    }

    /// Short label used in reports.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Jpeg2000 => "JPEG2000",
            Self::Hevc => "HEVC",
            Self::ProRes => "ProRes",
            Self::Uncompressed => "Uncompressed",
            Self::PcmAudio => "PCM",
            Self::TimedText => "TimedText",
            Self::ImmersiveAudio => "IAB",
            Self::Ancillary => "Ancillary",
            Self::Custom(_) => "Custom",
        }
    }
}

impl fmt::Display for EssenceContainerLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// MxfDescriptorInfo
// ---------------------------------------------------------------------------

/// High-level metadata extracted from an MXF file descriptor.
///
/// This summarises the key properties without needing to parse the full MXF
/// structure. An IMF validator can use this to cross-check CPL declarations
/// against the actual track file contents.
#[derive(Debug, Clone)]
pub struct MxfDescriptorInfo {
    /// Essence container label for this track file.
    pub container_label: EssenceContainerLabel,
    /// Track file duration in edit units (0 if unknown).
    pub duration_edit_units: u64,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Stored width in pixels (video only, 0 for audio/text).
    pub stored_width: u32,
    /// Stored height in pixels (video only, 0 for audio/text).
    pub stored_height: u32,
    /// Bit depth (video: bits per component, audio: bits per sample).
    pub bit_depth: u16,
    /// Audio channel count (0 for non-audio).
    pub channel_count: u16,
    /// Audio sample rate in Hz (0 for non-audio).
    pub sample_rate_hz: u32,
}

impl MxfDescriptorInfo {
    /// Create a new descriptor info for a video track.
    #[must_use]
    pub fn video(
        label: EssenceContainerLabel,
        width: u32,
        height: u32,
        bit_depth: u16,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            container_label: label,
            duration_edit_units: 0,
            edit_rate_num,
            edit_rate_den,
            stored_width: width,
            stored_height: height,
            bit_depth,
            channel_count: 0,
            sample_rate_hz: 0,
        }
    }

    /// Create a new descriptor info for an audio track.
    #[must_use]
    pub fn audio(
        bit_depth: u16,
        channel_count: u16,
        sample_rate_hz: u32,
        edit_rate_num: u32,
        edit_rate_den: u32,
    ) -> Self {
        Self {
            container_label: EssenceContainerLabel::PcmAudio,
            duration_edit_units: 0,
            edit_rate_num,
            edit_rate_den,
            stored_width: 0,
            stored_height: 0,
            bit_depth,
            channel_count,
            sample_rate_hz,
        }
    }

    /// Set the duration.
    #[must_use]
    pub fn with_duration(mut self, edit_units: u64) -> Self {
        self.duration_edit_units = edit_units;
        self
    }

    /// Edit rate as a floating-point value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn edit_rate_f64(&self) -> f64 {
        if self.edit_rate_den == 0 {
            return 0.0;
        }
        f64::from(self.edit_rate_num) / f64::from(self.edit_rate_den)
    }

    /// Duration in seconds (using edit rate).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        let rate = self.edit_rate_f64();
        if rate <= 0.0 {
            return 0.0;
        }
        self.duration_edit_units as f64 / rate
    }

    /// Returns `true` for UHD resolution (width >= 3840).
    #[must_use]
    pub fn is_uhd(&self) -> bool {
        self.stored_width >= 3840
    }

    /// Returns `true` if all essential fields look sane.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.edit_rate_num > 0 && self.edit_rate_den > 0 && self.bit_depth > 0
    }
}

impl fmt::Display for MxfDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}x{} {}bit {}/{}fps",
            self.container_label,
            self.stored_width,
            self.stored_height,
            self.bit_depth,
            self.edit_rate_num,
            self.edit_rate_den,
        )
    }
}

// ---------------------------------------------------------------------------
// DescriptorTrack
// ---------------------------------------------------------------------------

/// Groups a track index together with its MXF descriptor information.
#[derive(Debug, Clone)]
pub struct DescriptorTrack {
    /// Zero-based track index within the package.
    pub track_index: u32,
    /// UUID of the track file (as 32-hex-char string, no dashes).
    pub track_file_id: String,
    /// Descriptor metadata.
    pub descriptor: MxfDescriptorInfo,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl DescriptorTrack {
    /// Create a new descriptor track entry.
    #[must_use]
    pub fn new(
        track_index: u32,
        track_file_id: impl Into<String>,
        descriptor: MxfDescriptorInfo,
    ) -> Self {
        Self {
            track_index,
            track_file_id: track_file_id.into(),
            descriptor,
            label: None,
        }
    }

    /// Attach a label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Convenience: is this track carrying video essence?
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.descriptor.container_label.is_video()
    }

    /// Convenience: is this track carrying audio essence?
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.descriptor.container_label.is_audio()
    }
}

/// A collection of descriptor tracks for an IMF package.
#[derive(Debug, Clone, Default)]
pub struct DescriptorTrackList {
    tracks: Vec<DescriptorTrack>,
}

impl DescriptorTrackList {
    /// Create empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track.
    pub fn push(&mut self, track: DescriptorTrack) {
        self.tracks.push(track);
    }

    /// Number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Returns `true` if no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// All video tracks.
    #[must_use]
    pub fn video_tracks(&self) -> Vec<&DescriptorTrack> {
        self.tracks.iter().filter(|t| t.is_video()).collect()
    }

    /// All audio tracks.
    #[must_use]
    pub fn audio_tracks(&self) -> Vec<&DescriptorTrack> {
        self.tracks.iter().filter(|t| t.is_audio()).collect()
    }

    /// Iterator over all tracks.
    pub fn iter(&self) -> impl Iterator<Item = &DescriptorTrack> {
        self.tracks.iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_is_video() {
        assert!(EssenceContainerLabel::Jpeg2000.is_video());
        assert!(EssenceContainerLabel::Hevc.is_video());
        assert!(EssenceContainerLabel::ProRes.is_video());
        assert!(EssenceContainerLabel::Uncompressed.is_video());
        assert!(!EssenceContainerLabel::PcmAudio.is_video());
    }

    #[test]
    fn test_label_is_audio() {
        assert!(EssenceContainerLabel::PcmAudio.is_audio());
        assert!(EssenceContainerLabel::ImmersiveAudio.is_audio());
        assert!(!EssenceContainerLabel::Jpeg2000.is_audio());
    }

    #[test]
    fn test_label_is_text() {
        assert!(EssenceContainerLabel::TimedText.is_text());
        assert!(!EssenceContainerLabel::PcmAudio.is_text());
    }

    #[test]
    fn test_label_display() {
        assert_eq!(format!("{}", EssenceContainerLabel::Hevc), "HEVC");
        assert_eq!(format!("{}", EssenceContainerLabel::PcmAudio), "PCM");
    }

    #[test]
    fn test_descriptor_video() {
        let d = MxfDescriptorInfo::video(EssenceContainerLabel::Jpeg2000, 3840, 2160, 12, 24, 1);
        assert!(d.is_uhd());
        assert!(d.is_valid());
        assert!((d.edit_rate_f64() - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_descriptor_audio() {
        let d = MxfDescriptorInfo::audio(24, 6, 48000, 48000, 1);
        assert_eq!(d.channel_count, 6);
        assert_eq!(d.sample_rate_hz, 48000);
        assert!(d.is_valid());
    }

    #[test]
    fn test_descriptor_duration_seconds() {
        let d = MxfDescriptorInfo::video(EssenceContainerLabel::Hevc, 1920, 1080, 10, 24, 1)
            .with_duration(240);
        let secs = d.duration_seconds();
        assert!((secs - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_descriptor_zero_edit_rate() {
        let mut d = MxfDescriptorInfo::audio(24, 2, 48000, 0, 0);
        d.edit_rate_num = 0;
        d.edit_rate_den = 0;
        assert!((d.edit_rate_f64()).abs() < f64::EPSILON);
        assert!((d.duration_seconds()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_descriptor_not_uhd() {
        let d = MxfDescriptorInfo::video(EssenceContainerLabel::Hevc, 1920, 1080, 10, 24, 1);
        assert!(!d.is_uhd());
    }

    #[test]
    fn test_descriptor_display() {
        let d = MxfDescriptorInfo::video(EssenceContainerLabel::Jpeg2000, 4096, 2160, 12, 24, 1);
        let s = format!("{d}");
        assert!(s.contains("JPEG2000"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn test_track_new_and_label() {
        let desc = MxfDescriptorInfo::video(EssenceContainerLabel::Hevc, 3840, 2160, 10, 24, 1);
        let t = DescriptorTrack::new(0, "abc123", desc).with_label("main video");
        assert_eq!(t.track_index, 0);
        assert_eq!(t.label.as_deref(), Some("main video"));
        assert!(t.is_video());
        assert!(!t.is_audio());
    }

    #[test]
    fn test_track_list_push_and_filter() {
        let mut list = DescriptorTrackList::new();
        assert!(list.is_empty());

        let vdesc =
            MxfDescriptorInfo::video(EssenceContainerLabel::Jpeg2000, 3840, 2160, 12, 24, 1);
        list.push(DescriptorTrack::new(0, "vid001", vdesc));

        let adesc = MxfDescriptorInfo::audio(24, 6, 48000, 48000, 1);
        list.push(DescriptorTrack::new(1, "aud001", adesc));

        assert_eq!(list.len(), 2);
        assert_eq!(list.video_tracks().len(), 1);
        assert_eq!(list.audio_tracks().len(), 1);
    }

    #[test]
    fn test_custom_label() {
        let label = EssenceContainerLabel::Custom([0u8; 16]);
        assert_eq!(label.label(), "Custom");
        assert!(!label.is_video());
        assert!(!label.is_audio());
    }

    #[test]
    fn test_descriptor_invalid() {
        let d = MxfDescriptorInfo {
            container_label: EssenceContainerLabel::Hevc,
            duration_edit_units: 0,
            edit_rate_num: 0,
            edit_rate_den: 1,
            stored_width: 0,
            stored_height: 0,
            bit_depth: 0,
            channel_count: 0,
            sample_rate_hz: 0,
        };
        assert!(!d.is_valid());
    }
}
