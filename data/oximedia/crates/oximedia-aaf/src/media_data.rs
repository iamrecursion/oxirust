//! AAF media data descriptors and essence locator management
//!
//! Provides types for describing media essence data within AAF files, including
//! codec identification, container format, and file locator references per
//! SMPTE ST 377-1 Section 14.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Codec family identifier for essence data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecFamily {
    /// Uncompressed video (e.g. UYVY, v210).
    Uncompressed,
    /// JPEG-based codecs.
    Jpeg,
    /// MPEG-2 Long-GOP or I-frame.
    Mpeg2,
    /// H.264 / AVC.
    Avc,
    /// HEVC / H.265.
    Hevc,
    /// Apple ProRes family.
    ProRes,
    /// Avid DNxHD / DNxHR.
    DnxHd,
    /// PCM audio (uncompressed).
    Pcm,
    /// AAC audio.
    Aac,
    /// Unknown / other codec.
    Other,
}

impl CodecFamily {
    /// Whether this codec family represents video data.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(
            self,
            Self::Uncompressed
                | Self::Jpeg
                | Self::Mpeg2
                | Self::Avc
                | Self::Hevc
                | Self::ProRes
                | Self::DnxHd
        )
    }

    /// Whether this codec family represents audio data.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Pcm | Self::Aac)
    }

    /// Human-readable codec label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Uncompressed => "Uncompressed",
            Self::Jpeg => "JPEG",
            Self::Mpeg2 => "MPEG-2",
            Self::Avc => "AVC/H.264",
            Self::Hevc => "HEVC/H.265",
            Self::ProRes => "ProRes",
            Self::DnxHd => "DNxHD/DNxHR",
            Self::Pcm => "PCM",
            Self::Aac => "AAC",
            Self::Other => "Other",
        }
    }
}

/// Container format wrapping the essence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// AAF-internal embedded essence.
    AafInternal,
    /// MXF OP-Atom (Avid-style, one track per file).
    MxfOpAtom,
    /// MXF OP1a (single file, interleaved).
    MxfOp1a,
    /// QuickTime / MOV.
    QuickTime,
    /// WAVE audio container.
    Wave,
    /// Unknown container.
    Unknown,
}

impl ContainerFormat {
    /// Typical file extension for this container.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::AafInternal => "aaf",
            Self::MxfOpAtom | Self::MxfOp1a => "mxf",
            Self::QuickTime => "mov",
            Self::Wave => "wav",
            Self::Unknown => "",
        }
    }
}

/// Locator describing where essence data resides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EssenceLocator {
    /// Essence is embedded within the AAF file itself.
    Internal {
        /// Stream index within the structured storage.
        stream_index: u32,
    },
    /// Essence is in an external file on the network or filesystem.
    External {
        /// URL or file path to the external media file.
        url: String,
    },
}

impl EssenceLocator {
    /// Create an internal locator.
    #[must_use]
    pub const fn internal(stream_index: u32) -> Self {
        Self::Internal { stream_index }
    }

    /// Create an external locator.
    #[must_use]
    pub fn external(url: impl Into<String>) -> Self {
        Self::External { url: url.into() }
    }

    /// Whether the essence is embedded.
    #[must_use]
    pub const fn is_internal(&self) -> bool {
        matches!(self, Self::Internal { .. })
    }

    /// Whether the essence is in an external file.
    #[must_use]
    pub const fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }

    /// Get the external URL if this is an external locator.
    #[must_use]
    pub fn external_url(&self) -> Option<&str> {
        match self {
            Self::External { url } => Some(url.as_str()),
            Self::Internal { .. } => None,
        }
    }
}

/// Video-specific parameters for media data.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoParameters {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bits per component (8, 10, 12, 16).
    pub bit_depth: u8,
    /// Horizontal aspect ratio component.
    pub aspect_ratio_h: u32,
    /// Vertical aspect ratio component.
    pub aspect_ratio_v: u32,
    /// Whether the video is interlaced.
    pub interlaced: bool,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
}

impl VideoParameters {
    /// Create new video parameters.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        width: u32,
        height: u32,
        bit_depth: u8,
        aspect_ratio_h: u32,
        aspect_ratio_v: u32,
        interlaced: bool,
        frame_rate_num: u32,
        frame_rate_den: u32,
    ) -> Self {
        Self {
            width,
            height,
            bit_depth,
            aspect_ratio_h,
            aspect_ratio_v,
            interlaced,
            frame_rate_num,
            frame_rate_den,
        }
    }

    /// Frame rate as floating-point fps.
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        if self.frame_rate_den == 0 {
            return 0.0;
        }
        f64::from(self.frame_rate_num) / f64::from(self.frame_rate_den)
    }

    /// Total number of pixels per frame.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Display aspect ratio as a float (e.g. 1.778 for 16:9).
    #[must_use]
    pub fn display_aspect_ratio(&self) -> f64 {
        if self.aspect_ratio_v == 0 {
            return 0.0;
        }
        f64::from(self.aspect_ratio_h) / f64::from(self.aspect_ratio_v)
    }

    /// Standard HD 1920x1080 @ 23.976 progressive.
    #[must_use]
    pub const fn hd_1080p_23_976() -> Self {
        Self::new(1920, 1080, 10, 16, 9, false, 24000, 1001)
    }

    /// Standard UHD 3840x2160 @ 23.976 progressive.
    #[must_use]
    pub const fn uhd_2160p_23_976() -> Self {
        Self::new(3840, 2160, 10, 16, 9, false, 24000, 1001)
    }
}

/// Audio-specific parameters for media data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioParameters {
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Bits per sample (16, 24, 32).
    pub bit_depth: u8,
    /// Number of audio channels.
    pub channels: u16,
    /// Channel layout label (e.g. "stereo", "5.1").
    pub layout: String,
}

impl AudioParameters {
    /// Create new audio parameters.
    #[must_use]
    pub fn new(sample_rate: u32, bit_depth: u8, channels: u16, layout: impl Into<String>) -> Self {
        Self {
            sample_rate,
            bit_depth,
            channels,
            layout: layout.into(),
        }
    }

    /// Standard stereo 48 kHz / 24-bit.
    #[must_use]
    pub fn stereo_48k_24bit() -> Self {
        Self::new(48000, 24, 2, "stereo")
    }

    /// Standard 5.1 surround 48 kHz / 24-bit.
    #[must_use]
    pub fn surround_5_1_48k() -> Self {
        Self::new(48000, 24, 6, "5.1")
    }

    /// Bytes per sample per channel.
    #[must_use]
    pub const fn bytes_per_sample(&self) -> u32 {
        (self.bit_depth as u32).div_ceil(8)
    }

    /// Bytes per second of audio (all channels).
    #[must_use]
    pub fn bytes_per_second(&self) -> u64 {
        u64::from(self.sample_rate) * u64::from(self.bytes_per_sample()) * u64::from(self.channels)
    }
}

/// Complete media data descriptor for an essence stream.
#[derive(Debug, Clone)]
pub struct MediaDataDescriptor {
    /// Unique descriptor identifier.
    pub descriptor_id: String,
    /// Codec family.
    pub codec: CodecFamily,
    /// Container format.
    pub container: ContainerFormat,
    /// Locator for the essence data.
    pub locator: EssenceLocator,
    /// Video parameters (if video essence).
    pub video: Option<VideoParameters>,
    /// Audio parameters (if audio essence).
    pub audio: Option<AudioParameters>,
    /// Total essence size in bytes (0 = unknown).
    pub data_size_bytes: u64,
    /// Additional key-value properties.
    pub properties: HashMap<String, String>,
}

impl MediaDataDescriptor {
    /// Create a new media data descriptor.
    #[must_use]
    pub fn new(
        descriptor_id: impl Into<String>,
        codec: CodecFamily,
        container: ContainerFormat,
        locator: EssenceLocator,
    ) -> Self {
        Self {
            descriptor_id: descriptor_id.into(),
            codec,
            container,
            locator,
            video: None,
            audio: None,
            data_size_bytes: 0,
            properties: HashMap::new(),
        }
    }

    /// Set video parameters.
    pub fn with_video(mut self, params: VideoParameters) -> Self {
        self.video = Some(params);
        self
    }

    /// Set audio parameters.
    pub fn with_audio(mut self, params: AudioParameters) -> Self {
        self.audio = Some(params);
        self
    }

    /// Set the data size.
    pub fn with_data_size(mut self, size: u64) -> Self {
        self.data_size_bytes = size;
        self
    }

    /// Set a property.
    pub fn set_property(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.properties.insert(key.into(), value.into());
    }

    /// Get a property.
    #[must_use]
    pub fn get_property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(String::as_str)
    }

    /// Whether this descriptor describes video essence.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.video.is_some() || self.codec.is_video()
    }

    /// Whether this descriptor describes audio essence.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.audio.is_some() || self.codec.is_audio()
    }

    /// Estimated bitrate in bits per second (0 if unknown).
    #[must_use]
    pub fn estimated_bitrate_bps(&self) -> u64 {
        if let Some(ref audio) = self.audio {
            return audio.bytes_per_second() * 8;
        }
        0
    }
}

/// Registry of media data descriptors for an AAF file.
#[derive(Debug, Default)]
pub struct MediaDataRegistry {
    /// All descriptors.
    descriptors: Vec<MediaDataDescriptor>,
}

impl MediaDataRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a descriptor.
    pub fn add(&mut self, desc: MediaDataDescriptor) {
        self.descriptors.push(desc);
    }

    /// Number of descriptors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Find a descriptor by its ID.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&MediaDataDescriptor> {
        self.descriptors.iter().find(|d| d.descriptor_id == id)
    }

    /// All video descriptors.
    #[must_use]
    pub fn video_descriptors(&self) -> Vec<&MediaDataDescriptor> {
        self.descriptors.iter().filter(|d| d.is_video()).collect()
    }

    /// All audio descriptors.
    #[must_use]
    pub fn audio_descriptors(&self) -> Vec<&MediaDataDescriptor> {
        self.descriptors.iter().filter(|d| d.is_audio()).collect()
    }

    /// Total data size in bytes across all descriptors.
    #[must_use]
    pub fn total_data_size(&self) -> u64 {
        self.descriptors.iter().map(|d| d.data_size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_family_is_video() {
        assert!(CodecFamily::ProRes.is_video());
        assert!(CodecFamily::DnxHd.is_video());
        assert!(!CodecFamily::Pcm.is_video());
        assert!(!CodecFamily::Aac.is_video());
    }

    #[test]
    fn test_codec_family_is_audio() {
        assert!(CodecFamily::Pcm.is_audio());
        assert!(CodecFamily::Aac.is_audio());
        assert!(!CodecFamily::ProRes.is_audio());
    }

    #[test]
    fn test_codec_family_label() {
        assert_eq!(CodecFamily::ProRes.label(), "ProRes");
        assert_eq!(CodecFamily::Hevc.label(), "HEVC/H.265");
    }

    #[test]
    fn test_container_format_extension() {
        assert_eq!(ContainerFormat::MxfOpAtom.extension(), "mxf");
        assert_eq!(ContainerFormat::QuickTime.extension(), "mov");
        assert_eq!(ContainerFormat::Wave.extension(), "wav");
    }

    #[test]
    fn test_essence_locator_internal() {
        let loc = EssenceLocator::internal(42);
        assert!(loc.is_internal());
        assert!(!loc.is_external());
        assert!(loc.external_url().is_none());
    }

    #[test]
    fn test_essence_locator_external() {
        let loc = EssenceLocator::external("/media/clip001.mxf");
        assert!(loc.is_external());
        assert!(!loc.is_internal());
        assert_eq!(loc.external_url(), Some("/media/clip001.mxf"));
    }

    #[test]
    fn test_video_parameters_frame_rate() {
        let vp = VideoParameters::hd_1080p_23_976();
        assert!((vp.frame_rate() - 23.976).abs() < 0.001);
    }

    #[test]
    fn test_video_parameters_pixel_count() {
        let vp = VideoParameters::hd_1080p_23_976();
        assert_eq!(vp.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_video_parameters_display_aspect_ratio() {
        let vp = VideoParameters::hd_1080p_23_976();
        assert!((vp.display_aspect_ratio() - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_parameters_stereo() {
        let ap = AudioParameters::stereo_48k_24bit();
        assert_eq!(ap.sample_rate, 48000);
        assert_eq!(ap.channels, 2);
        assert_eq!(ap.bytes_per_sample(), 3);
    }

    #[test]
    fn test_audio_parameters_bytes_per_second() {
        let ap = AudioParameters::stereo_48k_24bit();
        // 48000 samples * 3 bytes * 2 channels = 288000
        assert_eq!(ap.bytes_per_second(), 288_000);
    }

    #[test]
    fn test_media_data_descriptor_video() {
        let desc = MediaDataDescriptor::new(
            "desc-001",
            CodecFamily::ProRes,
            ContainerFormat::MxfOpAtom,
            EssenceLocator::external("/media/v1.mxf"),
        )
        .with_video(VideoParameters::hd_1080p_23_976())
        .with_data_size(1_000_000);

        assert!(desc.is_video());
        assert!(!desc.is_audio());
        assert_eq!(desc.data_size_bytes, 1_000_000);
    }

    #[test]
    fn test_media_data_descriptor_audio() {
        let desc = MediaDataDescriptor::new(
            "desc-002",
            CodecFamily::Pcm,
            ContainerFormat::Wave,
            EssenceLocator::external("/media/a1.wav"),
        )
        .with_audio(AudioParameters::stereo_48k_24bit());

        assert!(desc.is_audio());
        assert!(!desc.is_video());
    }

    #[test]
    fn test_media_data_descriptor_properties() {
        let mut desc = MediaDataDescriptor::new(
            "desc-003",
            CodecFamily::DnxHd,
            ContainerFormat::MxfOp1a,
            EssenceLocator::internal(0),
        );
        desc.set_property("profile", "DNxHR HQ");
        assert_eq!(desc.get_property("profile"), Some("DNxHR HQ"));
        assert!(desc.get_property("missing").is_none());
    }

    #[test]
    fn test_media_data_registry() {
        let mut reg = MediaDataRegistry::new();
        assert!(reg.is_empty());

        reg.add(
            MediaDataDescriptor::new(
                "v1",
                CodecFamily::ProRes,
                ContainerFormat::MxfOpAtom,
                EssenceLocator::internal(0),
            )
            .with_video(VideoParameters::hd_1080p_23_976())
            .with_data_size(500),
        );
        reg.add(
            MediaDataDescriptor::new(
                "a1",
                CodecFamily::Pcm,
                ContainerFormat::Wave,
                EssenceLocator::external("/a.wav"),
            )
            .with_audio(AudioParameters::stereo_48k_24bit())
            .with_data_size(300),
        );

        assert_eq!(reg.len(), 2);
        assert_eq!(reg.video_descriptors().len(), 1);
        assert_eq!(reg.audio_descriptors().len(), 1);
        assert_eq!(reg.total_data_size(), 800);
    }

    #[test]
    fn test_media_data_registry_find_by_id() {
        let mut reg = MediaDataRegistry::new();
        reg.add(MediaDataDescriptor::new(
            "x",
            CodecFamily::Avc,
            ContainerFormat::AafInternal,
            EssenceLocator::internal(1),
        ));
        assert!(reg.find_by_id("x").is_some());
        assert!(reg.find_by_id("y").is_none());
    }

    #[test]
    fn test_estimated_bitrate_audio() {
        let desc = MediaDataDescriptor::new(
            "a2",
            CodecFamily::Pcm,
            ContainerFormat::Wave,
            EssenceLocator::internal(0),
        )
        .with_audio(AudioParameters::stereo_48k_24bit());
        // 288000 bytes/s * 8 = 2304000 bps
        assert_eq!(desc.estimated_bitrate_bps(), 2_304_000);
    }
}
