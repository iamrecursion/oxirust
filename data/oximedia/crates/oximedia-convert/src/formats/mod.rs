// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Format support for media conversion.
//!
//! This module provides comprehensive support for various media formats,
//! including video, audio, and image formats with patent-free codec support.

pub mod audio;
pub mod container;
pub mod image;
pub mod video;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported video codecs (patent-free only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoCodec {
    /// AV1 video codec
    Av1,
    /// VP9 video codec
    Vp9,
    /// VP8 video codec
    Vp8,
    /// Theora video codec
    Theora,
}

impl VideoCodec {
    /// Get the codec name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Av1 => "av1",
            Self::Vp9 => "vp9",
            Self::Vp8 => "vp8",
            Self::Theora => "theora",
        }
    }

    /// Check if codec supports alpha channel.
    #[must_use]
    pub const fn supports_alpha(self) -> bool {
        matches!(self, Self::Vp8 | Self::Vp9)
    }

    /// Get recommended quality range (0-63 for VP8/VP9, 0-255 for AV1).
    #[must_use]
    pub const fn quality_range(self) -> (u32, u32) {
        match self {
            Self::Av1 => (0, 255),
            Self::Vp9 | Self::Vp8 => (0, 63),
            Self::Theora => (0, 10),
        }
    }

    /// Get default quality value.
    #[must_use]
    pub const fn default_quality(self) -> u32 {
        match self {
            Self::Av1 => 128,
            Self::Vp9 => 31,
            Self::Vp8 => 10,
            Self::Theora => 7,
        }
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Supported audio codecs (patent-free only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodec {
    /// Opus audio codec
    Opus,
    /// Vorbis audio codec
    Vorbis,
    /// FLAC lossless audio codec
    Flac,
    /// PCM (uncompressed audio)
    Pcm,
}

impl AudioCodec {
    /// Get the codec name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Vorbis => "vorbis",
            Self::Flac => "flac",
            Self::Pcm => "pcm",
        }
    }

    /// Check if codec is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Flac | Self::Pcm)
    }

    /// Get supported sample rates.
    #[must_use]
    pub fn supported_sample_rates(self) -> &'static [u32] {
        match self {
            Self::Opus => &[8000, 12000, 16000, 24000, 48000],
            Self::Vorbis | Self::Flac | Self::Pcm => &[
                8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000,
            ],
        }
    }

    /// Get default bitrate in kbps (for lossy codecs).
    #[must_use]
    pub const fn default_bitrate(self) -> Option<u32> {
        match self {
            Self::Opus => Some(128),
            Self::Vorbis => Some(192),
            Self::Flac | Self::Pcm => None,
        }
    }
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Supported container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerFormat {
    /// MP4 container (ISO Base Media)
    Mp4,
    /// Matroska container (.mkv)
    Matroska,
    /// `WebM` container (subset of Matroska)
    Webm,
    /// Ogg container
    Ogg,
    /// MPEG-TS container
    MpegTs,
    /// WAV audio container
    Wav,
    /// FLAC audio container
    Flac,
}

impl ContainerFormat {
    /// Get the container format name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Matroska => "matroska",
            Self::Webm => "webm",
            Self::Ogg => "ogg",
            Self::MpegTs => "mpegts",
            Self::Wav => "wav",
            Self::Flac => "flac",
        }
    }

    /// Get typical file extension.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Matroska => "mkv",
            Self::Webm => "webm",
            Self::Ogg => "ogg",
            Self::MpegTs => "ts",
            Self::Wav => "wav",
            Self::Flac => "flac",
        }
    }

    /// Check if format supports video.
    #[must_use]
    pub const fn supports_video(self) -> bool {
        matches!(
            self,
            Self::Mp4 | Self::Matroska | Self::Webm | Self::Ogg | Self::MpegTs
        )
    }

    /// Check if format supports audio.
    #[must_use]
    pub const fn supports_audio(self) -> bool {
        true
    }

    /// Check if format supports subtitles.
    #[must_use]
    pub const fn supports_subtitles(self) -> bool {
        matches!(self, Self::Mp4 | Self::Matroska | Self::Webm | Self::MpegTs)
    }

    /// Get compatible video codecs.
    #[must_use]
    pub fn compatible_video_codecs(self) -> &'static [VideoCodec] {
        match self {
            Self::Mp4 => &[VideoCodec::Av1, VideoCodec::Vp9],
            Self::Matroska => &[
                VideoCodec::Av1,
                VideoCodec::Vp9,
                VideoCodec::Vp8,
                VideoCodec::Theora,
            ],
            Self::Webm => &[VideoCodec::Vp9, VideoCodec::Vp8],
            Self::Ogg => &[VideoCodec::Theora],
            Self::MpegTs => &[VideoCodec::Av1, VideoCodec::Vp9],
            Self::Wav | Self::Flac => &[],
        }
    }

    /// Get compatible audio codecs.
    #[must_use]
    pub fn compatible_audio_codecs(self) -> &'static [AudioCodec] {
        match self {
            Self::Mp4 => &[AudioCodec::Opus, AudioCodec::Flac],
            Self::Matroska => &[
                AudioCodec::Opus,
                AudioCodec::Vorbis,
                AudioCodec::Flac,
                AudioCodec::Pcm,
            ],
            Self::Webm => &[AudioCodec::Opus, AudioCodec::Vorbis],
            Self::Ogg => &[AudioCodec::Opus, AudioCodec::Vorbis, AudioCodec::Flac],
            Self::MpegTs => &[AudioCodec::Opus],
            Self::Wav => &[AudioCodec::Pcm],
            Self::Flac => &[AudioCodec::Flac],
        }
    }
}

impl fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Supported image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    /// PNG image format
    Png,
    /// WebP image format
    Webp,
    /// TIFF image format
    Tiff,
    /// DPX image format (film/broadcast)
    Dpx,
    /// `OpenEXR` image format (HDR)
    Exr,
}

impl ImageFormat {
    /// Get the format name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Webp => "webp",
            Self::Tiff => "tiff",
            Self::Dpx => "dpx",
            Self::Exr => "exr",
        }
    }

    /// Get file extension.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Webp => "webp",
            Self::Tiff => "tif",
            Self::Dpx => "dpx",
            Self::Exr => "exr",
        }
    }

    /// Check if format supports alpha channel.
    #[must_use]
    pub const fn supports_alpha(self) -> bool {
        matches!(self, Self::Png | Self::Webp | Self::Tiff | Self::Exr)
    }

    /// Check if format supports HDR.
    #[must_use]
    pub const fn supports_hdr(self) -> bool {
        matches!(self, Self::Exr | Self::Tiff)
    }

    /// Check if format is lossless.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Png | Self::Tiff | Self::Dpx | Self::Exr)
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Audio channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelLayout {
    /// Mono audio (1 channel)
    Mono,
    /// Stereo audio (2 channels)
    Stereo,
    /// 5.1 surround sound (6 channels)
    Surround5_1,
    /// 7.1 surround sound (8 channels)
    Surround7_1,
}

impl ChannelLayout {
    /// Get the number of channels.
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround5_1 => 6,
            Self::Surround7_1 => 8,
        }
    }

    /// Get the layout name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::Surround5_1 => "5.1",
            Self::Surround7_1 => "7.1",
        }
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_properties() {
        assert_eq!(VideoCodec::Av1.name(), "av1");
        assert!(VideoCodec::Vp9.supports_alpha());
        assert!(!VideoCodec::Av1.supports_alpha());
        assert_eq!(VideoCodec::Av1.quality_range(), (0, 255));
        assert_eq!(VideoCodec::Vp9.default_quality(), 31);
    }

    #[test]
    fn test_audio_codec_properties() {
        assert_eq!(AudioCodec::Opus.name(), "opus");
        assert!(AudioCodec::Flac.is_lossless());
        assert!(!AudioCodec::Opus.is_lossless());
        assert_eq!(AudioCodec::Opus.default_bitrate(), Some(128));
        assert_eq!(AudioCodec::Flac.default_bitrate(), None);
    }

    #[test]
    fn test_container_format_compatibility() {
        assert!(ContainerFormat::Webm.supports_video());
        assert!(ContainerFormat::Webm.supports_audio());
        assert!(ContainerFormat::Webm.supports_subtitles());
        assert!(ContainerFormat::Webm
            .compatible_video_codecs()
            .contains(&VideoCodec::Vp9));
        assert!(ContainerFormat::Webm
            .compatible_audio_codecs()
            .contains(&AudioCodec::Opus));
    }

    #[test]
    fn test_image_format_properties() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert!(ImageFormat::Png.supports_alpha());
        assert!(ImageFormat::Exr.supports_hdr());
        assert!(ImageFormat::Png.is_lossless());
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround5_1.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround7_1.name(), "7.1");
    }
}
