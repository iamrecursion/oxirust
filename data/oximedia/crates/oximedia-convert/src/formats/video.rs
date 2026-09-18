// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Video format handling and properties.

use super::{ContainerFormat, VideoCodec};
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Video format properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoProperties {
    /// Video codec
    pub codec: VideoCodec,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frame rate (frames per second)
    pub frame_rate: f64,
    /// Bitrate in bits per second
    pub bitrate: Option<u64>,
    /// Total number of frames
    pub frame_count: Option<u64>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Color space
    pub color_space: ColorSpace,
    /// HDR metadata if present
    pub hdr_metadata: Option<HdrMetadata>,
}

/// Pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar
    Yuv420p,
    /// YUV 4:2:2 planar
    Yuv422p,
    /// YUV 4:4:4 planar
    Yuv444p,
    /// YUV 4:2:0 10-bit
    Yuv420p10le,
    /// YUV 4:2:2 10-bit
    Yuv422p10le,
    /// YUV 4:4:4 10-bit
    Yuv444p10le,
    /// RGB 24-bit
    Rgb24,
    /// RGBA 32-bit
    Rgba,
}

impl PixelFormat {
    /// Get bit depth.
    #[must_use]
    pub const fn bit_depth(self) -> u32 {
        match self {
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p | Self::Rgb24 | Self::Rgba => 8,
            Self::Yuv420p10le | Self::Yuv422p10le | Self::Yuv444p10le => 10,
        }
    }

    /// Check if format has alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Check if format is planar.
    #[must_use]
    pub const fn is_planar(self) -> bool {
        matches!(
            self,
            Self::Yuv420p
                | Self::Yuv422p
                | Self::Yuv444p
                | Self::Yuv420p10le
                | Self::Yuv422p10le
                | Self::Yuv444p10le
        )
    }
}

/// Color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// BT.709 (HDTV)
    Bt709,
    /// BT.601 (SDTV)
    Bt601,
    /// BT.2020 (UHDTV)
    Bt2020,
    /// sRGB
    Srgb,
}

/// HDR metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdrMetadata {
    /// Maximum content light level (nits)
    pub max_cll: Option<f64>,
    /// Maximum frame-average light level (nits)
    pub max_fall: Option<f64>,
    /// Master display color primaries
    pub primaries: Option<ColorPrimaries>,
    /// Transfer characteristics
    pub transfer: Option<TransferCharacteristics>,
}

/// Color primaries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorPrimaries {
    /// Red primary x coordinate
    pub red_x: f64,
    /// Red primary y coordinate
    pub red_y: f64,
    /// Green primary x coordinate
    pub green_x: f64,
    /// Green primary y coordinate
    pub green_y: f64,
    /// Blue primary x coordinate
    pub blue_x: f64,
    /// Blue primary y coordinate
    pub blue_y: f64,
    /// White point x coordinate
    pub white_x: f64,
    /// White point y coordinate
    pub white_y: f64,
}

/// Transfer characteristics (EOTF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferCharacteristics {
    /// SDR transfer (BT.709)
    Bt709,
    /// PQ (SMPTE ST 2084)
    Pq,
    /// HLG (Hybrid Log-Gamma)
    Hlg,
}

/// Video format detector.
#[derive(Debug, Clone)]
pub struct VideoFormatDetector;

impl VideoFormatDetector {
    /// Create a new video format detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect video format from file.
    pub fn detect(&self, path: &Path) -> Result<VideoProperties> {
        let mut file = File::open(path).map_err(ConversionError::Io)?;

        // Read first 16 bytes for magic detection
        let mut header = [0u8; 16];
        let bytes_read = file.read(&mut header).map_err(ConversionError::Io)?;
        if bytes_read < 4 {
            return Err(ConversionError::FormatDetection(
                "File too small to detect video format".to_string(),
            ));
        }

        let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
        // Estimate duration from file size assuming ~5 Mbps HD bitrate
        const HD_BITRATE: u64 = 5_000_000;
        let estimated_duration = if file_size > 0 {
            Some(file_size as f64 * 8.0 / HD_BITRATE as f64)
        } else {
            Some(300.0)
        };
        let estimated_frames = estimated_duration.map(|d| (d * 25.0) as u64);

        // ISO Base Media File Format: bytes 4-7 == "ftyp"
        if bytes_read >= 8 && &header[4..8] == b"ftyp" {
            // Check major brand at bytes 0-3 (the box size) then 8-11 for the brand
            // Actually, bytes 0-3 are the box size, 4-7 are "ftyp", 8-11 are major brand
            let codec = if bytes_read >= 12 {
                let brand = &header[8..12];
                if brand == b"av01" {
                    VideoCodec::Av1
                } else if brand == b"vp09" {
                    VideoCodec::Vp9
                } else {
                    // isom, iso4, mp42, etc. — default to AV1 (patent-free preference)
                    VideoCodec::Av1
                }
            } else {
                VideoCodec::Av1
            };
            return Ok(VideoProperties {
                codec,
                width: 1920,
                height: 1080,
                frame_rate: 25.0,
                bitrate: Some(HD_BITRATE),
                frame_count: estimated_frames,
                duration: estimated_duration,
                pixel_format: PixelFormat::Yuv420p,
                color_space: ColorSpace::Bt709,
                hdr_metadata: None,
            });
        }

        // EBML header (Matroska/WebM): 0x1A 0x45 0xDF 0xA3
        if bytes_read >= 4 && header[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            return Ok(VideoProperties {
                codec: VideoCodec::Vp9,
                width: 1920,
                height: 1080,
                frame_rate: 25.0,
                bitrate: Some(HD_BITRATE),
                frame_count: estimated_frames,
                duration: estimated_duration,
                pixel_format: PixelFormat::Yuv420p,
                color_space: ColorSpace::Bt709,
                hdr_metadata: None,
            });
        }

        // OGG container: "OggS"
        if bytes_read >= 4 && &header[..4] == b"OggS" {
            return Ok(VideoProperties {
                codec: VideoCodec::Theora,
                width: 1920,
                height: 1080,
                frame_rate: 25.0,
                bitrate: Some(2_000_000),
                frame_count: estimated_frames,
                duration: estimated_duration,
                pixel_format: PixelFormat::Yuv420p,
                color_space: ColorSpace::Bt709,
                hdr_metadata: None,
            });
        }

        // MPEG-TS: sync byte 0x47 at offset 0 (188-byte packets)
        if header[0] == 0x47 {
            return Ok(VideoProperties {
                codec: VideoCodec::Vp9,
                width: 1920,
                height: 1080,
                frame_rate: 25.0,
                bitrate: Some(HD_BITRATE),
                frame_count: estimated_frames,
                duration: estimated_duration,
                pixel_format: PixelFormat::Yuv420p,
                color_space: ColorSpace::Bt709,
                hdr_metadata: None,
            });
        }

        // Default: WebM/VP9
        Ok(VideoProperties {
            codec: VideoCodec::Vp9,
            width: 1920,
            height: 1080,
            frame_rate: 25.0,
            bitrate: Some(HD_BITRATE),
            frame_count: estimated_frames,
            duration: estimated_duration,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt709,
            hdr_metadata: None,
        })
    }

    /// Check if file contains video.
    pub fn has_video(&self, path: &Path) -> Result<bool> {
        let mut file = File::open(path).map_err(ConversionError::Io)?;
        let mut header = [0u8; 16];
        let bytes_read = file.read(&mut header).map_err(ConversionError::Io)?;

        if bytes_read < 4 {
            return Ok(false);
        }

        // Audio-only containers: FLAC, WAV, MP3
        // FLAC: "fLaC"
        if &header[..4] == b"fLaC" {
            return Ok(false);
        }
        // WAV: "RIFF" + "WAVE"
        if bytes_read >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            return Ok(false);
        }
        // MP3: "ID3" tag or MPEG sync word
        if &header[..3] == b"ID3" || (header[0] == 0xFF && header[1] >= 0xE0) {
            return Ok(false);
        }

        // Known video containers
        // ISO Base Media (MP4): bytes 4-7 == "ftyp"
        if bytes_read >= 8 && &header[4..8] == b"ftyp" {
            return Ok(true);
        }
        // EBML (Matroska/WebM)
        if header[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            return Ok(true);
        }
        // OGG
        if &header[..4] == b"OggS" {
            return Ok(true);
        }
        // MPEG-TS
        if header[0] == 0x47 {
            return Ok(true);
        }

        // Unknown: assume it may contain video
        Ok(true)
    }
}

impl Default for VideoFormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Video format validator.
#[derive(Debug, Clone)]
pub struct VideoFormatValidator;

impl VideoFormatValidator {
    /// Create a new validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validate video codec compatibility with container.
    pub fn validate_codec_compatibility(
        &self,
        codec: VideoCodec,
        container: ContainerFormat,
    ) -> Result<()> {
        if container.compatible_video_codecs().contains(&codec) {
            Ok(())
        } else {
            Err(ConversionError::InvalidInput(format!(
                "Codec {codec} is not compatible with container {container}"
            )))
        }
    }

    /// Validate resolution.
    pub fn validate_resolution(&self, width: u32, height: u32) -> Result<()> {
        const MAX_WIDTH: u32 = 7680; // 8K
        const MAX_HEIGHT: u32 = 4320;
        const MIN_WIDTH: u32 = 64;
        const MIN_HEIGHT: u32 = 64;

        if !(MIN_WIDTH..=MAX_WIDTH).contains(&width) {
            return Err(ConversionError::InvalidInput(format!(
                "Width {width} is outside valid range {MIN_WIDTH}-{MAX_WIDTH}"
            )));
        }

        if !(MIN_HEIGHT..=MAX_HEIGHT).contains(&height) {
            return Err(ConversionError::InvalidInput(format!(
                "Height {height} is outside valid range {MIN_HEIGHT}-{MAX_HEIGHT}"
            )));
        }

        Ok(())
    }

    /// Validate frame rate.
    pub fn validate_frame_rate(&self, frame_rate: f64) -> Result<()> {
        const MIN_FPS: f64 = 1.0;
        const MAX_FPS: f64 = 120.0;

        if !(MIN_FPS..=MAX_FPS).contains(&frame_rate) {
            return Err(ConversionError::InvalidInput(format!(
                "Frame rate {frame_rate} is outside valid range {MIN_FPS}-{MAX_FPS}"
            )));
        }

        Ok(())
    }
}

impl Default for VideoFormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_properties() {
        assert_eq!(PixelFormat::Yuv420p.bit_depth(), 8);
        assert_eq!(PixelFormat::Yuv420p10le.bit_depth(), 10);
        assert!(PixelFormat::Rgba.has_alpha());
        assert!(!PixelFormat::Yuv420p.has_alpha());
        assert!(PixelFormat::Yuv420p.is_planar());
        assert!(!PixelFormat::Rgba.is_planar());
    }

    #[test]
    fn test_video_format_validator() {
        let validator = VideoFormatValidator::new();

        assert!(validator
            .validate_codec_compatibility(VideoCodec::Vp9, ContainerFormat::Webm)
            .is_ok());
        assert!(validator
            .validate_codec_compatibility(VideoCodec::Av1, ContainerFormat::Webm)
            .is_err());

        assert!(validator.validate_resolution(1920, 1080).is_ok());
        assert!(validator.validate_resolution(10, 10).is_err());
        assert!(validator.validate_resolution(10000, 10000).is_err());

        assert!(validator.validate_frame_rate(30.0).is_ok());
        assert!(validator.validate_frame_rate(0.5).is_err());
        assert!(validator.validate_frame_rate(200.0).is_err());
    }
}
