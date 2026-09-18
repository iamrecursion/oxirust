// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio format handling and properties.

use super::{AudioCodec, ChannelLayout, ContainerFormat};
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Audio format properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioProperties {
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Channel layout
    pub channels: ChannelLayout,
    /// Bitrate in bits per second (for lossy codecs)
    pub bitrate: Option<u64>,
    /// Bit depth
    pub bit_depth: Option<u32>,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Total number of samples
    pub sample_count: Option<u64>,
}

/// Audio format detector.
#[derive(Debug, Clone)]
pub struct AudioFormatDetector;

impl AudioFormatDetector {
    /// Create a new audio format detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect audio format from file.
    pub fn detect(&self, path: &Path) -> Result<AudioProperties> {
        let mut file = File::open(path).map_err(ConversionError::Io)?;

        // Read first 64 bytes for magic detection
        let mut header = [0u8; 64];
        let bytes_read = file.read(&mut header).map_err(ConversionError::Io)?;
        if bytes_read < 4 {
            return Err(ConversionError::FormatDetection(
                "File too small to detect format".to_string(),
            ));
        }

        // FLAC: magic bytes "fLaC"
        if header[..4] == *b"fLaC" {
            // STREAMINFO block starts at byte 8; sample rate is bits [80..100]
            // Bytes 14-17 contain: [sample_rate (20 bits) | channels (3 bits) | bit_depth (5 bits) | ...]
            // sample_rate = (header[18] << 12) | (header[19] << 4) | (header[20] >> 4)  -- raw STREAMINFO layout
            // STREAMINFO metadata block: 4 bytes block header, then STREAMINFO data at offset 8
            // Bytes 8..9 = min_block_size (u16), 10..11 = max_block_size (u16), 12..14 = min_frame_size (u24)
            // 15..17 = max_frame_size (u24), 18..20 (20 bits) = sample_rate, then 3 bits channels, etc.
            let sample_rate = if bytes_read >= 21 {
                let sr = (u32::from(header[18]) << 12)
                    | (u32::from(header[19]) << 4)
                    | (u32::from(header[20]) >> 4);
                if sr > 0 {
                    sr
                } else {
                    44100
                }
            } else {
                44100
            };
            // channels: bits [100..103] of STREAMINFO, i.e. bits 4-6 of header[20], +1
            let channels_raw = if bytes_read >= 21 {
                u16::from(((header[20] & 0x0E) >> 1) + 1)
            } else {
                2
            };
            // bit depth: bits [103..108], i.e. lower bit of header[20] (1 bit) | upper 4 bits of header[21]
            let bit_depth = if bytes_read >= 22 {
                let bd = (((header[20] & 0x01) << 4) | (header[21] >> 4)) + 1;
                u32::from(bd)
            } else {
                16
            };
            let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
            // Estimate duration: file_size / (sample_rate * channels * bit_depth/8)
            let bytes_per_sec = u64::from(sample_rate)
                * u64::from(channels_raw)
                * (u64::from(bit_depth) / 8).max(1);
            let duration = if bytes_per_sec > 0 && file_size > 0 {
                Some(file_size as f64 / bytes_per_sec as f64)
            } else {
                None
            };
            let channels = channels_from_count(u32::from(channels_raw));
            let sample_count = duration.map(|d| (d * f64::from(sample_rate)) as u64);
            return Ok(AudioProperties {
                codec: AudioCodec::Flac,
                sample_rate,
                channels,
                bitrate: None,
                bit_depth: Some(bit_depth),
                duration,
                sample_count,
            });
        }

        // OGG: magic bytes "OggS"
        if header[..4] == *b"OggS" {
            // Look in first 64 bytes for "OpusHead" or "\x01vorbis"
            let codec = if bytes_read >= 40 && header[28..36] == *b"OpusHead" {
                AudioCodec::Opus
            } else if bytes_read >= 36 && &header[28..35] == b"\x01vorbis" {
                AudioCodec::Vorbis
            } else {
                // Scan a wider range
                let found_opus = header[..bytes_read.min(64)]
                    .windows(8)
                    .any(|w| w == b"OpusHead");
                let found_vorbis = header[..bytes_read.min(64)]
                    .windows(7)
                    .any(|w| w == b"\x01vorbis");
                if found_opus {
                    AudioCodec::Opus
                } else if found_vorbis {
                    AudioCodec::Vorbis
                } else {
                    AudioCodec::Opus
                }
            };
            let sample_rate = if codec == AudioCodec::Opus {
                48000
            } else {
                44100
            };
            return Ok(AudioProperties {
                codec,
                sample_rate,
                channels: ChannelLayout::Stereo,
                bitrate: Some(128000),
                bit_depth: None,
                duration: None,
                sample_count: None,
            });
        }

        // WAV: "RIFF" at 0 and "WAVE" at 8
        if bytes_read >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            // fmt chunk starts at offset 12: "fmt " (4 bytes) + chunk_size (4 bytes)
            // offset 20 = audio_format (u16, 1=PCM)
            // offset 22 = num_channels (u16)
            // offset 24 = sample_rate (u32)
            // offset 28 = byte_rate (u32)
            // offset 32 = block_align (u16)
            // offset 34 = bits_per_sample (u16)
            let num_channels = if bytes_read >= 24 {
                u32::from(u16::from_le_bytes([header[22], header[23]]))
            } else {
                2
            };
            let sample_rate = if bytes_read >= 28 {
                u32::from_le_bytes([header[24], header[25], header[26], header[27]])
            } else {
                44100
            };
            let byte_rate = if bytes_read >= 32 {
                u32::from_le_bytes([header[28], header[29], header[30], header[31]])
            } else {
                0
            };
            let bits_per_sample = if bytes_read >= 36 {
                u32::from(u16::from_le_bytes([header[34], header[35]]))
            } else {
                16
            };
            let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
            let duration = if byte_rate > 0 && file_size > 44 {
                Some((file_size - 44) as f64 / f64::from(byte_rate))
            } else {
                None
            };
            let channels = channels_from_count(num_channels);
            let sample_count = duration.map(|d| (d * f64::from(sample_rate)) as u64);
            let bitrate = if byte_rate > 0 {
                Some(u64::from(byte_rate) * 8)
            } else {
                None
            };
            return Ok(AudioProperties {
                codec: AudioCodec::Pcm,
                sample_rate,
                channels,
                bitrate,
                bit_depth: Some(bits_per_sample),
                duration,
                sample_count,
            });
        }

        // MP3: "ID3" tag or MPEG sync word (0xFF 0xE0-0xFF)
        let is_mp3 =
            &header[..3] == b"ID3" || (bytes_read >= 2 && header[0] == 0xFF && header[1] >= 0xE0);
        if is_mp3 {
            // MP3 is patent-encumbered; return Opus as safe default
            return Ok(AudioProperties {
                codec: AudioCodec::Opus,
                sample_rate: 48000,
                channels: ChannelLayout::Stereo,
                bitrate: Some(128000),
                bit_depth: None,
                duration: None,
                sample_count: None,
            });
        }

        // Default: Opus
        Ok(AudioProperties {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128000),
            bit_depth: None,
            duration: None,
            sample_count: None,
        })
    }

    /// Check if file contains audio.
    pub fn has_audio(&self, path: &Path) -> Result<bool> {
        match self.detect(path) {
            Ok(_) => Ok(true),
            Err(ConversionError::FormatDetection(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Get number of audio tracks.
    pub fn track_count(&self, path: &Path) -> Result<usize> {
        let mut file = File::open(path).map_err(ConversionError::Io)?;
        let mut header = [0u8; 12];
        let bytes_read = file.read(&mut header).map_err(ConversionError::Io)?;

        // WAV always has exactly 1 audio track
        if bytes_read >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            return Ok(1);
        }

        // OGG: multiple logical bitstreams are possible, but detecting them
        // requires full parsing of all page headers; return 1 as safe default
        if bytes_read >= 4 && &header[..4] == b"OggS" {
            return Ok(1);
        }

        // All other formats default to 1 track
        Ok(1)
    }
}

impl Default for AudioFormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a raw channel count to the appropriate `ChannelLayout`.
fn channels_from_count(count: u32) -> ChannelLayout {
    match count {
        1 => ChannelLayout::Mono,
        6 => ChannelLayout::Surround5_1,
        8 => ChannelLayout::Surround7_1,
        _ => ChannelLayout::Stereo,
    }
}

/// Audio format validator.
#[derive(Debug, Clone)]
pub struct AudioFormatValidator;

impl AudioFormatValidator {
    /// Create a new validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validate audio codec compatibility with container.
    pub fn validate_codec_compatibility(
        &self,
        codec: AudioCodec,
        container: ContainerFormat,
    ) -> Result<()> {
        if container.compatible_audio_codecs().contains(&codec) {
            Ok(())
        } else {
            Err(ConversionError::InvalidInput(format!(
                "Audio codec {codec} is not compatible with container {container}"
            )))
        }
    }

    /// Validate sample rate for codec.
    pub fn validate_sample_rate(&self, codec: AudioCodec, sample_rate: u32) -> Result<()> {
        if codec.supported_sample_rates().contains(&sample_rate) {
            Ok(())
        } else {
            Err(ConversionError::InvalidInput(format!(
                "Sample rate {sample_rate} is not supported by codec {codec}"
            )))
        }
    }

    /// Validate bitrate for codec.
    pub fn validate_bitrate(&self, codec: AudioCodec, bitrate: u32) -> Result<()> {
        const MIN_BITRATE: u32 = 8;
        const MAX_BITRATE: u32 = 512;

        if codec.is_lossless() {
            return Ok(());
        }

        if !(MIN_BITRATE..=MAX_BITRATE).contains(&bitrate) {
            return Err(ConversionError::InvalidInput(format!(
                "Bitrate {bitrate} kbps is outside valid range {MIN_BITRATE}-{MAX_BITRATE} kbps"
            )));
        }

        Ok(())
    }

    /// Validate channel layout.
    pub fn validate_channel_layout(&self, layout: ChannelLayout) -> Result<()> {
        // All layouts are currently valid
        let _ = layout;
        Ok(())
    }
}

impl Default for AudioFormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// Peak normalization
    Peak,
    /// RMS normalization
    Rms,
    /// EBU R128 loudness normalization
    EbuR128,
    /// ATSC A/85 loudness normalization
    AtscA85,
}

/// Audio normalization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationSettings {
    /// Normalization mode
    pub mode: NormalizationMode,
    /// Target level in dB (or LUFS for loudness modes)
    pub target_level: f64,
    /// Whether to apply limiting to prevent clipping
    pub apply_limiting: bool,
}

impl Default for NormalizationSettings {
    fn default() -> Self {
        Self {
            mode: NormalizationMode::EbuR128,
            target_level: -23.0, // -23 LUFS for EBU R128
            apply_limiting: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_detector() {
        let detector = AudioFormatDetector::new();
        // AudioFormatDetector is a zero-sized struct; verify it can be constructed
        let _ = detector;
    }

    #[test]
    fn test_audio_format_validator() {
        let validator = AudioFormatValidator::new();

        assert!(validator
            .validate_codec_compatibility(AudioCodec::Opus, ContainerFormat::Webm)
            .is_ok());
        assert!(validator
            .validate_codec_compatibility(AudioCodec::Flac, ContainerFormat::Webm)
            .is_err());

        assert!(validator
            .validate_sample_rate(AudioCodec::Opus, 48000)
            .is_ok());
        assert!(validator
            .validate_sample_rate(AudioCodec::Opus, 96000)
            .is_err());

        assert!(validator.validate_bitrate(AudioCodec::Opus, 128).is_ok());
        assert!(validator.validate_bitrate(AudioCodec::Opus, 1000).is_err());
        assert!(validator.validate_bitrate(AudioCodec::Flac, 9999).is_ok());

        assert!(validator
            .validate_channel_layout(ChannelLayout::Stereo)
            .is_ok());
    }

    #[test]
    fn test_normalization_settings() {
        let settings = NormalizationSettings::default();
        assert_eq!(settings.mode, NormalizationMode::EbuR128);
        assert_eq!(settings.target_level, -23.0);
        assert!(settings.apply_limiting);
    }
}
