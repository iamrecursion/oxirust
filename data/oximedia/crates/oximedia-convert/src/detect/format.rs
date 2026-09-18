// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Format detection for media files.

use crate::{ConversionError, MediaProperties, Result};
use std::path::Path;

/// Detector for media file formats.
#[derive(Debug, Clone)]
pub struct FormatDetector {
    magic_bytes: Vec<FormatSignature>,
}

impl FormatDetector {
    /// Create a new format detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            magic_bytes: Self::init_signatures(),
        }
    }

    /// Detect the format of a media file.
    pub fn detect<P: AsRef<Path>>(&self, path: P) -> Result<MediaProperties> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "File not found: {}",
                path.display()
            )));
        }

        let metadata = std::fs::metadata(path).map_err(ConversionError::Io)?;

        if !metadata.is_file() {
            return Err(ConversionError::InvalidInput(
                "Path is not a file".to_string(),
            ));
        }

        // Read magic bytes
        let magic = self.read_magic_bytes(path)?;
        let format = self
            .detect_format_from_magic(&magic)
            .or_else(|| self.detect_format_from_extension(path))
            .ok_or_else(|| ConversionError::FormatDetection("Unknown format".to_string()))?;

        Ok(MediaProperties {
            format,
            file_size: metadata.len(),
            duration: None,
            width: None,
            height: None,
            video_codec: None,
            audio_codec: None,
            video_bitrate: None,
            audio_bitrate: None,
            frame_rate: None,
            audio_sample_rate: None,
            audio_channels: None,
        })
    }

    /// Detect format from magic bytes.
    #[must_use]
    pub fn detect_format_from_magic(&self, magic: &[u8]) -> Option<String> {
        for sig in &self.magic_bytes {
            if magic.len() >= sig.bytes.len() {
                let matches = if let Some(offset) = sig.offset {
                    if magic.len() >= offset + sig.bytes.len() {
                        &magic[offset..offset + sig.bytes.len()] == sig.bytes.as_slice()
                    } else {
                        false
                    }
                } else {
                    magic.starts_with(&sig.bytes)
                };

                if matches {
                    return Some(sig.format.to_string());
                }
            }
        }
        None
    }

    /// Detect format from file extension.
    pub fn detect_format_from_extension<P: AsRef<Path>>(&self, path: P) -> Option<String> {
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "mp4" | "m4v" | "m4a" => Some("mp4"),
                "mkv" => Some("matroska"),
                "webm" => Some("webm"),
                "avi" => Some("avi"),
                "mov" => Some("mov"),
                "flv" => Some("flv"),
                "wmv" => Some("wmv"),
                "mpg" | "mpeg" => Some("mpeg"),
                "ts" => Some("mpegts"),
                "m3u8" => Some("hls"),
                "mp3" => Some("mp3"),
                "aac" => Some("aac"),
                "flac" => Some("flac"),
                "wav" => Some("wav"),
                "ogg" => Some("ogg"),
                "opus" => Some("opus"),
                _ => None,
            })
            .map(String::from)
    }

    fn read_magic_bytes<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(ConversionError::Io)?;

        let mut buffer = vec![0u8; 64];
        let n = file.read(&mut buffer).map_err(ConversionError::Io)?;

        buffer.truncate(n);
        Ok(buffer)
    }

    fn init_signatures() -> Vec<FormatSignature> {
        vec![
            // MP4/M4V/M4A (ISO Base Media)
            FormatSignature {
                format: "mp4",
                bytes: vec![0x66, 0x74, 0x79, 0x70], // "ftyp"
                offset: Some(4),
            },
            // Matroska/WebM
            FormatSignature {
                format: "matroska",
                bytes: vec![0x1A, 0x45, 0xDF, 0xA3],
                offset: None,
            },
            // AVI
            FormatSignature {
                format: "avi",
                bytes: vec![0x52, 0x49, 0x46, 0x46], // "RIFF"
                offset: None,
            },
            // FLV
            FormatSignature {
                format: "flv",
                bytes: vec![0x46, 0x4C, 0x56], // "FLV"
                offset: None,
            },
            // MP3
            FormatSignature {
                format: "mp3",
                bytes: vec![0xFF, 0xFB],
                offset: None,
            },
            FormatSignature {
                format: "mp3",
                bytes: vec![0x49, 0x44, 0x33], // "ID3"
                offset: None,
            },
            // FLAC
            FormatSignature {
                format: "flac",
                bytes: vec![0x66, 0x4C, 0x61, 0x43], // "fLaC"
                offset: None,
            },
            // WAV
            FormatSignature {
                format: "wav",
                bytes: vec![0x52, 0x49, 0x46, 0x46], // "RIFF"
                offset: None,
            },
            // OGG
            FormatSignature {
                format: "ogg",
                bytes: vec![0x4F, 0x67, 0x67, 0x53], // "OggS"
                offset: None,
            },
            // MPEG TS
            FormatSignature {
                format: "mpegts",
                bytes: vec![0x47],
                offset: None,
            },
        ]
    }
}

impl Default for FormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct FormatSignature {
    format: &'static str,
    bytes: Vec<u8>,
    offset: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = FormatDetector::new();
        assert!(!detector.magic_bytes.is_empty());
    }

    #[test]
    fn test_extension_detection() {
        let detector = FormatDetector::new();

        assert_eq!(
            detector.detect_format_from_extension("test.mp4"),
            Some("mp4".to_string())
        );
        assert_eq!(
            detector.detect_format_from_extension("test.mkv"),
            Some("matroska".to_string())
        );
        assert_eq!(
            detector.detect_format_from_extension("test.mp3"),
            Some("mp3".to_string())
        );
    }

    #[test]
    fn test_magic_bytes_detection() {
        let detector = FormatDetector::new();

        // MP3 with ID3
        let mp3_magic = vec![0x49, 0x44, 0x33, 0x00];
        assert_eq!(
            detector.detect_format_from_magic(&mp3_magic),
            Some("mp3".to_string())
        );

        // FLAC
        let flac_magic = vec![0x66, 0x4C, 0x61, 0x43];
        assert_eq!(
            detector.detect_format_from_magic(&flac_magic),
            Some("flac".to_string())
        );

        // OGG
        let ogg_magic = vec![0x4F, 0x67, 0x67, 0x53];
        assert_eq!(
            detector.detect_format_from_magic(&ogg_magic),
            Some("ogg".to_string())
        );
    }

    #[test]
    fn test_unknown_format() {
        let detector = FormatDetector::new();

        let unknown_magic = vec![0x00, 0x00, 0x00, 0x00];
        assert_eq!(detector.detect_format_from_magic(&unknown_magic), None);

        assert_eq!(detector.detect_format_from_extension("test.unknown"), None);
    }
}
