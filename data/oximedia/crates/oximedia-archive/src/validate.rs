//! Media container and codec validation
//!
//! This module provides comprehensive validation for:
//! - Container formats (Matroska/MKV, MP4, AVI, etc.)
//! - Codec parameters
//! - Stream integrity
//! - Metadata validation
//! - Structural validation
//! - Format compliance checking

use crate::{ArchiveError, ArchiveResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use tokio::fs;
use tracing::{debug, info};

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub file_path: String,
    pub container_format: Option<String>,
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
    pub streams: Vec<StreamInfo>,
    pub metadata: Option<MediaMetadata>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub severity: ErrorSeverity,
    pub error_type: ErrorType,
    pub message: String,
    pub stream_index: Option<usize>,
}

/// Error severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Critical,
    Major,
    Minor,
    Warning,
}

/// Error type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    ContainerCorruption,
    StreamCorruption,
    CodecError,
    MetadataError,
    StructuralError,
    ComplianceViolation,
    Unknown,
}

/// Stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    pub index: usize,
    pub codec_type: String,
    pub codec_name: String,
    pub duration: Option<f64>,
    pub bitrate: Option<u64>,
    pub valid: bool,
    pub errors: Vec<String>,
}

/// Media metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    pub title: Option<String>,
    pub duration: Option<f64>,
    pub bitrate: Option<u64>,
    pub format_name: Option<String>,
    pub format_long_name: Option<String>,
    pub size: Option<u64>,
}

/// Validate a media file
pub async fn validate_file(path: &Path) -> ArchiveResult<ValidationResult> {
    info!("Validating file: {}", path.display());

    if !path.exists() {
        return Err(ArchiveError::Validation("File does not exist".to_string()));
    }

    let metadata = fs::metadata(path).await?;
    if !metadata.is_file() {
        return Err(ArchiveError::Validation("Path is not a file".to_string()));
    }

    // Detect container format
    let container_format = detect_container_format(path).await?;

    let mut result = ValidationResult {
        file_path: path.to_string_lossy().to_string(),
        container_format: Some(container_format.clone()),
        is_valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
        streams: Vec::new(),
        metadata: None,
    };

    // Validate using ffprobe (if available)
    if let Ok(ffprobe_result) = validate_with_ffprobe(path).await {
        result.streams = ffprobe_result.streams;
        result.metadata = Some(ffprobe_result.metadata);

        // Check for errors in streams
        for stream in &result.streams {
            if !stream.valid {
                result.is_valid = false;
                for error in &stream.errors {
                    result.errors.push(ValidationError {
                        severity: ErrorSeverity::Major,
                        error_type: ErrorType::StreamCorruption,
                        message: error.clone(),
                        stream_index: Some(stream.index),
                    });
                }
            }
        }
    }

    // Format-specific validation
    match container_format.as_str() {
        "matroska" | "mkv" => validate_matroska(path, &mut result).await?,
        "mp4" | "mov" => validate_mp4(path, &mut result).await?,
        "avi" => validate_avi(path, &mut result).await?,
        _ => {
            debug!("No specific validator for format: {}", container_format);
        }
    }

    // Structural validation
    validate_structure(path, &mut result).await?;

    // Compliance checking
    check_compliance(&container_format, &mut result).await?;

    if !result.errors.is_empty() {
        result.is_valid = false;
    }

    Ok(result)
}

/// Known file format identified by magic bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagicBytesMatch {
    /// Short identifier (e.g. "png", "flac").
    pub format_id: String,
    /// Human-readable format name.
    pub format_name: String,
    /// MIME type.
    pub mime_type: String,
    /// Offset at which the magic bytes were found.
    pub offset: usize,
    /// Length of the magic byte signature that matched.
    pub signature_len: usize,
    /// Confidence level (0.0 to 1.0). Magic-byte matches are high confidence;
    /// extension-only matches are lower.
    pub confidence: f64,
}

/// A magic bytes signature entry used for format identification.
struct MagicSignature {
    /// Offset in the file where the signature starts.
    offset: usize,
    /// The byte sequence to match.
    bytes: &'static [u8],
    /// Short format id.
    format_id: &'static str,
    /// Human-readable name.
    format_name: &'static str,
    /// MIME type.
    mime_type: &'static str,
}

/// Return the built-in table of magic byte signatures for media formats.
///
/// Covers containers, audio, image, subtitle, and archive formats relevant
/// to media preservation workflows. Ordered so that signatures requiring
/// larger offsets come after zero-offset ones; the first match wins in
/// most cases, but [`identify_format_by_magic`] returns *all* matches.
fn magic_signatures() -> Vec<MagicSignature> {
    vec![
        // --- Containers / Video ---
        MagicSignature {
            offset: 0,
            bytes: &[0x1A, 0x45, 0xDF, 0xA3],
            format_id: "matroska",
            format_name: "Matroska/WebM (EBML)",
            mime_type: "video/x-matroska",
        },
        MagicSignature {
            offset: 4,
            bytes: b"ftyp",
            format_id: "mp4",
            format_name: "ISO Base Media (MP4/MOV)",
            mime_type: "video/mp4",
        },
        MagicSignature {
            offset: 0,
            bytes: b"RIFF",
            format_id: "riff",
            format_name: "RIFF container",
            mime_type: "application/octet-stream",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x00, 0x00, 0x01, 0xBA],
            format_id: "mpeg-ps",
            format_name: "MPEG Program Stream",
            mime_type: "video/mpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x47],
            format_id: "mpeg-ts",
            format_name: "MPEG Transport Stream",
            mime_type: "video/mp2t",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11],
            format_id: "asf",
            format_name: "ASF/WMV/WMA",
            mime_type: "video/x-ms-asf",
        },
        MagicSignature {
            offset: 0,
            bytes: b"FLV\x01",
            format_id: "flv",
            format_name: "Flash Video",
            mime_type: "video/x-flv",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x06, 0x0E, 0x2B, 0x34],
            format_id: "mxf",
            format_name: "MXF (Material eXchange Format)",
            mime_type: "application/mxf",
        },
        // --- Audio ---
        MagicSignature {
            offset: 0,
            bytes: b"fLaC",
            format_id: "flac",
            format_name: "FLAC",
            mime_type: "audio/flac",
        },
        MagicSignature {
            offset: 0,
            bytes: b"OggS",
            format_id: "ogg",
            format_name: "Ogg container (Vorbis/Opus/Theora)",
            mime_type: "audio/ogg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xFB],
            format_id: "mp3",
            format_name: "MP3 (MPEG Audio Layer III)",
            mime_type: "audio/mpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xF3],
            format_id: "mp3",
            format_name: "MP3 (MPEG Audio Layer III)",
            mime_type: "audio/mpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xF2],
            format_id: "mp3",
            format_name: "MP3 (MPEG Audio Layer III)",
            mime_type: "audio/mpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: b"ID3",
            format_id: "mp3",
            format_name: "MP3 with ID3 tag",
            mime_type: "audio/mpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xF1],
            format_id: "aac",
            format_name: "AAC (ADTS)",
            mime_type: "audio/aac",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xF9],
            format_id: "aac",
            format_name: "AAC (ADTS)",
            mime_type: "audio/aac",
        },
        MagicSignature {
            offset: 0,
            bytes: b".snd",
            format_id: "au",
            format_name: "Sun/NeXT AU audio",
            mime_type: "audio/basic",
        },
        MagicSignature {
            offset: 8,
            bytes: b"AIFF",
            format_id: "aiff",
            format_name: "AIFF audio",
            mime_type: "audio/aiff",
        },
        // --- Image ---
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0xD8, 0xFF],
            format_id: "jpeg",
            format_name: "JPEG",
            mime_type: "image/jpeg",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            format_id: "png",
            format_name: "PNG",
            mime_type: "image/png",
        },
        MagicSignature {
            offset: 0,
            bytes: b"GIF87a",
            format_id: "gif",
            format_name: "GIF87a",
            mime_type: "image/gif",
        },
        MagicSignature {
            offset: 0,
            bytes: b"GIF89a",
            format_id: "gif",
            format_name: "GIF89a",
            mime_type: "image/gif",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x49, 0x49, 0x2A, 0x00],
            format_id: "tiff",
            format_name: "TIFF (little-endian)",
            mime_type: "image/tiff",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x4D, 0x4D, 0x00, 0x2A],
            format_id: "tiff",
            format_name: "TIFF (big-endian)",
            mime_type: "image/tiff",
        },
        MagicSignature {
            offset: 0,
            bytes: b"BM",
            format_id: "bmp",
            format_name: "BMP",
            mime_type: "image/bmp",
        },
        MagicSignature {
            offset: 0,
            bytes: b"RIFF",
            format_id: "webp_check",
            format_name: "RIFF (check WEBP)",
            mime_type: "image/webp",
        },
        MagicSignature {
            offset: 0,
            bytes: &[
                0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
            ],
            format_id: "jxl",
            format_name: "JPEG XL (container)",
            mime_type: "image/jxl",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFF, 0x0A],
            format_id: "jxl",
            format_name: "JPEG XL (naked codestream)",
            mime_type: "image/jxl",
        },
        MagicSignature {
            offset: 0,
            bytes: b"SDPX",
            format_id: "dpx",
            format_name: "DPX (big-endian)",
            mime_type: "image/x-dpx",
        },
        MagicSignature {
            offset: 0,
            bytes: b"XPDS",
            format_id: "dpx",
            format_name: "DPX (little-endian)",
            mime_type: "image/x-dpx",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x76, 0x2F, 0x31, 0x01],
            format_id: "exr",
            format_name: "OpenEXR",
            mime_type: "image/x-exr",
        },
        MagicSignature {
            offset: 0,
            bytes: b"DNG",
            format_id: "dng",
            format_name: "Adobe DNG",
            mime_type: "image/x-adobe-dng",
        },
        // --- Subtitle / Caption ---
        // (text-based, but some have byte-order marks or specific patterns)
        MagicSignature {
            offset: 0,
            bytes: &[0xEF, 0xBB, 0xBF],
            format_id: "utf8-bom",
            format_name: "UTF-8 BOM (text/subtitle)",
            mime_type: "text/plain",
        },
        // --- Archive / Compression ---
        MagicSignature {
            offset: 0,
            bytes: &[0x50, 0x4B, 0x03, 0x04],
            format_id: "zip",
            format_name: "ZIP archive",
            mime_type: "application/zip",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x1F, 0x8B],
            format_id: "gzip",
            format_name: "GZIP",
            mime_type: "application/gzip",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x28, 0xB5, 0x2F, 0xFD],
            format_id: "zstd",
            format_name: "Zstandard",
            mime_type: "application/zstd",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
            format_id: "xz",
            format_name: "XZ",
            mime_type: "application/x-xz",
        },
        MagicSignature {
            offset: 0,
            bytes: &[0x04, 0x22, 0x4D, 0x18],
            format_id: "lz4",
            format_name: "LZ4 frame",
            mime_type: "application/x-lz4",
        },
        MagicSignature {
            offset: 0,
            bytes: b"7z\xBC\xAF\x27\x1C",
            format_id: "7z",
            format_name: "7-Zip",
            mime_type: "application/x-7z-compressed",
        },
        MagicSignature {
            offset: 0,
            bytes: b"Rar!\x1A\x07",
            format_id: "rar",
            format_name: "RAR archive",
            mime_type: "application/vnd.rar",
        },
        // --- Multimedia metadata ---
        MagicSignature {
            offset: 0,
            bytes: b"FORM",
            format_id: "iff",
            format_name: "IFF (Interchange File Format)",
            mime_type: "application/octet-stream",
        },
    ]
}

/// Identify a file format using magic bytes analysis.
///
/// Reads up to the first 64 bytes of the file and matches against all known
/// signatures. Returns all matches (there may be more than one for ambiguous
/// headers like RIFF, which can be AVI, WAV, or WebP). Results are sorted by
/// descending confidence.
pub async fn identify_format_by_magic(path: &Path) -> ArchiveResult<Vec<MagicBytesMatch>> {
    let mut file = fs::File::open(path).await?;
    let mut buffer = vec![0u8; 64];

    use tokio::io::AsyncReadExt;
    let bytes_read = file.read(&mut buffer).await?;
    buffer.truncate(bytes_read);

    let mut matches = identify_format_from_bytes(&buffer);

    // RIFF sub-format disambiguation
    if buffer.len() >= 12 && buffer.starts_with(b"RIFF") {
        let sub = &buffer[8..12];
        if sub == b"AVI " {
            matches.retain(|m| m.format_id != "riff" && m.format_id != "webp_check");
            matches.push(MagicBytesMatch {
                format_id: "avi".to_string(),
                format_name: "AVI".to_string(),
                mime_type: "video/x-msvideo".to_string(),
                offset: 0,
                signature_len: 12,
                confidence: 0.95,
            });
        } else if sub == b"WAVE" {
            matches.retain(|m| m.format_id != "riff" && m.format_id != "webp_check");
            matches.push(MagicBytesMatch {
                format_id: "wav".to_string(),
                format_name: "WAV audio".to_string(),
                mime_type: "audio/wav".to_string(),
                offset: 0,
                signature_len: 12,
                confidence: 0.95,
            });
        } else if sub == b"WEBP" {
            matches.retain(|m| m.format_id != "riff" && m.format_id != "webp_check");
            matches.push(MagicBytesMatch {
                format_id: "webp".to_string(),
                format_name: "WebP image".to_string(),
                mime_type: "image/webp".to_string(),
                offset: 0,
                signature_len: 12,
                confidence: 0.95,
            });
        }
    }

    // Remove the temporary webp_check entry if it slipped through
    matches.retain(|m| m.format_id != "webp_check");

    // Sort by descending confidence, then by signature length (longer = more specific)
    matches.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.signature_len.cmp(&a.signature_len))
    });

    Ok(matches)
}

/// Identify format from an in-memory byte buffer (no file I/O).
pub fn identify_format_from_bytes(buffer: &[u8]) -> Vec<MagicBytesMatch> {
    let signatures = magic_signatures();
    let mut matches = Vec::new();

    for sig in &signatures {
        let end = sig.offset + sig.bytes.len();
        if buffer.len() >= end && buffer[sig.offset..end] == *sig.bytes {
            matches.push(MagicBytesMatch {
                format_id: sig.format_id.to_string(),
                format_name: sig.format_name.to_string(),
                mime_type: sig.mime_type.to_string(),
                offset: sig.offset,
                signature_len: sig.bytes.len(),
                confidence: 0.90,
            });
        }
    }

    matches
}

/// Detect container format.
///
/// Uses magic-byte identification first, then falls back to file extension.
pub async fn detect_container_format(path: &Path) -> ArchiveResult<String> {
    // Try magic bytes first
    let magic_matches = identify_format_by_magic(path).await?;
    if let Some(best) = magic_matches.first() {
        return Ok(best.format_id.clone());
    }

    // Fall back to extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        let id = match ext_str.as_str() {
            "mkv" | "mka" | "mks" => "matroska",
            "mp4" | "m4v" | "m4a" => "mp4",
            "mov" => "mov",
            "avi" => "avi",
            "webm" => "webm",
            "flac" => "flac",
            "ogg" | "oga" | "ogv" => "ogg",
            "wav" => "wav",
            "mp3" => "mp3",
            "aac" => "aac",
            "mxf" => "mxf",
            "dpx" => "dpx",
            "exr" => "exr",
            "tiff" | "tif" => "tiff",
            "png" => "png",
            "jpg" | "jpeg" => "jpeg",
            "gif" => "gif",
            "bmp" => "bmp",
            "webp" => "webp",
            "jxl" => "jxl",
            "srt" => "srt",
            "vtt" => "vtt",
            "ass" | "ssa" => "ass",
            _ => "unknown",
        };
        return Ok(id.to_string());
    }

    Ok("unknown".to_string())
}

/// Validate using ffprobe
#[derive(Debug)]
struct FfprobeResult {
    streams: Vec<StreamInfo>,
    metadata: MediaMetadata,
}

async fn validate_with_ffprobe(path: &Path) -> ArchiveResult<FfprobeResult> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            path.to_str()
                .ok_or_else(|| ArchiveError::Validation("Invalid path".to_string()))?,
        ])
        .output()
        .map_err(|e| ArchiveError::Validation(format!("ffprobe not available: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ArchiveError::Validation(format!(
            "ffprobe failed: {stderr}"
        )));
    }

    let json_str = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| ArchiveError::Validation(format!("Failed to parse ffprobe output: {e}")))?;

    // Parse streams
    let mut streams = Vec::new();
    if let Some(streams_array) = json["streams"].as_array() {
        for (index, stream) in streams_array.iter().enumerate() {
            let codec_type = stream["codec_type"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let codec_name = stream["codec_name"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let duration = stream["duration"]
                .as_str()
                .and_then(|s| s.parse::<f64>().ok());
            let bitrate = stream["bit_rate"]
                .as_str()
                .and_then(|s| s.parse::<u64>().ok());

            streams.push(StreamInfo {
                index,
                codec_type,
                codec_name,
                duration,
                bitrate,
                valid: true,
                errors: Vec::new(),
            });
        }
    }

    // Parse format metadata
    let format = &json["format"];
    let metadata = MediaMetadata {
        title: format["tags"]["title"].as_str().map(String::from),
        duration: format["duration"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok()),
        bitrate: format["bit_rate"]
            .as_str()
            .and_then(|s| s.parse::<u64>().ok()),
        format_name: format["format_name"].as_str().map(String::from),
        format_long_name: format["format_long_name"].as_str().map(String::from),
        size: format["size"].as_str().and_then(|s| s.parse::<u64>().ok()),
    };

    Ok(FfprobeResult { streams, metadata })
}

/// Validate Matroska/MKV container
async fn validate_matroska(path: &Path, result: &mut ValidationResult) -> ArchiveResult<()> {
    debug!(
        "Performing Matroska-specific validation for {}",
        path.display()
    );

    // Check EBML header
    let mut file = fs::File::open(path).await?;
    let mut header = vec![0u8; 4];

    use tokio::io::AsyncReadExt;
    file.read_exact(&mut header).await?;

    if header != [0x1A, 0x45, 0xDF, 0xA3] {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Critical,
            error_type: ErrorType::ContainerCorruption,
            message: "Invalid EBML header".to_string(),
            stream_index: None,
        });
        return Ok(());
    }

    // Use mkvinfo if available
    if let Ok(output) = Command::new("mkvinfo")
        .arg(
            path.to_str()
                .ok_or_else(|| ArchiveError::Validation("Invalid path".to_string()))?,
        )
        .output()
    {
        if !output.status.success() {
            result.errors.push(ValidationError {
                severity: ErrorSeverity::Major,
                error_type: ErrorType::StructuralError,
                message: "mkvinfo validation failed".to_string(),
                stream_index: None,
            });
        }
    }

    Ok(())
}

/// Validate MP4 container
async fn validate_mp4(path: &Path, result: &mut ValidationResult) -> ArchiveResult<()> {
    debug!("Performing MP4-specific validation for {}", path.display());

    // Check ftyp box
    let mut file = fs::File::open(path).await?;
    let mut header = vec![0u8; 12];

    use tokio::io::AsyncReadExt;
    file.read_exact(&mut header).await?;

    if header[4..8] != [b'f', b't', b'y', b'p'] {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Critical,
            error_type: ErrorType::ContainerCorruption,
            message: "Invalid MP4 ftyp box".to_string(),
            stream_index: None,
        });
        return Ok(());
    }

    // Validate atom structure
    validate_mp4_atoms(path, result).await?;

    Ok(())
}

/// Validate MP4 atom structure
async fn validate_mp4_atoms(path: &Path, result: &mut ValidationResult) -> ArchiveResult<()> {
    // Basic atom validation
    let file = fs::File::open(path).await?;
    let metadata = file.metadata().await?;
    let file_size = metadata.len();

    // Check if file is large enough to contain required atoms
    if file_size < 32 {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Critical,
            error_type: ErrorType::StructuralError,
            message: "File too small to be a valid MP4".to_string(),
            stream_index: None,
        });
    }

    // Use MP4Box if available for detailed validation
    if let Ok(output) = Command::new("MP4Box")
        .args([
            "-info",
            path.to_str()
                .ok_or_else(|| ArchiveError::Validation("Invalid path".to_string()))?,
        ])
        .output()
    {
        if !output.status.success() {
            result
                .warnings
                .push("MP4Box validation reported issues".to_string());
        }
    }

    Ok(())
}

/// Validate AVI container
async fn validate_avi(path: &Path, result: &mut ValidationResult) -> ArchiveResult<()> {
    debug!("Performing AVI-specific validation for {}", path.display());

    // Check RIFF header
    let mut file = fs::File::open(path).await?;
    let mut header = vec![0u8; 12];

    use tokio::io::AsyncReadExt;
    file.read_exact(&mut header).await?;

    if &header[0..4] != b"RIFF" {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Critical,
            error_type: ErrorType::ContainerCorruption,
            message: "Invalid RIFF header".to_string(),
            stream_index: None,
        });
        return Ok(());
    }

    if &header[8..12] != b"AVI " {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Critical,
            error_type: ErrorType::ContainerCorruption,
            message: "Invalid AVI signature".to_string(),
            stream_index: None,
        });
        return Ok(());
    }

    Ok(())
}

/// Validate file structure
async fn validate_structure(path: &Path, result: &mut ValidationResult) -> ArchiveResult<()> {
    let metadata = fs::metadata(path).await?;
    let file_size = metadata.len();

    // Check minimum file size
    if file_size < 1024 {
        result
            .warnings
            .push(format!("File size is very small: {file_size} bytes"));
    }

    // Check if file is readable throughout
    let mut file = fs::File::open(path).await?;
    let mut buffer = vec![0u8; 8192];
    let mut total_read = 0u64;

    use tokio::io::AsyncReadExt;
    loop {
        match file.read(&mut buffer).await {
            Ok(0) => break,
            Ok(n) => total_read += n as u64,
            Err(e) => {
                result.errors.push(ValidationError {
                    severity: ErrorSeverity::Critical,
                    error_type: ErrorType::StructuralError,
                    message: format!("Read error at byte {total_read}: {e}"),
                    stream_index: None,
                });
                return Ok(());
            }
        }
    }

    if total_read != file_size {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Major,
            error_type: ErrorType::StructuralError,
            message: format!(
                "File size mismatch: expected {file_size} bytes, read {total_read} bytes"
            ),
            stream_index: None,
        });
    }

    Ok(())
}

/// Check format compliance
async fn check_compliance(
    container_format: &str,
    result: &mut ValidationResult,
) -> ArchiveResult<()> {
    match container_format {
        "matroska" | "mkv" => {
            // Check Matroska compliance
            check_matroska_compliance(result).await?;
        }
        "mp4" | "mov" => {
            // Check MP4 compliance
            check_mp4_compliance(result).await?;
        }
        _ => {
            debug!("No compliance checks for format: {}", container_format);
        }
    }

    Ok(())
}

/// Check Matroska compliance
async fn check_matroska_compliance(result: &mut ValidationResult) -> ArchiveResult<()> {
    // Check for required Matroska elements
    // This is a simplified check - full compliance checking would require parsing EBML

    if result.streams.is_empty() {
        result
            .warnings
            .push("No streams found in Matroska file".to_string());
    }

    // Check for video stream if this is a video file
    let has_video = result.streams.iter().any(|s| s.codec_type == "video");
    let has_audio = result.streams.iter().any(|s| s.codec_type == "audio");

    if !has_video && !has_audio {
        result.errors.push(ValidationError {
            severity: ErrorSeverity::Major,
            error_type: ErrorType::ComplianceViolation,
            message: "No video or audio streams found".to_string(),
            stream_index: None,
        });
    }

    Ok(())
}

/// Check MP4 compliance
async fn check_mp4_compliance(result: &mut ValidationResult) -> ArchiveResult<()> {
    // Check for required MP4 boxes (simplified)

    if result.streams.is_empty() {
        result
            .warnings
            .push("No streams found in MP4 file".to_string());
    }

    // Check for common codec compliance
    for stream in &result.streams {
        if stream.codec_type == "video" {
            match stream.codec_name.as_str() {
                "h264" | "hevc" | "vp9" | "av1" => {
                    // Common codecs - OK
                }
                _ => {
                    result.warnings.push(format!(
                        "Uncommon video codec in MP4: {}",
                        stream.codec_name
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Codec parameter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecValidation {
    pub codec_name: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validate codec parameters
#[allow(dead_code)]
pub async fn validate_codec_parameters(
    codec_name: &str,
    stream_info: &StreamInfo,
) -> ArchiveResult<CodecValidation> {
    let mut validation = CodecValidation {
        codec_name: codec_name.to_string(),
        is_valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Check bitrate
    if stream_info.codec_type == "video" {
        if let Some(bitrate) = stream_info.bitrate {
            if bitrate < 100_000 {
                validation
                    .warnings
                    .push(format!("Very low bitrate: {bitrate} bps"));
            }
            if bitrate > 100_000_000 {
                validation
                    .warnings
                    .push(format!("Very high bitrate: {bitrate} bps"));
            }
        }
    }

    // Check duration
    if let Some(duration) = stream_info.duration {
        if duration <= 0.0 {
            validation
                .errors
                .push("Invalid duration: must be positive".to_string());
            validation.is_valid = false;
        }
    }

    Ok(validation)
}

/// Stream integrity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamIntegrityResult {
    pub stream_index: usize,
    pub is_intact: bool,
    pub frame_errors: u32,
    pub packet_errors: u32,
    pub details: Vec<String>,
}

/// Check stream integrity using ffmpeg
#[allow(dead_code)]
pub async fn check_stream_integrity(
    path: &Path,
    stream_index: usize,
) -> ArchiveResult<StreamIntegrityResult> {
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str()
                .ok_or_else(|| ArchiveError::Validation("Invalid path".to_string()))?,
            "-map",
            &format!("0:{stream_index}"),
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|e| ArchiveError::Validation(format!("ffmpeg not available: {e}")))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let frame_errors = stderr.matches("error").count() as u32;
    let packet_errors = stderr.matches("corrupt").count() as u32;

    let is_intact = frame_errors == 0 && packet_errors == 0;

    let mut details = Vec::new();
    if !is_intact {
        for line in stderr.lines() {
            if line.contains("error") || line.contains("corrupt") {
                details.push(line.to_string());
            }
        }
    }

    Ok(StreamIntegrityResult {
        stream_index,
        is_intact,
        frame_errors,
        packet_errors,
        details,
    })
}

/// Metadata validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataValidation {
    pub has_duration: bool,
    pub has_bitrate: bool,
    pub has_title: bool,
    pub is_complete: bool,
    pub missing_fields: Vec<String>,
}

/// Validate metadata completeness
pub fn validate_metadata(metadata: &MediaMetadata) -> MetadataValidation {
    let has_duration = metadata.duration.is_some();
    let has_bitrate = metadata.bitrate.is_some();
    let has_title = metadata.title.is_some();

    let mut missing_fields = Vec::new();
    if !has_duration {
        missing_fields.push("duration".to_string());
    }
    if !has_bitrate {
        missing_fields.push("bitrate".to_string());
    }

    let is_complete = missing_fields.is_empty();

    MetadataValidation {
        has_duration,
        has_bitrate,
        has_title,
        is_complete,
        missing_fields,
    }
}

/// Deep validation (comprehensive check)
#[allow(dead_code)]
pub async fn deep_validate(path: &Path) -> ArchiveResult<DeepValidationResult> {
    info!("Performing deep validation for {}", path.display());

    let mut result = DeepValidationResult {
        file_path: path.to_string_lossy().to_string(),
        validation_result: validate_file(path).await?,
        stream_integrity: Vec::new(),
        codec_validations: Vec::new(),
        metadata_validation: None,
    };

    // Check each stream's integrity
    for stream in &result.validation_result.streams {
        if let Ok(integrity) = check_stream_integrity(path, stream.index).await {
            result.stream_integrity.push(integrity);
        }
    }

    // Validate codec parameters
    for stream in &result.validation_result.streams {
        if let Ok(codec_val) = validate_codec_parameters(&stream.codec_name, stream).await {
            result.codec_validations.push(codec_val);
        }
    }

    // Validate metadata
    if let Some(ref metadata) = result.validation_result.metadata {
        result.metadata_validation = Some(validate_metadata(metadata));
    }

    Ok(result)
}

/// Deep validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepValidationResult {
    pub file_path: String,
    pub validation_result: ValidationResult,
    pub stream_integrity: Vec<StreamIntegrityResult>,
    pub codec_validations: Vec<CodecValidation>,
    pub metadata_validation: Option<MetadataValidation>,
}

impl DeepValidationResult {
    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.validation_result.is_valid
            && self.stream_integrity.iter().all(|s| s.is_intact)
            && self.codec_validations.iter().all(|c| c.is_valid)
    }
}
