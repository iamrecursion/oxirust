//! Configuration types for adaptive streaming packaging.

use crate::error::{PackagerError, PackagerResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Packaging format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PackagingFormat {
    /// HLS with MPEG-TS segments.
    HlsTs,
    /// HLS with fragmented MP4 segments.
    #[default]
    HlsFmp4,
    /// DASH with fragmented MP4 segments.
    Dash,
    /// Both HLS and DASH.
    Both,
}

/// Segment format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentFormat {
    /// MPEG-TS segments (HLS).
    MpegTs,
    /// Fragmented MP4 segments.
    Fmp4,
    /// CMAF segments.
    Cmaf,
}

/// Encryption method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EncryptionMethod {
    /// No encryption.
    #[default]
    None,
    /// AES-128 encryption (HLS).
    Aes128,
    /// SAMPLE-AES encryption (HLS).
    SampleAes,
    /// Common Encryption (CENC) for DASH.
    Cenc,
}

/// Bitrate ladder entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitrateEntry {
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Video codec (AV1, VP9, VP8).
    pub codec: String,
    /// Frame rate (optional, defaults to source).
    pub framerate: Option<f64>,
}

impl BitrateEntry {
    /// Create a new bitrate entry.
    #[must_use]
    pub fn new(bitrate: u32, width: u32, height: u32, codec: &str) -> Self {
        Self {
            bitrate,
            width,
            height,
            codec: codec.to_string(),
            framerate: None,
        }
    }

    /// Set the frame rate.
    #[must_use]
    pub fn with_framerate(mut self, fps: f64) -> Self {
        self.framerate = Some(fps);
        self
    }
}

/// Bitrate ladder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitrateLadder {
    /// List of bitrate entries.
    pub entries: Vec<BitrateEntry>,
    /// Whether to generate automatically based on source.
    pub auto_generate: bool,
}

impl Default for BitrateLadder {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            auto_generate: true,
        }
    }
}

impl BitrateLadder {
    /// Create a new bitrate ladder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a bitrate entry.
    pub fn add_entry(&mut self, entry: BitrateEntry) {
        self.entries.push(entry);
    }

    /// Enable auto-generation.
    #[must_use]
    pub fn with_auto_generate(mut self, enabled: bool) -> Self {
        self.auto_generate = enabled;
        self
    }

    /// Validate the ladder configuration.
    pub fn validate(&self) -> PackagerResult<()> {
        if !self.auto_generate && self.entries.is_empty() {
            return Err(PackagerError::invalid_config(
                "Bitrate ladder has no entries and auto-generation is disabled",
            ));
        }

        for entry in &self.entries {
            if entry.bitrate == 0 {
                return Err(PackagerError::invalid_config("Bitrate cannot be zero"));
            }
            if entry.width == 0 || entry.height == 0 {
                return Err(PackagerError::invalid_config(
                    "Width and height must be greater than zero",
                ));
            }
        }

        Ok(())
    }
}

/// Segment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentConfig {
    /// Target segment duration.
    pub duration: Duration,
    /// Segment format.
    pub format: SegmentFormat,
    /// Enable keyframe alignment.
    pub keyframe_alignment: bool,
    /// Enable fast start (moov before mdat).
    pub fast_start: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(6),
            format: SegmentFormat::Fmp4,
            keyframe_alignment: true,
            fast_start: true,
        }
    }
}

/// Encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption method.
    pub method: EncryptionMethod,
    /// Encryption key (16 bytes for AES-128).
    pub key: Option<Vec<u8>>,
    /// Key URI for HLS.
    pub key_uri: Option<String>,
    /// IV (initialization vector).
    pub iv: Option<Vec<u8>>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            method: EncryptionMethod::None,
            key: None,
            key_uri: None,
            iv: None,
        }
    }
}

impl EncryptionConfig {
    /// Check if encryption is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.method != EncryptionMethod::None
    }

    /// Validate the encryption configuration.
    pub fn validate(&self) -> PackagerResult<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        match self.method {
            EncryptionMethod::None => Ok(()),
            EncryptionMethod::Aes128 | EncryptionMethod::SampleAes => {
                if self.key.is_none() {
                    return Err(PackagerError::invalid_config("Encryption key is required"));
                }
                if let Some(key) = &self.key {
                    if key.len() != 16 {
                        return Err(PackagerError::invalid_config(
                            "AES-128 key must be 16 bytes",
                        ));
                    }
                }
                if self.key_uri.is_none() {
                    return Err(PackagerError::invalid_config("Key URI is required for HLS"));
                }
                Ok(())
            }
            EncryptionMethod::Cenc => {
                if self.key.is_none() {
                    return Err(PackagerError::invalid_config("Encryption key is required"));
                }
                Ok(())
            }
        }
    }
}

/// Output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory.
    pub directory: PathBuf,
    /// Base URL for manifests.
    pub base_url: Option<String>,
    /// Enable S3 upload.
    pub s3_upload: bool,
    /// S3 bucket name.
    pub s3_bucket: Option<String>,
    /// S3 key prefix.
    pub s3_prefix: Option<String>,
    /// Keep local files after upload.
    pub keep_local: bool,
    /// Maximum number of segments to keep.
    pub max_segments: Option<usize>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("output"),
            base_url: None,
            s3_upload: false,
            s3_bucket: None,
            s3_prefix: None,
            keep_local: true,
            max_segments: None,
        }
    }
}

/// Main packager configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PackagerConfig {
    /// Packaging format.
    pub format: PackagingFormat,
    /// Bitrate ladder.
    pub ladder: BitrateLadder,
    /// Segment configuration.
    pub segment: SegmentConfig,
    /// Encryption configuration.
    pub encryption: EncryptionConfig,
    /// Output configuration.
    pub output: OutputConfig,
    /// Enable low latency mode.
    pub low_latency: bool,
    /// Manifest version tracking.
    pub manifest_versioning: bool,
    /// Optional explicit variant set for multi-variant playlist generation.
    ///
    /// When set, `MultivariantPlaylistBuilder` uses this set directly instead
    /// of deriving variants from the bitrate ladder.
    #[serde(skip)]
    pub variant_set: Option<crate::variant_stream::VariantSet>,
    /// Real source-media parameters (resolution, framerate, codec) used to
    /// derive an automatic bitrate ladder when `ladder.auto_generate` is set.
    ///
    /// The packager does not invent a source resolution: if this is `None`
    /// and auto-generation is requested, packaging fails with a clear error
    /// rather than emitting a ladder unrelated to the real input. Populate it
    /// from your own probe of the input (or via [`Self::with_source_info`]).
    #[serde(skip)]
    pub source_media: Option<crate::ladder::SourceInfo>,
}

impl PackagerConfig {
    /// Create a new packager configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the packaging format.
    #[must_use]
    pub fn with_format(mut self, format: PackagingFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the bitrate ladder.
    #[must_use]
    pub fn with_ladder(mut self, ladder: BitrateLadder) -> Self {
        self.ladder = ladder;
        self
    }

    /// Set the segment configuration.
    #[must_use]
    pub fn with_segment_config(mut self, segment: SegmentConfig) -> Self {
        self.segment = segment;
        self
    }

    /// Set the encryption configuration.
    #[must_use]
    pub fn with_encryption(mut self, encryption: EncryptionConfig) -> Self {
        self.encryption = encryption;
        self
    }

    /// Set the output configuration.
    #[must_use]
    pub fn with_output(mut self, output: OutputConfig) -> Self {
        self.output = output;
        self
    }

    /// Enable low latency mode.
    #[must_use]
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Set an explicit variant set for multi-variant playlist generation.
    #[must_use]
    pub fn with_variant_set(mut self, vs: crate::variant_stream::VariantSet) -> Self {
        self.variant_set = Some(vs);
        self
    }

    /// Set the real source-media parameters used for automatic bitrate ladder
    /// generation (see [`Self::source_media`]).
    #[must_use]
    pub fn with_source_info(mut self, source: crate::ladder::SourceInfo) -> Self {
        self.source_media = Some(source);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> PackagerResult<()> {
        self.ladder.validate()?;
        self.encryption.validate()?;

        if self.output.s3_upload && self.output.s3_bucket.is_none() {
            return Err(PackagerError::invalid_config("S3 bucket is required"));
        }

        if let Some(vs) = &self.variant_set {
            vs.validate()?;
        }

        Ok(())
    }
}
