//! Side-car metadata file management.
//!
//! A "side-car" is a companion file that lives alongside a media file and stores
//! supplemental metadata: proxy registry info, processing history, editorial notes,
//! checksum, timecode, color metadata, etc.
//!
//! Side-car files use the `.oxsc` extension (OxiMedia Side-Car) and are serialized
//! as JSON for human-readability and easy debugging.

use crate::{ProxyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Side-car format version for future compatibility.
const SIDECAR_VERSION: u32 = 1;

/// Checksum algorithm used for integrity verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// MD5 (fast, not cryptographically secure).
    Md5,
    /// SHA-256 (secure, recommended for archival).
    Sha256,
    /// CRC32 (fastest, lowest security).
    Crc32,
    /// xxHash64 (very fast, good distribution).
    XxHash64,
}

impl ChecksumAlgorithm {
    /// Get algorithm name string.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Md5 => "md5",
            Self::Sha256 => "sha256",
            Self::Crc32 => "crc32",
            Self::XxHash64 => "xxhash64",
        }
    }
}

/// File integrity checksum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checksum {
    /// Algorithm used.
    pub algorithm: ChecksumAlgorithm,
    /// Hex-encoded checksum value.
    pub value: String,
    /// Size of file at time of checksum computation.
    pub file_size: u64,
}

impl Checksum {
    /// Create a new checksum record.
    #[must_use]
    pub fn new(algorithm: ChecksumAlgorithm, value: impl Into<String>, file_size: u64) -> Self {
        Self {
            algorithm,
            value: value.into(),
            file_size,
        }
    }
}

/// Timecode metadata stored in a side-car.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarTimecode {
    /// Start timecode string (e.g. "01:00:00:00").
    pub start: String,
    /// Duration in frames.
    pub duration_frames: u64,
    /// Frame rate as fraction string (e.g. "24/1").
    pub frame_rate: String,
    /// Whether drop-frame timecode is used.
    pub drop_frame: bool,
}

impl SidecarTimecode {
    /// Create a new timecode entry.
    #[must_use]
    pub fn new(
        start: impl Into<String>,
        duration_frames: u64,
        frame_rate: impl Into<String>,
        drop_frame: bool,
    ) -> Self {
        Self {
            start: start.into(),
            duration_frames,
            frame_rate: frame_rate.into(),
            drop_frame,
        }
    }
}

/// Processing history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRecord {
    /// Name of the operation (e.g. "proxy_generate", "color_grade").
    pub operation: String,
    /// ISO8601 timestamp.
    pub timestamp: String,
    /// Tool/software that performed the operation.
    pub tool: String,
    /// Tool version string.
    pub tool_version: String,
    /// Additional parameters or notes.
    pub params: HashMap<String, String>,
}

impl ProcessingRecord {
    /// Create a new processing record.
    #[must_use]
    pub fn new(operation: impl Into<String>, tool: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            timestamp: String::new(),
            tool: tool.into(),
            tool_version: String::new(),
            params: HashMap::new(),
        }
    }

    /// Add a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// The main side-car data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarData {
    /// Side-car format version.
    pub version: u32,
    /// Path to the associated media file (relative or absolute).
    pub media_path: PathBuf,
    /// File integrity checksum.
    pub checksum: Option<Checksum>,
    /// Timecode information.
    pub timecode: Option<SidecarTimecode>,
    /// Processing history (most recent last).
    pub history: Vec<ProcessingRecord>,
    /// Proxy paths (spec name → proxy path).
    pub proxies: HashMap<String, PathBuf>,
    /// Arbitrary user/application metadata.
    pub metadata: HashMap<String, String>,
    /// Whether this file has been verified against its checksum.
    pub integrity_verified: bool,
    /// Optional editorial notes.
    pub notes: String,
}

impl SidecarData {
    /// Create a new side-car for the given media path.
    #[must_use]
    pub fn new(media_path: PathBuf) -> Self {
        Self {
            version: SIDECAR_VERSION,
            media_path,
            checksum: None,
            timecode: None,
            history: Vec::new(),
            proxies: HashMap::new(),
            metadata: HashMap::new(),
            integrity_verified: false,
            notes: String::new(),
        }
    }

    /// Set the checksum.
    pub fn set_checksum(&mut self, checksum: Checksum) {
        self.integrity_verified = false;
        self.checksum = Some(checksum);
    }

    /// Set timecode info.
    pub fn set_timecode(&mut self, tc: SidecarTimecode) {
        self.timecode = Some(tc);
    }

    /// Register a proxy path under a spec name.
    pub fn add_proxy(&mut self, spec_name: impl Into<String>, path: PathBuf) {
        self.proxies.insert(spec_name.into(), path);
    }

    /// Add a processing record to history.
    pub fn add_history(&mut self, record: ProcessingRecord) {
        self.history.push(record);
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Number of proxies registered.
    #[must_use]
    pub fn proxy_count(&self) -> usize {
        self.proxies.len()
    }
}

/// Manages reading and writing side-car files.
pub struct SideCar;

impl SideCar {
    /// Get the side-car path for a media file.
    ///
    /// The side-car is placed alongside the media file with `.oxsc` appended.
    #[must_use]
    pub fn path_for(media: &Path) -> PathBuf {
        let mut p = media.to_path_buf();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        p.set_file_name(format!("{name}.oxsc"));
        p
    }

    /// Load a side-car file from its media path.
    ///
    /// # Errors
    ///
    /// Returns an error if the side-car does not exist or cannot be parsed.
    pub fn load(media: &Path) -> Result<SidecarData> {
        let sc_path = Self::path_for(media);
        if !sc_path.exists() {
            return Err(ProxyError::FileNotFound(sc_path.display().to_string()));
        }
        let content = std::fs::read_to_string(&sc_path).map_err(ProxyError::IoError)?;
        serde_json::from_str(&content)
            .map_err(|e| ProxyError::MetadataError(format!("Side-car parse error: {e}")))
    }

    /// Load a side-car or create a default one if none exists.
    ///
    /// # Errors
    ///
    /// Returns an error only if an existing side-car file can't be read.
    pub fn load_or_create(media: &Path) -> Result<SidecarData> {
        let sc_path = Self::path_for(media);
        if sc_path.exists() {
            Self::load(media)
        } else {
            Ok(SidecarData::new(media.to_path_buf()))
        }
    }

    /// Save a side-car file alongside the media path.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file write fails.
    pub fn save(media: &Path, data: &SidecarData) -> Result<()> {
        let sc_path = Self::path_for(media);
        let json = serde_json::to_string_pretty(data)
            .map_err(|e| ProxyError::MetadataError(e.to_string()))?;
        std::fs::write(&sc_path, json).map_err(ProxyError::IoError)
    }

    /// Delete the side-car file for a media path (if it exists).
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be deleted.
    pub fn delete(media: &Path) -> Result<()> {
        let sc_path = Self::path_for(media);
        if sc_path.exists() {
            std::fs::remove_file(sc_path).map_err(ProxyError::IoError)?;
        }
        Ok(())
    }

    /// Check whether a side-car exists for a given media path.
    #[must_use]
    pub fn exists(media: &Path) -> bool {
        Self::path_for(media).exists()
    }

    /// Compute a simple (fake/mock) checksum string for testing without crypto deps.
    ///
    /// This is a placeholder that produces a deterministic hash for the file size.
    /// Real implementations would use sha2 / md5 crates.
    #[must_use]
    pub fn mock_checksum(data: &[u8]) -> String {
        // Simple FNV-1a 64-bit hash as a stand-in
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_algorithm_names() {
        assert_eq!(ChecksumAlgorithm::Md5.name(), "md5");
        assert_eq!(ChecksumAlgorithm::Sha256.name(), "sha256");
        assert_eq!(ChecksumAlgorithm::Crc32.name(), "crc32");
        assert_eq!(ChecksumAlgorithm::XxHash64.name(), "xxhash64");
    }

    #[test]
    fn test_checksum_new() {
        let c = Checksum::new(ChecksumAlgorithm::Sha256, "abc123", 4096);
        assert_eq!(c.algorithm, ChecksumAlgorithm::Sha256);
        assert_eq!(c.value, "abc123");
        assert_eq!(c.file_size, 4096);
    }

    #[test]
    fn test_sidecar_timecode_new() {
        let tc = SidecarTimecode::new("01:00:00:00", 86400, "24/1", false);
        assert_eq!(tc.start, "01:00:00:00");
        assert_eq!(tc.duration_frames, 86400);
        assert!(!tc.drop_frame);
    }

    #[test]
    fn test_processing_record_new() {
        let r = ProcessingRecord::new("proxy_generate", "OxiMedia");
        assert_eq!(r.operation, "proxy_generate");
        assert_eq!(r.tool, "OxiMedia");
    }

    #[test]
    fn test_processing_record_with_param() {
        let r = ProcessingRecord::new("encode", "ffmpeg")
            .with_param("codec", "h264")
            .with_param("bitrate", "2000000");
        assert_eq!(
            r.params.get("codec").expect("should succeed in test"),
            "h264"
        );
        assert_eq!(
            r.params.get("bitrate").expect("should succeed in test"),
            "2000000"
        );
    }

    #[test]
    fn test_sidecar_data_new() {
        let data = SidecarData::new(PathBuf::from("/media/clip.mov"));
        assert_eq!(data.version, 1);
        assert!(data.history.is_empty());
        assert!(data.proxies.is_empty());
        assert!(!data.integrity_verified);
    }

    #[test]
    fn test_sidecar_data_set_metadata() {
        let mut data = SidecarData::new(PathBuf::from("/media/clip.mov"));
        data.set_metadata("camera", "ARRI ALEXA");
        assert_eq!(data.get_metadata("camera"), Some("ARRI ALEXA"));
        assert_eq!(data.get_metadata("missing"), None);
    }

    #[test]
    fn test_sidecar_data_add_proxy() {
        let mut data = SidecarData::new(PathBuf::from("/media/clip.mov"));
        data.add_proxy("Quarter H.264", PathBuf::from("/proxy/clip.mp4"));
        assert_eq!(data.proxy_count(), 1);
        assert!(data.proxies.contains_key("Quarter H.264"));
    }

    #[test]
    fn test_sidecar_data_add_history() {
        let mut data = SidecarData::new(PathBuf::from("/media/clip.mov"));
        data.add_history(ProcessingRecord::new("ingest", "OxiMedia"));
        data.add_history(ProcessingRecord::new("proxy_generate", "OxiMedia"));
        assert_eq!(data.history.len(), 2);
        assert_eq!(data.history[1].operation, "proxy_generate");
    }

    #[test]
    fn test_sidecar_data_checksum() {
        let mut data = SidecarData::new(PathBuf::from("/media/clip.mov"));
        data.set_checksum(Checksum::new(ChecksumAlgorithm::Sha256, "deadbeef", 1024));
        assert!(data.checksum.is_some());
        assert!(!data.integrity_verified); // Reset on set
    }

    #[test]
    fn test_sidecar_path_for() {
        let media = Path::new("/media/project/clip001.mov");
        let sc_path = SideCar::path_for(media);
        assert_eq!(sc_path, PathBuf::from("/media/project/clip001.mov.oxsc"));
    }

    #[test]
    fn test_sidecar_path_for_no_extension() {
        let media = Path::new("/media/clip");
        let sc_path = SideCar::path_for(media);
        assert_eq!(sc_path, PathBuf::from("/media/clip.oxsc"));
    }

    #[test]
    fn test_sidecar_exists_false() {
        let media = Path::new("/nonexistent/clip.mov");
        assert!(!SideCar::exists(media));
    }

    #[test]
    fn test_sidecar_load_not_found() {
        let media = Path::new("/nonexistent/clip.mov");
        let result = SideCar::load(media);
        assert!(result.is_err());
        matches!(result, Err(crate::ProxyError::FileNotFound(_)));
    }

    #[test]
    fn test_sidecar_save_and_load() {
        let dir = std::env::temp_dir();
        let media = dir.join("test_sidecar_media.mov");
        // Clean up first
        let _ = SideCar::delete(&media);

        let mut data = SidecarData::new(media.clone());
        data.set_metadata("test", "value123");
        data.add_proxy("Quarter", PathBuf::from("/proxy/q.mp4"));
        data.add_history(ProcessingRecord::new("test_op", "OxiMedia v1.0"));

        SideCar::save(&media, &data).expect("should succeed in test");
        assert!(SideCar::exists(&media));

        let loaded = SideCar::load(&media).expect("should succeed in test");
        assert_eq!(loaded.get_metadata("test"), Some("value123"));
        assert_eq!(loaded.proxy_count(), 1);
        assert_eq!(loaded.history.len(), 1);

        SideCar::delete(&media).expect("should succeed in test");
        assert!(!SideCar::exists(&media));
    }

    #[test]
    fn test_sidecar_load_or_create_new() {
        let media = Path::new("/nonexistent/fresh.mov");
        let data = SideCar::load_or_create(media).expect("should succeed in test");
        assert_eq!(data.media_path, PathBuf::from("/nonexistent/fresh.mov"));
        assert!(data.history.is_empty());
    }

    #[test]
    fn test_mock_checksum_deterministic() {
        let data = b"hello world";
        let h1 = SideCar::mock_checksum(data);
        let h2 = SideCar::mock_checksum(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16); // 8-byte hex = 16 chars
    }

    #[test]
    fn test_mock_checksum_different_inputs() {
        let h1 = SideCar::mock_checksum(b"abc");
        let h2 = SideCar::mock_checksum(b"def");
        assert_ne!(h1, h2);
    }
}
