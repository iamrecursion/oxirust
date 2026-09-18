//! Pre-roll verification for playlist items before air.
//!
//! `PreRollVerifier` performs a set of pre-air checks on a `PlaylistItem`:
//!
//! 1. **Media existence** — the file path is present in the filesystem.
//! 2. **Duration validity** — the metadata duration (in seconds) is within
//!    broadcast-safe bounds (`> 0` and `<= MAX_DURATION_SECS`).
//! 3. **Codec validity** — the file extension implies a broadcast-safe codec
//!    (MXF, MOV, MP4, TS, MKV, or WAV/AIFF for audio-only items).
//! 4. **Estimated ready time** — how long the system is predicted to need
//!    before the item can be considered pre-rolled and ready to air.
//!
//! The verifier deliberately avoids heavy demuxing in the hot path; instead
//! it performs lightweight file-system and metadata-cache checks so that
//! playout engines can call it on every item transition without stalling.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::debug;

/// Maximum duration (seconds) an individual playlist item is allowed to have.
/// Items longer than this are flagged as `duration_valid = false`.
const MAX_DURATION_SECS: f64 = 86_400.0; // 24 hours

/// Minimum duration (seconds) required for a playlist item to be considered
/// broadcast-safe.
const MIN_DURATION_SECS: f64 = 0.1; // 100 ms

/// File extensions recognised as broadcast-safe video/audio containers.
#[allow(dead_code)]
const VALID_EXTENSIONS: &[&str] = &[
    "mxf", "mov", "mp4", "m4v", "ts", "mts", "m2ts", "mkv", "avi", "wav", "aiff", "aif", "flac",
    "mka", "y4m",
];

/// Codec validity tags derived from file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecHint {
    /// Container is a professional broadcast format (MXF, GXF).
    ProfessionalBroadcast,
    /// Container is a common delivery format (MOV, MP4).
    CommonDelivery,
    /// Container is a transport-stream format (TS, M2TS).
    TransportStream,
    /// Container is an open-format (MKV, WebM).
    OpenContainer,
    /// Audio-only container (WAV, AIFF, FLAC).
    AudioOnly,
    /// Extension was not recognised as a broadcast-safe container.
    Unknown,
}

impl CodecHint {
    fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "mxf" | "gxf" => Self::ProfessionalBroadcast,
            "mov" | "mp4" | "m4v" => Self::CommonDelivery,
            "ts" | "mts" | "m2ts" => Self::TransportStream,
            "mkv" | "mka" | "webm" => Self::OpenContainer,
            "wav" | "aiff" | "aif" | "flac" | "y4m" => Self::AudioOnly,
            _ => Self::Unknown,
        }
    }

    /// Returns `true` if the codec hint corresponds to a known broadcast-safe
    /// container.
    pub fn is_valid_broadcast_codec(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// A playlist item descriptor supplied to `PreRollVerifier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistItem {
    /// Unique identifier for this item in the playlist.
    pub id: String,
    /// Absolute filesystem path to the media file.
    pub file_path: PathBuf,
    /// Declared duration in seconds (from the scheduling system / NLE export).
    /// If `None` the verifier will attempt to infer it from filesystem metadata
    /// (file size heuristic) or return `duration_valid = false`.
    pub declared_duration_secs: Option<f64>,
    /// Optional title for logging.
    pub title: Option<String>,
}

impl PlaylistItem {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        file_path: impl Into<PathBuf>,
        declared_duration_secs: Option<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            file_path: file_path.into(),
            declared_duration_secs,
            title: None,
        }
    }

    /// Attach a title for display / logging purposes.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Result of a pre-roll verification pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRollResult {
    /// `true` if the media file exists and is readable on the filesystem.
    pub media_exists: bool,
    /// `true` if the declared (or inferred) duration is within broadcast-safe
    /// bounds (`MIN_DURATION_SECS`..=`MAX_DURATION_SECS`).
    pub duration_valid: bool,
    /// `true` if the file extension maps to a known broadcast-safe container.
    pub codec_valid: bool,
    /// The `CodecHint` derived from the file extension.
    pub codec_hint: CodecHint,
    /// Estimated time the system needs before the item can be considered
    /// pre-rolled.  Currently a heuristic based on file size.
    pub estimated_ready_time: Duration,
    /// The actual duration used for the `duration_valid` check.  May be
    /// `None` if neither declared nor inferred duration was available.
    pub effective_duration_secs: Option<f64>,
    /// Human-readable failure reason (empty string if all checks pass).
    pub failure_reason: String,
}

impl PreRollResult {
    /// `true` if all three checks passed.
    pub fn is_ready(&self) -> bool {
        self.media_exists && self.duration_valid && self.codec_valid
    }
}

/// Pre-roll verifier for broadcast playout items.
///
/// # Design
///
/// The verifier is intentionally **synchronous** and **allocation-light**: it
/// calls `std::fs::metadata` once per item and performs no I/O beyond that.
/// Callers that need async integration can wrap calls with
/// `tokio::task::spawn_blocking`.
pub struct PreRollVerifier {
    /// Expected minimum pre-roll duration (used in estimated_ready_time
    /// calculation).
    preroll_duration: Duration,
    /// Network / storage read speed estimate in bytes per second (used to
    /// compute `estimated_ready_time` from file size).
    estimated_read_bps: u64,
}

impl PreRollVerifier {
    /// Create a verifier with the given pre-roll duration assumption and an
    /// estimated storage read speed.
    ///
    /// `preroll_duration` — minimum lead time required before the item airs.
    /// `estimated_read_bps` — expected storage throughput in bytes/s (default
    ///   is 100 MB/s for professional SAN storage).
    pub fn new(preroll_duration: Duration, estimated_read_bps: u64) -> Self {
        Self {
            preroll_duration,
            estimated_read_bps,
        }
    }

    /// Verify a single playlist item.
    ///
    /// Performs three checks in sequence:
    /// 1. File existence via `std::fs::metadata`.
    /// 2. Duration validity against `declared_duration_secs`.
    /// 3. Codec validity from file extension.
    pub fn verify(&self, item: &PlaylistItem) -> PreRollResult {
        let path: &Path = &item.file_path;
        debug!("Pre-roll verify: {:?}", path);

        // ── 1. Media existence ──────────────────────────────────────────────
        let metadata_result = std::fs::metadata(path);
        let (media_exists, file_size_bytes) = match metadata_result {
            Ok(meta) if meta.is_file() => (true, meta.len()),
            _ => (false, 0u64),
        };

        // ── 2. Duration validity ────────────────────────────────────────────
        let effective_duration_secs = if let Some(d) = item.declared_duration_secs {
            Some(d)
        } else if media_exists && file_size_bytes > 0 {
            // Very rough heuristic: assume ~10 Mbit/s video → 1.25 MB/s → divide
            // by 1_250_000 to get approximate seconds.
            Some(file_size_bytes as f64 / 1_250_000.0)
        } else {
            None
        };

        let duration_valid = match effective_duration_secs {
            Some(d) => d >= MIN_DURATION_SECS && d <= MAX_DURATION_SECS,
            None => false,
        };

        // ── 3. Codec validity ───────────────────────────────────────────────
        let codec_hint = path
            .extension()
            .and_then(|e| e.to_str())
            .map(CodecHint::from_extension)
            .unwrap_or(CodecHint::Unknown);
        let codec_valid = codec_hint.is_valid_broadcast_codec();

        // ── Estimated ready time ────────────────────────────────────────────
        // = pre-roll lead time + file transfer time
        let transfer_secs = if self.estimated_read_bps > 0 && file_size_bytes > 0 {
            Duration::from_secs_f64(file_size_bytes as f64 / self.estimated_read_bps as f64)
        } else {
            Duration::ZERO
        };
        let estimated_ready_time = self.preroll_duration + transfer_secs;

        // ── Failure reason ─────────────────────────────────────────────────
        let mut reasons = Vec::new();
        if !media_exists {
            reasons.push(format!("media file not found: {}", path.display()));
        }
        if !duration_valid {
            reasons.push(format!(
                "duration invalid (got {:?})",
                effective_duration_secs
            ));
        }
        if !codec_valid {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("<none>");
            reasons.push(format!("unrecognised codec/container extension: .{ext}"));
        }
        let failure_reason = reasons.join("; ");

        PreRollResult {
            media_exists,
            duration_valid,
            codec_valid,
            codec_hint,
            estimated_ready_time,
            effective_duration_secs,
            failure_reason,
        }
    }
}

impl Default for PreRollVerifier {
    fn default() -> Self {
        Self::new(
            Duration::from_secs(5), // 5-second pre-roll lead time
            100 * 1024 * 1024,      // 100 MB/s
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_temp_mxf(content: &[u8]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("test_item.mxf");
        let mut f = std::fs::File::create(&path).expect("file create should succeed");
        f.write_all(content).expect("write should succeed");
        (dir, path)
    }

    #[test]
    fn test_verify_existing_mxf_with_declared_duration() {
        let (_dir, path) = make_temp_mxf(&[0u8; 1024]);
        let item = PlaylistItem::new("item1", &path, Some(30.0));
        let verifier = PreRollVerifier::default();
        let result = verifier.verify(&item);
        assert!(result.media_exists);
        assert!(result.duration_valid);
        assert!(result.codec_valid);
        assert!(result.is_ready());
        assert_eq!(result.codec_hint, CodecHint::ProfessionalBroadcast);
    }

    #[test]
    fn test_verify_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("nonexistent.mxf");
        let item = PlaylistItem::new("item2", &path, Some(30.0));
        let verifier = PreRollVerifier::default();
        let result = verifier.verify(&item);
        assert!(!result.media_exists);
        assert!(!result.is_ready());
        assert!(result.failure_reason.contains("not found"));
    }

    #[test]
    fn test_verify_unknown_extension() {
        let (_dir, path_mxf) = make_temp_mxf(&[0u8; 512]);
        // rename to .xyz
        let xyz_path = path_mxf.with_extension("xyz");
        std::fs::rename(&path_mxf, &xyz_path).expect("rename should succeed");
        let item = PlaylistItem::new("item3", &xyz_path, Some(10.0));
        let verifier = PreRollVerifier::default();
        let result = verifier.verify(&item);
        assert!(result.media_exists);
        assert!(!result.codec_valid);
        assert_eq!(result.codec_hint, CodecHint::Unknown);
        assert!(!result.is_ready());
    }

    #[test]
    fn test_verify_zero_duration_invalid() {
        let (_dir, path) = make_temp_mxf(&[0u8; 100]);
        let item = PlaylistItem::new("item4", &path, Some(0.0));
        let verifier = PreRollVerifier::default();
        let result = verifier.verify(&item);
        assert!(!result.duration_valid);
    }

    #[test]
    fn test_verify_excessive_duration_invalid() {
        let (_dir, path) = make_temp_mxf(&[0u8; 100]);
        let item = PlaylistItem::new("item5", &path, Some(90_000.0));
        let verifier = PreRollVerifier::default();
        let result = verifier.verify(&item);
        assert!(!result.duration_valid);
    }

    #[test]
    fn test_verify_mp4_is_valid_codec() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("clip.mp4");
        std::fs::write(&path, &[0u8; 256]).expect("write should succeed");
        let item = PlaylistItem::new("item6", &path, Some(5.0));
        let result = PreRollVerifier::default().verify(&item);
        assert!(result.codec_valid);
        assert_eq!(result.codec_hint, CodecHint::CommonDelivery);
    }

    #[test]
    fn test_verify_ts_is_transport_stream() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("stream.ts");
        std::fs::write(&path, &[0u8; 512]).expect("write should succeed");
        let item = PlaylistItem::new("item7", &path, Some(60.0));
        let result = PreRollVerifier::default().verify(&item);
        assert_eq!(result.codec_hint, CodecHint::TransportStream);
        assert!(result.is_ready());
    }

    #[test]
    fn test_estimated_ready_time_includes_preroll() {
        let (_dir, path) = make_temp_mxf(&[0u8; 512]);
        let item = PlaylistItem::new("item8", &path, Some(30.0));
        let verifier = PreRollVerifier::new(Duration::from_secs(10), 100 * 1024 * 1024);
        let result = verifier.verify(&item);
        assert!(result.estimated_ready_time >= Duration::from_secs(10));
    }

    #[test]
    fn test_verify_wav_audio_only() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("music.wav");
        std::fs::write(&path, &[0u8; 1024]).expect("write should succeed");
        let item = PlaylistItem::new("item9", &path, Some(120.0));
        let result = PreRollVerifier::default().verify(&item);
        assert_eq!(result.codec_hint, CodecHint::AudioOnly);
        assert!(result.codec_valid);
        assert!(result.is_ready());
    }
}
