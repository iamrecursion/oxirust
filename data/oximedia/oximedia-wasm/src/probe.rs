//! Format probing for WASM.
//!
//! This module exposes the `probe_format` functionality to JavaScript,
//! allowing detection of media container formats from raw bytes.

use crate::container::ContainerFormat;
use crate::types::{JsStreamInfo, MediaInfo};
use wasm_bindgen::prelude::*;

use crate::utils::to_js_error;

/// Result of format probing.
///
/// Contains the detected format and confidence score.
///
/// # JavaScript Example
///
/// ```javascript
/// const data = new Uint8Array([0x1A, 0x45, 0xDF, 0xA3, ...]);
/// const result = oximedia.probe_format(data);
/// console.log('Format:', result.format());
/// console.log('Confidence:', result.confidence());
/// ```
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmProbeResult {
    format: ContainerFormat,
    confidence: f32,
}

impl WasmProbeResult {
    /// Internal constructor (not exposed to JavaScript).
    #[must_use]
    pub(crate) const fn new_internal(format: ContainerFormat, confidence: f32) -> Self {
        Self { format, confidence }
    }
}

#[wasm_bindgen]
impl WasmProbeResult {
    /// Returns the detected container format as a string.
    ///
    /// Possible values:
    /// - `"Matroska"` - Matroska/`WebM` container
    /// - `"Ogg"` - Ogg container
    /// - `"Flac"` - FLAC audio
    /// - `"Wav"` - WAV audio
    /// - `"Mp4"` - MP4/ISOBMFF container (AV1/VP9 only)
    #[must_use]
    pub fn format(&self) -> String {
        format!("{:?}", self.format)
    }

    /// Returns the confidence score from 0.0 to 1.0.
    ///
    /// Higher values indicate greater confidence in the detection.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Returns a human-readable description of the format.
    #[must_use]
    pub fn description(&self) -> String {
        match self.format {
            ContainerFormat::Matroska => "Matroska/WebM container (.mkv, .webm)".to_string(),
            ContainerFormat::Ogg => "Ogg container (.ogg, .opus, .oga)".to_string(),
            ContainerFormat::Flac => "FLAC audio (.flac)".to_string(),
            ContainerFormat::Wav => "WAV audio (.wav)".to_string(),
            ContainerFormat::Mp4 => "MP4/ISOBMFF container (.mp4)".to_string(),
        }
    }

    /// Returns true if the format is a video container.
    #[must_use]
    pub fn is_video_container(&self) -> bool {
        matches!(
            self.format,
            ContainerFormat::Matroska | ContainerFormat::Mp4
        )
    }

    /// Returns true if the format is an audio-only container.
    #[must_use]
    pub fn is_audio_only(&self) -> bool {
        matches!(self.format, ContainerFormat::Flac | ContainerFormat::Wav)
    }
}

/// Probe the container format from raw bytes.
///
/// Analyzes the first few bytes of media data to detect the container format.
/// Returns the detected format and a confidence score.
///
/// # Arguments
///
/// * `data` - At least the first 12 bytes of the file (more bytes improve detection)
///
/// # Errors
///
/// Throws a JavaScript exception if the format cannot be detected.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// // WebM/Matroska header
/// const data = new Uint8Array([
///     0x1A, 0x45, 0xDF, 0xA3,  // EBML magic
///     0x01, 0x00, 0x00, 0x00,
///     0x00, 0x00, 0x00, 0x1F,
/// ]);
///
/// try {
///     const result = oximedia.probe_format(data);
///     console.log('Detected format:', result.format());
///     console.log('Confidence:', (result.confidence() * 100).toFixed(1) + '%');
/// } catch (e) {
///     console.error('Failed to probe format:', e);
/// }
/// ```
#[wasm_bindgen]
pub fn probe_format(data: &[u8]) -> Result<WasmProbeResult, JsValue> {
    let result = crate::container::probe_format(data).map_err(to_js_error)?;

    Ok(WasmProbeResult {
        format: result.format,
        confidence: result.confidence,
    })
}

/// Probe media data and return comprehensive `MediaInfo` as a JavaScript value.
///
/// Unlike [`probe_format`] which returns a `WasmProbeResult` wrapper object,
/// this function returns a plain JavaScript object that can be inspected
/// directly without calling getter methods.
///
/// # Arguments
///
/// * `data` - At least the first 12 bytes of the file (more bytes improve detection)
///
/// # Returns
///
/// A serialized [`MediaInfo`] object with `format`, `confidence`, `streams`, etc.
///
/// # Errors
///
/// Throws a JavaScript exception if the format cannot be detected or serialization fails.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// const data = new Uint8Array([0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]);
/// try {
///     const info = oximedia.probe_media(data);
///     console.log('Format:', info.format);
///     console.log('Confidence:', (info.confidence * 100).toFixed(1) + '%');
///     console.log('Is video:', info.is_video_container);
///     for (const stream of info.streams) {
///         console.log(`Stream ${stream.index}: ${stream.codec} (${stream.media_type})`);
///     }
/// } catch (e) {
///     console.error('Probe failed:', e);
/// }
/// ```
#[wasm_bindgen]
pub fn probe_media(data: &[u8]) -> Result<JsValue, JsValue> {
    let result = crate::container::probe_format(data).map_err(to_js_error)?;

    let probe_result = WasmProbeResult {
        format: result.format,
        confidence: result.confidence,
    };

    // Build a demuxer to extract stream information
    let mut demuxer = crate::demuxer::WasmDemuxer::new(data);
    let streams: Vec<JsStreamInfo> = match demuxer.probe() {
        Ok(_) => demuxer.streams().iter().map(JsStreamInfo::from).collect(),
        Err(_) => Vec::new(),
    };

    let stream_count = streams.len();
    let info = MediaInfo {
        format: probe_result.format(),
        format_description: probe_result.description(),
        confidence: probe_result.confidence(),
        is_video_container: probe_result.is_video_container(),
        is_audio_only: probe_result.is_audio_only(),
        streams,
        stream_count,
    };

    serde_json::to_string(&info)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
        .and_then(|json| {
            js_sys::JSON::parse(&json)
                .map_err(|e| crate::utils::js_err(&format!("JSON parse error: {e:?}")))
        })
}

/// Result of hashing raw content bytes via [`probe_hash`].
///
/// Provides real, deterministic digests of the input bytes using
/// dependency-free algorithms, so the WASM bundle's dependency footprint
/// and size stay unchanged. These mirror the pure-Rust routines in
/// `oximedia-py`'s `media_hash` module (CRC-32 ISO 3309 + FNV-1a 64-bit),
/// giving the browser build parity with that capability -- unlike the
/// Python module (which is not yet registered with `#[pymodule]`), this
/// one is wired up to a real, callable entry point.
///
/// For cryptographic-strength whole-file hashing (SHA-256), see the CLI's
/// `oximedia probe --hash`, which delegates to `oximedia-archive-pro`'s
/// `ChecksumGenerator`. That crate pulls in `tokio`/`rayon`/`walkdir` for
/// file-system streaming and is not part of the WASM dependency graph, so
/// it is intentionally not used here.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmHashResult {
    crc32_hex: String,
    fnv1a64_hex: String,
    simple128_hex: String,
    byte_length: usize,
}

#[wasm_bindgen]
impl WasmHashResult {
    /// Returns the CRC-32 (ISO 3309 / ITU-T V.42) digest as 8 lowercase hex
    /// characters.
    #[must_use]
    pub fn crc32(&self) -> String {
        self.crc32_hex.clone()
    }

    /// Returns the FNV-1a 64-bit digest as 16 lowercase hex characters.
    #[must_use]
    pub fn fnv1a64(&self) -> String {
        self.fnv1a64_hex.clone()
    }

    /// Returns a 128-bit composite digest (FNV-1a of the bytes forward,
    /// concatenated with FNV-1a of the bytes reversed) as 32 lowercase hex
    /// characters -- a longer identifier with lower collision probability
    /// than either hash alone.
    #[must_use]
    pub fn simple128(&self) -> String {
        self.simple128_hex.clone()
    }

    /// Returns the number of input bytes that were hashed.
    #[must_use]
    pub fn byte_length(&self) -> usize {
        self.byte_length
    }
}

/// Computes a real, deterministic content hash/fingerprint of `data`.
///
/// This is the WASM-side counterpart of the CLI's `oximedia probe --hash`
/// flag and of `oximedia-py`'s `media_hash` module: given the same input
/// bytes, all three report the same CRC-32 and FNV-1a 64-bit digests (the
/// CLI additionally reports a SHA-256 digest via `oximedia-archive-pro`,
/// which is out of scope for the WASM build -- see the [`WasmHashResult`]
/// docs).
///
/// Unlike [`probe_format`], which only needs the first few header bytes,
/// this hashes every byte passed in -- callers wanting a whole-file digest
/// must pass the complete file contents, not just a header sniff.
///
/// # Arguments
///
/// * `data` - The complete byte buffer to hash (e.g. an entire file loaded
///   into a `Uint8Array`).
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// const response = await fetch('video.webm');
/// const data = new Uint8Array(await response.arrayBuffer());
/// const hash = oximedia.probe_hash(data);
/// console.log('CRC32:', hash.crc32());
/// console.log('FNV1a64:', hash.fnv1a64());
/// console.log('bytes hashed:', hash.byte_length());
/// ```
#[wasm_bindgen]
pub fn probe_hash(data: &[u8]) -> WasmHashResult {
    let fnv_forward = fnv1a_64(data);
    let fnv_reversed = fnv1a_64_reversed(data);
    WasmHashResult {
        crc32_hex: format!("{:08x}", crc32(data)),
        fnv1a64_hex: format!("{fnv_forward:016x}"),
        simple128_hex: format!("{fnv_forward:016x}{fnv_reversed:016x}"),
        byte_length: data.len(),
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust hash helpers (no external deps; algorithm-for-algorithm mirrors
// oximedia-py's `media_hash.rs` so the two builds agree on digests for the
// same input bytes).
// ---------------------------------------------------------------------------

/// Computes CRC-32 (ISO 3309 / ITU-T V.42) of `data`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Computes the FNV-1a 64-bit hash of `data`.
fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Computes the FNV-1a 64-bit hash of `data` read back-to-front (the
/// second half of [`probe_hash`]'s 128-bit composite digest).
fn fnv1a_64_reversed(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data.iter().rev() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_webm() {
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
        let result = probe_format(&data).expect("probe should succeed");
        assert_eq!(result.format(), "Matroska");
        assert!(result.confidence() > 0.9);
        assert!(result.is_video_container());
    }

    #[test]
    fn test_probe_ogg() {
        let data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        let result = probe_format(data).expect("probe should succeed");
        assert_eq!(result.format(), "Ogg");
    }

    #[test]
    fn test_probe_flac() {
        let data = b"fLaC\x00\x00\x00\x22";
        let result = probe_format(data).expect("probe should succeed");
        assert_eq!(result.format(), "Flac");
        assert!(result.is_audio_only());
    }

    #[test]
    fn test_probe_unknown() {
        // Test the underlying container probe without WASM JsValue conversion
        // which panics in native test environments.
        let data = [0xFF; 16];
        assert!(crate::container::probe_format(&data).is_err());
    }

    #[test]
    fn test_media_info_serialization() {
        // Validate that MediaInfo can be constructed and serialized to JSON.
        // The full probe_media() uses js_sys::JSON::parse which is not available
        // in native test environments, so we test the data layer directly.
        use crate::types::{JsStreamInfo, MediaInfo};

        let info = MediaInfo {
            format: "Matroska".to_string(),
            format_description: "Matroska/WebM container (.mkv, .webm)".to_string(),
            confidence: 0.95,
            is_video_container: true,
            is_audio_only: false,
            streams: vec![JsStreamInfo {
                index: 0,
                codec: "Vp9".to_string(),
                media_type: "Video".to_string(),
                duration_seconds: None,
                timebase_num: 1,
                timebase_den: 1000,
                width: Some(1920),
                height: Some(1080),
                sample_rate: None,
                channels: None,
            }],
            stream_count: 1,
        };

        let json = serde_json::to_string(&info).expect("serde_json::to_string should succeed");
        assert!(json.contains("\"format\":\"Matroska\""));
        assert!(json.contains("\"confidence\":0.95"));
        assert!(json.contains("\"is_video_container\":true"));
        assert!(json.contains("\"stream_count\":1"));
        assert!(json.contains("\"codec\":\"Vp9\""));
    }

    #[test]
    fn test_probe_media_unknown_returns_err() {
        // Verify that probe_media on unknown data returns an Err.
        // We test via the underlying container::probe_format to avoid JsValue in native.
        let data = [0xFF; 16];
        assert!(crate::container::probe_format(&data).is_err());
    }

    #[test]
    fn test_probe_hash_deterministic() {
        let data = b"hello wasm hashing";
        let h1 = probe_hash(data);
        let h2 = probe_hash(data);
        assert_eq!(h1.crc32(), h2.crc32());
        assert_eq!(h1.fnv1a64(), h2.fnv1a64());
        assert_eq!(h1.simple128(), h2.simple128());
        assert_eq!(h1.byte_length(), data.len());
    }

    #[test]
    fn test_probe_hash_differs_for_different_input() {
        let h1 = probe_hash(b"input one");
        let h2 = probe_hash(b"input two");
        assert_ne!(h1.crc32(), h2.crc32());
        assert_ne!(h1.fnv1a64(), h2.fnv1a64());
        assert_ne!(h1.simple128(), h2.simple128());
    }

    #[test]
    fn test_probe_hash_matches_known_crc32_check_value() {
        // CRC-32 (ISO 3309 / "CRC-32/ISO-HDLC", the zlib/PKZIP variant used
        // here: poly 0xEDB88320 reflected, init 0xFFFFFFFF, final XOR) of
        // the ASCII string "123456789" is the well-known check value
        // 0xCBF43926, used across implementations to validate correctness.
        let h = probe_hash(b"123456789");
        assert_eq!(h.crc32(), "cbf43926");
    }

    #[test]
    fn test_probe_hash_empty_input() {
        let h = probe_hash(b"");
        assert_eq!(h.byte_length(), 0);
        assert_eq!(h.crc32().len(), 8);
        assert_eq!(h.fnv1a64().len(), 16);
        assert_eq!(h.simple128().len(), 32);
    }

    #[test]
    fn test_probe_hash_simple128_is_concat_of_forward_and_reversed_fnv() {
        let data = b"asymmetric payload";
        let h = probe_hash(data);
        let mut reversed = data.to_vec();
        reversed.reverse();
        let expected = format!(
            "{}{}",
            probe_hash(data).fnv1a64(),
            probe_hash(&reversed).fnv1a64()
        );
        assert_eq!(h.simple128(), expected);
    }
}
