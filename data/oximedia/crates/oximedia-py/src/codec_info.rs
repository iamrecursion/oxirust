//! Python-bindable codec and stream information types.
//!
//! Provides lightweight, serialisable data structures that represent codec
//! metadata for streams and container probing results, without any PyO3
//! `#[pyclass]` annotations so the module compiles regardless of the Python
//! ABI being present.

#![allow(dead_code)]

/// Codec metadata for a single stream.
#[derive(Clone, Debug)]
pub struct PyCodecInfo {
    /// Codec short name (e.g. `"h264"`, `"aac"`, `"av1"`).
    pub name: String,
    /// Broad codec type: `"video"`, `"audio"`, or `"data"`.
    pub codec_type: String,
    /// Optional codec profile (e.g. `"High"`, `"Main"`).
    pub profile: Option<String>,
    /// Optional codec level string (e.g. `"4.1"`).
    pub level: Option<String>,
    /// Nominal bitrate in kbit/s (0 = unknown).
    pub bitrate_kbps: u32,
}

impl PyCodecInfo {
    /// Returns `true` if this is a video codec.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.codec_type == "video"
    }

    /// Returns `true` if this is an audio codec.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.codec_type == "audio"
    }

    /// Returns a concise human-readable description of the codec.
    ///
    /// Format: `"<name> [<profile>] [L<level>] <bitrate_kbps> kbps"`.
    #[must_use]
    pub fn short_description(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if let Some(ref p) = self.profile {
            parts.push(p.clone());
        }
        if let Some(ref l) = self.level {
            parts.push(format!("L{l}"));
        }
        if self.bitrate_kbps > 0 {
            parts.push(format!("{} kbps", self.bitrate_kbps));
        }
        parts.join(" ")
    }
}

/// A single stream within a media container.
#[derive(Clone, Debug)]
pub struct PyStreamInfo {
    /// Zero-based stream index inside the container.
    pub index: u32,
    /// Codec information for this stream.
    pub codec: PyCodecInfo,
    /// Stream duration in milliseconds.
    pub duration_ms: u64,
    /// Optional ISO 639-2/T language tag (e.g. `"eng"`, `"jpn"`).
    pub language: Option<String>,
}

impl PyStreamInfo {
    /// Returns `true` if the stream has a language tag set.
    #[must_use]
    pub fn has_language(&self) -> bool {
        self.language.is_some()
    }
}

/// The result of probing a media file.
#[derive(Clone, Debug)]
pub struct PyProbeResult {
    /// All streams found in the container.
    pub streams: Vec<PyStreamInfo>,
    /// Container format name (e.g. `"matroska"`, `"mp4"`).
    pub format_name: String,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
}

impl PyProbeResult {
    /// Returns references to all video streams.
    #[must_use]
    pub fn video_streams(&self) -> Vec<&PyStreamInfo> {
        self.streams.iter().filter(|s| s.codec.is_video()).collect()
    }

    /// Returns references to all audio streams.
    #[must_use]
    pub fn audio_streams(&self) -> Vec<&PyStreamInfo> {
        self.streams.iter().filter(|s| s.codec.is_audio()).collect()
    }

    /// Returns the total number of streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ─── helpers for tests ───────────────────────────────────────────────────────

#[cfg(test)]
fn video_codec(name: &str) -> PyCodecInfo {
    PyCodecInfo {
        name: name.to_string(),
        codec_type: "video".to_string(),
        profile: None,
        level: None,
        bitrate_kbps: 0,
    }
}

#[cfg(test)]
fn audio_codec(name: &str) -> PyCodecInfo {
    PyCodecInfo {
        name: name.to_string(),
        codec_type: "audio".to_string(),
        profile: None,
        level: None,
        bitrate_kbps: 0,
    }
}

#[cfg(test)]
fn make_stream(index: u32, codec: PyCodecInfo) -> PyStreamInfo {
    PyStreamInfo {
        index,
        codec,
        duration_ms: 60_000,
        language: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PyCodecInfo ────────────────────────────────────────────────────────

    #[test]
    fn test_is_video_true() {
        assert!(video_codec("h264").is_video());
    }

    #[test]
    fn test_is_audio_true() {
        assert!(audio_codec("aac").is_audio());
    }

    #[test]
    fn test_is_video_false_for_audio() {
        assert!(!audio_codec("opus").is_video());
    }

    #[test]
    fn test_is_audio_false_for_video() {
        assert!(!video_codec("av1").is_audio());
    }

    #[test]
    fn test_short_description_name_only() {
        let c = video_codec("av1");
        assert_eq!(c.short_description(), "av1");
    }

    #[test]
    fn test_short_description_with_profile_and_level() {
        let c = PyCodecInfo {
            name: "h264".to_string(),
            codec_type: "video".to_string(),
            profile: Some("High".to_string()),
            level: Some("4.1".to_string()),
            bitrate_kbps: 4_000,
        };
        let desc = c.short_description();
        assert!(desc.contains("h264"));
        assert!(desc.contains("High"));
        assert!(desc.contains("L4.1"));
        assert!(desc.contains("4000 kbps"));
    }

    #[test]
    fn test_short_description_zero_bitrate_excluded() {
        let c = video_codec("vp9");
        assert!(!c.short_description().contains("kbps"));
    }

    // ── PyStreamInfo ────────────────────────────────────────────────────────

    #[test]
    fn test_has_language_true() {
        let mut s = make_stream(0, audio_codec("aac"));
        s.language = Some("eng".to_string());
        assert!(s.has_language());
    }

    #[test]
    fn test_has_language_false() {
        let s = make_stream(0, audio_codec("aac"));
        assert!(!s.has_language());
    }

    #[test]
    fn test_stream_index_stored() {
        let s = make_stream(3, video_codec("av1"));
        assert_eq!(s.index, 3);
    }

    // ── PyProbeResult ───────────────────────────────────────────────────────

    #[test]
    fn test_stream_count_empty() {
        let pr = PyProbeResult {
            streams: vec![],
            format_name: "matroska".to_string(),
            duration_ms: 0,
        };
        assert_eq!(pr.stream_count(), 0);
    }

    #[test]
    fn test_stream_count_multiple() {
        let pr = PyProbeResult {
            streams: vec![
                make_stream(0, video_codec("av1")),
                make_stream(1, audio_codec("opus")),
                make_stream(2, audio_codec("opus")),
            ],
            format_name: "matroska".to_string(),
            duration_ms: 60_000,
        };
        assert_eq!(pr.stream_count(), 3);
    }

    #[test]
    fn test_video_streams_filtered() {
        let pr = PyProbeResult {
            streams: vec![
                make_stream(0, video_codec("av1")),
                make_stream(1, audio_codec("opus")),
            ],
            format_name: "matroska".to_string(),
            duration_ms: 60_000,
        };
        assert_eq!(pr.video_streams().len(), 1);
        assert_eq!(pr.audio_streams().len(), 1);
    }

    #[test]
    fn test_audio_streams_multiple() {
        let pr = PyProbeResult {
            streams: vec![
                make_stream(0, video_codec("vp9")),
                make_stream(1, audio_codec("vorbis")),
                make_stream(2, audio_codec("opus")),
            ],
            format_name: "webm".to_string(),
            duration_ms: 30_000,
        };
        assert_eq!(pr.audio_streams().len(), 2);
    }

    #[test]
    fn test_format_name_stored() {
        let pr = PyProbeResult {
            streams: vec![],
            format_name: "mp4".to_string(),
            duration_ms: 0,
        };
        assert_eq!(pr.format_name, "mp4");
    }
}
