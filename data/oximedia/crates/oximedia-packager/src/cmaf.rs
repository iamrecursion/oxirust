// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! CMAF (Common Media Application Format) packaging support.
//!
//! Provides configuration, track descriptors, manifest generation, and
//! segment structures for CMAF-compliant adaptive streaming workflows.

// ---------------------------------------------------------------------------
// CmafConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for a CMAF packaging session.
#[derive(Debug, Clone)]
pub struct CmafConfig {
    /// Duration of each fragment (chunk) in milliseconds.
    pub fragment_duration_ms: u32,
    /// Duration of each addressable segment in milliseconds.
    pub segment_duration_ms: u32,
    /// Whether to emit segments using chunked transfer encoding.
    pub use_chunked_transfer: bool,
    /// Enable low-latency CMAF (LL-CMAF) chunked delivery.
    pub low_latency: bool,
}

impl Default for CmafConfig {
    fn default() -> Self {
        Self {
            fragment_duration_ms: 1_000,
            segment_duration_ms: 6_000,
            use_chunked_transfer: false,
            low_latency: false,
        }
    }
}

// ---------------------------------------------------------------------------
// CmafTrack
// ---------------------------------------------------------------------------

/// Describes a single media track within a CMAF presentation.
#[derive(Debug, Clone)]
pub struct CmafTrack {
    /// Unique track identifier within the presentation.
    pub track_id: u32,
    /// Codec string (e.g. `"av01"`, `"vp09"`, `"opus"`, `"flac"`).
    pub codec: String,
    /// Encoded bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// `true` for video tracks, `false` for audio tracks.
    pub is_video: bool,
}

impl CmafTrack {
    /// Create a new CMAF track descriptor.
    #[must_use]
    pub fn new(track_id: u32, codec: impl Into<String>, bitrate_kbps: u32, is_video: bool) -> Self {
        Self {
            track_id,
            codec: codec.into(),
            bitrate_kbps,
            is_video,
        }
    }

    /// Returns `"video"` or `"audio"` depending on the track type.
    #[must_use]
    pub fn media_type(&self) -> &'static str {
        if self.is_video {
            "video"
        } else {
            "audio"
        }
    }
}

// ---------------------------------------------------------------------------
// CmafManifest
// ---------------------------------------------------------------------------

/// High-level descriptor for a CMAF presentation (used to drive manifest
/// generation).
#[derive(Debug, Clone)]
pub struct CmafManifest {
    /// All tracks (video and audio) in this presentation.
    pub tracks: Vec<CmafTrack>,
    /// Base URL prepended to all segment URLs in the manifest.
    pub base_url: String,
    /// Total presentation duration in milliseconds, or `None` for live.
    pub presentation_duration_ms: Option<u64>,
}

impl CmafManifest {
    /// Create a new CMAF manifest descriptor.
    #[must_use]
    pub fn new(tracks: Vec<CmafTrack>, base_url: impl Into<String>) -> Self {
        Self {
            tracks,
            base_url: base_url.into(),
            presentation_duration_ms: None,
        }
    }

    /// Set the total presentation duration (makes this a VOD manifest).
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.presentation_duration_ms = Some(duration_ms);
        self
    }
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Generate a CMAF manifest string for the given tracks and configuration.
///
/// The output is a simple text-based manifest similar to an HLS playlist but
/// describing CMAF tracks. For production use this would be an MPD or M3U8;
/// here we emit a lightweight custom format suitable for unit testing and
/// prototyping.
#[must_use]
pub fn generate_cmaf_manifest(tracks: &[CmafTrack], config: &CmafConfig, base_url: &str) -> String {
    let mut out = String::new();

    out.push_str("#CMAF-MANIFEST\n");
    out.push_str(&format!(
        "#FRAGMENT-DURATION:{}\n",
        config.fragment_duration_ms
    ));
    out.push_str(&format!(
        "#SEGMENT-DURATION:{}\n",
        config.segment_duration_ms
    ));
    if config.low_latency {
        out.push_str("#LOW-LATENCY:YES\n");
    }
    if config.use_chunked_transfer {
        out.push_str("#CHUNKED-TRANSFER:YES\n");
    }
    out.push_str(&format!("#BASE-URL:{base_url}\n"));
    out.push('\n');

    for track in tracks {
        out.push_str(&format!(
            "TRACK id={} type={} codec={} bitrate={}kbps\n",
            track.track_id,
            track.media_type(),
            track.codec,
            track.bitrate_kbps,
        ));
        out.push_str(&format!(
            "  INIT {base_url}/track{}/init.cmf{}\n",
            track.track_id,
            if track.is_video { "v" } else { "a" },
        ));
        out.push_str(&format!(
            "  SEGMENTS {base_url}/track{}/seg-$Number$.cmf{}\n",
            track.track_id,
            if track.is_video { "v" } else { "a" },
        ));
    }

    out
}

/// Build the segment URL for a given track and 0-based segment number.
#[must_use]
pub fn cmaf_track_url(track: &CmafTrack, segment_num: u32) -> String {
    let ext = if track.is_video { "cmfv" } else { "cmfa" };
    format!("track{}/seg-{:05}.{}", track.track_id, segment_num, ext)
}

// ---------------------------------------------------------------------------
// CmafSegment
// ---------------------------------------------------------------------------

/// A single CMAF segment (fragment) for a particular track.
#[derive(Debug, Clone)]
pub struct CmafSegment {
    /// Track this segment belongs to.
    pub track_id: u32,
    /// 0-based sequence number within the track.
    pub sequence: u32,
    /// Duration of this segment in milliseconds.
    pub duration_ms: u32,
    /// Raw payload bytes (moof + mdat boxes in a real CMAF stream).
    pub data: Vec<u8>,
}

impl CmafSegment {
    /// Create a new, empty segment.
    #[must_use]
    pub fn new(track_id: u32, seq: u32, duration_ms: u32) -> Self {
        Self {
            track_id,
            sequence: seq,
            duration_ms,
            data: Vec::new(),
        }
    }

    /// Return the size of the payload in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Append raw bytes to the segment payload.
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmaf_config_default() {
        let c = CmafConfig::default();
        assert_eq!(c.fragment_duration_ms, 1_000);
        assert_eq!(c.segment_duration_ms, 6_000);
        assert!(!c.low_latency);
        assert!(!c.use_chunked_transfer);
    }

    #[test]
    fn test_cmaf_track_media_type_video() {
        let t = CmafTrack::new(1, "av01", 3_000, true);
        assert_eq!(t.media_type(), "video");
    }

    #[test]
    fn test_cmaf_track_media_type_audio() {
        let t = CmafTrack::new(2, "opus", 128, false);
        assert_eq!(t.media_type(), "audio");
    }

    #[test]
    fn test_cmaf_track_fields() {
        let t = CmafTrack::new(5, "vp09", 1_500, true);
        assert_eq!(t.track_id, 5);
        assert_eq!(t.codec, "vp09");
        assert_eq!(t.bitrate_kbps, 1_500);
        assert!(t.is_video);
    }

    #[test]
    fn test_generate_cmaf_manifest_contains_base_url() {
        let tracks = vec![CmafTrack::new(1, "av01", 2_000, true)];
        let config = CmafConfig::default();
        let manifest = generate_cmaf_manifest(&tracks, &config, "https://cdn.example.com");
        assert!(manifest.contains("https://cdn.example.com"));
    }

    #[test]
    fn test_generate_cmaf_manifest_contains_track_info() {
        let tracks = vec![
            CmafTrack::new(1, "av01", 2_000, true),
            CmafTrack::new(2, "opus", 128, false),
        ];
        let config = CmafConfig::default();
        let manifest = generate_cmaf_manifest(&tracks, &config, "https://cdn.example.com");
        assert!(manifest.contains("type=video"));
        assert!(manifest.contains("type=audio"));
        assert!(manifest.contains("codec=av01"));
        assert!(manifest.contains("codec=opus"));
    }

    #[test]
    fn test_generate_cmaf_manifest_low_latency_flag() {
        let tracks = vec![];
        let config = CmafConfig {
            low_latency: true,
            ..CmafConfig::default()
        };
        let manifest = generate_cmaf_manifest(&tracks, &config, "http://localhost");
        assert!(manifest.contains("#LOW-LATENCY:YES"));
    }

    #[test]
    fn test_generate_cmaf_manifest_chunked_transfer_flag() {
        let config = CmafConfig {
            use_chunked_transfer: true,
            ..CmafConfig::default()
        };
        let manifest = generate_cmaf_manifest(&[], &config, "http://localhost");
        assert!(manifest.contains("#CHUNKED-TRANSFER:YES"));
    }

    #[test]
    fn test_cmaf_track_url_video() {
        let track = CmafTrack::new(3, "av01", 4_000, true);
        let url = cmaf_track_url(&track, 0);
        assert!(url.contains("track3"));
        assert!(url.ends_with("cmfv"));
    }

    #[test]
    fn test_cmaf_track_url_audio() {
        let track = CmafTrack::new(4, "opus", 64, false);
        let url = cmaf_track_url(&track, 7);
        assert!(url.contains("track4"));
        assert!(url.ends_with("cmfa"));
        assert!(url.contains("00007"));
    }

    #[test]
    fn test_cmaf_segment_new() {
        let seg = CmafSegment::new(1, 0, 6_000);
        assert_eq!(seg.track_id, 1);
        assert_eq!(seg.sequence, 0);
        assert_eq!(seg.duration_ms, 6_000);
        assert_eq!(seg.size_bytes(), 0);
    }

    #[test]
    fn test_cmaf_segment_push_bytes() {
        let mut seg = CmafSegment::new(1, 0, 1_000);
        seg.push_bytes(b"hello");
        seg.push_bytes(b" world");
        assert_eq!(seg.size_bytes(), 11);
        assert_eq!(&seg.data, b"hello world");
    }

    #[test]
    fn test_cmaf_manifest_with_duration() {
        let m = CmafManifest::new(vec![], "http://example.com").with_duration_ms(120_000);
        assert_eq!(m.presentation_duration_ms, Some(120_000));
    }
}
