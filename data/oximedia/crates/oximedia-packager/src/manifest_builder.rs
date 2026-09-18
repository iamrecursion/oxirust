#![allow(dead_code)]
//! Manifest building for HLS and DASH adaptive streams.
//!
//! Provides a builder that accumulates track descriptors and can emit
//! HLS master playlist or DASH MPD text.

use std::fmt::Write as FmtWrite;

/// The manifest/playlist format to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifestFormat {
    /// HTTP Live Streaming master playlist (M3U8).
    Hls,
    /// MPEG-DASH Media Presentation Description (MPD XML).
    Dash,
    /// Microsoft Smooth Streaming manifest (XML).
    Smooth,
}

impl ManifestFormat {
    /// Returns the MIME type for the manifest document itself.
    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Hls => "application/vnd.apple.mpegurl",
            Self::Dash => "application/dash+xml",
            Self::Smooth => "application/vnd.ms-sstr+xml",
        }
    }

    /// Returns the conventional file extension (without leading dot).
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Hls => "m3u8",
            Self::Dash => "mpd",
            Self::Smooth => "ism",
        }
    }
}

impl std::fmt::Display for ManifestFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hls => write!(f, "HLS"),
            Self::Dash => write!(f, "DASH"),
            Self::Smooth => write!(f, "Smooth"),
        }
    }
}

/// Describes a single rendition (video or audio track) to include in the manifest.
#[derive(Debug, Clone)]
pub struct ManifestTrack {
    /// URI of the media playlist or segment.
    uri: String,
    /// Bandwidth in bits per second.
    bandwidth_bps: u64,
    /// Video width in pixels (0 for audio-only).
    width: u32,
    /// Video height in pixels (0 for audio-only).
    height: u32,
    /// Codec string (e.g. "av01.0.05M.08").
    codecs: String,
    /// Frame rate (0.0 for audio-only).
    frame_rate: f64,
    /// Whether this track carries audio.
    is_audio: bool,
}

impl ManifestTrack {
    /// Create a new video track descriptor.
    pub fn video(
        uri: impl Into<String>,
        bandwidth_bps: u64,
        width: u32,
        height: u32,
        codecs: impl Into<String>,
        frame_rate: f64,
    ) -> Self {
        Self {
            uri: uri.into(),
            bandwidth_bps,
            width,
            height,
            codecs: codecs.into(),
            frame_rate,
            is_audio: false,
        }
    }

    /// Create a new audio-only track descriptor.
    pub fn audio(uri: impl Into<String>, bandwidth_bps: u64, codecs: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            bandwidth_bps,
            width: 0,
            height: 0,
            codecs: codecs.into(),
            frame_rate: 0.0,
            is_audio: true,
        }
    }

    /// Returns bandwidth in kbps.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bandwidth_kbps(&self) -> f64 {
        self.bandwidth_bps as f64 / 1000.0
    }

    /// Returns the track URI.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Returns the codec string.
    #[must_use]
    pub fn codecs(&self) -> &str {
        &self.codecs
    }

    /// Returns `true` if this is an audio-only track.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.is_audio
    }
}

/// Builds streaming manifests from a collection of tracks.
pub struct ManifestBuilder {
    tracks: Vec<ManifestTrack>,
    target_duration_s: u32,
    media_sequence: u64,
    base_url: String,
}

impl ManifestBuilder {
    /// Create a new manifest builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            target_duration_s: 6,
            media_sequence: 0,
            base_url: String::new(),
        }
    }

    /// Set the HLS target segment duration in seconds.
    #[must_use]
    pub fn with_target_duration(mut self, seconds: u32) -> Self {
        self.target_duration_s = seconds;
        self
    }

    /// Set the media sequence number for live playlists.
    #[must_use]
    pub fn with_media_sequence(mut self, seq: u64) -> Self {
        self.media_sequence = seq;
        self
    }

    /// Set a base URL to prepend to relative URIs.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Add a track to the manifest.
    pub fn add_track(&mut self, track: ManifestTrack) {
        self.tracks.push(track);
    }

    /// Returns the number of tracks registered.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Generate an HLS master playlist (M3U8) string.
    #[must_use]
    pub fn build_hls(&self) -> String {
        let mut out = String::from("#EXTM3U\n#EXT-X-VERSION:6\n");
        for track in &self.tracks {
            if track.is_audio {
                let _ = write!(
                    out,
                    "#EXT-X-STREAM-INF:BANDWIDTH={},CODECS=\"{}\"\n{}{}\n",
                    track.bandwidth_bps, track.codecs, self.base_url, track.uri
                );
            } else {
                let _ = write!(
                    out,
                    "#EXT-X-STREAM-INF:BANDWIDTH={},RESOLUTION={}x{},CODECS=\"{}\",FRAME-RATE={:.3}\n{}{}\n",
                    track.bandwidth_bps,
                    track.width,
                    track.height,
                    track.codecs,
                    track.frame_rate,
                    self.base_url,
                    track.uri
                );
            }
        }
        out
    }

    /// Generate a minimal DASH MPD XML string.
    #[must_use]
    pub fn build_dash(&self) -> String {
        let mut out = String::from(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" profiles=\"urn:mpeg:dash:profile:isoff-live:2011\" type=\"static\">\n\
             <Period>\n",
        );

        for (i, track) in self.tracks.iter().enumerate() {
            if track.is_audio {
                let _ = write!(
                    out,
                    "  <AdaptationSet id=\"{}\" mimeType=\"audio/mp4\" codecs=\"{}\">\n\
                         <Representation id=\"r{}\" bandwidth=\"{}\">\n\
                           <BaseURL>{}{}</BaseURL>\n\
                         </Representation>\n\
                   </AdaptationSet>\n",
                    i, track.codecs, i, track.bandwidth_bps, self.base_url, track.uri
                );
            } else {
                let _ = write!(
                    out,
                    "  <AdaptationSet id=\"{}\" mimeType=\"video/mp4\" codecs=\"{}\">\n\
                         <Representation id=\"r{}\" bandwidth=\"{}\" width=\"{}\" height=\"{}\">\n\
                           <BaseURL>{}{}</BaseURL>\n\
                         </Representation>\n\
                   </AdaptationSet>\n",
                    i,
                    track.codecs,
                    i,
                    track.bandwidth_bps,
                    track.width,
                    track.height,
                    self.base_url,
                    track.uri
                );
            }
        }

        out.push_str("</Period>\n</MPD>");
        out
    }
}

impl Default for ManifestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_format_mime_hls() {
        assert_eq!(
            ManifestFormat::Hls.mime_type(),
            "application/vnd.apple.mpegurl"
        );
    }

    #[test]
    fn test_manifest_format_mime_dash() {
        assert_eq!(ManifestFormat::Dash.mime_type(), "application/dash+xml");
    }

    #[test]
    fn test_manifest_format_extension() {
        assert_eq!(ManifestFormat::Hls.extension(), "m3u8");
        assert_eq!(ManifestFormat::Dash.extension(), "mpd");
        assert_eq!(ManifestFormat::Smooth.extension(), "ism");
    }

    #[test]
    fn test_manifest_format_display() {
        assert_eq!(ManifestFormat::Hls.to_string(), "HLS");
        assert_eq!(ManifestFormat::Dash.to_string(), "DASH");
    }

    #[test]
    fn test_track_bandwidth_kbps() {
        let track = ManifestTrack::video("v1.m3u8", 3_000_000, 1280, 720, "av01.0.05M.08", 30.0);
        let kbps = track.bandwidth_kbps();
        assert!((kbps - 3000.0).abs() < 0.1);
    }

    #[test]
    fn test_track_audio_is_audio() {
        let track = ManifestTrack::audio("audio.m3u8", 128_000, "opus");
        assert!(track.is_audio());
    }

    #[test]
    fn test_track_video_not_audio() {
        let track = ManifestTrack::video("v.m3u8", 2_000_000, 1280, 720, "av01", 25.0);
        assert!(!track.is_audio());
    }

    #[test]
    fn test_track_uri_and_codecs() {
        let track = ManifestTrack::video("stream.m3u8", 1_000_000, 854, 480, "vp09", 24.0);
        assert_eq!(track.uri(), "stream.m3u8");
        assert_eq!(track.codecs(), "vp09");
    }

    #[test]
    fn test_builder_track_count() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "v.m3u8", 1_000_000, 1280, 720, "av01", 30.0,
        ));
        builder.add_track(ManifestTrack::audio("a.m3u8", 128_000, "opus"));
        assert_eq!(builder.track_count(), 2);
    }

    #[test]
    fn test_build_hls_contains_extm3u() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "v.m3u8", 2_000_000, 1280, 720, "av01", 30.0,
        ));
        let hls = builder.build_hls();
        assert!(hls.starts_with("#EXTM3U"));
    }

    #[test]
    fn test_build_hls_contains_bandwidth() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "v.m3u8", 2_000_000, 1280, 720, "av01", 30.0,
        ));
        let hls = builder.build_hls();
        assert!(hls.contains("2000000"));
    }

    #[test]
    fn test_build_dash_contains_mpd() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "v.mp4", 3_000_000, 1920, 1080, "av01", 25.0,
        ));
        let dash = builder.build_dash();
        assert!(dash.contains("<MPD"));
        assert!(dash.contains("</MPD>"));
    }

    #[test]
    fn test_build_dash_contains_adaptation_set() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::audio("a.mp4", 128_000, "opus"));
        let dash = builder.build_dash();
        assert!(dash.contains("AdaptationSet"));
    }

    #[test]
    fn test_builder_with_base_url() {
        let mut builder = ManifestBuilder::new().with_base_url("https://cdn.example.com/");
        builder.add_track(ManifestTrack::video(
            "720p.m3u8",
            2_000_000,
            1280,
            720,
            "av01",
            30.0,
        ));
        let hls = builder.build_hls();
        assert!(hls.contains("https://cdn.example.com/720p.m3u8"));
    }

    // =========================================================================
    // HLS M3U8 format validity tests
    // (TODO item: "Test HLS M3U8 output against Apple's mediastreamvalidator
    //  or equivalent format checks")
    // =========================================================================

    /// A well-formed HLS master playlist must start with `#EXTM3U` (required
    /// by the HLS spec as the very first line), contain at least one
    /// `#EXT-X-STREAM-INF` tag, include `BANDWIDTH=`, `RESOLUTION=`, and the
    /// track URI on the next line.
    #[test]
    fn test_hls_m3u8_required_tags_present() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "1080p.m3u8",
            5_000_000,
            1920,
            1080,
            "av01.0.09M.08",
            25.0,
        ));
        builder.add_track(ManifestTrack::video(
            "720p.m3u8",
            2_500_000,
            1280,
            720,
            "av01.0.05M.08",
            25.0,
        ));
        builder.add_track(ManifestTrack::audio("audio.m3u8", 128_000, "opus"));

        let m3u8 = builder.build_hls();

        // Required: #EXTM3U must be the first line.
        assert!(
            m3u8.starts_with("#EXTM3U"),
            "HLS playlist must start with #EXTM3U"
        );

        // Required: at least one EXT-X-STREAM-INF.
        assert!(
            m3u8.contains("#EXT-X-STREAM-INF:"),
            "HLS playlist must contain #EXT-X-STREAM-INF"
        );

        // Required: BANDWIDTH attribute present on every STREAM-INF line.
        for line in m3u8.lines() {
            if line.starts_with("#EXT-X-STREAM-INF:") {
                assert!(
                    line.contains("BANDWIDTH="),
                    "every #EXT-X-STREAM-INF must carry BANDWIDTH=: {line}"
                );
            }
        }

        // Required: RESOLUTION attribute present for video tracks.
        assert!(
            m3u8.contains("RESOLUTION="),
            "HLS playlist with video tracks must contain RESOLUTION="
        );

        // Check URIs are present.
        assert!(
            m3u8.contains("1080p.m3u8"),
            "1080p URI must appear in playlist"
        );
        assert!(
            m3u8.contains("720p.m3u8"),
            "720p URI must appear in playlist"
        );
    }

    /// Every `#EXT-X-STREAM-INF` tag must be immediately followed by a URI
    /// line (not another tag, not an empty line).  This validates the two-line
    /// STREAM-INF structure required by the HLS spec (RFC 8216 §4.3.4.2).
    #[test]
    fn test_hls_stream_inf_followed_by_uri() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "stream_480p.m3u8",
            1_500_000,
            854,
            480,
            "av01",
            30.0,
        ));

        let m3u8 = builder.build_hls();
        let lines: Vec<&str> = m3u8.lines().collect();

        for (i, &line) in lines.iter().enumerate() {
            if line.starts_with("#EXT-X-STREAM-INF:") {
                let next = lines.get(i + 1).copied().unwrap_or("");
                assert!(
                    !next.is_empty() && !next.starts_with('#'),
                    "#EXT-X-STREAM-INF at line {i} must be followed by a URI, got: {next:?}"
                );
            }
        }
    }

    // =========================================================================
    // DASH MPD format validity tests
    // (TODO item: "Verify DASH MPD output validates against MPD schema")
    // =========================================================================

    /// A well-formed DASH MPD must contain `<MPD`, the MPEG-DASH namespace URI,
    /// `<AdaptationSet`, and `<Representation` elements.
    #[test]
    fn test_dash_mpd_required_elements_present() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::video(
            "video_1080p.mp4",
            5_000_000,
            1920,
            1080,
            "av01.0.09M.08",
            25.0,
        ));
        builder.add_track(ManifestTrack::video(
            "video_720p.mp4",
            2_500_000,
            1280,
            720,
            "av01.0.05M.08",
            25.0,
        ));
        builder.add_track(ManifestTrack::audio("audio.mp4", 128_000, "opus"));

        let mpd = builder.build_dash();

        // Must contain the MPD root element.
        assert!(mpd.contains("<MPD"), "DASH MPD must contain <MPD element");
        assert!(
            mpd.contains("</MPD>"),
            "DASH MPD must contain closing </MPD>"
        );

        // Must reference the MPEG-DASH namespace.
        assert!(
            mpd.contains("xmlns="),
            "DASH MPD must declare the xmlns namespace"
        );
        assert!(
            mpd.contains("urn:mpeg:dash"),
            "DASH MPD xmlns must reference urn:mpeg:dash"
        );

        // Must have AdaptationSet elements.
        assert!(
            mpd.contains("AdaptationSet"),
            "DASH MPD must contain AdaptationSet"
        );

        // Must have Representation elements.
        assert!(
            mpd.contains("Representation"),
            "DASH MPD must contain Representation"
        );

        // Video Representation must carry width and height attributes.
        assert!(
            mpd.contains("width="),
            "DASH MPD video Representation must have width="
        );
        assert!(
            mpd.contains("height="),
            "DASH MPD video Representation must have height="
        );

        // Must have a Period element wrapping the adaptation sets.
        assert!(
            mpd.contains("<Period>"),
            "DASH MPD must have a <Period> element"
        );
        assert!(mpd.contains("</Period>"), "DASH MPD must close </Period>");
    }

    /// An audio-only DASH MPD must declare `mimeType="audio/mp4"` on the
    /// AdaptationSet (confirming codec-to-mimeType mapping is correct).
    #[test]
    fn test_dash_mpd_audio_mime_type() {
        let mut builder = ManifestBuilder::new();
        builder.add_track(ManifestTrack::audio("audio_en.mp4", 192_000, "opus"));

        let mpd = builder.build_dash();

        assert!(
            mpd.contains("audio/mp4"),
            "audio AdaptationSet must carry mimeType audio/mp4"
        );
        assert!(
            mpd.contains("opus"),
            "audio AdaptationSet codecs must be 'opus'"
        );
    }

    // =========================================================================
    // Bitrate ladder ordering test
    // (TODO item: "Add test for bitrate ladder ordering in manifests —
    //  highest to lowest bandwidth")
    // =========================================================================

    /// When multiple video renditions are added, the HLS master playlist must
    /// list them in ascending BANDWIDTH order (lowest first) as recommended by
    /// Apple HLS authoring guidelines — clients start from the lowest rung.
    ///
    /// Note: the spec does not mandate a specific ordering in the master
    /// playlist, but listing lowest-bandwidth first is the conventional
    /// recommendation for initial segment selection.  Our `ManifestBuilder`
    /// preserves insertion order, so this test verifies the caller's
    /// responsibility for ordering and that we don't silently reorder.
    #[test]
    fn test_hls_bitrate_ladder_ordering_preserved() {
        let mut builder = ManifestBuilder::new();

        // Add in descending order (highest first) to verify the builder
        // preserves the insertion order and does not sort internally.
        builder.add_track(ManifestTrack::video(
            "1080p.m3u8",
            5_000_000,
            1920,
            1080,
            "av01",
            25.0,
        ));
        builder.add_track(ManifestTrack::video(
            "720p.m3u8",
            2_500_000,
            1280,
            720,
            "av01",
            25.0,
        ));
        builder.add_track(ManifestTrack::video(
            "480p.m3u8",
            1_200_000,
            854,
            480,
            "av01",
            25.0,
        ));

        let m3u8 = builder.build_hls();

        // Find the positions of each bandwidth value in the output string.
        let pos_5m = m3u8
            .find("5000000")
            .expect("5 Mbps entry must appear in M3U8");
        let pos_2_5m = m3u8
            .find("2500000")
            .expect("2.5 Mbps entry must appear in M3U8");
        let pos_1_2m = m3u8
            .find("1200000")
            .expect("1.2 Mbps entry must appear in M3U8");

        // Since tracks are added in descending order, the playlist must reflect
        // that same order (5M before 2.5M before 1.2M).
        assert!(
            pos_5m < pos_2_5m,
            "5 Mbps rendition must appear before 2.5 Mbps in M3U8 output"
        );
        assert!(
            pos_2_5m < pos_1_2m,
            "2.5 Mbps rendition must appear before 1.2 Mbps in M3U8 output"
        );
    }

    /// Verify that a DASH MPD lists multiple Representation elements with
    /// bandwidth attributes in the order they were inserted.
    #[test]
    fn test_dash_mpd_representation_order_preserved() {
        let mut builder = ManifestBuilder::new();

        builder.add_track(ManifestTrack::video(
            "720p.mp4", 2_500_000, 1280, 720, "av01", 25.0,
        ));
        builder.add_track(ManifestTrack::video(
            "1080p.mp4",
            5_000_000,
            1920,
            1080,
            "av01",
            25.0,
        ));

        let mpd = builder.build_dash();

        let pos_2_5m = mpd.find("2500000").expect("2.5 Mbps in MPD");
        let pos_5m = mpd.find("5000000").expect("5 Mbps in MPD");

        // 720p was inserted first → must appear before 1080p in the MPD.
        assert!(
            pos_2_5m < pos_5m,
            "MPD must list Representations in insertion order: 2.5M before 5M"
        );
    }
}
