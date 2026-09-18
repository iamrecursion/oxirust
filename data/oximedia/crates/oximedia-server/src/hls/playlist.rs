//! HLS playlist generation.

use crate::error::ServerResult;
use crate::hls::HlsConfig;
use parking_lot::RwLock;
use std::collections::VecDeque;

/// Segment information.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SegmentInfo {
    /// Segment filename.
    pub filename: String,

    /// Duration in seconds.
    pub duration: f64,

    /// Sequence number.
    pub sequence: u64,

    /// Is discontinuity.
    pub discontinuity: bool,
}

/// Master playlist for multiple variants.
pub struct MasterPlaylist {
    /// Variants.
    variants: Vec<VariantInfo>,
}

/// Variant information.
#[derive(Debug, Clone)]
pub struct VariantInfo {
    /// Bandwidth in bits per second.
    pub bandwidth: u64,

    /// Resolution.
    pub resolution: Option<(u32, u32)>,

    /// Codecs.
    pub codecs: String,

    /// Playlist URI.
    pub uri: String,
}

impl MasterPlaylist {
    /// Creates a new master playlist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
        }
    }

    /// Adds a variant.
    pub fn add_variant(&mut self, variant: VariantInfo) {
        self.variants.push(variant);
    }

    /// Generates the M3U8 content.
    #[must_use]
    pub fn generate(&self) -> String {
        let mut content = String::from("#EXTM3U\n");
        content.push_str("#EXT-X-VERSION:3\n\n");

        for variant in &self.variants {
            content.push_str(&format!(
                "#EXT-X-STREAM-INF:BANDWIDTH={},CODECS=\"{}\"",
                variant.bandwidth, variant.codecs
            ));

            if let Some((width, height)) = variant.resolution {
                content.push_str(&format!(",RESOLUTION={}x{}", width, height));
            }

            content.push('\n');
            content.push_str(&format!("{}\n\n", variant.uri));
        }

        content
    }
}

impl Default for MasterPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

/// Media playlist for a single variant.
pub struct MediaPlaylist {
    /// Target duration.
    target_duration: u64,

    /// Media sequence.
    media_sequence: u64,

    /// Segments.
    segments: VecDeque<SegmentInfo>,

    /// Maximum playlist length.
    max_length: usize,

    /// Is ended.
    ended: bool,
}

impl MediaPlaylist {
    /// Creates a new media playlist.
    #[must_use]
    pub fn new(target_duration: u64, max_length: usize) -> Self {
        Self {
            target_duration,
            media_sequence: 0,
            segments: VecDeque::new(),
            max_length,
            ended: false,
        }
    }

    /// Adds a segment.
    pub fn add_segment(&mut self, filename: String, duration: f64) {
        let segment = SegmentInfo {
            filename,
            duration,
            sequence: self.media_sequence + self.segments.len() as u64,
            discontinuity: false,
        };

        self.segments.push_back(segment);

        // Remove old segments if exceeding max length
        while self.segments.len() > self.max_length {
            self.segments.pop_front();
            self.media_sequence += 1;
        }
    }

    /// Marks the playlist as ended.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Generates the M3U8 content.
    #[must_use]
    pub fn generate(&self) -> String {
        let mut content = String::from("#EXTM3U\n");
        content.push_str("#EXT-X-VERSION:3\n");
        content.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration));
        content.push_str(&format!(
            "#EXT-X-MEDIA-SEQUENCE:{}\n\n",
            self.media_sequence
        ));

        for segment in &self.segments {
            if segment.discontinuity {
                content.push_str("#EXT-X-DISCONTINUITY\n");
            }

            content.push_str(&format!("#EXTINF:{:.3},\n", segment.duration));
            content.push_str(&format!("{}\n", segment.filename));
        }

        if self.ended {
            content.push_str("#EXT-X-ENDLIST\n");
        }

        content
    }
}

/// Playlist generator.
#[allow(dead_code)]
pub struct PlaylistGenerator {
    /// Configuration.
    config: HlsConfig,

    /// Media playlist.
    media_playlist: RwLock<MediaPlaylist>,
}

impl PlaylistGenerator {
    /// Creates a new playlist generator.
    #[must_use]
    pub fn new(config: HlsConfig) -> Self {
        let target_duration = config.segment_duration.as_secs();
        let media_playlist = MediaPlaylist::new(target_duration, config.playlist_length);

        Self {
            config,
            media_playlist: RwLock::new(media_playlist),
        }
    }

    /// Adds a segment to the playlist.
    pub fn add_segment(&self, filename: &str, duration: f64) -> ServerResult<()> {
        let mut playlist = self.media_playlist.write();
        playlist.add_segment(filename.to_string(), duration);
        Ok(())
    }

    /// Marks the playlist as ended.
    pub fn end(&self) {
        let mut playlist = self.media_playlist.write();
        playlist.end();
    }

    /// Generates the playlist content.
    #[must_use]
    pub fn generate(&self) -> String {
        let playlist = self.media_playlist.read();
        playlist.generate()
    }
}
