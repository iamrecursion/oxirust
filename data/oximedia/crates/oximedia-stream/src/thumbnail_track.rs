//! Thumbnail and trick-play track generation for HLS I-frame-only playlists.
//!
//! Generates `#EXT-X-I-FRAMES-ONLY` playlists for fast forward/rewind and
//! storyboard preview thumbnails.

use crate::StreamError;

// ─── Image Format ─────────────────────────────────────────────────────────────

/// Image encoding format for thumbnail data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,
    /// WebP image.
    WebP,
    /// Raw YUV420 frame (uncompressed).
    RawYuv420,
}

impl ImageFormat {
    /// Return the MIME type string.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::WebP => "image/webp",
            Self::RawYuv420 => "application/octet-stream",
        }
    }

    /// Return the common file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::WebP => "webp",
            Self::RawYuv420 => "yuv",
        }
    }
}

// ─── Thumbnail Segment ────────────────────────────────────────────────────────

/// A single thumbnail image with timing information.
#[derive(Debug, Clone)]
pub struct ThumbnailSegment {
    /// Presentation timestamp for this thumbnail in milliseconds.
    pub start_time_ms: u64,
    /// Encoded image data.
    pub image_data: Vec<u8>,
    /// Image encoding format.
    pub format: ImageFormat,
    /// Duration this thumbnail represents in milliseconds (for playlist use).
    pub duration_ms: u64,
}

impl ThumbnailSegment {
    /// Create a new thumbnail segment.
    pub fn new(start_time_ms: u64, image_data: Vec<u8>, format: ImageFormat) -> Self {
        Self {
            start_time_ms,
            image_data,
            format,
            duration_ms: 0,
        }
    }

    /// Return the start time as floating-point seconds.
    pub fn start_time_secs(&self) -> f64 {
        self.start_time_ms as f64 / 1000.0
    }

    /// Return the duration as floating-point seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }

    /// Auto-generate a URI for this thumbnail.
    pub fn auto_uri(&self, index: usize) -> String {
        format!("thumb_{:04}.{}", index, self.format.extension())
    }
}

// ─── Thumbnail Track ─────────────────────────────────────────────────────────

/// A complete thumbnail / trick-play track.
#[derive(Debug, Clone)]
pub struct ThumbnailTrack {
    /// Desired thumbnail capture interval in frames.
    pub interval_frames: u32,
    /// Width of each thumbnail image in pixels.
    pub width: u32,
    /// Height of each thumbnail image in pixels.
    pub height: u32,
    /// Ordered list of thumbnail segments.
    pub segments: Vec<ThumbnailSegment>,
    /// Source video frame rate (frames per second) — used to compute intervals.
    pub frame_rate: f64,
}

impl ThumbnailTrack {
    /// Return the thumbnail interval in milliseconds.
    pub fn interval_ms(&self) -> u64 {
        if self.frame_rate > 0.0 {
            ((self.interval_frames as f64 / self.frame_rate) * 1000.0).round() as u64
        } else {
            0
        }
    }

    /// Generate an HLS `#EXT-X-I-FRAMES-ONLY` playlist.
    ///
    /// The output is a valid M3U8 string describing the I-frame-only rendition.
    pub fn generate_iframe_playlist(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:4\n");

        // Compute target duration (ceiling of the longest segment)
        let target_secs = self
            .segments
            .iter()
            .map(|s| s.duration_ms)
            .max()
            .map(|ms| (ms + 999) / 1000)
            .unwrap_or(2);

        out.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_secs));
        out.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
        out.push_str("#EXT-X-I-FRAMES-ONLY\n\n");

        for (idx, seg) in self.segments.iter().enumerate() {
            let uri = seg.auto_uri(idx);
            let byte_len = seg.image_data.len();
            out.push_str(&format!(
                "#EXT-X-BYTERANGE:{}@0\n#EXTINF:{:.3},\n{}\n",
                byte_len,
                seg.duration_secs(),
                uri,
            ));
        }

        out.push_str("#EXT-X-ENDLIST\n");
        out
    }

    /// Generate a WebVTT thumbnail track for video.js / Shaka Player storyboards.
    ///
    /// Each cue references the thumbnail image at the corresponding timestamp.
    pub fn generate_vtt_storyboard(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for (idx, seg) in self.segments.iter().enumerate() {
            let start = seg.start_time_ms;
            let end = start + seg.duration_ms.max(1);
            let uri = seg.auto_uri(idx);
            out.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                idx + 1,
                format_vtt_time(start),
                format_vtt_time(end),
                uri,
            ));
        }
        out
    }
}

/// Format milliseconds as `HH:MM:SS.mmm` for WebVTT.
fn format_vtt_time(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

// ─── Thumbnail Track Builder ──────────────────────────────────────────────────

/// Incrementally constructs a [`ThumbnailTrack`] from raw video frames.
#[derive(Debug)]
pub struct ThumbnailTrackBuilder {
    interval_frames: u32,
    width: u32,
    height: u32,
    format: ImageFormat,
    frame_rate: f64,
    frames: Vec<(u64, Vec<u8>)>,
}

impl ThumbnailTrackBuilder {
    /// Create a new builder.
    ///
    /// * `interval_frames` — capture one thumbnail every N frames.
    /// * `width` / `height` — output thumbnail dimensions.
    /// * `format` — output image format.
    /// * `frame_rate` — source video frame rate (fps).
    pub fn new(
        interval_frames: u32,
        width: u32,
        height: u32,
        format: ImageFormat,
        frame_rate: f64,
    ) -> Self {
        Self {
            interval_frames,
            width,
            height,
            format,
            frame_rate,
            frames: Vec::new(),
        }
    }

    /// Add a frame at the given presentation timestamp (milliseconds).
    pub fn add_frame(&mut self, time_ms: u64, data: Vec<u8>) {
        self.frames.push((time_ms, data));
    }

    /// Consume the builder and return the finalised [`ThumbnailTrack`].
    ///
    /// Returns an error if no frames have been added.
    pub fn build(self) -> Result<ThumbnailTrack, StreamError> {
        if self.frames.is_empty() {
            return Err(StreamError::Generic(
                "cannot build ThumbnailTrack: no frames added".to_string(),
            ));
        }

        // Sort frames by timestamp to ensure monotonic ordering.
        let mut frames = self.frames;
        frames.sort_by_key(|(t, _)| *t);

        let mut segments: Vec<ThumbnailSegment> = Vec::with_capacity(frames.len());

        for (i, (time_ms, data)) in frames.iter().enumerate() {
            let mut seg = ThumbnailSegment::new(*time_ms, data.clone(), self.format.clone());
            // Duration is the gap to the next frame (or a default for the last).
            seg.duration_ms = if i + 1 < frames.len() {
                frames[i + 1].0.saturating_sub(*time_ms)
            } else {
                // Use the average interval for the last segment.
                if self.frame_rate > 0.0 {
                    ((self.interval_frames as f64 / self.frame_rate) * 1000.0).round() as u64
                } else {
                    2000
                }
            };
            segments.push(seg);
        }

        Ok(ThumbnailTrack {
            interval_frames: self.interval_frames,
            width: self.width,
            height: self.height,
            segments,
            frame_rate: self.frame_rate,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_builder() -> ThumbnailTrackBuilder {
        ThumbnailTrackBuilder::new(30, 320, 180, ImageFormat::Jpeg, 30.0)
    }

    fn sample_track() -> ThumbnailTrack {
        let mut b = make_builder();
        b.add_frame(0, vec![0xFF, 0xD8, 0xFF]);
        b.add_frame(1000, vec![0xFF, 0xD8, 0xFE]);
        b.add_frame(2000, vec![0xFF, 0xD8, 0xFD]);
        b.build().expect("build")
    }

    #[test]
    fn test_builder_empty_fails() {
        let b = make_builder();
        assert!(b.build().is_err());
    }

    #[test]
    fn test_builder_produces_correct_segment_count() {
        let track = sample_track();
        assert_eq!(track.segments.len(), 3);
    }

    #[test]
    fn test_segment_duration_computed() {
        let track = sample_track();
        // First segment: 1000 ms gap
        assert_eq!(track.segments[0].duration_ms, 1000);
        // Second segment: 1000 ms gap
        assert_eq!(track.segments[1].duration_ms, 1000);
    }

    #[test]
    fn test_iframe_playlist_header() {
        let track = sample_track();
        let pl = track.generate_iframe_playlist();
        assert!(pl.starts_with("#EXTM3U\n"));
        assert!(pl.contains("#EXT-X-I-FRAMES-ONLY"));
    }

    #[test]
    fn test_iframe_playlist_contains_extinf() {
        let track = sample_track();
        let pl = track.generate_iframe_playlist();
        assert!(pl.contains("#EXTINF:"));
    }

    #[test]
    fn test_iframe_playlist_ends_with_endlist() {
        let track = sample_track();
        let pl = track.generate_iframe_playlist();
        assert!(pl.trim_end().ends_with("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_vtt_storyboard_header() {
        let track = sample_track();
        let vtt = track.generate_vtt_storyboard();
        assert!(vtt.starts_with("WEBVTT\n"));
    }

    #[test]
    fn test_vtt_storyboard_cue_count() {
        let track = sample_track();
        let vtt = track.generate_vtt_storyboard();
        // Count "WEBVTT" arrows (-->) as proxy for cue count
        let cue_count = vtt.matches(" --> ").count();
        assert_eq!(cue_count, 3);
    }

    #[test]
    fn test_interval_ms_calculation() {
        let track = sample_track(); // 30 frames @ 30 fps = 1000 ms
        assert_eq!(track.interval_ms(), 1000);
    }

    #[test]
    fn test_image_format_mime() {
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    }

    #[test]
    fn test_auto_uri_format() {
        let seg = ThumbnailSegment::new(0, vec![], ImageFormat::Jpeg);
        assert_eq!(seg.auto_uri(5), "thumb_0005.jpg");
    }
}
