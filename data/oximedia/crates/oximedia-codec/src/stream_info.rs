//! Stream information parsing and media container metadata.
//!
//! Provides types for describing individual elementary streams within a media
//! container, a parser that reads a simple header, and a `MediaInfo` aggregate
//! that groups all streams found in a file.

#![allow(dead_code)]

/// Type of elementary stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle or text stream.
    Subtitle,
    /// Data / metadata stream.
    Data,
    /// Unknown stream type.
    Unknown,
}

/// Information record for a single elementary stream.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Zero-based stream index within the container.
    pub index: u32,
    /// Stream type.
    pub stream_type: StreamType,
    /// Codec identifier string (e.g. `"h264"`, `"aac"`).
    pub codec_id: String,
    /// Bitrate in bits per second (0 if unknown).
    pub bitrate_bps: u64,
    /// Duration in milliseconds (0 if unknown).
    pub duration_ms_val: u64,
    /// Sample rate (audio) or frame rate numerator (video); 0 if unused.
    pub rate_num: u32,
    /// Frame rate denominator (video only); 0 if unused.
    pub rate_den: u32,
    /// Width in pixels (video only); 0 if unused.
    pub width: u32,
    /// Height in pixels (video only); 0 if unused.
    pub height: u32,
    /// Channel count (audio only); 0 if unused.
    pub channels: u32,
}

impl StreamInfo {
    /// Create a minimal stream descriptor.
    pub fn new(index: u32, stream_type: StreamType, codec_id: impl Into<String>) -> Self {
        Self {
            index,
            stream_type,
            codec_id: codec_id.into(),
            bitrate_bps: 0,
            duration_ms_val: 0,
            rate_num: 0,
            rate_den: 1,
            width: 0,
            height: 0,
            channels: 0,
        }
    }

    /// Returns `true` if this is a video stream.
    pub fn is_video(&self) -> bool {
        self.stream_type == StreamType::Video
    }

    /// Returns `true` if this is an audio stream.
    pub fn is_audio(&self) -> bool {
        self.stream_type == StreamType::Audio
    }

    /// Returns the duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms_val
    }

    /// Returns the frame rate as a float (for video streams).
    pub fn frame_rate(&self) -> f64 {
        if self.rate_den == 0 {
            return 0.0;
        }
        self.rate_num as f64 / self.rate_den as f64
    }

    /// Builder: set bitrate.
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Builder: set duration.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms_val = ms;
        self
    }

    /// Builder: set video dimensions.
    pub fn with_video_dims(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Builder: set frame rate (numerator / denominator).
    pub fn with_frame_rate(mut self, num: u32, den: u32) -> Self {
        self.rate_num = num;
        self.rate_den = den;
        self
    }

    /// Builder: set audio properties.
    pub fn with_audio(mut self, sample_rate: u32, channels: u32) -> Self {
        self.rate_num = sample_rate;
        self.channels = channels;
        self
    }
}

/// Parses a minimal text header format into a list of [`StreamInfo`] records.
///
/// Expected format per line:
/// `<index>,<type>,<codec_id>,<bitrate_bps>,<duration_ms>,<rate_num>,<rate_den>,<w>,<h>,<ch>`
///
/// Lines starting with `#` are treated as comments and skipped.
#[derive(Debug, Default)]
pub struct StreamInfoParser;

impl StreamInfoParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse a multi-line header string into a `Vec<StreamInfo>`.
    ///
    /// Malformed lines are silently skipped.
    pub fn parse_header(&self, header: &str) -> Vec<StreamInfo> {
        let mut streams = Vec::new();
        for line in header.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split(',').collect();
            if parts.len() < 10 {
                continue;
            }
            let index: u32 = match parts[0].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let stream_type = match parts[1].trim() {
                "video" => StreamType::Video,
                "audio" => StreamType::Audio,
                "subtitle" => StreamType::Subtitle,
                "data" => StreamType::Data,
                _ => StreamType::Unknown,
            };
            let codec_id = parts[2].trim().to_string();
            let bitrate_bps: u64 = parts[3].trim().parse().unwrap_or(0);
            let duration_ms: u64 = parts[4].trim().parse().unwrap_or(0);
            let rate_num: u32 = parts[5].trim().parse().unwrap_or(0);
            let rate_den: u32 = parts[6].trim().parse().unwrap_or(1);
            let width: u32 = parts[7].trim().parse().unwrap_or(0);
            let height: u32 = parts[8].trim().parse().unwrap_or(0);
            let channels: u32 = parts[9].trim().parse().unwrap_or(0);
            streams.push(StreamInfo {
                index,
                stream_type,
                codec_id,
                bitrate_bps,
                duration_ms_val: duration_ms,
                rate_num,
                rate_den,
                width,
                height,
                channels,
            });
        }
        streams
    }
}

/// Aggregated media information for a complete container.
#[derive(Debug, Default)]
pub struct MediaInfo {
    /// All streams present in the container.
    pub streams: Vec<StreamInfo>,
    /// Container format string (e.g. `"mp4"`, `"mkv"`).
    pub container_format: String,
    /// Total file size in bytes.
    pub file_size_bytes: u64,
}

impl MediaInfo {
    /// Create an empty `MediaInfo`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the total number of streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Returns a slice of video streams only.
    pub fn video_streams(&self) -> Vec<&StreamInfo> {
        self.streams.iter().filter(|s| s.is_video()).collect()
    }

    /// Returns a slice of audio streams only.
    pub fn audio_streams(&self) -> Vec<&StreamInfo> {
        self.streams.iter().filter(|s| s.is_audio()).collect()
    }

    /// Returns the primary video stream (index 0 among video streams), if any.
    pub fn primary_video(&self) -> Option<&StreamInfo> {
        self.video_streams().into_iter().next()
    }

    /// Returns the primary audio stream (index 0 among audio streams), if any.
    pub fn primary_audio(&self) -> Option<&StreamInfo> {
        self.audio_streams().into_iter().next()
    }

    /// Returns the total bitrate across all streams in kbps.
    pub fn total_bitrate_kbps(&self) -> u64 {
        self.streams.iter().map(|s| s.bitrate_bps / 1000).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_stream() -> StreamInfo {
        StreamInfo::new(0, StreamType::Video, "h264")
            .with_video_dims(1920, 1080)
            .with_frame_rate(30, 1)
            .with_bitrate(5_000_000)
            .with_duration_ms(60_000)
    }

    fn make_audio_stream() -> StreamInfo {
        StreamInfo::new(1, StreamType::Audio, "aac")
            .with_audio(48_000, 2)
            .with_bitrate(128_000)
            .with_duration_ms(60_000)
    }

    #[test]
    fn test_stream_info_is_video() {
        let s = make_video_stream();
        assert!(s.is_video());
        assert!(!s.is_audio());
    }

    #[test]
    fn test_stream_info_is_audio() {
        let s = make_audio_stream();
        assert!(s.is_audio());
        assert!(!s.is_video());
    }

    #[test]
    fn test_stream_info_duration_ms() {
        let s = make_video_stream();
        assert_eq!(s.duration_ms(), 60_000);
    }

    #[test]
    fn test_stream_info_frame_rate() {
        let s = make_video_stream();
        assert!((s.frame_rate() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_stream_info_zero_den_frame_rate() {
        let s = StreamInfo::new(0, StreamType::Video, "av1").with_frame_rate(30, 0);
        assert!((s.frame_rate()).abs() < 0.001);
    }

    #[test]
    fn test_parser_parses_video_line() {
        let header = "0,video,h264,5000000,60000,30,1,1920,1080,0";
        let parser = StreamInfoParser::new();
        let streams = parser.parse_header(header);
        assert_eq!(streams.len(), 1);
        assert!(streams[0].is_video());
        assert_eq!(streams[0].codec_id, "h264");
        assert_eq!(streams[0].width, 1920);
    }

    #[test]
    fn test_parser_parses_audio_line() {
        let header = "1,audio,aac,128000,60000,48000,0,0,0,2";
        let parser = StreamInfoParser::new();
        let streams = parser.parse_header(header);
        assert_eq!(streams.len(), 1);
        assert!(streams[0].is_audio());
        assert_eq!(streams[0].channels, 2);
    }

    #[test]
    fn test_parser_skips_comment_lines() {
        let header = "# This is a comment\n0,video,av1,4000000,30000,24,1,1280,720,0";
        let parser = StreamInfoParser::new();
        let streams = parser.parse_header(header);
        assert_eq!(streams.len(), 1);
    }

    #[test]
    fn test_parser_skips_malformed_lines() {
        let header = "bad,line\n0,video,vp9,0,0,30,1,1920,1080,0";
        let parser = StreamInfoParser::new();
        let streams = parser.parse_header(header);
        assert_eq!(streams.len(), 1);
    }

    #[test]
    fn test_media_info_stream_count() {
        let mut info = MediaInfo::new();
        info.streams.push(make_video_stream());
        info.streams.push(make_audio_stream());
        assert_eq!(info.stream_count(), 2);
    }

    #[test]
    fn test_media_info_video_streams() {
        let mut info = MediaInfo::new();
        info.streams.push(make_video_stream());
        info.streams.push(make_audio_stream());
        assert_eq!(info.video_streams().len(), 1);
    }

    #[test]
    fn test_media_info_primary_video() {
        let mut info = MediaInfo::new();
        info.streams.push(make_video_stream());
        assert!(info.primary_video().is_some());
    }

    #[test]
    fn test_media_info_primary_audio() {
        let mut info = MediaInfo::new();
        info.streams.push(make_audio_stream());
        assert!(info.primary_audio().is_some());
    }

    #[test]
    fn test_media_info_total_bitrate_kbps() {
        let mut info = MediaInfo::new();
        info.streams.push(make_video_stream()); // 5000 kbps
        info.streams.push(make_audio_stream()); // 128 kbps
        assert_eq!(info.total_bitrate_kbps(), 5128);
    }
}
