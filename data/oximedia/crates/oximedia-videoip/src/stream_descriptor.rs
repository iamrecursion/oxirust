//! Stream descriptor types for `VideoIP`.
//!
//! Describes the properties of a video/audio stream so that receivers can
//! negotiate compatible formats before establishing a connection.

#![allow(dead_code)]

use std::collections::HashMap;

/// Classification of a stream by its encoding type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Uncompressed video (e.g. v210, UYVY).
    UncompressedVideo,
    /// Compressed video (e.g. AV1, VP9, H.264).
    CompressedVideo,
    /// Uncompressed PCM audio.
    UncompressedAudio,
    /// Compressed audio (e.g. Opus, AAC).
    CompressedAudio,
    /// Data / ancillary stream (timecode, tally, etc.).
    Ancillary,
}

impl StreamType {
    /// Returns `true` for compressed video or audio stream types.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            StreamType::CompressedVideo | StreamType::CompressedAudio
        )
    }

    /// Returns `true` for any video stream type.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            StreamType::UncompressedVideo | StreamType::CompressedVideo
        )
    }

    /// Returns `true` for any audio stream type.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            StreamType::UncompressedAudio | StreamType::CompressedAudio
        )
    }
}

/// Describes the complete set of parameters for a single stream.
#[derive(Debug, Clone)]
pub struct StreamDescriptor {
    /// Unique stream identifier string.
    id: String,
    /// Type of stream.
    stream_type: StreamType,
    /// Pixel / sample width.
    width: u32,
    /// Pixel / sample height (for video) or channel count (for audio).
    height: u32,
    /// Frame rate (fps) for video; sample rate (Hz) for audio.
    rate: f64,
    /// Human-readable codec name (e.g. "av1", "`pcm_s24le`").
    codec: String,
    /// Arbitrary metadata key-value pairs.
    metadata: HashMap<String, String>,
}

impl StreamDescriptor {
    /// Creates a new `StreamDescriptor`.
    pub fn new(
        id: impl Into<String>,
        stream_type: StreamType,
        width: u32,
        height: u32,
        rate: f64,
        codec: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            stream_type,
            width,
            height,
            rate,
            codec: codec.into(),
            metadata: HashMap::new(),
        }
    }

    /// Returns the stream ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the stream type.
    #[must_use]
    pub fn stream_type(&self) -> &StreamType {
        &self.stream_type
    }

    /// Returns `true` if the video stream is HD (width ≥ 1280).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.stream_type.is_video() && self.width >= 1280
    }

    /// Returns `true` if the video stream is UHD (width ≥ 3840).
    #[must_use]
    pub fn is_uhd(&self) -> bool {
        self.stream_type.is_video() && self.width >= 3840
    }

    /// Returns `true` if `width` and `height` match this descriptor exactly.
    #[must_use]
    pub fn matches_resolution(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    /// Returns the codec name.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// Returns the frame / sample rate.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Sets an arbitrary metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Returns a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// A registry of `StreamDescriptor` instances, keyed by stream ID.
#[derive(Debug, Clone, Default)]
pub struct StreamDescriptorRegistry {
    descriptors: HashMap<String, StreamDescriptor>,
}

impl StreamDescriptorRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
        }
    }

    /// Registers a stream descriptor. Returns `false` if the ID already exists.
    pub fn register(&mut self, desc: StreamDescriptor) -> bool {
        let id = desc.id().to_owned();
        if self.descriptors.contains_key(&id) {
            return false;
        }
        self.descriptors.insert(id, desc);
        true
    }

    /// Returns a reference to the descriptor with the given ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&StreamDescriptor> {
        self.descriptors.get(id)
    }

    /// Finds all descriptors whose `stream_type` and resolution both match.
    #[must_use]
    pub fn find_compatible(
        &self,
        stream_type: &StreamType,
        width: u32,
        height: u32,
    ) -> Vec<&StreamDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.stream_type() == stream_type && d.matches_resolution(width, height))
            .collect()
    }

    /// Returns the total number of registered descriptors.
    #[must_use]
    pub fn count(&self) -> usize {
        self.descriptors.len()
    }

    /// Removes a descriptor by ID. Returns `true` if it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.descriptors.remove(id).is_some()
    }

    /// Returns all descriptors matching the given `StreamType`.
    #[must_use]
    pub fn by_type(&self, stream_type: &StreamType) -> Vec<&StreamDescriptor> {
        self.descriptors
            .values()
            .filter(|d| d.stream_type() == stream_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_video() -> StreamDescriptor {
        StreamDescriptor::new("hd-1", StreamType::CompressedVideo, 1920, 1080, 60.0, "av1")
    }

    fn sd_video() -> StreamDescriptor {
        StreamDescriptor::new(
            "sd-1",
            StreamType::UncompressedVideo,
            720,
            576,
            25.0,
            "v210",
        )
    }

    fn uhd_video() -> StreamDescriptor {
        StreamDescriptor::new(
            "uhd-1",
            StreamType::CompressedVideo,
            3840,
            2160,
            30.0,
            "av1",
        )
    }

    #[test]
    fn test_stream_type_is_compressed_video() {
        assert!(StreamType::CompressedVideo.is_compressed());
    }

    #[test]
    fn test_stream_type_uncompressed_not_compressed() {
        assert!(!StreamType::UncompressedVideo.is_compressed());
        assert!(!StreamType::UncompressedAudio.is_compressed());
    }

    #[test]
    fn test_stream_type_is_video() {
        assert!(StreamType::CompressedVideo.is_video());
        assert!(StreamType::UncompressedVideo.is_video());
        assert!(!StreamType::CompressedAudio.is_video());
    }

    #[test]
    fn test_stream_type_is_audio() {
        assert!(StreamType::CompressedAudio.is_audio());
        assert!(StreamType::UncompressedAudio.is_audio());
        assert!(!StreamType::CompressedVideo.is_audio());
    }

    #[test]
    fn test_is_hd_true() {
        assert!(hd_video().is_hd());
    }

    #[test]
    fn test_is_hd_false_for_sd() {
        assert!(!sd_video().is_hd());
    }

    #[test]
    fn test_is_uhd() {
        assert!(uhd_video().is_uhd());
        assert!(!hd_video().is_uhd());
    }

    #[test]
    fn test_matches_resolution_true() {
        assert!(hd_video().matches_resolution(1920, 1080));
    }

    #[test]
    fn test_matches_resolution_false() {
        assert!(!hd_video().matches_resolution(1280, 720));
    }

    #[test]
    fn test_metadata_set_get() {
        let mut d = hd_video();
        d.set_metadata("bitrate", "50000000");
        assert_eq!(d.get_metadata("bitrate"), Some("50000000"));
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = StreamDescriptorRegistry::new();
        assert!(reg.register(hd_video()));
        assert!(reg.get("hd-1").is_some());
    }

    #[test]
    fn test_registry_duplicate_returns_false() {
        let mut reg = StreamDescriptorRegistry::new();
        reg.register(hd_video());
        assert!(!reg.register(hd_video()));
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_find_compatible() {
        let mut reg = StreamDescriptorRegistry::new();
        reg.register(hd_video());
        reg.register(sd_video());
        let found = reg.find_compatible(&StreamType::CompressedVideo, 1920, 1080);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id(), "hd-1");
    }

    #[test]
    fn test_registry_by_type() {
        let mut reg = StreamDescriptorRegistry::new();
        reg.register(hd_video());
        reg.register(uhd_video());
        reg.register(sd_video());
        let compressed = reg.by_type(&StreamType::CompressedVideo);
        assert_eq!(compressed.len(), 2);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = StreamDescriptorRegistry::new();
        reg.register(sd_video());
        assert!(reg.remove("sd-1"));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_ancillary_not_compressed() {
        assert!(!StreamType::Ancillary.is_compressed());
    }
}
