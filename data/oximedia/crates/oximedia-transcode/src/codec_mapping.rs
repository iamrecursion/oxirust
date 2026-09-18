//! Container-to-codec mapping and codec compatibility utilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A container format identifier (e.g., `"mp4"`, `"webm"`, `"mkv"`).
pub type ContainerFormat = String;

/// A codec identifier (e.g., `"h264"`, `"vp9"`, `"opus"`).
pub type CodecId = String;

/// Category of a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecKind {
    /// Video codec.
    Video,
    /// Audio codec.
    Audio,
    /// Subtitle codec.
    Subtitle,
}

/// Maps container formats to supported video and audio codecs.
///
/// # Example
///
/// ```
/// use oximedia_transcode::codec_mapping::{CodecMapping, CodecKind};
///
/// let mapping = CodecMapping::default();
/// let video_codecs = mapping.supported_codecs("mp4", CodecKind::Video);
/// assert!(video_codecs.contains(&"h264".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct CodecMapping {
    /// video codecs per container.
    video: HashMap<ContainerFormat, Vec<CodecId>>,
    /// audio codecs per container.
    audio: HashMap<ContainerFormat, Vec<CodecId>>,
    /// subtitle codecs per container.
    subtitle: HashMap<ContainerFormat, Vec<CodecId>>,
}

impl Default for CodecMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecMapping {
    /// Creates a `CodecMapping` populated with standard container/codec combinations.
    #[must_use]
    pub fn new() -> Self {
        let mut video: HashMap<ContainerFormat, Vec<CodecId>> = HashMap::new();
        let mut audio: HashMap<ContainerFormat, Vec<CodecId>> = HashMap::new();
        let mut subtitle: HashMap<ContainerFormat, Vec<CodecId>> = HashMap::new();

        // MP4
        video.insert(
            "mp4".into(),
            vec![
                "h264".into(),
                "h265".into(),
                "hevc".into(),
                "av1".into(),
                "mpeg4".into(),
            ],
        );
        audio.insert(
            "mp4".into(),
            vec![
                "aac".into(),
                "mp3".into(),
                "ac3".into(),
                "eac3".into(),
                "flac".into(),
            ],
        );
        subtitle.insert("mp4".into(), vec!["mov_text".into(), "dvdsub".into()]);

        // WebM
        video.insert(
            "webm".into(),
            vec!["vp8".into(), "vp9".into(), "av1".into()],
        );
        audio.insert("webm".into(), vec!["opus".into(), "vorbis".into()]);
        subtitle.insert("webm".into(), vec!["webvtt".into()]);

        // MKV / Matroska
        video.insert(
            "mkv".into(),
            vec![
                "h264".into(),
                "h265".into(),
                "hevc".into(),
                "av1".into(),
                "vp9".into(),
                "vp8".into(),
                "mpeg4".into(),
                "theora".into(),
                "ffv1".into(),
            ],
        );
        audio.insert(
            "mkv".into(),
            vec![
                "aac".into(),
                "mp3".into(),
                "opus".into(),
                "vorbis".into(),
                "flac".into(),
                "ac3".into(),
                "truehd".into(),
                "dts".into(),
                "pcm_s16le".into(),
            ],
        );
        subtitle.insert(
            "mkv".into(),
            vec![
                "srt".into(),
                "ass".into(),
                "ssa".into(),
                "webvtt".into(),
                "dvdsub".into(),
            ],
        );

        // MOV
        video.insert(
            "mov".into(),
            vec![
                "h264".into(),
                "h265".into(),
                "prores".into(),
                "dnxhd".into(),
                "av1".into(),
            ],
        );
        audio.insert(
            "mov".into(),
            vec![
                "aac".into(),
                "pcm_s16le".into(),
                "pcm_s24le".into(),
                "mp3".into(),
            ],
        );
        subtitle.insert("mov".into(), vec!["mov_text".into()]);

        // AVI
        video.insert(
            "avi".into(),
            vec![
                "h264".into(),
                "mpeg4".into(),
                "xvid".into(),
                "divx".into(),
                "wmv2".into(),
            ],
        );
        audio.insert(
            "avi".into(),
            vec!["mp3".into(), "aac".into(), "pcm_s16le".into(), "ac3".into()],
        );
        subtitle.insert("avi".into(), vec![]);

        // TS (MPEG-TS)
        video.insert(
            "ts".into(),
            vec!["h264".into(), "h265".into(), "mpeg2video".into()],
        );
        audio.insert("ts".into(), vec!["aac".into(), "mp3".into(), "ac3".into()]);
        subtitle.insert("ts".into(), vec!["dvbsub".into()]);

        // FLV
        video.insert("flv".into(), vec!["h264".into(), "flv1".into()]);
        audio.insert("flv".into(), vec!["aac".into(), "mp3".into()]);
        subtitle.insert("flv".into(), vec![]);

        // OGG
        video.insert("ogg".into(), vec!["theora".into()]);
        audio.insert(
            "ogg".into(),
            vec!["vorbis".into(), "opus".into(), "flac".into()],
        );
        subtitle.insert("ogg".into(), vec![]);

        Self {
            video,
            audio,
            subtitle,
        }
    }

    /// Returns supported codecs for the given container and kind.
    ///
    /// Returns an empty `Vec` if the container is unknown.
    #[must_use]
    pub fn supported_codecs(&self, container: &str, kind: CodecKind) -> Vec<CodecId> {
        let map = match kind {
            CodecKind::Video => &self.video,
            CodecKind::Audio => &self.audio,
            CodecKind::Subtitle => &self.subtitle,
        };
        map.get(container).cloned().unwrap_or_default()
    }

    /// Returns `true` when `codec` is compatible with `container` for the given `kind`.
    #[must_use]
    pub fn is_compatible(&self, container: &str, codec: &str, kind: CodecKind) -> bool {
        self.supported_codecs(container, kind)
            .contains(&codec.to_string())
    }

    /// Returns all container formats known to this mapping.
    #[must_use]
    pub fn known_containers(&self) -> Vec<ContainerFormat> {
        let mut containers: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for k in self.video.keys() {
            containers.insert(k.as_str());
        }
        for k in self.audio.keys() {
            containers.insert(k.as_str());
        }
        let mut result: Vec<ContainerFormat> = containers.into_iter().map(String::from).collect();
        result.sort_unstable();
        result
    }

    /// Finds containers that support both the given video and audio codecs.
    #[must_use]
    pub fn find_compatible_containers(
        &self,
        video_codec: &str,
        audio_codec: &str,
    ) -> Vec<ContainerFormat> {
        let mut result = Vec::new();
        for container in self.known_containers() {
            let has_video = self.is_compatible(&container, video_codec, CodecKind::Video);
            let has_audio = self.is_compatible(&container, audio_codec, CodecKind::Audio);
            if has_video && has_audio {
                result.push(container);
            }
        }
        result
    }

    /// Adds a custom codec mapping for a container.
    pub fn add_codec(
        &mut self,
        container: impl Into<ContainerFormat>,
        codec: impl Into<CodecId>,
        kind: CodecKind,
    ) {
        let map = match kind {
            CodecKind::Video => &mut self.video,
            CodecKind::Audio => &mut self.audio,
            CodecKind::Subtitle => &mut self.subtitle,
        };
        map.entry(container.into()).or_default().push(codec.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp4_video_codecs() {
        let mapping = CodecMapping::default();
        let codecs = mapping.supported_codecs("mp4", CodecKind::Video);
        assert!(codecs.contains(&"h264".to_string()));
        assert!(codecs.contains(&"h265".to_string()));
    }

    #[test]
    fn test_webm_audio_codecs() {
        let mapping = CodecMapping::default();
        let codecs = mapping.supported_codecs("webm", CodecKind::Audio);
        assert!(codecs.contains(&"opus".to_string()));
        assert!(codecs.contains(&"vorbis".to_string()));
        assert!(!codecs.contains(&"aac".to_string()));
    }

    #[test]
    fn test_is_compatible() {
        let mapping = CodecMapping::default();
        assert!(mapping.is_compatible("mp4", "h264", CodecKind::Video));
        assert!(!mapping.is_compatible("webm", "h264", CodecKind::Video));
        assert!(mapping.is_compatible("mkv", "opus", CodecKind::Audio));
    }

    #[test]
    fn test_unknown_container_returns_empty() {
        let mapping = CodecMapping::default();
        let codecs = mapping.supported_codecs("unknown_fmt", CodecKind::Video);
        assert!(codecs.is_empty());
    }

    #[test]
    fn test_known_containers_non_empty() {
        let mapping = CodecMapping::default();
        let containers = mapping.known_containers();
        assert!(!containers.is_empty());
        assert!(containers.contains(&"mp4".to_string()));
        assert!(containers.contains(&"webm".to_string()));
        assert!(containers.contains(&"mkv".to_string()));
    }

    #[test]
    fn test_find_compatible_containers() {
        let mapping = CodecMapping::default();
        let containers = mapping.find_compatible_containers("vp9", "opus");
        assert!(containers.contains(&"webm".to_string()));
        assert!(containers.contains(&"mkv".to_string()));
        assert!(!containers.contains(&"mp4".to_string()));
    }

    #[test]
    fn test_add_custom_codec() {
        let mut mapping = CodecMapping::new();
        mapping.add_codec("mp4", "custom_codec", CodecKind::Video);
        assert!(mapping.is_compatible("mp4", "custom_codec", CodecKind::Video));
    }

    #[test]
    fn test_subtitle_codecs_mkv() {
        let mapping = CodecMapping::default();
        let subs = mapping.supported_codecs("mkv", CodecKind::Subtitle);
        assert!(subs.contains(&"ass".to_string()));
        assert!(subs.contains(&"srt".to_string()));
    }
}
