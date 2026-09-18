//! Track selection and filtering.
//!
//! Provides sophisticated track selection based on various criteria.

#![forbid(unsafe_code)]

use oximedia_core::CodecId;

use crate::StreamInfo;

/// Criteria for selecting tracks.
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Codec types to include.
    pub codecs: Option<Vec<CodecId>>,
    /// Language codes to include (e.g., "eng", "jpn").
    pub languages: Option<Vec<String>>,
    /// Stream indices to include.
    pub indices: Option<Vec<usize>>,
    /// Minimum quality level (0-100).
    pub min_quality: Option<u32>,
    /// Maximum bitrate in bits per second.
    pub max_bitrate: Option<u64>,
    /// Whether to include only default tracks.
    pub default_only: bool,
    /// Whether to include forced tracks.
    pub include_forced: bool,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            codecs: None,
            languages: None,
            indices: None,
            min_quality: None,
            max_bitrate: None,
            default_only: false,
            include_forced: true,
        }
    }
}

impl SelectionCriteria {
    /// Creates a new selection criteria with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            codecs: None,
            languages: None,
            indices: None,
            min_quality: None,
            max_bitrate: None,
            default_only: false,
            include_forced: true,
        }
    }

    /// Sets the codec filter.
    #[must_use]
    pub fn with_codecs(mut self, codecs: Vec<CodecId>) -> Self {
        self.codecs = Some(codecs);
        self
    }

    /// Sets the language filter.
    #[must_use]
    pub fn with_languages(mut self, languages: Vec<String>) -> Self {
        self.languages = Some(languages);
        self
    }

    /// Sets the index filter.
    #[must_use]
    pub fn with_indices(mut self, indices: Vec<usize>) -> Self {
        self.indices = Some(indices);
        self
    }

    /// Sets the minimum quality level.
    #[must_use]
    pub const fn with_min_quality(mut self, quality: u32) -> Self {
        self.min_quality = Some(quality);
        self
    }

    /// Sets the maximum bitrate.
    #[must_use]
    pub const fn with_max_bitrate(mut self, bitrate: u64) -> Self {
        self.max_bitrate = Some(bitrate);
        self
    }

    /// Sets whether to include only default tracks.
    #[must_use]
    pub const fn with_default_only(mut self, enabled: bool) -> Self {
        self.default_only = enabled;
        self
    }

    /// Sets whether to include forced tracks.
    #[must_use]
    pub const fn with_include_forced(mut self, enabled: bool) -> Self {
        self.include_forced = enabled;
        self
    }
}

/// Track type categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
    /// Data track.
    Data,
}

impl TrackType {
    /// Returns the track type for a codec.
    #[must_use]
    pub const fn from_codec(codec: CodecId) -> Self {
        match codec {
            CodecId::Av1 | CodecId::Vp8 | CodecId::Vp9 => Self::Video,
            CodecId::Opus | CodecId::Flac | CodecId::Vorbis => Self::Audio,
            _ => Self::Data,
        }
    }
}

/// Selector for filtering and choosing tracks.
pub struct TrackSelector {
    criteria: SelectionCriteria,
}

impl TrackSelector {
    /// Creates a new track selector with default criteria.
    #[must_use]
    pub fn new() -> Self {
        Self {
            criteria: SelectionCriteria::default(),
        }
    }

    /// Creates a new track selector with custom criteria.
    #[must_use]
    pub const fn with_criteria(criteria: SelectionCriteria) -> Self {
        Self { criteria }
    }

    /// Returns the selection criteria.
    #[must_use]
    pub const fn criteria(&self) -> &SelectionCriteria {
        &self.criteria
    }

    /// Sets the selection criteria.
    pub fn set_criteria(&mut self, criteria: SelectionCriteria) {
        self.criteria = criteria;
    }

    /// Filters streams based on the selection criteria.
    #[must_use]
    pub fn select(&self, streams: &[StreamInfo]) -> Vec<usize> {
        streams
            .iter()
            .enumerate()
            .filter_map(|(index, stream)| {
                if self.matches(stream) {
                    Some(index)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Checks if a stream matches the criteria.
    fn matches(&self, stream: &StreamInfo) -> bool {
        // Check codec filter
        if let Some(ref codecs) = self.criteria.codecs {
            if !codecs.contains(&stream.codec) {
                return false;
            }
        }

        // Check index filter
        if let Some(ref indices) = self.criteria.indices {
            if !indices.contains(&stream.index) {
                return false;
            }
        }

        // Check language filter (if metadata contains language)
        if let Some(ref languages) = self.criteria.languages {
            if let Some(lang) = stream.metadata.get("language") {
                if !languages.iter().any(|l| l.eq_ignore_ascii_case(lang)) {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Selects the best track for each type.
    #[must_use]
    pub fn select_best_per_type(&self, streams: &[StreamInfo]) -> Vec<usize> {
        let mut selected = Vec::new();

        // Select best video track
        if let Some(video_idx) = self.select_best_by_type(streams, TrackType::Video) {
            selected.push(video_idx);
        }

        // Select best audio track
        if let Some(audio_idx) = self.select_best_by_type(streams, TrackType::Audio) {
            selected.push(audio_idx);
        }

        // Select best subtitle track
        if let Some(subtitle_idx) = self.select_best_by_type(streams, TrackType::Subtitle) {
            selected.push(subtitle_idx);
        }

        selected
    }

    /// Selects the best track of a specific type.
    fn select_best_by_type(&self, streams: &[StreamInfo], track_type: TrackType) -> Option<usize> {
        streams
            .iter()
            .enumerate()
            .filter(|(_, stream)| TrackType::from_codec(stream.codec) == track_type)
            .filter(|(_, stream)| self.matches(stream))
            .max_by_key(|(_, stream)| self.score_stream(stream))
            .map(|(index, _)| index)
    }

    /// Scores a stream for quality ranking.
    #[allow(clippy::unused_self, clippy::cast_possible_wrap)]
    fn score_stream(&self, stream: &StreamInfo) -> i32 {
        let mut score = 0;

        // Prefer higher quality codecs
        score += match stream.codec {
            CodecId::Av1 | CodecId::Flac => 100,
            CodecId::Opus => 90,
            CodecId::Vp9 => 80,
            CodecId::Vorbis => 70,
            CodecId::Vp8 => 60,
            _ => 0,
        };

        // Prefer higher sample rates for audio
        if let Some(sample_rate) = stream.codec_params.sample_rate {
            score += (sample_rate / 1000) as i32;
        }

        score
    }

    /// Returns all video track indices.
    #[must_use]
    pub fn video_tracks(&self, streams: &[StreamInfo]) -> Vec<usize> {
        self.tracks_by_type(streams, TrackType::Video)
    }

    /// Returns all audio track indices.
    #[must_use]
    pub fn audio_tracks(&self, streams: &[StreamInfo]) -> Vec<usize> {
        self.tracks_by_type(streams, TrackType::Audio)
    }

    /// Returns all subtitle track indices.
    #[must_use]
    pub fn subtitle_tracks(&self, streams: &[StreamInfo]) -> Vec<usize> {
        self.tracks_by_type(streams, TrackType::Subtitle)
    }

    /// Returns track indices by type.
    fn tracks_by_type(&self, streams: &[StreamInfo], track_type: TrackType) -> Vec<usize> {
        streams
            .iter()
            .enumerate()
            .filter(|(_, stream)| TrackType::from_codec(stream.codec) == track_type)
            .filter(|(_, stream)| self.matches(stream))
            .map(|(index, _)| index)
            .collect()
    }
}

impl Default for TrackSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset selection configurations.
pub struct SelectionPresets;

impl SelectionPresets {
    /// Creates criteria for selecting all video tracks.
    #[must_use]
    pub fn all_video() -> SelectionCriteria {
        SelectionCriteria::new().with_codecs(vec![CodecId::Av1, CodecId::Vp9, CodecId::Vp8])
    }

    /// Creates criteria for selecting all audio tracks.
    #[must_use]
    pub fn all_audio() -> SelectionCriteria {
        SelectionCriteria::new().with_codecs(vec![CodecId::Opus, CodecId::Flac, CodecId::Vorbis])
    }

    /// Creates criteria for selecting high-quality tracks.
    #[must_use]
    pub fn high_quality() -> SelectionCriteria {
        SelectionCriteria::new()
            .with_codecs(vec![CodecId::Av1, CodecId::Flac])
            .with_min_quality(80)
    }

    /// Creates criteria for selecting low-bandwidth tracks.
    #[must_use]
    pub fn low_bandwidth() -> SelectionCriteria {
        SelectionCriteria::new().with_max_bitrate(1_000_000) // 1 Mbps
    }

    /// Creates criteria for selecting English tracks.
    #[must_use]
    pub fn english() -> SelectionCriteria {
        SelectionCriteria::new().with_languages(vec!["eng".into(), "en".into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn create_test_stream(index: usize, codec: CodecId, language: Option<&str>) -> StreamInfo {
        let metadata = if let Some(lang) = language {
            crate::stream::Metadata::default().with_entry("language", lang)
        } else {
            crate::stream::Metadata::default()
        };

        let mut stream = StreamInfo::new(index, codec, Rational::new(1, 48000));
        stream.metadata = metadata;
        stream
    }

    #[test]
    fn test_selection_criteria() {
        let criteria = SelectionCriteria::new()
            .with_codecs(vec![CodecId::Opus])
            .with_languages(vec!["eng".into()])
            .with_min_quality(50)
            .with_max_bitrate(128_000)
            .with_default_only(true);

        assert!(criteria.codecs.is_some());
        assert!(criteria.languages.is_some());
        assert_eq!(criteria.min_quality, Some(50));
        assert_eq!(criteria.max_bitrate, Some(128_000));
        assert!(criteria.default_only);
    }

    #[test]
    fn test_track_type() {
        assert_eq!(TrackType::from_codec(CodecId::Av1), TrackType::Video);
        assert_eq!(TrackType::from_codec(CodecId::Opus), TrackType::Audio);
    }

    #[test]
    fn test_track_selector() {
        let streams = vec![
            create_test_stream(0, CodecId::Av1, Some("eng")),
            create_test_stream(1, CodecId::Opus, Some("eng")),
            create_test_stream(2, CodecId::Opus, Some("jpn")),
        ];

        let criteria = SelectionCriteria::new().with_languages(vec!["eng".into()]);
        let selector = TrackSelector::with_criteria(criteria);

        let selected = selector.select(&streams);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_track_selector_by_type() {
        let streams = vec![
            create_test_stream(0, CodecId::Av1, None),
            create_test_stream(1, CodecId::Opus, None),
            create_test_stream(2, CodecId::Vp9, None),
        ];

        let selector = TrackSelector::new();

        let video_tracks = selector.video_tracks(&streams);
        assert_eq!(video_tracks.len(), 2);

        let audio_tracks = selector.audio_tracks(&streams);
        assert_eq!(audio_tracks.len(), 1);
    }

    #[test]
    fn test_selection_presets() {
        let video_criteria = SelectionPresets::all_video();
        assert!(video_criteria.codecs.is_some());

        let audio_criteria = SelectionPresets::all_audio();
        assert!(audio_criteria.codecs.is_some());

        let hq_criteria = SelectionPresets::high_quality();
        assert_eq!(hq_criteria.min_quality, Some(80));

        let eng_criteria = SelectionPresets::english();
        assert!(eng_criteria.languages.is_some());
    }
}
