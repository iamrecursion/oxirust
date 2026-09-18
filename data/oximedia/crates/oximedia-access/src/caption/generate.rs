//! Caption generation from audio with multi-speaker differentiation.

use crate::caption::{Caption, CaptionQuality, CaptionType};
use crate::error::{AccessError, AccessResult};
use oximedia_audio::frame::AudioBuffer;
use oximedia_subtitle::Subtitle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for caption generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionConfig {
    /// Language code (e.g., "en", "es", "fr").
    pub language: String,
    /// Caption type.
    pub caption_type: CaptionType,
    /// Quality level.
    pub quality: CaptionQuality,
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum lines per caption.
    pub max_lines: usize,
    /// Minimum caption duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum caption duration in milliseconds.
    pub max_duration_ms: i64,
    /// Enable speaker identification.
    pub identify_speakers: bool,
    /// Enable sound effects descriptions.
    pub include_sound_effects: bool,
    /// Enable music descriptions.
    pub include_music_description: bool,
}

/// Speaker color assignment for multi-speaker differentiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerColor {
    /// Text color RGBA.
    pub text_color: (u8, u8, u8, u8),
    /// Label prefix (e.g., ">> SPEAKER 1:").
    pub label_prefix: String,
}

/// Label format for speaker identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeakerLabelFormat {
    /// Use the speaker name directly (e.g., "John:").
    Name,
    /// Use chevron prefix (e.g., ">> John:").
    Chevron,
    /// Use brackets (e.g., "\[John\]").
    Bracket,
    /// Use parentheses (e.g., "(John)").
    Parenthesis,
    /// No label; only differentiate by color.
    ColorOnly,
}

/// Configuration for multi-speaker differentiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerConfig {
    /// Label format for speaker identification.
    pub label_format: SpeakerLabelFormat,
    /// Whether to use color coding per speaker.
    pub color_coding: bool,
    /// Maximum number of distinct speakers to track.
    pub max_speakers: usize,
    /// Default speaker name when identity is unknown.
    pub unknown_speaker_label: String,
    /// Predefined color palette for speakers (RGBA).
    pub color_palette: Vec<(u8, u8, u8, u8)>,
}

impl Default for MultiSpeakerConfig {
    fn default() -> Self {
        Self {
            label_format: SpeakerLabelFormat::Chevron,
            color_coding: true,
            max_speakers: 8,
            unknown_speaker_label: "Speaker".to_string(),
            color_palette: vec![
                (255, 255, 255, 255), // White
                (255, 255, 0, 255),   // Yellow
                (0, 255, 255, 255),   // Cyan
                (0, 255, 0, 255),     // Green
                (255, 165, 0, 255),   // Orange
                (255, 105, 180, 255), // Pink
                (173, 216, 230, 255), // Light blue
                (255, 200, 200, 255), // Light red
            ],
        }
    }
}

impl MultiSpeakerConfig {
    /// Format a speaker label according to the configured format.
    #[must_use]
    pub fn format_label(&self, speaker_name: &str) -> String {
        match self.label_format {
            SpeakerLabelFormat::Name => format!("{speaker_name}:"),
            SpeakerLabelFormat::Chevron => format!(">> {speaker_name}:"),
            SpeakerLabelFormat::Bracket => format!("[{speaker_name}]"),
            SpeakerLabelFormat::Parenthesis => format!("({speaker_name})"),
            SpeakerLabelFormat::ColorOnly => String::new(),
        }
    }

    /// Get the color for a speaker index.
    #[must_use]
    pub fn get_speaker_color(&self, index: usize) -> SpeakerColor {
        let color = if index < self.color_palette.len() {
            self.color_palette[index]
        } else {
            // Cycle through palette for extra speakers
            self.color_palette[index % self.color_palette.len()]
        };
        let label = format!("{} {}", self.unknown_speaker_label, index + 1);
        SpeakerColor {
            text_color: color,
            label_prefix: self.format_label(&label),
        }
    }
}

/// Tracks speaker assignments and provides consistent color/label mapping.
#[derive(Debug, Clone)]
pub struct SpeakerTracker {
    config: MultiSpeakerConfig,
    /// Maps speaker name to assigned index.
    speaker_map: HashMap<String, usize>,
    /// Next speaker index.
    next_index: usize,
}

impl SpeakerTracker {
    /// Create a new speaker tracker.
    #[must_use]
    pub fn new(config: MultiSpeakerConfig) -> Self {
        Self {
            config,
            speaker_map: HashMap::new(),
            next_index: 0,
        }
    }

    /// Register a speaker and return their assigned index.
    pub fn register_speaker(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.speaker_map.get(name) {
            return idx;
        }
        let idx = self.next_index;
        if self.next_index < self.config.max_speakers {
            self.speaker_map.insert(name.to_string(), idx);
            self.next_index += 1;
        }
        idx.min(self.config.max_speakers.saturating_sub(1))
    }

    /// Get the color assignment for a speaker.
    #[must_use]
    pub fn get_color(&self, name: &str) -> SpeakerColor {
        let idx = self.speaker_map.get(name).copied().unwrap_or(0);
        let color = self.config.get_speaker_color(idx);
        SpeakerColor {
            text_color: color.text_color,
            label_prefix: self.config.format_label(name),
        }
    }

    /// Get the number of tracked speakers.
    #[must_use]
    pub fn speaker_count(&self) -> usize {
        self.speaker_map.len()
    }

    /// Get all tracked speaker names.
    #[must_use]
    pub fn speakers(&self) -> Vec<&str> {
        let mut speakers: Vec<(&str, &usize)> = self
            .speaker_map
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        speakers.sort_by_key(|(_, &idx)| idx);
        speakers.into_iter().map(|(name, _)| name).collect()
    }

    /// Get config reference.
    #[must_use]
    pub fn config(&self) -> &MultiSpeakerConfig {
        &self.config
    }
}

/// A speech segment from voice activity detection with speaker identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSegment {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Speaker identifier.
    pub speaker: String,
    /// Transcribed text for this segment.
    pub text: String,
    /// Confidence score for speaker identification (0.0 to 1.0).
    pub speaker_confidence: f32,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            caption_type: CaptionType::Closed,
            quality: CaptionQuality::Standard,
            max_chars_per_line: 42,
            max_lines: 2,
            min_duration_ms: 1000,
            max_duration_ms: 7000,
            identify_speakers: true,
            include_sound_effects: true,
            include_music_description: true,
        }
    }
}

impl CaptionConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(language: String, caption_type: CaptionType) -> Self {
        Self {
            language,
            caption_type,
            ..Default::default()
        }
    }

    /// Set quality level.
    #[must_use]
    pub const fn with_quality(mut self, quality: CaptionQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set maximum characters per line.
    #[must_use]
    pub const fn with_max_chars_per_line(mut self, max_chars: usize) -> Self {
        self.max_chars_per_line = max_chars;
        self
    }

    /// Enable speaker identification.
    #[must_use]
    pub const fn with_speaker_identification(mut self, enable: bool) -> Self {
        self.identify_speakers = enable;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> AccessResult<()> {
        if self.max_chars_per_line == 0 {
            return Err(AccessError::CaptionFailed(
                "Max characters per line must be positive".to_string(),
            ));
        }

        if self.max_lines == 0 {
            return Err(AccessError::CaptionFailed(
                "Max lines must be positive".to_string(),
            ));
        }

        if self.min_duration_ms <= 0 {
            return Err(AccessError::CaptionFailed(
                "Minimum duration must be positive".to_string(),
            ));
        }

        if self.max_duration_ms < self.min_duration_ms {
            return Err(AccessError::CaptionFailed(
                "Maximum duration must be >= minimum duration".to_string(),
            ));
        }

        Ok(())
    }
}

/// Caption generator with multi-speaker differentiation.
///
/// Generates captions from audio using speech-to-text with support for
/// speaker identification, color coding, and label formatting.
pub struct CaptionGenerator {
    config: CaptionConfig,
    speaker_config: MultiSpeakerConfig,
    speaker_tracker: SpeakerTracker,
}

impl CaptionGenerator {
    /// Create a new caption generator.
    #[must_use]
    pub fn new(config: CaptionConfig) -> Self {
        let speaker_config = MultiSpeakerConfig::default();
        let speaker_tracker = SpeakerTracker::new(speaker_config.clone());
        Self {
            config,
            speaker_config,
            speaker_tracker,
        }
    }

    /// Create generator with default configuration.
    #[must_use]
    pub fn default() -> Self {
        Self::new(CaptionConfig::default())
    }

    /// Create generator with multi-speaker configuration.
    #[must_use]
    pub fn with_speaker_config(mut self, config: MultiSpeakerConfig) -> Self {
        self.speaker_tracker = SpeakerTracker::new(config.clone());
        self.speaker_config = config;
        self
    }

    /// Generate captions from audio.
    ///
    /// This is an integration point for speech-to-text services.
    /// In production, this would call services like:
    /// - AWS Transcribe
    /// - Google Cloud Speech-to-Text
    /// - Microsoft Azure Speech
    /// - `OpenAI` Whisper
    /// - Local STT engines
    pub fn generate_from_audio(&self, _audio: &AudioBuffer) -> AccessResult<Vec<Caption>> {
        self.config.validate()?;

        let captions = vec![
            self.create_caption(1000, 3000, "Example caption text.", None),
            self.create_caption(4000, 6000, "Another caption segment.", None),
        ];

        Ok(captions)
    }

    /// Generate captions from speech segments with multi-speaker differentiation.
    ///
    /// Takes pre-segmented speech with speaker identities and produces
    /// formatted captions with speaker labels and color assignments.
    pub fn generate_from_segments(
        &mut self,
        segments: &[SpeechSegment],
    ) -> AccessResult<Vec<Caption>> {
        self.config.validate()?;

        if segments.is_empty() {
            return Err(AccessError::CaptionFailed(
                "No speech segments provided".to_string(),
            ));
        }

        let mut captions = Vec::new();
        let mut prev_speaker: Option<String> = None;

        for segment in segments {
            // Register speaker for consistent color assignment
            self.speaker_tracker.register_speaker(&segment.speaker);
            let speaker_color = self.speaker_tracker.get_color(&segment.speaker);

            // Determine if we need a speaker label change
            let needs_label = match &prev_speaker {
                Some(prev) => prev != &segment.speaker,
                None => true,
            };

            let display_text = if needs_label && self.config.identify_speakers {
                let label = &speaker_color.label_prefix;
                if label.is_empty() {
                    segment.text.clone()
                } else {
                    format!("{label} {}", segment.text)
                }
            } else {
                segment.text.clone()
            };

            let formatted = self.format_text(&display_text);
            let mut caption = self.create_caption(
                segment.start_time_ms,
                segment.end_time_ms,
                &formatted,
                Some(segment.speaker.clone()),
            );

            // Assign speaker confidence as caption confidence
            caption.confidence = segment.speaker_confidence;

            captions.push(caption);
            prev_speaker = Some(segment.speaker.clone());
        }

        Ok(captions)
    }

    /// Generate from existing transcript.
    pub fn generate_from_transcript(
        &self,
        transcript: &str,
        timestamps: &[(i64, i64)],
    ) -> AccessResult<Vec<Caption>> {
        self.config.validate()?;

        if timestamps.is_empty() {
            return Err(AccessError::CaptionFailed(
                "No timestamps provided".to_string(),
            ));
        }

        let words: Vec<&str> = transcript.split_whitespace().collect();
        let words_per_segment = words.len() / timestamps.len();

        let mut captions = Vec::new();

        for (i, (start, end)) in timestamps.iter().enumerate() {
            let word_start = i * words_per_segment;
            let word_end = ((i + 1) * words_per_segment).min(words.len());

            if word_start < words.len() {
                let text = words[word_start..word_end].join(" ");
                let formatted = self.format_text(&text);
                captions.push(self.create_caption(*start, *end, &formatted, None));
            }
        }

        Ok(captions)
    }

    /// Generate from transcript with per-entry speaker assignments.
    pub fn generate_from_diarized_transcript(
        &mut self,
        entries: &[(i64, i64, &str, &str)], // (start, end, speaker, text)
    ) -> AccessResult<Vec<Caption>> {
        let segments: Vec<SpeechSegment> = entries
            .iter()
            .map(|(start, end, speaker, text)| SpeechSegment {
                start_time_ms: *start,
                end_time_ms: *end,
                speaker: (*speaker).to_string(),
                text: (*text).to_string(),
                speaker_confidence: 1.0,
            })
            .collect();
        self.generate_from_segments(&segments)
    }

    /// Get a summary of all detected speakers and their color assignments.
    #[must_use]
    pub fn speaker_summary(&self) -> Vec<(String, SpeakerColor)> {
        self.speaker_tracker
            .speakers()
            .iter()
            .map(|name| (name.to_string(), self.speaker_tracker.get_color(name)))
            .collect()
    }

    /// Format text according to caption rules.
    fn format_text(&self, text: &str) -> String {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.len() + word.len() < self.config.max_chars_per_line {
                if !current_line.is_empty() {
                    current_line.push(' ');
                }
                current_line.push_str(word);
            } else {
                if !current_line.is_empty() {
                    lines.push(current_line.clone());
                    current_line.clear();
                }
                current_line.push_str(word);
            }

            if lines.len() >= self.config.max_lines {
                break;
            }
        }

        if !current_line.is_empty() && lines.len() < self.config.max_lines {
            lines.push(current_line);
        }

        lines.join("\n")
    }

    /// Create a caption with proper formatting.
    fn create_caption(
        &self,
        start_time: i64,
        end_time: i64,
        text: &str,
        speaker: Option<String>,
    ) -> Caption {
        let formatted_text = self.format_text(text);

        let subtitle = Subtitle::new(start_time, end_time, formatted_text);

        let mut caption = Caption::new(subtitle, self.config.caption_type);

        if let Some(speaker_name) = speaker {
            caption = caption.with_speaker(speaker_name);
        }

        caption
    }

    /// Add sound effect description.
    #[must_use]
    pub fn add_sound_effect(&self, time: i64, effect: &str) -> Caption {
        let text = format!("[{effect}]");
        let subtitle = Subtitle::new(time, time + 1000, text);
        Caption::new(subtitle, self.config.caption_type)
    }

    /// Add music description.
    #[must_use]
    pub fn add_music_description(
        &self,
        start_time: i64,
        end_time: i64,
        description: &str,
    ) -> Caption {
        let text = format!("\u{266a} {description} \u{266a}");
        let subtitle = Subtitle::new(start_time, end_time, text);
        Caption::new(subtitle, self.config.caption_type)
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &CaptionConfig {
        &self.config
    }

    /// Get the speaker tracker.
    #[must_use]
    pub fn speaker_tracker(&self) -> &SpeakerTracker {
        &self.speaker_tracker
    }

    /// Validate caption duration.
    pub fn validate_caption(&self, caption: &Caption) -> AccessResult<()> {
        let duration = caption.duration();

        if duration < self.config.min_duration_ms {
            return Err(AccessError::CaptionFailed(format!(
                "Caption duration too short: {}ms < {}ms",
                duration, self.config.min_duration_ms
            )));
        }

        if duration > self.config.max_duration_ms {
            return Err(AccessError::CaptionFailed(format!(
                "Caption duration too long: {}ms > {}ms",
                duration, self.config.max_duration_ms
            )));
        }

        Ok(())
    }

    /// Split long caption into multiple segments.
    #[must_use]
    pub fn split_caption(&self, caption: &Caption) -> Vec<Caption> {
        if caption.duration() <= self.config.max_duration_ms {
            return vec![caption.clone()];
        }

        let words: Vec<&str> = caption.text().split_whitespace().collect();
        let segment_count = (caption.duration() / self.config.max_duration_ms) + 1;
        let words_per_segment = words.len() / segment_count as usize;

        let mut segments = Vec::new();
        let duration_per_segment = caption.duration() / segment_count;

        for i in 0..segment_count as usize {
            let start_word = i * words_per_segment;
            let end_word = ((i + 1) * words_per_segment).min(words.len());

            if start_word < words.len() {
                let text = words[start_word..end_word].join(" ");
                let start_time = caption.start_time() + (i as i64 * duration_per_segment);
                let end_time = start_time + duration_per_segment;

                let subtitle = Subtitle::new(start_time, end_time, text);
                segments.push(Caption::new(subtitle, caption.caption_type));
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CaptionConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.max_chars_per_line, 42);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = CaptionConfig::default();
        assert!(config.validate().is_ok());

        config.max_chars_per_line = 0;
        assert!(config.validate().is_err());

        config.max_chars_per_line = 42;
        config.min_duration_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generator_creation() {
        let generator = CaptionGenerator::default();
        assert_eq!(generator.config().language, "en");
    }

    #[test]
    fn test_format_text() {
        let generator = CaptionGenerator::new(CaptionConfig::default().with_max_chars_per_line(20));

        let text = "This is a very long caption that should be split into multiple lines";
        let formatted = generator.format_text(text);

        assert!(formatted.contains('\n'));
    }

    #[test]
    fn test_generate_from_transcript() {
        let generator = CaptionGenerator::default();
        let transcript = "This is a test transcript with some words";
        let timestamps = vec![(1000, 3000), (4000, 6000)];

        let captions = generator
            .generate_from_transcript(transcript, &timestamps)
            .expect("test expectation failed");
        assert_eq!(captions.len(), 2);
    }

    #[test]
    fn test_split_caption() {
        let config = CaptionConfig::default().with_max_chars_per_line(10);
        let generator = CaptionGenerator::new(config);

        let long_text = "Word ".repeat(100);
        let subtitle = Subtitle::new(0, 10000, long_text);
        let caption = Caption::new(subtitle, CaptionType::Closed);

        let segments = generator.split_caption(&caption);
        assert!(segments.len() > 1);
    }

    // ============================================================
    // Multi-speaker differentiation tests
    // ============================================================

    #[test]
    fn test_speaker_tracker_register() {
        let config = MultiSpeakerConfig::default();
        let mut tracker = SpeakerTracker::new(config);

        let idx1 = tracker.register_speaker("Alice");
        let idx2 = tracker.register_speaker("Bob");
        let idx1_again = tracker.register_speaker("Alice");

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx1_again, 0); // Same speaker gets same index
        assert_eq!(tracker.speaker_count(), 2);
    }

    #[test]
    fn test_speaker_tracker_max_speakers() {
        let config = MultiSpeakerConfig {
            max_speakers: 3,
            ..MultiSpeakerConfig::default()
        };
        let mut tracker = SpeakerTracker::new(config);

        tracker.register_speaker("A");
        tracker.register_speaker("B");
        tracker.register_speaker("C");
        let idx = tracker.register_speaker("D"); // Exceeds max

        // Should clamp to max_speakers - 1
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_speaker_color_assignment() {
        let config = MultiSpeakerConfig::default();
        let mut tracker = SpeakerTracker::new(config);

        tracker.register_speaker("Alice");
        tracker.register_speaker("Bob");

        let alice_color = tracker.get_color("Alice");
        let bob_color = tracker.get_color("Bob");

        // Alice gets first color (white), Bob gets second (yellow)
        assert_eq!(alice_color.text_color, (255, 255, 255, 255));
        assert_eq!(bob_color.text_color, (255, 255, 0, 255));
    }

    #[test]
    fn test_speaker_label_formats() {
        let mut config = MultiSpeakerConfig::default();

        config.label_format = SpeakerLabelFormat::Name;
        assert_eq!(config.format_label("Alice"), "Alice:");

        config.label_format = SpeakerLabelFormat::Chevron;
        assert_eq!(config.format_label("Alice"), ">> Alice:");

        config.label_format = SpeakerLabelFormat::Bracket;
        assert_eq!(config.format_label("Alice"), "[Alice]");

        config.label_format = SpeakerLabelFormat::Parenthesis;
        assert_eq!(config.format_label("Alice"), "(Alice)");

        config.label_format = SpeakerLabelFormat::ColorOnly;
        assert_eq!(config.format_label("Alice"), "");
    }

    #[test]
    fn test_generate_from_segments_multi_speaker() {
        let config = CaptionConfig::default().with_speaker_identification(true);
        let mut generator =
            CaptionGenerator::new(config).with_speaker_config(MultiSpeakerConfig::default());

        let segments = vec![
            SpeechSegment {
                start_time_ms: 0,
                end_time_ms: 2000,
                speaker: "Alice".to_string(),
                text: "Hello, how are you?".to_string(),
                speaker_confidence: 0.95,
            },
            SpeechSegment {
                start_time_ms: 2500,
                end_time_ms: 4500,
                speaker: "Bob".to_string(),
                text: "I am doing well, thanks.".to_string(),
                speaker_confidence: 0.88,
            },
            SpeechSegment {
                start_time_ms: 5000,
                end_time_ms: 7000,
                speaker: "Alice".to_string(),
                text: "That is great to hear.".to_string(),
                speaker_confidence: 0.92,
            },
        ];

        let captions = generator
            .generate_from_segments(&segments)
            .expect("segment generation should succeed");

        assert_eq!(captions.len(), 3);

        // First caption should have Alice's speaker label
        assert_eq!(captions[0].speaker.as_deref(), Some("Alice"));
        assert!(captions[0].text().contains(">> Alice:"));

        // Second caption should have Bob's speaker label (different speaker)
        assert_eq!(captions[1].speaker.as_deref(), Some("Bob"));
        assert!(captions[1].text().contains(">> Bob:"));

        // Third caption should have Alice's label again (speaker changed back)
        assert!(captions[2].text().contains(">> Alice:"));
    }

    #[test]
    fn test_generate_from_segments_same_speaker_no_repeat_label() {
        let config = CaptionConfig::default().with_speaker_identification(true);
        let mut generator =
            CaptionGenerator::new(config).with_speaker_config(MultiSpeakerConfig::default());

        let segments = vec![
            SpeechSegment {
                start_time_ms: 0,
                end_time_ms: 2000,
                speaker: "Alice".to_string(),
                text: "First sentence.".to_string(),
                speaker_confidence: 0.95,
            },
            SpeechSegment {
                start_time_ms: 2500,
                end_time_ms: 4500,
                speaker: "Alice".to_string(),
                text: "Second sentence.".to_string(),
                speaker_confidence: 0.95,
            },
        ];

        let captions = generator
            .generate_from_segments(&segments)
            .expect("segment generation should succeed");

        // First gets label, second should NOT (same speaker continues)
        assert!(captions[0].text().contains(">> Alice:"));
        assert!(!captions[1].text().contains(">> Alice:"));
    }

    #[test]
    fn test_generate_from_segments_color_only() {
        let config = CaptionConfig::default().with_speaker_identification(true);
        let speaker_config = MultiSpeakerConfig {
            label_format: SpeakerLabelFormat::ColorOnly,
            ..MultiSpeakerConfig::default()
        };
        let mut generator = CaptionGenerator::new(config).with_speaker_config(speaker_config);

        let segments = vec![SpeechSegment {
            start_time_ms: 0,
            end_time_ms: 2000,
            speaker: "Alice".to_string(),
            text: "Hello world.".to_string(),
            speaker_confidence: 0.95,
        }];

        let captions = generator
            .generate_from_segments(&segments)
            .expect("segment generation should succeed");

        // ColorOnly mode: no label prefix, text should be raw
        assert!(!captions[0].text().contains("Alice"));
        assert!(captions[0].text().contains("Hello world"));
    }

    #[test]
    fn test_generate_from_diarized_transcript() {
        let config = CaptionConfig::default().with_speaker_identification(true);
        let mut generator =
            CaptionGenerator::new(config).with_speaker_config(MultiSpeakerConfig::default());

        let entries: Vec<(i64, i64, &str, &str)> = vec![
            (0, 2000, "Alice", "Good morning."),
            (2500, 4500, "Bob", "Good morning to you."),
        ];

        let captions = generator
            .generate_from_diarized_transcript(&entries)
            .expect("diarized generation should succeed");

        assert_eq!(captions.len(), 2);
        assert_eq!(captions[0].speaker.as_deref(), Some("Alice"));
        assert_eq!(captions[1].speaker.as_deref(), Some("Bob"));
    }

    #[test]
    fn test_speaker_summary() {
        let config = CaptionConfig::default();
        let mut generator =
            CaptionGenerator::new(config).with_speaker_config(MultiSpeakerConfig::default());

        let segments = vec![
            SpeechSegment {
                start_time_ms: 0,
                end_time_ms: 2000,
                speaker: "Alice".to_string(),
                text: "Hello.".to_string(),
                speaker_confidence: 0.95,
            },
            SpeechSegment {
                start_time_ms: 3000,
                end_time_ms: 5000,
                speaker: "Bob".to_string(),
                text: "Hi.".to_string(),
                speaker_confidence: 0.90,
            },
        ];

        let _ = generator.generate_from_segments(&segments);

        let summary = generator.speaker_summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].0, "Alice");
        assert_eq!(summary[1].0, "Bob");
    }

    #[test]
    fn test_generate_from_segments_empty() {
        let config = CaptionConfig::default();
        let mut generator = CaptionGenerator::new(config);
        let result = generator.generate_from_segments(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_speaker_confidence_propagation() {
        let config = CaptionConfig::default();
        let mut generator = CaptionGenerator::new(config);

        let segments = vec![SpeechSegment {
            start_time_ms: 0,
            end_time_ms: 2000,
            speaker: "Alice".to_string(),
            text: "Test.".to_string(),
            speaker_confidence: 0.72,
        }];

        let captions = generator
            .generate_from_segments(&segments)
            .expect("should succeed");
        assert!((captions[0].confidence - 0.72).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sound_effect_and_music() {
        let generator = CaptionGenerator::default();

        let sfx = generator.add_sound_effect(5000, "door slams");
        assert!(sfx.text().contains("[door slams]"));
        assert_eq!(sfx.start_time(), 5000);

        let music = generator.add_music_description(10000, 20000, "dramatic orchestral music");
        assert!(music.text().contains("dramatic orchestral music"));
    }
}
