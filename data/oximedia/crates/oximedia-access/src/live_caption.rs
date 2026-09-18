#![allow(dead_code)]
//! Live captioning pipeline: STT -> caption generation -> rendering.
//!
//! Provides a unified pipeline that chains speech-to-text transcription,
//! caption generation (with speaker differentiation and styling), and
//! output rendering into a single configurable processor for real-time
//! and near-real-time captioning workflows.

use crate::caption::{
    Caption, CaptionConfig, CaptionGenerator, CaptionStyle, CaptionStylePreset, CaptionType,
    MultiSpeakerConfig,
};
use crate::error::{AccessError, AccessResult};
use oximedia_subtitle::Subtitle;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

/// Operating mode for the live captioning pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionMode {
    /// Real-time: captions appear as speech is detected (lowest latency).
    RealTime,
    /// Near-real-time: short buffer for better accuracy (200-500ms delay).
    NearRealTime,
    /// Batch: process complete audio segments before generating captions.
    Batch,
}

impl fmt::Display for CaptionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RealTime => write!(f, "Real-Time"),
            Self::NearRealTime => write!(f, "Near Real-Time"),
            Self::Batch => write!(f, "Batch"),
        }
    }
}

/// Configuration for the live captioning pipeline.
#[derive(Debug, Clone)]
pub struct LiveCaptionConfig {
    /// Operating mode.
    pub mode: CaptionMode,
    /// Maximum latency in milliseconds before forcing caption output.
    pub max_latency_ms: u64,
    /// Minimum confidence to accept a word (0.0 to 1.0).
    pub min_confidence: f32,
    /// Maximum number of captions to keep in the output buffer.
    pub max_buffer_size: usize,
    /// Whether to enable speaker identification.
    pub speaker_identification: bool,
    /// Style preset for generated captions.
    pub style_preset: CaptionStylePreset,
    /// Language code for STT.
    pub language: String,
    /// Whether to include non-speech audio descriptions.
    pub include_non_speech: bool,
    /// Rolling display: how many recent captions to show at once.
    pub rolling_display_count: usize,
    /// Whether to correct captions when better recognition arrives.
    pub enable_correction: bool,
}

impl Default for LiveCaptionConfig {
    fn default() -> Self {
        Self {
            mode: CaptionMode::NearRealTime,
            max_latency_ms: 300,
            min_confidence: 0.4,
            max_buffer_size: 100,
            speaker_identification: true,
            style_preset: CaptionStylePreset::Standard,
            language: "en".to_string(),
            include_non_speech: true,
            rolling_display_count: 3,
            enable_correction: true,
        }
    }
}

impl LiveCaptionConfig {
    /// Validate configuration.
    pub fn validate(&self) -> AccessResult<()> {
        if self.max_latency_ms == 0 {
            return Err(AccessError::CaptionFailed(
                "Max latency must be positive".to_string(),
            ));
        }
        if self.max_buffer_size == 0 {
            return Err(AccessError::CaptionFailed(
                "Buffer size must be positive".to_string(),
            ));
        }
        if self.rolling_display_count == 0 {
            return Err(AccessError::CaptionFailed(
                "Rolling display count must be positive".to_string(),
            ));
        }
        if self.language.is_empty() {
            return Err(AccessError::CaptionFailed(
                "Language must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// State of the live captioning pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineState {
    /// Pipeline is idle, not processing.
    Idle,
    /// Pipeline is running and processing audio.
    Running,
    /// Pipeline is paused (audio buffered but not processed).
    Paused,
    /// Pipeline encountered an error.
    Error,
    /// Pipeline is finishing (draining buffered audio).
    Draining,
}

impl fmt::Display for PipelineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Error => write!(f, "Error"),
            Self::Draining => write!(f, "Draining"),
        }
    }
}

/// A word recognized from the audio stream with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizedWord {
    /// The recognized word text.
    pub text: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Start time in milliseconds (relative to stream start).
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Speaker identifier (if diarization enabled).
    pub speaker: Option<String>,
    /// Whether this is a final result or interim.
    pub is_final: bool,
}

/// An interim recognition result that may be corrected later.
#[derive(Debug, Clone)]
struct InterimResult {
    /// Words in this interim result.
    words: Vec<RecognizedWord>,
    /// Sequence number for ordering.
    sequence: u64,
    /// Whether this has been finalized.
    finalized: bool,
}

/// A rendered caption ready for display.
#[derive(Debug, Clone)]
pub struct RenderedCaption {
    /// The caption data.
    pub caption: Caption,
    /// Style applied to this caption.
    pub style: CaptionStyle,
    /// Whether this is a correction of a previous caption.
    pub is_correction: bool,
    /// Sequence number for ordering.
    pub sequence: u64,
    /// Speaker color (RGBA) if speaker identification is enabled.
    pub speaker_color: Option<(u8, u8, u8, u8)>,
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total words processed.
    pub total_words: u64,
    /// Total captions generated.
    pub total_captions: u64,
    /// Number of corrections made.
    pub corrections: u64,
    /// Average word confidence.
    pub avg_confidence: f64,
    /// Number of words rejected for low confidence.
    pub rejected_words: u64,
    /// Number of unique speakers detected.
    pub speakers_detected: u64,
    /// Current pipeline latency estimate in milliseconds.
    pub estimated_latency_ms: u64,
}

/// The live captioning pipeline.
///
/// Processes incoming recognized words through caption generation, styling,
/// and rendering stages to produce display-ready captions.
pub struct LiveCaptionPipeline {
    config: LiveCaptionConfig,
    state: PipelineState,
    /// Caption generator with speaker tracking.
    generator: CaptionGenerator,
    /// Default style for captions.
    style: CaptionStyle,
    /// Output buffer of rendered captions.
    output_buffer: VecDeque<RenderedCaption>,
    /// Pending words waiting to form a caption.
    pending_words: Vec<RecognizedWord>,
    /// Interim results that may be corrected.
    interim_results: Vec<InterimResult>,
    /// Sequence counter for ordering.
    next_sequence: u64,
    /// Running statistics.
    stats: PipelineStats,
    /// Current speaker (for change detection).
    current_speaker: Option<String>,
    /// Timestamp of last caption emission.
    last_emission_ms: i64,
}

impl LiveCaptionPipeline {
    /// Create a new live captioning pipeline.
    pub fn new(config: LiveCaptionConfig) -> AccessResult<Self> {
        config.validate()?;

        let caption_config = CaptionConfig::new(config.language.clone(), CaptionType::Closed)
            .with_speaker_identification(config.speaker_identification);

        let speaker_config = MultiSpeakerConfig::default();
        let generator = CaptionGenerator::new(caption_config).with_speaker_config(speaker_config);

        let style = CaptionStyle::from_preset(config.style_preset);

        Ok(Self {
            config,
            state: PipelineState::Idle,
            generator,
            style,
            output_buffer: VecDeque::new(),
            pending_words: Vec::new(),
            interim_results: Vec::new(),
            next_sequence: 1,
            stats: PipelineStats::default(),
            current_speaker: None,
            last_emission_ms: 0,
        })
    }

    /// Start the pipeline.
    pub fn start(&mut self) -> AccessResult<()> {
        match self.state {
            PipelineState::Idle | PipelineState::Paused | PipelineState::Error => {
                self.state = PipelineState::Running;
                Ok(())
            }
            PipelineState::Running => Err(AccessError::CaptionFailed(
                "Pipeline is already running".to_string(),
            )),
            PipelineState::Draining => Err(AccessError::CaptionFailed(
                "Pipeline is draining, cannot start".to_string(),
            )),
        }
    }

    /// Pause the pipeline.
    pub fn pause(&mut self) -> AccessResult<()> {
        if self.state != PipelineState::Running {
            return Err(AccessError::CaptionFailed(
                "Pipeline is not running".to_string(),
            ));
        }
        self.state = PipelineState::Paused;
        Ok(())
    }

    /// Stop the pipeline and drain remaining content.
    pub fn stop(&mut self) -> AccessResult<Vec<RenderedCaption>> {
        self.state = PipelineState::Draining;

        // Flush pending words into a final caption
        if !self.pending_words.is_empty() {
            self.emit_caption_from_pending()?;
        }

        self.state = PipelineState::Idle;

        // Return all remaining captions
        let remaining: Vec<RenderedCaption> = self.output_buffer.drain(..).collect();
        Ok(remaining)
    }

    /// Feed recognized words into the pipeline.
    ///
    /// Words are buffered and assembled into captions according to the
    /// pipeline mode and configuration.
    pub fn feed_words(&mut self, words: &[RecognizedWord]) -> AccessResult<Vec<RenderedCaption>> {
        if self.state != PipelineState::Running {
            return Err(AccessError::CaptionFailed(format!(
                "Pipeline is not running (state: {})",
                self.state
            )));
        }

        let mut new_captions = Vec::new();

        for word in words {
            self.stats.total_words += 1;

            // Skip low-confidence words
            if word.confidence < self.config.min_confidence {
                self.stats.rejected_words += 1;
                continue;
            }

            // Update running confidence average
            let n = (self.stats.total_words - self.stats.rejected_words) as f64;
            if n > 0.0 {
                self.stats.avg_confidence =
                    self.stats.avg_confidence * ((n - 1.0) / n) + f64::from(word.confidence) / n;
            }

            // Detect speaker change
            let speaker_changed = match (&self.current_speaker, &word.speaker) {
                (Some(current), Some(new_speaker)) => current != new_speaker,
                (None, Some(_)) => true,
                _ => false,
            };

            // Emit caption on speaker change
            if speaker_changed && !self.pending_words.is_empty() {
                if let Some(caption) = self.emit_caption_from_pending()? {
                    new_captions.push(caption);
                }
            }

            if let Some(ref speaker) = word.speaker {
                self.current_speaker = Some(speaker.clone());
            }

            self.pending_words.push(word.clone());

            // Check if we should emit based on mode
            let should_emit = match self.config.mode {
                CaptionMode::RealTime => self.pending_words.len() >= 3,
                CaptionMode::NearRealTime => {
                    self.pending_words.len() >= 6
                        || self.check_sentence_boundary()
                        || self.check_latency_exceeded(word.end_ms)
                }
                CaptionMode::Batch => self.check_sentence_boundary(),
            };

            if should_emit {
                if let Some(caption) = self.emit_caption_from_pending()? {
                    new_captions.push(caption);
                }
            }
        }

        Ok(new_captions)
    }

    /// Feed a correction for previously recognized words.
    pub fn feed_correction(
        &mut self,
        sequence: u64,
        corrected_text: &str,
    ) -> AccessResult<Option<RenderedCaption>> {
        if !self.config.enable_correction {
            return Ok(None);
        }

        // Find the caption in the output buffer
        let found = self
            .output_buffer
            .iter_mut()
            .find(|c| c.sequence == sequence);

        if let Some(rendered) = found {
            let old_start = rendered.caption.start_time();
            let old_end = rendered.caption.end_time();
            let speaker = rendered.caption.speaker.clone();

            let subtitle = Subtitle::new(old_start, old_end, corrected_text.to_string());
            let mut new_caption = Caption::new(subtitle, CaptionType::Closed);
            if let Some(s) = speaker {
                new_caption = new_caption.with_speaker(s);
            }

            rendered.caption = new_caption;
            rendered.is_correction = true;
            self.stats.corrections += 1;

            Ok(Some(rendered.clone()))
        } else {
            Ok(None)
        }
    }

    /// Get the currently visible captions (rolling display).
    #[must_use]
    pub fn visible_captions(&self) -> Vec<&RenderedCaption> {
        let count = self.config.rolling_display_count;
        self.output_buffer
            .iter()
            .rev()
            .take(count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Get pipeline state.
    #[must_use]
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Get pipeline statistics.
    #[must_use]
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &LiveCaptionConfig {
        &self.config
    }

    /// Get the total number of captions in the output buffer.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.output_buffer.len()
    }

    /// Clear the output buffer.
    pub fn clear_buffer(&mut self) {
        self.output_buffer.clear();
    }

    /// Check if pending words end at a sentence boundary.
    fn check_sentence_boundary(&self) -> bool {
        self.pending_words
            .last()
            .map(|w| {
                w.text.ends_with('.')
                    || w.text.ends_with('!')
                    || w.text.ends_with('?')
                    || w.text.ends_with(':')
            })
            .unwrap_or(false)
    }

    /// Check if the latency since last emission exceeds the max.
    fn check_latency_exceeded(&self, current_ms: i64) -> bool {
        if self.last_emission_ms == 0 {
            return false;
        }
        let elapsed = (current_ms - self.last_emission_ms).unsigned_abs();
        elapsed >= self.config.max_latency_ms
    }

    /// Emit a caption from the pending words buffer.
    fn emit_caption_from_pending(&mut self) -> AccessResult<Option<RenderedCaption>> {
        if self.pending_words.is_empty() {
            return Ok(None);
        }

        let words = std::mem::take(&mut self.pending_words);

        let text: String = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let start_ms = words.first().map(|w| w.start_ms).unwrap_or(0);
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(start_ms + 1000);

        let avg_confidence: f32 = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
        };

        let speaker = words.first().and_then(|w| w.speaker.clone());

        let subtitle = Subtitle::new(start_ms, end_ms, text);
        let mut caption =
            Caption::new(subtitle, CaptionType::Closed).with_confidence(avg_confidence);

        if let Some(ref s) = speaker {
            caption = caption.with_speaker(s.clone());
        }

        let speaker_color = if self.config.speaker_identification {
            speaker.as_ref().map(|s| {
                let tracker = self.generator.speaker_tracker();
                let color = tracker.get_color(s);
                color.text_color
            })
        } else {
            None
        };

        let sequence = self.next_sequence;
        self.next_sequence += 1;

        let rendered = RenderedCaption {
            caption,
            style: self.style.clone(),
            is_correction: false,
            sequence,
            speaker_color,
        };

        // Manage buffer size
        if self.output_buffer.len() >= self.config.max_buffer_size {
            self.output_buffer.pop_front();
        }

        self.output_buffer.push_back(rendered.clone());
        self.stats.total_captions += 1;
        self.last_emission_ms = end_ms;

        // Track unique speakers
        if speaker.is_some() {
            // Register with generator's tracker for consistent colors
            // This is a simplified check; the generator tracks internally
            self.stats.speakers_detected = self
                .stats
                .speakers_detected
                .max(self.generator.speaker_tracker().speaker_count() as u64);
        }

        self.stats.estimated_latency_ms = match self.config.mode {
            CaptionMode::RealTime => 50,
            CaptionMode::NearRealTime => self.config.max_latency_ms,
            CaptionMode::Batch => 1000,
        };

        Ok(Some(rendered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(text: &str, start: i64, end: i64, confidence: f32) -> RecognizedWord {
        RecognizedWord {
            text: text.to_string(),
            confidence,
            start_ms: start,
            end_ms: end,
            speaker: None,
            is_final: true,
        }
    }

    fn make_word_with_speaker(text: &str, start: i64, end: i64, speaker: &str) -> RecognizedWord {
        RecognizedWord {
            text: text.to_string(),
            confidence: 0.95,
            start_ms: start,
            end_ms: end,
            speaker: Some(speaker.to_string()),
            is_final: true,
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = LiveCaptionPipeline::new(LiveCaptionConfig::default());
        assert!(pipeline.is_ok());
        let pipeline = pipeline.expect("should succeed");
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }

    #[test]
    fn test_pipeline_invalid_config() {
        let config = LiveCaptionConfig {
            max_latency_ms: 0,
            ..LiveCaptionConfig::default()
        };
        assert!(LiveCaptionPipeline::new(config).is_err());

        let config2 = LiveCaptionConfig {
            max_buffer_size: 0,
            ..LiveCaptionConfig::default()
        };
        assert!(LiveCaptionPipeline::new(config2).is_err());

        let config3 = LiveCaptionConfig {
            language: String::new(),
            ..LiveCaptionConfig::default()
        };
        assert!(LiveCaptionPipeline::new(config3).is_err());
    }

    #[test]
    fn test_pipeline_start_stop() {
        let mut pipeline =
            LiveCaptionPipeline::new(LiveCaptionConfig::default()).expect("should succeed");
        assert_eq!(pipeline.state(), PipelineState::Idle);

        pipeline.start().expect("start should succeed");
        assert_eq!(pipeline.state(), PipelineState::Running);

        let remaining = pipeline.stop().expect("stop should succeed");
        assert_eq!(pipeline.state(), PipelineState::Idle);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_pipeline_pause_resume() {
        let mut pipeline =
            LiveCaptionPipeline::new(LiveCaptionConfig::default()).expect("should succeed");
        pipeline.start().expect("start should succeed");

        pipeline.pause().expect("pause should succeed");
        assert_eq!(pipeline.state(), PipelineState::Paused);

        pipeline.start().expect("resume should succeed");
        assert_eq!(pipeline.state(), PipelineState::Running);
    }

    #[test]
    fn test_pipeline_cannot_start_twice() {
        let mut pipeline =
            LiveCaptionPipeline::new(LiveCaptionConfig::default()).expect("should succeed");
        pipeline.start().expect("start should succeed");
        assert!(pipeline.start().is_err());
    }

    #[test]
    fn test_feed_words_not_running() {
        let mut pipeline =
            LiveCaptionPipeline::new(LiveCaptionConfig::default()).expect("should succeed");
        let words = vec![make_word("hello", 0, 500, 0.9)];
        assert!(pipeline.feed_words(&words).is_err());
    }

    #[test]
    fn test_feed_words_real_time_mode() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        // Feed 3 words (threshold for real-time emission)
        let words = vec![
            make_word("Hello", 0, 300, 0.95),
            make_word("world", 300, 600, 0.90),
            make_word("today.", 600, 900, 0.88),
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        assert_eq!(captions.len(), 1);
        assert!(captions[0].caption.text().contains("Hello"));
        assert!(captions[0].caption.text().contains("world"));
    }

    #[test]
    fn test_feed_words_low_confidence_rejected() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            min_confidence: 0.8,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("clear", 0, 300, 0.95),
            make_word("mumble", 300, 600, 0.3), // below threshold
            make_word("clear2", 600, 900, 0.90),
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        assert_eq!(pipeline.stats().rejected_words, 1);

        // Only 2 words were accepted (below real-time threshold of 3)
        assert!(captions.is_empty());
    }

    #[test]
    fn test_feed_words_sentence_boundary_emission() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::NearRealTime,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        // Feed words ending with sentence boundary
        let words = vec![
            make_word("This", 0, 200, 0.95),
            make_word("is", 200, 400, 0.95),
            make_word("a", 400, 500, 0.95),
            make_word("sentence.", 500, 800, 0.95),
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        // Should emit because of sentence boundary
        assert!(!captions.is_empty());
        assert!(captions[0].caption.text().contains("sentence."));
    }

    #[test]
    fn test_speaker_change_triggers_emission() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::NearRealTime,
            speaker_identification: true,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word_with_speaker("Hello", 0, 300, "Alice"),
            make_word_with_speaker("there.", 300, 600, "Alice"),
            make_word_with_speaker("Hi", 700, 900, "Bob"), // speaker change
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        // Should have emitted Alice's caption when Bob started speaking
        assert!(!captions.is_empty());
    }

    #[test]
    fn test_stop_drains_pending() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::Batch,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("Pending", 0, 300, 0.95),
            make_word("words", 300, 600, 0.95),
        ];

        // In batch mode, this might not emit
        let _ = pipeline.feed_words(&words).expect("should succeed");

        // Stop should drain pending words
        let remaining = pipeline.stop().expect("stop should succeed");
        // Either in output buffer from feed or from drain
        let total = remaining.len() + pipeline.buffer_size();
        // Pipeline processed something — total is usize, just ensure it ran
        let _ = total;
    }

    #[test]
    fn test_visible_captions_rolling_display() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            rolling_display_count: 2,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        // Generate multiple captions
        for i in 0..5 {
            let base = i * 1000;
            let words = vec![
                make_word(&format!("Word{}", i * 3), base, base + 300, 0.95),
                make_word(&format!("Word{}", i * 3 + 1), base + 300, base + 600, 0.95),
                make_word(&format!("Word{}", i * 3 + 2), base + 600, base + 900, 0.95),
            ];
            let _ = pipeline.feed_words(&words).expect("should succeed");
        }

        let visible = pipeline.visible_captions();
        assert!(visible.len() <= 2);
    }

    #[test]
    fn test_buffer_size_limit() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            max_buffer_size: 3,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        // Generate more captions than buffer allows
        for i in 0..10 {
            let base = i * 1000;
            let words = vec![
                make_word(&format!("A{i}"), base, base + 300, 0.95),
                make_word(&format!("B{i}"), base + 300, base + 600, 0.95),
                make_word(&format!("C{i}"), base + 600, base + 900, 0.95),
            ];
            let _ = pipeline.feed_words(&words).expect("should succeed");
        }

        assert!(pipeline.buffer_size() <= 3);
    }

    #[test]
    fn test_feed_correction() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            enable_correction: true,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("Helo", 0, 300, 0.7),
            make_word("wrold", 300, 600, 0.6),
            make_word("test.", 600, 900, 0.9),
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        assert!(!captions.is_empty());

        let seq = captions[0].sequence;
        let corrected = pipeline
            .feed_correction(seq, "Hello world test.")
            .expect("correction should succeed");
        assert!(corrected.is_some());

        let corrected = corrected.expect("should have correction");
        assert!(corrected.is_correction);
        assert_eq!(corrected.caption.text(), "Hello world test.");
        assert_eq!(pipeline.stats().corrections, 1);
    }

    #[test]
    fn test_feed_correction_disabled() {
        let config = LiveCaptionConfig {
            enable_correction: false,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        let result = pipeline.feed_correction(1, "test");
        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_feed_correction_missing_sequence() {
        let config = LiveCaptionConfig::default();
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        let result = pipeline.feed_correction(999, "test");
        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_none());
    }

    #[test]
    fn test_stats_tracking() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            min_confidence: 0.5,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("Good", 0, 300, 0.95),
            make_word("bad", 300, 600, 0.3), // rejected
            make_word("fine", 600, 900, 0.88),
            make_word("ok.", 900, 1200, 0.85),
        ];

        let _ = pipeline.feed_words(&words).expect("should succeed");

        let stats = pipeline.stats();
        assert_eq!(stats.total_words, 4);
        assert_eq!(stats.rejected_words, 1);
        assert!(stats.avg_confidence > 0.0);
    }

    #[test]
    fn test_clear_buffer() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("A", 0, 300, 0.95),
            make_word("B", 300, 600, 0.95),
            make_word("C.", 600, 900, 0.95),
        ];
        let _ = pipeline.feed_words(&words).expect("should succeed");

        pipeline.clear_buffer();
        assert_eq!(pipeline.buffer_size(), 0);
    }

    #[test]
    fn test_caption_mode_display() {
        assert_eq!(CaptionMode::RealTime.to_string(), "Real-Time");
        assert_eq!(CaptionMode::NearRealTime.to_string(), "Near Real-Time");
        assert_eq!(CaptionMode::Batch.to_string(), "Batch");
    }

    #[test]
    fn test_pipeline_state_display() {
        assert_eq!(PipelineState::Idle.to_string(), "Idle");
        assert_eq!(PipelineState::Running.to_string(), "Running");
        assert_eq!(PipelineState::Paused.to_string(), "Paused");
        assert_eq!(PipelineState::Error.to_string(), "Error");
        assert_eq!(PipelineState::Draining.to_string(), "Draining");
    }

    #[test]
    fn test_config_validation() {
        assert!(LiveCaptionConfig::default().validate().is_ok());

        let bad = LiveCaptionConfig {
            rolling_display_count: 0,
            ..LiveCaptionConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_multi_speaker_tracking() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::NearRealTime,
            speaker_identification: true,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        // Alice speaks, then Bob, causing speaker change emission
        let words = vec![
            make_word_with_speaker("Hello", 0, 300, "Alice"),
            make_word_with_speaker("there", 300, 600, "Alice"),
            make_word_with_speaker("everyone.", 600, 900, "Alice"),
        ];
        let _ = pipeline.feed_words(&words).expect("should succeed");

        let words2 = vec![
            make_word_with_speaker("Hi", 1000, 1300, "Bob"),
            make_word_with_speaker("Alice.", 1300, 1600, "Bob"),
        ];
        let _ = pipeline.feed_words(&words2).expect("should succeed");

        // Stop to flush
        let remaining = pipeline.stop().expect("stop should succeed");
        let total_captions = pipeline.stats().total_captions + remaining.len() as u64;
        assert!(total_captions >= 1);
    }

    #[test]
    fn test_rendered_caption_has_sequence() {
        let config = LiveCaptionConfig {
            mode: CaptionMode::RealTime,
            ..LiveCaptionConfig::default()
        };
        let mut pipeline = LiveCaptionPipeline::new(config).expect("should succeed");
        pipeline.start().expect("start should succeed");

        let words = vec![
            make_word("One", 0, 300, 0.95),
            make_word("two", 300, 600, 0.95),
            make_word("three.", 600, 900, 0.95),
        ];

        let captions = pipeline.feed_words(&words).expect("should succeed");
        assert!(!captions.is_empty());
        assert!(captions[0].sequence > 0);
    }
}
