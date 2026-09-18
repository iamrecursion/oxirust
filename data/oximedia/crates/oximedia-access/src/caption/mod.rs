//! Closed caption generation and styling.
//!
//! Provides advanced closed caption generation with smart positioning,
//! styling, and synchronization.

pub mod generate;
pub mod position;
pub mod style;
pub mod sync;

pub use generate::{
    CaptionConfig, CaptionGenerator, MultiSpeakerConfig, SpeakerColor, SpeakerLabelFormat,
    SpeakerTracker, SpeechSegment,
};
pub use position::{CaptionPosition, CaptionPositioner};
pub use style::{CaptionStyle, CaptionStylePreset};
pub use sync::{CaptionSynchronizer, SyncAdjustment, SyncDriftAnalysis, SyncQuality};

use oximedia_subtitle::Subtitle;
use serde::{Deserialize, Serialize};

/// Caption type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionType {
    /// Closed captions (CC) - can be toggled on/off.
    Closed,
    /// Open captions - always visible (burned in).
    Open,
    /// Subtitles for translation.
    Subtitle,
}

/// Caption format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionFormat {
    /// CEA-608 closed captions.
    Cea608,
    /// CEA-708 closed captions.
    Cea708,
    /// `WebVTT` format.
    WebVtt,
    /// `SubRip` (SRT) format.
    Srt,
    /// Advanced `SubStation` Alpha (ASS).
    Ass,
}

/// Caption quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionQuality {
    /// Basic quality - automated generation.
    Basic,
    /// Standard quality - automated with corrections.
    Standard,
    /// High quality - human reviewed.
    High,
    /// Professional quality - professional captioner.
    Professional,
}

/// A generated caption with metadata.
#[derive(Debug, Clone)]
pub struct Caption {
    /// Underlying subtitle data.
    pub subtitle: Subtitle,
    /// Caption type.
    pub caption_type: CaptionType,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Speaker identification.
    pub speaker: Option<String>,
}

impl Caption {
    /// Create a new caption.
    #[must_use]
    pub fn new(subtitle: Subtitle, caption_type: CaptionType) -> Self {
        Self {
            subtitle,
            caption_type,
            confidence: 1.0,
            speaker: None,
        }
    }

    /// Set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set speaker.
    #[must_use]
    pub fn with_speaker(mut self, speaker: String) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Get start time in milliseconds.
    #[must_use]
    pub const fn start_time(&self) -> i64 {
        self.subtitle.start_time
    }

    /// Get end time in milliseconds.
    #[must_use]
    pub const fn end_time(&self) -> i64 {
        self.subtitle.end_time
    }

    /// Get caption text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.subtitle.text
    }

    /// Get duration in milliseconds.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.subtitle.duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_creation() {
        let subtitle = Subtitle::new(1000, 3000, "Test caption".to_string());
        let caption = Caption::new(subtitle, CaptionType::Closed);

        assert_eq!(caption.start_time(), 1000);
        assert_eq!(caption.end_time(), 3000);
        assert_eq!(caption.text(), "Test caption");
        assert_eq!(caption.duration(), 2000);
    }

    #[test]
    fn test_caption_confidence() {
        let subtitle = Subtitle::new(1000, 2000, "Test".to_string());
        let caption = Caption::new(subtitle, CaptionType::Closed).with_confidence(0.95);

        assert!((caption.confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_caption_speaker() {
        let subtitle = Subtitle::new(1000, 2000, "Hello".to_string());
        let caption = Caption::new(subtitle, CaptionType::Closed).with_speaker("John".to_string());

        assert_eq!(caption.speaker.as_deref(), Some("John"));
    }
}
