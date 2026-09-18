//! Transcript generation and export.

pub mod export;
pub mod format;
pub mod generate;
pub mod readability;

pub use export::TranscriptExporter;
pub use format::{TranscriptFormat, TranscriptFormatter};
pub use generate::TranscriptGenerator;
pub use readability::{TargetAudience, TranscriptReadability, TranscriptReadabilityAssessor};

use serde::{Deserialize, Serialize};

/// A transcript with timed entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    /// Transcript entries.
    pub entries: Vec<TranscriptEntry>,
    /// Metadata.
    pub metadata: TranscriptMetadata,
}

/// A single transcript entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// Start time in milliseconds.
    pub start_time_ms: i64,
    /// End time in milliseconds.
    pub end_time_ms: i64,
    /// Speaker name (optional).
    pub speaker: Option<String>,
    /// Transcript text.
    pub text: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Transcript metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptMetadata {
    /// Title.
    pub title: Option<String>,
    /// Language code.
    pub language: String,
    /// Creation timestamp.
    pub created_at: Option<String>,
    /// Total duration in milliseconds.
    pub duration_ms: i64,
}

impl Transcript {
    /// Create a new empty transcript.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry.
    pub fn add_entry(&mut self, entry: TranscriptEntry) {
        self.entries.push(entry);
    }

    /// Get total duration.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.entries.last().map_or(0, |e| e.end_time_ms)
    }

    /// Get text only (no timestamps).
    #[must_use]
    pub fn get_text(&self) -> String {
        self.entries
            .iter()
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl TranscriptEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(start_time_ms: i64, end_time_ms: i64, text: String) -> Self {
        Self {
            start_time_ms,
            end_time_ms,
            speaker: None,
            text,
            confidence: 1.0,
        }
    }

    /// Get duration in milliseconds.
    #[must_use]
    pub const fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.start_time_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_creation() {
        let mut transcript = Transcript::new();
        transcript.add_entry(TranscriptEntry::new(0, 1000, "Hello".to_string()));
        transcript.add_entry(TranscriptEntry::new(1000, 2000, "World".to_string()));

        assert_eq!(transcript.entries.len(), 2);
        assert_eq!(transcript.duration_ms(), 2000);
        assert_eq!(transcript.get_text(), "Hello World");
    }
}
