//! Transcript formatting.

use crate::transcript::Transcript;

/// Transcript format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptFormat {
    /// Plain text with timestamps.
    Plain,
    /// `WebVTT` format.
    Vtt,
    /// SRT format.
    Srt,
    /// JSON format.
    Json,
}

/// Formats transcripts to different output formats.
pub struct TranscriptFormatter;

impl TranscriptFormatter {
    /// Format transcript to specified format.
    #[must_use]
    pub fn format(transcript: &Transcript, format: TranscriptFormat) -> String {
        match format {
            TranscriptFormat::Plain => Self::to_plain(transcript),
            TranscriptFormat::Vtt => Self::to_vtt(transcript),
            TranscriptFormat::Srt => Self::to_srt(transcript),
            TranscriptFormat::Json => Self::to_json(transcript),
        }
    }

    fn to_plain(transcript: &Transcript) -> String {
        let mut output = String::new();

        for entry in &transcript.entries {
            let timestamp = Self::format_timestamp(entry.start_time_ms);
            let speaker = entry.speaker.as_deref().unwrap_or("Speaker");
            output.push_str(&format!("[{}] {}: {}\n", timestamp, speaker, entry.text));
        }

        output
    }

    fn to_vtt(transcript: &Transcript) -> String {
        let mut output = String::from("WEBVTT\n\n");

        for entry in &transcript.entries {
            let start = Self::format_vtt_time(entry.start_time_ms);
            let end = Self::format_vtt_time(entry.end_time_ms);
            output.push_str(&format!("{} --> {}\n{}\n\n", start, end, entry.text));
        }

        output
    }

    fn to_srt(transcript: &Transcript) -> String {
        let mut output = String::new();

        for (i, entry) in transcript.entries.iter().enumerate() {
            let start = Self::format_srt_time(entry.start_time_ms);
            let end = Self::format_srt_time(entry.end_time_ms);
            output.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                i + 1,
                start,
                end,
                entry.text
            ));
        }

        output
    }

    fn to_json(transcript: &Transcript) -> String {
        serde_json::to_string_pretty(transcript).unwrap_or_default()
    }

    fn format_timestamp(ms: i64) -> String {
        let seconds = ms / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        format!("{:02}:{:02}:{:02}", hours, minutes % 60, seconds % 60)
    }

    fn format_vtt_time(ms: i64) -> String {
        let seconds = ms / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        let millis = ms % 1000;
        format!(
            "{:02}:{:02}:{:02}.{:03}",
            hours,
            minutes % 60,
            seconds % 60,
            millis
        )
    }

    fn format_srt_time(ms: i64) -> String {
        let seconds = ms / 1000;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        let millis = ms % 1000;
        format!(
            "{:02}:{:02}:{:02},{:03}",
            hours,
            minutes % 60,
            seconds % 60,
            millis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::TranscriptEntry;

    #[test]
    fn test_format_plain() {
        let mut transcript = Transcript::new();
        transcript.add_entry(TranscriptEntry::new(0, 1000, "Test".to_string()));

        let output = TranscriptFormatter::format(&transcript, TranscriptFormat::Plain);
        assert!(output.contains("Test"));
    }

    #[test]
    fn test_format_vtt() {
        let mut transcript = Transcript::new();
        transcript.add_entry(TranscriptEntry::new(0, 1000, "Test".to_string()));

        let output = TranscriptFormatter::format(&transcript, TranscriptFormat::Vtt);
        assert!(output.starts_with("WEBVTT"));
    }
}
