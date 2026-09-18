//! Subtitle accessibility features.
//!
//! Provides SDH (Subtitles for Deaf and Hard of Hearing) support,
//! reading speed checks, and sound description formatting.

/// Accessibility profile for subtitle rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AccessibilityProfile {
    /// Subtitles for Deaf and Hard of Hearing — includes non-speech sounds.
    SDH,
    /// Hard of Hearing — simplified sound descriptions.
    HoH,
    /// Visually impaired — audio descriptions and enhanced text.
    VisuallyImpaired,
    /// Cognitive accessibility — simplified language and pacing.
    Cognitive,
}

impl AccessibilityProfile {
    /// Human-readable description of this accessibility profile.
    #[allow(dead_code)]
    pub fn description(&self) -> &str {
        match self {
            Self::SDH => "Subtitles for Deaf and Hard of Hearing (includes non-speech sounds)",
            Self::HoH => "Hard of Hearing — simplified sound descriptions",
            Self::VisuallyImpaired => "Visually Impaired — audio descriptions and enhanced text",
            Self::Cognitive => "Cognitive accessibility — simplified language and pacing",
        }
    }
}

/// Categories of non-speech sounds for SDH subtitles.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SoundType {
    /// Music playing in the background or foreground.
    Music,
    /// Audience or crowd applause.
    Applause,
    /// Laughing sounds.
    Laughter,
    /// Door opening or closing sounds.
    Door,
    /// Telephone ringing or notification.
    Phone,
    /// Alarm or siren sounds.
    Alarm,
    /// Thunder or storm sounds.
    Thunder,
    /// Footstep sounds.
    Footsteps,
}

impl SoundType {
    /// Default display label for the sound type.
    #[allow(dead_code)]
    pub fn label(&self) -> &str {
        match self {
            Self::Music => "MUSIC",
            Self::Applause => "APPLAUSE",
            Self::Laughter => "LAUGHTER",
            Self::Door => "DOOR",
            Self::Phone => "PHONE",
            Self::Alarm => "ALARM",
            Self::Thunder => "THUNDER",
            Self::Footsteps => "FOOTSTEPS",
        }
    }
}

/// Description of a detected or manually-coded sound.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SoundDescription {
    /// Type of sound.
    pub sound_type: SoundType,
    /// Human-readable description.
    pub description: String,
    /// Confidence of detection (0.0–1.0).
    pub confidence: f32,
}

impl SoundDescription {
    /// Create a new sound description.
    #[allow(dead_code)]
    pub fn new(sound_type: SoundType, description: impl Into<String>, confidence: f32) -> Self {
        Self {
            sound_type,
            description: description.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// An SDH subtitle event describing a non-speech sound occurrence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SdhEvent {
    /// Start timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Duration of the event in milliseconds.
    pub duration_ms: u64,
    /// Sound description.
    pub sound: SoundDescription,
    /// Optional speaker identification.
    pub speaker: Option<String>,
}

impl SdhEvent {
    /// Create a new SDH event.
    #[allow(dead_code)]
    pub fn new(
        timestamp_ms: u64,
        duration_ms: u64,
        sound: SoundDescription,
        speaker: Option<String>,
    ) -> Self {
        Self {
            timestamp_ms,
            duration_ms,
            sound,
            speaker,
        }
    }
}

/// Formatter for SDH events into subtitle text.
pub struct SdhFormatter;

impl SdhFormatter {
    /// Format an SDH event into a subtitle string.
    ///
    /// Examples:
    /// - Music → `[MUSIC PLAYING]`
    /// - Laughter with speaker → `(John laughing)`
    #[allow(dead_code)]
    pub fn format(event: &SdhEvent) -> String {
        let label = event.sound.sound_type.label();
        let desc = &event.sound.description;

        match event.sound.sound_type {
            SoundType::Music => {
                if desc.is_empty() {
                    format!("[{label} PLAYING]")
                } else {
                    format!("[{label}: {desc}]")
                }
            }
            SoundType::Laughter => {
                if let Some(ref speaker) = event.speaker {
                    format!("({speaker} laughing)")
                } else {
                    format!("({desc})")
                }
            }
            SoundType::Applause => format!("[{label}]"),
            _ => {
                if desc.is_empty() {
                    format!("[{label}]")
                } else {
                    format!("[{desc}]")
                }
            }
        }
    }
}

/// Result of a reading speed check.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReadingSpeedResult {
    /// Characters per second.
    pub chars_per_sec: f32,
    /// Words per minute (approximate: chars / 5 / secs * 60).
    pub words_per_min: f32,
    /// Whether the reading speed exceeds the UK standard (17 chars/sec).
    pub is_too_fast: bool,
    /// Recommended minimum duration in milliseconds.
    pub recommended_duration_ms: u64,
}

/// Utility for checking subtitle reading speed.
pub struct ReadingSpeed;

/// Maximum characters per second — UK standard.
pub const MAX_CHARS_PER_SEC_UK: f32 = 17.0;
/// Maximum characters per second — fast standard.
pub const MAX_CHARS_PER_SEC_FAST: f32 = 25.0;

impl ReadingSpeed {
    /// Check reading speed for a subtitle text and duration.
    ///
    /// Returns a `ReadingSpeedResult` with metrics and recommendations.
    #[allow(dead_code)]
    pub fn check(text: &str, duration_ms: u64) -> ReadingSpeedResult {
        let char_count = text.chars().count() as f32;
        let duration_secs = duration_ms as f32 / 1000.0;

        let chars_per_sec = if duration_secs > 0.0 {
            char_count / duration_secs
        } else {
            f32::INFINITY
        };

        // Approximate words per minute: chars / 5 chars-per-word * 60 secs
        let words_per_min = (char_count / 5.0) / duration_secs * 60.0;

        let is_too_fast = chars_per_sec > MAX_CHARS_PER_SEC_UK;

        // Recommended duration = char_count / max_speed * 1000 ms
        let recommended_duration_ms = ((char_count / MAX_CHARS_PER_SEC_UK) * 1000.0).ceil() as u64;

        ReadingSpeedResult {
            chars_per_sec,
            words_per_min,
            is_too_fast,
            recommended_duration_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessibility_profile_description_not_empty() {
        assert!(!AccessibilityProfile::SDH.description().is_empty());
        assert!(!AccessibilityProfile::HoH.description().is_empty());
        assert!(!AccessibilityProfile::VisuallyImpaired
            .description()
            .is_empty());
        assert!(!AccessibilityProfile::Cognitive.description().is_empty());
    }

    #[test]
    fn test_sound_type_labels() {
        assert_eq!(SoundType::Music.label(), "MUSIC");
        assert_eq!(SoundType::Laughter.label(), "LAUGHTER");
        assert_eq!(SoundType::Alarm.label(), "ALARM");
        assert_eq!(SoundType::Thunder.label(), "THUNDER");
    }

    #[test]
    fn test_sound_description_confidence_clamped() {
        let sd = SoundDescription::new(SoundType::Music, "dramatic", 1.5);
        assert!(sd.confidence <= 1.0);
        let sd2 = SoundDescription::new(SoundType::Door, "slam", -0.5);
        assert!(sd2.confidence >= 0.0);
    }

    #[test]
    fn test_sdh_formatter_music_no_desc() {
        let ev = SdhEvent::new(
            0,
            2000,
            SoundDescription::new(SoundType::Music, "", 1.0),
            None,
        );
        let fmt = SdhFormatter::format(&ev);
        assert_eq!(fmt, "[MUSIC PLAYING]");
    }

    #[test]
    fn test_sdh_formatter_music_with_desc() {
        let ev = SdhEvent::new(
            0,
            2000,
            SoundDescription::new(SoundType::Music, "soft piano", 1.0),
            None,
        );
        let fmt = SdhFormatter::format(&ev);
        assert!(fmt.contains("soft piano"));
    }

    #[test]
    fn test_sdh_formatter_laughter_with_speaker() {
        let ev = SdhEvent::new(
            0,
            1500,
            SoundDescription::new(SoundType::Laughter, "laughing", 0.9),
            Some("Alice".to_string()),
        );
        let fmt = SdhFormatter::format(&ev);
        assert!(fmt.contains("Alice"));
        assert!(fmt.contains("laughing"));
    }

    #[test]
    fn test_sdh_formatter_applause() {
        let ev = SdhEvent::new(
            0,
            3000,
            SoundDescription::new(SoundType::Applause, "", 1.0),
            None,
        );
        let fmt = SdhFormatter::format(&ev);
        assert_eq!(fmt, "[APPLAUSE]");
    }

    #[test]
    fn test_sdh_formatter_alarm_with_desc() {
        let ev = SdhEvent::new(
            0,
            2000,
            SoundDescription::new(SoundType::Alarm, "fire alarm sounding", 0.95),
            None,
        );
        let fmt = SdhFormatter::format(&ev);
        assert!(fmt.contains("fire alarm sounding"));
    }

    #[test]
    fn test_reading_speed_check_slow() {
        // 17 chars over 2 seconds = 8.5 chars/sec — acceptable
        let result = ReadingSpeed::check("Hello, World 17c", 2_000);
        assert!(!result.is_too_fast);
        assert!(result.chars_per_sec < MAX_CHARS_PER_SEC_UK);
    }

    #[test]
    fn test_reading_speed_check_too_fast() {
        // 34 chars over 1 second = 34 chars/sec — too fast
        let result = ReadingSpeed::check("This text is way too fast to read!", 1_000);
        assert!(result.is_too_fast);
    }

    #[test]
    fn test_reading_speed_recommended_duration() {
        // 17 chars → at 17 chars/sec = exactly 1000ms recommended
        let result = ReadingSpeed::check("Hello, World 17c", 500);
        // recommended should be at least 1000ms for 16+ chars at 17 c/s
        assert!(result.recommended_duration_ms >= 900);
    }

    #[test]
    fn test_sdh_event_construction() {
        let ev = SdhEvent::new(
            1000,
            500,
            SoundDescription::new(SoundType::Footsteps, "footsteps on gravel", 0.8),
            Some("Bob".to_string()),
        );
        assert_eq!(ev.timestamp_ms, 1000);
        assert_eq!(ev.duration_ms, 500);
        assert!(ev.speaker.is_some());
    }
}
