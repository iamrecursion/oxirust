//! Accessibility features: reading level analysis, speaker identification,
//! and sound description generation.

#![allow(dead_code)]
#![allow(missing_docs)]

use std::collections::HashMap;

// ── Reading level analysis ────────────────────────────────────────────────────

/// Reading difficulty level (simplified Flesch-Kincaid tiers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReadingLevel {
    /// Very easy (grade 1–3)
    Elementary,
    /// Easy (grade 4–6)
    Intermediate,
    /// Moderate (grade 7–9)
    MiddleSchool,
    /// Difficult (grade 10–12)
    HighSchool,
    /// Professional / academic
    Advanced,
}

impl ReadingLevel {
    /// Return a human-readable label
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Elementary => "Elementary",
            Self::Intermediate => "Intermediate",
            Self::MiddleSchool => "Middle School",
            Self::HighSchool => "High School",
            Self::Advanced => "Advanced",
        }
    }
}

/// Metrics extracted from a block of caption text
#[derive(Debug, Clone)]
pub struct ReadabilityMetrics {
    pub word_count: usize,
    pub sentence_count: usize,
    pub syllable_count: usize,
    pub avg_words_per_sentence: f32,
    pub avg_syllables_per_word: f32,
    pub flesch_reading_ease: f32,
    pub level: ReadingLevel,
}

/// Count syllables in a single word (heuristic: vowel groups)
fn count_syllables(word: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y'];
    let mut count = 0usize;
    let mut prev_vowel = false;
    for ch in word.chars() {
        if vowels.contains(&ch) {
            if !prev_vowel {
                count += 1;
            }
            prev_vowel = true;
        } else {
            prev_vowel = false;
        }
    }
    count.max(1)
}

/// Analyse readability of a piece of text
#[allow(clippy::cast_precision_loss)]
pub fn analyse_readability(text: &str) -> ReadabilityMetrics {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    let sentence_count = sentences.len().max(1);

    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|w| !w.is_empty())
        .collect();

    let word_count = words.len().max(1);
    let syllable_count: usize = words.iter().map(|w| count_syllables(w)).sum();

    let asl = word_count as f32 / sentence_count as f32;
    let asw = syllable_count as f32 / word_count as f32;
    let fre = 206.835 - 1.015 * asl - 84.6 * asw;

    let level = match fre as i32 {
        90..=200 => ReadingLevel::Elementary,
        70..=89 => ReadingLevel::Intermediate,
        50..=69 => ReadingLevel::MiddleSchool,
        30..=49 => ReadingLevel::HighSchool,
        _ => ReadingLevel::Advanced,
    };

    ReadabilityMetrics {
        word_count,
        sentence_count,
        syllable_count,
        avg_words_per_sentence: asl,
        avg_syllables_per_word: asw,
        flesch_reading_ease: fre.clamp(-100.0, 200.0),
        level,
    }
}

// ── Speaker identification ────────────────────────────────────────────────────

/// Unique speaker identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpeakerId(pub String);

impl SpeakerId {
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl std::fmt::Display for SpeakerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata for a known speaker
#[derive(Debug, Clone)]
pub struct SpeakerProfile {
    pub id: SpeakerId,
    pub display_name: String,
    /// Caption label used in brackets, e.g. "\[JOHN\]"
    pub caption_label: String,
    /// Typical speech rate in words-per-minute (informational)
    pub wpm: Option<u32>,
}

impl SpeakerProfile {
    #[must_use]
    pub fn new(id: &str, display_name: &str) -> Self {
        let label = format!("[{}]", display_name.to_uppercase());
        Self {
            id: SpeakerId::new(id),
            display_name: display_name.to_string(),
            caption_label: label,
            wpm: None,
        }
    }

    #[must_use]
    pub fn with_wpm(mut self, wpm: u32) -> Self {
        self.wpm = Some(wpm);
        self
    }
}

/// A caption segment attributed to a speaker
#[derive(Debug, Clone)]
pub struct AttributedSegment {
    pub text: String,
    pub speaker_id: Option<SpeakerId>,
    pub begin_ms: u64,
    pub end_ms: u64,
}

impl AttributedSegment {
    #[must_use]
    pub fn new(text: &str, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.to_string(),
            speaker_id: None,
            begin_ms,
            end_ms,
        }
    }

    /// Apply a speaker label prefix to the text
    #[must_use]
    pub fn labelled_text(&self, registry: &SpeakerRegistry) -> String {
        match &self.speaker_id {
            Some(id) => {
                if let Some(profile) = registry.get(id) {
                    format!("{} {}", profile.caption_label, self.text)
                } else {
                    self.text.clone()
                }
            }
            None => self.text.clone(),
        }
    }
}

/// Registry mapping speaker ids to profiles
#[derive(Debug, Default)]
pub struct SpeakerRegistry {
    profiles: HashMap<String, SpeakerProfile>,
}

impl SpeakerRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, profile: SpeakerProfile) {
        self.profiles.insert(profile.id.0.clone(), profile);
    }

    #[must_use]
    pub fn get(&self, id: &SpeakerId) -> Option<&SpeakerProfile> {
        self.profiles.get(&id.0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ── Sound description ─────────────────────────────────────────────────────────

/// Category of a sound description
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundCategory {
    Music,
    Effect,
    Ambient,
    Silence,
    Laughter,
    Applause,
    Alert,
    Other,
}

impl SoundCategory {
    #[must_use]
    pub fn default_label(self) -> &'static str {
        match self {
            Self::Music => "[MUSIC]",
            Self::Effect => "[SOUND EFFECT]",
            Self::Ambient => "[AMBIENT NOISE]",
            Self::Silence => "[SILENCE]",
            Self::Laughter => "[LAUGHTER]",
            Self::Applause => "[APPLAUSE]",
            Self::Alert => "[ALERT]",
            Self::Other => "[SOUND]",
        }
    }
}

/// A non-speech audio event annotated for accessibility
#[derive(Debug, Clone)]
pub struct SoundDescription {
    pub category: SoundCategory,
    /// Human-readable description (may be more specific than the label)
    pub description: String,
    pub begin_ms: u64,
    pub end_ms: u64,
    /// Loudness estimate (0–100, informational)
    pub loudness_estimate: Option<u8>,
}

impl SoundDescription {
    #[must_use]
    pub fn new(category: SoundCategory, description: &str, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            category,
            description: description.to_string(),
            begin_ms,
            end_ms,
            loudness_estimate: None,
        }
    }

    /// Format for insertion into caption text
    #[must_use]
    pub fn caption_text(&self) -> String {
        format!("{} {}", self.category.default_label(), self.description)
    }

    /// Duration of this sound event
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.begin_ms)
    }
}

/// Manage a track of sound descriptions for a piece of content
#[derive(Debug, Default)]
pub struct SoundDescriptionTrack {
    events: Vec<SoundDescription>,
}

impl SoundDescriptionTrack {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, event: SoundDescription) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.begin_ms);
    }

    /// Return all events active at the given time
    #[must_use]
    pub fn active_at(&self, time_ms: u64) -> Vec<&SoundDescription> {
        self.events
            .iter()
            .filter(|e| time_ms >= e.begin_ms && time_ms < e.end_ms)
            .collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Find events of a specific category
    #[must_use]
    pub fn by_category(&self, cat: SoundCategory) -> Vec<&SoundDescription> {
        self.events.iter().filter(|e| e.category == cat).collect()
    }
}

// ── Accessibility checker ─────────────────────────────────────────────────────

/// An accessibility issue found in a caption track
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessibilityIssue {
    /// Caption rate exceeds viewer comfort threshold (words-per-minute)
    ExcessiveCaptionRate { segment_idx: usize, actual_wpm: u32 },
    /// Text is too complex for the expected reading level
    ComplexLanguage { segment_idx: usize },
    /// Caption window is too short for the text length
    TooShort {
        segment_idx: usize,
        min_ms: u64,
        actual_ms: u64,
    },
    /// Speaker not identified when multiple speakers are present
    UnidentifiedSpeaker { segment_idx: usize },
}

/// Configuration for the accessibility checker
#[derive(Debug, Clone)]
pub struct AccessibilityConfig {
    /// Maximum acceptable words-per-minute for reading captions
    pub max_caption_wpm: u32,
    /// Minimum display time per caption in milliseconds
    pub min_display_ms: u64,
    /// Expected maximum reading level for the target audience
    pub max_reading_level: ReadingLevel,
    /// Whether multi-speaker content requires speaker attribution
    pub require_speaker_attribution: bool,
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            max_caption_wpm: 180,
            min_display_ms: 1000,
            max_reading_level: ReadingLevel::Intermediate,
            require_speaker_attribution: false,
        }
    }
}

/// Run accessibility checks on a list of attributed segments
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn check_accessibility(
    segments: &[AttributedSegment],
    config: &AccessibilityConfig,
) -> Vec<AccessibilityIssue> {
    let mut issues = Vec::new();

    for (i, seg) in segments.iter().enumerate() {
        let duration_secs = seg.end_ms.saturating_sub(seg.begin_ms) as f32 / 1000.0;
        let word_count = seg.text.split_whitespace().count();

        // Duration check
        let actual_ms = seg.end_ms.saturating_sub(seg.begin_ms);
        if actual_ms < config.min_display_ms {
            issues.push(AccessibilityIssue::TooShort {
                segment_idx: i,
                min_ms: config.min_display_ms,
                actual_ms,
            });
        }

        // Caption rate check
        if duration_secs > 0.0 {
            let wpm = (word_count as f32 / duration_secs * 60.0) as u32;
            if wpm > config.max_caption_wpm {
                issues.push(AccessibilityIssue::ExcessiveCaptionRate {
                    segment_idx: i,
                    actual_wpm: wpm,
                });
            }
        }

        // Reading level check
        let metrics = analyse_readability(&seg.text);
        if metrics.level > config.max_reading_level {
            issues.push(AccessibilityIssue::ComplexLanguage { segment_idx: i });
        }

        // Speaker attribution check
        if config.require_speaker_attribution && seg.speaker_id.is_none() {
            issues.push(AccessibilityIssue::UnidentifiedSpeaker { segment_idx: i });
        }
    }

    issues
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_syllables_basic() {
        assert_eq!(count_syllables("cat"), 1);
        assert_eq!(count_syllables("hello"), 2);
        assert_eq!(count_syllables("beautiful"), 3);
    }

    #[test]
    fn test_count_syllables_minimum_one() {
        // Even a word with no detected vowels gets at least 1
        assert!(count_syllables("gym") >= 1);
    }

    #[test]
    fn test_analyse_readability_word_count() {
        let metrics = analyse_readability("The cat sat on the mat.");
        assert_eq!(metrics.word_count, 6);
    }

    #[test]
    fn test_analyse_readability_sentence_count() {
        let metrics = analyse_readability("Hello! World. How are you?");
        assert_eq!(metrics.sentence_count, 3);
    }

    #[test]
    fn test_reading_level_ordering() {
        assert!(ReadingLevel::Elementary < ReadingLevel::Advanced);
    }

    #[test]
    fn test_reading_level_label() {
        assert_eq!(ReadingLevel::Intermediate.label(), "Intermediate");
    }

    #[test]
    fn test_speaker_profile_label() {
        let p = SpeakerProfile::new("spk1", "Alice");
        assert_eq!(p.caption_label, "[ALICE]");
    }

    #[test]
    fn test_speaker_registry_register_get() {
        let mut reg = SpeakerRegistry::new();
        reg.register(SpeakerProfile::new("s1", "Bob"));
        assert!(reg.get(&SpeakerId::new("s1")).is_some());
        assert!(reg.get(&SpeakerId::new("unknown")).is_none());
    }

    #[test]
    fn test_attributed_segment_labelled_text() {
        let mut reg = SpeakerRegistry::new();
        reg.register(SpeakerProfile::new("s1", "Alice"));
        let mut seg = AttributedSegment::new("Good morning.", 0, 2000);
        seg.speaker_id = Some(SpeakerId::new("s1"));
        let text = seg.labelled_text(&reg);
        assert!(text.starts_with("[ALICE]"));
    }

    #[test]
    fn test_attributed_segment_no_speaker() {
        let reg = SpeakerRegistry::new();
        let seg = AttributedSegment::new("Good morning.", 0, 2000);
        assert_eq!(seg.labelled_text(&reg), "Good morning.");
    }

    #[test]
    fn test_sound_category_label() {
        assert_eq!(SoundCategory::Laughter.default_label(), "[LAUGHTER]");
    }

    #[test]
    fn test_sound_description_caption_text() {
        let sd = SoundDescription::new(SoundCategory::Music, "upbeat jazz", 0, 5000);
        assert!(sd.caption_text().contains("[MUSIC]"));
        assert!(sd.caption_text().contains("upbeat jazz"));
    }

    #[test]
    fn test_sound_description_duration() {
        let sd = SoundDescription::new(SoundCategory::Effect, "door slam", 1000, 3500);
        assert_eq!(sd.duration_ms(), 2500);
    }

    #[test]
    fn test_sound_description_track_active_at() {
        let mut track = SoundDescriptionTrack::new();
        track.add(SoundDescription::new(
            SoundCategory::Music,
            "theme",
            0,
            10000,
        ));
        track.add(SoundDescription::new(
            SoundCategory::Applause,
            "clap",
            5000,
            7000,
        ));
        assert_eq!(track.active_at(6000).len(), 2);
        assert_eq!(track.active_at(8000).len(), 1);
    }

    #[test]
    fn test_sound_description_track_by_category() {
        let mut track = SoundDescriptionTrack::new();
        track.add(SoundDescription::new(
            SoundCategory::Music,
            "intro",
            0,
            5000,
        ));
        track.add(SoundDescription::new(
            SoundCategory::Laughter,
            "laugh",
            2000,
            3000,
        ));
        assert_eq!(track.by_category(SoundCategory::Music).len(), 1);
    }

    #[test]
    fn test_accessibility_check_too_short() {
        let segs = vec![AttributedSegment {
            text: "Hi.".to_string(),
            speaker_id: None,
            begin_ms: 0,
            end_ms: 200, // 200ms < 1000ms minimum
        }];
        let config = AccessibilityConfig::default();
        let issues = check_accessibility(&segs, &config);
        assert!(issues
            .iter()
            .any(|i| matches!(i, AccessibilityIssue::TooShort { .. })));
    }

    #[test]
    fn test_accessibility_check_speaker_attribution() {
        let segs = vec![AttributedSegment {
            text: "Welcome to the show.".to_string(),
            speaker_id: None,
            begin_ms: 0,
            end_ms: 3000,
        }];
        let config = AccessibilityConfig {
            require_speaker_attribution: true,
            ..Default::default()
        };
        let issues = check_accessibility(&segs, &config);
        assert!(issues
            .iter()
            .any(|i| matches!(i, AccessibilityIssue::UnidentifiedSpeaker { .. })));
    }

    #[test]
    fn test_accessibility_check_no_issues_simple() {
        let segs = vec![AttributedSegment {
            text: "The cat sat.".to_string(),
            speaker_id: None,
            begin_ms: 0,
            end_ms: 4000,
        }];
        let config = AccessibilityConfig::default();
        let issues = check_accessibility(&segs, &config);
        // Should have no excessive-rate or too-short issues
        let rate_issues: Vec<_> = issues
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    AccessibilityIssue::ExcessiveCaptionRate { .. }
                        | AccessibilityIssue::TooShort { .. }
                )
            })
            .collect();
        assert!(rate_issues.is_empty());
    }
}
