//! SDH (Subtitles for Deaf and Hard-of-Hearing) generator.
//!
//! This module provides tools for augmenting existing captions with
//! sound descriptions — non-speech audio events such as music, sound effects,
//! laughter and applause — in the format commonly used in broadcast SDH subtitles.
//!
//! Sound descriptions are wrapped in square brackets, e.g. `[MUSIC: dramatic strings]`,
//! and can be positioned at the start or end of a caption line.
//!
//! # Standards compliance
//! The bracket notation follows OFCOM SDH guidelines, BBC subtitle guidelines,
//! and Netflix partner standards for hearing-impaired subtitles.

use std::fmt;

// ── Sound description types ───────────────────────────────────────────────────

/// Describes a non-speech audio event for SDH purposes.
#[derive(Debug, Clone, PartialEq)]
pub enum SoundDescription {
    /// Background or featured music with an optional descriptor.
    ///
    /// `Music("dramatic strings")` → `[MUSIC: dramatic strings]`
    Music(String),
    /// A discrete sound effect with a description.
    ///
    /// `SoundEffect("door slams")` → `[SOUND EFFECT: door slams]`
    SoundEffect(String),
    /// Audience or character laughter.
    Laughter,
    /// Audience or character applause.
    Applause,
    /// A custom free-form sound description.
    ///
    /// `Custom("tense breathing")` → `[tense breathing]`
    Custom(String),
    /// Phone or radio-filtered speech indication.
    FilteredSpeech,
    /// Silence — may be used to indicate an intentional pause of significance.
    Silence,
}

impl SoundDescription {
    /// Render this description as a bracketed SDH string.
    #[must_use]
    pub fn to_bracketed_string(&self) -> String {
        match self {
            Self::Music(desc) => {
                if desc.is_empty() {
                    "[MUSIC]".to_string()
                } else {
                    format!("[MUSIC: {desc}]")
                }
            }
            Self::SoundEffect(desc) => {
                if desc.is_empty() {
                    "[SOUND EFFECT]".to_string()
                } else {
                    format!("[SOUND EFFECT: {desc}]")
                }
            }
            Self::Laughter => "[LAUGHTER]".to_string(),
            Self::Applause => "[APPLAUSE]".to_string(),
            Self::Custom(desc) => format!("[{desc}]"),
            Self::FilteredSpeech => "[FILTERED SPEECH]".to_string(),
            Self::Silence => "[SILENCE]".to_string(),
        }
    }
}

impl fmt::Display for SoundDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_bracketed_string())
    }
}

// ── Position ──────────────────────────────────────────────────────────────────

/// Where the sound description tag should appear relative to the caption text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SdhPosition {
    /// Sound description appears before the caption text on a separate line.
    #[default]
    Before,
    /// Sound description appears after the caption text on a separate line.
    After,
    /// Sound description appears inline at the beginning of the caption text.
    InlineStart,
    /// Sound description appears inline at the end of the caption text.
    InlineEnd,
    /// Sound description replaces the caption text entirely (for caption-less events).
    Replace,
}

// ── SdhTag ────────────────────────────────────────────────────────────────────

/// An SDH annotation: a sound description at a specified position.
#[derive(Debug, Clone)]
pub struct SdhTag {
    /// The sound description to annotate.
    pub sound: SoundDescription,
    /// Where to place the annotation relative to existing text.
    pub position: SdhPosition,
}

impl SdhTag {
    /// Create a new SDH tag.
    #[must_use]
    pub fn new(sound: SoundDescription, position: SdhPosition) -> Self {
        Self { sound, position }
    }

    /// Convenience: create a "before" tag.
    #[must_use]
    pub fn before(sound: SoundDescription) -> Self {
        Self::new(sound, SdhPosition::Before)
    }

    /// Convenience: create an "after" tag.
    #[must_use]
    pub fn after(sound: SoundDescription) -> Self {
        Self::new(sound, SdhPosition::After)
    }

    /// Convenience: create a "replace" tag for sound-only captions.
    #[must_use]
    pub fn replace(sound: SoundDescription) -> Self {
        Self::new(sound, SdhPosition::Replace)
    }

    /// Apply this tag to a caption text string.
    ///
    /// Returns the annotated text according to the configured position.
    /// If the input `text` is empty and position is not `Replace`, the tag
    /// is still appended so the cue contains something displayable.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let tag = self.sound.to_bracketed_string();
        match self.position {
            SdhPosition::Before => {
                if text.is_empty() {
                    tag
                } else {
                    format!("{tag}\n{text}")
                }
            }
            SdhPosition::After => {
                if text.is_empty() {
                    tag
                } else {
                    format!("{text}\n{tag}")
                }
            }
            SdhPosition::InlineStart => {
                if text.is_empty() {
                    tag
                } else {
                    format!("{tag} {text}")
                }
            }
            SdhPosition::InlineEnd => {
                if text.is_empty() {
                    tag
                } else {
                    format!("{text} {tag}")
                }
            }
            SdhPosition::Replace => tag,
        }
    }
}

// ── SdhCaption ────────────────────────────────────────────────────────────────

/// A caption entry augmented with optional SDH tags.
#[derive(Debug, Clone)]
pub struct SdhCaption {
    /// Original caption text (may be empty for sound-only events).
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// SDH tags to apply to this caption.
    pub tags: Vec<SdhTag>,
}

impl SdhCaption {
    /// Create a new SDH caption without tags.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
            tags: Vec::new(),
        }
    }

    /// Add a tag to this caption.
    pub fn add_tag(&mut self, tag: SdhTag) {
        self.tags.push(tag);
    }

    /// Builder-style tag addition.
    #[must_use]
    pub fn with_tag(mut self, tag: SdhTag) -> Self {
        self.tags.push(tag);
        self
    }

    /// Render the final caption text with all SDH tags applied in order.
    ///
    /// Tags are applied sequentially. A `Replace` tag will discard any previous text.
    #[must_use]
    pub fn render(&self) -> String {
        if self.tags.is_empty() {
            return self.text.clone();
        }
        let mut current = self.text.clone();
        for tag in &self.tags {
            current = tag.apply(&current);
        }
        current
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Whether this caption has any SDH annotations.
    #[must_use]
    pub fn has_sdh(&self) -> bool {
        !self.tags.is_empty()
    }
}

// ── SdhGenerator ─────────────────────────────────────────────────────────────

/// Configuration for the SDH generator.
#[derive(Debug, Clone)]
pub struct SdhConfig {
    /// Whether to use upper-case bracket labels (e.g. `[MUSIC]` vs `[music]`).
    pub uppercase_labels: bool,
    /// Maximum characters per line before wrapping sound descriptions to new line.
    pub max_chars_per_line: usize,
    /// Default position for new sound descriptions.
    pub default_position: SdhPosition,
}

impl Default for SdhConfig {
    fn default() -> Self {
        Self {
            uppercase_labels: true,
            max_chars_per_line: 42,
            default_position: SdhPosition::Before,
        }
    }
}

/// Generator that injects SDH sound descriptions into an existing caption sequence.
#[derive(Debug, Clone)]
pub struct SdhGenerator {
    config: SdhConfig,
}

impl SdhGenerator {
    /// Create a new generator with the given configuration.
    #[must_use]
    pub fn new(config: SdhConfig) -> Self {
        Self { config }
    }

    /// Inject a sound description event into a caption sequence at the specified
    /// time position, inserting a new cue if no existing cue overlaps the time,
    /// or annotating the overlapping cue.
    ///
    /// Returns the (potentially extended) list of captions.
    #[must_use]
    pub fn inject(
        &self,
        captions: Vec<SdhCaption>,
        sound: SoundDescription,
        at_ms: u64,
        duration_ms: u64,
    ) -> Vec<SdhCaption> {
        let mut result = captions;
        // Find any caption that overlaps the target time
        let overlapping_idx = result
            .iter()
            .position(|c| c.start_ms <= at_ms && c.end_ms > at_ms);

        match overlapping_idx {
            Some(idx) => {
                result[idx].add_tag(SdhTag::new(sound, self.config.default_position));
            }
            None => {
                // Insert a new sound-only cue
                let end_ms = at_ms.saturating_add(duration_ms).max(at_ms + 1);
                let cue = SdhCaption::new("", at_ms, end_ms).with_tag(SdhTag::replace(sound));
                // Insert at correct position by start time
                let insert_pos = result
                    .iter()
                    .position(|c| c.start_ms > at_ms)
                    .unwrap_or(result.len());
                result.insert(insert_pos, cue);
            }
        }
        result
    }

    /// Annotate all captions in a music segment with music tags.
    ///
    /// All captions whose start time falls within `[music_start_ms, music_end_ms)`
    /// will have a music tag prepended.
    #[must_use]
    pub fn annotate_music_segment(
        &self,
        captions: Vec<SdhCaption>,
        music_start_ms: u64,
        music_end_ms: u64,
        description: &str,
    ) -> Vec<SdhCaption> {
        captions
            .into_iter()
            .map(|mut cap| {
                if cap.start_ms >= music_start_ms && cap.start_ms < music_end_ms {
                    cap.add_tag(SdhTag::new(
                        SoundDescription::Music(description.to_string()),
                        self.config.default_position,
                    ));
                }
                cap
            })
            .collect()
    }

    /// Apply SDH processing to strip any previously applied tags (cleanup pass).
    /// Returns captions with their original text, removing rendered SDH annotations.
    ///
    /// This is a no-op since `SdhCaption` stores tags separately from text.
    #[must_use]
    pub fn strip_sdh(captions: Vec<SdhCaption>) -> Vec<SdhCaption> {
        captions
            .into_iter()
            .map(|mut cap| {
                cap.tags.clear();
                cap
            })
            .collect()
    }

    /// Validate the SDH captions: check that no caption has zero duration
    /// and that all rendered texts are non-empty.
    ///
    /// Returns a list of validation error strings.
    #[must_use]
    pub fn validate(captions: &[SdhCaption]) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, cap) in captions.iter().enumerate() {
            if cap.duration_ms() == 0 {
                errors.push(format!("caption {i} has zero duration"));
            }
            let rendered = cap.render();
            if rendered.trim().is_empty() {
                errors.push(format!("caption {i} renders to empty text"));
            }
        }
        errors
    }
}

impl Default for SdhGenerator {
    fn default() -> Self {
        Self::new(SdhConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SoundDescription tests ───────────────────────────────────────────────

    #[test]
    fn test_sound_description_music_with_label() {
        let s = SoundDescription::Music("upbeat jazz".to_string());
        assert_eq!(s.to_bracketed_string(), "[MUSIC: upbeat jazz]");
    }

    #[test]
    fn test_sound_description_music_empty() {
        let s = SoundDescription::Music(String::new());
        assert_eq!(s.to_bracketed_string(), "[MUSIC]");
    }

    #[test]
    fn test_sound_description_sound_effect() {
        let s = SoundDescription::SoundEffect("door slams".to_string());
        assert_eq!(s.to_bracketed_string(), "[SOUND EFFECT: door slams]");
    }

    #[test]
    fn test_sound_description_laughter() {
        assert_eq!(
            SoundDescription::Laughter.to_bracketed_string(),
            "[LAUGHTER]"
        );
    }

    #[test]
    fn test_sound_description_applause() {
        assert_eq!(
            SoundDescription::Applause.to_bracketed_string(),
            "[APPLAUSE]"
        );
    }

    #[test]
    fn test_sound_description_custom() {
        let s = SoundDescription::Custom("tense breathing".to_string());
        assert_eq!(s.to_bracketed_string(), "[tense breathing]");
    }

    // ── SdhTag::apply tests ──────────────────────────────────────────────────

    #[test]
    fn test_sdh_tag_apply_before() {
        let tag = SdhTag::before(SoundDescription::Music("jazz".to_string()));
        let result = tag.apply("Hello world");
        assert_eq!(result, "[MUSIC: jazz]\nHello world");
    }

    #[test]
    fn test_sdh_tag_apply_after() {
        let tag = SdhTag::after(SoundDescription::Laughter);
        let result = tag.apply("That was funny");
        assert_eq!(result, "That was funny\n[LAUGHTER]");
    }

    #[test]
    fn test_sdh_tag_apply_replace() {
        let tag = SdhTag::replace(SoundDescription::Applause);
        let result = tag.apply("any existing text");
        assert_eq!(result, "[APPLAUSE]");
    }

    #[test]
    fn test_sdh_tag_apply_inline_start() {
        let tag = SdhTag::new(SoundDescription::FilteredSpeech, SdhPosition::InlineStart);
        let result = tag.apply("Hello?");
        assert_eq!(result, "[FILTERED SPEECH] Hello?");
    }

    #[test]
    fn test_sdh_tag_apply_inline_end() {
        let tag = SdhTag::new(SoundDescription::Silence, SdhPosition::InlineEnd);
        let result = tag.apply("…");
        assert_eq!(result, "… [SILENCE]");
    }

    // ── SdhCaption tests ─────────────────────────────────────────────────────

    #[test]
    fn test_sdh_caption_render_no_tags() {
        let cap = SdhCaption::new("Hello world", 0, 2000);
        assert_eq!(cap.render(), "Hello world");
    }

    #[test]
    fn test_sdh_caption_render_with_music() {
        let cap = SdhCaption::new("Opening scene", 0, 5000).with_tag(SdhTag::before(
            SoundDescription::Music("orchestral swell".to_string()),
        ));
        let rendered = cap.render();
        assert!(rendered.contains("[MUSIC: orchestral swell]"));
        assert!(rendered.contains("Opening scene"));
    }

    #[test]
    fn test_sdh_caption_has_sdh_flag() {
        let cap_plain = SdhCaption::new("Hello", 0, 1000);
        let cap_sdh =
            SdhCaption::new("Hello", 0, 1000).with_tag(SdhTag::after(SoundDescription::Laughter));
        assert!(!cap_plain.has_sdh());
        assert!(cap_sdh.has_sdh());
    }

    // ── SdhGenerator tests ───────────────────────────────────────────────────

    #[test]
    fn test_sdh_generator_inject_overlapping() {
        let gen = SdhGenerator::default();
        let captions = vec![SdhCaption::new("Hello", 0, 3000)];
        let result = gen.inject(
            captions,
            SoundDescription::Music("exciting".to_string()),
            1000,
            2000,
        );
        assert_eq!(result.len(), 1);
        assert!(result[0].has_sdh());
    }

    #[test]
    fn test_sdh_generator_inject_new_cue() {
        let gen = SdhGenerator::default();
        let captions = vec![SdhCaption::new("Hello", 0, 1000)];
        let result = gen.inject(captions, SoundDescription::Applause, 2000, 1000);
        // Should have inserted a new cue
        assert_eq!(result.len(), 2);
        let applause_cue = result
            .iter()
            .find(|c| c.start_ms == 2000)
            .expect("cue inserted");
        assert!(applause_cue.render().contains("[APPLAUSE]"));
    }

    #[test]
    fn test_sdh_strip_sdh() {
        let cap = SdhCaption::new("Text", 0, 1000)
            .with_tag(SdhTag::before(SoundDescription::Music("jazz".to_string())));
        let stripped = SdhGenerator::strip_sdh(vec![cap]);
        assert!(!stripped[0].has_sdh());
        assert_eq!(stripped[0].render(), "Text");
    }

    #[test]
    fn test_sdh_validate_zero_duration() {
        let cap = SdhCaption::new("Hello", 1000, 1000); // zero duration
        let errors = SdhGenerator::validate(&[cap]);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("zero duration"));
    }
}
