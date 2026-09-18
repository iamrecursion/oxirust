//! Multi-language subtitle support with ISO 639-1 validated language codes.
//!
//! This module provides:
//!
//! - [`LanguageCode`] — ISO 639-1 validated 2-letter language code newtype.
//! - [`CaptionEntry`] — A single timed caption entry with text.
//! - [`MultiLangCaption`] — Container for caption tracks in multiple languages.
//! - [`MultiLangCaptionBuilder`] — Builder for constructing `MultiLangCaption`.
//!
//! ## SRT output
//!
//! [`MultiLangCaption::to_srt`] formats a language track as standard SubRip
//! (`.srt`) text, with 1-based sequence numbers and `HH:MM:SS,mmm` timestamps.
//!
//! ## Timing merge
//!
//! [`MultiLangCaption::merge_timing`] aligns a secondary-language track to a
//! primary track by matching overlapping timestamps, returning a merged
//! [`Vec<CaptionEntry>`] whose timing follows the primary track and whose text
//! is taken from the secondary track.

use std::collections::HashMap;

use crate::CaptionGenError;

// ─── Language code ────────────────────────────────────────────────────────────

/// ISO 639-1 language code newtype (two lowercase ASCII letters, e.g. `"en"`).
///
/// Construction always validates the code; use [`LanguageCode::new`] or
/// [`LanguageCode::try_from`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageCode(String);

impl LanguageCode {
    /// Create a validated ISO 639-1 language code.
    ///
    /// The code must be exactly two ASCII lowercase letters (`a-z`).
    ///
    /// # Errors
    ///
    /// Returns [`CaptionGenError::InvalidParameter`] if the code is not exactly
    /// two lowercase ASCII letters.
    pub fn new(code: &str) -> Result<Self, CaptionGenError> {
        let code = code.trim();
        if code.len() != 2 || !code.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(CaptionGenError::InvalidParameter(format!(
                "ISO 639-1 language code must be exactly two lowercase ASCII letters, got {:?}",
                code
            )));
        }
        Ok(Self(code.to_string()))
    }

    /// Return the inner code string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<&str> for LanguageCode {
    type Error = CaptionGenError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for LanguageCode {
    type Error = CaptionGenError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
}

// ─── Caption entry ────────────────────────────────────────────────────────────

/// A single timed caption entry.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionEntry {
    /// 1-based sequence number.
    pub id: u32,
    /// Display start time in milliseconds.
    pub start_ms: u64,
    /// Display end time in milliseconds.
    pub end_ms: u64,
    /// Caption text (may contain newlines for multi-line captions).
    pub text: String,
}

impl CaptionEntry {
    /// Create a new caption entry.
    pub fn new(id: u32, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            id,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration of this entry in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

// ─── Multi-language caption container ────────────────────────────────────────

/// Container for subtitle tracks in multiple languages.
///
/// Each language track is a `Vec<CaptionEntry>` keyed by its [`LanguageCode`].
#[derive(Debug, Clone)]
pub struct MultiLangCaption {
    pub entries: HashMap<LanguageCode, Vec<CaptionEntry>>,
}

impl MultiLangCaption {
    /// Returns the set of language codes present in this container.
    pub fn languages(&self) -> impl Iterator<Item = &LanguageCode> {
        self.entries.keys()
    }

    /// Returns the entries for a given language, or `None` if absent.
    pub fn track(&self, lang: &LanguageCode) -> Option<&[CaptionEntry]> {
        self.entries.get(lang).map(|v| v.as_slice())
    }

    /// Format a language track as SRT (SubRip) text.
    ///
    /// Returns an error if the language is not present in this container.
    pub fn to_srt(&self, lang: &LanguageCode) -> Result<String, CaptionGenError> {
        let track = self.entries.get(lang).ok_or_else(|| {
            CaptionGenError::InvalidParameter(format!(
                "language {:?} not found in MultiLangCaption",
                lang.as_str()
            ))
        })?;

        if track.is_empty() {
            return Ok(String::new());
        }

        let mut out = String::with_capacity(track.len() * 80);
        for (idx, entry) in track.iter().enumerate() {
            let seq = idx as u32 + 1;
            out.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                seq,
                ms_to_srt_timestamp(entry.start_ms),
                ms_to_srt_timestamp(entry.end_ms),
                entry.text
            ));
        }
        Ok(out)
    }

    /// Merge timing from a primary language track onto a secondary track.
    ///
    /// For each entry in the primary track, finds the best-overlapping entry
    /// in the secondary track and adopts the primary's timestamps.  Entries
    /// in the secondary track with no overlap are omitted.
    ///
    /// Returns an error if either language is not present.
    pub fn merge_timing(
        &self,
        primary: &LanguageCode,
        secondary: &LanguageCode,
    ) -> Result<Vec<CaptionEntry>, CaptionGenError> {
        let primary_track = self.entries.get(primary).ok_or_else(|| {
            CaptionGenError::InvalidParameter(format!(
                "primary language {:?} not found",
                primary.as_str()
            ))
        })?;
        let secondary_track = self.entries.get(secondary).ok_or_else(|| {
            CaptionGenError::InvalidParameter(format!(
                "secondary language {:?} not found",
                secondary.as_str()
            ))
        })?;

        let mut merged: Vec<CaptionEntry> = Vec::with_capacity(primary_track.len());

        for (idx, pentry) in primary_track.iter().enumerate() {
            // Find the secondary entry with the greatest temporal overlap.
            let best = secondary_track
                .iter()
                .filter_map(|sentry| {
                    let overlap_start = pentry.start_ms.max(sentry.start_ms);
                    let overlap_end = pentry.end_ms.min(sentry.end_ms);
                    if overlap_end > overlap_start {
                        Some((sentry, overlap_end - overlap_start))
                    } else {
                        None
                    }
                })
                .max_by_key(|(_, overlap)| *overlap)
                .map(|(sentry, _)| sentry);

            if let Some(sentry) = best {
                merged.push(CaptionEntry {
                    id: idx as u32 + 1,
                    start_ms: pentry.start_ms,
                    end_ms: pentry.end_ms,
                    text: sentry.text.clone(),
                });
            }
        }

        Ok(merged)
    }
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Builder for [`MultiLangCaption`].
///
/// ```rust,no_run
/// # use oximedia_caption_gen::multilang::{MultiLangCaptionBuilder, CaptionEntry, LanguageCode};
/// let lang_en = LanguageCode::new("en").unwrap();
/// let entries = vec![CaptionEntry::new(1, 0, 2000, "Hello")];
/// let caption = MultiLangCaptionBuilder::new()
///     .add_track(lang_en, entries)
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct MultiLangCaptionBuilder {
    entries: HashMap<LanguageCode, Vec<CaptionEntry>>,
}

impl MultiLangCaptionBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a caption track for the given language.
    ///
    /// If a track already exists for this language, it is replaced.
    /// Returns `self` by value for method chaining with `.build()`.
    pub fn add_track(mut self, lang: LanguageCode, entries: Vec<CaptionEntry>) -> Self {
        self.entries.insert(lang, entries);
        self
    }

    /// Consume the builder and produce a [`MultiLangCaption`].
    pub fn build(self) -> MultiLangCaption {
        MultiLangCaption {
            entries: self.entries,
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Format milliseconds as SRT timestamp `HH:MM:SS,mmm`.
fn ms_to_srt_timestamp(ms: u64) -> String {
    let total_secs = ms / 1_000;
    let millis = ms % 1_000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, millis)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LanguageCode ──────────────────────────────────────────────────────────

    #[test]
    fn lang_code_valid_en() {
        let code = LanguageCode::new("en").expect("en should be valid");
        assert_eq!(code.as_str(), "en");
    }

    #[test]
    fn lang_code_valid_ja() {
        let code = LanguageCode::new("ja").expect("ja should be valid");
        assert_eq!(code.as_str(), "ja");
    }

    #[test]
    fn lang_code_valid_zh() {
        assert!(LanguageCode::new("zh").is_ok());
    }

    #[test]
    fn lang_code_invalid_empty() {
        assert!(LanguageCode::new("").is_err());
    }

    #[test]
    fn lang_code_invalid_one_letter() {
        assert!(LanguageCode::new("e").is_err());
    }

    #[test]
    fn lang_code_invalid_three_letters() {
        assert!(LanguageCode::new("eng").is_err());
    }

    #[test]
    fn lang_code_invalid_uppercase() {
        assert!(LanguageCode::new("EN").is_err());
    }

    #[test]
    fn lang_code_invalid_digit() {
        assert!(LanguageCode::new("e1").is_err());
    }

    #[test]
    fn lang_code_try_from_str() {
        let code: Result<LanguageCode, _> = "fr".try_into();
        assert!(code.is_ok());
    }

    #[test]
    fn lang_code_display() {
        let code = LanguageCode::new("de").expect("new should succeed");
        assert_eq!(code.to_string(), "de");
    }

    // ── CaptionEntry ──────────────────────────────────────────────────────────

    #[test]
    fn caption_entry_duration() {
        let entry = CaptionEntry::new(1, 1000, 4000, "Hello");
        assert_eq!(entry.duration_ms(), 3000);
    }

    #[test]
    fn caption_entry_duration_zero_on_equal_timestamps() {
        let entry = CaptionEntry::new(1, 2000, 2000, "X");
        assert_eq!(entry.duration_ms(), 0);
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    #[test]
    fn builder_creates_empty_multilang() {
        let caption = MultiLangCaptionBuilder::new().build();
        assert_eq!(caption.entries.len(), 0);
    }

    #[test]
    fn builder_add_track() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let entries = vec![CaptionEntry::new(1, 0, 2000, "Hello")];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), entries)
            .build();
        assert!(caption.track(&en).is_some());
        assert_eq!(caption.track(&en).expect("track should succeed").len(), 1);
    }

    #[test]
    fn builder_add_two_tracks() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let es = LanguageCode::new("es").expect("new should succeed");
        let en_entries = vec![CaptionEntry::new(1, 0, 2000, "Hello")];
        let es_entries = vec![CaptionEntry::new(1, 0, 2000, "Hola")];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), en_entries)
            .add_track(es.clone(), es_entries)
            .build();
        assert!(caption.track(&en).is_some());
        assert!(caption.track(&es).is_some());
    }

    #[test]
    fn builder_add_track_replaces_existing() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let first = vec![CaptionEntry::new(1, 0, 1000, "First")];
        let second = vec![CaptionEntry::new(1, 0, 1000, "Second")];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), first)
            .add_track(en.clone(), second)
            .build();
        assert_eq!(
            caption.track(&en).expect("track should succeed")[0].text,
            "Second"
        );
    }

    // ── to_srt ────────────────────────────────────────────────────────────────

    #[test]
    fn to_srt_basic() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let entries = vec![
            CaptionEntry::new(1, 0, 2000, "Hello"),
            CaptionEntry::new(2, 3000, 5000, "World"),
        ];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), entries)
            .build();
        let srt = caption.to_srt(&en).expect("to srt should succeed");
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
        assert!(srt.contains("00:00:03,000 --> 00:00:05,000"));
        assert!(srt.contains("Hello"));
        assert!(srt.contains("World"));
    }

    #[test]
    fn to_srt_empty_track_returns_empty_string() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), vec![])
            .build();
        let srt = caption.to_srt(&en).expect("to srt should succeed");
        assert!(srt.is_empty());
    }

    #[test]
    fn to_srt_missing_language_returns_error() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let fr = LanguageCode::new("fr").expect("new should succeed");
        let caption = MultiLangCaptionBuilder::new().add_track(en, vec![]).build();
        assert!(caption.to_srt(&fr).is_err());
    }

    #[test]
    fn to_srt_timestamp_format() {
        // Test timestamp formatting: 1 hour, 2 min, 3 sec, 456 ms
        let ms = 1 * 3_600_000 + 2 * 60_000 + 3 * 1_000 + 456;
        let ts = ms_to_srt_timestamp(ms);
        assert_eq!(ts, "01:02:03,456");
    }

    // ── merge_timing ──────────────────────────────────────────────────────────

    #[test]
    fn merge_timing_basic_overlap() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let ja = LanguageCode::new("ja").expect("new should succeed");
        let en_entries = vec![CaptionEntry::new(1, 0, 3000, "Hello")];
        let ja_entries = vec![CaptionEntry::new(1, 500, 3500, "こんにちは")];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), en_entries)
            .add_track(ja.clone(), ja_entries)
            .build();
        let merged = caption
            .merge_timing(&en, &ja)
            .expect("merge timing should succeed");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].start_ms, 0); // primary timing
        assert_eq!(merged[0].end_ms, 3000); // primary timing
        assert_eq!(merged[0].text, "こんにちは"); // secondary text
    }

    #[test]
    fn merge_timing_no_overlap_excluded() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let ja = LanguageCode::new("ja").expect("new should succeed");
        let en_entries = vec![CaptionEntry::new(1, 0, 1000, "Hello")];
        let ja_entries = vec![CaptionEntry::new(1, 5000, 7000, "こんにちは")]; // far away
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), en_entries)
            .add_track(ja.clone(), ja_entries)
            .build();
        let merged = caption
            .merge_timing(&en, &ja)
            .expect("merge timing should succeed");
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_timing_picks_best_overlap() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let es = LanguageCode::new("es").expect("new should succeed");
        let en_entries = vec![CaptionEntry::new(1, 0, 5000, "Long sentence")];
        let es_entries = vec![
            CaptionEntry::new(1, 0, 500, "Short"),   // 500ms overlap
            CaptionEntry::new(2, 0, 4000, "Better"), // 4000ms overlap — wins
        ];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), en_entries)
            .add_track(es.clone(), es_entries)
            .build();
        let merged = caption
            .merge_timing(&en, &es)
            .expect("merge timing should succeed");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text, "Better");
    }

    #[test]
    fn merge_timing_missing_primary_returns_error() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let fr = LanguageCode::new("fr").expect("new should succeed");
        let es = LanguageCode::new("es").expect("new should succeed");
        let caption = MultiLangCaptionBuilder::new().add_track(en, vec![]).build();
        assert!(caption.merge_timing(&fr, &es).is_err());
    }

    #[test]
    fn merge_timing_missing_secondary_returns_error() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let fr = LanguageCode::new("fr").expect("new should succeed");
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), vec![CaptionEntry::new(1, 0, 1000, "X")])
            .build();
        assert!(caption.merge_timing(&en, &fr).is_err());
    }

    #[test]
    fn merge_timing_ids_renumbered() {
        let en = LanguageCode::new("en").expect("new should succeed");
        let de = LanguageCode::new("de").expect("new should succeed");
        let en_entries = vec![
            CaptionEntry::new(1, 0, 1000, "Hello"),
            CaptionEntry::new(2, 2000, 3000, "World"),
        ];
        let de_entries = vec![
            CaptionEntry::new(5, 200, 1200, "Hallo"),
            CaptionEntry::new(6, 2100, 3100, "Welt"),
        ];
        let caption = MultiLangCaptionBuilder::new()
            .add_track(en.clone(), en_entries)
            .add_track(de.clone(), de_entries)
            .build();
        let merged = caption
            .merge_timing(&en, &de)
            .expect("merge timing should succeed");
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 1);
        assert_eq!(merged[1].id, 2);
    }
}
