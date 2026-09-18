//! Chapter metadata support for media files.
//!
//! This module provides parsing and construction of chapter metadata across
//! multiple container formats:
//!
//! - **Matroska/WebM** chapters (ChapterAtom hierarchy)
//! - **MP4/M4A** chapters (timed text / `chpl` atom)
//! - **ID3v2 CHAP** frames (chapter markers in MP3)
//!
//! # Overview
//!
//! A chapter is a named time range within a media file.  Chapters can be
//! nested (e.g., a part containing several scenes) and may carry additional
//! metadata such as language, country, and artwork.
//!
//! The [`ChapterList`] type is the primary entry point.  It stores an ordered
//! list of [`Chapter`] entries and provides serialization to/from the
//! generic `Metadata` container.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::chapter::{Chapter, ChapterList};
//!
//! let mut chapters = ChapterList::new();
//! chapters.add(Chapter::new(0, 120_000, "Introduction"));
//! chapters.add(Chapter::new(120_000, 360_000, "Main Content"));
//! chapters.add(Chapter::new(360_000, 480_000, "Conclusion"));
//!
//! assert_eq!(chapters.len(), 3);
//! assert_eq!(chapters.total_duration_ms(), Some(480_000));
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;

/// A single chapter entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// Start time in milliseconds from the beginning of the media.
    pub start_ms: u64,
    /// End time in milliseconds (exclusive).
    pub end_ms: u64,
    /// Chapter title / display name.
    pub title: String,
    /// Optional language code (ISO 639-2, e.g., "eng").
    pub language: Option<String>,
    /// Optional country code (ISO 3166-1 alpha-2).
    pub country: Option<String>,
    /// Unique identifier for this chapter (used by ID3v2 CHAP / Matroska).
    pub uid: Option<String>,
    /// Nested sub-chapters.
    pub children: Vec<Chapter>,
    /// Arbitrary key-value attributes (format-specific extras).
    pub attributes: HashMap<String, String>,
}

impl Chapter {
    /// Create a new chapter with start/end times and a title.
    pub fn new(start_ms: u64, end_ms: u64, title: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            title: title.into(),
            language: None,
            country: None,
            uid: None,
            children: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Set the language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the country.
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the UID.
    pub fn with_uid(mut self, uid: impl Into<String>) -> Self {
        self.uid = Some(uid.into());
        self
    }

    /// Add a nested sub-chapter.
    pub fn add_child(&mut self, child: Chapter) {
        self.children.push(child);
    }

    /// Set an extra attribute.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Duration of this chapter in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns true if this chapter contains nested children.
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Total number of chapters including all descendants.
    pub fn total_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_count()).sum::<usize>()
    }

    /// Format start time as HH:MM:SS.mmm string.
    pub fn start_time_formatted(&self) -> String {
        format_time_ms(self.start_ms)
    }

    /// Format end time as HH:MM:SS.mmm string.
    pub fn end_time_formatted(&self) -> String {
        format_time_ms(self.end_ms)
    }

    /// Validate that the chapter's time range is well-formed.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.start_ms > self.end_ms {
            issues.push(format!(
                "Chapter '{}': start ({}) > end ({})",
                self.title, self.start_ms, self.end_ms
            ));
        }

        if self.title.is_empty() {
            issues.push("Chapter has empty title".to_string());
        }

        // Validate children time ranges are within parent bounds
        for child in &self.children {
            if child.start_ms < self.start_ms {
                issues.push(format!(
                    "Sub-chapter '{}' starts ({}) before parent '{}' ({})",
                    child.title, child.start_ms, self.title, self.start_ms
                ));
            }
            if child.end_ms > self.end_ms {
                issues.push(format!(
                    "Sub-chapter '{}' ends ({}) after parent '{}' ({})",
                    child.title, child.end_ms, self.title, self.end_ms
                ));
            }
            issues.extend(child.validate());
        }

        issues
    }
}

/// An ordered list of chapter entries.
#[derive(Debug, Clone, Default)]
pub struct ChapterList {
    /// Top-level chapters (may each contain nested sub-chapters).
    chapters: Vec<Chapter>,
    /// Edition UID (Matroska concept, optional).
    edition_uid: Option<String>,
    /// If true, this chapter edition is the default.
    is_default: bool,
    /// If true, chapter edition is hidden from the user.
    is_hidden: bool,
}

impl ChapterList {
    /// Create an empty chapter list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a top-level chapter.
    pub fn add(&mut self, chapter: Chapter) {
        self.chapters.push(chapter);
    }

    /// Insert a chapter at a specific index.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds.
    pub fn insert(&mut self, index: usize, chapter: Chapter) -> Result<(), Error> {
        if index > self.chapters.len() {
            return Err(Error::ParseError(format!(
                "Chapter index {index} out of bounds (len={})",
                self.chapters.len()
            )));
        }
        self.chapters.insert(index, chapter);
        Ok(())
    }

    /// Remove a chapter by index, returning it.
    pub fn remove(&mut self, index: usize) -> Option<Chapter> {
        if index < self.chapters.len() {
            Some(self.chapters.remove(index))
        } else {
            None
        }
    }

    /// Get a chapter by index.
    pub fn get(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }

    /// Get a mutable chapter by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Chapter> {
        self.chapters.get_mut(index)
    }

    /// Number of top-level chapters.
    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    /// Returns true if there are no chapters.
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Total number of chapters including all nested sub-chapters.
    pub fn total_count(&self) -> usize {
        self.chapters.iter().map(|c| c.total_count()).sum()
    }

    /// Get all top-level chapters.
    pub fn chapters(&self) -> &[Chapter] {
        &self.chapters
    }

    /// Total duration in milliseconds (end of last chapter).
    pub fn total_duration_ms(&self) -> Option<u64> {
        self.chapters.iter().map(|c| c.end_ms).max()
    }

    /// Set the edition UID.
    pub fn set_edition_uid(&mut self, uid: impl Into<String>) {
        self.edition_uid = Some(uid.into());
    }

    /// Get the edition UID.
    pub fn edition_uid(&self) -> Option<&str> {
        self.edition_uid.as_deref()
    }

    /// Set whether this is the default edition.
    pub fn set_default(&mut self, is_default: bool) {
        self.is_default = is_default;
    }

    /// Whether this is the default edition.
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Set whether this edition is hidden.
    pub fn set_hidden(&mut self, is_hidden: bool) {
        self.is_hidden = is_hidden;
    }

    /// Whether this edition is hidden.
    pub fn is_hidden(&self) -> bool {
        self.is_hidden
    }

    /// Sort chapters by start time.
    pub fn sort_by_time(&mut self) {
        self.chapters.sort_by_key(|c| c.start_ms);
        for ch in &mut self.chapters {
            sort_children_recursive(ch);
        }
    }

    /// Find the chapter containing a given timestamp (in ms).
    ///
    /// Returns the most specific (deepest nested) chapter that contains the
    /// given time.
    pub fn chapter_at(&self, time_ms: u64) -> Option<&Chapter> {
        for ch in &self.chapters {
            if let Some(found) = find_chapter_at(ch, time_ms) {
                return Some(found);
            }
        }
        None
    }

    /// Validate the entire chapter list for consistency.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for overlapping chapters
        let mut sorted: Vec<&Chapter> = self.chapters.iter().collect();
        sorted.sort_by_key(|c| c.start_ms);

        for window in sorted.windows(2) {
            if window[0].end_ms > window[1].start_ms {
                issues.push(format!(
                    "Chapters '{}' and '{}' overlap",
                    window[0].title, window[1].title
                ));
            }
        }

        // Validate each chapter
        for ch in &self.chapters {
            issues.extend(ch.validate());
        }

        issues
    }

    /// Serialize chapter list into a `Metadata` container.
    ///
    /// Fields are stored as numbered keys:
    /// - `chapter_count` = number of chapters
    /// - `chapter_N_start` = start time in ms
    /// - `chapter_N_end` = end time in ms
    /// - `chapter_N_title` = chapter title
    /// - `chapter_N_language` = language (if set)
    pub fn to_metadata(&self, format: MetadataFormat) -> Metadata {
        let mut metadata = Metadata::new(format);

        metadata.insert(
            "chapter_count".to_string(),
            MetadataValue::Integer(self.chapters.len() as i64),
        );

        if let Some(ref uid) = self.edition_uid {
            metadata.insert(
                "chapter_edition_uid".to_string(),
                MetadataValue::Text(uid.clone()),
            );
        }

        for (i, ch) in self.chapters.iter().enumerate() {
            serialize_chapter(&mut metadata, ch, &format!("chapter_{i}"));
        }

        metadata
    }

    /// Deserialize a chapter list from a `Metadata` container.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata lacks required fields.
    pub fn from_metadata(metadata: &Metadata) -> Result<Self, Error> {
        let count = metadata
            .get("chapter_count")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as usize;

        let mut list = ChapterList::new();

        if let Some(uid) = metadata
            .get("chapter_edition_uid")
            .and_then(|v| v.as_text())
        {
            list.set_edition_uid(uid);
        }

        for i in 0..count {
            let prefix = format!("chapter_{i}");
            let ch = deserialize_chapter(metadata, &prefix)?;
            list.add(ch);
        }

        Ok(list)
    }

    /// Generate an ID3v2 CHAP-style representation.
    ///
    /// Returns key-value pairs suitable for embedding in ID3v2 metadata.
    pub fn to_id3v2_chap_fields(&self) -> Vec<(String, MetadataValue)> {
        let mut fields = Vec::new();

        // ID3v2 CTOC (Table of Contents)
        let toc_entries: Vec<String> = (0..self.chapters.len())
            .map(|i| format!("chp{i}"))
            .collect();
        fields.push(("CTOC".to_string(), MetadataValue::TextList(toc_entries)));

        // Each CHAP frame
        for (i, ch) in self.chapters.iter().enumerate() {
            let element_id = format!("chp{i}");
            // ID3v2 CHAP stores times in milliseconds as 32-bit integers
            let start = ch.start_ms.min(u32::MAX as u64) as i64;
            let end = ch.end_ms.min(u32::MAX as u64) as i64;

            fields.push((
                format!("CHAP:{element_id}:start"),
                MetadataValue::Integer(start),
            ));
            fields.push((
                format!("CHAP:{element_id}:end"),
                MetadataValue::Integer(end),
            ));
            fields.push((
                format!("CHAP:{element_id}:title"),
                MetadataValue::Text(ch.title.clone()),
            ));
        }

        fields
    }

    /// Generate MP4 `chpl` atom-style representation.
    ///
    /// Returns chapter entries as (start_100ns, title) pairs.
    pub fn to_mp4_chpl(&self) -> Vec<(u64, String)> {
        self.chapters
            .iter()
            .map(|ch| {
                // MP4 chpl uses 100-nanosecond units
                let start_100ns = ch.start_ms.saturating_mul(10_000);
                (start_100ns, ch.title.clone())
            })
            .collect()
    }

    /// Create a ChapterList from MP4 `chpl` entries.
    ///
    /// Each entry is (start_100ns, title). End times are inferred from
    /// the next chapter's start (or `total_duration_100ns` for the last).
    pub fn from_mp4_chpl(entries: &[(u64, String)], total_duration_100ns: u64) -> Self {
        let mut list = ChapterList::new();

        for (i, (start_100ns, title)) in entries.iter().enumerate() {
            let start_ms = start_100ns / 10_000;
            let end_ms = if i + 1 < entries.len() {
                entries[i + 1].0 / 10_000
            } else {
                total_duration_100ns / 10_000
            };
            list.add(Chapter::new(start_ms, end_ms, title.clone()));
        }

        list
    }

    /// Generate Matroska-style chapter XML snippet (simplified).
    pub fn to_matroska_xml(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<Chapters>\n");
        xml.push_str("  <EditionEntry>\n");

        if let Some(ref uid) = self.edition_uid {
            xml.push_str(&format!("    <EditionUID>{uid}</EditionUID>\n"));
        }
        xml.push_str(&format!(
            "    <EditionFlagDefault>{}</EditionFlagDefault>\n",
            if self.is_default { 1 } else { 0 }
        ));
        xml.push_str(&format!(
            "    <EditionFlagHidden>{}</EditionFlagHidden>\n",
            if self.is_hidden { 1 } else { 0 }
        ));

        for ch in &self.chapters {
            write_matroska_chapter_atom(&mut xml, ch, 4);
        }

        xml.push_str("  </EditionEntry>\n");
        xml.push_str("</Chapters>\n");
        xml
    }
}

// ---- Internal helpers ----

fn format_time_ms(ms: u64) -> String {
    let total_secs = ms / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn sort_children_recursive(chapter: &mut Chapter) {
    chapter.children.sort_by_key(|c| c.start_ms);
    for child in &mut chapter.children {
        sort_children_recursive(child);
    }
}

fn find_chapter_at(chapter: &Chapter, time_ms: u64) -> Option<&Chapter> {
    if time_ms >= chapter.start_ms && time_ms < chapter.end_ms {
        // Check children first for the most specific match
        for child in &chapter.children {
            if let Some(found) = find_chapter_at(child, time_ms) {
                return Some(found);
            }
        }
        Some(chapter)
    } else {
        None
    }
}

fn serialize_chapter(metadata: &mut Metadata, chapter: &Chapter, prefix: &str) {
    metadata.insert(
        format!("{prefix}_start"),
        MetadataValue::Integer(chapter.start_ms as i64),
    );
    metadata.insert(
        format!("{prefix}_end"),
        MetadataValue::Integer(chapter.end_ms as i64),
    );
    metadata.insert(
        format!("{prefix}_title"),
        MetadataValue::Text(chapter.title.clone()),
    );

    if let Some(ref lang) = chapter.language {
        metadata.insert(
            format!("{prefix}_language"),
            MetadataValue::Text(lang.clone()),
        );
    }
    if let Some(ref country) = chapter.country {
        metadata.insert(
            format!("{prefix}_country"),
            MetadataValue::Text(country.clone()),
        );
    }
    if let Some(ref uid) = chapter.uid {
        metadata.insert(format!("{prefix}_uid"), MetadataValue::Text(uid.clone()));
    }

    // Sub-chapters
    if !chapter.children.is_empty() {
        metadata.insert(
            format!("{prefix}_child_count"),
            MetadataValue::Integer(chapter.children.len() as i64),
        );
        for (i, child) in chapter.children.iter().enumerate() {
            serialize_chapter(metadata, child, &format!("{prefix}_child_{i}"));
        }
    }
}

fn deserialize_chapter(metadata: &Metadata, prefix: &str) -> Result<Chapter, Error> {
    let start_ms = metadata
        .get(&format!("{prefix}_start"))
        .and_then(|v| v.as_integer())
        .ok_or_else(|| Error::ParseError(format!("Missing {prefix}_start")))?
        as u64;

    let end_ms = metadata
        .get(&format!("{prefix}_end"))
        .and_then(|v| v.as_integer())
        .ok_or_else(|| Error::ParseError(format!("Missing {prefix}_end")))? as u64;

    let title = metadata
        .get(&format!("{prefix}_title"))
        .and_then(|v| v.as_text())
        .ok_or_else(|| Error::ParseError(format!("Missing {prefix}_title")))?
        .to_string();

    let mut chapter = Chapter::new(start_ms, end_ms, title);

    chapter.language = metadata
        .get(&format!("{prefix}_language"))
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    chapter.country = metadata
        .get(&format!("{prefix}_country"))
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    chapter.uid = metadata
        .get(&format!("{prefix}_uid"))
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    // Sub-chapters
    let child_count = metadata
        .get(&format!("{prefix}_child_count"))
        .and_then(|v| v.as_integer())
        .unwrap_or(0) as usize;

    for i in 0..child_count {
        let child = deserialize_chapter(metadata, &format!("{prefix}_child_{i}"))?;
        chapter.add_child(child);
    }

    Ok(chapter)
}

fn write_matroska_chapter_atom(xml: &mut String, chapter: &Chapter, indent: usize) {
    let pad = " ".repeat(indent);
    xml.push_str(&format!("{pad}<ChapterAtom>\n"));

    if let Some(ref uid) = chapter.uid {
        xml.push_str(&format!("{pad}  <ChapterUID>{uid}</ChapterUID>\n"));
    }

    // Matroska uses nanosecond timestamps
    let start_ns = chapter.start_ms.saturating_mul(1_000_000);
    let end_ns = chapter.end_ms.saturating_mul(1_000_000);
    xml.push_str(&format!(
        "{pad}  <ChapterTimeStart>{start_ns}</ChapterTimeStart>\n"
    ));
    xml.push_str(&format!(
        "{pad}  <ChapterTimeEnd>{end_ns}</ChapterTimeEnd>\n"
    ));

    xml.push_str(&format!("{pad}  <ChapterDisplay>\n"));
    xml.push_str(&format!(
        "{pad}    <ChapString>{}</ChapString>\n",
        chapter.title
    ));
    if let Some(ref lang) = chapter.language {
        xml.push_str(&format!("{pad}    <ChapLanguage>{lang}</ChapLanguage>\n"));
    }
    if let Some(ref country) = chapter.country {
        xml.push_str(&format!("{pad}    <ChapCountry>{country}</ChapCountry>\n"));
    }
    xml.push_str(&format!("{pad}  </ChapterDisplay>\n"));

    for child in &chapter.children {
        write_matroska_chapter_atom(xml, child, indent + 2);
    }

    xml.push_str(&format!("{pad}</ChapterAtom>\n"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_new() {
        let ch = Chapter::new(0, 60_000, "Intro");
        assert_eq!(ch.start_ms, 0);
        assert_eq!(ch.end_ms, 60_000);
        assert_eq!(ch.title, "Intro");
        assert_eq!(ch.duration_ms(), 60_000);
        assert!(!ch.has_children());
    }

    #[test]
    fn test_chapter_with_builders() {
        let ch = Chapter::new(0, 1000, "Test")
            .with_language("eng")
            .with_country("US")
            .with_uid("ch001");
        assert_eq!(ch.language.as_deref(), Some("eng"));
        assert_eq!(ch.country.as_deref(), Some("US"));
        assert_eq!(ch.uid.as_deref(), Some("ch001"));
    }

    #[test]
    fn test_chapter_nested() {
        let mut parent = Chapter::new(0, 120_000, "Part 1");
        parent.add_child(Chapter::new(0, 60_000, "Scene 1"));
        parent.add_child(Chapter::new(60_000, 120_000, "Scene 2"));

        assert!(parent.has_children());
        assert_eq!(parent.total_count(), 3);
    }

    #[test]
    fn test_chapter_time_formatting() {
        let ch = Chapter::new(3_661_500, 7_200_000, "Test");
        assert_eq!(ch.start_time_formatted(), "01:01:01.500");
        assert_eq!(ch.end_time_formatted(), "02:00:00.000");
    }

    #[test]
    fn test_chapter_validate_valid() {
        let ch = Chapter::new(0, 60_000, "Valid");
        assert!(ch.validate().is_empty());
    }

    #[test]
    fn test_chapter_validate_inverted_times() {
        let ch = Chapter::new(60_000, 30_000, "Bad");
        let issues = ch.validate();
        assert!(!issues.is_empty());
        assert!(issues[0].contains("start"));
    }

    #[test]
    fn test_chapter_validate_child_out_of_bounds() {
        let mut parent = Chapter::new(10_000, 20_000, "Parent");
        parent.add_child(Chapter::new(5_000, 15_000, "Child"));
        let issues = parent.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_chapter_attributes() {
        let mut ch = Chapter::new(0, 1000, "Test");
        ch.set_attribute("key1", "val1");
        assert_eq!(ch.attributes.get("key1").map(|s| s.as_str()), Some("val1"));
    }

    #[test]
    fn test_chapter_list_basic() {
        let mut list = ChapterList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        list.add(Chapter::new(0, 60_000, "Chapter 1"));
        list.add(Chapter::new(60_000, 120_000, "Chapter 2"));
        list.add(Chapter::new(120_000, 180_000, "Chapter 3"));

        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());
        assert_eq!(list.total_duration_ms(), Some(180_000));
    }

    #[test]
    fn test_chapter_list_insert_remove() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 100, "A"));
        list.add(Chapter::new(200, 300, "C"));

        list.insert(1, Chapter::new(100, 200, "B"))
            .expect("insert should succeed");
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(1).map(|c| c.title.as_str()), Some("B"));

        let removed = list.remove(0);
        assert_eq!(removed.map(|c| c.title), Some("A".to_string()));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_chapter_list_insert_out_of_bounds() {
        let mut list = ChapterList::new();
        let result = list.insert(5, Chapter::new(0, 1, "X"));
        assert!(result.is_err());
    }

    #[test]
    fn test_chapter_list_sort_by_time() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(200, 300, "C"));
        list.add(Chapter::new(0, 100, "A"));
        list.add(Chapter::new(100, 200, "B"));

        list.sort_by_time();

        assert_eq!(list.get(0).map(|c| c.title.as_str()), Some("A"));
        assert_eq!(list.get(1).map(|c| c.title.as_str()), Some("B"));
        assert_eq!(list.get(2).map(|c| c.title.as_str()), Some("C"));
    }

    #[test]
    fn test_chapter_list_chapter_at() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 60_000, "Intro"));
        list.add(Chapter::new(60_000, 120_000, "Main"));
        list.add(Chapter::new(120_000, 180_000, "End"));

        assert_eq!(
            list.chapter_at(30_000).map(|c| c.title.as_str()),
            Some("Intro")
        );
        assert_eq!(
            list.chapter_at(60_000).map(|c| c.title.as_str()),
            Some("Main")
        );
        assert_eq!(
            list.chapter_at(150_000).map(|c| c.title.as_str()),
            Some("End")
        );
        assert!(list.chapter_at(200_000).is_none());
    }

    #[test]
    fn test_chapter_list_chapter_at_nested() {
        let mut list = ChapterList::new();
        let mut part = Chapter::new(0, 120_000, "Part 1");
        part.add_child(Chapter::new(0, 60_000, "Scene 1"));
        part.add_child(Chapter::new(60_000, 120_000, "Scene 2"));
        list.add(part);

        // Should find the most specific (deepest) match
        assert_eq!(
            list.chapter_at(30_000).map(|c| c.title.as_str()),
            Some("Scene 1")
        );
        assert_eq!(
            list.chapter_at(90_000).map(|c| c.title.as_str()),
            Some("Scene 2")
        );
    }

    #[test]
    fn test_chapter_list_validate_overlapping() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 100, "A"));
        list.add(Chapter::new(50, 150, "B")); // overlaps with A
        let issues = list.validate();
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.contains("overlap")));
    }

    #[test]
    fn test_chapter_list_validate_clean() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 100, "A"));
        list.add(Chapter::new(100, 200, "B"));
        list.add(Chapter::new(200, 300, "C"));
        assert!(list.validate().is_empty());
    }

    #[test]
    fn test_chapter_list_edition_properties() {
        let mut list = ChapterList::new();
        list.set_edition_uid("ed-001");
        list.set_default(true);
        list.set_hidden(false);

        assert_eq!(list.edition_uid(), Some("ed-001"));
        assert!(list.is_default());
        assert!(!list.is_hidden());
    }

    #[test]
    fn test_chapter_list_total_count_with_nesting() {
        let mut list = ChapterList::new();
        let mut part1 = Chapter::new(0, 100, "P1");
        part1.add_child(Chapter::new(0, 50, "P1.1"));
        part1.add_child(Chapter::new(50, 100, "P1.2"));
        list.add(part1);
        list.add(Chapter::new(100, 200, "P2"));

        assert_eq!(list.total_count(), 4); // P1 + P1.1 + P1.2 + P2
    }

    #[test]
    fn test_chapter_list_metadata_round_trip() {
        let mut list = ChapterList::new();
        list.set_edition_uid("ed-42");
        list.add(
            Chapter::new(0, 60_000, "Chapter One")
                .with_language("eng")
                .with_uid("c1"),
        );
        list.add(Chapter::new(60_000, 120_000, "Chapter Two"));

        let metadata = list.to_metadata(MetadataFormat::Matroska);
        let restored =
            ChapterList::from_metadata(&metadata).expect("deserialization should succeed");

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.edition_uid(), Some("ed-42"));

        let ch0 = restored.get(0).expect("chapter 0");
        assert_eq!(ch0.start_ms, 0);
        assert_eq!(ch0.end_ms, 60_000);
        assert_eq!(ch0.title, "Chapter One");
        assert_eq!(ch0.language.as_deref(), Some("eng"));
        assert_eq!(ch0.uid.as_deref(), Some("c1"));

        let ch1 = restored.get(1).expect("chapter 1");
        assert_eq!(ch1.title, "Chapter Two");
    }

    #[test]
    fn test_chapter_list_metadata_with_children_round_trip() {
        let mut list = ChapterList::new();
        let mut parent = Chapter::new(0, 120_000, "Part 1");
        parent.add_child(Chapter::new(0, 60_000, "Scene 1"));
        parent.add_child(Chapter::new(60_000, 120_000, "Scene 2"));
        list.add(parent);

        let metadata = list.to_metadata(MetadataFormat::Matroska);
        let restored =
            ChapterList::from_metadata(&metadata).expect("deserialization should succeed");

        assert_eq!(restored.len(), 1);
        let ch = restored.get(0).expect("chapter 0");
        assert_eq!(ch.children.len(), 2);
        assert_eq!(ch.children[0].title, "Scene 1");
        assert_eq!(ch.children[1].title, "Scene 2");
    }

    #[test]
    fn test_chapter_list_to_id3v2_chap() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 30_000, "Intro"));
        list.add(Chapter::new(30_000, 60_000, "Verse"));

        let fields = list.to_id3v2_chap_fields();

        // CTOC + 3 fields per chapter (start, end, title) = 1 + 6 = 7
        assert_eq!(fields.len(), 7);

        // Verify CTOC
        let ctoc = &fields[0];
        assert_eq!(ctoc.0, "CTOC");

        // Verify first chapter
        let has_intro_title = fields.iter().any(|(k, v)| {
            k == "CHAP:chp0:title" && matches!(v, MetadataValue::Text(t) if t == "Intro")
        });
        assert!(has_intro_title);
    }

    #[test]
    fn test_chapter_list_mp4_chpl_round_trip() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 60_000, "Chapter 1"));
        list.add(Chapter::new(60_000, 120_000, "Chapter 2"));

        let chpl = list.to_mp4_chpl();
        assert_eq!(chpl.len(), 2);
        assert_eq!(chpl[0].0, 0); // 0 ms * 10000
        assert_eq!(chpl[0].1, "Chapter 1");
        assert_eq!(chpl[1].0, 600_000_000); // 60000 ms * 10000

        // Round trip
        let total_100ns = 120_000 * 10_000;
        let restored = ChapterList::from_mp4_chpl(&chpl, total_100ns);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get(0).map(|c| c.start_ms), Some(0));
        assert_eq!(restored.get(0).map(|c| c.end_ms), Some(60_000));
        assert_eq!(restored.get(1).map(|c| c.start_ms), Some(60_000));
        assert_eq!(restored.get(1).map(|c| c.end_ms), Some(120_000));
    }

    #[test]
    fn test_chapter_list_matroska_xml() {
        let mut list = ChapterList::new();
        list.set_edition_uid("12345");
        list.set_default(true);
        list.add(
            Chapter::new(0, 60_000, "Chapter 1")
                .with_uid("c1")
                .with_language("eng"),
        );

        let xml = list.to_matroska_xml();
        assert!(xml.contains("<Chapters>"));
        assert!(xml.contains("<EditionUID>12345</EditionUID>"));
        assert!(xml.contains("<EditionFlagDefault>1</EditionFlagDefault>"));
        assert!(xml.contains("<ChapterUID>c1</ChapterUID>"));
        assert!(xml.contains("<ChapString>Chapter 1</ChapString>"));
        assert!(xml.contains("<ChapLanguage>eng</ChapLanguage>"));
        // Start time in nanoseconds
        assert!(xml.contains("<ChapterTimeStart>0</ChapterTimeStart>"));
        assert!(xml.contains("<ChapterTimeEnd>60000000000</ChapterTimeEnd>"));
    }

    #[test]
    fn test_format_time_ms() {
        assert_eq!(format_time_ms(0), "00:00:00.000");
        assert_eq!(format_time_ms(1500), "00:00:01.500");
        assert_eq!(format_time_ms(3_661_500), "01:01:01.500");
        assert_eq!(format_time_ms(86_400_000), "24:00:00.000");
    }

    #[test]
    fn test_chapter_list_remove_out_of_bounds() {
        let mut list = ChapterList::new();
        assert!(list.remove(0).is_none());
    }

    #[test]
    fn test_chapter_list_get_mut() {
        let mut list = ChapterList::new();
        list.add(Chapter::new(0, 100, "Original"));

        if let Some(ch) = list.get_mut(0) {
            ch.title = "Modified".to_string();
        }
        assert_eq!(list.get(0).map(|c| c.title.as_str()), Some("Modified"));
    }

    #[test]
    fn test_chapter_duration_saturating() {
        let ch = Chapter::new(100, 50, "Inverted");
        assert_eq!(ch.duration_ms(), 0); // saturating_sub
    }

    #[test]
    fn test_empty_chapter_list_total_duration() {
        let list = ChapterList::new();
        assert_eq!(list.total_duration_ms(), None);
    }
}
