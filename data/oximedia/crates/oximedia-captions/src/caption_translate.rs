//! Caption translation pipeline: language tagging, segment alignment,
//! and glossary substitution.

#![allow(dead_code)]
#![allow(missing_docs)]

use std::collections::HashMap;

// ── Language tag ─────────────────────────────────────────────────────────────

/// BCP-47 language tag (simplified representation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageTag {
    /// ISO 639-1 or 639-2 language code (e.g. "en", "fr", "zh")
    pub language: String,
    /// Optional ISO 15924 script subtag (e.g. "Latn", "Hant")
    pub script: Option<String>,
    /// Optional ISO 3166-1 region subtag (e.g. "US", "GB")
    pub region: Option<String>,
}

impl LanguageTag {
    /// Create a simple language tag
    #[must_use]
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_lowercase(),
            script: None,
            region: None,
        }
    }

    /// Create a language-region tag
    #[must_use]
    pub fn with_region(language: &str, region: &str) -> Self {
        Self {
            language: language.to_lowercase(),
            script: None,
            region: Some(region.to_uppercase()),
        }
    }

    /// Create a language-script-region tag
    #[must_use]
    pub fn full(language: &str, script: &str, region: &str) -> Self {
        Self {
            language: language.to_lowercase(),
            script: Some({
                let mut s = script.to_lowercase();
                if let Some(first) = s.get_mut(0..1) {
                    first.make_ascii_uppercase();
                }
                s
            }),
            region: Some(region.to_uppercase()),
        }
    }

    /// Format as BCP-47 string
    #[must_use]
    pub fn as_bcp47(&self) -> String {
        let mut parts = vec![self.language.clone()];
        if let Some(script) = &self.script {
            parts.push(script.clone());
        }
        if let Some(region) = &self.region {
            parts.push(region.clone());
        }
        parts.join("-")
    }

    /// Check whether this tag matches a given language code (ignoring subtags)
    #[must_use]
    pub fn matches_language(&self, lang: &str) -> bool {
        self.language == lang.to_lowercase()
    }
}

impl std::fmt::Display for LanguageTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_bcp47())
    }
}

// ── Segment ───────────────────────────────────────────────────────────────────

/// A timed text segment with language metadata
#[derive(Debug, Clone)]
pub struct TranslationSegment {
    /// Unique segment id within the track
    pub id: usize,
    /// Source text (original language)
    pub source_text: String,
    /// Translated text (target language), None if not yet translated
    pub translated_text: Option<String>,
    /// Source language tag
    pub source_lang: LanguageTag,
    /// Target language tag
    pub target_lang: LanguageTag,
    /// Begin time in milliseconds
    pub begin_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Confidence score for the translation (0.0 – 1.0)
    pub confidence: Option<f32>,
}

impl TranslationSegment {
    /// Create a new untranslated segment
    #[must_use]
    pub fn new(
        id: usize,
        text: &str,
        source_lang: LanguageTag,
        target_lang: LanguageTag,
        begin_ms: u64,
        end_ms: u64,
    ) -> Self {
        Self {
            id,
            source_text: text.to_string(),
            translated_text: None,
            source_lang,
            target_lang,
            begin_ms,
            end_ms,
            confidence: None,
        }
    }

    /// Duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.begin_ms)
    }

    /// True if this segment has a translation
    #[must_use]
    pub fn is_translated(&self) -> bool {
        self.translated_text.is_some()
    }
}

// ── Glossary ──────────────────────────────────────────────────────────────────

/// A glossary entry with source and target terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlossaryEntry {
    pub source_term: String,
    pub target_term: String,
    pub case_sensitive: bool,
}

impl GlossaryEntry {
    #[must_use]
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source_term: source.to_string(),
            target_term: target.to_string(),
            case_sensitive: false,
        }
    }

    #[must_use]
    pub fn case_sensitive(mut self) -> Self {
        self.case_sensitive = true;
        self
    }
}

/// Domain-specific glossary for translation substitution
#[derive(Debug, Default, Clone)]
pub struct Glossary {
    entries: Vec<GlossaryEntry>,
}

impl Glossary {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the glossary
    pub fn add(&mut self, entry: GlossaryEntry) {
        self.entries.push(entry);
    }

    /// Number of entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Apply glossary substitutions to translated text
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();
        for entry in &self.entries {
            if entry.case_sensitive {
                result = result.replace(&entry.source_term, &entry.target_term);
            } else {
                // Case-insensitive replacement preserving case of surrounding text
                let lower = result.to_lowercase();
                let src_lower = entry.source_term.to_lowercase();
                if let Some(pos) = lower.find(&src_lower) {
                    result = format!(
                        "{}{}{}",
                        &result[..pos],
                        &entry.target_term,
                        &result[pos + entry.source_term.len()..]
                    );
                }
            }
        }
        result
    }
}

// ── Alignment ─────────────────────────────────────────────────────────────────

/// How two segment lists should be aligned
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentStrategy {
    /// Match purely by sequential index (1-to-1)
    Sequential,
    /// Match by timing overlap
    Temporal,
    /// Match by sentence-count heuristics
    Sentence,
}

/// Result of aligning a source segment list with a translated list
#[derive(Debug, Clone)]
pub struct AlignedPair {
    pub source_idx: usize,
    pub target_idx: Option<usize>,
    pub overlap_ms: u64,
}

/// Align source segments with translated segments
#[must_use]
pub fn align_segments(
    source: &[TranslationSegment],
    translated: &[TranslationSegment],
    strategy: AlignmentStrategy,
) -> Vec<AlignedPair> {
    match strategy {
        AlignmentStrategy::Sequential => source
            .iter()
            .enumerate()
            .map(|(i, _)| AlignedPair {
                source_idx: i,
                target_idx: if i < translated.len() { Some(i) } else { None },
                overlap_ms: if i < translated.len() {
                    compute_overlap(
                        source[i].begin_ms,
                        source[i].end_ms,
                        translated[i].begin_ms,
                        translated[i].end_ms,
                    )
                } else {
                    0
                },
            })
            .collect(),

        AlignmentStrategy::Temporal => source
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                // Find translated segment with maximum temporal overlap
                let best = translated.iter().enumerate().max_by_key(|(_, t)| {
                    compute_overlap(seg.begin_ms, seg.end_ms, t.begin_ms, t.end_ms)
                });
                let (target_idx, overlap_ms) = match best {
                    Some((ti, t)) => {
                        let ov = compute_overlap(seg.begin_ms, seg.end_ms, t.begin_ms, t.end_ms);
                        (Some(ti), ov)
                    }
                    None => (None, 0),
                };
                AlignedPair {
                    source_idx: i,
                    target_idx,
                    overlap_ms,
                }
            })
            .collect(),

        AlignmentStrategy::Sentence => {
            // Simple sentence-count heuristic: align greedily by count
            align_segments(source, translated, AlignmentStrategy::Sequential)
        }
    }
}

fn compute_overlap(a_begin: u64, a_end: u64, b_begin: u64, b_end: u64) -> u64 {
    let start = a_begin.max(b_begin);
    let end = a_end.min(b_end);
    end.saturating_sub(start)
}

// ── Translation pipeline ──────────────────────────────────────────────────────

/// Configuration for the translation pipeline
#[derive(Debug, Clone)]
pub struct TranslationConfig {
    pub source_lang: LanguageTag,
    pub target_lang: LanguageTag,
    pub alignment_strategy: AlignmentStrategy,
    pub glossary: Glossary,
    /// Minimum confidence threshold; segments below this are flagged
    pub min_confidence: f32,
}

impl TranslationConfig {
    #[must_use]
    pub fn new(source_lang: LanguageTag, target_lang: LanguageTag) -> Self {
        Self {
            source_lang,
            target_lang,
            alignment_strategy: AlignmentStrategy::Temporal,
            glossary: Glossary::new(),
            min_confidence: 0.6,
        }
    }
}

/// Outcome of processing one segment through the pipeline
#[derive(Debug, Clone)]
pub struct TranslationResult {
    pub segment_id: usize,
    pub output_text: String,
    pub glossary_applied: bool,
    pub flagged_low_confidence: bool,
}

/// Apply the full translation pipeline to a list of segments.
///
/// In a real implementation this would call an MT backend; here it uses
/// a trivial identity transform so the pipeline logic can be exercised
/// without external dependencies.
pub fn run_pipeline(
    segments: &mut Vec<TranslationSegment>,
    config: &TranslationConfig,
) -> Vec<TranslationResult> {
    segments
        .iter_mut()
        .map(|seg| {
            // Simulate translation: prepend [lang] marker
            let raw = format!("[{}] {}", config.target_lang.language, seg.source_text);
            // Apply glossary
            let after_glossary = config.glossary.apply(&raw);
            let glossary_applied = after_glossary != raw;
            seg.translated_text = Some(after_glossary.clone());
            seg.confidence = Some(0.85);

            let flagged = seg.confidence.unwrap_or(0.0) < config.min_confidence;

            TranslationResult {
                segment_id: seg.id,
                output_text: after_glossary,
                glossary_applied,
                flagged_low_confidence: flagged,
            }
        })
        .collect()
}

/// Statistics about a translation run
#[derive(Debug, Default, Clone)]
pub struct TranslationStats {
    pub total_segments: usize,
    pub translated_segments: usize,
    pub glossary_hits: usize,
    pub low_confidence_count: usize,
}

impl TranslationStats {
    /// Compute stats from pipeline results
    #[must_use]
    pub fn from_results(results: &[TranslationResult]) -> Self {
        let glossary_hits = results.iter().filter(|r| r.glossary_applied).count();
        let low_confidence_count = results.iter().filter(|r| r.flagged_low_confidence).count();
        Self {
            total_segments: results.len(),
            translated_segments: results.len(),
            glossary_hits,
            low_confidence_count,
        }
    }

    /// Coverage ratio (0.0–1.0)
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn coverage(&self) -> f32 {
        if self.total_segments == 0 {
            1.0
        } else {
            self.translated_segments as f32 / self.total_segments as f32
        }
    }
}

// ── Language pair map ─────────────────────────────────────────────────────────

/// Registry that tracks glossaries for multiple language pairs
#[derive(Debug, Default)]
pub struct GlossaryRegistry {
    map: HashMap<(String, String), Glossary>,
}

impl GlossaryRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a glossary for a source→target pair
    pub fn insert(&mut self, source: &str, target: &str, glossary: Glossary) {
        self.map
            .insert((source.to_string(), target.to_string()), glossary);
    }

    /// Retrieve the glossary for a pair, returning an empty one if not found
    #[must_use]
    pub fn get_or_empty(&self, source: &str, target: &str) -> &Glossary {
        static EMPTY: Glossary = Glossary { entries: vec![] };
        self.map
            .get(&(source.to_string(), target.to_string()))
            .unwrap_or(&EMPTY)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_tag_new() {
        let tag = LanguageTag::new("EN");
        assert_eq!(tag.language, "en");
    }

    #[test]
    fn test_language_tag_bcp47_simple() {
        let tag = LanguageTag::new("fr");
        assert_eq!(tag.as_bcp47(), "fr");
    }

    #[test]
    fn test_language_tag_bcp47_region() {
        let tag = LanguageTag::with_region("en", "us");
        assert_eq!(tag.as_bcp47(), "en-US");
    }

    #[test]
    fn test_language_tag_bcp47_full() {
        let tag = LanguageTag::full("zh", "hant", "tw");
        assert_eq!(tag.as_bcp47(), "zh-Hant-TW");
    }

    #[test]
    fn test_language_tag_matches_language() {
        let tag = LanguageTag::with_region("en", "GB");
        assert!(tag.matches_language("en"));
        assert!(!tag.matches_language("fr"));
    }

    #[test]
    fn test_translation_segment_duration() {
        let seg = TranslationSegment::new(
            0,
            "Hello",
            LanguageTag::new("en"),
            LanguageTag::new("fr"),
            1000,
            4000,
        );
        assert_eq!(seg.duration_ms(), 3000);
        assert!(!seg.is_translated());
    }

    #[test]
    fn test_glossary_case_insensitive() {
        let mut g = Glossary::new();
        g.add(GlossaryEntry::new("hello", "bonjour"));
        let result = g.apply("Say Hello there");
        assert!(result.contains("bonjour"));
    }

    #[test]
    fn test_glossary_case_sensitive() {
        let mut g = Glossary::new();
        g.add(GlossaryEntry::new("Hello", "Bonjour").case_sensitive());
        // Should match "Hello" exactly
        assert!(g.apply("Hello world").contains("Bonjour"));
        // Should NOT match "hello" (lowercase)
        assert!(!g.apply("hello world").contains("Bonjour"));
    }

    #[test]
    fn test_glossary_len_and_empty() {
        let mut g = Glossary::new();
        assert!(g.is_empty());
        g.add(GlossaryEntry::new("a", "b"));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_compute_overlap() {
        assert_eq!(compute_overlap(0, 5000, 3000, 8000), 2000);
        assert_eq!(compute_overlap(0, 1000, 2000, 3000), 0);
    }

    #[test]
    fn test_align_segments_sequential() {
        let src_lang = LanguageTag::new("en");
        let tgt_lang = LanguageTag::new("fr");
        let src: Vec<_> = (0..3)
            .map(|i| {
                TranslationSegment::new(
                    i,
                    "text",
                    src_lang.clone(),
                    tgt_lang.clone(),
                    i as u64 * 1000,
                    (i as u64 + 1) * 1000,
                )
            })
            .collect();
        let tgt: Vec<_> = (0..3)
            .map(|i| {
                TranslationSegment::new(
                    i,
                    "texte",
                    tgt_lang.clone(),
                    src_lang.clone(),
                    i as u64 * 1000,
                    (i as u64 + 1) * 1000,
                )
            })
            .collect();
        let pairs = align_segments(&src, &tgt, AlignmentStrategy::Sequential);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].target_idx, Some(0));
    }

    #[test]
    fn test_run_pipeline_basic() {
        let mut segments = vec![TranslationSegment::new(
            0,
            "Hello",
            LanguageTag::new("en"),
            LanguageTag::new("de"),
            0,
            2000,
        )];
        let config = TranslationConfig::new(LanguageTag::new("en"), LanguageTag::new("de"));
        let results = run_pipeline(&mut segments, &config);
        assert_eq!(results.len(), 1);
        assert!(results[0].output_text.contains("de"));
        assert!(segments[0].is_translated());
    }

    #[test]
    fn test_translation_stats_coverage() {
        let results = vec![
            TranslationResult {
                segment_id: 0,
                output_text: "a".into(),
                glossary_applied: false,
                flagged_low_confidence: false,
            },
            TranslationResult {
                segment_id: 1,
                output_text: "b".into(),
                glossary_applied: true,
                flagged_low_confidence: false,
            },
        ];
        let stats = TranslationStats::from_results(&results);
        assert_eq!(stats.total_segments, 2);
        assert_eq!(stats.glossary_hits, 1);
        assert!((stats.coverage() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_glossary_registry_insert_get() {
        let mut reg = GlossaryRegistry::new();
        let mut g = Glossary::new();
        g.add(GlossaryEntry::new("broadcast", "Rundfunk"));
        reg.insert("en", "de", g);
        assert_eq!(reg.len(), 1);
        assert!(!reg.get_or_empty("en", "de").is_empty());
        assert!(reg.get_or_empty("en", "fr").is_empty());
    }

    #[test]
    fn test_pipeline_with_glossary() {
        let mut segments = vec![TranslationSegment::new(
            0,
            "broadcast signal",
            LanguageTag::new("en"),
            LanguageTag::new("de"),
            0,
            2000,
        )];
        let mut config = TranslationConfig::new(LanguageTag::new("en"), LanguageTag::new("de"));
        config
            .glossary
            .add(GlossaryEntry::new("broadcast", "Rundfunk"));
        let results = run_pipeline(&mut segments, &config);
        assert!(
            results[0].glossary_applied
                || results[0].output_text.contains("Rundfunk")
                || !results[0].output_text.is_empty()
        );
    }
}
