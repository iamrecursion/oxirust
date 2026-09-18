//! Subtitle translation support.
//!
//! Provides language identification, translation memory with fuzzy matching,
//! and bilingual subtitle structures.

/// Text direction for a language.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TextDirection {
    /// Left-to-right (most Latin-script languages).
    LeftToRight,
    /// Right-to-left (Arabic, Hebrew, etc.).
    RightToLeft,
}

/// A language with its BCP-47 code, display name, and text direction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Language {
    /// BCP-47 language code (e.g. "en", "ar").
    pub code: String,
    /// Human-readable language name.
    pub name: String,
    /// Text direction.
    pub direction: TextDirection,
    /// Whether this is a right-to-left language.
    pub is_rtl: bool,
}

impl Language {
    /// Create a new language descriptor.
    #[allow(dead_code)]
    pub fn new(code: impl Into<String>, name: impl Into<String>, direction: TextDirection) -> Self {
        let is_rtl = direction == TextDirection::RightToLeft;
        Self {
            code: code.into(),
            name: name.into(),
            direction,
            is_rtl,
        }
    }

    /// Look up a language by its BCP-47 code.
    ///
    /// Known languages: en, fr, de, es, ar, he, zh, ja, ko, ru
    #[allow(dead_code)]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "en" => Some(Self::new("en", "English", TextDirection::LeftToRight)),
            "fr" => Some(Self::new("fr", "French", TextDirection::LeftToRight)),
            "de" => Some(Self::new("de", "German", TextDirection::LeftToRight)),
            "es" => Some(Self::new("es", "Spanish", TextDirection::LeftToRight)),
            "ar" => Some(Self::new("ar", "Arabic", TextDirection::RightToLeft)),
            "he" => Some(Self::new("he", "Hebrew", TextDirection::RightToLeft)),
            "zh" => Some(Self::new("zh", "Chinese", TextDirection::LeftToRight)),
            "ja" => Some(Self::new("ja", "Japanese", TextDirection::LeftToRight)),
            "ko" => Some(Self::new("ko", "Korean", TextDirection::LeftToRight)),
            "ru" => Some(Self::new("ru", "Russian", TextDirection::LeftToRight)),
            _ => None,
        }
    }
}

/// A unit of translation with source and target text.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TranslationUnit {
    /// Source language code.
    pub source_lang: String,
    /// Target language code.
    pub target_lang: String,
    /// Original source text.
    pub source_text: String,
    /// Translated text.
    pub translated_text: String,
    /// Translation confidence (0.0–1.0).
    pub confidence: f32,
}

impl TranslationUnit {
    /// Create a new translation unit.
    #[allow(dead_code)]
    pub fn new(
        source_lang: impl Into<String>,
        target_lang: impl Into<String>,
        source_text: impl Into<String>,
        translated_text: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            source_lang: source_lang.into(),
            target_lang: target_lang.into(),
            source_text: source_text.into(),
            translated_text: translated_text.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Compute the Levenshtein edit distance between two strings.
///
/// Standard dynamic-programming implementation.
#[allow(dead_code)]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }

    dp[m][n]
}

/// A translation memory for reusing previous translations via fuzzy matching.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TranslationMemory {
    /// Stored translation units.
    pub units: Vec<TranslationUnit>,
}

impl TranslationMemory {
    /// Create an empty translation memory.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { units: Vec::new() }
    }

    /// Add a translation unit to the memory.
    #[allow(dead_code)]
    pub fn add(&mut self, unit: TranslationUnit) {
        self.units.push(unit);
    }

    /// Find the best matching translation unit for a query text.
    ///
    /// Uses Levenshtein distance to compute similarity.
    /// Returns the unit and a similarity score in [0.0, 1.0], or `None`
    /// if the memory is empty or no match is above a minimal threshold.
    ///
    /// # Arguments
    ///
    /// * `text` - The source text to match.
    /// * `src` - Source language code filter.
    /// * `tgt` - Target language code filter.
    #[allow(dead_code)]
    pub fn find_match<'a>(
        &'a self,
        text: &str,
        src: &str,
        tgt: &str,
    ) -> Option<(&'a TranslationUnit, f32)> {
        let candidates: Vec<&TranslationUnit> = self
            .units
            .iter()
            .filter(|u| u.source_lang == src && u.target_lang == tgt)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let mut best: Option<(&TranslationUnit, f32)> = None;

        for unit in candidates {
            let dist = levenshtein(text, &unit.source_text);
            let max_len = text.len().max(unit.source_text.len());
            let similarity = if max_len == 0 {
                1.0
            } else {
                1.0 - (dist as f32 / max_len as f32)
            };

            if best.is_none() || similarity > best.as_ref().map(|b| b.1).unwrap_or(0.0) {
                best = Some((unit, similarity));
            }
        }

        best
    }
}

/// A bilingual subtitle entry for dual-language display.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BilingualSubtitle {
    /// Original source-language text.
    pub source: String,
    /// Translated target-language text.
    pub translation: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

impl BilingualSubtitle {
    /// Create a new bilingual subtitle.
    #[allow(dead_code)]
    pub fn new(
        source: impl Into<String>,
        translation: impl Into<String>,
        start_ms: u64,
        end_ms: u64,
    ) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
            start_ms,
            end_ms,
        }
    }

    /// Duration of this subtitle in milliseconds.
    #[allow(dead_code)]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Whether this subtitle is active at the given timestamp.
    #[allow(dead_code)]
    pub fn is_active(&self, timestamp_ms: u64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }

    /// Format both lines for display, source on top.
    #[allow(dead_code)]
    pub fn formatted(&self) -> String {
        format!("{}\n{}", self.source, self.translation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_code_known() {
        let lang = Language::from_code("en").expect("should succeed in test");
        assert_eq!(lang.code, "en");
        assert_eq!(lang.name, "English");
        assert!(!lang.is_rtl);
    }

    #[test]
    fn test_language_from_code_rtl() {
        let lang = Language::from_code("ar").expect("should succeed in test");
        assert!(lang.is_rtl);
        assert_eq!(lang.direction, TextDirection::RightToLeft);
    }

    #[test]
    fn test_language_from_code_hebrew() {
        let lang = Language::from_code("he").expect("should succeed in test");
        assert!(lang.is_rtl);
    }

    #[test]
    fn test_language_from_code_unknown() {
        assert!(Language::from_code("xx").is_none());
    }

    #[test]
    fn test_language_all_known_codes() {
        for code in &["en", "fr", "de", "es", "ar", "he", "zh", "ja", "ko", "ru"] {
            assert!(Language::from_code(code).is_some(), "Missing: {code}");
        }
    }

    #[test]
    fn test_levenshtein_equal() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_levenshtein_simple_substitution() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_levenshtein_single_insertion() {
        assert_eq!(levenshtein("cat", "cats"), 1);
    }

    #[test]
    fn test_translation_memory_exact_match() {
        let mut mem = TranslationMemory::new();
        mem.add(TranslationUnit::new(
            "en",
            "fr",
            "Hello world",
            "Bonjour le monde",
            1.0,
        ));
        let result = mem.find_match("Hello world", "en", "fr");
        assert!(result.is_some());
        let (unit, score) = result.expect("should succeed in test");
        assert!(
            (score - 1.0).abs() < 1e-5,
            "Expected exact match score 1.0, got {score}"
        );
        assert_eq!(unit.translated_text, "Bonjour le monde");
    }

    #[test]
    fn test_translation_memory_fuzzy_match() {
        let mut mem = TranslationMemory::new();
        mem.add(TranslationUnit::new(
            "en",
            "de",
            "Hello world",
            "Hallo Welt",
            1.0,
        ));
        // Slightly different query
        let result = mem.find_match("Hello World!", "en", "de");
        assert!(result.is_some());
        let (_, score) = result.expect("should succeed in test");
        assert!(
            score > 0.5,
            "Fuzzy match should have reasonable similarity, got {score}"
        );
    }

    #[test]
    fn test_translation_memory_no_match_wrong_lang() {
        let mut mem = TranslationMemory::new();
        mem.add(TranslationUnit::new("en", "fr", "Hello", "Bonjour", 1.0));
        // Different language pair
        let result = mem.find_match("Hello", "en", "de");
        assert!(result.is_none());
    }

    #[test]
    fn test_translation_memory_empty() {
        let mem = TranslationMemory::new();
        assert!(mem.find_match("test", "en", "fr").is_none());
    }

    #[test]
    fn test_bilingual_subtitle_is_active() {
        let sub = BilingualSubtitle::new("Hello", "Hola", 1000, 4000);
        assert!(sub.is_active(2000));
        assert!(!sub.is_active(500));
        assert!(!sub.is_active(4000));
    }

    #[test]
    fn test_bilingual_subtitle_duration() {
        let sub = BilingualSubtitle::new("Hello", "Hola", 1000, 4000);
        assert_eq!(sub.duration_ms(), 3000);
    }

    #[test]
    fn test_bilingual_subtitle_formatted() {
        let sub = BilingualSubtitle::new("Hello", "Hola", 0, 1000);
        let fmt = sub.formatted();
        assert!(fmt.contains("Hello"));
        assert!(fmt.contains("Hola"));
        assert!(fmt.contains('\n'));
    }
}
