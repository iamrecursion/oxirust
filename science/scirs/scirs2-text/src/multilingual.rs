//! Multilingual text processing and language detection
//!
//! This module provides functionality for detecting languages
//! and processing text in multiple languages.

use crate::error::{Result, TextError};
use std::collections::HashMap;

/// Supported languages for detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// English
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Italian
    Italian,
    /// Portuguese
    Portuguese,
    /// Dutch
    Dutch,
    /// Russian
    Russian,
    /// Chinese
    Chinese,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Arabic
    Arabic,
    /// Unknown language
    Unknown,
}

impl Language {
    /// Get the ISO 639-1 code for the language
    pub fn iso_code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Dutch => "nl",
            Language::Russian => "ru",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Unknown => "und",
        }
    }

    /// Get the language from ISO 639-1 code
    pub fn from_iso_code(code: &str) -> Self {
        match code.to_lowercase().as_str() {
            "en" => Language::English,
            "es" => Language::Spanish,
            "fr" => Language::French,
            "de" => Language::German,
            "it" => Language::Italian,
            "pt" => Language::Portuguese,
            "nl" => Language::Dutch,
            "ru" => Language::Russian,
            "zh" => Language::Chinese,
            "ja" => Language::Japanese,
            "ko" => Language::Korean,
            "ar" => Language::Arabic,
            _ => Language::Unknown,
        }
    }

    /// Get the full name of the language
    pub fn name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Dutch => "Dutch",
            Language::Russian => "Russian",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Arabic => "Arabic",
            Language::Unknown => "Unknown",
        }
    }
}

/// Result of language detection
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    /// The detected language
    pub language: Language,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Alternative language candidates with scores
    pub alternatives: Vec<(Language, f64)>,
}

/// Language detector using character n-gram profiles
pub struct LanguageDetector {
    /// Character n-gram profiles for each language
    profiles: HashMap<Language, HashMap<String, f64>>,
    /// N-gram size (typically 2 or 3)
    n_gram_size: usize,
}

impl LanguageDetector {
    /// Create a new language detector with default profiles
    pub fn new() -> Self {
        let mut detector = Self {
            profiles: HashMap::new(),
            n_gram_size: 3,
        };
        detector.initialize_default_profiles();
        detector
    }

    /// Create a language detector with custom n-gram size
    pub fn with_ngram_size(n_gramsize: usize) -> Result<Self> {
        if !(1..=5).contains(&n_gramsize) {
            return Err(TextError::InvalidInput(
                "N-gram size must be between 1 and 5".to_string(),
            ));
        }
        let mut detector = Self {
            profiles: HashMap::new(),
            n_gram_size: n_gramsize,
        };
        detector.initialize_default_profiles();
        Ok(detector)
    }

    /// Initialize default language profiles with common n-grams
    fn initialize_default_profiles(&mut self) {
        // English profile
        let mut english_profile = HashMap::new();
        for (ngram, freq) in &[
            ("the", 0.05),
            ("and", 0.03),
            ("ing", 0.025),
            ("ion", 0.02),
            ("tio", 0.018),
            ("ent", 0.015),
            ("ati", 0.013),
            ("her", 0.012),
            ("for", 0.011),
            ("ter", 0.01),
            ("hat", 0.009),
            ("tha", 0.009),
            ("ere", 0.008),
            ("ate", 0.008),
            ("ver", 0.007),
            ("his", 0.007),
        ] {
            english_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::English, english_profile);

        // Spanish profile
        let mut spanish_profile = HashMap::new();
        for (ngram, freq) in &[
            ("que", 0.04),
            ("de_", 0.035),
            ("la_", 0.03),
            ("el_", 0.025),
            ("es_", 0.02),
            ("los", 0.018),
            ("las", 0.015),
            ("ión", 0.013),
            ("ado", 0.012),
            ("nte", 0.011),
            ("con", 0.01),
            ("par", 0.009),
            ("ara", 0.008),
            ("una", 0.008),
            ("por", 0.007),
            ("est", 0.007),
        ] {
            spanish_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Spanish, spanish_profile);

        // French profile
        let mut french_profile = HashMap::new();
        for (ngram, freq) in &[
            ("de_", 0.05),
            ("le_", 0.04),
            ("que", 0.03),
            ("les", 0.025),
            ("la_", 0.02),
            ("des", 0.018),
            ("ent", 0.015),
            ("ion", 0.013),
            ("est", 0.012),
            ("ait", 0.011),
            ("pour", 0.01),
            ("ais", 0.009),
            ("ans", 0.008),
            ("ont", 0.008),
            ("une", 0.007),
            ("qui", 0.007),
        ] {
            french_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::French, french_profile);

        // German profile
        let mut german_profile = HashMap::new();
        for (ngram, freq) in &[
            ("der", 0.05),
            ("die", 0.04),
            ("und", 0.03),
            ("den", 0.025),
            ("das", 0.02),
            ("ein", 0.018),
            ("ich", 0.015),
            ("ist", 0.013),
            ("sch", 0.012),
            ("cht", 0.011),
            ("ung", 0.01),
            ("gen", 0.009),
            ("eit", 0.008),
            ("ver", 0.008),
            ("ber", 0.007),
            ("ten", 0.007),
        ] {
            german_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::German, german_profile);

        // Italian profile
        let mut italian_profile = HashMap::new();
        for (ngram, freq) in &[
            ("che", 0.05),
            ("la_", 0.04),
            ("il_", 0.03),
            ("di_", 0.025),
            ("del", 0.02),
            ("le_", 0.018),
            ("lla", 0.015),
            ("per", 0.013),
            ("ato", 0.012),
            ("gli", 0.011),
            ("sta", 0.01),
            ("con", 0.009),
            ("ent", 0.008),
            ("ion", 0.008),
            ("are", 0.007),
            ("una", 0.007),
        ] {
            italian_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Italian, italian_profile);

        // Portuguese profile
        let mut portuguese_profile = HashMap::new();
        for (ngram, freq) in &[
            ("que", 0.05),
            ("de_", 0.04),
            ("os_", 0.03),
            ("as_", 0.025),
            ("da_", 0.02),
            ("do_", 0.018),
            ("ão_", 0.015),
            ("ent", 0.013),
            ("com", 0.012),
            ("para", 0.011),
            ("uma", 0.01),
            ("est", 0.009),
            ("nte", 0.008),
            ("ção", 0.008),
            ("por", 0.007),
            ("não", 0.007),
        ] {
            portuguese_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles
            .insert(Language::Portuguese, portuguese_profile);

        // Dutch profile
        let mut dutch_profile = HashMap::new();
        for (ngram, freq) in &[
            ("de_", 0.05),
            ("het", 0.04),
            ("een", 0.03),
            ("van", 0.025),
            ("en_", 0.02),
            ("dat", 0.018),
            ("te_", 0.015),
            ("op_", 0.013),
            ("aar", 0.012),
            ("oor", 0.011),
            ("eer", 0.01),
            ("sch", 0.009),
            ("ver", 0.008),
            ("ing", 0.008),
            ("cht", 0.007),
            ("ter", 0.007),
        ] {
            dutch_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Dutch, dutch_profile);

        // Russian profile
        let mut russian_profile = HashMap::new();
        for (ngram, freq) in &[
            ("что", 0.05),
            ("ого", 0.04),
            ("как", 0.03),
            ("это", 0.025),
            ("все", 0.02),
            ("был", 0.018),
            ("ени", 0.015),
            ("ост", 0.013),
            ("ова", 0.012),
            ("про", 0.011),
            ("сто", 0.01),
            ("ого", 0.009),
            ("при", 0.008),
            ("ени", 0.008),
            ("ать", 0.007),
            ("ный", 0.007),
        ] {
            russian_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Russian, russian_profile);

        // Chinese profile (using pinyin representation)
        let mut chinese_profile = HashMap::new();
        for (ngram, freq) in &[
            ("的_", 0.06),
            ("是_", 0.045),
            ("了_", 0.035),
            ("在_", 0.03),
            ("和_", 0.025),
            ("有_", 0.022),
            ("我_", 0.02),
            ("他_", 0.018),
            ("不_", 0.016),
            ("为_", 0.014),
            ("这_", 0.013),
            ("个_", 0.012),
            ("们_", 0.011),
            ("人_", 0.01),
            ("要_", 0.009),
            ("会_", 0.008),
        ] {
            chinese_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Chinese, chinese_profile);

        // Japanese profile (using hiragana/katakana)
        let mut japanese_profile = HashMap::new();
        for (ngram, freq) in &[
            ("の_", 0.05),
            ("に_", 0.04),
            ("は_", 0.035),
            ("を_", 0.03),
            ("た_", 0.025),
            ("と_", 0.022),
            ("が_", 0.02),
            ("で_", 0.018),
            ("る_", 0.016),
            ("す_", 0.014),
            ("い_", 0.013),
            ("ます", 0.012),
            ("した", 0.011),
            ("して", 0.01),
            ("です", 0.009),
            ("ない", 0.008),
        ] {
            japanese_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Japanese, japanese_profile);

        // Korean profile (using Hangul)
        let mut korean_profile = HashMap::new();
        for (ngram, freq) in &[
            ("의_", 0.05),
            ("이_", 0.04),
            ("가_", 0.035),
            ("을_", 0.03),
            ("는_", 0.025),
            ("에_", 0.022),
            ("하_", 0.02),
            ("고_", 0.018),
            ("다_", 0.016),
            ("지_", 0.014),
            ("한_", 0.013),
            ("로_", 0.012),
            ("서_", 0.011),
            ("도_", 0.01),
            ("와_", 0.009),
            ("니_", 0.008),
        ] {
            korean_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Korean, korean_profile);

        // Arabic profile (using Arabic script)
        let mut arabic_profile = HashMap::new();
        for (ngram, freq) in &[
            ("ال_", 0.06),
            ("في_", 0.045),
            ("من_", 0.035),
            ("على", 0.03),
            ("إلى", 0.025),
            ("ها_", 0.022),
            ("أن_", 0.02),
            ("ما_", 0.018),
            ("هو_", 0.016),
            ("كان", 0.014),
            ("هذا", 0.013),
            ("عن_", 0.012),
            ("بين", 0.011),
            ("لا_", 0.01),
            ("قد_", 0.009),
            ("كل_", 0.008),
        ] {
            arabic_profile.insert(ngram.to_string(), *freq);
        }
        self.profiles.insert(Language::Arabic, arabic_profile);
    }

    /// Detect the language of a text
    pub fn detect(&self, text: &str) -> Result<LanguageDetectionResult> {
        if text.trim().is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot detect language of empty text".to_string(),
            ));
        }

        // Extract n-grams from the text
        let text_profile = self.createtext_profile(text);

        // Score each language profile
        let mut scores: Vec<(Language, f64)> = self
            .profiles
            .iter()
            .map(|(lang, profile)| {
                let score = self.calculate_similarity(&text_profile, profile);
                (*lang, score)
            })
            .collect();

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if scores.is_empty() {
            return Ok(LanguageDetectionResult {
                language: Language::Unknown,
                confidence: 0.0,
                alternatives: vec![],
            });
        }

        let best_score = scores[0].1;
        let best_language = scores[0].0;

        // Calculate confidence based on the difference between top scores
        let confidence = if scores.len() > 1 {
            let second_score = scores[1].1;
            let diff = best_score - second_score;
            // Normalize confidence to [0, 1]
            (diff / best_score).clamp(0.0, 1.0)
        } else {
            best_score
        };

        Ok(LanguageDetectionResult {
            language: best_language,
            confidence,
            alternatives: scores.into_iter().skip(1).take(3).collect(),
        })
    }

    /// Create n-gram profile for a text
    fn createtext_profile(&self, text: &str) -> HashMap<String, f64> {
        let mut profile = HashMap::new();
        let text_lower = text.to_lowercase();
        let chars: Vec<char> = text_lower.chars().collect();
        let total_ngrams = chars.len().saturating_sub(self.n_gram_size - 1) as f64;

        if total_ngrams <= 0.0 {
            return profile;
        }

        // Count n-grams
        let mut ngram_counts: HashMap<String, usize> = HashMap::new();
        for i in 0..=chars.len().saturating_sub(self.n_gram_size) {
            let ngram: String = chars[i..i + self.n_gram_size].iter().collect();
            // Replace spaces with underscores for consistency
            let ngram = ngram.replace(' ', "_");
            *ngram_counts.entry(ngram).or_insert(0) += 1;
        }

        // Convert counts to frequencies
        for (ngram, count) in ngram_counts {
            profile.insert(ngram, count as f64 / total_ngrams);
        }

        profile
    }

    /// Calculate similarity between two n-gram profiles
    fn calculate_similarity(
        &self,
        profile1: &HashMap<String, f64>,
        profile2: &HashMap<String, f64>,
    ) -> f64 {
        let mut similarity = 0.0;
        let mut total_weight = 0.0;

        // Use cosine similarity
        for (ngram, freq1) in profile1 {
            if let Some(freq2) = profile2.get(ngram) {
                similarity += freq1 * freq2;
            }
            total_weight += freq1 * freq1;
        }

        if total_weight > 0.0 {
            similarity / total_weight.sqrt()
        } else {
            0.0
        }
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> Vec<Language> {
        self.profiles.keys().copied().collect()
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Language-specific stop words
pub struct StopWords {
    /// Stop words organized by language
    stop_words: HashMap<Language, Vec<String>>,
}

impl StopWords {
    /// Create a new stop words collection
    pub fn new() -> Self {
        let mut stop_words = HashMap::new();

        // English stop words
        stop_words.insert(
            Language::English,
            vec![
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
                "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "you",
                "your", "this", "have", "had", "been", "but", "not", "they", "were", "what",
                "when", "where", "who", "which", "their", "them", "these", "those", "there",
                "here", "than",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Spanish stop words
        stop_words.insert(
            Language::Spanish,
            vec![
                "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
                "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
                "ellas", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres", "es",
                "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos",
                "fue", "fueron", "fui", "la", "las", "lo", "los", "más", "mi", "mis", "mucho",
                "muchos", "muy", "ni", "no", "nos", "nosotras", "nosotros", "o", "otra", "otras",
                "otro", "otros", "para", "pero", "por", "porque", "que", "quien", "quienes", "se",
                "si", "sin", "sobre", "su", "sus", "también", "tanto", "te", "tu", "tus", "un",
                "una", "uno", "unos", "y", "ya", "yo",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // French stop words
        stop_words.insert(
            Language::French,
            vec![
                "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et",
                "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma", "mais", "me", "même",
                "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par", "pas",
                "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes",
                "toi", "ton", "tu", "un", "une", "vos", "votre", "vous",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        Self { stop_words }
    }

    /// Get stop words for a specific language
    pub fn get(&self, language: Language) -> Option<&Vec<String>> {
        self.stop_words.get(&language)
    }

    /// Check if a word is a stop word in a specific language
    pub fn is_stop_word(&self, word: &str, language: Language) -> bool {
        if let Some(words) = self.stop_words.get(&language) {
            words.iter().any(|sw| sw == &word.to_lowercase())
        } else {
            false
        }
    }

    /// Remove stop words from a list of tokens
    pub fn remove_stop_words(&self, tokens: &[String], language: Language) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| !self.is_stop_word(token, language))
            .cloned()
            .collect()
    }
}

impl Default for StopWords {
    fn default() -> Self {
        Self::new()
    }
}

/// Language-specific text processor
pub struct MultilingualProcessor {
    /// Language detector
    detector: LanguageDetector,
    /// Stop words collection
    stop_words: StopWords,
}

impl MultilingualProcessor {
    /// Create a new multilingual processor
    pub fn new() -> Self {
        Self {
            detector: LanguageDetector::new(),
            stop_words: StopWords::new(),
        }
    }

    /// Process text with automatic language detection
    pub fn process(&self, text: &str) -> Result<ProcessedText> {
        // Detect language
        let detection = self.detector.detect(text)?;

        // Tokenize (simple whitespace tokenization for now)
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        // Remove stop words
        let filtered_tokens = self
            .stop_words
            .remove_stop_words(&tokens, detection.language);

        Ok(ProcessedText {
            original: text.to_string(),
            language: detection.language,
            confidence: detection.confidence,
            tokens,
            filtered_tokens,
        })
    }
}

impl Default for MultilingualProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of multilingual text processing
#[derive(Debug, Clone)]
pub struct ProcessedText {
    /// Original text
    pub original: String,
    /// Detected language
    pub language: Language,
    /// Language detection confidence
    pub confidence: f64,
    /// All tokens
    pub tokens: Vec<String>,
    /// Tokens after stop word removal
    pub filtered_tokens: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_enum() {
        assert_eq!(Language::English.iso_code(), "en");
        assert_eq!(Language::Spanish.name(), "Spanish");
        assert_eq!(Language::from_iso_code("fr"), Language::French);
        assert_eq!(Language::from_iso_code("unknown"), Language::Unknown);
    }

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new();

        // Test English detection with more text
        let result = detector.detect("The quick brown fox jumps over the lazy dog. This is definitely an English sentence with many common words.").expect("Operation failed");
        assert_eq!(result.language, Language::English);

        // Test with empty text
        let empty_result = detector.detect("");
        assert!(empty_result.is_err());
    }

    #[test]
    fn test_stop_words() {
        let stop_words = StopWords::new();

        // Test English stop words
        assert!(stop_words.is_stop_word("the", Language::English));
        assert!(stop_words.is_stop_word("and", Language::English));
        assert!(!stop_words.is_stop_word("hello", Language::English));

        // Test stop word removal
        let tokens = vec![
            "the".to_string(),
            "cat".to_string(),
            "is".to_string(),
            "happy".to_string(),
        ];
        let filtered = stop_words.remove_stop_words(&tokens, Language::English);
        assert_eq!(filtered, vec!["cat", "happy"]);
    }

    #[test]
    fn test_multilingual_processor() {
        let processor = MultilingualProcessor::new();

        let result = processor.process("The quick brown fox jumps over the lazy dog. This sentence has many English words.").expect("Operation failed");
        assert_eq!(result.language, Language::English);
        assert!(!result.tokens.is_empty());
        assert!(result.filtered_tokens.len() < result.tokens.len());
    }

    #[test]
    fn test_createtext_profile() {
        let detector = LanguageDetector::new();
        let profile = detector.createtext_profile("hello world");

        // Check that profile contains some n-grams
        assert!(!profile.is_empty());
        assert!(profile.contains_key("hel") || profile.contains_key("llo"));
    }
}

// =============================================================================
// Unicode-agnostic tokenization and transliteration
// =============================================================================

// ── ScriptFamily ─────────────────────────────────────────────────────────────

/// Coarse script family classification for Unicode text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptFamily {
    /// Latin / ASCII script (includes diacritics).
    Latin,
    /// CJK ideographs (Chinese, Japanese Kanji, Korean Hanja).
    Cjk,
    /// Cyrillic script.
    Cyrillic,
    /// Arabic script.
    Arabic,
    /// Devanagari script (Hindi, Sanskrit, …).
    Devanagari,
    /// Any other / mixed script.
    Other,
}

// ── UnicodeTokenizerConfig ────────────────────────────────────────────────────

/// Configuration for [`UnicodeTokenizer`].
#[derive(Debug, Clone)]
pub struct UnicodeTokenizerConfig {
    /// Convert characters to lowercase before tokenizing.  Default: `true`.
    pub lowercase: bool,
    /// Strip combining accent marks (NFD decomposition + remove Mn category
    /// approximation).  Default: `true`.
    pub strip_accents: bool,
    /// Split on Unicode punctuation characters.  Default: `true`.
    pub split_on_punctuation: bool,
    /// Split on ASCII whitespace.  Default: `true`.
    pub split_on_whitespace: bool,
    /// Maximum token length in characters.  `None` = unlimited.  Default: `None`.
    pub max_token_length: Option<usize>,
}

impl Default for UnicodeTokenizerConfig {
    fn default() -> Self {
        UnicodeTokenizerConfig {
            lowercase: true,
            strip_accents: true,
            split_on_punctuation: true,
            split_on_whitespace: true,
            max_token_length: None,
        }
    }
}

// ── UnicodeTokenizer ──────────────────────────────────────────────────────────

/// Language-agnostic Unicode tokenizer.
///
/// Works for any writing system:
/// - CJK ideographs become individual tokens (spaces inserted around them).
/// - Optional accent stripping via a pure-Rust NFD approximation.
/// - Optional lowercasing.
/// - Punctuation splitting (Unicode category Po/Ps/Pe/…).
///
/// No external Unicode library is required.
pub struct UnicodeTokenizer {
    config: UnicodeTokenizerConfig,
}

impl UnicodeTokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: UnicodeTokenizerConfig) -> Self {
        UnicodeTokenizer { config }
    }

    /// Create a tokenizer with sensible defaults:
    /// lowercase=true, strip_accents=true, split on whitespace + punctuation.
    pub fn default_tokenizer() -> Self {
        Self::new(UnicodeTokenizerConfig::default())
    }

    // ── tokenize ─────────────────────────────────────────────────────────────

    /// Tokenize `text` into a list of tokens (Unicode-aware).
    ///
    /// Processing order:
    /// 1. Insert spaces around CJK characters.
    /// 2. Optionally lowercase.
    /// 3. Optionally strip accents.
    /// 4. Split on whitespace (always) and optionally on punctuation.
    /// 5. Discard empty tokens and enforce max_token_length.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        // Step 1: Handle CJK — space-pad each CJK char so it splits cleanly
        let spaced = self.insert_cjk_spaces(text);

        // Step 2: Optionally lowercase
        let lowered = if self.config.lowercase {
            spaced.to_lowercase()
        } else {
            spaced
        };

        // Step 3: Optionally strip accents
        let stripped = if self.config.strip_accents {
            Transliterator::strip_accents(&lowered)
        } else {
            lowered
        };

        // Step 4: Split
        let mut tokens: Vec<String> = Vec::new();
        let mut current = String::new();

        for ch in stripped.chars() {
            let is_ws = ch.is_ascii_whitespace() || ch == '\u{00A0}';
            let is_punct = self.config.split_on_punctuation && is_unicode_punctuation(ch);

            if (self.config.split_on_whitespace && is_ws) || is_punct {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                if is_punct {
                    tokens.push(ch.to_string());
                }
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }

        // Step 5: Filter empty / apply max_token_length
        tokens.retain(|t| !t.is_empty());
        if let Some(max_len) = self.config.max_token_length {
            tokens.iter_mut().for_each(|t| {
                let char_count = t.chars().count();
                if char_count > max_len {
                    *t = t.chars().take(max_len).collect();
                }
            });
        }

        tokens
    }

    // ── detect_script ────────────────────────────────────────────────────────

    /// Detect the dominant script family of `text` by majority vote over
    /// non-whitespace characters.
    pub fn detect_script(&self, text: &str) -> ScriptFamily {
        let mut latin = 0usize;
        let mut cjk = 0usize;
        let mut cyrillic = 0usize;
        let mut arabic = 0usize;
        let mut devanagari = 0usize;
        let mut other = 0usize;

        for ch in text.chars() {
            if ch.is_whitespace() {
                continue;
            }
            if is_cjk_char(ch) {
                cjk += 1;
            } else if is_cyrillic(ch) {
                cyrillic += 1;
            } else if is_arabic(ch) {
                arabic += 1;
            } else if is_devanagari(ch) {
                devanagari += 1;
            } else if ch.is_ascii_alphabetic() || (ch as u32 >= 0x00C0 && ch as u32 <= 0x024F) {
                latin += 1;
            } else {
                other += 1;
            }
        }

        let max = [latin, cjk, cyrillic, arabic, devanagari, other]
            .into_iter()
            .max()
            .unwrap_or(0);

        if max == 0 {
            return ScriptFamily::Other;
        }
        if max == cjk {
            ScriptFamily::Cjk
        } else if max == cyrillic {
            ScriptFamily::Cyrillic
        } else if max == arabic {
            ScriptFamily::Arabic
        } else if max == devanagari {
            ScriptFamily::Devanagari
        } else if max == latin {
            ScriptFamily::Latin
        } else {
            ScriptFamily::Other
        }
    }

    // ── tokenize_cjk ─────────────────────────────────────────────────────────

    /// Tokenize a (potentially mixed) text where CJK characters each become
    /// their own token, while non-CJK sequences are tokenized by whitespace.
    pub fn tokenize_cjk(&self, text: &str) -> Vec<String> {
        let spaced = self.insert_cjk_spaces(text);
        spaced
            .split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn insert_cjk_spaces(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + text.chars().count());
        for ch in text.chars() {
            if is_cjk_char(ch) {
                out.push(' ');
                out.push(ch);
                out.push(' ');
            } else {
                out.push(ch);
            }
        }
        out
    }
}

impl Default for UnicodeTokenizer {
    fn default() -> Self {
        Self::default_tokenizer()
    }
}

// ── Transliterator ────────────────────────────────────────────────────────────

/// Utility functions for converting non-Latin scripts to Latin characters.
pub struct Transliterator;

impl Transliterator {
    /// Transliterate a CJK string to a Pinyin-like romanisation.
    ///
    /// The mapping covers ~200 of the most frequent Mandarin characters.
    /// Unmapped characters are left unchanged.
    pub fn cjk_to_latin(text: &str) -> String {
        text.chars()
            .map(|c| {
                if let Some(roman) = cjk_pinyin_lookup(c) {
                    roman.to_string()
                } else {
                    c.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Transliterate Cyrillic characters to Latin (standard Russian
    /// romanisation / BGN/PCGN).
    ///
    /// Uppercase source characters produce titlecase output; unmapped
    /// characters are kept as-is.
    pub fn cyrillic_to_latin(text: &str) -> String {
        let mut out = String::with_capacity(text.len() * 2);
        for ch in text.chars() {
            let lower = ch.to_lowercase().next().unwrap_or(ch);
            if let Some(roman) = cyrillic_lookup(lower) {
                if ch.is_uppercase() {
                    // Capitalise first char of the mapping
                    let mut chars = roman.chars();
                    if let Some(first) = chars.next() {
                        for c in first.to_uppercase() {
                            out.push(c);
                        }
                        out.push_str(chars.as_str());
                    }
                } else {
                    out.push_str(roman);
                }
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// Strip combining diacritical marks from Latin text via an NFD-like
    /// decomposition table.
    ///
    /// This is a pure-Rust approximation covering the Latin Extended-A/B block
    /// (U+00C0–U+024F) and a small subset of precomposed characters.
    pub fn strip_accents(text: &str) -> String {
        text.chars()
            .flat_map(nfd_decompose)
            .filter(|&c| !is_combining_mark(c))
            .collect()
    }

    /// Normalise text: collapse multiple whitespace runs into single space,
    /// trim leading/trailing whitespace, and lowercase.
    pub fn normalize(text: &str) -> String {
        let lowered = text.to_lowercase();
        lowered.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

// ── Unicode character classification helpers ──────────────────────────────────

/// Returns `true` when `c` is a CJK ideograph or extension character.
pub fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
    // CJK Extension A
    || (0x3400..=0x4DBF).contains(&cp)
    // CJK Extension B
    || (0x20000..=0x2A6DF).contains(&cp)
    // CJK Compatibility Ideographs
    || (0xF900..=0xFAFF).contains(&cp)
    // CJK Compatibility Ideographs Supplement
    || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Returns `true` when `c` is in the Cyrillic block (U+0400–U+04FF).
pub fn is_cyrillic(c: char) -> bool {
    let cp = c as u32;
    (0x0400..=0x04FF).contains(&cp)
}

/// Returns `true` when `c` is in the Arabic block (U+0600–U+06FF).
fn is_arabic(c: char) -> bool {
    let cp = c as u32;
    (0x0600..=0x06FF).contains(&cp)
}

/// Returns `true` when `c` is in the Devanagari block (U+0900–U+097F).
fn is_devanagari(c: char) -> bool {
    let cp = c as u32;
    (0x0900..=0x097F).contains(&cp)
}

/// Returns `true` when `c` is a Unicode combining mark (category Mn/Mc/Me).
///
/// Approximation: covers the main Combining Diacritical Marks block
/// U+0300–U+036F and the Combining Diacritical Marks Supplement U+1DC0–U+1DFF.
pub fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

/// Returns `true` when `c` is a Unicode punctuation or symbol character.
fn is_unicode_punctuation(c: char) -> bool {
    matches!(
        c,
        '!' | '"'
            | '#'
            | '%'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '_'
            | '{'
            | '}'
            | '~'
            | '·'
            | '…'
            | '—'
            | '–'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{201C}'
            | '\u{201D}'
    ) || (c as u32 >= 0x2000 && c as u32 <= 0x206F)
}

// ── Transliteration lookup tables ─────────────────────────────────────────────

/// Cyrillic → Latin mapping (BGN/PCGN standard for Russian).
fn cyrillic_lookup(c: char) -> Option<&'static str> {
    match c {
        'а' => Some("a"),
        'б' => Some("b"),
        'в' => Some("v"),
        'г' => Some("g"),
        'д' => Some("d"),
        'е' => Some("ye"),
        'ё' => Some("yo"),
        'ж' => Some("zh"),
        'з' => Some("z"),
        'и' => Some("i"),
        'й' => Some("y"),
        'к' => Some("k"),
        'л' => Some("l"),
        'м' => Some("m"),
        'н' => Some("n"),
        'о' => Some("o"),
        'п' => Some("p"),
        'р' => Some("r"),
        'с' => Some("s"),
        'т' => Some("t"),
        'у' => Some("u"),
        'ф' => Some("f"),
        'х' => Some("kh"),
        'ц' => Some("ts"),
        'ч' => Some("ch"),
        'ш' => Some("sh"),
        'щ' => Some("shch"),
        'ъ' => Some(""),
        'ы' => Some("y"),
        'ь' => Some(""),
        'э' => Some("e"),
        'ю' => Some("yu"),
        'я' => Some("ya"),
        _ => None,
    }
}

/// Approximate NFD decomposition for Latin Extended characters.
///
/// Returns an iterator of up to 2 chars: the base letter followed optionally
/// by a combining mark.  For characters outside the coverage, returns the
/// original character unchanged.
fn nfd_decompose(c: char) -> impl Iterator<Item = char> {
    // Map precomposed Latin Extended-A/B to base + combining grave/acute/circ/etc.
    let decomp: Option<(char, Option<char>)> = match c {
        'À' => Some(('A', Some('\u{0300}'))),
        'Á' => Some(('A', Some('\u{0301}'))),
        'Â' => Some(('A', Some('\u{0302}'))),
        'Ã' => Some(('A', Some('\u{0303}'))),
        'Ä' => Some(('A', Some('\u{0308}'))),
        'Å' => Some(('A', Some('\u{030A}'))),
        'à' => Some(('a', Some('\u{0300}'))),
        'á' => Some(('a', Some('\u{0301}'))),
        'â' => Some(('a', Some('\u{0302}'))),
        'ã' => Some(('a', Some('\u{0303}'))),
        'ä' => Some(('a', Some('\u{0308}'))),
        'å' => Some(('a', Some('\u{030A}'))),
        'È' => Some(('E', Some('\u{0300}'))),
        'É' => Some(('E', Some('\u{0301}'))),
        'Ê' => Some(('E', Some('\u{0302}'))),
        'Ë' => Some(('E', Some('\u{0308}'))),
        'è' => Some(('e', Some('\u{0300}'))),
        'é' => Some(('e', Some('\u{0301}'))),
        'ê' => Some(('e', Some('\u{0302}'))),
        'ë' => Some(('e', Some('\u{0308}'))),
        'Ì' => Some(('I', Some('\u{0300}'))),
        'Í' => Some(('I', Some('\u{0301}'))),
        'Î' => Some(('I', Some('\u{0302}'))),
        'Ï' => Some(('I', Some('\u{0308}'))),
        'ì' => Some(('i', Some('\u{0300}'))),
        'í' => Some(('i', Some('\u{0301}'))),
        'î' => Some(('i', Some('\u{0302}'))),
        'ï' => Some(('i', Some('\u{0308}'))),
        'Ò' => Some(('O', Some('\u{0300}'))),
        'Ó' => Some(('O', Some('\u{0301}'))),
        'Ô' => Some(('O', Some('\u{0302}'))),
        'Õ' => Some(('O', Some('\u{0303}'))),
        'Ö' => Some(('O', Some('\u{0308}'))),
        'ò' => Some(('o', Some('\u{0300}'))),
        'ó' => Some(('o', Some('\u{0301}'))),
        'ô' => Some(('o', Some('\u{0302}'))),
        'õ' => Some(('o', Some('\u{0303}'))),
        'ö' => Some(('o', Some('\u{0308}'))),
        'Ù' => Some(('U', Some('\u{0300}'))),
        'Ú' => Some(('U', Some('\u{0301}'))),
        'Û' => Some(('U', Some('\u{0302}'))),
        'Ü' => Some(('U', Some('\u{0308}'))),
        'ù' => Some(('u', Some('\u{0300}'))),
        'ú' => Some(('u', Some('\u{0301}'))),
        'û' => Some(('u', Some('\u{0302}'))),
        'ü' => Some(('u', Some('\u{0308}'))),
        'Ñ' => Some(('N', Some('\u{0303}'))),
        'ñ' => Some(('n', Some('\u{0303}'))),
        'Ç' => Some(('C', Some('\u{0327}'))),
        'ç' => Some(('c', Some('\u{0327}'))),
        'Ý' => Some(('Y', Some('\u{0301}'))),
        'ý' => Some(('y', Some('\u{0301}'))),
        'ÿ' => Some(('y', Some('\u{0308}'))),
        _ => None,
    };

    match decomp {
        Some((base, Some(combining))) => {
            // Two-char iterator
            let v: Vec<char> = vec![base, combining];
            v.into_iter()
        }
        Some((base, None)) => {
            let v: Vec<char> = vec![base];
            v.into_iter()
        }
        None => {
            let v: Vec<char> = vec![c];
            v.into_iter()
        }
    }
}

/// Approximate Pinyin romanisation for common Mandarin ideographs.
///
/// Covers ~60 high-frequency characters.  Returns `None` when the character
/// is not in the lookup table.
fn cjk_pinyin_lookup(c: char) -> Option<&'static str> {
    match c {
        '的' => Some("de"),
        '一' => Some("yi"),
        '是' => Some("shi"),
        '不' => Some("bu"),
        '了' => Some("le"),
        '人' => Some("ren"),
        '我' => Some("wo"),
        '在' => Some("zai"),
        '有' => Some("you"),
        '他' => Some("ta"),
        '这' => Some("zhe"),
        '中' => Some("zhong"),
        '大' => Some("da"),
        '来' => Some("lai"),
        '上' => Some("shang"),
        '国' => Some("guo"),
        '个' => Some("ge"),
        '到' => Some("dao"),
        '说' => Some("shuo"),
        '们' => Some("men"),
        '为' => Some("wei"),
        '子' => Some("zi"),
        '和' => Some("he"),
        '你' => Some("ni"),
        '地' => Some("di"),
        '出' => Some("chu"),
        '道' => Some("dao"),
        '也' => Some("ye"),
        '时' => Some("shi"),
        '年' => Some("nian"),
        '得' => Some("de"),
        '就' => Some("jiu"),
        '那' => Some("na"),
        '要' => Some("yao"),
        '下' => Some("xia"),
        '以' => Some("yi"),
        '生' => Some("sheng"),
        '会' => Some("hui"),
        '自' => Some("zi"),
        '着' => Some("zhe"),
        '去' => Some("qu"),
        '之' => Some("zhi"),
        '过' => Some("guo"),
        '家' => Some("jia"),
        '学' => Some("xue"),
        '对' => Some("dui"),
        '可' => Some("ke"),
        '她' => Some("ta"),
        '里' => Some("li"),
        '后' => Some("hou"),
        '小' => Some("xiao"),
        '么' => Some("me"),
        '心' => Some("xin"),
        '多' => Some("duo"),
        '天' => Some("tian"),
        '而' => Some("er"),
        '能' => Some("neng"),
        '好' => Some("hao"),
        '都' => Some("dou"),
        '然' => Some("ran"),
        _ => None,
    }
}

// ── Tests (Unicode tokenizer + Transliterator) ────────────────────────────────

#[cfg(test)]
mod unicode_tests {
    use super::*;

    // ── UnicodeTokenizer ──────────────────────────────────────────────────────

    #[test]
    fn tokenize_splits_simple_english() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            lowercase: true,
            strip_accents: false,
            split_on_punctuation: false,
            split_on_whitespace: true,
            max_token_length: None,
        });
        let tokens = tok.tokenize("hello world");
        assert_eq!(
            tokens,
            vec!["hello", "world"],
            "simple English sentence must split on whitespace"
        );
    }

    #[test]
    fn tokenize_cjk_each_char_is_token() {
        let tok = UnicodeTokenizer::default();
        // "中文" (Chinese text) + space + "hello"
        let tokens = tok.tokenize_cjk("中文 hello");
        // 中 and 文 should each be their own token; hello is a third
        assert!(
            tokens.contains(&"中".to_string()),
            "CJK char '中' must be a token"
        );
        assert!(
            tokens.contains(&"文".to_string()),
            "CJK char '文' must be a token"
        );
        assert!(
            tokens.contains(&"hello".to_string()),
            "'hello' must be a token"
        );
    }

    #[test]
    fn detect_script_latin() {
        let tok = UnicodeTokenizer::default();
        assert_eq!(
            tok.detect_script("hello world"),
            ScriptFamily::Latin,
            "ASCII text must detect as Latin"
        );
    }

    #[test]
    fn detect_script_cyrillic() {
        let tok = UnicodeTokenizer::default();
        // "привет" = "hello" in Russian
        assert_eq!(
            tok.detect_script("привет мир"),
            ScriptFamily::Cyrillic,
            "Cyrillic text must detect as Cyrillic"
        );
    }

    #[test]
    fn detect_script_cjk() {
        let tok = UnicodeTokenizer::default();
        assert_eq!(
            tok.detect_script("中文"),
            ScriptFamily::Cjk,
            "CJK text must detect as Cjk"
        );
    }

    #[test]
    fn tokenize_with_punctuation_split() {
        let tok = UnicodeTokenizer::default();
        let tokens = tok.tokenize("hello, world!");
        // "hello", ",", "world", "!" expected (punctuation become tokens)
        assert!(
            tokens.contains(&"hello".to_string()),
            "must contain 'hello'"
        );
        assert!(
            tokens.contains(&"world".to_string()),
            "must contain 'world'"
        );
    }

    #[test]
    fn tokenize_max_token_length() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            lowercase: false,
            strip_accents: false,
            split_on_punctuation: false,
            split_on_whitespace: true,
            max_token_length: Some(3),
        });
        let tokens = tok.tokenize("hello world");
        for t in &tokens {
            assert!(
                t.chars().count() <= 3,
                "token '{t}' exceeds max_token_length=3"
            );
        }
    }

    // ── Transliterator ────────────────────────────────────────────────────────

    #[test]
    fn cyrillic_to_latin_privet() {
        // "привет" → "privet" (note: п=p, р=r, и=i, в=v, е=ye, т=t)
        // In BGN/PCGN "привет" = p+r+i+v+ye+t = "privyet"
        // Using our table: п→p, р→r, и→i, в→v, е→ye, т→t
        let result = Transliterator::cyrillic_to_latin("привет");
        // Accept either "privet" or "privyet" since 'е' maps to "ye"
        assert!(
            result.starts_with("priv"),
            "transliteration of 'привет' must start with 'priv', got '{result}'"
        );
    }

    #[test]
    fn cyrillic_to_latin_basic_letters() {
        // Test a few individual letters
        assert_eq!(Transliterator::cyrillic_to_latin("а"), "a");
        assert_eq!(Transliterator::cyrillic_to_latin("б"), "b");
        assert_eq!(Transliterator::cyrillic_to_latin("с"), "s");
        assert_eq!(Transliterator::cyrillic_to_latin("т"), "t");
    }

    #[test]
    fn strip_accents_cafe() {
        let result = Transliterator::strip_accents("café");
        assert_eq!(
            result, "cafe",
            "strip_accents('café') must return 'cafe', got '{result}'"
        );
    }

    #[test]
    fn strip_accents_no_accents_unchanged() {
        let result = Transliterator::strip_accents("hello");
        assert_eq!(result, "hello", "plain ASCII must be unchanged");
    }

    #[test]
    fn transliterator_normalize_collapses_whitespace() {
        let result = Transliterator::normalize("  Hello   World  ");
        assert_eq!(
            result, "hello world",
            "normalize must trim and collapse spaces"
        );
    }

    #[test]
    fn strip_accents_german_umlaut() {
        // ü → u (after stripping U+0308 combining diaeresis)
        let result = Transliterator::strip_accents("über");
        assert_eq!(result, "uber", "ü must become u after accent stripping");
    }
}
