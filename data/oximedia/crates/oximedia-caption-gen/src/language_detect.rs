//! Language detection for caption transcript text.
//!
//! Auto-detects the natural language of a transcript so that downstream
//! components (line-breaking, hyphenation, CPS limits) can apply
//! locale-specific rules.
//!
//! ## Method
//!
//! Detection uses a **byte-trigram language model** built from a small set of
//! high-frequency trigrams curated for each supported language.  No Unicode
//! statistical tables are loaded from disk — everything is compiled in.
//!
//! For short inputs (< 30 characters) the detector falls back to
//! `LanguageCode::Unknown` rather than guessing.
//!
//! ## Supported languages (BCP-47 tags)
//!
//! | Code  | Language   |
//! |-------|------------|
//! | `en`  | English    |
//! | `es`  | Spanish    |
//! | `fr`  | French     |
//! | `de`  | German     |
//! | `it`  | Italian    |
//! | `pt`  | Portuguese |
//! | `nl`  | Dutch      |
//! | `ja`  | Japanese   |
//! | `zh`  | Chinese    |
//! | `ko`  | Korean     |
//! | `ar`  | Arabic     |
//! | `ru`  | Russian    |

use std::collections::HashMap;

// ─── Language code ─────────────────────────────────────────────────────────────

/// A BCP-47 language code.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageCode(pub String);

impl LanguageCode {
    /// English.
    pub fn en() -> Self {
        Self("en".to_string())
    }
    /// Spanish.
    pub fn es() -> Self {
        Self("es".to_string())
    }
    /// French.
    pub fn fr() -> Self {
        Self("fr".to_string())
    }
    /// German.
    pub fn de() -> Self {
        Self("de".to_string())
    }
    /// Italian.
    pub fn it() -> Self {
        Self("it".to_string())
    }
    /// Portuguese.
    pub fn pt() -> Self {
        Self("pt".to_string())
    }
    /// Dutch.
    pub fn nl() -> Self {
        Self("nl".to_string())
    }
    /// Japanese.
    pub fn ja() -> Self {
        Self("ja".to_string())
    }
    /// Chinese (Mandarin).
    pub fn zh() -> Self {
        Self("zh".to_string())
    }
    /// Korean.
    pub fn ko() -> Self {
        Self("ko".to_string())
    }
    /// Arabic.
    pub fn ar() -> Self {
        Self("ar".to_string())
    }
    /// Russian.
    pub fn ru() -> Self {
        Self("ru".to_string())
    }
    /// Unknown / undetected.
    pub fn unknown() -> Self {
        Self("und".to_string())
    }

    /// Whether this code represents an unknown language.
    pub fn is_unknown(&self) -> bool {
        self.0 == "und"
    }

    /// Whether text in this language is written right-to-left.
    pub fn is_rtl(&self) -> bool {
        matches!(self.0.as_str(), "ar" | "he" | "fa" | "ur")
    }

    /// Whether this language uses CJK (logographic) script, which affects
    /// line-breaking (no spaces between words).
    pub fn is_cjk(&self) -> bool {
        matches!(self.0.as_str(), "ja" | "zh" | "ko")
    }
}

impl std::fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─── Detection result ─────────────────────────────────────────────────────────

/// The result of a language detection operation.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// The most likely language code.
    pub language: LanguageCode,
    /// Normalised confidence score in \[0.0, 1.0\].
    pub confidence: f32,
    /// Top-3 alternative candidates, ordered by descending score.
    pub alternatives: Vec<(LanguageCode, f32)>,
}

impl DetectionResult {
    /// Returns `true` when the detected language is reliable (confidence ≥ 0.50).
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.50
    }
}

// ─── Language profiles ────────────────────────────────────────────────────────

/// A trigram language profile: language code + list of high-frequency trigrams.
struct LangProfile {
    code: &'static str,
    trigrams: &'static [&'static str],
}

/// Build the static trigram profiles.
///
/// Each language has a curated set of highly discriminative byte-trigrams
/// derived from common function words and inflectional morphology.
fn build_profiles() -> Vec<LangProfile> {
    vec![
        LangProfile {
            code: "en",
            trigrams: &[
                "the", " th", "he ", "ing", " an", "nd ", "ed ", " in", "nt ", "ion", " of", "of ",
                "tio", " to", "to ", " is", "is ", " it", " a ", "er ", "hat", "at ", "tha", "his",
                "he ", "and", "re ", "on ", "ing", "ent",
            ],
        },
        LangProfile {
            code: "es",
            trigrams: &[
                " de", "de ", "que", " qu", "ue ", "la ", " la", "el ", " el", " en", "en ", "los",
                "os ", " lo", "es ", " es", "por", "or ", "ado", "ión", "nte", "con", "una", " un",
                "un ", "tra", "par",
            ],
        },
        LangProfile {
            code: "fr",
            trigrams: &[
                " de", "de ", " le", "le ", "les", " le", "que", " qu", "ent", " en", "en ", " la",
                "la ", "tion", "ion", "ons", "est", "es ", " es", "des", " de", "ons", "nt ",
                "une", " un", "un ", "ait", "pas", " pa",
            ],
        },
        LangProfile {
            code: "de",
            trigrams: &[
                "die", "ie ", " di", "der", "er ", " de", "und", "nd ", " un", "in ", " in", "en ",
                "den", " da", "das", "ich", "ch ", " ic", "ein", "in ", "ist", "st ", "cht", "sch",
                "che", "hen", "eit", "ung", "ng ", "mit",
            ],
        },
        LangProfile {
            code: "it",
            trigrams: &[
                " di", "di ", "del", " de", "la ", " la", "che", "he ", " ch", " il", "il ", "lla",
                "ell", "al ", " al", "una", " un", "un ", "con", "on ", "per", "er ", "ent", "nto",
                "zione", "ion", "ita", "ta ", "ato",
            ],
        },
        LangProfile {
            code: "pt",
            trigrams: &[
                " de", "de ", "que", " qu", "ue ", "da ", " da", " do", "do ", "os ", " os", " a ",
                "as ", " as", "com", "om ", "uma", "ma ", " um", "um ", "ção", "ões", "ara", "ra ",
                "nte", "ão ", "por", "or ", "ois",
            ],
        },
        LangProfile {
            code: "nl",
            trigrams: &[
                "de ", " de", "van", "an ", " va", "het", "et ", " he", "een", "en ", " ee", "in ",
                " in", "dat", "at ", " da", "ver", "er ", "aar", "ar ", "ing", "ng ", "ijk", "ijk",
                "oor", "or ", "ste", "te ", "men",
            ],
        },
        LangProfile {
            code: "ru",
            trigrams: &[
                "ого", "ого", "ние", "ие ", "ель", "ль ", "ати", "ти ", "ова", "ва ", "ена", "на ",
                "ест", "сть", "то ", " то", "ных", "ых ", "ого", " в ", "это", "что", "как", " на",
                "ной", "ой ", "ский", "ски", " не",
            ],
        },
        // CJK languages: detected via Unicode script ranges rather than trigrams.
        LangProfile {
            code: "ja",
            trigrams: &["は", "の", "に", "を", "が", "で", "て", "た", "し", "と"],
        },
        LangProfile {
            code: "zh",
            trigrams: &["的", "了", "在", "是", "我", "有", "和", "人", "这", "中"],
        },
        LangProfile {
            code: "ko",
            trigrams: &["이", "는", "을", "가", "에", "의", "로", "하", "을", "기"],
        },
        LangProfile {
            code: "ar",
            trigrams: &[
                "ال", "لا", "في", " في", "من", "من ", " من", "على", "لى ", "إلى", "هذا", "ذا ",
                "كان", "ان ", "الت", "الم", "الأ",
            ],
        },
    ]
}

// ─── Unicode script range helpers ────────────────────────────────────────────

/// Count characters in the CJK Unified Ideographs block + common extensions.
fn count_cjk_chars(text: &str) -> usize {
    text.chars()
        .filter(|&c| {
            // CJK Unified Ideographs: U+4E00–U+9FFF
            // CJK Extension A: U+3400–U+4DBF
            // Compatibility ideographs: U+F900–U+FAFF
            ('\u{4E00}'..='\u{9FFF}').contains(&c)
                || ('\u{3400}'..='\u{4DBF}').contains(&c)
                || ('\u{F900}'..='\u{FAFF}').contains(&c)
        })
        .count()
}

/// Count hiragana/katakana characters (distinctive of Japanese).
fn count_hiragana_katakana(text: &str) -> usize {
    text.chars()
        .filter(|&c| {
            ('\u{3040}'..='\u{309F}').contains(&c) // Hiragana
                || ('\u{30A0}'..='\u{30FF}').contains(&c) // Katakana
        })
        .count()
}

/// Count Hangul syllable characters (distinctive of Korean).
fn count_hangul(text: &str) -> usize {
    text.chars()
        .filter(|&c| ('\u{AC00}'..='\u{D7AF}').contains(&c))
        .count()
}

/// Count Arabic script characters.
fn count_arabic(text: &str) -> usize {
    text.chars()
        .filter(|&c| ('\u{0600}'..='\u{06FF}').contains(&c))
        .count()
}

/// Count Cyrillic characters (dominant in Russian, Bulgarian, etc.).
fn count_cyrillic(text: &str) -> usize {
    text.chars()
        .filter(|&c| ('\u{0400}'..='\u{04FF}').contains(&c))
        .count()
}

/// Script-based fast-path detection for non-Latin scripts.
///
/// Returns `Some((code, confidence))` when a dominant script is detected.
fn detect_by_script(text: &str) -> Option<(&'static str, f32)> {
    let total: usize = text.chars().filter(|c| !c.is_whitespace()).count();
    if total == 0 {
        return None;
    }

    let hiragana = count_hiragana_katakana(text);
    let hangul = count_hangul(text);
    let arabic = count_arabic(text);
    let cyrillic = count_cyrillic(text);
    let cjk = count_cjk_chars(text);

    // Hiragana/katakana is uniquely Japanese.
    if hiragana as f32 / total as f32 > 0.10 {
        return Some(("ja", 0.95));
    }
    // Korean Hangul syllables.
    if hangul as f32 / total as f32 > 0.15 {
        return Some(("ko", 0.95));
    }
    // Arabic script.
    if arabic as f32 / total as f32 > 0.15 {
        return Some(("ar", 0.92));
    }
    // Cyrillic (assume Russian as most common).
    if cyrillic as f32 / total as f32 > 0.30 {
        return Some(("ru", 0.85));
    }
    // CJK characters without hiragana/katakana → likely Chinese.
    if cjk as f32 / total as f32 > 0.20 {
        return Some(("zh", 0.88));
    }

    None
}

// ─── Trigram extraction ────────────────────────────────────────────────────────

/// Extract character trigrams from `text`, normalised to lowercase.
fn extract_trigrams(text: &str) -> HashMap<String, u32> {
    let normalised: String = text.chars().flat_map(|c| c.to_lowercase()).collect();
    let chars: Vec<char> = normalised.chars().collect();
    let mut counts: HashMap<String, u32> = HashMap::new();

    for window in chars.windows(3) {
        let tri: String = window.iter().collect();
        *counts.entry(tri).or_insert(0) += 1;
    }
    counts
}

// ─── Detector ─────────────────────────────────────────────────────────────────

/// Language detector.
pub struct LanguageDetector {
    profiles: Vec<LangProfile>,
    /// Minimum text length (in characters) required before returning a result.
    min_length: usize,
}

impl LanguageDetector {
    /// Create a detector with default settings.
    pub fn new() -> Self {
        Self {
            profiles: build_profiles(),
            min_length: 30,
        }
    }

    /// Create a detector with a custom minimum text length.
    pub fn with_min_length(min_length: usize) -> Self {
        Self {
            profiles: build_profiles(),
            min_length,
        }
    }

    /// Detect the language of `text`.
    ///
    /// Returns `LanguageCode::unknown()` with confidence 0.0 when the text is
    /// shorter than `min_length` characters.
    pub fn detect(&self, text: &str) -> DetectionResult {
        let char_count = text.chars().count();
        if char_count < self.min_length {
            return DetectionResult {
                language: LanguageCode::unknown(),
                confidence: 0.0,
                alternatives: Vec::new(),
            };
        }

        // Fast-path: detect non-Latin scripts via Unicode character ranges.
        if let Some((code, confidence)) = detect_by_script(text) {
            return DetectionResult {
                language: LanguageCode(code.to_string()),
                confidence,
                alternatives: Vec::new(),
            };
        }

        // Trigram-based scoring for Latin-script languages.
        let trigrams = extract_trigrams(text);
        // Only score Latin-script profiles (skip CJK/Arabic/Cyrillic profiles
        // since those are handled by the script detector above).
        let latin_codes: &[&str] = &["en", "es", "fr", "de", "it", "pt", "nl"];
        let mut scores: Vec<(LanguageCode, f32)> = self
            .profiles
            .iter()
            .filter(|p| latin_codes.contains(&p.code))
            .map(|p| {
                let score = self.score_against_profile(&trigrams, p);
                (LanguageCode(p.code.to_string()), score)
            })
            .collect();

        // Sort descending.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total: f32 = scores.iter().map(|(_, s)| s).sum();
        let normalised: Vec<(LanguageCode, f32)> = scores
            .iter()
            .map(|(code, score)| {
                let conf = if total > 0.0 { score / total } else { 0.0 };
                (code.clone(), conf)
            })
            .collect();

        let best = normalised
            .first()
            .cloned()
            .unwrap_or_else(|| (LanguageCode::unknown(), 0.0));

        let alternatives: Vec<(LanguageCode, f32)> =
            normalised.into_iter().skip(1).take(3).collect();

        DetectionResult {
            language: best.0,
            confidence: best.1,
            alternatives,
        }
    }

    /// Detect language from a slice of transcript segments by concatenating
    /// their text.
    pub fn detect_from_segments(
        &self,
        segments: &[crate::alignment::TranscriptSegment],
    ) -> DetectionResult {
        let combined: String = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        self.detect(&combined)
    }

    fn score_against_profile(&self, trigrams: &HashMap<String, u32>, profile: &LangProfile) -> f32 {
        let mut score = 0.0f32;
        for &tri in profile.trigrams {
            let count = trigrams.get(tri).copied().unwrap_or(0);
            score += count as f32;
        }
        score
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Locale-aware line-breaking hints ────────────────────────────────────────

/// Line-breaking recommendations for a given language.
#[derive(Debug, Clone)]
pub struct LineBreakHints {
    /// Recommended maximum characters per line for this language/script.
    pub max_chars_per_line: u8,
    /// Recommended maximum characters per second (reading speed).
    pub max_cps: f32,
    /// Whether word-boundary detection is needed (false for CJK).
    pub needs_word_boundary: bool,
    /// Whether text is RTL.
    pub rtl: bool,
}

impl LineBreakHints {
    /// Return locale-appropriate line-breaking hints for the given language.
    pub fn for_language(code: &LanguageCode) -> Self {
        match code.0.as_str() {
            "ja" | "zh" => Self {
                max_chars_per_line: 13, // CJK characters are wider
                max_cps: 7.0,
                needs_word_boundary: false,
                rtl: false,
            },
            "ko" => Self {
                max_chars_per_line: 15,
                max_cps: 9.0,
                needs_word_boundary: false,
                rtl: false,
            },
            "ar" => Self {
                max_chars_per_line: 40,
                max_cps: 14.0,
                needs_word_boundary: true,
                rtl: true,
            },
            "ru" => Self {
                max_chars_per_line: 40,
                max_cps: 15.0,
                needs_word_boundary: true,
                rtl: false,
            },
            "de" => Self {
                // German compound words are long.
                max_chars_per_line: 45,
                max_cps: 16.0,
                needs_word_boundary: true,
                rtl: false,
            },
            // Default (Latin scripts).
            _ => Self {
                max_chars_per_line: 42,
                max_cps: 17.0,
                needs_word_boundary: true,
                rtl: false,
            },
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> LanguageDetector {
        // Use a smaller minimum for tests.
        LanguageDetector::with_min_length(10)
    }

    // ─── LanguageCode ─────────────────────────────────────────────────────────

    #[test]
    fn language_code_display() {
        assert_eq!(LanguageCode::en().to_string(), "en");
        assert_eq!(LanguageCode::unknown().to_string(), "und");
    }

    #[test]
    fn language_code_is_unknown() {
        assert!(LanguageCode::unknown().is_unknown());
        assert!(!LanguageCode::en().is_unknown());
    }

    #[test]
    fn language_code_is_rtl_arabic() {
        assert!(LanguageCode::ar().is_rtl());
        assert!(!LanguageCode::en().is_rtl());
    }

    #[test]
    fn language_code_is_cjk() {
        assert!(LanguageCode::ja().is_cjk());
        assert!(LanguageCode::zh().is_cjk());
        assert!(LanguageCode::ko().is_cjk());
        assert!(!LanguageCode::en().is_cjk());
    }

    // ─── Trigram extraction ───────────────────────────────────────────────────

    #[test]
    fn trigrams_basic() {
        let tris = extract_trigrams("abcdef");
        assert!(tris.contains_key("abc"));
        assert!(tris.contains_key("bcd"));
    }

    #[test]
    fn trigrams_empty_string() {
        let tris = extract_trigrams("");
        assert!(tris.is_empty());
    }

    #[test]
    fn trigrams_counts_repeats() {
        let tris = extract_trigrams("aaaa");
        assert_eq!(*tris.get("aaa").unwrap_or(&0), 2);
    }

    // ─── LanguageDetector ─────────────────────────────────────────────────────

    #[test]
    fn detect_returns_unknown_for_short_text() {
        let det = LanguageDetector::new(); // default min_length = 30
        let result = det.detect("Hi");
        assert!(result.language.is_unknown());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn detect_english_text() {
        let det = detector();
        let text = "the quick brown fox jumps over the lazy dog and the cat sat on the mat";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::en(),
            "detected: {:?}",
            result.language
        );
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn detect_spanish_text() {
        let det = detector();
        let text = "el perro corre por el parque y la gente lo mira con curiosidad";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::es(),
            "detected: {:?}",
            result.language
        );
    }

    #[test]
    fn detect_french_text() {
        let det = detector();
        let text = "le chat est assis sur le tapis et regarde par la fenêtre avec attention";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::fr(),
            "detected: {:?}",
            result.language
        );
    }

    #[test]
    fn detect_german_text() {
        let det = detector();
        let text = "die schnelle braune Katze springt über den faulen Hund und schläft danach";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::de(),
            "detected: {:?}",
            result.language
        );
    }

    #[test]
    fn detect_japanese_text() {
        let det = LanguageDetector::with_min_length(5);
        let text = "これは日本語のテキストです。私はここにいます。";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::ja(),
            "detected: {:?}",
            result.language
        );
    }

    #[test]
    fn detect_arabic_text() {
        let det = LanguageDetector::with_min_length(5);
        let text = "هذا نص عربي يحتوي على كلمات كثيرة";
        let result = det.detect(text);
        assert_eq!(
            result.language,
            LanguageCode::ar(),
            "detected: {:?}",
            result.language
        );
    }

    #[test]
    fn detection_result_reliability() {
        let reliable = DetectionResult {
            language: LanguageCode::en(),
            confidence: 0.75,
            alternatives: Vec::new(),
        };
        assert!(reliable.is_reliable());

        let unreliable = DetectionResult {
            language: LanguageCode::unknown(),
            confidence: 0.30,
            alternatives: Vec::new(),
        };
        assert!(!unreliable.is_reliable());
    }

    #[test]
    fn alternatives_present() {
        let det = detector();
        let text = "the quick brown fox jumps over the lazy dog and the cat sat on the mat";
        let result = det.detect(text);
        // Should have at most 3 alternatives.
        assert!(result.alternatives.len() <= 3);
    }

    // ─── LineBreakHints ───────────────────────────────────────────────────────

    #[test]
    fn hints_english_defaults() {
        let hints = LineBreakHints::for_language(&LanguageCode::en());
        assert_eq!(hints.max_chars_per_line, 42);
        assert!(hints.needs_word_boundary);
        assert!(!hints.rtl);
    }

    #[test]
    fn hints_japanese_cjk() {
        let hints = LineBreakHints::for_language(&LanguageCode::ja());
        assert!(hints.max_chars_per_line < 20);
        assert!(!hints.needs_word_boundary);
    }

    #[test]
    fn hints_arabic_rtl() {
        let hints = LineBreakHints::for_language(&LanguageCode::ar());
        assert!(hints.rtl);
    }

    #[test]
    fn hints_german_wider_line() {
        let hints_de = LineBreakHints::for_language(&LanguageCode::de());
        let hints_en = LineBreakHints::for_language(&LanguageCode::en());
        assert!(hints_de.max_chars_per_line >= hints_en.max_chars_per_line);
    }

    // ─── detect_from_segments ─────────────────────────────────────────────────

    #[test]
    fn detect_from_segments_concatenates() {
        use crate::alignment::TranscriptSegment;
        let segs = vec![
            TranscriptSegment {
                text: "the quick brown fox".to_string(),
                start_ms: 0,
                end_ms: 2000,
                speaker_id: None,
                words: Vec::new(),
            },
            TranscriptSegment {
                text: "jumps over the lazy dog and the cat".to_string(),
                start_ms: 2000,
                end_ms: 4000,
                speaker_id: None,
                words: Vec::new(),
            },
        ];
        let det = detector();
        let result = det.detect_from_segments(&segs);
        assert_eq!(
            result.language,
            LanguageCode::en(),
            "detected: {:?}",
            result.language
        );
    }
}
