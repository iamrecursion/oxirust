#![allow(dead_code)]
//! Text readability analysis for accessibility compliance.
//!
//! Provides readability scoring using standard formulas (Flesch-Kincaid,
//! Coleman-Liau, ARI) to ensure captions, descriptions, and transcripts
//! are accessible to target audiences.

use std::fmt;

/// Readability grade level category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadabilityLevel {
    /// Very easy to read (grade 1-5).
    VeryEasy,
    /// Easy to read (grade 6-7).
    Easy,
    /// Fairly easy (grade 8-9).
    FairlyEasy,
    /// Standard (grade 10-11).
    Standard,
    /// Fairly difficult (grade 12-13).
    FairlyDifficult,
    /// Difficult (grade 14-16).
    Difficult,
    /// Very difficult (grade 17+).
    VeryDifficult,
}

impl fmt::Display for ReadabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::VeryEasy => "Very Easy",
            Self::Easy => "Easy",
            Self::FairlyEasy => "Fairly Easy",
            Self::Standard => "Standard",
            Self::FairlyDifficult => "Fairly Difficult",
            Self::Difficult => "Difficult",
            Self::VeryDifficult => "Very Difficult",
        };
        write!(f, "{label}")
    }
}

/// Statistics extracted from text for readability calculation.
#[derive(Debug, Clone)]
pub struct TextStats {
    /// Total number of words.
    pub word_count: usize,
    /// Total number of sentences.
    pub sentence_count: usize,
    /// Total number of syllables.
    pub syllable_count: usize,
    /// Total number of characters (letters only).
    pub char_count: usize,
    /// Number of complex words (3+ syllables).
    pub complex_word_count: usize,
}

/// Compute readability scores from text.
#[derive(Debug)]
pub struct ReadabilityAnalyzer {
    /// Minimum sentence length to consider valid.
    min_sentence_length: usize,
}

impl Default for ReadabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadabilityAnalyzer {
    /// Create a new analyzer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_sentence_length: 1,
        }
    }

    /// Create an analyzer with a custom minimum sentence length.
    #[must_use]
    pub fn with_min_sentence_length(min_len: usize) -> Self {
        Self {
            min_sentence_length: min_len.max(1),
        }
    }

    /// Count syllables in a single word (English heuristic).
    #[must_use]
    pub fn count_syllables(word: &str) -> usize {
        let w = word.to_lowercase();
        let chars: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
        if chars.is_empty() {
            return 0;
        }
        if chars.len() <= 3 {
            return 1;
        }

        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
        let mut count = 0_usize;
        let mut prev_vowel = false;

        for (i, &ch) in chars.iter().enumerate() {
            let is_vowel = vowels.contains(&ch);
            if is_vowel && !prev_vowel {
                count += 1;
            }
            prev_vowel = is_vowel;
            // Silent e at end
            if i == chars.len() - 1 && ch == 'e' && count > 1 {
                count -= 1;
            }
        }

        count.max(1)
    }

    /// Extract text statistics from input text.
    #[must_use]
    pub fn analyze_text(&self, text: &str) -> TextStats {
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        let sentence_count = text
            .chars()
            .filter(|&c| c == '.' || c == '!' || c == '?')
            .count()
            .max(if word_count > 0 { 1 } else { 0 });

        let mut syllable_count = 0_usize;
        let mut complex_word_count = 0_usize;
        let mut char_count = 0_usize;

        for word in &words {
            let cleaned: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            char_count += cleaned.len();
            let syls = Self::count_syllables(&cleaned);
            syllable_count += syls;
            if syls >= 3 {
                complex_word_count += 1;
            }
        }

        TextStats {
            word_count,
            sentence_count,
            syllable_count,
            char_count,
            complex_word_count,
        }
    }

    /// Compute the Flesch-Kincaid Grade Level.
    ///
    /// Returns the U.S. school grade level needed to understand the text.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn flesch_kincaid_grade(&self, stats: &TextStats) -> f64 {
        if stats.word_count == 0 || stats.sentence_count == 0 {
            return 0.0;
        }
        let words = stats.word_count as f64;
        let sentences = stats.sentence_count as f64;
        let syllables = stats.syllable_count as f64;

        0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    }

    /// Compute the Flesch Reading Ease score (0-100, higher is easier).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn flesch_reading_ease(&self, stats: &TextStats) -> f64 {
        if stats.word_count == 0 || stats.sentence_count == 0 {
            return 0.0;
        }
        let words = stats.word_count as f64;
        let sentences = stats.sentence_count as f64;
        let syllables = stats.syllable_count as f64;

        206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    }

    /// Compute the Coleman-Liau Index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coleman_liau_index(&self, stats: &TextStats) -> f64 {
        if stats.word_count == 0 {
            return 0.0;
        }
        let words = stats.word_count as f64;
        let l = (stats.char_count as f64 / words) * 100.0;
        let s = (stats.sentence_count as f64 / words) * 100.0;

        0.0588 * l - 0.296 * s - 15.8
    }

    /// Compute the Automated Readability Index (ARI).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn automated_readability_index(&self, stats: &TextStats) -> f64 {
        if stats.word_count == 0 || stats.sentence_count == 0 {
            return 0.0;
        }
        let words = stats.word_count as f64;
        let sentences = stats.sentence_count as f64;
        let chars = stats.char_count as f64;

        4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43
    }

    /// Compute the Gunning Fog Index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn gunning_fog_index(&self, stats: &TextStats) -> f64 {
        if stats.word_count == 0 || stats.sentence_count == 0 {
            return 0.0;
        }
        let words = stats.word_count as f64;
        let sentences = stats.sentence_count as f64;
        let complex = stats.complex_word_count as f64;

        0.4 * ((words / sentences) + 100.0 * (complex / words))
    }

    /// Map a grade level to a readability level category.
    #[must_use]
    pub fn grade_to_level(grade: f64) -> ReadabilityLevel {
        if grade < 6.0 {
            ReadabilityLevel::VeryEasy
        } else if grade < 8.0 {
            ReadabilityLevel::Easy
        } else if grade < 10.0 {
            ReadabilityLevel::FairlyEasy
        } else if grade < 12.0 {
            ReadabilityLevel::Standard
        } else if grade < 14.0 {
            ReadabilityLevel::FairlyDifficult
        } else if grade < 17.0 {
            ReadabilityLevel::Difficult
        } else {
            ReadabilityLevel::VeryDifficult
        }
    }

    /// Check if text meets a target readability level (at or below the grade).
    #[must_use]
    pub fn meets_level(&self, text: &str, target: ReadabilityLevel) -> bool {
        let stats = self.analyze_text(text);
        let grade = self.flesch_kincaid_grade(&stats);
        let actual_level = Self::grade_to_level(grade);
        actual_level <= target
    }
}

/// Result of a full readability assessment.
#[derive(Debug, Clone)]
pub struct ReadabilityReport {
    /// The analyzed text statistics.
    pub stats: TextStats,
    /// Flesch-Kincaid grade level.
    pub fk_grade: f64,
    /// Flesch reading ease score.
    pub fre_score: f64,
    /// Coleman-Liau index.
    pub cli_index: f64,
    /// Automated Readability Index.
    pub ari_index: f64,
    /// Gunning Fog index.
    pub fog_index: f64,
    /// Overall readability level.
    pub level: ReadabilityLevel,
}

impl ReadabilityReport {
    /// Generate a full readability report for the given text.
    #[must_use]
    pub fn generate(text: &str) -> Self {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text(text);
        let fk_grade = analyzer.flesch_kincaid_grade(&stats);
        let fre_score = analyzer.flesch_reading_ease(&stats);
        let cli_index = analyzer.coleman_liau_index(&stats);
        let ari_index = analyzer.automated_readability_index(&stats);
        let fog_index = analyzer.gunning_fog_index(&stats);
        let level = ReadabilityAnalyzer::grade_to_level(fk_grade);

        Self {
            stats,
            fk_grade,
            fre_score,
            cli_index,
            ari_index,
            fog_index,
            level,
        }
    }

    /// Average grade level across all indices.
    #[must_use]
    pub fn average_grade(&self) -> f64 {
        (self.fk_grade + self.cli_index + self.ari_index + self.fog_index) / 4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllable_count_simple() {
        assert_eq!(ReadabilityAnalyzer::count_syllables("cat"), 1);
        assert_eq!(ReadabilityAnalyzer::count_syllables("the"), 1);
        assert_eq!(ReadabilityAnalyzer::count_syllables("hello"), 2);
    }

    #[test]
    fn test_syllable_count_complex() {
        assert!(ReadabilityAnalyzer::count_syllables("beautiful") >= 2);
        assert!(ReadabilityAnalyzer::count_syllables("understanding") >= 3);
    }

    #[test]
    fn test_syllable_count_empty() {
        assert_eq!(ReadabilityAnalyzer::count_syllables(""), 0);
        assert_eq!(ReadabilityAnalyzer::count_syllables("123"), 0);
    }

    #[test]
    fn test_analyze_text_basic() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("The cat sat on the mat.");
        assert_eq!(stats.word_count, 6);
        assert_eq!(stats.sentence_count, 1);
        assert!(stats.syllable_count >= 6);
        assert!(stats.char_count > 0);
    }

    #[test]
    fn test_analyze_text_multiple_sentences() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("Hello world. How are you? I am fine!");
        assert_eq!(stats.sentence_count, 3);
        assert_eq!(stats.word_count, 8);
    }

    #[test]
    fn test_analyze_text_empty() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("");
        assert_eq!(stats.word_count, 0);
        assert_eq!(stats.sentence_count, 0);
    }

    #[test]
    fn test_flesch_kincaid_grade_empty() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("");
        assert!((analyzer.flesch_kincaid_grade(&stats) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flesch_kincaid_grade_simple_text() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("The cat sat. The dog ran. The bird flew.");
        let grade = analyzer.flesch_kincaid_grade(&stats);
        // Simple text should have a low grade level
        assert!(
            grade < 10.0,
            "Grade {grade} should be below 10 for simple text"
        );
    }

    #[test]
    fn test_flesch_reading_ease() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("The cat sat. The dog ran.");
        let ease = analyzer.flesch_reading_ease(&stats);
        // Simple text should have high readability ease
        assert!(
            ease > 50.0,
            "Ease score {ease} should be above 50 for simple text"
        );
    }

    #[test]
    fn test_coleman_liau_empty() {
        let analyzer = ReadabilityAnalyzer::new();
        let stats = analyzer.analyze_text("");
        assert!((analyzer.coleman_liau_index(&stats) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_grade_to_level_mapping() {
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(3.0),
            ReadabilityLevel::VeryEasy
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(7.0),
            ReadabilityLevel::Easy
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(9.0),
            ReadabilityLevel::FairlyEasy
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(11.0),
            ReadabilityLevel::Standard
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(13.0),
            ReadabilityLevel::FairlyDifficult
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(15.0),
            ReadabilityLevel::Difficult
        );
        assert_eq!(
            ReadabilityAnalyzer::grade_to_level(20.0),
            ReadabilityLevel::VeryDifficult
        );
    }

    #[test]
    fn test_readability_report_generate() {
        let report = ReadabilityReport::generate("The cat sat on the mat. The dog ran fast.");
        assert!(report.stats.word_count > 0);
        assert!(report.fre_score != 0.0 || report.stats.word_count == 0);
    }

    #[test]
    fn test_readability_report_average_grade() {
        let report = ReadabilityReport::generate("Simple words. Short text. Easy read.");
        let avg = report.average_grade();
        // Average should be a finite number
        assert!(avg.is_finite());
    }

    #[test]
    fn test_readability_level_display() {
        assert_eq!(format!("{}", ReadabilityLevel::VeryEasy), "Very Easy");
        assert_eq!(format!("{}", ReadabilityLevel::Standard), "Standard");
        assert_eq!(
            format!("{}", ReadabilityLevel::VeryDifficult),
            "Very Difficult"
        );
    }

    #[test]
    fn test_meets_level() {
        let analyzer = ReadabilityAnalyzer::new();
        // Simple text should meet the standard level
        let simple = "The cat sat. The dog ran. The bird flew.";
        assert!(analyzer.meets_level(simple, ReadabilityLevel::Standard));
    }
}
