//! Reading-speed analysis for subtitle tracks.
//!
//! Checks each subtitle cue against per-level CPS (characters per second)
//! limits and produces a compliance report.

#![allow(dead_code)]

/// Classifies the reading level of an audience, each with a CPS ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadingLevel {
    /// Children or learners: up to 14 CPS.
    Children,
    /// Standard broadcast: up to 17 CPS (EBU R37 / BBC guidelines).
    Standard,
    /// Verbatim / fast-paced content: up to 22 CPS.
    Fast,
    /// No CPS limit enforced.
    Unlimited,
}

impl ReadingLevel {
    /// Returns the maximum allowed characters per second for this level.
    /// `None` means no limit is enforced.
    #[must_use]
    pub fn cps_limit(self) -> Option<f64> {
        match self {
            Self::Children => Some(14.0),
            Self::Standard => Some(17.0),
            Self::Fast => Some(22.0),
            Self::Unlimited => None,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Children => "children",
            Self::Standard => "standard",
            Self::Fast => "fast",
            Self::Unlimited => "unlimited",
        }
    }
}

/// A check result for a single subtitle cue.
#[derive(Debug, Clone)]
pub struct ReadingSpeedCheck {
    /// Index of the cue within the track.
    pub cue_index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Actual characters per second for this cue.
    pub actual_cps: f64,
    /// CPS limit that was applied.
    pub limit_cps: Option<f64>,
    /// Whether this cue violates the CPS limit.
    pub violation: bool,
}

impl ReadingSpeedCheck {
    /// Returns `true` when the cue exceeds the allowed reading speed.
    #[must_use]
    pub fn is_too_fast(&self) -> bool {
        self.violation
    }

    /// Returns the excess CPS above the limit, or `0.0` if within limits.
    #[must_use]
    pub fn excess_cps(&self) -> f64 {
        match self.limit_cps {
            Some(limit) if self.actual_cps > limit => self.actual_cps - limit,
            _ => 0.0,
        }
    }
}

/// A subtitle cue used as input for reading-speed analysis.
#[derive(Debug, Clone)]
pub struct SpeedCue {
    /// Index of this cue.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Plain-text content of the cue.
    pub text: String,
}

impl SpeedCue {
    /// Creates a new `SpeedCue`.
    #[must_use]
    pub fn new(index: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Duration in seconds. Returns `0.0` for zero-length or inverted cues.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        let dur_ms = (self.end_ms - self.start_ms).max(0);
        dur_ms as f64 / 1000.0
    }

    /// Number of printable characters (excluding whitespace for CPS calc).
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Computes characters per second.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn cps(&self) -> f64 {
        let dur = self.duration_secs();
        if dur <= 0.0 {
            return 0.0;
        }
        self.char_count() as f64 / dur
    }
}

/// Analyses subtitle reading speed against a configurable level.
#[derive(Debug, Clone)]
pub struct ReadingSpeedAnalyzer {
    /// Target reading level.
    pub level: ReadingLevel,
}

impl ReadingSpeedAnalyzer {
    /// Creates a new analyser for the given reading level.
    #[must_use]
    pub fn new(level: ReadingLevel) -> Self {
        Self { level }
    }

    /// Analyses all cues and returns per-cue checks.
    #[must_use]
    pub fn analyze(&self, cues: &[SpeedCue]) -> Vec<ReadingSpeedCheck> {
        let limit = self.level.cps_limit();
        cues.iter()
            .map(|cue| {
                let actual_cps = cue.cps();
                let violation = limit.is_some_and(|l| actual_cps > l);
                ReadingSpeedCheck {
                    cue_index: cue.index,
                    start_ms: cue.start_ms,
                    end_ms: cue.end_ms,
                    actual_cps,
                    limit_cps: limit,
                    violation,
                }
            })
            .collect()
    }

    /// Returns only the cues that violate the reading-speed limit.
    #[must_use]
    pub fn violations(&self, cues: &[SpeedCue]) -> Vec<ReadingSpeedCheck> {
        self.analyze(cues)
            .into_iter()
            .filter(|c| c.violation)
            .collect()
    }
}

/// Summary report for reading-speed compliance.
#[derive(Debug, Clone)]
pub struct ReadingSpeedReport {
    /// Reading level that was checked.
    pub level: ReadingLevel,
    /// Total number of cues analysed.
    pub total_cues: usize,
    /// Number of cues that violate the CPS limit.
    pub violation_count: usize,
    /// Maximum CPS observed across all cues.
    pub max_cps: f64,
    /// Average CPS across all cues.
    pub avg_cps: f64,
}

impl ReadingSpeedReport {
    /// Builds a report from a set of per-cue checks.
    #[must_use]
    pub fn from_checks(level: ReadingLevel, checks: &[ReadingSpeedCheck]) -> Self {
        let total_cues = checks.len();
        let violation_count = checks.iter().filter(|c| c.violation).count();
        let max_cps = checks.iter().map(|c| c.actual_cps).fold(0.0f64, f64::max);
        let avg_cps = if total_cues == 0 {
            0.0
        } else {
            checks.iter().map(|c| c.actual_cps).sum::<f64>() / total_cues as f64
        };
        Self {
            level,
            total_cues,
            violation_count,
            max_cps,
            avg_cps,
        }
    }

    /// Percentage of cues that are within the reading-speed limit (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compliance_pct(&self) -> f64 {
        if self.total_cues == 0 {
            return 100.0;
        }
        let compliant = self.total_cues.saturating_sub(self.violation_count);
        compliant as f64 / self.total_cues as f64 * 100.0
    }

    /// Returns `true` when all cues are within the CPS limit.
    #[must_use]
    pub fn is_fully_compliant(&self) -> bool {
        self.violation_count == 0
    }
}

// ============================================================================
// Word-Complexity-Aware Reading Speed Analysis
// ============================================================================

/// Word complexity factors used in advanced reading-speed estimation.
///
/// Instead of a flat character-per-second metric, this accounts for:
/// - **Word length**: longer words take proportionally more time to read.
/// - **Syllable count**: polysyllabic words slow reading.
/// - **Uncommon characters**: words with digits, mixed case, or non-Latin
///   scripts carry an additional penalty.
/// - **Punctuation density**: heavy punctuation signals more complex text.
#[derive(Debug, Clone)]
pub struct WordComplexityConfig {
    /// Base weight per character (default 1.0).
    pub base_char_weight: f64,
    /// Additional weight per estimated syllable beyond the first (default 0.3).
    pub syllable_penalty: f64,
    /// Penalty per word that is longer than `long_word_threshold` chars (default 0.5).
    pub long_word_penalty: f64,
    /// Characters in a word to be classified as "long" (default 8).
    pub long_word_threshold: usize,
    /// Penalty applied for each digit character (default 0.2).
    pub digit_penalty: f64,
    /// Penalty applied per punctuation mark (default 0.1).
    pub punctuation_penalty: f64,
}

impl Default for WordComplexityConfig {
    fn default() -> Self {
        Self {
            base_char_weight: 1.0,
            syllable_penalty: 0.3,
            long_word_penalty: 0.5,
            long_word_threshold: 8,
            digit_penalty: 0.2,
            punctuation_penalty: 0.1,
        }
    }
}

/// Estimate the number of syllables in an English-like word.
///
/// Uses a simple vowel-cluster heuristic:
/// 1. Count groups of consecutive vowels.
/// 2. Subtract one for a trailing silent 'e' (unless the word is very short).
/// 3. Ensure at least 1 syllable per word.
#[must_use]
pub fn estimate_syllables(word: &str) -> usize {
    if word.is_empty() {
        return 0;
    }
    let lower: Vec<char> = word.chars().map(|c| c.to_ascii_lowercase()).collect();
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];

    let mut count = 0usize;
    let mut prev_vowel = false;

    for &c in &lower {
        let is_v = vowels.contains(&c);
        if is_v && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_v;
    }

    // Subtract trailing silent 'e'
    if lower.len() > 2 && lower.last() == Some(&'e') && !vowels.contains(&lower[lower.len() - 2]) {
        count = count.saturating_sub(1);
    }

    count.max(1)
}

/// Compute the weighted complexity score for a piece of text.
///
/// The score represents an "effective character count" that is higher
/// for complex text than for simple text of the same length.
#[must_use]
pub fn word_complexity_score(text: &str, config: &WordComplexityConfig) -> f64 {
    let mut score = 0.0;

    for word in text.split_whitespace() {
        // Base score: number of non-whitespace characters
        let char_count = word.chars().filter(|c| c.is_alphanumeric()).count();
        score += char_count as f64 * config.base_char_weight;

        // Syllable penalty
        let syllables = estimate_syllables(word);
        if syllables > 1 {
            score += (syllables - 1) as f64 * config.syllable_penalty;
        }

        // Long-word penalty
        if char_count > config.long_word_threshold {
            score += config.long_word_penalty;
        }

        // Digit penalty
        let digit_count = word.chars().filter(|c| c.is_ascii_digit()).count();
        score += digit_count as f64 * config.digit_penalty;

        // Punctuation penalty
        let punct_count = word.chars().filter(|c| c.is_ascii_punctuation()).count();
        score += punct_count as f64 * config.punctuation_penalty;
    }

    score
}

/// A check result for complexity-aware reading speed analysis.
#[derive(Debug, Clone)]
pub struct ComplexitySpeedCheck {
    /// Index of the cue within the track.
    pub cue_index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Weighted complexity score.
    pub complexity_score: f64,
    /// Effective weighted CPS (complexity_score / duration_secs).
    pub effective_cps: f64,
    /// Simple CPS (chars / duration).
    pub simple_cps: f64,
    /// CPS limit that was applied.
    pub limit_cps: Option<f64>,
    /// Whether this cue violates the limit.
    pub violation: bool,
}

/// Analyser that uses word-complexity weighting.
#[derive(Debug, Clone)]
pub struct ComplexityReadingSpeedAnalyzer {
    /// Reading level for CPS limit.
    pub level: ReadingLevel,
    /// Complexity configuration.
    pub config: WordComplexityConfig,
}

impl ComplexityReadingSpeedAnalyzer {
    /// Create a new analyser with the given level and default complexity config.
    #[must_use]
    pub fn new(level: ReadingLevel) -> Self {
        Self {
            level,
            config: WordComplexityConfig::default(),
        }
    }

    /// Create with custom complexity configuration.
    #[must_use]
    pub fn with_config(level: ReadingLevel, config: WordComplexityConfig) -> Self {
        Self { level, config }
    }

    /// Analyse all cues with complexity weighting.
    #[must_use]
    pub fn analyze(&self, cues: &[SpeedCue]) -> Vec<ComplexitySpeedCheck> {
        let limit = self.level.cps_limit();
        cues.iter()
            .map(|cue| {
                let dur = cue.duration_secs();
                let complexity_score = word_complexity_score(&cue.text, &self.config);
                let effective_cps = if dur > 0.0 {
                    complexity_score / dur
                } else {
                    0.0
                };
                let simple_cps = cue.cps();
                let violation = limit.is_some_and(|l| effective_cps > l);

                ComplexitySpeedCheck {
                    cue_index: cue.index,
                    start_ms: cue.start_ms,
                    end_ms: cue.end_ms,
                    complexity_score,
                    effective_cps,
                    simple_cps,
                    limit_cps: limit,
                    violation,
                }
            })
            .collect()
    }

    /// Return only the cues that violate the complexity-weighted CPS limit.
    #[must_use]
    pub fn violations(&self, cues: &[SpeedCue]) -> Vec<ComplexitySpeedCheck> {
        self.analyze(cues)
            .into_iter()
            .filter(|c| c.violation)
            .collect()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cue(index: usize, start_ms: i64, end_ms: i64, text: &str) -> SpeedCue {
        SpeedCue::new(index, start_ms, end_ms, text)
    }

    #[test]
    fn test_reading_level_cps_limit_children() {
        assert_eq!(ReadingLevel::Children.cps_limit(), Some(14.0));
    }

    #[test]
    fn test_reading_level_cps_limit_unlimited() {
        assert_eq!(ReadingLevel::Unlimited.cps_limit(), None);
    }

    #[test]
    fn test_reading_level_labels() {
        assert_eq!(ReadingLevel::Standard.label(), "standard");
        assert_eq!(ReadingLevel::Fast.label(), "fast");
    }

    #[test]
    fn test_speed_cue_duration_secs() {
        let cue = make_cue(0, 0, 2000, "hello");
        assert!((cue.duration_secs() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_speed_cue_char_count_excludes_spaces() {
        let cue = make_cue(0, 0, 1000, "hello world");
        // 'h','e','l','l','o','w','o','r','l','d' = 10
        assert_eq!(cue.char_count(), 10);
    }

    #[test]
    fn test_speed_cue_cps() {
        // 10 non-space chars, 2 seconds → 5 CPS
        let cue = make_cue(0, 0, 2000, "hello world");
        assert!((cue.cps() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_speed_cue_zero_duration_cps() {
        let cue = make_cue(0, 1000, 1000, "hello");
        assert_eq!(cue.cps(), 0.0);
    }

    #[test]
    fn test_reading_speed_check_is_too_fast() {
        let check = ReadingSpeedCheck {
            cue_index: 0,
            start_ms: 0,
            end_ms: 1000,
            actual_cps: 20.0,
            limit_cps: Some(17.0),
            violation: true,
        };
        assert!(check.is_too_fast());
        assert!((check.excess_cps() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_reading_speed_check_not_too_fast() {
        let check = ReadingSpeedCheck {
            cue_index: 0,
            start_ms: 0,
            end_ms: 1000,
            actual_cps: 10.0,
            limit_cps: Some(17.0),
            violation: false,
        };
        assert!(!check.is_too_fast());
        assert_eq!(check.excess_cps(), 0.0);
    }

    #[test]
    fn test_analyzer_no_violations_slow_text() {
        let cues = vec![make_cue(0, 0, 5000, "Hello")];
        let analyzer = ReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let violations = analyzer.violations(&cues);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_analyzer_detects_violation() {
        // "abcdefghijklmnopqrstuvwxyz" = 26 non-space chars in 1 second → 26 CPS
        let cues = vec![make_cue(0, 0, 1000, "abcdefghijklmnopqrstuvwxyz")];
        let analyzer = ReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let violations = analyzer.violations(&cues);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_report_compliance_pct_full() {
        let cues = vec![
            make_cue(0, 0, 5000, "Short"),
            make_cue(1, 5000, 10000, "Also short"),
        ];
        let analyzer = ReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let checks = analyzer.analyze(&cues);
        let report = ReadingSpeedReport::from_checks(ReadingLevel::Standard, &checks);
        assert!((report.compliance_pct() - 100.0).abs() < 0.01);
        assert!(report.is_fully_compliant());
    }

    #[test]
    fn test_report_compliance_pct_partial() {
        let cues = vec![
            make_cue(0, 0, 1000, "abcdefghijklmnopqrstuvwxyz"), // violates
            make_cue(1, 1000, 6000, "fine"),                    // ok
        ];
        let analyzer = ReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let checks = analyzer.analyze(&cues);
        let report = ReadingSpeedReport::from_checks(ReadingLevel::Standard, &checks);
        assert!((report.compliance_pct() - 50.0).abs() < 0.01);
        assert!(!report.is_fully_compliant());
    }

    #[test]
    fn test_report_empty_cues() {
        let report = ReadingSpeedReport::from_checks(ReadingLevel::Standard, &[]);
        assert!((report.compliance_pct() - 100.0).abs() < 0.01);
        assert_eq!(report.total_cues, 0);
    }

    // ── Word-complexity tests ──────────────────────────────────────────

    #[test]
    fn test_estimate_syllables_one() {
        assert_eq!(estimate_syllables("cat"), 1);
        assert_eq!(estimate_syllables("dog"), 1);
    }

    #[test]
    fn test_estimate_syllables_two() {
        assert_eq!(estimate_syllables("hello"), 2);
        assert_eq!(estimate_syllables("water"), 2);
    }

    #[test]
    fn test_estimate_syllables_polysyllabic() {
        // "international" has ~5 syllables
        let s = estimate_syllables("international");
        assert!(s >= 4, "got {s} for 'international'");
    }

    #[test]
    fn test_estimate_syllables_empty() {
        assert_eq!(estimate_syllables(""), 0);
    }

    #[test]
    fn test_estimate_syllables_single_char() {
        assert_eq!(estimate_syllables("a"), 1);
        assert_eq!(estimate_syllables("x"), 1);
    }

    #[test]
    fn test_word_complexity_score_simple_text() {
        let config = WordComplexityConfig::default();
        let score_simple = word_complexity_score("Hello world", &config);
        // 10 alphanumeric chars + syllable penalties
        assert!(score_simple > 10.0, "score={score_simple}");
    }

    #[test]
    fn test_word_complexity_score_complex_higher() {
        let config = WordComplexityConfig::default();
        let simple = word_complexity_score("Hi there", &config);
        let complex = word_complexity_score("Internationally unprecedented", &config);
        // Complex text should have a higher score per-character
        let simple_per_char = simple / 7.0; // 7 non-space chars
        let complex_per_char = complex / 29.0; // 29 non-space chars roughly
        assert!(
            complex_per_char > simple_per_char,
            "complex_pc={complex_per_char}, simple_pc={simple_per_char}"
        );
    }

    #[test]
    fn test_word_complexity_score_digits_penalty() {
        let config = WordComplexityConfig::default();
        let no_digits = word_complexity_score("Hello world", &config);
        let with_digits = word_complexity_score("Hello 12345", &config);
        assert!(with_digits > no_digits, "digits should add penalty");
    }

    #[test]
    fn test_word_complexity_score_punctuation_penalty() {
        let config = WordComplexityConfig::default();
        let plain = word_complexity_score("Hello world", &config);
        let with_punct = word_complexity_score("Hello, world!!!", &config);
        assert!(with_punct > plain, "punctuation should add penalty");
    }

    #[test]
    fn test_complexity_analyzer_no_violation_slow() {
        let cues = vec![make_cue(0, 0, 10000, "Hello")];
        let analyzer = ComplexityReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let violations = analyzer.violations(&cues);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_complexity_analyzer_detects_fast_complex_text() {
        // Dense complex text in 1 second
        let cues = vec![make_cue(
            0,
            0,
            1000,
            "Internationally unprecedented characterization",
        )];
        let analyzer = ComplexityReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let violations = analyzer.violations(&cues);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].effective_cps > violations[0].simple_cps);
    }

    #[test]
    fn test_complexity_check_fields() {
        let cues = vec![make_cue(0, 0, 2000, "Hello world")];
        let analyzer = ComplexityReadingSpeedAnalyzer::new(ReadingLevel::Standard);
        let checks = analyzer.analyze(&cues);
        assert_eq!(checks.len(), 1);
        assert!(checks[0].complexity_score > 0.0);
        assert!(checks[0].effective_cps > 0.0);
        assert_eq!(checks[0].limit_cps, Some(17.0));
        assert!(!checks[0].violation);
    }

    #[test]
    fn test_word_complexity_config_custom() {
        let config = WordComplexityConfig {
            base_char_weight: 2.0,
            syllable_penalty: 1.0,
            long_word_penalty: 2.0,
            long_word_threshold: 4,
            digit_penalty: 0.5,
            punctuation_penalty: 0.5,
        };
        let score = word_complexity_score("Hello", &config);
        // 5 alphanumeric * 2.0 = 10.0 + syllable penalty (2-1)*1.0 = 1.0 + long_word (5>4)=2.0 = 13.0
        assert!((score - 13.0).abs() < 0.01, "score={score}");
    }

    #[test]
    fn test_word_complexity_empty_text() {
        let config = WordComplexityConfig::default();
        let score = word_complexity_score("", &config);
        assert!((score - 0.0).abs() < 0.01);
    }
}
