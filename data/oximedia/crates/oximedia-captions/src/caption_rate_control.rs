#![allow(dead_code)]
//! Caption reading-rate control and validation.
//!
//! Ensures captions conform to recommended reading speed guidelines by
//! computing words-per-minute (WPM) and characters-per-second (CPS) metrics,
//! and optionally adjusting timings to meet target rates.

/// Default maximum characters per second (broadcast standard).
const DEFAULT_MAX_CPS: f64 = 20.0;

/// Default minimum characters per second (avoid overly long display).
const DEFAULT_MIN_CPS: f64 = 3.0;

/// Default maximum words per minute.
const DEFAULT_MAX_WPM: f64 = 200.0;

/// Default minimum words per minute.
const DEFAULT_MIN_WPM: f64 = 60.0;

/// A caption entry with text and timing for rate analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct RateCaption {
    /// Caption text content (plain text, no markup).
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

impl RateCaption {
    /// Create a new rate caption.
    #[must_use]
    pub fn new(text: String, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text,
            start_ms,
            end_ms,
        }
    }

    /// Duration of this caption in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Number of characters in the text (excluding whitespace-only content).
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Number of words in the text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Characters per second for this caption.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cps(&self) -> f64 {
        let dur_sec = self.duration_ms() as f64 / 1000.0;
        if dur_sec <= 0.0 {
            return 0.0;
        }
        self.char_count() as f64 / dur_sec
    }

    /// Words per minute for this caption.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wpm(&self) -> f64 {
        let dur_min = self.duration_ms() as f64 / 60_000.0;
        if dur_min <= 0.0 {
            return 0.0;
        }
        self.word_count() as f64 / dur_min
    }
}

/// Configuration for rate control limits.
#[derive(Debug, Clone)]
pub struct RateControlConfig {
    /// Maximum allowed characters per second.
    pub max_cps: f64,
    /// Minimum allowed characters per second.
    pub min_cps: f64,
    /// Maximum allowed words per minute.
    pub max_wpm: f64,
    /// Minimum allowed words per minute.
    pub min_wpm: f64,
}

impl Default for RateControlConfig {
    fn default() -> Self {
        Self {
            max_cps: DEFAULT_MAX_CPS,
            min_cps: DEFAULT_MIN_CPS,
            max_wpm: DEFAULT_MAX_WPM,
            min_wpm: DEFAULT_MIN_WPM,
        }
    }
}

/// A single rate violation found during analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct RateViolation {
    /// Zero-based index of the offending caption.
    pub caption_index: usize,
    /// The kind of violation.
    pub kind: RateViolationKind,
    /// Actual measured value.
    pub actual: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
}

/// The kind of rate violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateViolationKind {
    /// CPS exceeds the maximum.
    CpsTooHigh,
    /// CPS is below the minimum.
    CpsTooLow,
    /// WPM exceeds the maximum.
    WpmTooHigh,
    /// WPM is below the minimum.
    WpmTooLow,
}

/// Overall rate statistics for a caption track.
#[derive(Debug, Clone)]
pub struct RateStats {
    /// Number of captions analysed.
    pub caption_count: usize,
    /// Average CPS across all captions.
    pub avg_cps: f64,
    /// Maximum CPS found.
    pub max_cps: f64,
    /// Minimum CPS found (among non-empty captions).
    pub min_cps: f64,
    /// Average WPM across all captions.
    pub avg_wpm: f64,
    /// All violations found.
    pub violations: Vec<RateViolation>,
}

impl RateStats {
    /// Whether no violations were found.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.violations.is_empty()
    }

    /// Count of CPS violations.
    #[must_use]
    pub fn cps_violation_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| {
                matches!(
                    v.kind,
                    RateViolationKind::CpsTooHigh | RateViolationKind::CpsTooLow
                )
            })
            .count()
    }

    /// Count of WPM violations.
    #[must_use]
    pub fn wpm_violation_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| {
                matches!(
                    v.kind,
                    RateViolationKind::WpmTooHigh | RateViolationKind::WpmTooLow
                )
            })
            .count()
    }
}

/// Analyse reading rate across a list of captions.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn analyse_rates(captions: &[RateCaption], config: &RateControlConfig) -> RateStats {
    let mut violations = Vec::new();
    let mut total_cps = 0.0_f64;
    let mut total_wpm = 0.0_f64;
    let mut max_cps = 0.0_f64;
    let mut min_cps = f64::MAX;
    let mut counted = 0usize;

    for (i, cap) in captions.iter().enumerate() {
        if cap.duration_ms() == 0 || cap.char_count() == 0 {
            continue;
        }

        let cps = cap.cps();
        let wpm = cap.wpm();

        total_cps += cps;
        total_wpm += wpm;
        counted += 1;

        if cps > max_cps {
            max_cps = cps;
        }
        if cps < min_cps {
            min_cps = cps;
        }

        if cps > config.max_cps {
            violations.push(RateViolation {
                caption_index: i,
                kind: RateViolationKind::CpsTooHigh,
                actual: cps,
                threshold: config.max_cps,
            });
        }
        if cps < config.min_cps {
            violations.push(RateViolation {
                caption_index: i,
                kind: RateViolationKind::CpsTooLow,
                actual: cps,
                threshold: config.min_cps,
            });
        }
        if wpm > config.max_wpm {
            violations.push(RateViolation {
                caption_index: i,
                kind: RateViolationKind::WpmTooHigh,
                actual: wpm,
                threshold: config.max_wpm,
            });
        }
        if wpm < config.min_wpm {
            violations.push(RateViolation {
                caption_index: i,
                kind: RateViolationKind::WpmTooLow,
                actual: wpm,
                threshold: config.min_wpm,
            });
        }
    }

    let avg_cps = if counted > 0 {
        total_cps / counted as f64
    } else {
        0.0
    };
    let avg_wpm = if counted > 0 {
        total_wpm / counted as f64
    } else {
        0.0
    };
    if min_cps == f64::MAX {
        min_cps = 0.0;
    }

    RateStats {
        caption_count: captions.len(),
        avg_cps,
        max_cps,
        min_cps,
        avg_wpm,
        violations,
    }
}

/// Compute the ideal end time for a caption to achieve a target CPS.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn ideal_end_time(start_ms: u64, char_count: usize, target_cps: f64) -> u64 {
    if target_cps <= 0.0 || char_count == 0 {
        return start_ms;
    }
    let needed_sec = char_count as f64 / target_cps;
    let needed_ms = (needed_sec * 1000.0).round() as u64;
    start_ms + needed_ms
}

/// Suggest adjusted end times for captions that violate maximum CPS.
/// Returns a list of `(index, new_end_ms)` pairs.
#[must_use]
pub fn suggest_timing_fixes(captions: &[RateCaption], max_cps: f64) -> Vec<(usize, u64)> {
    let mut fixes = Vec::new();
    for (i, cap) in captions.iter().enumerate() {
        if cap.duration_ms() == 0 {
            continue;
        }
        let cps = cap.cps();
        if cps > max_cps {
            let new_end = ideal_end_time(cap.start_ms, cap.char_count(), max_cps);
            fixes.push((i, new_end));
        }
    }
    fixes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(text: &str, start: u64, end: u64) -> RateCaption {
        RateCaption::new(text.to_string(), start, end)
    }

    #[test]
    fn test_cps_calculation() {
        let c = cap("Hello world", 0, 2000);
        let cps = c.cps();
        // 11 chars / 2.0 sec = 5.5
        assert!((cps - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_wpm_calculation() {
        // "Hello world" = 2 words, duration = 6 seconds = 0.1 minutes
        let c = cap("Hello world", 0, 6000);
        let wpm = c.wpm();
        // 2 / 0.1 = 20
        assert!((wpm - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_zero_duration_cps() {
        let c = cap("Hello", 1000, 1000);
        assert!((c.cps()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_word_count() {
        let c = cap("The quick brown fox jumps", 0, 5000);
        assert_eq!(c.word_count(), 5);
    }

    #[test]
    fn test_char_count() {
        let c = cap("abc", 0, 1000);
        assert_eq!(c.char_count(), 3);
    }

    #[test]
    fn test_rate_analysis_compliant() {
        let captions = vec![
            cap("Hello there my friend", 0, 3000),
            cap("Welcome to the show", 3200, 6000),
        ];
        let stats = analyse_rates(&captions, &RateControlConfig::default());
        assert!(stats.is_compliant());
    }

    #[test]
    fn test_rate_analysis_too_fast() {
        // 50 chars in 1 second = 50 CPS, way above 20 max
        let captions = vec![cap(
            "This is a really long caption that has many chars",
            0,
            1000,
        )];
        let stats = analyse_rates(&captions, &RateControlConfig::default());
        assert!(!stats.is_compliant());
        assert!(stats.cps_violation_count() > 0);
    }

    #[test]
    fn test_rate_analysis_too_slow() {
        // 3 chars in 10 seconds = 0.3 CPS, below 3.0 min
        let captions = vec![cap("Hi!", 0, 10000)];
        let config = RateControlConfig::default();
        let stats = analyse_rates(&captions, &config);
        assert!(stats
            .violations
            .iter()
            .any(|v| v.kind == RateViolationKind::CpsTooLow));
    }

    #[test]
    fn test_ideal_end_time() {
        // 20 chars at 10 CPS = 2 seconds = 2000ms from start
        let end = ideal_end_time(5000, 20, 10.0);
        assert_eq!(end, 7000);
    }

    #[test]
    fn test_suggest_timing_fixes() {
        let captions = vec![
            cap("Short", 0, 5000), // 5 chars / 5s = 1 CPS (fine at max 20)
            cap("This is extremely fast text with lots of chars", 5000, 5500), // too fast
        ];
        let fixes = suggest_timing_fixes(&captions, 20.0);
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].0, 1);
        assert!(fixes[0].1 > 5500);
    }

    #[test]
    fn test_empty_captions() {
        let stats = analyse_rates(&[], &RateControlConfig::default());
        assert_eq!(stats.caption_count, 0);
        assert!(stats.is_compliant());
    }

    #[test]
    fn test_custom_config() {
        let config = RateControlConfig {
            max_cps: 10.0,
            min_cps: 5.0,
            max_wpm: 150.0,
            min_wpm: 80.0,
        };
        // 10 chars in 1 second = 10 CPS, exactly at limit
        let captions = vec![cap("0123456789", 0, 1000)];
        let stats = analyse_rates(&captions, &config);
        // At limit, not over. But WPM: 1 word / (1/60 min) = 60 WPM < 80 min
        assert!(stats
            .violations
            .iter()
            .any(|v| v.kind == RateViolationKind::WpmTooLow));
    }

    #[test]
    fn test_duration_ms() {
        let c = cap("test", 500, 2500);
        assert_eq!(c.duration_ms(), 2000);
    }
}
