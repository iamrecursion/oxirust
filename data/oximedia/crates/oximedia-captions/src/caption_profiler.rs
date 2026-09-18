#![allow(dead_code)]
//! Caption profiling and performance analysis.
//!
//! This module provides tools for profiling caption tracks, analyzing timing
//! distributions, reading speed, character density, and overall caption
//! quality metrics for broadcast compliance and viewer experience optimization.

use std::collections::HashMap;
use std::fmt;

/// Words-per-minute reading speed thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadingSpeedConfig {
    /// Minimum acceptable words per minute.
    pub min_wpm: f64,
    /// Maximum acceptable words per minute.
    pub max_wpm: f64,
    /// Target (ideal) words per minute.
    pub target_wpm: f64,
    /// Minimum caption display duration in seconds.
    pub min_duration_secs: f64,
    /// Maximum caption display duration in seconds.
    pub max_duration_secs: f64,
}

impl Default for ReadingSpeedConfig {
    fn default() -> Self {
        Self {
            min_wpm: 100.0,
            max_wpm: 220.0,
            target_wpm: 160.0,
            min_duration_secs: 0.7,
            max_duration_secs: 7.0,
        }
    }
}

/// Severity level for profiling issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueSeverity {
    /// Informational note.
    Info,
    /// Potential problem.
    Warning,
    /// Compliance violation.
    Error,
    /// Critical compliance failure.
    Critical,
}

impl fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single profiling issue found in a caption track.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfilingIssue {
    /// Index of the caption with the issue.
    pub caption_index: usize,
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Short code for the issue type.
    pub code: String,
    /// Human-readable description.
    pub description: String,
    /// Start time in seconds of the affected caption.
    pub start_time_secs: f64,
    /// End time in seconds of the affected caption.
    pub end_time_secs: f64,
}

/// Timing statistics for a caption entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptionTiming {
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Gap to the next caption in seconds (None if last).
    pub gap_to_next_secs: Option<f64>,
    /// Word count.
    pub word_count: usize,
    /// Character count.
    pub char_count: usize,
    /// Words per minute for this caption.
    pub wpm: f64,
    /// Characters per second for this caption.
    pub cps: f64,
}

/// A single caption entry for profiling analysis.
#[derive(Debug, Clone)]
pub struct CaptionEntry {
    /// Text content of the caption.
    pub text: String,
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
    /// Number of lines in the caption.
    pub line_count: usize,
}

impl CaptionEntry {
    /// Create a new caption entry.
    pub fn new(text: impl Into<String>, start_secs: f64, end_secs: f64) -> Self {
        let text = text.into();
        let line_count = text.lines().count().max(1);
        Self {
            text,
            start_secs,
            end_secs,
            line_count,
        }
    }

    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }

    /// Count words in the text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Count characters (excluding whitespace).
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Calculate words per minute.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn wpm(&self) -> f64 {
        let duration = self.duration_secs();
        if duration <= 0.0 {
            return 0.0;
        }
        (self.word_count() as f64 / duration) * 60.0
    }

    /// Calculate characters per second.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn cps(&self) -> f64 {
        let duration = self.duration_secs();
        if duration <= 0.0 {
            return 0.0;
        }
        self.char_count() as f64 / duration
    }
}

/// Aggregate profiling statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileStats {
    /// Total number of captions.
    pub total_captions: usize,
    /// Total duration of all captions in seconds.
    pub total_caption_duration_secs: f64,
    /// Total program duration from first start to last end.
    pub program_duration_secs: f64,
    /// Average words per minute across all captions.
    pub avg_wpm: f64,
    /// Maximum words per minute found.
    pub max_wpm: f64,
    /// Minimum words per minute found.
    pub min_wpm: f64,
    /// Average characters per second.
    pub avg_cps: f64,
    /// Average duration per caption in seconds.
    pub avg_duration_secs: f64,
    /// Average gap between captions in seconds.
    pub avg_gap_secs: f64,
    /// Total word count.
    pub total_words: usize,
    /// Total character count.
    pub total_chars: usize,
    /// Number of issues by severity.
    pub issue_counts: HashMap<IssueSeverity, usize>,
}

impl Default for ProfileStats {
    fn default() -> Self {
        Self {
            total_captions: 0,
            total_caption_duration_secs: 0.0,
            program_duration_secs: 0.0,
            avg_wpm: 0.0,
            max_wpm: 0.0,
            min_wpm: f64::MAX,
            avg_cps: 0.0,
            avg_duration_secs: 0.0,
            avg_gap_secs: 0.0,
            total_words: 0,
            total_chars: 0,
            issue_counts: HashMap::new(),
        }
    }
}

/// The caption profiler engine.
#[derive(Debug, Clone)]
pub struct CaptionProfiler {
    /// Configuration for reading speed checks.
    pub config: ReadingSpeedConfig,
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum lines per caption.
    pub max_lines_per_caption: usize,
    /// Minimum gap between captions in seconds.
    pub min_gap_secs: f64,
}

impl Default for CaptionProfiler {
    fn default() -> Self {
        Self {
            config: ReadingSpeedConfig::default(),
            max_chars_per_line: 42,
            max_lines_per_caption: 2,
            min_gap_secs: 0.067, // 2 frames at 30fps
        }
    }
}

/// Full profiling report.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Aggregate statistics.
    pub stats: ProfileStats,
    /// All issues found.
    pub issues: Vec<ProfilingIssue>,
    /// Per-caption timing details.
    pub timings: Vec<CaptionTiming>,
}

impl CaptionProfiler {
    /// Create a new profiler with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Profile a list of caption entries and produce a full report.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn profile(&self, entries: &[CaptionEntry]) -> ProfileReport {
        let mut issues = Vec::new();
        let mut timings = Vec::new();
        let mut stats = ProfileStats::default();

        stats.total_captions = entries.len();
        let mut total_wpm = 0.0;
        let mut total_cps = 0.0;
        let mut total_gap = 0.0;
        let mut gap_count = 0usize;

        for (i, entry) in entries.iter().enumerate() {
            let duration = entry.duration_secs();
            let wpm = entry.wpm();
            let cps = entry.cps();
            let wc = entry.word_count();
            let cc = entry.char_count();

            stats.total_caption_duration_secs += duration;
            stats.total_words += wc;
            stats.total_chars += cc;
            total_wpm += wpm;
            total_cps += cps;

            if wpm > stats.max_wpm {
                stats.max_wpm = wpm;
            }
            if wpm < stats.min_wpm && wpm > 0.0 {
                stats.min_wpm = wpm;
            }

            // Gap to next
            let gap = if i + 1 < entries.len() {
                let g = entries[i + 1].start_secs - entry.end_secs;
                total_gap += g.max(0.0);
                gap_count += 1;
                Some(g)
            } else {
                None
            };

            timings.push(CaptionTiming {
                start_secs: entry.start_secs,
                end_secs: entry.end_secs,
                duration_secs: duration,
                gap_to_next_secs: gap,
                word_count: wc,
                char_count: cc,
                wpm,
                cps,
            });

            // Duration checks
            if duration < self.config.min_duration_secs {
                issues.push(ProfilingIssue {
                    caption_index: i,
                    severity: IssueSeverity::Warning,
                    code: "SHORT_DURATION".to_string(),
                    description: format!(
                        "Caption duration {duration:.2}s is below minimum {:.2}s",
                        self.config.min_duration_secs
                    ),
                    start_time_secs: entry.start_secs,
                    end_time_secs: entry.end_secs,
                });
            }
            if duration > self.config.max_duration_secs {
                issues.push(ProfilingIssue {
                    caption_index: i,
                    severity: IssueSeverity::Warning,
                    code: "LONG_DURATION".to_string(),
                    description: format!(
                        "Caption duration {duration:.2}s exceeds maximum {:.2}s",
                        self.config.max_duration_secs
                    ),
                    start_time_secs: entry.start_secs,
                    end_time_secs: entry.end_secs,
                });
            }

            // Reading speed checks
            if wpm > self.config.max_wpm {
                issues.push(ProfilingIssue {
                    caption_index: i,
                    severity: IssueSeverity::Error,
                    code: "FAST_READING".to_string(),
                    description: format!(
                        "Reading speed {wpm:.0} WPM exceeds maximum {:.0} WPM",
                        self.config.max_wpm
                    ),
                    start_time_secs: entry.start_secs,
                    end_time_secs: entry.end_secs,
                });
            }

            // Line count check
            if entry.line_count > self.max_lines_per_caption {
                issues.push(ProfilingIssue {
                    caption_index: i,
                    severity: IssueSeverity::Error,
                    code: "TOO_MANY_LINES".to_string(),
                    description: format!(
                        "Caption has {} lines, max is {}",
                        entry.line_count, self.max_lines_per_caption
                    ),
                    start_time_secs: entry.start_secs,
                    end_time_secs: entry.end_secs,
                });
            }

            // Gap check
            if let Some(g) = gap {
                if g < 0.0 {
                    issues.push(ProfilingIssue {
                        caption_index: i,
                        severity: IssueSeverity::Critical,
                        code: "OVERLAP".to_string(),
                        description: format!("Caption overlaps with next by {:.3}s", -g),
                        start_time_secs: entry.start_secs,
                        end_time_secs: entry.end_secs,
                    });
                } else if g < self.min_gap_secs && g > 0.0 {
                    issues.push(ProfilingIssue {
                        caption_index: i,
                        severity: IssueSeverity::Warning,
                        code: "SHORT_GAP".to_string(),
                        description: format!(
                            "Gap to next caption {g:.3}s is below minimum {:.3}s",
                            self.min_gap_secs
                        ),
                        start_time_secs: entry.start_secs,
                        end_time_secs: entry.end_secs,
                    });
                }
            }
        }

        let n = entries.len() as f64;
        if n > 0.0 {
            stats.avg_wpm = total_wpm / n;
            stats.avg_cps = total_cps / n;
            stats.avg_duration_secs = stats.total_caption_duration_secs / n;
        }
        if gap_count > 0 {
            stats.avg_gap_secs = total_gap / gap_count as f64;
        }
        if stats.min_wpm == f64::MAX {
            stats.min_wpm = 0.0;
        }

        if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
            stats.program_duration_secs = last.end_secs - first.start_secs;
        }

        // Count issues by severity
        for issue in &issues {
            *stats.issue_counts.entry(issue.severity).or_insert(0) += 1;
        }

        ProfileReport {
            stats,
            issues,
            timings,
        }
    }

    /// Check if a profile report passes all critical checks.
    #[must_use]
    pub fn passes_compliance(&self, report: &ProfileReport) -> bool {
        !report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Critical || i.severity == IssueSeverity::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries() -> Vec<CaptionEntry> {
        vec![
            CaptionEntry::new("Hello world, this is a test caption.", 0.0, 3.0),
            CaptionEntry::new("Another caption with more words here.", 3.1, 6.0),
            CaptionEntry::new("Short one.", 6.5, 8.0),
        ]
    }

    #[test]
    fn test_caption_entry_duration() {
        let entry = CaptionEntry::new("Hello world", 1.0, 4.0);
        assert!((entry.duration_secs() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_caption_entry_word_count() {
        let entry = CaptionEntry::new("Hello world this is four words plus two", 0.0, 3.0);
        assert_eq!(entry.word_count(), 8);
    }

    #[test]
    fn test_caption_entry_char_count() {
        let entry = CaptionEntry::new("Hi there", 0.0, 1.0);
        assert_eq!(entry.char_count(), 7); // "Hithere" = 7 non-whitespace
    }

    #[test]
    fn test_caption_entry_wpm() {
        // 6 words in 3 seconds = 120 WPM
        let entry = CaptionEntry::new("one two three four five six", 0.0, 3.0);
        let wpm = entry.wpm();
        assert!((wpm - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_caption_entry_wpm_zero_duration() {
        let entry = CaptionEntry::new("Hello world", 1.0, 1.0);
        assert!((entry.wpm() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_caption_entry_cps() {
        // "abc" = 3 chars, 1 second => 3.0 cps
        let entry = CaptionEntry::new("abc", 0.0, 1.0);
        assert!((entry.cps() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profiler_basic_stats() {
        let profiler = CaptionProfiler::new();
        let entries = make_entries();
        let report = profiler.profile(&entries);
        assert_eq!(report.stats.total_captions, 3);
        assert!(report.stats.avg_wpm > 0.0);
        assert!(report.stats.total_words > 0);
    }

    #[test]
    fn test_profiler_no_issues_for_normal_captions() {
        let profiler = CaptionProfiler::new();
        let entries = make_entries();
        let report = profiler.profile(&entries);
        // Normal captions should not have critical issues
        let critical = report
            .issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count();
        assert_eq!(critical, 0);
    }

    #[test]
    fn test_profiler_detects_overlap() {
        let profiler = CaptionProfiler::new();
        let entries = vec![
            CaptionEntry::new("First caption", 0.0, 3.0),
            CaptionEntry::new("Overlapping", 2.5, 5.0), // starts before first ends
        ];
        let report = profiler.profile(&entries);
        assert!(report.issues.iter().any(|i| i.code == "OVERLAP"));
    }

    #[test]
    fn test_profiler_detects_short_duration() {
        let profiler = CaptionProfiler::new();
        let entries = vec![
            CaptionEntry::new("Flash", 0.0, 0.3), // Very short
        ];
        let report = profiler.profile(&entries);
        assert!(report.issues.iter().any(|i| i.code == "SHORT_DURATION"));
    }

    #[test]
    fn test_profiler_detects_fast_reading() {
        let profiler = CaptionProfiler::new();
        // 20 words in 1 second = 1200 WPM - way too fast
        let text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty";
        let entries = vec![CaptionEntry::new(text, 0.0, 1.0)];
        let report = profiler.profile(&entries);
        assert!(report.issues.iter().any(|i| i.code == "FAST_READING"));
    }

    #[test]
    fn test_profiler_detects_too_many_lines() {
        let profiler = CaptionProfiler::new();
        let entries = vec![
            CaptionEntry::new("Line 1\nLine 2\nLine 3", 0.0, 3.0), // 3 lines > max 2
        ];
        let report = profiler.profile(&entries);
        assert!(report.issues.iter().any(|i| i.code == "TOO_MANY_LINES"));
    }

    #[test]
    fn test_profiler_compliance_check() {
        let profiler = CaptionProfiler::new();
        let entries = make_entries();
        let report = profiler.profile(&entries);
        assert!(profiler.passes_compliance(&report));
    }

    #[test]
    fn test_issue_severity_display() {
        assert_eq!(format!("{}", IssueSeverity::Info), "INFO");
        assert_eq!(format!("{}", IssueSeverity::Warning), "WARN");
        assert_eq!(format!("{}", IssueSeverity::Error), "ERROR");
        assert_eq!(format!("{}", IssueSeverity::Critical), "CRITICAL");
    }

    #[test]
    fn test_profiler_empty_entries() {
        let profiler = CaptionProfiler::new();
        let report = profiler.profile(&[]);
        assert_eq!(report.stats.total_captions, 0);
        assert!(report.issues.is_empty());
        assert!((report.stats.min_wpm - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reading_speed_config_default() {
        let config = ReadingSpeedConfig::default();
        assert!((config.target_wpm - 160.0).abs() < f64::EPSILON);
        assert!(config.min_wpm < config.max_wpm);
    }
}
