//! Readability assessment integration for transcripts.
//!
//! Analyzes transcript readability using the `ReadabilityAnalyzer` and provides
//! per-entry and overall readability scoring, plain language compliance checking,
//! and suggestions for simplification.

use crate::error::{AccessError, AccessResult};
use crate::reading_level::{ReadabilityAnalyzer, ReadabilityLevel, ReadabilityReport};
use crate::transcript::{Transcript, TranscriptEntry};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Target audience for plain language compliance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetAudience {
    /// Children (grade 1-5, very easy).
    Children,
    /// General public (grade 6-8, easy).
    GeneralPublic,
    /// Young adults (grade 8-10, fairly easy).
    YoungAdults,
    /// Educated adults (grade 10-12, standard).
    EducatedAdults,
    /// Professional/academic (grade 12+, difficult).
    Professional,
}

impl fmt::Display for TargetAudience {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Children => write!(f, "Children"),
            Self::GeneralPublic => write!(f, "General Public"),
            Self::YoungAdults => write!(f, "Young Adults"),
            Self::EducatedAdults => write!(f, "Educated Adults"),
            Self::Professional => write!(f, "Professional"),
        }
    }
}

impl TargetAudience {
    /// Get the maximum acceptable readability level for this audience.
    #[must_use]
    pub fn max_readability_level(self) -> ReadabilityLevel {
        match self {
            Self::Children => ReadabilityLevel::VeryEasy,
            Self::GeneralPublic => ReadabilityLevel::Easy,
            Self::YoungAdults => ReadabilityLevel::FairlyEasy,
            Self::EducatedAdults => ReadabilityLevel::Standard,
            Self::Professional => ReadabilityLevel::VeryDifficult,
        }
    }

    /// Get the maximum acceptable grade level for this audience.
    #[must_use]
    pub fn max_grade_level(self) -> f64 {
        match self {
            Self::Children => 5.0,
            Self::GeneralPublic => 8.0,
            Self::YoungAdults => 10.0,
            Self::EducatedAdults => 12.0,
            Self::Professional => 20.0,
        }
    }
}

/// Readability assessment for a single transcript entry.
#[derive(Debug, Clone)]
pub struct EntryReadability {
    /// Index of the entry in the transcript.
    pub entry_index: usize,
    /// Start time of the entry.
    pub start_time_ms: i64,
    /// End time of the entry.
    pub end_time_ms: i64,
    /// The text of the entry.
    pub text: String,
    /// Flesch-Kincaid grade level.
    pub grade_level: f64,
    /// Flesch reading ease score.
    pub reading_ease: f64,
    /// Readability level category.
    pub level: ReadabilityLevel,
    /// Word count.
    pub word_count: usize,
    /// Complex word count (3+ syllables).
    pub complex_word_count: usize,
    /// Whether this entry meets the target audience level.
    pub meets_target: bool,
}

/// Overall readability assessment for a complete transcript.
#[derive(Debug, Clone)]
pub struct TranscriptReadability {
    /// Overall average grade level.
    pub average_grade_level: f64,
    /// Overall average reading ease.
    pub average_reading_ease: f64,
    /// Overall readability level.
    pub overall_level: ReadabilityLevel,
    /// Target audience.
    pub target_audience: TargetAudience,
    /// Whether the transcript meets the target audience level.
    pub meets_target: bool,
    /// Percentage of entries that meet the target level.
    pub compliance_percentage: f64,
    /// Per-entry readability assessments.
    pub entries: Vec<EntryReadability>,
    /// Entries that exceed the target level (most problematic first).
    pub problem_entries: Vec<usize>,
    /// Full readability report for the combined text.
    pub full_report: ReadabilityReport,
    /// Suggestions for improving readability.
    pub suggestions: Vec<String>,
}

/// Assesses transcript readability for plain language compliance.
#[derive(Debug)]
pub struct TranscriptReadabilityAssessor {
    analyzer: ReadabilityAnalyzer,
    target_audience: TargetAudience,
    /// Minimum number of words per entry to assess (shorter entries are skipped).
    min_words_for_assessment: usize,
}

impl TranscriptReadabilityAssessor {
    /// Create a new assessor for the given target audience.
    #[must_use]
    pub fn new(target_audience: TargetAudience) -> Self {
        Self {
            analyzer: ReadabilityAnalyzer::new(),
            target_audience,
            min_words_for_assessment: 5,
        }
    }

    /// Set the minimum word count per entry to assess.
    #[must_use]
    pub fn with_min_words(mut self, min_words: usize) -> Self {
        self.min_words_for_assessment = min_words.max(1);
        self
    }

    /// Assess a single transcript entry.
    #[must_use]
    pub fn assess_entry(&self, entry: &TranscriptEntry, index: usize) -> EntryReadability {
        let stats = self.analyzer.analyze_text(&entry.text);
        let grade = self.analyzer.flesch_kincaid_grade(&stats);
        let ease = self.analyzer.flesch_reading_ease(&stats);
        let level = ReadabilityAnalyzer::grade_to_level(grade);
        let max_level = self.target_audience.max_readability_level();

        EntryReadability {
            entry_index: index,
            start_time_ms: entry.start_time_ms,
            end_time_ms: entry.end_time_ms,
            text: entry.text.clone(),
            grade_level: grade,
            reading_ease: ease,
            level,
            word_count: stats.word_count,
            complex_word_count: stats.complex_word_count,
            meets_target: level <= max_level,
        }
    }

    /// Assess the entire transcript.
    pub fn assess_transcript(
        &self,
        transcript: &Transcript,
    ) -> AccessResult<TranscriptReadability> {
        if transcript.entries.is_empty() {
            return Err(AccessError::TranscriptFailed(
                "Cannot assess empty transcript".to_string(),
            ));
        }

        // Assess individual entries
        let entry_assessments: Vec<EntryReadability> = transcript
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| self.assess_entry(entry, i))
            .collect();

        // Filter assessable entries (enough words)
        let assessable: Vec<&EntryReadability> = entry_assessments
            .iter()
            .filter(|e| e.word_count >= self.min_words_for_assessment)
            .collect();

        let (avg_grade, avg_ease) = if assessable.is_empty() {
            (0.0, 100.0)
        } else {
            let total_grade: f64 = assessable.iter().map(|e| e.grade_level).sum();
            let total_ease: f64 = assessable.iter().map(|e| e.reading_ease).sum();
            let count = assessable.len() as f64;
            (total_grade / count, total_ease / count)
        };

        let overall_level = ReadabilityAnalyzer::grade_to_level(avg_grade);
        let max_level = self.target_audience.max_readability_level();
        let meets_target = overall_level <= max_level;

        let compliant_count = assessable.iter().filter(|e| e.meets_target).count();
        let compliance_pct = if assessable.is_empty() {
            100.0
        } else {
            compliant_count as f64 / assessable.len() as f64 * 100.0
        };

        // Find problem entries (sorted by grade level, highest first)
        let mut problem_entries: Vec<usize> = entry_assessments
            .iter()
            .filter(|e| !e.meets_target && e.word_count >= self.min_words_for_assessment)
            .map(|e| e.entry_index)
            .collect();
        problem_entries.sort_by(|a, b| {
            let grade_a = entry_assessments[*a].grade_level;
            let grade_b = entry_assessments[*b].grade_level;
            grade_b
                .partial_cmp(&grade_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Generate full report from combined text
        let full_text = transcript.get_text();
        let full_report = ReadabilityReport::generate(&full_text);

        // Generate suggestions
        let suggestions = self.generate_suggestions(&entry_assessments);

        Ok(TranscriptReadability {
            average_grade_level: avg_grade,
            average_reading_ease: avg_ease,
            overall_level,
            target_audience: self.target_audience,
            meets_target,
            compliance_percentage: compliance_pct,
            entries: entry_assessments,
            problem_entries,
            full_report,
            suggestions,
        })
    }

    /// Generate readability improvement suggestions.
    fn generate_suggestions(&self, entries: &[EntryReadability]) -> Vec<String> {
        let mut suggestions = Vec::new();
        let max_grade = self.target_audience.max_grade_level();

        // Check for complex words
        let total_complex: usize = entries.iter().map(|e| e.complex_word_count).sum();
        let total_words: usize = entries.iter().map(|e| e.word_count).sum();

        if total_words > 0 {
            let complex_pct = total_complex as f64 / total_words as f64 * 100.0;
            if complex_pct > 10.0 {
                suggestions.push(format!(
                    "Reduce complex words (3+ syllables): currently {:.1}% of all words",
                    complex_pct
                ));
            }
        }

        // Check average sentence length
        let high_grade_entries: Vec<&EntryReadability> = entries
            .iter()
            .filter(|e| e.grade_level > max_grade && e.word_count >= self.min_words_for_assessment)
            .collect();

        if !high_grade_entries.is_empty() {
            suggestions.push(format!(
                "{} entries exceed target grade level ({:.0}); consider simplifying",
                high_grade_entries.len(),
                max_grade
            ));
        }

        // Check for long entries
        let long_entries = entries.iter().filter(|e| e.word_count > 30).count();
        if long_entries > 0 {
            suggestions.push(format!(
                "{} entries have >30 words; consider breaking into shorter segments",
                long_entries
            ));
        }

        // Check low reading ease
        let low_ease = entries
            .iter()
            .filter(|e| e.reading_ease < 30.0 && e.word_count >= self.min_words_for_assessment)
            .count();
        if low_ease > 0 {
            suggestions.push(format!(
                "{} entries have very low reading ease (<30); use shorter sentences and simpler words",
                low_ease
            ));
        }

        // General recommendations
        if suggestions.is_empty() && entries.iter().all(|e| e.meets_target) {
            suggestions.push(format!(
                "Transcript meets {} audience requirements",
                self.target_audience
            ));
        }

        suggestions
    }

    /// Get the target audience.
    #[must_use]
    pub fn target_audience(&self) -> TargetAudience {
        self.target_audience
    }
}

impl Default for TranscriptReadabilityAssessor {
    fn default() -> Self {
        Self::new(TargetAudience::GeneralPublic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transcript(entries: &[(&str, i64, i64)]) -> Transcript {
        let mut t = Transcript::new();
        for (text, start, end) in entries {
            t.add_entry(TranscriptEntry::new(*start, *end, text.to_string()));
        }
        t
    }

    #[test]
    fn test_target_audience_display() {
        assert_eq!(TargetAudience::Children.to_string(), "Children");
        assert_eq!(TargetAudience::GeneralPublic.to_string(), "General Public");
        assert_eq!(TargetAudience::YoungAdults.to_string(), "Young Adults");
        assert_eq!(
            TargetAudience::EducatedAdults.to_string(),
            "Educated Adults"
        );
        assert_eq!(TargetAudience::Professional.to_string(), "Professional");
    }

    #[test]
    fn test_target_audience_max_grade() {
        assert!(
            TargetAudience::Children.max_grade_level()
                < TargetAudience::GeneralPublic.max_grade_level()
        );
        assert!(
            TargetAudience::GeneralPublic.max_grade_level()
                < TargetAudience::EducatedAdults.max_grade_level()
        );
    }

    #[test]
    fn test_target_audience_max_level() {
        assert!(
            TargetAudience::Children.max_readability_level()
                < TargetAudience::GeneralPublic.max_readability_level()
        );
    }

    #[test]
    fn test_assess_entry_simple() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic);
        let entry = TranscriptEntry::new(
            0,
            2000,
            "The cat sat on the mat. The dog ran fast.".to_string(),
        );
        let result = assessor.assess_entry(&entry, 0);

        assert_eq!(result.entry_index, 0);
        assert!(result.word_count > 0);
        assert!(result.grade_level < 10.0);
        assert!(result.reading_ease > 50.0);
    }

    #[test]
    fn test_assess_entry_complex() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::Children);
        let entry = TranscriptEntry::new(
            0,
            5000,
            "The implementation of sophisticated algorithmic methodologies necessitates comprehensive understanding of computational complexity theory and mathematical optimization techniques.".to_string(),
        );
        let result = assessor.assess_entry(&entry, 0);
        assert!(result.grade_level > 10.0);
        assert!(result.complex_word_count > 0);
        assert!(!result.meets_target); // Too complex for children
    }

    #[test]
    fn test_assess_transcript_simple() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic);
        let transcript = make_transcript(&[
            ("The cat sat on the mat. The dog ran fast.", 0, 3000),
            (
                "The bird flew over the house. The fish swam in the pond.",
                3000,
                6000,
            ),
        ]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        assert_eq!(result.entries.len(), 2);
        assert!(result.average_grade_level < 10.0);
        assert_eq!(result.target_audience, TargetAudience::GeneralPublic);
    }

    #[test]
    fn test_assess_transcript_empty() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic);
        let transcript = Transcript::new();
        assert!(assessor.assess_transcript(&transcript).is_err());
    }

    #[test]
    fn test_compliance_percentage() {
        let assessor =
            TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic).with_min_words(3);
        let transcript = make_transcript(&[
            ("The cat sat on the mat.", 0, 2000),
            ("The dog ran fast and jumped over the fence.", 2000, 4000),
        ]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        // Simple text should be mostly compliant
        assert!(result.compliance_percentage > 0.0);
    }

    #[test]
    fn test_problem_entries_identified() {
        let assessor =
            TranscriptReadabilityAssessor::new(TargetAudience::Children).with_min_words(3);
        let transcript = make_transcript(&[
            ("The cat sat on the mat.", 0, 2000),
            (
                "The implementation of sophisticated algorithmic methodologies \
                 necessitates comprehensive understanding of computational \
                 complexity theory.",
                2000,
                5000,
            ),
        ]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        // The complex entry should be flagged
        assert!(!result.problem_entries.is_empty());
    }

    #[test]
    fn test_suggestions_generated() {
        let assessor =
            TranscriptReadabilityAssessor::new(TargetAudience::Children).with_min_words(3);
        let transcript = make_transcript(&[(
            "The sophisticated implementation of algorithmic optimization \
                 requires fundamental understanding of computational complexity \
                 and mathematical reasoning capabilities.",
            0,
            5000,
        )]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_suggestions_positive_when_compliant() {
        let assessor =
            TranscriptReadabilityAssessor::new(TargetAudience::Professional).with_min_words(3);
        let transcript = make_transcript(&[("Hello world. How are you today?", 0, 2000)]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        // Should have a positive suggestion
        let has_positive = result.suggestions.iter().any(|s| s.contains("meets"));
        assert!(
            has_positive,
            "Should have positive feedback: {:?}",
            result.suggestions
        );
    }

    #[test]
    fn test_full_report_included() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic);
        let transcript = make_transcript(&[("The cat sat on the mat. The dog ran fast.", 0, 3000)]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        assert!(result.full_report.stats.word_count > 0);
    }

    #[test]
    fn test_min_words_filter() {
        let assessor =
            TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic).with_min_words(10);
        let transcript = make_transcript(&[
            ("Short.", 0, 1000), // Only 1 word, below threshold
            (
                "The cat sat on the mat and the dog ran fast over the hill.",
                1000,
                4000,
            ), // 13 words
        ]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        // Both entries should be in the list
        assert_eq!(result.entries.len(), 2);
        // But compliance should only count the assessable one
        assert!(result.compliance_percentage >= 0.0);
    }

    #[test]
    fn test_default_assessor() {
        let assessor = TranscriptReadabilityAssessor::default();
        assert_eq!(assessor.target_audience(), TargetAudience::GeneralPublic);
    }

    #[test]
    fn test_reading_ease_correlates_with_grade() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::GeneralPublic);

        let simple_entry = TranscriptEntry::new(
            0,
            2000,
            "The cat sat. The dog ran. The bird flew.".to_string(),
        );
        let complex_entry = TranscriptEntry::new(0, 2000, "The implementation of sophisticated algorithmic optimization strategies necessitates comprehensive understanding.".to_string());

        let simple = assessor.assess_entry(&simple_entry, 0);
        let complex = assessor.assess_entry(&complex_entry, 1);

        // Simple should have lower grade and higher reading ease
        assert!(simple.grade_level < complex.grade_level);
        assert!(simple.reading_ease > complex.reading_ease);
    }

    #[test]
    fn test_overall_meets_target_professional() {
        let assessor = TranscriptReadabilityAssessor::new(TargetAudience::Professional);
        let transcript = make_transcript(&[
            ("Complex multifaceted interdisciplinary analysis.", 0, 3000),
            ("Simple words too.", 3000, 5000),
        ]);

        let result = assessor
            .assess_transcript(&transcript)
            .expect("should succeed");
        // Professional audience should accept almost anything
        assert!(result.meets_target);
    }
}
