//! Translation quality checking.

use serde::{Deserialize, Serialize};

/// Translation quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationQuality {
    /// Overall quality score (0.0 to 1.0).
    pub score: f32,
    /// Fluency score.
    pub fluency: f32,
    /// Adequacy score.
    pub adequacy: f32,
    /// Issues detected.
    pub issues: Vec<QualityIssue>,
}

/// Quality issue in translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type.
    pub issue_type: QualityIssueType,
    /// Issue description.
    pub description: String,
    /// Severity (0.0 to 1.0, higher is more severe).
    pub severity: f32,
}

/// Type of quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Missing translation.
    MissingTranslation,
    /// Inconsistent terminology.
    InconsistentTerminology,
    /// Grammar error.
    GrammarError,
    /// Cultural inappropriateness.
    CulturalIssue,
    /// Timing constraint violation.
    TimingIssue,
    /// Length constraint violation.
    LengthIssue,
}

/// Checks translation quality.
pub struct TranslationQualityChecker;

impl TranslationQualityChecker {
    /// Check translation quality.
    #[must_use]
    pub fn check(_source: &str, _translation: &str) -> TranslationQuality {
        // Placeholder: Perform quality checks
        // In production:
        // - Compare length ratios
        // - Check grammar
        // - Verify terminology consistency
        // - Detect cultural issues

        TranslationQuality {
            score: 0.9,
            fluency: 0.92,
            adequacy: 0.88,
            issues: vec![],
        }
    }

    /// Check if quality meets threshold.
    #[must_use]
    pub fn meets_threshold(quality: &TranslationQuality, threshold: f32) -> bool {
        quality.score >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_check() {
        let quality = TranslationQualityChecker::check("Hello", "Hola");
        assert!(quality.score > 0.0);
    }

    #[test]
    fn test_meets_threshold() {
        let quality = TranslationQuality {
            score: 0.9,
            fluency: 0.9,
            adequacy: 0.9,
            issues: vec![],
        };

        assert!(TranslationQualityChecker::meets_threshold(&quality, 0.8));
        assert!(!TranslationQualityChecker::meets_threshold(&quality, 0.95));
    }
}
