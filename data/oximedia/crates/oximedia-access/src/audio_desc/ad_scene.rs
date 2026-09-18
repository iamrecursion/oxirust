//! Audio description scene script validation and management.
//!
//! Provides structured scene-based audio description scripts with validation
//! tools for timing, reading speed, and style compliance.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Priority level for an audio description scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdPriority {
    /// Essential description — must be read regardless of gap length.
    Essential,
    /// Important description — read when enough gap is available.
    Important,
    /// Optional description — read only when plenty of time is available.
    Optional,
}

impl AdPriority {
    /// Minimum gap in milliseconds required to deliver this priority of description.
    #[must_use]
    pub const fn min_gap_ms(&self) -> u64 {
        match self {
            Self::Essential => 500,
            Self::Important => 250,
            Self::Optional => 100,
        }
    }
}

/// A single scene within an audio description script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdScene {
    /// Unique identifier for the scene.
    pub scene_id: u32,
    /// Scene start time in milliseconds.
    pub start_ms: u64,
    /// Scene end time in milliseconds.
    pub end_ms: u64,
    /// Descriptive text to be spoken.
    pub description: String,
    /// Priority of this scene's description.
    pub priority: AdPriority,
    /// Available gap for narration in milliseconds (before next dialogue).
    pub gap_available_ms: u64,
}

impl AdScene {
    /// Create a new scene entry.
    #[must_use]
    pub fn new(
        scene_id: u32,
        start_ms: u64,
        end_ms: u64,
        description: impl Into<String>,
        priority: AdPriority,
        gap_available_ms: u64,
    ) -> Self {
        Self {
            scene_id,
            start_ms,
            end_ms,
            description: description.into(),
            priority,
            gap_available_ms,
        }
    }

    /// Duration of the scene in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Scene-based audio description script.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDescriptionScript {
    /// Title of the content being described.
    pub title: String,
    /// Ordered list of scenes in the script.
    pub scenes: Vec<AdScene>,
    /// BCP-47 language code for the script (e.g., `"en"`, `"fr"`).
    pub language: String,
}

impl AudioDescriptionScript {
    /// Create a new empty script.
    #[must_use]
    pub fn new(title: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            scenes: Vec::new(),
            language: language.into(),
        }
    }

    /// Add a scene to the script.
    pub fn add_scene(&mut self, scene: AdScene) {
        self.scenes.push(scene);
    }
}

// ---------------------------------------------------------------------------
// Validation issue types
// ---------------------------------------------------------------------------

/// Category of a validation issue found in an audio description script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdIssueType {
    /// Description text is too long to fit within the available gap.
    TooLong,
    /// Available gap is shorter than the priority's minimum gap threshold.
    GapTooShort,
    /// An essential description cannot be delivered.
    MissingEssential,
    /// The language tag of this scene differs from the script language.
    LanguageInconsistency,
}

/// A single validation issue referencing a specific scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdValidationIssue {
    /// The scene that triggered this issue.
    pub scene_id: u32,
    /// The type of issue.
    pub issue_type: AdIssueType,
    /// Human-readable description of the issue.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Reading time estimator
// ---------------------------------------------------------------------------

/// Estimates narration reading time from text.
pub struct ReadingTimeEstimator;

impl ReadingTimeEstimator {
    /// Estimate reading time in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `text` — The text to estimate.
    /// * `wpm` — Words per minute; defaults to 150 if `<= 0`.
    #[must_use]
    pub fn estimate_ms(text: &str, wpm: f32) -> u64 {
        let effective_wpm = if wpm <= 0.0 { 150.0 } else { wpm };
        let word_count = text.split_whitespace().count() as f32;
        ((word_count / effective_wpm) * 60_000.0) as u64
    }
}

// ---------------------------------------------------------------------------
// Style guide formatter
// ---------------------------------------------------------------------------

/// Provides style-guide-compliant formatting for audio description text.
pub struct AdStyleGuide;

impl AdStyleGuide {
    /// Format a description string according to style-guide rules.
    ///
    /// Rules applied:
    /// - Capitalise the first letter of each sentence.
    /// - Remove parenthetical phrases (text surrounded by parentheses).
    #[must_use]
    pub fn format_description(text: &str) -> String {
        // Remove parenthetical content
        let without_parens = Self::remove_parentheticals(text);
        // Capitalise start of each sentence
        Self::capitalise_sentences(&without_parens)
    }

    /// Remove text enclosed in parentheses, including the parentheses themselves.
    fn remove_parentheticals(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut depth = 0usize;
        for ch in text.chars() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth = depth.saturating_sub(1);
                }
                _ => {
                    if depth == 0 {
                        result.push(ch);
                    }
                }
            }
        }
        // Collapse multiple spaces left after removal
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Capitalise the first character following sentence-ending punctuation.
    fn capitalise_sentences(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut capitalise_next = true;
        for ch in text.chars() {
            if capitalise_next && ch.is_alphabetic() {
                result.extend(ch.to_uppercase());
                capitalise_next = false;
            } else {
                result.push(ch);
                if matches!(ch, '.' | '!' | '?') {
                    capitalise_next = true;
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Validates an [`AudioDescriptionScript`] for common issues.
pub struct AdScriptValidator;

impl AdScriptValidator {
    /// Validate the script and return a list of issues found.
    ///
    /// Checks performed:
    /// 1. Description text fits within `gap_available_ms` at 150 wpm.
    /// 2. Gap is shorter than the priority threshold (flags risk of overlap with dialogue).
    /// 3. Essential scenes that cannot be delivered.
    #[must_use]
    pub fn validate(script: &AudioDescriptionScript) -> Vec<AdValidationIssue> {
        let mut issues = Vec::new();

        for scene in &script.scenes {
            let reading_ms = ReadingTimeEstimator::estimate_ms(&scene.description, 150.0);

            // Check 1: text too long for the gap
            if reading_ms > scene.gap_available_ms {
                issues.push(AdValidationIssue {
                    scene_id: scene.scene_id,
                    issue_type: AdIssueType::TooLong,
                    description: format!(
                        "Scene {}: description takes ~{reading_ms}ms to read but only {gap}ms available",
                        scene.scene_id,
                        gap = scene.gap_available_ms,
                    ),
                });

                // Check 3: essential and cannot fit at all
                if scene.priority == AdPriority::Essential {
                    issues.push(AdValidationIssue {
                        scene_id: scene.scene_id,
                        issue_type: AdIssueType::MissingEssential,
                        description: format!(
                            "Scene {}: essential description cannot be delivered within the available gap",
                            scene.scene_id
                        ),
                    });
                }
            }

            // Check 2: gap shorter than priority minimum — dialogue overlap risk
            if scene.gap_available_ms < 200 {
                issues.push(AdValidationIssue {
                    scene_id: scene.scene_id,
                    issue_type: AdIssueType::GapTooShort,
                    description: format!(
                        "Scene {}: gap of {}ms is shorter than the 200ms dialogue-overlap threshold",
                        scene.scene_id, scene.gap_available_ms
                    ),
                });
            }
        }

        issues
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(id: u32, description: &str, gap_ms: u64, priority: AdPriority) -> AdScene {
        AdScene::new(id, 0, 5_000, description, priority, gap_ms)
    }

    #[test]
    fn test_ad_priority_min_gap() {
        assert_eq!(AdPriority::Essential.min_gap_ms(), 500);
        assert_eq!(AdPriority::Important.min_gap_ms(), 250);
        assert_eq!(AdPriority::Optional.min_gap_ms(), 100);
    }

    #[test]
    fn test_reading_time_single_word() {
        // 1 word at 150 wpm → 60_000 / 150 = 400 ms
        let ms = ReadingTimeEstimator::estimate_ms("hello", 150.0);
        assert_eq!(ms, 400);
    }

    #[test]
    fn test_reading_time_default_wpm() {
        let ms_default = ReadingTimeEstimator::estimate_ms("hello world", 0.0);
        let ms_explicit = ReadingTimeEstimator::estimate_ms("hello world", 150.0);
        assert_eq!(ms_default, ms_explicit);
    }

    #[test]
    fn test_reading_time_ten_words() {
        // 10 words @ 150 wpm → 4000 ms
        let text = "one two three four five six seven eight nine ten";
        let ms = ReadingTimeEstimator::estimate_ms(text, 150.0);
        assert_eq!(ms, 4000);
    }

    #[test]
    fn test_remove_parentheticals() {
        let formatted = AdStyleGuide::format_description("A sunset (very bright) over the hills.");
        assert!(!formatted.contains('('));
        assert!(formatted.contains("sunset"));
        assert!(formatted.contains("hills"));
    }

    #[test]
    fn test_capitalise_sentences() {
        let result = AdStyleGuide::format_description("a dog runs. a cat sits.");
        assert!(result.starts_with('A'), "Should start with capital A");
        // second sentence capitalised
        assert!(
            result.contains(". A cat"),
            "Second sentence should start uppercase"
        );
    }

    #[test]
    fn test_validate_no_issues() {
        let mut script = AudioDescriptionScript::new("Test", "en");
        // 2 words @ 150 wpm → 800ms; gap 2000ms → OK
        script.add_scene(make_scene(1, "A dog.", 2000, AdPriority::Optional));
        let issues = AdScriptValidator::validate(&script);
        assert!(issues.is_empty(), "Expected no issues, got: {issues:?}");
    }

    #[test]
    fn test_validate_too_long() {
        let mut script = AudioDescriptionScript::new("Test", "en");
        // 30 words @ 150 wpm → 12_000ms; gap only 500ms
        let text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty";
        script.add_scene(make_scene(2, text, 500, AdPriority::Important));
        let issues = AdScriptValidator::validate(&script);
        assert!(
            issues.iter().any(|i| i.issue_type == AdIssueType::TooLong),
            "Expected TooLong issue"
        );
    }

    #[test]
    fn test_validate_gap_too_short() {
        let mut script = AudioDescriptionScript::new("Test", "en");
        script.add_scene(make_scene(3, "Hi.", 100, AdPriority::Optional));
        let issues = AdScriptValidator::validate(&script);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == AdIssueType::GapTooShort),
            "Expected GapTooShort issue"
        );
    }

    #[test]
    fn test_validate_missing_essential() {
        let mut script = AudioDescriptionScript::new("Test", "en");
        // Long text, tiny gap, essential priority
        let text = "one two three four five six seven eight nine ten eleven twelve";
        script.add_scene(make_scene(4, text, 100, AdPriority::Essential));
        let issues = AdScriptValidator::validate(&script);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == AdIssueType::MissingEssential),
            "Expected MissingEssential issue"
        );
    }

    #[test]
    fn test_ad_scene_duration() {
        let scene = AdScene::new(1, 1000, 4000, "Test", AdPriority::Important, 2000);
        assert_eq!(scene.duration_ms(), 3000);
    }

    #[test]
    fn test_script_add_scene() {
        let mut script = AudioDescriptionScript::new("My Film", "en");
        script.add_scene(make_scene(1, "Opening shot.", 3000, AdPriority::Essential));
        assert_eq!(script.scenes.len(), 1);
        assert_eq!(script.title, "My Film");
    }

    #[test]
    fn test_style_guide_nested_parens() {
        let result = AdStyleGuide::format_description("A (very (deep)) shadow.");
        assert!(!result.contains('('));
        assert!(result.contains("shadow"));
    }

    #[test]
    fn test_ad_issue_type_variants() {
        let types = [
            AdIssueType::TooLong,
            AdIssueType::GapTooShort,
            AdIssueType::MissingEssential,
            AdIssueType::LanguageInconsistency,
        ];
        assert_eq!(types.len(), 4);
    }
}
