#![allow(dead_code)]
//! Review templates for standardised review workflows.
//!
//! Provides reusable templates that define the structure of a review session,
//! including required checklist items, default reviewers, and scoring criteria.

use std::collections::HashMap;

/// A checklist criterion within a review template.
#[derive(Debug, Clone)]
pub struct TemplateCriterion {
    /// Unique name of this criterion.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this criterion is mandatory.
    pub required: bool,
    /// Weight used when computing a weighted score (0.0..=1.0).
    pub weight: f64,
    /// Category grouping for display.
    pub category: String,
}

impl TemplateCriterion {
    /// Create a new criterion.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
        weight: f64,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required,
            weight: weight.clamp(0.0, 1.0),
            category: String::from("General"),
        }
    }

    /// Set the category for this criterion.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }
}

/// A scored evaluation of a single criterion.
#[derive(Debug, Clone)]
pub struct CriterionScore {
    /// The criterion name being scored.
    pub criterion_name: String,
    /// Score value between 0 and 100.
    pub score: u32,
    /// Optional reviewer notes.
    pub notes: Option<String>,
    /// Whether this criterion passed.
    pub passed: bool,
}

impl CriterionScore {
    /// Create a new criterion score.
    #[must_use]
    pub fn new(criterion_name: impl Into<String>, score: u32, passed: bool) -> Self {
        Self {
            criterion_name: criterion_name.into(),
            score: score.min(100),
            notes: None,
            passed,
        }
    }

    /// Attach notes to this score.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// The kind of review template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateKind {
    /// Technical quality check (codec, resolution, bitrate).
    Technical,
    /// Creative / editorial review.
    Creative,
    /// Compliance and legal review.
    Compliance,
    /// Client approval round.
    ClientApproval,
    /// Internal QA pass.
    QualityAssurance,
}

impl TemplateKind {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Technical => "Technical Review",
            Self::Creative => "Creative Review",
            Self::Compliance => "Compliance Review",
            Self::ClientApproval => "Client Approval",
            Self::QualityAssurance => "Quality Assurance",
        }
    }
}

/// A reusable review template.
#[derive(Debug, Clone)]
pub struct ReviewTemplate {
    /// Template identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Template kind.
    pub kind: TemplateKind,
    /// Ordered list of criteria.
    pub criteria: Vec<TemplateCriterion>,
    /// Default reviewer roles required.
    pub required_roles: Vec<String>,
    /// Minimum overall score to pass (0..=100).
    pub pass_threshold: u32,
}

impl ReviewTemplate {
    /// Create a new review template.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: TemplateKind) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind,
            criteria: Vec::new(),
            required_roles: Vec::new(),
            pass_threshold: 70,
        }
    }

    /// Add a criterion to the template.
    #[must_use]
    pub fn with_criterion(mut self, criterion: TemplateCriterion) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Add a required reviewer role.
    #[must_use]
    pub fn with_required_role(mut self, role: impl Into<String>) -> Self {
        self.required_roles.push(role.into());
        self
    }

    /// Set the pass threshold score.
    #[must_use]
    pub fn with_pass_threshold(mut self, threshold: u32) -> Self {
        self.pass_threshold = threshold.min(100);
        self
    }

    /// Count the number of required criteria.
    #[must_use]
    pub fn required_count(&self) -> usize {
        self.criteria.iter().filter(|c| c.required).count()
    }

    /// Count the number of optional criteria.
    #[must_use]
    pub fn optional_count(&self) -> usize {
        self.criteria.iter().filter(|c| !c.required).count()
    }

    /// Return the total weight of all criteria.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn total_weight(&self) -> f64 {
        self.criteria.iter().map(|c| c.weight).sum()
    }

    /// Group criteria by category.
    #[must_use]
    pub fn criteria_by_category(&self) -> HashMap<String, Vec<&TemplateCriterion>> {
        let mut map: HashMap<String, Vec<&TemplateCriterion>> = HashMap::new();
        for criterion in &self.criteria {
            map.entry(criterion.category.clone())
                .or_default()
                .push(criterion);
        }
        map
    }
}

/// A completed scorecard based on a template.
#[derive(Debug, Clone)]
pub struct ReviewScorecard {
    /// The template ID this scorecard is based on.
    pub template_id: String,
    /// Individual criterion scores.
    pub scores: Vec<CriterionScore>,
    /// Reviewer identifier.
    pub reviewer: String,
}

impl ReviewScorecard {
    /// Create a new scorecard.
    #[must_use]
    pub fn new(template_id: impl Into<String>, reviewer: impl Into<String>) -> Self {
        Self {
            template_id: template_id.into(),
            scores: Vec::new(),
            reviewer: reviewer.into(),
        }
    }

    /// Add a score entry.
    pub fn add_score(&mut self, score: CriterionScore) {
        self.scores.push(score);
    }

    /// Compute the simple average score across all entries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let total: u32 = self.scores.iter().map(|s| s.score).sum();
        f64::from(total) / self.scores.len() as f64
    }

    /// Compute the weighted score given a template.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weighted_score(&self, template: &ReviewTemplate) -> f64 {
        let weight_map: HashMap<&str, f64> = template
            .criteria
            .iter()
            .map(|c| (c.name.as_str(), c.weight))
            .collect();
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;
        for s in &self.scores {
            let w = weight_map
                .get(s.criterion_name.as_str())
                .copied()
                .unwrap_or(1.0);
            weighted_sum += f64::from(s.score) * w;
            weight_total += w;
        }
        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }

    /// Check whether all required criteria passed.
    #[must_use]
    pub fn all_required_passed(&self, template: &ReviewTemplate) -> bool {
        let required_names: Vec<&str> = template
            .criteria
            .iter()
            .filter(|c| c.required)
            .map(|c| c.name.as_str())
            .collect();
        for name in &required_names {
            let found = self
                .scores
                .iter()
                .find(|s| s.criterion_name.as_str() == *name);
            match found {
                Some(s) if s.passed => {}
                _ => return false,
            }
        }
        true
    }

    /// Determine whether the overall review passes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn passes(&self, template: &ReviewTemplate) -> bool {
        if !self.all_required_passed(template) {
            return false;
        }
        self.weighted_score(template) >= f64::from(template.pass_threshold)
    }

    /// Return names of failed criteria.
    #[must_use]
    pub fn failed_criteria(&self) -> Vec<&str> {
        self.scores
            .iter()
            .filter(|s| !s.passed)
            .map(|s| s.criterion_name.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_template() -> ReviewTemplate {
        ReviewTemplate::new("tmpl-1", "Technical Check", TemplateKind::Technical)
            .with_criterion(TemplateCriterion::new(
                "resolution",
                "Check resolution",
                true,
                0.4,
            ))
            .with_criterion(TemplateCriterion::new(
                "bitrate",
                "Check bitrate",
                true,
                0.3,
            ))
            .with_criterion(
                TemplateCriterion::new("color", "Color accuracy", false, 0.3)
                    .with_category("Visual"),
            )
            .with_required_role("engineer")
            .with_pass_threshold(70)
    }

    #[test]
    fn test_template_creation() {
        let t = sample_template();
        assert_eq!(t.name, "Technical Check");
        assert_eq!(t.kind, TemplateKind::Technical);
        assert_eq!(t.criteria.len(), 3);
    }

    #[test]
    fn test_required_count() {
        let t = sample_template();
        assert_eq!(t.required_count(), 2);
        assert_eq!(t.optional_count(), 1);
    }

    #[test]
    fn test_total_weight() {
        let t = sample_template();
        assert!((t.total_weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_criteria_by_category() {
        let t = sample_template();
        let map = t.criteria_by_category();
        assert!(map.contains_key("General"));
        assert!(map.contains_key("Visual"));
    }

    #[test]
    fn test_template_kind_label() {
        assert_eq!(TemplateKind::Technical.label(), "Technical Review");
        assert_eq!(TemplateKind::Creative.label(), "Creative Review");
        assert_eq!(TemplateKind::Compliance.label(), "Compliance Review");
        assert_eq!(TemplateKind::ClientApproval.label(), "Client Approval");
        assert_eq!(TemplateKind::QualityAssurance.label(), "Quality Assurance");
    }

    #[test]
    fn test_criterion_weight_clamped() {
        let c = TemplateCriterion::new("x", "desc", false, 5.0);
        assert!((c.weight - 1.0).abs() < f64::EPSILON);
        let c2 = TemplateCriterion::new("y", "desc", false, -1.0);
        assert!((c2.weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scorecard_average() {
        let mut sc = ReviewScorecard::new("tmpl-1", "alice");
        sc.add_score(CriterionScore::new("resolution", 80, true));
        sc.add_score(CriterionScore::new("bitrate", 60, true));
        assert!((sc.average_score() - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scorecard_weighted_score() {
        let t = sample_template();
        let mut sc = ReviewScorecard::new("tmpl-1", "bob");
        sc.add_score(CriterionScore::new("resolution", 100, true));
        sc.add_score(CriterionScore::new("bitrate", 100, true));
        sc.add_score(CriterionScore::new("color", 100, true));
        assert!((sc.weighted_score(&t) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scorecard_passes() {
        let t = sample_template();
        let mut sc = ReviewScorecard::new("tmpl-1", "charlie");
        sc.add_score(CriterionScore::new("resolution", 80, true));
        sc.add_score(CriterionScore::new("bitrate", 80, true));
        sc.add_score(CriterionScore::new("color", 80, true));
        assert!(sc.passes(&t));
    }

    #[test]
    fn test_scorecard_fails_required() {
        let t = sample_template();
        let mut sc = ReviewScorecard::new("tmpl-1", "dave");
        sc.add_score(CriterionScore::new("resolution", 90, true));
        sc.add_score(CriterionScore::new("bitrate", 90, false)); // failed
        sc.add_score(CriterionScore::new("color", 90, true));
        assert!(!sc.passes(&t));
    }

    #[test]
    fn test_failed_criteria() {
        let mut sc = ReviewScorecard::new("tmpl-1", "eve");
        sc.add_score(CriterionScore::new("a", 50, false));
        sc.add_score(CriterionScore::new("b", 90, true));
        sc.add_score(CriterionScore::new("c", 30, false));
        let failed = sc.failed_criteria();
        assert_eq!(failed.len(), 2);
        assert!(failed.contains(&"a"));
        assert!(failed.contains(&"c"));
    }

    #[test]
    fn test_empty_scorecard_average() {
        let sc = ReviewScorecard::new("tmpl-1", "nobody");
        assert!((sc.average_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_criterion_score_with_notes() {
        let cs = CriterionScore::new("x", 85, true).with_notes("Looks good");
        assert_eq!(cs.notes, Some("Looks good".to_string()));
    }
}
