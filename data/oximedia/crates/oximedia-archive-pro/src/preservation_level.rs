#![allow(dead_code)]

//! Preservation level classification and assessment for archived media.
//!
//! This module defines a framework for classifying the preservation effort
//! applied to archival packages. Levels range from basic bit-level preservation
//! through full logical preservation with ongoing format migration.

use std::collections::HashMap;

/// Defines the preservation level applied to an archival object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PreservationLevel {
    /// Bit-level only: ensure bits are unchanged over time.
    BitLevel,
    /// Logical preservation: maintain readability of the content.
    Logical,
    /// Semantic preservation: maintain meaning and usability.
    Semantic,
    /// Full preservation: active format migration and emulation support.
    Full,
}

impl PreservationLevel {
    /// Returns a human-readable description of the level.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::BitLevel => "Bit-level preservation: fixity checks ensure binary integrity",
            Self::Logical => "Logical preservation: content remains renderable and accessible",
            Self::Semantic => {
                "Semantic preservation: meaning and context are maintained across migrations"
            }
            Self::Full => "Full preservation: proactive migration, emulation, and access support",
        }
    }

    /// Returns the numeric rank (1-4) for ordering.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub const fn rank(&self) -> u8 {
        match self {
            Self::BitLevel => 1,
            Self::Logical => 2,
            Self::Semantic => 3,
            Self::Full => 4,
        }
    }

    /// Returns true if this level includes fixity checking.
    #[must_use]
    pub const fn includes_fixity(&self) -> bool {
        // All levels include fixity
        true
    }

    /// Returns true if this level includes format migration planning.
    #[must_use]
    pub const fn includes_migration(&self) -> bool {
        matches!(self, Self::Semantic | Self::Full)
    }

    /// Returns true if this level includes emulation support.
    #[must_use]
    pub const fn includes_emulation(&self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Criteria used to assess the preservation level of an object.
#[derive(Debug, Clone)]
pub struct PreservationAssessment {
    /// Whether fixity checks are configured and passing.
    pub fixity_verified: bool,
    /// Whether the format is currently renderable.
    pub format_renderable: bool,
    /// Whether descriptive metadata is present.
    pub has_descriptive_metadata: bool,
    /// Whether provenance information is recorded.
    pub has_provenance: bool,
    /// Whether a migration pathway exists for the format.
    pub migration_pathway_exists: bool,
    /// Whether an emulation environment is available.
    pub emulation_available: bool,
    /// Number of redundant copies.
    pub redundant_copies: u32,
    /// Whether the object is in a preservation-grade format.
    pub preservation_format: bool,
}

impl PreservationAssessment {
    /// Creates a new assessment with default (worst-case) values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fixity_verified: false,
            format_renderable: false,
            has_descriptive_metadata: false,
            has_provenance: false,
            migration_pathway_exists: false,
            emulation_available: false,
            redundant_copies: 0,
            preservation_format: false,
        }
    }

    /// Determines the achieved preservation level based on the assessment criteria.
    #[must_use]
    pub fn achieved_level(&self) -> PreservationLevel {
        if self.fixity_verified
            && self.format_renderable
            && self.has_descriptive_metadata
            && self.has_provenance
            && self.migration_pathway_exists
            && self.emulation_available
            && self.preservation_format
        {
            PreservationLevel::Full
        } else if self.fixity_verified
            && self.format_renderable
            && self.has_descriptive_metadata
            && self.migration_pathway_exists
        {
            PreservationLevel::Semantic
        } else if self.fixity_verified && self.format_renderable {
            PreservationLevel::Logical
        } else if self.fixity_verified {
            PreservationLevel::BitLevel
        } else {
            PreservationLevel::BitLevel
        }
    }

    /// Returns a list of recommendations to reach the target level.
    #[must_use]
    pub fn recommendations_for(&self, target: PreservationLevel) -> Vec<String> {
        let mut recs = Vec::new();

        if !self.fixity_verified {
            recs.push("Configure and run fixity checks".to_string());
        }
        if target >= PreservationLevel::Logical && !self.format_renderable {
            recs.push("Ensure format is renderable with available tools".to_string());
        }
        if target >= PreservationLevel::Semantic && !self.has_descriptive_metadata {
            recs.push("Add descriptive metadata (Dublin Core, MODS, etc.)".to_string());
        }
        if target >= PreservationLevel::Semantic && !self.has_provenance {
            recs.push("Record provenance and chain-of-custody information".to_string());
        }
        if target >= PreservationLevel::Semantic && !self.migration_pathway_exists {
            recs.push("Identify a format migration pathway".to_string());
        }
        if target >= PreservationLevel::Full && !self.emulation_available {
            recs.push("Configure an emulation environment".to_string());
        }
        if target >= PreservationLevel::Full && !self.preservation_format {
            recs.push("Convert to a preservation-grade format".to_string());
        }
        if self.redundant_copies < 3 {
            recs.push(format!(
                "Increase redundant copies from {} to at least 3",
                self.redundant_copies
            ));
        }

        recs
    }

    /// Computes a numeric score from 0.0 to 1.0 representing overall preservation health.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn health_score(&self) -> f64 {
        let mut score = 0.0_f64;
        let total = 8.0_f64;

        if self.fixity_verified {
            score += 1.0;
        }
        if self.format_renderable {
            score += 1.0;
        }
        if self.has_descriptive_metadata {
            score += 1.0;
        }
        if self.has_provenance {
            score += 1.0;
        }
        if self.migration_pathway_exists {
            score += 1.0;
        }
        if self.emulation_available {
            score += 1.0;
        }
        if self.preservation_format {
            score += 1.0;
        }
        // Copies contribution: min(copies, 3) / 3
        let copy_score = f64::from(self.redundant_copies.min(3)) / 3.0;
        score += copy_score;

        score / total
    }
}

impl Default for PreservationAssessment {
    fn default() -> Self {
        Self::new()
    }
}

/// A policy that maps content types to required preservation levels.
#[derive(Debug, Clone)]
pub struct PreservationPolicy {
    /// Map from content type identifier to required level.
    requirements: HashMap<String, PreservationLevel>,
    /// Default level for unspecified content types.
    default_level: PreservationLevel,
}

impl PreservationPolicy {
    /// Creates a new policy with the given default level.
    #[must_use]
    pub fn new(default_level: PreservationLevel) -> Self {
        Self {
            requirements: HashMap::new(),
            default_level,
        }
    }

    /// Adds a requirement for a specific content type.
    pub fn set_requirement(&mut self, content_type: &str, level: PreservationLevel) {
        self.requirements.insert(content_type.to_string(), level);
    }

    /// Gets the required level for a content type.
    #[must_use]
    pub fn required_level(&self, content_type: &str) -> PreservationLevel {
        self.requirements
            .get(content_type)
            .copied()
            .unwrap_or(self.default_level)
    }

    /// Checks whether an assessment meets the policy for a given content type.
    #[must_use]
    pub fn meets_policy(&self, content_type: &str, assessment: &PreservationAssessment) -> bool {
        let required = self.required_level(content_type);
        assessment.achieved_level() >= required
    }

    /// Returns the number of configured requirements.
    #[must_use]
    pub fn requirement_count(&self) -> usize {
        self.requirements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preservation_level_ordering() {
        assert!(PreservationLevel::BitLevel < PreservationLevel::Logical);
        assert!(PreservationLevel::Logical < PreservationLevel::Semantic);
        assert!(PreservationLevel::Semantic < PreservationLevel::Full);
    }

    #[test]
    fn test_preservation_level_rank() {
        assert_eq!(PreservationLevel::BitLevel.rank(), 1);
        assert_eq!(PreservationLevel::Logical.rank(), 2);
        assert_eq!(PreservationLevel::Semantic.rank(), 3);
        assert_eq!(PreservationLevel::Full.rank(), 4);
    }

    #[test]
    fn test_level_descriptions_not_empty() {
        for level in [
            PreservationLevel::BitLevel,
            PreservationLevel::Logical,
            PreservationLevel::Semantic,
            PreservationLevel::Full,
        ] {
            assert!(!level.description().is_empty());
        }
    }

    #[test]
    fn test_level_includes_fixity() {
        assert!(PreservationLevel::BitLevel.includes_fixity());
        assert!(PreservationLevel::Full.includes_fixity());
    }

    #[test]
    fn test_level_includes_migration() {
        assert!(!PreservationLevel::BitLevel.includes_migration());
        assert!(!PreservationLevel::Logical.includes_migration());
        assert!(PreservationLevel::Semantic.includes_migration());
        assert!(PreservationLevel::Full.includes_migration());
    }

    #[test]
    fn test_level_includes_emulation() {
        assert!(!PreservationLevel::Semantic.includes_emulation());
        assert!(PreservationLevel::Full.includes_emulation());
    }

    #[test]
    fn test_assessment_default_is_bit_level() {
        let a = PreservationAssessment::new();
        // Even without fixity, the fallback is BitLevel
        assert_eq!(a.achieved_level(), PreservationLevel::BitLevel);
    }

    #[test]
    fn test_assessment_logical_level() {
        let a = PreservationAssessment {
            fixity_verified: true,
            format_renderable: true,
            ..PreservationAssessment::new()
        };
        assert_eq!(a.achieved_level(), PreservationLevel::Logical);
    }

    #[test]
    fn test_assessment_semantic_level() {
        let a = PreservationAssessment {
            fixity_verified: true,
            format_renderable: true,
            has_descriptive_metadata: true,
            migration_pathway_exists: true,
            ..PreservationAssessment::new()
        };
        assert_eq!(a.achieved_level(), PreservationLevel::Semantic);
    }

    #[test]
    fn test_assessment_full_level() {
        let a = PreservationAssessment {
            fixity_verified: true,
            format_renderable: true,
            has_descriptive_metadata: true,
            has_provenance: true,
            migration_pathway_exists: true,
            emulation_available: true,
            preservation_format: true,
            redundant_copies: 3,
        };
        assert_eq!(a.achieved_level(), PreservationLevel::Full);
    }

    #[test]
    fn test_health_score_zero() {
        let a = PreservationAssessment::new();
        let score = a.health_score();
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_score_full() {
        let a = PreservationAssessment {
            fixity_verified: true,
            format_renderable: true,
            has_descriptive_metadata: true,
            has_provenance: true,
            migration_pathway_exists: true,
            emulation_available: true,
            preservation_format: true,
            redundant_copies: 3,
        };
        let score = a.health_score();
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recommendations() {
        let a = PreservationAssessment::new();
        let recs = a.recommendations_for(PreservationLevel::Full);
        assert!(recs.len() >= 5);
        assert!(recs.iter().any(|r| r.contains("fixity")));
    }

    #[test]
    fn test_policy_default_level() {
        let policy = PreservationPolicy::new(PreservationLevel::Logical);
        assert_eq!(policy.required_level("unknown"), PreservationLevel::Logical);
    }

    #[test]
    fn test_policy_specific_requirement() {
        let mut policy = PreservationPolicy::new(PreservationLevel::BitLevel);
        policy.set_requirement("video/x-matroska", PreservationLevel::Full);
        assert_eq!(
            policy.required_level("video/x-matroska"),
            PreservationLevel::Full
        );
        assert_eq!(
            policy.required_level("audio/flac"),
            PreservationLevel::BitLevel
        );
    }

    #[test]
    fn test_policy_meets_check() {
        let mut policy = PreservationPolicy::new(PreservationLevel::BitLevel);
        policy.set_requirement("video", PreservationLevel::Logical);

        let good = PreservationAssessment {
            fixity_verified: true,
            format_renderable: true,
            ..PreservationAssessment::new()
        };
        assert!(policy.meets_policy("video", &good));

        let bad = PreservationAssessment::new();
        // BitLevel fallback still >= BitLevel requirement for "other"
        assert!(policy.meets_policy("other", &bad));
    }
}
