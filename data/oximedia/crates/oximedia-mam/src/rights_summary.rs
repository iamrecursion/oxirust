//! Rights coverage summary for media assets.
//!
//! Tracks which rights categories are cleared and provides a
//! consolidated view of an asset's rights status.

#![allow(dead_code)]

use std::collections::HashMap;

/// The coverage state of a single rights category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RightsCoverage {
    /// Rights are fully cleared for all intended uses.
    Cleared,
    /// Clearance is still in progress.
    Pending,
    /// Rights could not be obtained.
    NotCleared,
    /// Rights are not applicable for this asset.
    NotApplicable,
}

impl RightsCoverage {
    /// Returns `true` when rights are cleared or not applicable.
    #[must_use]
    pub fn is_cleared(&self) -> bool {
        matches!(self, Self::Cleared | Self::NotApplicable)
    }

    /// Returns a short label suitable for display.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Cleared => "Cleared",
            Self::Pending => "Pending",
            Self::NotCleared => "Not Cleared",
            Self::NotApplicable => "N/A",
        }
    }
}

/// A consolidated rights summary for a single asset.
#[derive(Debug, Clone)]
pub struct RightsSummary {
    /// Asset this summary belongs to.
    pub asset_id: u64,
    /// Per-category coverage map.
    rights: HashMap<String, RightsCoverage>,
    /// Optional notes recorded by rights administrator.
    pub notes: Option<String>,
}

impl RightsSummary {
    /// Create a new summary with no rights entries.
    #[must_use]
    pub fn new(asset_id: u64) -> Self {
        Self {
            asset_id,
            rights: HashMap::new(),
            notes: None,
        }
    }

    /// Returns `true` when every recorded right is cleared.
    #[must_use]
    pub fn is_fully_cleared(&self) -> bool {
        !self.rights.is_empty() && self.rights.values().all(|r| r.is_cleared())
    }

    /// Number of rights categories recorded.
    #[must_use]
    pub fn category_count(&self) -> usize {
        self.rights.len()
    }

    /// Returns the coverage for a specific category.
    #[must_use]
    pub fn coverage_for(&self, category: &str) -> Option<&RightsCoverage> {
        self.rights.get(category)
    }

    /// Returns all categories that are not yet cleared.
    #[must_use]
    pub fn pending_categories(&self) -> Vec<&str> {
        self.rights
            .iter()
            .filter(|(_, v)| !v.is_cleared())
            .map(|(k, _)| k.as_str())
            .collect()
    }
}

/// Builder for constructing a [`RightsSummary`].
#[derive(Debug, Default)]
pub struct RightsSummaryBuilder {
    asset_id: u64,
    rights: HashMap<String, RightsCoverage>,
    notes: Option<String>,
}

impl RightsSummaryBuilder {
    /// Create a builder for the given asset id.
    #[must_use]
    pub fn new(asset_id: u64) -> Self {
        Self {
            asset_id,
            ..Default::default()
        }
    }

    /// Add a rights category with its coverage state.
    #[must_use]
    pub fn add_right(mut self, category: impl Into<String>, coverage: RightsCoverage) -> Self {
        self.rights.insert(category.into(), coverage);
        self
    }

    /// Attach an optional notes string.
    #[must_use]
    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Consume the builder and produce a [`RightsSummary`].
    #[must_use]
    pub fn build(self) -> RightsSummary {
        RightsSummary {
            asset_id: self.asset_id,
            rights: self.rights,
            notes: self.notes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rights_coverage_is_cleared() {
        assert!(RightsCoverage::Cleared.is_cleared());
        assert!(RightsCoverage::NotApplicable.is_cleared());
        assert!(!RightsCoverage::Pending.is_cleared());
        assert!(!RightsCoverage::NotCleared.is_cleared());
    }

    #[test]
    fn test_rights_coverage_label() {
        assert_eq!(RightsCoverage::Cleared.label(), "Cleared");
        assert_eq!(RightsCoverage::Pending.label(), "Pending");
        assert_eq!(RightsCoverage::NotCleared.label(), "Not Cleared");
        assert_eq!(RightsCoverage::NotApplicable.label(), "N/A");
    }

    #[test]
    fn test_rights_summary_new_is_not_fully_cleared() {
        let summary = RightsSummary::new(1);
        // No rights recorded — not fully cleared
        assert!(!summary.is_fully_cleared());
    }

    #[test]
    fn test_rights_summary_builder_fully_cleared() {
        let summary = RightsSummaryBuilder::new(42)
            .add_right("music", RightsCoverage::Cleared)
            .add_right("footage", RightsCoverage::Cleared)
            .add_right("talent", RightsCoverage::NotApplicable)
            .build();
        assert!(summary.is_fully_cleared());
        assert_eq!(summary.asset_id, 42);
    }

    #[test]
    fn test_rights_summary_not_fully_cleared_when_pending() {
        let summary = RightsSummaryBuilder::new(7)
            .add_right("music", RightsCoverage::Cleared)
            .add_right("footage", RightsCoverage::Pending)
            .build();
        assert!(!summary.is_fully_cleared());
    }

    #[test]
    fn test_rights_summary_category_count() {
        let summary = RightsSummaryBuilder::new(1)
            .add_right("a", RightsCoverage::Cleared)
            .add_right("b", RightsCoverage::Pending)
            .add_right("c", RightsCoverage::NotCleared)
            .build();
        assert_eq!(summary.category_count(), 3);
    }

    #[test]
    fn test_rights_summary_coverage_for() {
        let summary = RightsSummaryBuilder::new(1)
            .add_right("music", RightsCoverage::Cleared)
            .build();
        assert_eq!(
            summary.coverage_for("music"),
            Some(&RightsCoverage::Cleared)
        );
        assert!(summary.coverage_for("footage").is_none());
    }

    #[test]
    fn test_rights_summary_pending_categories() {
        let summary = RightsSummaryBuilder::new(1)
            .add_right("music", RightsCoverage::Cleared)
            .add_right("footage", RightsCoverage::Pending)
            .add_right("talent", RightsCoverage::NotCleared)
            .build();
        let mut pending = summary.pending_categories();
        pending.sort_unstable();
        assert_eq!(pending, vec!["footage", "talent"]);
    }

    #[test]
    fn test_rights_summary_notes() {
        let summary = RightsSummaryBuilder::new(1)
            .add_right("music", RightsCoverage::Cleared)
            .notes("Approved by legal on 2024-01-01")
            .build();
        assert!(summary
            .notes
            .as_deref()
            .expect("should succeed in test")
            .contains("legal"));
    }

    #[test]
    fn test_rights_summary_no_notes() {
        let summary = RightsSummaryBuilder::new(1).build();
        assert!(summary.notes.is_none());
    }

    #[test]
    fn test_rights_summary_add_right_overwrite() {
        // Adding the same category twice should use the last value
        let summary = RightsSummaryBuilder::new(1)
            .add_right("music", RightsCoverage::Pending)
            .add_right("music", RightsCoverage::Cleared)
            .build();
        assert_eq!(
            summary.coverage_for("music"),
            Some(&RightsCoverage::Cleared)
        );
        assert_eq!(summary.category_count(), 1);
    }

    #[test]
    fn test_fully_cleared_single_not_applicable() {
        let summary = RightsSummaryBuilder::new(99)
            .add_right("talent", RightsCoverage::NotApplicable)
            .build();
        assert!(summary.is_fully_cleared());
    }

    #[test]
    fn test_pending_categories_empty_when_all_cleared() {
        let summary = RightsSummaryBuilder::new(1)
            .add_right("music", RightsCoverage::Cleared)
            .add_right("talent", RightsCoverage::NotApplicable)
            .build();
        assert!(summary.pending_categories().is_empty());
    }
}
