//! Dolby Vision trim-pass management.
//!
//! A trim pass maps source mastered content to a specific target display
//! (e.g. 100-nit SDR TV, 1000-nit HDR monitor). Different pass types
//! operate at different scope (global vs. per-shot).

#![allow(dead_code)]

/// Identifies a DV trim-pass by its level number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrimPassType {
    /// Level 1 global trim – frame-level brightness target.
    L1,
    /// Level 2 trim – per-target-display tone and saturation.
    L2,
    /// Level 3 trim – per-shot, reserved for future use.
    L3,
    /// Level 4 trim – alternative display target.
    L4,
    /// Level 6 trim – fallback SDR / HDR10 metadata.
    L6,
}

impl TrimPassType {
    /// Returns `true` when this pass applies globally (not per-shot).
    #[must_use]
    pub fn is_global(self) -> bool {
        matches!(self, Self::L1 | Self::L6)
    }

    /// Human-readable label for reporting.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::L1 => "L1-global",
            Self::L2 => "L2-per-target",
            Self::L3 => "L3-per-shot",
            Self::L4 => "L4-alt-target",
            Self::L6 => "L6-fallback",
        }
    }

    /// Priority order (lower = applied first).
    #[must_use]
    pub fn priority(self) -> u8 {
        match self {
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L3 => 3,
            Self::L4 => 4,
            Self::L6 => 5,
        }
    }
}

// ---------------------------------------------------------------------------

/// A single configured trim pass with its activation state.
#[derive(Debug, Clone)]
pub struct TrimPass {
    /// Which level this pass belongs to.
    pub pass_type: TrimPassType,
    /// Whether this pass is currently active.
    pub enabled: bool,
    /// Target peak luminance in nits (0 means unconfigured).
    pub target_nits: f32,
}

impl TrimPass {
    /// Create a new pass with the given type and peak luminance.
    #[must_use]
    pub fn new(pass_type: TrimPassType, target_nits: f32) -> Self {
        Self {
            pass_type,
            enabled: true,
            target_nits,
        }
    }

    /// Returns `true` if the pass is enabled and properly configured.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled && self.target_nits > 0.0
    }

    /// Disable this pass.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this pass.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

// ---------------------------------------------------------------------------

/// An ordered collection of [`TrimPass`] entries for a stream or shot.
#[derive(Debug, Clone, Default)]
pub struct TrimPassSet {
    passes: Vec<TrimPass>,
}

impl TrimPassSet {
    /// Create an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pass to the set. Duplicate types are allowed (some streams carry
    /// multiple L2 entries for different target displays).
    pub fn add(&mut self, pass: TrimPass) {
        self.passes.push(pass);
    }

    /// Return all currently active passes, sorted by priority.
    #[must_use]
    pub fn active_passes(&self) -> Vec<&TrimPass> {
        let mut active: Vec<&TrimPass> = self.passes.iter().filter(|p| p.is_active()).collect();
        active.sort_by_key(|p| p.pass_type.priority());
        active
    }

    /// Returns `true` if all passes pass basic validation (target_nits > 0 when enabled).
    #[must_use]
    pub fn validate(&self) -> bool {
        self.passes
            .iter()
            .all(|p| !p.enabled || p.target_nits > 0.0)
    }

    /// Total number of passes (active + inactive).
    #[must_use]
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Returns `true` if the set contains no passes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Count passes matching a specific type.
    #[must_use]
    pub fn count_type(&self, pass_type: TrimPassType) -> usize {
        self.passes
            .iter()
            .filter(|p| p.pass_type == pass_type)
            .count()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_is_global() {
        assert!(TrimPassType::L1.is_global());
    }

    #[test]
    fn test_l6_is_global() {
        assert!(TrimPassType::L6.is_global());
    }

    #[test]
    fn test_l2_is_not_global() {
        assert!(!TrimPassType::L2.is_global());
    }

    #[test]
    fn test_l3_is_not_global() {
        assert!(!TrimPassType::L3.is_global());
    }

    #[test]
    fn test_trim_pass_type_labels_distinct() {
        let labels: Vec<_> = [
            TrimPassType::L1,
            TrimPassType::L2,
            TrimPassType::L3,
            TrimPassType::L4,
            TrimPassType::L6,
        ]
        .iter()
        .map(|t| t.label())
        .collect();
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
    }

    #[test]
    fn test_trim_pass_is_active_when_enabled_and_nits_positive() {
        let p = TrimPass::new(TrimPassType::L2, 1000.0);
        assert!(p.is_active());
    }

    #[test]
    fn test_trim_pass_not_active_when_disabled() {
        let mut p = TrimPass::new(TrimPassType::L2, 1000.0);
        p.disable();
        assert!(!p.is_active());
    }

    #[test]
    fn test_trim_pass_not_active_when_zero_nits() {
        let p = TrimPass::new(TrimPassType::L2, 0.0);
        assert!(!p.is_active());
    }

    #[test]
    fn test_trim_pass_enable_disable_roundtrip() {
        let mut p = TrimPass::new(TrimPassType::L1, 400.0);
        p.disable();
        assert!(!p.is_active());
        p.enable();
        assert!(p.is_active());
    }

    #[test]
    fn test_set_add_and_len() {
        let mut s = TrimPassSet::new();
        s.add(TrimPass::new(TrimPassType::L1, 1000.0));
        s.add(TrimPass::new(TrimPassType::L2, 400.0));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_set_active_passes_sorted_by_priority() {
        let mut s = TrimPassSet::new();
        s.add(TrimPass::new(TrimPassType::L6, 100.0));
        s.add(TrimPass::new(TrimPassType::L1, 1000.0));
        let active = s.active_passes();
        assert_eq!(active[0].pass_type, TrimPassType::L1);
        assert_eq!(active[1].pass_type, TrimPassType::L6);
    }

    #[test]
    fn test_set_validate_passes_when_all_configured() {
        let mut s = TrimPassSet::new();
        s.add(TrimPass::new(TrimPassType::L1, 1000.0));
        assert!(s.validate());
    }

    #[test]
    fn test_set_validate_fails_when_enabled_pass_has_zero_nits() {
        let mut s = TrimPassSet::new();
        s.add(TrimPass::new(TrimPassType::L2, 0.0));
        assert!(!s.validate());
    }

    #[test]
    fn test_set_count_type() {
        let mut s = TrimPassSet::new();
        s.add(TrimPass::new(TrimPassType::L2, 1000.0));
        s.add(TrimPass::new(TrimPassType::L2, 400.0));
        s.add(TrimPass::new(TrimPassType::L1, 100.0));
        assert_eq!(s.count_type(TrimPassType::L2), 2);
        assert_eq!(s.count_type(TrimPassType::L1), 1);
    }

    #[test]
    fn test_set_is_empty() {
        let s = TrimPassSet::new();
        assert!(s.is_empty());
    }
}
