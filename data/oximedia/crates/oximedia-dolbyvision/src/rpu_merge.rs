//! RPU merging: combine metadata from multiple Dolby Vision RPU sources.
//!
//! Real-world workflows sometimes produce separate RPU streams (e.g. one from an
//! on-set grading tool, another from a mastering system) that must be reconciled
//! into a single authoritative RPU per frame.  This module provides the
//! `RpuMerger` type which implements configurable merge strategies for each
//! metadata level.

use crate::{DolbyVisionError, DolbyVisionRpu, Level1Metadata, Result};

/// Strategy used to resolve conflicts when two RPUs carry different values for
/// the same metadata level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Always take the value from the first (primary) RPU.
    PreferPrimary,
    /// Always take the value from the second (secondary) RPU.
    PreferSecondary,
    /// When both sources have a level, take the one with the higher max-PQ.
    PreferHigherPeak,
    /// When both sources have a level, take the one with the lower max-PQ.
    PreferLowerPeak,
    /// Merge Level 1 min/max/avg by taking the statistical union
    /// (min of mins, max of maxes, average of avgs).  Falls back to
    /// `PreferPrimary` for non-L1 levels.
    StatisticalUnion,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::PreferPrimary
    }
}

/// Configuration for the merge operation.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Strategy for Level 1 (frame-level PQ range).
    pub level1_strategy: MergeStrategy,
    /// Strategy for Level 2 (trim passes).
    pub level2_strategy: MergeStrategy,
    /// Strategy for Level 4 (global dimming).
    pub level4_strategy: MergeStrategy,
    /// Strategy for Level 5 (active area).
    pub level5_strategy: MergeStrategy,
    /// Strategy for Level 6 (fallback HDR10 metadata).
    pub level6_strategy: MergeStrategy,
    /// Strategy for Level 7 (source display color volume).
    pub level7_strategy: MergeStrategy,
    /// Strategy for Level 8 (target display).
    pub level8_strategy: MergeStrategy,
    /// Strategy for Level 9 (source display).
    pub level9_strategy: MergeStrategy,
    /// Strategy for Level 11 (content type).
    pub level11_strategy: MergeStrategy,
    /// If `true`, validate the merged RPU before returning it.
    pub validate_result: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            level1_strategy: MergeStrategy::StatisticalUnion,
            level2_strategy: MergeStrategy::PreferPrimary,
            level4_strategy: MergeStrategy::PreferPrimary,
            level5_strategy: MergeStrategy::PreferPrimary,
            level6_strategy: MergeStrategy::PreferHigherPeak,
            level7_strategy: MergeStrategy::PreferPrimary,
            level8_strategy: MergeStrategy::PreferPrimary,
            level9_strategy: MergeStrategy::PreferPrimary,
            level11_strategy: MergeStrategy::PreferPrimary,
            validate_result: true,
        }
    }
}

/// Merge two Dolby Vision RPUs into one authoritative RPU.
///
/// The `primary` RPU provides the profile and header; the `secondary` RPU
/// supplies any metadata levels absent from `primary`, or its values are
/// merged according to `config`.
///
/// # Errors
///
/// Returns [`DolbyVisionError`] if the RPUs come from incompatible profiles or
/// if post-merge validation fails (when `config.validate_result` is `true`).
pub fn merge_rpus(
    primary: &DolbyVisionRpu,
    secondary: &DolbyVisionRpu,
    config: &MergeConfig,
) -> Result<DolbyVisionRpu> {
    // Profile compatibility check
    if primary.profile != secondary.profile {
        return Err(DolbyVisionError::Generic(format!(
            "Cannot merge RPUs with different profiles: {:?} vs {:?}",
            primary.profile, secondary.profile
        )));
    }

    let mut merged = primary.clone();

    // ── Level 1 ──────────────────────────────────────────────────────────────
    merged.level1 = merge_option_level1(
        primary.level1.as_ref(),
        secondary.level1.as_ref(),
        config.level1_strategy,
    );

    // ── Level 2 ──────────────────────────────────────────────────────────────
    merged.level2 = choose_option(
        primary.level2.as_ref(),
        secondary.level2.as_ref(),
        config.level2_strategy,
        None,
    );

    // ── Level 4 ──────────────────────────────────────────────────────────────
    merged.level4 = choose_option(
        primary.level4.as_ref(),
        secondary.level4.as_ref(),
        config.level4_strategy,
        None,
    );

    // ── Level 5 ──────────────────────────────────────────────────────────────
    merged.level5 = choose_option(
        primary.level5.as_ref(),
        secondary.level5.as_ref(),
        config.level5_strategy,
        None,
    );

    // ── Level 6 ──────────────────────────────────────────────────────────────
    merged.level6 = merge_option_level6_peak(
        primary.level6.as_ref(),
        secondary.level6.as_ref(),
        config.level6_strategy,
    );

    // ── Level 7 ──────────────────────────────────────────────────────────────
    merged.level7 = choose_option(
        primary.level7.as_ref(),
        secondary.level7.as_ref(),
        config.level7_strategy,
        None,
    );

    // ── Level 8 ──────────────────────────────────────────────────────────────
    merged.level8 = choose_option(
        primary.level8.as_ref(),
        secondary.level8.as_ref(),
        config.level8_strategy,
        None,
    );

    // ── Level 9 ──────────────────────────────────────────────────────────────
    merged.level9 = choose_option(
        primary.level9.as_ref(),
        secondary.level9.as_ref(),
        config.level9_strategy,
        None,
    );

    // ── Level 11 ─────────────────────────────────────────────────────────────
    merged.level11 = choose_option(
        primary.level11.as_ref(),
        secondary.level11.as_ref(),
        config.level11_strategy,
        None,
    );

    // Validate if requested
    if config.validate_result {
        merged.validate()?;
    }

    Ok(merged)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Choose between two optional values using a `MergeStrategy`.
///
/// `peak_key` is an optional function extracting a numeric sort key used by
/// `PreferHigherPeak` / `PreferLowerPeak`.  When `None`, those strategies
/// fall back to `PreferPrimary`.
fn choose_option<T: Clone>(
    primary: Option<&T>,
    secondary: Option<&T>,
    strategy: MergeStrategy,
    _peak_key: Option<fn(&T) -> u16>,
) -> Option<T> {
    match (primary, secondary) {
        (Some(p), None) => Some(p.clone()),
        (None, Some(s)) => Some(s.clone()),
        (None, None) => None,
        (Some(p), Some(s)) => match strategy {
            MergeStrategy::PreferSecondary => Some(s.clone()),
            _ => Some(p.clone()),
        },
    }
}

/// Merge Level 1 metadata using the chosen strategy.
fn merge_option_level1(
    primary: Option<&Level1Metadata>,
    secondary: Option<&Level1Metadata>,
    strategy: MergeStrategy,
) -> Option<Level1Metadata> {
    match (primary, secondary) {
        (Some(p), None) => Some(p.clone()),
        (None, Some(s)) => Some(s.clone()),
        (None, None) => None,
        (Some(p), Some(s)) => match strategy {
            MergeStrategy::StatisticalUnion => {
                let min_pq = p.min_pq.min(s.min_pq);
                let max_pq = p.max_pq.max(s.max_pq);
                let avg_pq = ((u32::from(p.avg_pq) + u32::from(s.avg_pq)) / 2) as u16;
                Some(Level1Metadata {
                    min_pq,
                    max_pq,
                    avg_pq: avg_pq.clamp(min_pq, max_pq),
                })
            }
            MergeStrategy::PreferHigherPeak => {
                if s.max_pq > p.max_pq {
                    Some(s.clone())
                } else {
                    Some(p.clone())
                }
            }
            MergeStrategy::PreferLowerPeak => {
                if s.max_pq < p.max_pq {
                    Some(s.clone())
                } else {
                    Some(p.clone())
                }
            }
            MergeStrategy::PreferSecondary => Some(s.clone()),
            _ => Some(p.clone()),
        },
    }
}

/// Merge Level 6 metadata, respecting peak-preference strategies.
fn merge_option_level6_peak(
    primary: Option<&crate::Level6Metadata>,
    secondary: Option<&crate::Level6Metadata>,
    strategy: MergeStrategy,
) -> Option<crate::Level6Metadata> {
    match (primary, secondary) {
        (Some(p), None) => Some(p.clone()),
        (None, Some(s)) => Some(s.clone()),
        (None, None) => None,
        (Some(p), Some(s)) => match strategy {
            MergeStrategy::PreferHigherPeak => {
                if s.max_display_mastering_luminance > p.max_display_mastering_luminance {
                    Some(s.clone())
                } else {
                    Some(p.clone())
                }
            }
            MergeStrategy::PreferLowerPeak => {
                if s.max_display_mastering_luminance < p.max_display_mastering_luminance {
                    Some(s.clone())
                } else {
                    Some(p.clone())
                }
            }
            MergeStrategy::PreferSecondary => Some(s.clone()),
            _ => Some(p.clone()),
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DolbyVisionRpu, Level1Metadata, Level6Metadata, Profile};

    fn make_rpu_with_l1(min: u16, max: u16, avg: u16) -> DolbyVisionRpu {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: min,
            max_pq: max,
            avg_pq: avg,
        });
        rpu
    }

    #[test]
    fn test_merge_identical_profiles_ok() {
        let p = DolbyVisionRpu::new(Profile::Profile8);
        let s = DolbyVisionRpu::new(Profile::Profile8);
        let result = merge_rpus(&p, &s, &MergeConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_different_profiles_error() {
        let p = DolbyVisionRpu::new(Profile::Profile8);
        let s = DolbyVisionRpu::new(Profile::Profile5);
        let result = merge_rpus(&p, &s, &MergeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_statistical_union_l1() {
        let primary = make_rpu_with_l1(100, 3000, 1500);
        let secondary = make_rpu_with_l1(50, 4000, 2000);
        let cfg = MergeConfig {
            level1_strategy: MergeStrategy::StatisticalUnion,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l1 = merged.level1.expect("L1 should be present");
        assert_eq!(l1.min_pq, 50);
        assert_eq!(l1.max_pq, 4000);
        // avg should be average of 1500 and 2000 = 1750
        assert_eq!(l1.avg_pq, 1750);
    }

    #[test]
    fn test_merge_prefer_primary_l1() {
        let primary = make_rpu_with_l1(100, 3000, 1500);
        let secondary = make_rpu_with_l1(50, 4000, 2000);
        let cfg = MergeConfig {
            level1_strategy: MergeStrategy::PreferPrimary,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l1 = merged.level1.expect("L1 should be present");
        assert_eq!(l1.max_pq, 3000); // primary's value
    }

    #[test]
    fn test_merge_prefer_secondary_l1() {
        let primary = make_rpu_with_l1(100, 3000, 1500);
        let secondary = make_rpu_with_l1(50, 4000, 2000);
        let cfg = MergeConfig {
            level1_strategy: MergeStrategy::PreferSecondary,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l1 = merged.level1.expect("L1 should be present");
        assert_eq!(l1.max_pq, 4000); // secondary's value
    }

    #[test]
    fn test_merge_prefer_higher_peak_l1() {
        let primary = make_rpu_with_l1(100, 3000, 1500);
        let secondary = make_rpu_with_l1(50, 4000, 2000);
        let cfg = MergeConfig {
            level1_strategy: MergeStrategy::PreferHigherPeak,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l1 = merged.level1.expect("L1 should be present");
        assert_eq!(l1.max_pq, 4000);
    }

    #[test]
    fn test_merge_prefer_lower_peak_l1() {
        let primary = make_rpu_with_l1(100, 3000, 1500);
        let secondary = make_rpu_with_l1(50, 4000, 2000);
        let cfg = MergeConfig {
            level1_strategy: MergeStrategy::PreferLowerPeak,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l1 = merged.level1.expect("L1 should be present");
        assert_eq!(l1.max_pq, 3000);
    }

    #[test]
    fn test_merge_primary_only_l1() {
        let mut primary = DolbyVisionRpu::new(Profile::Profile8);
        primary.level1 = Some(Level1Metadata {
            min_pq: 10,
            max_pq: 2000,
            avg_pq: 1000,
        });
        let secondary = DolbyVisionRpu::new(Profile::Profile8);
        let merged = merge_rpus(&primary, &secondary, &MergeConfig::default())
            .expect("merge should succeed");
        let l1 = merged.level1.expect("L1 from primary should be present");
        assert_eq!(l1.max_pq, 2000);
    }

    #[test]
    fn test_merge_secondary_only_l1() {
        let primary = DolbyVisionRpu::new(Profile::Profile8);
        let mut secondary = DolbyVisionRpu::new(Profile::Profile8);
        secondary.level1 = Some(Level1Metadata {
            min_pq: 10,
            max_pq: 3000,
            avg_pq: 1500,
        });
        let merged = merge_rpus(&primary, &secondary, &MergeConfig::default())
            .expect("merge should succeed");
        let l1 = merged.level1.expect("L1 from secondary should be present");
        assert_eq!(l1.max_pq, 3000);
    }

    #[test]
    fn test_merge_level6_prefer_higher_peak() {
        let mut primary = DolbyVisionRpu::new(Profile::Profile8);
        primary.level6 = Some(Level6Metadata::bt2020());

        let mut secondary = DolbyVisionRpu::new(Profile::Profile8);
        let mut l6 = Level6Metadata::bt2020();
        l6.max_display_mastering_luminance = 4000;
        secondary.level6 = Some(l6);

        let cfg = MergeConfig {
            level6_strategy: MergeStrategy::PreferHigherPeak,
            validate_result: false,
            ..MergeConfig::default()
        };
        let merged = merge_rpus(&primary, &secondary, &cfg).expect("merge should succeed");
        let l6 = merged.level6.expect("L6 should be present");
        assert_eq!(l6.max_display_mastering_luminance, 4000);
    }

    #[test]
    fn test_merge_config_default() {
        let cfg = MergeConfig::default();
        assert_eq!(cfg.level1_strategy, MergeStrategy::StatisticalUnion);
        assert!(cfg.validate_result);
    }

    #[test]
    fn test_merge_strategy_default() {
        assert_eq!(MergeStrategy::default(), MergeStrategy::PreferPrimary);
    }
}
