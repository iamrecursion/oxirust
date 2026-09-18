//! Dolby Vision metadata validation.
//!
//! This module provides a structured validator for Dolby Vision metadata blocks.
//! It checks L1 PQ ordering, trim-value bounds, and L6 MaxCLL/MaxFALL constraints
//! against the published Dolby specification limits.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dolbyvision::metadata_validator::{
//!     DvMetadata, DvMetadataValidator, Level1Info, Level6Info,
//! };
//!
//! let meta = DvMetadata {
//!     level1: Some(Level1Info { min_pq: 64, max_pq: 3500, avg_pq: 1200 }),
//!     level2_trims: vec![],
//!     level6: Some(Level6Info { max_cll: 1000, max_fall: 400 }),
//!     required_levels: vec![],
//! };
//!
//! let validator = DvMetadataValidator::new();
//! let result = validator.validate(&meta);
//! assert!(result.is_valid());
//! ```

use thiserror::Error;

// ─── Validation error ─────────────────────────────────────────────────────────

/// Specific validation error for a Dolby Vision metadata block.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DvValidationError {
    /// L1 `min_pq` is above the 12-bit maximum (4095).
    #[error("L1 min_pq {value} exceeds maximum {max}")]
    L1MinPqOutOfRange {
        /// Actual value found.
        value: u16,
        /// Allowed maximum.
        max: u16,
    },

    /// L1 `max_pq` is above the 12-bit maximum (4095).
    #[error("L1 max_pq {value} exceeds maximum {max}")]
    L1MaxPqOutOfRange {
        /// Actual value found.
        value: u16,
        /// Allowed maximum.
        max: u16,
    },

    /// L1 `avg_pq` is above the 12-bit maximum (4095).
    #[error("L1 avg_pq {value} exceeds maximum {max}")]
    L1AvgPqOutOfRange {
        /// Actual value found.
        value: u16,
        /// Allowed maximum.
        max: u16,
    },

    /// L1 values do not satisfy `min_pq <= avg_pq <= max_pq`.
    #[error("L1 ordering violation: min={min} avg={avg} max={max}")]
    L1OrderingViolation {
        /// `min_pq` value.
        min: u16,
        /// `avg_pq` value.
        avg: u16,
        /// `max_pq` value.
        max: u16,
    },

    /// L2 `trim_slope` is outside the valid signed 12-bit range.
    #[error("L2 trim_slope {value} out of range [{min}, {max}]")]
    L2TrimSlopeOutOfRange {
        /// Actual value.
        value: i16,
        /// Minimum allowed.
        min: i16,
        /// Maximum allowed.
        max: i16,
    },

    /// L2 `trim_offset` is outside the valid signed 12-bit range.
    #[error("L2 trim_offset {value} out of range [{min}, {max}]")]
    L2TrimOffsetOutOfRange {
        /// Actual value.
        value: i16,
        /// Minimum allowed.
        min: i16,
        /// Maximum allowed.
        max: i16,
    },

    /// L6 `max_cll` exceeds 10 000 nits.
    #[error("L6 MaxCLL {value} exceeds maximum {max} nits")]
    L6MaxCllOutOfRange {
        /// Actual value.
        value: u16,
        /// Allowed maximum.
        max: u16,
    },

    /// L6 `max_fall` exceeds 10 000 nits.
    #[error("L6 MaxFALL {value} exceeds maximum {max} nits")]
    L6MaxFallOutOfRange {
        /// Actual value.
        value: u16,
        /// Allowed maximum.
        max: u16,
    },

    /// L6 `max_fall` is greater than `max_cll`, which is physically impossible.
    #[error("L6 MaxFALL {max_fall} > MaxCLL {max_cll}: average cannot exceed peak")]
    L6MaxCllExceedsMaxFall {
        /// MaxCLL value.
        max_cll: u16,
        /// MaxFALL value.
        max_fall: u16,
    },

    /// A level listed in `required_levels` is absent from the metadata.
    #[error("Required metadata level {level} is missing")]
    MissingRequiredLevel {
        /// Level number that must be present.
        level: u8,
    },

    /// The metadata claims a profile that does not match the content type.
    #[error("Profile mismatch: expected {expected}, found {found}")]
    ProfileMismatch {
        /// Expected profile name.
        expected: String,
        /// Actual profile name.
        found: String,
    },
}

// ─── Metadata structures ──────────────────────────────────────────────────────

/// Level-1 PQ frame statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Level1Info {
    /// Minimum PQ code value (0–4095).
    pub min_pq: u16,
    /// Maximum PQ code value (0–4095).
    pub max_pq: u16,
    /// Average PQ code value (0–4095).
    pub avg_pq: u16,
}

/// Level-2 trim pass for a single target display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Level2TrimInfo {
    /// Target display index (decoder-assigned).
    pub target_display_index: u8,
    /// Trim slope (signed 12-bit range: -2048 .. 2047).
    pub trim_slope: i16,
    /// Trim offset (signed 12-bit range: -2048 .. 2047).
    pub trim_offset: i16,
    /// Trim power (signed 12-bit range: -2048 .. 2047).
    pub trim_power: i16,
}

/// Level-6 MaxCLL and MaxFALL values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Level6Info {
    /// Maximum Content Light Level in nits (0–10 000).
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level in nits (0–10 000).
    pub max_fall: u16,
}

/// Aggregated Dolby Vision metadata block submitted for validation.
#[derive(Debug, Clone, Default)]
pub struct DvMetadata {
    /// Level-1 frame statistics, if present.
    pub level1: Option<Level1Info>,
    /// Level-2 trim passes (zero or more target displays).
    pub level2_trims: Vec<Level2TrimInfo>,
    /// Level-6 MaxCLL / MaxFALL, if present.
    pub level6: Option<Level6Info>,
    /// Levels that the validator must confirm are present.
    pub required_levels: Vec<u8>,
}

// ─── ValidationResult ─────────────────────────────────────────────────────────

/// The result of running a [`DvMetadataValidator`] over a [`DvMetadata`] block.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Structural and specification errors.  Non-empty means the metadata is
    /// non-conformant.
    pub errors: Vec<DvValidationError>,
    /// Advisory messages that do not fail validation by themselves.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Returns `true` when no errors were found.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Number of errors collected.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Number of warnings collected.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

// ─── Validator ────────────────────────────────────────────────────────────────

/// Validates [`DvMetadata`] against Dolby Vision specification constraints.
///
/// Two modes are available:
///
/// * [`validate`] – permissive: missing L1 is only a warning.
/// * [`validate_strict`] – strict: missing L1 is a hard error.
///
/// [`validate`]: DvMetadataValidator::validate
/// [`validate_strict`]: DvMetadataValidator::validate_strict
#[derive(Debug, Clone, Default)]
pub struct DvMetadataValidator;

/// PQ 12-bit maximum code value.
const PQ_MAX: u16 = 4095;
/// Maximum valid nit value for MaxCLL / MaxFALL per SMPTE ST 2086.
const NITS_MAX: u16 = 10_000;
/// Signed 12-bit trim value boundaries.
const TRIM_MIN: i16 = -2048;
const TRIM_MAX: i16 = 2047;

impl DvMetadataValidator {
    /// Create a new validator instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate `metadata` in normal mode.
    ///
    /// A missing Level-1 block is reported as a warning only.
    #[must_use]
    pub fn validate(&self, metadata: &DvMetadata) -> ValidationResult {
        let mut result = ValidationResult::default();
        self.check_level1(metadata, &mut result, false);
        self.check_level2_trims(metadata, &mut result);
        self.check_level6(metadata, &mut result);
        self.check_required_levels(metadata, &mut result);
        result
    }

    /// Validate `metadata` in strict mode.
    ///
    /// Absence of Level-1 metadata is treated as a hard error.
    #[must_use]
    pub fn validate_strict(&self, metadata: &DvMetadata) -> ValidationResult {
        let mut result = ValidationResult::default();
        self.check_level1(metadata, &mut result, true);
        self.check_level2_trims(metadata, &mut result);
        self.check_level6(metadata, &mut result);
        self.check_required_levels(metadata, &mut result);
        result
    }

    // ── Internal checks ───────────────────────────────────────────────────────

    fn check_level1(&self, metadata: &DvMetadata, result: &mut ValidationResult, strict: bool) {
        match &metadata.level1 {
            None => {
                if strict {
                    result
                        .errors
                        .push(DvValidationError::MissingRequiredLevel { level: 1 });
                } else {
                    result
                        .warnings
                        .push("Level-1 metadata is absent".to_string());
                }
            }
            Some(l1) => {
                if l1.min_pq > PQ_MAX {
                    result.errors.push(DvValidationError::L1MinPqOutOfRange {
                        value: l1.min_pq,
                        max: PQ_MAX,
                    });
                }
                if l1.max_pq > PQ_MAX {
                    result.errors.push(DvValidationError::L1MaxPqOutOfRange {
                        value: l1.max_pq,
                        max: PQ_MAX,
                    });
                }
                if l1.avg_pq > PQ_MAX {
                    result.errors.push(DvValidationError::L1AvgPqOutOfRange {
                        value: l1.avg_pq,
                        max: PQ_MAX,
                    });
                }
                // Ordering: min <= avg <= max (only check when values are in range)
                if l1.min_pq <= PQ_MAX && l1.avg_pq <= PQ_MAX && l1.max_pq <= PQ_MAX {
                    if l1.min_pq > l1.avg_pq || l1.avg_pq > l1.max_pq {
                        result.errors.push(DvValidationError::L1OrderingViolation {
                            min: l1.min_pq,
                            avg: l1.avg_pq,
                            max: l1.max_pq,
                        });
                    }
                }
            }
        }
    }

    fn check_level2_trims(&self, metadata: &DvMetadata, result: &mut ValidationResult) {
        for trim in &metadata.level2_trims {
            if trim.trim_slope < TRIM_MIN || trim.trim_slope > TRIM_MAX {
                result
                    .errors
                    .push(DvValidationError::L2TrimSlopeOutOfRange {
                        value: trim.trim_slope,
                        min: TRIM_MIN,
                        max: TRIM_MAX,
                    });
            }
            if trim.trim_offset < TRIM_MIN || trim.trim_offset > TRIM_MAX {
                result
                    .errors
                    .push(DvValidationError::L2TrimOffsetOutOfRange {
                        value: trim.trim_offset,
                        min: TRIM_MIN,
                        max: TRIM_MAX,
                    });
            }
        }
    }

    fn check_level6(&self, metadata: &DvMetadata, result: &mut ValidationResult) {
        let l6 = match &metadata.level6 {
            None => return,
            Some(v) => v,
        };

        if l6.max_cll > NITS_MAX {
            result.errors.push(DvValidationError::L6MaxCllOutOfRange {
                value: l6.max_cll,
                max: NITS_MAX,
            });
        }
        if l6.max_fall > NITS_MAX {
            result.errors.push(DvValidationError::L6MaxFallOutOfRange {
                value: l6.max_fall,
                max: NITS_MAX,
            });
        }
        // MaxFALL physically cannot exceed MaxCLL.
        if l6.max_fall > l6.max_cll {
            result
                .errors
                .push(DvValidationError::L6MaxCllExceedsMaxFall {
                    max_cll: l6.max_cll,
                    max_fall: l6.max_fall,
                });
        }
    }

    fn check_required_levels(&self, metadata: &DvMetadata, result: &mut ValidationResult) {
        for &level in &metadata.required_levels {
            let present = match level {
                1 => metadata.level1.is_some(),
                2 => !metadata.level2_trims.is_empty(),
                6 => metadata.level6.is_some(),
                _ => false,
            };
            if !present {
                result
                    .errors
                    .push(DvValidationError::MissingRequiredLevel { level });
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_metadata() -> DvMetadata {
        DvMetadata {
            level1: Some(Level1Info {
                min_pq: 64,
                max_pq: 3500,
                avg_pq: 1200,
            }),
            level2_trims: vec![Level2TrimInfo {
                target_display_index: 0,
                trim_slope: 0,
                trim_offset: 0,
                trim_power: 0,
            }],
            level6: Some(Level6Info {
                max_cll: 1000,
                max_fall: 400,
            }),
            required_levels: vec![],
        }
    }

    // ── Happy path ───────────────────────────────────────────────────────────

    #[test]
    fn test_valid_metadata_passes() {
        let v = DvMetadataValidator::new();
        let result = v.validate(&valid_metadata());
        assert!(result.is_valid(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_validation_result_methods() {
        let result = ValidationResult {
            errors: vec![DvValidationError::MissingRequiredLevel { level: 1 }],
            warnings: vec!["w1".to_string(), "w2".to_string()],
        };
        assert!(!result.is_valid());
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.warning_count(), 2);
    }

    #[test]
    fn test_empty_metadata_permissive_has_warning() {
        let v = DvMetadataValidator::new();
        let meta = DvMetadata::default();
        let result = v.validate(&meta);
        assert!(result.is_valid(), "no errors expected in permissive mode");
        assert!(!result.warnings.is_empty(), "expect warning for absent L1");
    }

    // ── L1 range checks ──────────────────────────────────────────────────────

    #[test]
    fn test_l1_min_pq_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level1 = Some(Level1Info {
            min_pq: 5000,
            max_pq: 5000,
            avg_pq: 5000,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L1MinPqOutOfRange { .. })));
    }

    #[test]
    fn test_l1_max_pq_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level1 = Some(Level1Info {
            min_pq: 0,
            max_pq: 4096,
            avg_pq: 100,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L1MaxPqOutOfRange { .. })));
    }

    #[test]
    fn test_l1_avg_pq_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level1 = Some(Level1Info {
            min_pq: 0,
            max_pq: 3000,
            avg_pq: 9999,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L1AvgPqOutOfRange { .. })));
    }

    #[test]
    fn test_l1_ordering_violation_min_gt_max() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level1 = Some(Level1Info {
            min_pq: 2000,
            max_pq: 500,
            avg_pq: 1000,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L1OrderingViolation { .. })));
    }

    #[test]
    fn test_l1_ordering_violation_avg_gt_max() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level1 = Some(Level1Info {
            min_pq: 100,
            max_pq: 1000,
            avg_pq: 1500,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L1OrderingViolation { .. })));
    }

    // ── L2 trim checks ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_trim_slope_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level2_trims = vec![Level2TrimInfo {
            target_display_index: 0,
            trim_slope: 3000, // > 2047
            trim_offset: 0,
            trim_power: 0,
        }];
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L2TrimSlopeOutOfRange { .. })));
    }

    #[test]
    fn test_l2_trim_offset_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level2_trims = vec![Level2TrimInfo {
            target_display_index: 0,
            trim_slope: 0,
            trim_offset: -3000, // < -2048
            trim_power: 0,
        }];
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L2TrimOffsetOutOfRange { .. })));
    }

    // ── L6 checks ────────────────────────────────────────────────────────────

    #[test]
    fn test_l6_max_cll_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level6 = Some(Level6Info {
            max_cll: 10_001,
            max_fall: 400,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L6MaxCllOutOfRange { .. })));
    }

    #[test]
    fn test_l6_max_fall_out_of_range() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level6 = Some(Level6Info {
            max_cll: 1000,
            max_fall: 10_001,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L6MaxFallOutOfRange { .. })));
    }

    #[test]
    fn test_l6_max_fall_exceeds_max_cll() {
        let v = DvMetadataValidator::new();
        let mut meta = valid_metadata();
        meta.level6 = Some(Level6Info {
            max_cll: 400,
            max_fall: 1000,
        });
        let result = v.validate(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::L6MaxCllExceedsMaxFall { .. })));
    }

    // ── Required levels ──────────────────────────────────────────────────────

    #[test]
    fn test_missing_required_level() {
        let v = DvMetadataValidator::new();
        let mut meta = DvMetadata::default();
        meta.required_levels = vec![1, 6];
        let result = v.validate(&meta);
        // Both level 1 and level 6 are absent
        let missing: Vec<u8> = result
            .errors
            .iter()
            .filter_map(|e| {
                if let DvValidationError::MissingRequiredLevel { level } = e {
                    Some(*level)
                } else {
                    None
                }
            })
            .collect();
        assert!(missing.contains(&1));
        assert!(missing.contains(&6));
    }

    // ── Multiple errors accumulate ────────────────────────────────────────────

    #[test]
    fn test_multiple_errors_accumulate() {
        let v = DvMetadataValidator::new();
        let meta = DvMetadata {
            level1: Some(Level1Info {
                min_pq: 4096, // out of range
                max_pq: 100,  // < min → ordering violation too
                avg_pq: 50,
            }),
            level2_trims: vec![Level2TrimInfo {
                target_display_index: 0,
                trim_slope: 4000, // out of range
                trim_offset: 0,
                trim_power: 0,
            }],
            level6: Some(Level6Info {
                max_cll: 200,
                max_fall: 500, // > max_cll
            }),
            required_levels: vec![],
        };
        let result = v.validate(&meta);
        assert!(
            result.error_count() >= 3,
            "expected ≥3 errors, got {:?}",
            result.errors
        );
    }

    // ── Strict mode ──────────────────────────────────────────────────────────

    #[test]
    fn test_strict_mode_requires_l1() {
        let v = DvMetadataValidator::new();
        let meta = DvMetadata {
            level1: None,
            level2_trims: vec![],
            level6: None,
            required_levels: vec![],
        };
        let result = v.validate_strict(&meta);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, DvValidationError::MissingRequiredLevel { level: 1 })));
    }

    #[test]
    fn test_strict_mode_valid_with_l1() {
        let v = DvMetadataValidator::new();
        let result = v.validate_strict(&valid_metadata());
        assert!(result.is_valid(), "errors: {:?}", result.errors);
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_dv_validation_error_display() {
        let e = DvValidationError::L1MinPqOutOfRange {
            value: 5000,
            max: 4095,
        };
        let s = e.to_string();
        assert!(s.contains("5000") && s.contains("4095"));

        let e2 = DvValidationError::L6MaxCllExceedsMaxFall {
            max_cll: 400,
            max_fall: 1000,
        };
        let s2 = e2.to_string();
        assert!(s2.contains("400") && s2.contains("1000"));

        let e3 = DvValidationError::ProfileMismatch {
            expected: "8.1".to_string(),
            found: "8.4".to_string(),
        };
        let s3 = e3.to_string();
        assert!(s3.contains("8.1") && s3.contains("8.4"));
    }
}
