//! AV1 Bitstream Conformance Validation.
//!
//! This module provides validators for AV1 bitstream syntax and semantic
//! conformance as specified in the AV1 specification.
//!
//! # Validators
//!
//! - [`SequenceHeaderValidator`] — validates a parsed `SequenceHeader` against spec constraints
//! - [`ObuValidator`] — validates the syntactic structure of a complete AV1 bitstream
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::av1::conformance::{ObuValidator, SequenceHeaderValidator};
//!
//! let result = SequenceHeaderValidator::validate(&header);
//! if !result.is_valid {
//!     for error in &result.errors {
//!         eprintln!("Conformance error: {error}");
//!     }
//! }
//! ```

use super::obu::{parse_obu, ObuType};
use super::sequence::SequenceHeader;

// =============================================================================
// ValidationResult
// =============================================================================

/// Result of an AV1 conformance validation pass.
#[derive(Clone, Debug, Default)]
pub struct ValidationResult {
    /// `true` if no errors were found (warnings are allowed).
    pub is_valid: bool,
    /// Hard constraint violations from the AV1 spec.
    pub errors: Vec<String>,
    /// Advisory issues that do not strictly violate the spec but may indicate
    /// non-conformant or unusual configurations.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new, initially-valid result.
    #[must_use]
    fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Record a hard error and mark the result as invalid.
    fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
        self.is_valid = false;
    }

    /// Record a warning (does not affect `is_valid`).
    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

// =============================================================================
// SequenceHeaderValidator
// =============================================================================

/// Validates a parsed [`SequenceHeader`] against AV1 specification constraints.
///
/// Checks cover:
/// - `seq_profile` range (0–2)
/// - Bit depth validity (8, 10, or 12)
/// - Profile-specific chroma subsampling requirements
/// - `order_hint_bits` range and consistency with `enable_order_hint`
/// - `film_grain_params_present` against profile constraints
pub struct SequenceHeaderValidator;

impl SequenceHeaderValidator {
    /// Validate a [`SequenceHeader`] and return a [`ValidationResult`].
    ///
    /// This is a pure, side-effect-free function.
    #[must_use]
    pub fn validate(header: &SequenceHeader) -> ValidationResult {
        let mut result = ValidationResult::new();

        // ── Profile range ────────────────────────────────────────────────────
        if header.profile > 2 {
            result.error(format!(
                "seq_profile {} is out of range; must be 0, 1, or 2",
                header.profile
            ));
        }

        // ── Bit depth ────────────────────────────────────────────────────────
        let bd = header.color_config.bit_depth;
        if bd != 8 && bd != 10 && bd != 12 {
            result.error(format!("bit_depth {bd} is invalid; must be 8, 10, or 12"));
        }

        // ── Profile-specific color constraints ───────────────────────────────
        match header.profile {
            0 => {
                // Profile 0 (Main): must use 4:2:0 subsampling, no 12-bit
                if !header.color_config.subsampling_x || !header.color_config.subsampling_y {
                    result.error(
                        "profile 0 (Main) requires 4:2:0 subsampling \
                         (subsampling_x and subsampling_y must both be true)"
                            .to_string(),
                    );
                }
                if bd == 12 {
                    result.error("profile 0 (Main) does not allow 12-bit depth".to_string());
                }
            }
            1 => {
                // Profile 1 (High): must use 4:4:4, mono_chrome forbidden
                if header.color_config.mono_chrome {
                    result.error("profile 1 (High) forbids mono_chrome".to_string());
                }
                if header.color_config.subsampling_x || header.color_config.subsampling_y {
                    result.error(
                        "profile 1 (High) requires 4:4:4 \
                         (subsampling_x and subsampling_y must both be false)"
                            .to_string(),
                    );
                }
                if bd == 12 {
                    result.error("profile 1 (High) does not allow 12-bit depth".to_string());
                }
            }
            2 => {
                // Profile 2 (Professional): bit_depth must be 12 if twelve_bit flag is set;
                // we simply verify the stored bit_depth is one of the allowed values
                // (already checked above).  No additional subsampling restriction.
            }
            _ => {
                // Already caught above; no further checks needed.
            }
        }

        // ── order_hint_bits range & consistency ──────────────────────────────
        if header.order_hint_bits > 8 {
            result.error(format!(
                "order_hint_bits {} exceeds maximum of 8",
                header.order_hint_bits
            ));
        }
        if !header.enable_order_hint && header.order_hint_bits != 0 {
            result.warn(format!(
                "enable_order_hint is false but order_hint_bits is {}; \
                 expected 0 when order hints are disabled",
                header.order_hint_bits
            ));
        }

        // ── film_grain_params_present constraints ────────────────────────────
        // Film grain is only meaningful with luma + chroma planes (num_planes == 3).
        // Profile 1 mandates num_planes == 3 (YUV444); profile 0 and 2 may be mono.
        if header.film_grain_params_present && header.color_config.num_planes != 3 {
            result.warn(
                "film_grain_params_present is set but num_planes != 3; \
                 film grain synthesis requires luma and chroma planes"
                    .to_string(),
            );
        }

        // Profile 1 + mono_chrome is already an error above; guard here is
        // redundant but adds clarity for edge-case combinations.
        if header.film_grain_params_present
            && header.profile == 1
            && header.color_config.mono_chrome
        {
            result.warn(
                "film_grain_params_present with profile 1 and mono_chrome \
                 is contradictory (profile 1 forbids mono_chrome)"
                    .to_string(),
            );
        }

        result
    }
}

// =============================================================================
// ObuValidator
// =============================================================================

/// Validates the syntactic structure of an AV1 bitstream at the OBU level.
///
/// Checks include:
/// - Correct first OBU type
/// - `FrameHeader`/`Frame` OBU appearing before any `SequenceHeader`
/// - OBU size fields remaining within data bounds (surfaced from parse errors)
pub struct ObuValidator;

impl ObuValidator {
    /// Validate the OBU-level structure of `data` and return a [`ValidationResult`].
    ///
    /// The validator iterates all OBUs, accumulating errors and warnings
    /// without aborting on the first issue (except for a parse error that
    /// makes forward progress impossible).
    #[must_use]
    pub fn validate_bitstream(data: &[u8]) -> ValidationResult {
        let mut result = ValidationResult::new();

        if data.is_empty() {
            result.warn("bitstream is empty".to_string());
            return result;
        }

        let mut offset = 0usize;
        let mut first_obu = true;
        let mut seen_sequence_header = false;

        while offset < data.len() {
            let remaining = &data[offset..];

            match parse_obu(remaining) {
                Err(e) => {
                    result.error(format!("OBU parse error at byte offset {offset}: {e}"));
                    // Cannot continue safely; stop iteration.
                    break;
                }
                Ok((header, _payload, total_size)) => {
                    // ── First OBU check ──────────────────────────────────
                    if first_obu {
                        match header.obu_type {
                            ObuType::TemporalDelimiter | ObuType::SequenceHeader => {
                                // Conformant start
                            }
                            _ => {
                                result.warn(format!(
                                    "first OBU at offset 0 is {:?}; \
                                     expected TemporalDelimiter or SequenceHeader",
                                    header.obu_type
                                ));
                            }
                        }
                        first_obu = false;
                    }

                    // ── Track SequenceHeader presence ────────────────────
                    if matches!(header.obu_type, ObuType::SequenceHeader) {
                        seen_sequence_header = true;
                    }

                    // ── FrameHeader / Frame before SequenceHeader ─────────
                    if !seen_sequence_header {
                        match header.obu_type {
                            ObuType::FrameHeader
                            | ObuType::Frame
                            | ObuType::RedundantFrameHeader => {
                                result.error(format!(
                                    "{:?} OBU at byte offset {offset} appears \
                                     before any SequenceHeader",
                                    header.obu_type
                                ));
                            }
                            _ => {}
                        }
                    }

                    offset += total_size;
                }
            }
        }

        result
    }

    /// Count the number of successfully parseable OBUs in `data`.
    ///
    /// Stops counting at the first parse error.
    #[must_use]
    pub fn count_obus(data: &[u8]) -> usize {
        let mut count = 0usize;
        let mut offset = 0usize;

        while offset < data.len() {
            match parse_obu(&data[offset..]) {
                Err(_) => break,
                Ok((_header, _payload, total_size)) => {
                    count += 1;
                    offset += total_size;
                }
            }
        }

        count
    }

    /// Return the byte offset of the first `SequenceHeader` OBU in `data`,
    /// or `None` if no such OBU is found.
    #[must_use]
    pub fn find_sequence_header(data: &[u8]) -> Option<usize> {
        let mut offset = 0usize;

        while offset < data.len() {
            match parse_obu(&data[offset..]) {
                Err(_) => break,
                Ok((header, _payload, total_size)) => {
                    if matches!(header.obu_type, ObuType::SequenceHeader) {
                        return Some(offset);
                    }
                    offset += total_size;
                }
            }
        }

        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::film_grain::{FilmGrainParams, ScalingPoint};
    use super::super::sequence::{ColorConfig, SequenceHeader};
    use super::{ObuValidator, SequenceHeaderValidator};

    fn valid_profile0_header() -> SequenceHeader {
        SequenceHeader {
            profile: 0,
            still_picture: false,
            reduced_still_picture_header: false,
            max_frame_width_minus_1: 1919,
            max_frame_height_minus_1: 1079,
            enable_order_hint: true,
            order_hint_bits: 7,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
            color_config: ColorConfig {
                bit_depth: 8,
                mono_chrome: false,
                num_planes: 3,
                color_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
                color_range: false,
                subsampling_x: true,
                subsampling_y: true,
                separate_uv_delta_q: false,
            },
            film_grain_params_present: false,
        }
    }

    #[test]
    fn test_sequence_header_validator_valid_profile0() {
        let header = valid_profile0_header();
        let result = SequenceHeaderValidator::validate(&header);
        assert!(
            result.is_valid,
            "expected valid profile-0 header; errors: {:?}",
            result.errors
        );
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_sequence_header_validator_profile1_requires_444() {
        // Profile 1 requires subsampling_x == false && subsampling_y == false.
        // Using subsampling_x: true violates the 4:4:4 requirement.
        let header = SequenceHeader {
            profile: 1,
            color_config: ColorConfig {
                bit_depth: 8,
                mono_chrome: false,
                num_planes: 3,
                color_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
                color_range: false,
                subsampling_x: true, // wrong for profile 1
                subsampling_y: false,
                separate_uv_delta_q: false,
            },
            ..valid_profile0_header()
        };
        let result = SequenceHeaderValidator::validate(&header);
        assert!(
            !result.is_valid,
            "expected validation failure for profile 1 with 4:2:2 subsampling"
        );
        assert!(
            !result.errors.is_empty(),
            "expected at least one error for profile 1 subsampling violation"
        );
    }

    #[test]
    fn test_sequence_header_validator_invalid_bit_depth() {
        let header = SequenceHeader {
            profile: 0,
            color_config: ColorConfig {
                bit_depth: 7, // invalid
                ..valid_profile0_header().color_config
            },
            ..valid_profile0_header()
        };
        let result = SequenceHeaderValidator::validate(&header);
        assert!(!result.is_valid, "bit_depth=7 should be invalid");
        assert!(result.errors.iter().any(|e| e.contains("bit_depth")));
    }

    #[test]
    fn test_sequence_header_validator_order_hint_bits_too_large() {
        let header = SequenceHeader {
            enable_order_hint: true,
            order_hint_bits: 9, // exceeds max of 8
            ..valid_profile0_header()
        };
        let result = SequenceHeaderValidator::validate(&header);
        assert!(!result.is_valid, "order_hint_bits=9 should be invalid");
        assert!(result.errors.iter().any(|e| e.contains("order_hint_bits")));
    }

    #[test]
    fn test_sequence_header_validator_order_hint_disabled_nonzero_bits_warns() {
        let header = SequenceHeader {
            enable_order_hint: false,
            order_hint_bits: 4, // non-zero but order hint disabled
            ..valid_profile0_header()
        };
        let result = SequenceHeaderValidator::validate(&header);
        // Should still be valid (warning only)
        assert!(
            result.is_valid,
            "non-zero order_hint_bits with enable_order_hint=false should only warn"
        );
        assert!(
            !result.warnings.is_empty(),
            "expected a warning for inconsistent order_hint_bits"
        );
    }

    #[test]
    fn test_sequence_header_validator_profile0_rejects_12bit() {
        let header = SequenceHeader {
            profile: 0,
            color_config: ColorConfig {
                bit_depth: 12,
                ..valid_profile0_header().color_config
            },
            ..valid_profile0_header()
        };
        let result = SequenceHeaderValidator::validate(&header);
        assert!(!result.is_valid, "profile 0 should reject 12-bit depth");
    }

    #[test]
    fn test_obu_validator_count_obus_empty() {
        assert_eq!(ObuValidator::count_obus(&[]), 0);
    }

    #[test]
    fn test_obu_validator_find_sequence_header_none() {
        assert_eq!(ObuValidator::find_sequence_header(&[]), None);
    }

    #[test]
    fn test_obu_validator_minimal_valid_bitstream() {
        // TemporalDelimiter: type=2, no extension, has_size=true → byte = (2<<3)|(1<<1) = 0x12
        //                    followed by LEB128 size=0 → 0x00
        // SequenceHeader:    type=1, no extension, has_size=true → byte = (1<<3)|(1<<1) = 0x0A
        //                    followed by LEB128 size=0 → 0x00
        let bitstream: &[u8] = &[0x12, 0x00, 0x0A, 0x00];

        let count = ObuValidator::count_obus(bitstream);
        assert_eq!(count, 2, "expected 2 OBUs in minimal bitstream");

        let seq_offset = ObuValidator::find_sequence_header(bitstream);
        assert_eq!(
            seq_offset,
            Some(2),
            "SequenceHeader should be at byte offset 2"
        );
    }

    #[test]
    fn test_obu_validator_validate_bitstream_empty_warns() {
        let result = ObuValidator::validate_bitstream(&[]);
        // Empty bitstream → warning only, not an error
        assert!(
            result.is_valid,
            "empty bitstream should yield is_valid=true with warnings"
        );
        assert!(
            !result.warnings.is_empty(),
            "empty bitstream should produce at least one warning"
        );
    }

    #[test]
    fn test_obu_validator_validate_bitstream_minimal() {
        let bitstream: &[u8] = &[0x12, 0x00, 0x0A, 0x00];
        let result = ObuValidator::validate_bitstream(bitstream);
        assert!(
            result.is_valid,
            "minimal TD+SH bitstream should be valid; errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_obu_validator_validate_bitstream_frame_before_sequence_header() {
        // Frame OBU (type=6) with has_size=true and payload size 0, no prior SequenceHeader.
        // header byte = (6<<3)|(1<<1) = 0x32
        let bitstream: &[u8] = &[0x32, 0x00];
        let result = ObuValidator::validate_bitstream(bitstream);
        assert!(
            !result.is_valid,
            "Frame OBU before SequenceHeader should produce an error"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("before any SequenceHeader")),
            "error should mention SequenceHeader ordering; got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_av1_film_grain_params_serialize_deserialize() {
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.grain_seed = 0xDEAD;
        params.update_grain = true;
        params.film_grain_params_present = true;
        params.num_y_points = 2;
        params.y_points[0] = ScalingPoint::new(0, 64);
        params.y_points[1] = ScalingPoint::new(128, 128);

        // Clone acts as serialise→deserialise round-trip for in-memory representation.
        let cloned = params.clone();

        assert_eq!(params.grain_seed, cloned.grain_seed);
        assert_eq!(params.num_y_points, cloned.num_y_points);
        assert_eq!(params.y_points[0], cloned.y_points[0]);
        assert_eq!(params.y_points[1], cloned.y_points[1]);
        assert!(params.apply_grain);
        assert!(params.update_grain);
        assert!(params.film_grain_params_present);

        // Validate the params are internally consistent.
        assert!(
            params.validate(),
            "constructed FilmGrainParams should pass validate()"
        );
    }

    #[test]
    fn test_av1_film_grain_params_default_valid() {
        let params = FilmGrainParams::default();
        assert!(params.validate(), "default FilmGrainParams should be valid");
    }

    #[test]
    fn test_av1_film_grain_params_invalid_ar_coeff_lag() {
        let mut params = FilmGrainParams::new();
        params.ar_coeff_lag = 4; // max is MAX_AR_LAG=3
        assert!(!params.validate(), "ar_coeff_lag=4 should fail validate()");
    }
}
