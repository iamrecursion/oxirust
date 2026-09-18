//! Dolby Vision RPU (Reference Processing Unit) metadata validation.
//!
//! This module validates RPU NAL unit bytes for Dolby Vision profile/level
//! compliance and backward-compatibility constraints.
//!
//! # RPU Header Layout (Simplified)
//!
//! An RPU NAL unit RBSP begins with:
//!
//! | Offset | Field             | Bits | Description                            |
//! |--------|-------------------|------|----------------------------------------|
//! | 0      | `rpu_nal_prefix`  | 8    | Must be `0x19` (decimal 25)            |
//! | 1      | `rpu_type`        | 4    | CM version (0 = CM v2.9)               |
//! | 1      | `rpu_format`      | 4    | Data format indicator                  |
//! | 2      | `vdr_rpu_profile` | 4    | DV profile (upper nibble)              |
//! | 2      | `vdr_rpu_level`   | 4    | DV level (lower nibble)                |
//!
//! At least 3 bytes of RBSP data are required for validation.
//! Start-code prefixes (`00 00 01` / `00 00 00 01`) are stripped before parsing.
//!
//! # Profile/Level Semantics
//!
//! | Profile | Description                              | Backward compat |
//! |---------|------------------------------------------|-----------------|
//! | 4       | Dual-layer, HEVC BL + EL                 | No              |
//! | 5       | Single-layer, HDR-only                   | No              |
//! | 7       | Dual-layer, HDR10 base                   | No              |
//! | 8       | CM v4.0, single-layer with BL compat.    | Yes (required)  |
//!
//! Level constraints (max mastering display peak luminance):
//!
//! | Level | Max nits |
//! |-------|----------|
//! | 1     | 600      |
//! | 2     | 600      |
//! | 3     | 600      |
//! | 4     | 1000     |
//! | 5     | 2000     |
//! | 6     | 4000     |
//! | 7     | 4000     |
//! | 8     | 4000     |
//! | 9     | 4000     |
//! | 10    | 4000     |
//! | 11    | 4000     |
//! | 12    | 4000     |
//! | 13    | 4000     |

/// Minimum number of RBSP bytes required for RPU header parsing.
const RPU_MIN_BYTES: usize = 3;

/// Expected first byte of a valid RPU RBSP (`rpu_nal_prefix`).
const RPU_NAL_PREFIX: u8 = 0x19;

/// Maximum start-code prefix length to scan past (4 bytes: 00 00 00 01).
const START_CODE_MAX_LEN: usize = 4;

/// Profile 8 — CM v4.0 with mandatory backward-layer compatibility.
const PROFILE_BACKWARD_COMPAT: u8 = 8;

/// Configuration for RPU metadata validation.
///
/// Controls which DV profiles and levels are accepted, and whether profile-8
/// backward-compatibility checking is enforced.
#[derive(Debug, Clone)]
pub struct RpuValidationConfig {
    /// DV profiles accepted by this validator (e.g. `[4, 5, 7, 8]`).
    ///
    /// An empty slice means *all defined profiles are accepted*.
    pub allowed_profiles: Vec<u8>,
    /// DV levels accepted by this validator (e.g. `[1, 4, 9]`).
    ///
    /// An empty slice means *all levels are accepted*.
    pub allowed_levels: Vec<u8>,
    /// When `true`, profile 8 content is additionally checked to confirm that a
    /// backward-compatible base layer flag is present in the RPU.
    ///
    /// Profile 8 is the CM v4.0 single-layer format designed for dual-track
    /// delivery; without the BL+EL compatibility signal the SDR/HDR10 fallback
    /// path breaks.
    pub require_backward_compat: bool,
}

impl Default for RpuValidationConfig {
    /// Returns a permissive default that accepts profiles 4, 5, 7, 8 and
    /// levels 1 through 13, with backward-compatibility enforcement enabled.
    fn default() -> Self {
        Self {
            allowed_profiles: vec![4, 5, 7, 8],
            allowed_levels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            require_backward_compat: true,
        }
    }
}

impl RpuValidationConfig {
    /// Create a new config with explicit profile and level lists.
    #[must_use]
    pub fn new(
        allowed_profiles: Vec<u8>,
        allowed_levels: Vec<u8>,
        require_backward_compat: bool,
    ) -> Self {
        Self {
            allowed_profiles,
            allowed_levels,
            require_backward_compat,
        }
    }

    /// Returns `true` if `profile` is in the allowed list, or if the list is empty.
    #[must_use]
    fn profile_allowed(&self, profile: u8) -> bool {
        self.allowed_profiles.is_empty() || self.allowed_profiles.contains(&profile)
    }

    /// Returns `true` if `level` is in the allowed list, or if the list is empty.
    #[must_use]
    fn level_allowed(&self, level: u8) -> bool {
        self.allowed_levels.is_empty() || self.allowed_levels.contains(&level)
    }
}

/// The result of validating a single RPU NAL unit.
#[derive(Debug, Clone)]
pub struct RpuValidationResult {
    /// DV profile extracted from the RPU header.
    pub profile: u8,
    /// DV level extracted from the RPU header.
    pub level: u8,
    /// `true` if the RPU passes all configured constraints.
    pub is_valid: bool,
    /// Human-readable error messages explaining any validation failures.
    ///
    /// Empty when `is_valid` is `true`.
    pub errors: Vec<String>,
}

impl RpuValidationResult {
    /// Creates a valid result with no errors.
    #[must_use]
    fn valid(profile: u8, level: u8) -> Self {
        Self {
            profile,
            level,
            is_valid: true,
            errors: Vec::new(),
        }
    }

    /// Creates an invalid result from a single error message (profile/level = 0).
    #[must_use]
    fn from_error(message: impl Into<String>) -> Self {
        Self {
            profile: 0,
            level: 0,
            is_valid: false,
            errors: vec![message.into()],
        }
    }

    /// Creates an invalid result carrying known profile/level values and errors.
    #[must_use]
    fn with_violations(profile: u8, level: u8, errors: Vec<String>) -> Self {
        Self {
            profile,
            level,
            is_valid: false,
            errors,
        }
    }

    /// Returns the maximum peak luminance allowed for this level in nits.
    ///
    /// Returns `None` for unrecognised levels.
    #[must_use]
    pub fn max_nits_for_level(level: u8) -> Option<u32> {
        match level {
            1..=3 => Some(600),
            4 => Some(1000),
            5 => Some(2000),
            6..=13 => Some(4000),
            _ => None,
        }
    }
}

/// Strip a leading H.264/H.265-style start-code prefix from `data`.
///
/// Handles both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`) variants.
/// Returns the slice after the start code, or `data` unchanged if no prefix is
/// found within the first [`START_CODE_MAX_LEN`] bytes.
fn strip_start_code(data: &[u8]) -> &[u8] {
    // 4-byte start code: 00 00 00 01
    if data.len() >= 4 && data[..4] == [0x00, 0x00, 0x00, 0x01] {
        return &data[4..];
    }
    // 3-byte start code: 00 00 01
    if data.len() >= 3 && data[..3] == [0x00, 0x00, 0x01] {
        return &data[3..];
    }
    data
}

/// Validate a raw RPU NAL unit byte slice against the given configuration.
///
/// # Parsing
///
/// The function first strips any leading start-code prefix, then interprets the
/// RBSP as:
///
/// - Byte 0 (`rpu_nal_prefix`): must equal `0x19`.
/// - Byte 1: `rpu_type` (bits 7-4) and `rpu_format` (bits 3-0) — consumed but
///   not validated beyond confirming the prefix.
/// - Byte 2: `vdr_rpu_profile` (bits 7-4, upper nibble) and `vdr_rpu_level`
///   (bits 3-0, lower nibble).
///
/// # Errors (returned as fields of [`RpuValidationResult`])
///
/// | Condition | Error message |
/// |-----------|---------------|
/// | Empty / too-short slice | `"insufficient data: need at least 3 bytes"` |
/// | `rpu_nal_prefix` ≠ 0x19 | `"invalid RPU NAL prefix: expected 0x19, got …"` |
/// | Profile not in `allowed_profiles` | `"unsupported profile …"` |
/// | Level not in `allowed_levels` | `"unsupported level …"` |
/// | Profile 8 + `require_backward_compat` but BL compat flag absent | `"profile 8 requires backward-compatible base layer …"` |
#[must_use]
pub fn validate_rpu_metadata(rpu_data: &[u8], config: &RpuValidationConfig) -> RpuValidationResult {
    // Strip any start-code prefix
    let rbsp = strip_start_code(rpu_data);

    // Guard: need at least 3 bytes of RBSP
    if rbsp.len() < RPU_MIN_BYTES {
        return RpuValidationResult::from_error(format!(
            "insufficient data: need at least {RPU_MIN_BYTES} bytes, got {}",
            rbsp.len()
        ));
    }

    // Byte 0: rpu_nal_prefix
    if rbsp[0] != RPU_NAL_PREFIX {
        return RpuValidationResult::from_error(format!(
            "invalid RPU NAL prefix: expected 0x{RPU_NAL_PREFIX:02X}, got 0x{:02X}",
            rbsp[0]
        ));
    }

    // Byte 2: vdr_rpu_profile (upper nibble) | vdr_rpu_level (lower nibble)
    let profile_byte = rbsp[2];
    let profile = profile_byte >> 4; // bits [7:4]
    let level = profile_byte & 0x0F; // bits [3:0]

    let mut errors: Vec<String> = Vec::new();

    // Profile check
    if !config.profile_allowed(profile) {
        errors.push(format!(
            "unsupported profile {profile}: not in allowed list {:?}",
            config.allowed_profiles
        ));
    }

    // Level check
    if !config.level_allowed(level) {
        errors.push(format!(
            "unsupported level {level}: not in allowed list {:?}",
            config.allowed_levels
        ));
    }

    // Profile 8 backward-compatibility check.
    //
    // For profile 8, the RPU carries a `bl_video_full_range_flag` and a
    // backward-compatibility signal in the extended RPU data.  In the
    // simplified header we check for the presence of a 4th byte with any
    // non-zero value as a proxy for the BL compat signal.  A real parser
    // would inspect `vdr_dm_data()` → `ext_metadata_block()` type 2.
    if profile == PROFILE_BACKWARD_COMPAT && config.require_backward_compat {
        let compat_present = rbsp.len() >= 4 && rbsp[3] != 0x00;
        if !compat_present {
            errors.push(format!(
                "profile 8 requires backward-compatible base layer (BL+EL): \
                 compat signal absent or byte[3] is 0x00"
            ));
        }
    }

    if errors.is_empty() {
        RpuValidationResult::valid(profile, level)
    } else {
        RpuValidationResult::with_violations(profile, level, errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a minimal valid RPU RBSP for the given profile and level.
    ///
    /// Layout:
    /// - byte 0: 0x19 (rpu_nal_prefix)
    /// - byte 1: 0x00 (rpu_type=0, rpu_format=0)
    /// - byte 2: (profile << 4) | level
    /// - byte 3: compat_byte (used for profile-8 BL compat signal)
    fn make_rpu(profile: u8, level: u8, compat_byte: u8) -> Vec<u8> {
        vec![
            RPU_NAL_PREFIX,
            0x00,
            (profile << 4) | (level & 0x0F),
            compat_byte,
        ]
    }

    /// Profile 5 RPU header → is_valid = true for allowed=[5]
    #[test]
    fn test_rpu_validation_valid_profile_5() {
        let rpu = make_rpu(5, 4, 0x00);
        let config = RpuValidationConfig {
            allowed_profiles: vec![5],
            allowed_levels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert_eq!(result.profile, 5, "profile should be 5");
        assert_eq!(result.level, 4, "level should be 4");
        assert!(
            result.is_valid,
            "profile 5, level 4 should be valid; errors: {:?}",
            result.errors
        );
        assert!(result.errors.is_empty());
    }

    /// Profile 3 RPU → errors contains "unsupported profile"
    #[test]
    fn test_rpu_validation_invalid_profile() {
        let rpu = make_rpu(3, 4, 0x00);
        let config = RpuValidationConfig {
            allowed_profiles: vec![4, 5, 7, 8],
            allowed_levels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(!result.is_valid, "profile 3 should be invalid");
        let has_profile_error = result
            .errors
            .iter()
            .any(|e| e.contains("unsupported profile"));
        assert!(
            has_profile_error,
            "errors should contain 'unsupported profile'; got: {:?}",
            result.errors
        );
    }

    /// L4 with allowed=[1,9] → invalid (level 4 not in allowed list)
    #[test]
    fn test_rpu_validation_level_constraint() {
        let rpu = make_rpu(5, 4, 0x00);
        let config = RpuValidationConfig {
            allowed_profiles: vec![5],
            allowed_levels: vec![1, 9], // L4 intentionally excluded
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(!result.is_valid, "level 4 not in [1,9] should be invalid");
        let has_level_error = result
            .errors
            .iter()
            .any(|e| e.contains("unsupported level"));
        assert!(
            has_level_error,
            "errors should contain 'unsupported level'; got: {:?}",
            result.errors
        );
    }

    /// Empty bytes → error "insufficient data"
    #[test]
    fn test_rpu_validation_empty_data() {
        let config = RpuValidationConfig::default();
        let result = validate_rpu_metadata(&[], &config);
        assert!(!result.is_valid, "empty data should be invalid");
        let has_data_error = result
            .errors
            .iter()
            .any(|e| e.contains("insufficient data"));
        assert!(
            has_data_error,
            "errors should contain 'insufficient data'; got: {:?}",
            result.errors
        );
    }

    /// Start-code prefix (00 00 01) is correctly stripped before parsing.
    #[test]
    fn test_rpu_validation_strips_3byte_start_code() {
        let mut rpu = vec![0x00u8, 0x00, 0x01]; // 3-byte start code
        rpu.extend_from_slice(&make_rpu(5, 1, 0x00));
        let config = RpuValidationConfig {
            allowed_profiles: vec![5],
            allowed_levels: vec![1],
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(
            result.is_valid,
            "should be valid after stripping 3-byte start code; errors: {:?}",
            result.errors
        );
        assert_eq!(result.profile, 5);
        assert_eq!(result.level, 1);
    }

    /// Start-code prefix (00 00 00 01) is correctly stripped before parsing.
    #[test]
    fn test_rpu_validation_strips_4byte_start_code() {
        let mut rpu = vec![0x00u8, 0x00, 0x00, 0x01]; // 4-byte start code
        rpu.extend_from_slice(&make_rpu(7, 6, 0x00));
        let config = RpuValidationConfig {
            allowed_profiles: vec![7],
            allowed_levels: vec![6],
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(
            result.is_valid,
            "should be valid after stripping 4-byte start code; errors: {:?}",
            result.errors
        );
        assert_eq!(result.profile, 7);
        assert_eq!(result.level, 6);
    }

    /// Profile 8 with compat byte present → valid
    #[test]
    fn test_rpu_validation_profile8_compat_present() {
        // compat_byte = 0x01 (non-zero) → BL compat signal present
        let rpu = make_rpu(8, 9, 0x01);
        let config = RpuValidationConfig {
            allowed_profiles: vec![8],
            allowed_levels: vec![9],
            require_backward_compat: true,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(
            result.is_valid,
            "profile 8 with compat byte should be valid; errors: {:?}",
            result.errors
        );
    }

    /// Profile 8 with compat byte absent → invalid when require_backward_compat=true
    #[test]
    fn test_rpu_validation_profile8_compat_absent() {
        // compat_byte = 0x00 → BL compat signal absent
        let rpu = make_rpu(8, 9, 0x00);
        let config = RpuValidationConfig {
            allowed_profiles: vec![8],
            allowed_levels: vec![9],
            require_backward_compat: true,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(
            !result.is_valid,
            "profile 8 without compat should be invalid"
        );
        let has_compat_error = result
            .errors
            .iter()
            .any(|e| e.contains("backward-compatible base layer"));
        assert!(
            has_compat_error,
            "errors should mention backward-compatible base layer; got: {:?}",
            result.errors
        );
    }

    /// Invalid `rpu_nal_prefix` byte → error
    #[test]
    fn test_rpu_validation_invalid_prefix() {
        let rpu = vec![0xFFu8, 0x00, 0x54]; // wrong prefix
        let config = RpuValidationConfig::default();
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(!result.is_valid, "wrong prefix should be invalid");
        let has_prefix_error = result
            .errors
            .iter()
            .any(|e| e.contains("invalid RPU NAL prefix"));
        assert!(
            has_prefix_error,
            "errors should mention invalid RPU NAL prefix; got: {:?}",
            result.errors
        );
    }

    /// Verify level max_nits_for_level helper.
    #[test]
    fn test_rpu_max_nits_for_level() {
        assert_eq!(RpuValidationResult::max_nits_for_level(1), Some(600));
        assert_eq!(RpuValidationResult::max_nits_for_level(4), Some(1000));
        assert_eq!(RpuValidationResult::max_nits_for_level(9), Some(4000));
        assert_eq!(RpuValidationResult::max_nits_for_level(0), None);
        assert_eq!(RpuValidationResult::max_nits_for_level(14), None);
    }

    /// Multiple violations in one RPU (bad profile + bad level)
    #[test]
    fn test_rpu_validation_multiple_violations() {
        let rpu = make_rpu(3, 6, 0x00); // profile 3 not allowed, level 6 not allowed
        let config = RpuValidationConfig {
            allowed_profiles: vec![5],
            allowed_levels: vec![1, 4],
            require_backward_compat: false,
        };
        let result = validate_rpu_metadata(&rpu, &config);
        assert!(!result.is_valid);
        assert_eq!(
            result.errors.len(),
            2,
            "expected 2 violation errors; got: {:?}",
            result.errors
        );
    }
}
