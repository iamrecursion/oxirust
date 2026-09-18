//! Dolby Vision delivery specification types and validation.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// DvProfile
// ---------------------------------------------------------------------------

/// Dolby Vision delivery profile identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DvProfile {
    /// Profile 4: IPT-PQ, no backward compatibility.
    Profile4,
    /// Profile 5: IPT-PQ, backward compatible with HDR10.
    Profile5,
    /// Profile 7: MEL dual-track with HDR10 base.
    Profile7,
    /// Profile 8.1: BL + RPU, low-latency HDR10 backward compat.
    Profile8_1,
    /// Profile 8.2: BL + RPU, HDR10 backward compat (standard latency).
    Profile8_2,
    /// Profile 8.4: BL + RPU, HLG backward compat.
    Profile8_4,
}

impl DvProfile {
    /// Human-readable name of the base layer format for this profile.
    #[must_use]
    pub fn base_layer_format(&self) -> &str {
        match self {
            DvProfile::Profile4 => "IPT-PQ",
            DvProfile::Profile5 => "IPT-PQ/HDR10",
            DvProfile::Profile7 => "HDR10-MEL",
            DvProfile::Profile8_1 | DvProfile::Profile8_2 => "HDR10",
            DvProfile::Profile8_4 => "HLG",
        }
    }

    /// Returns `true` if the profile includes backward compatibility signalling.
    #[must_use]
    pub fn supports_backward_compat(&self) -> bool {
        matches!(
            self,
            DvProfile::Profile5
                | DvProfile::Profile7
                | DvProfile::Profile8_1
                | DvProfile::Profile8_2
                | DvProfile::Profile8_4
        )
    }
}

// ---------------------------------------------------------------------------
// DvDeliverySpec
// ---------------------------------------------------------------------------

/// Delivery specification for a Dolby Vision asset.
#[derive(Debug, Clone, PartialEq)]
pub struct DvDeliverySpec {
    /// Target Dolby Vision profile.
    pub profile: DvProfile,
    /// Maximum Content Light Level in nits.
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level in nits.
    pub max_fall: u16,
    /// Maximum display luminance in nits.
    pub max_luminance_nits: f32,
}

impl DvDeliverySpec {
    /// Create a new delivery specification.
    #[must_use]
    pub fn new(profile: DvProfile, max_cll: u16, max_fall: u16, max_luminance_nits: f32) -> Self {
        Self {
            profile,
            max_cll,
            max_fall,
            max_luminance_nits,
        }
    }

    /// Returns `true` if this spec is compatible with HDR10 playback.
    ///
    /// HDR10 compatibility requires: an HDR10-backward-compatible profile,
    /// non-zero MaxCLL and MaxFALL, and luminance ≥ 1000 nits.
    #[must_use]
    pub fn is_hdr10_compatible(&self) -> bool {
        matches!(
            self.profile,
            DvProfile::Profile5
                | DvProfile::Profile7
                | DvProfile::Profile8_1
                | DvProfile::Profile8_2
        ) && self.max_cll > 0
            && self.max_fall > 0
            && self.max_luminance_nits >= 1000.0
    }
}

// ---------------------------------------------------------------------------
// DeliveryValidator
// ---------------------------------------------------------------------------

/// Validates a [`DvDeliverySpec`] against common delivery requirements.
pub struct DeliveryValidator;

impl DeliveryValidator {
    /// Validate a delivery spec.
    ///
    /// Returns a list of human-readable error strings.  An empty list means
    /// the spec is valid.
    #[must_use]
    pub fn validate(spec: &DvDeliverySpec) -> Vec<String> {
        let mut errors = Vec::new();

        if spec.max_cll == 0 {
            errors.push("MaxCLL must be greater than 0".to_string());
        }
        if spec.max_fall == 0 {
            errors.push("MaxFALL must be greater than 0".to_string());
        }
        if spec.max_luminance_nits < 100.0 {
            errors.push(format!(
                "max_luminance_nits ({}) is below minimum of 100 nits",
                spec.max_luminance_nits
            ));
        }
        if spec.max_luminance_nits > 10_000.0 {
            errors.push(format!(
                "max_luminance_nits ({}) exceeds maximum of 10000 nits",
                spec.max_luminance_nits
            ));
        }

        errors
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DvProfile ---

    #[test]
    fn test_profile4_base_layer() {
        assert_eq!(DvProfile::Profile4.base_layer_format(), "IPT-PQ");
    }

    #[test]
    fn test_profile5_base_layer() {
        assert_eq!(DvProfile::Profile5.base_layer_format(), "IPT-PQ/HDR10");
    }

    #[test]
    fn test_profile7_base_layer() {
        assert_eq!(DvProfile::Profile7.base_layer_format(), "HDR10-MEL");
    }

    #[test]
    fn test_profile8_1_base_layer() {
        assert_eq!(DvProfile::Profile8_1.base_layer_format(), "HDR10");
    }

    #[test]
    fn test_profile8_4_base_layer() {
        assert_eq!(DvProfile::Profile8_4.base_layer_format(), "HLG");
    }

    #[test]
    fn test_backward_compat_profile4_false() {
        assert!(!DvProfile::Profile4.supports_backward_compat());
    }

    #[test]
    fn test_backward_compat_profile5_true() {
        assert!(DvProfile::Profile5.supports_backward_compat());
    }

    #[test]
    fn test_backward_compat_profile8_4_true() {
        assert!(DvProfile::Profile8_4.supports_backward_compat());
    }

    // --- DvDeliverySpec ---

    #[test]
    fn test_hdr10_compatible_profile5() {
        let spec = DvDeliverySpec::new(DvProfile::Profile5, 1000, 400, 4000.0);
        assert!(spec.is_hdr10_compatible());
    }

    #[test]
    fn test_hdr10_compatible_profile8_2() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_2, 1000, 400, 1000.0);
        assert!(spec.is_hdr10_compatible());
    }

    #[test]
    fn test_hdr10_not_compatible_profile4() {
        let spec = DvDeliverySpec::new(DvProfile::Profile4, 1000, 400, 4000.0);
        assert!(!spec.is_hdr10_compatible());
    }

    #[test]
    fn test_hdr10_not_compatible_low_luminance() {
        let spec = DvDeliverySpec::new(DvProfile::Profile5, 1000, 400, 500.0);
        assert!(!spec.is_hdr10_compatible());
    }

    #[test]
    fn test_hdr10_not_compatible_zero_cll() {
        let spec = DvDeliverySpec::new(DvProfile::Profile5, 0, 400, 4000.0);
        assert!(!spec.is_hdr10_compatible());
    }

    // --- DeliveryValidator ---

    #[test]
    fn test_validator_ok() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 1000, 400, 4000.0);
        assert!(DeliveryValidator::validate(&spec).is_empty());
    }

    #[test]
    fn test_validator_zero_cll() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 0, 400, 4000.0);
        let errs = DeliveryValidator::validate(&spec);
        assert!(errs.iter().any(|e| e.contains("MaxCLL")));
    }

    #[test]
    fn test_validator_zero_fall() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 1000, 0, 4000.0);
        let errs = DeliveryValidator::validate(&spec);
        assert!(errs.iter().any(|e| e.contains("MaxFALL")));
    }

    #[test]
    fn test_validator_luminance_too_low() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 1000, 400, 50.0);
        let errs = DeliveryValidator::validate(&spec);
        assert!(errs.iter().any(|e| e.contains("below minimum")));
    }

    #[test]
    fn test_validator_luminance_too_high() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 1000, 400, 20_000.0);
        let errs = DeliveryValidator::validate(&spec);
        assert!(errs.iter().any(|e| e.contains("exceeds maximum")));
    }

    #[test]
    fn test_validator_multiple_errors() {
        let spec = DvDeliverySpec::new(DvProfile::Profile8_1, 0, 0, 50.0);
        let errs = DeliveryValidator::validate(&spec);
        assert!(errs.len() >= 3);
    }
}
