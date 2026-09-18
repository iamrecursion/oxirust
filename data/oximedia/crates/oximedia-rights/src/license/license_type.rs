//! License types

use serde::{Deserialize, Serialize};

/// Type of content license
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LicenseType {
    /// Royalty-free: one-time payment, unlimited use
    RoyaltyFree,
    /// Rights-managed: usage-based pricing
    RightsManaged,
    /// Exclusive: sole user rights
    Exclusive,
    /// Non-exclusive: shared rights
    NonExclusive,
    /// Creative Commons Attribution (CC BY)
    CreativeCommonsBy,
    /// Creative Commons Attribution-ShareAlike (CC BY-SA)
    CreativeCommonsBySa,
    /// Creative Commons Attribution-NoDerivs (CC BY-ND)
    CreativeCommonsByNd,
    /// Creative Commons Attribution-NonCommercial (CC BY-NC)
    CreativeCommonsByNc,
    /// Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
    CreativeCommonsByNcSa,
    /// Creative Commons Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)
    CreativeCommonsByNcNd,
    /// Creative Commons Zero (CC0) / Public Domain
    CreativeCommonsZero,
    /// Public domain
    PublicDomain,
    /// Custom license type
    Custom(String),
}

impl LicenseType {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            LicenseType::RoyaltyFree => "royalty_free",
            LicenseType::RightsManaged => "rights_managed",
            LicenseType::Exclusive => "exclusive",
            LicenseType::NonExclusive => "non_exclusive",
            LicenseType::CreativeCommonsBy => "cc_by",
            LicenseType::CreativeCommonsBySa => "cc_by_sa",
            LicenseType::CreativeCommonsByNd => "cc_by_nd",
            LicenseType::CreativeCommonsByNc => "cc_by_nc",
            LicenseType::CreativeCommonsByNcSa => "cc_by_nc_sa",
            LicenseType::CreativeCommonsByNcNd => "cc_by_nc_nd",
            LicenseType::CreativeCommonsZero => "cc0",
            LicenseType::PublicDomain => "public_domain",
            LicenseType::Custom(s) => s,
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "royalty_free" => LicenseType::RoyaltyFree,
            "rights_managed" => LicenseType::RightsManaged,
            "exclusive" => LicenseType::Exclusive,
            "non_exclusive" => LicenseType::NonExclusive,
            "cc_by" => LicenseType::CreativeCommonsBy,
            "cc_by_sa" => LicenseType::CreativeCommonsBySa,
            "cc_by_nd" => LicenseType::CreativeCommonsByNd,
            "cc_by_nc" => LicenseType::CreativeCommonsByNc,
            "cc_by_nc_sa" => LicenseType::CreativeCommonsByNcSa,
            "cc_by_nc_nd" => LicenseType::CreativeCommonsByNcNd,
            "cc0" => LicenseType::CreativeCommonsZero,
            "public_domain" => LicenseType::PublicDomain,
            other => LicenseType::Custom(other.to_string()),
        }
    }

    /// Check if license requires attribution
    pub fn requires_attribution(&self) -> bool {
        matches!(
            self,
            LicenseType::CreativeCommonsBy
                | LicenseType::CreativeCommonsBySa
                | LicenseType::CreativeCommonsByNd
                | LicenseType::CreativeCommonsByNc
                | LicenseType::CreativeCommonsByNcSa
                | LicenseType::CreativeCommonsByNcNd
        )
    }

    /// Check if license allows commercial use
    pub fn allows_commercial_use(&self) -> bool {
        !matches!(
            self,
            LicenseType::CreativeCommonsByNc
                | LicenseType::CreativeCommonsByNcSa
                | LicenseType::CreativeCommonsByNcNd
        )
    }

    /// Check if license allows modifications
    pub fn allows_modifications(&self) -> bool {
        !matches!(
            self,
            LicenseType::CreativeCommonsByNd | LicenseType::CreativeCommonsByNcNd
        )
    }

    /// Check if license is exclusive
    pub fn is_exclusive(&self) -> bool {
        matches!(self, LicenseType::Exclusive)
    }

    /// Get human-readable description
    pub fn description(&self) -> &str {
        match self {
            LicenseType::RoyaltyFree => "Royalty-free license with one-time payment",
            LicenseType::RightsManaged => "Rights-managed license with usage-based pricing",
            LicenseType::Exclusive => "Exclusive rights for sole use",
            LicenseType::NonExclusive => "Non-exclusive rights allowing shared use",
            LicenseType::CreativeCommonsBy => {
                "Creative Commons Attribution: Free use with attribution"
            }
            LicenseType::CreativeCommonsBySa => {
                "Creative Commons Attribution-ShareAlike: Free use with attribution and share-alike"
            }
            LicenseType::CreativeCommonsByNd => {
                "Creative Commons Attribution-NoDerivs: Free use with attribution, no modifications"
            }
            LicenseType::CreativeCommonsByNc => {
                "Creative Commons Attribution-NonCommercial: Free non-commercial use with attribution"
            }
            LicenseType::CreativeCommonsByNcSa => {
                "Creative Commons Attribution-NonCommercial-ShareAlike: Free non-commercial use with attribution and share-alike"
            }
            LicenseType::CreativeCommonsByNcNd => {
                "Creative Commons Attribution-NonCommercial-NoDerivs: Free non-commercial use with attribution, no modifications"
            }
            LicenseType::CreativeCommonsZero => {
                "Creative Commons Zero: Public domain dedication"
            }
            LicenseType::PublicDomain => "Public domain, no restrictions",
            LicenseType::Custom(_) => "Custom license terms",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_type_conversion() {
        assert_eq!(LicenseType::RoyaltyFree.as_str(), "royalty_free");
        assert_eq!(
            LicenseType::from_str("royalty_free"),
            LicenseType::RoyaltyFree
        );
    }

    #[test]
    fn test_attribution_requirement() {
        assert!(LicenseType::CreativeCommonsBy.requires_attribution());
        assert!(!LicenseType::PublicDomain.requires_attribution());
    }

    #[test]
    fn test_commercial_use() {
        assert!(LicenseType::RoyaltyFree.allows_commercial_use());
        assert!(!LicenseType::CreativeCommonsByNc.allows_commercial_use());
    }

    #[test]
    fn test_modifications() {
        assert!(LicenseType::CreativeCommonsBy.allows_modifications());
        assert!(!LicenseType::CreativeCommonsByNd.allows_modifications());
    }

    #[test]
    fn test_exclusivity() {
        assert!(LicenseType::Exclusive.is_exclusive());
        assert!(!LicenseType::NonExclusive.is_exclusive());
    }
}
