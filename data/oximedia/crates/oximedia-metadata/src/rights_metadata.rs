//! Rights and copyright metadata.
//!
//! This module provides types for representing copyright status,
//! Creative Commons licensing, and associated rights information.

#![allow(dead_code)]

/// The copyright status of a media asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyrightStatus {
    /// Asset is owned by a named copyright holder.
    CopyrightHolder,
    /// Asset is in the public domain.
    PublicDomain,
    /// Asset is licensed under a Creative Commons license.
    CreativeCommons,
    /// All rights are reserved by the copyright holder.
    AllRightsReserved,
    /// Copyright status is unknown.
    Unknown,
}

impl CopyrightStatus {
    /// Returns true if this status permits reuse without explicit permission.
    pub fn allows_reuse(&self) -> bool {
        matches!(self, Self::PublicDomain | Self::CreativeCommons)
    }
}

/// Rights and copyright metadata for a media asset.
#[derive(Debug, Clone)]
pub struct RightsMetadata {
    /// Name of the copyright holder.
    pub copyright_holder: String,
    /// Year of copyright (if known).
    pub year: Option<u16>,
    /// Copyright status.
    pub status: CopyrightStatus,
    /// URL pointing to the license text (if any).
    pub license_url: Option<String>,
    /// Human-readable usage terms.
    pub usage_terms: String,
}

impl RightsMetadata {
    /// Create new rights metadata.
    pub fn new(
        copyright_holder: String,
        year: Option<u16>,
        status: CopyrightStatus,
        license_url: Option<String>,
        usage_terms: String,
    ) -> Self {
        Self {
            copyright_holder,
            year,
            status,
            license_url,
            usage_terms,
        }
    }

    /// Returns true if the asset is in the public domain.
    pub fn is_public_domain(&self) -> bool {
        self.status == CopyrightStatus::PublicDomain
    }

    /// Returns true if a license URL is present.
    pub fn has_license(&self) -> bool {
        self.license_url.is_some()
    }
}

/// A Creative Commons license variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreativeCommonsLicense {
    /// Attribution (CC BY)
    By,
    /// Attribution-ShareAlike (CC BY-SA)
    BySa,
    /// Attribution-NonCommercial (CC BY-NC)
    ByNc,
    /// Attribution-NoDerivatives (CC BY-ND)
    ByNd,
    /// Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
    ByNcSa,
    /// Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND)
    ByNcNd,
}

impl CreativeCommonsLicense {
    /// Returns true if this license permits commercial use.
    pub fn allows_commercial(&self) -> bool {
        matches!(self, Self::By | Self::BySa | Self::ByNd)
    }

    /// Returns true if this license permits creation of derivative works.
    pub fn allows_derivatives(&self) -> bool {
        matches!(self, Self::By | Self::BySa | Self::ByNc | Self::ByNcSa)
    }

    /// Returns the SPDX identifier for this license.
    pub fn spdx_id(&self) -> &str {
        match self {
            Self::By => "CC-BY-4.0",
            Self::BySa => "CC-BY-SA-4.0",
            Self::ByNc => "CC-BY-NC-4.0",
            Self::ByNd => "CC-BY-ND-4.0",
            Self::ByNcSa => "CC-BY-NC-SA-4.0",
            Self::ByNcNd => "CC-BY-NC-ND-4.0",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copyright_status_allows_reuse_public_domain() {
        assert!(CopyrightStatus::PublicDomain.allows_reuse());
    }

    #[test]
    fn test_copyright_status_allows_reuse_creative_commons() {
        assert!(CopyrightStatus::CreativeCommons.allows_reuse());
    }

    #[test]
    fn test_copyright_status_does_not_allow_reuse() {
        assert!(!CopyrightStatus::CopyrightHolder.allows_reuse());
        assert!(!CopyrightStatus::AllRightsReserved.allows_reuse());
        assert!(!CopyrightStatus::Unknown.allows_reuse());
    }

    #[test]
    fn test_rights_metadata_is_public_domain() {
        let rights = RightsMetadata::new(
            String::new(),
            None,
            CopyrightStatus::PublicDomain,
            None,
            "No restrictions".to_string(),
        );
        assert!(rights.is_public_domain());
    }

    #[test]
    fn test_rights_metadata_is_not_public_domain() {
        let rights = RightsMetadata::new(
            "Acme Corp".to_string(),
            Some(2024),
            CopyrightStatus::AllRightsReserved,
            None,
            "All rights reserved".to_string(),
        );
        assert!(!rights.is_public_domain());
    }

    #[test]
    fn test_rights_metadata_has_license() {
        let rights = RightsMetadata::new(
            "Author".to_string(),
            Some(2023),
            CopyrightStatus::CreativeCommons,
            Some("https://creativecommons.org/licenses/by/4.0/".to_string()),
            "CC BY 4.0".to_string(),
        );
        assert!(rights.has_license());
    }

    #[test]
    fn test_rights_metadata_no_license() {
        let rights = RightsMetadata::new(
            "Author".to_string(),
            None,
            CopyrightStatus::Unknown,
            None,
            String::new(),
        );
        assert!(!rights.has_license());
    }

    #[test]
    fn test_rights_metadata_year_field() {
        let rights = RightsMetadata::new(
            "Corp".to_string(),
            Some(1999),
            CopyrightStatus::AllRightsReserved,
            None,
            "All rights reserved".to_string(),
        );
        assert_eq!(rights.year, Some(1999));
    }

    #[test]
    fn test_cc_license_allows_commercial() {
        assert!(CreativeCommonsLicense::By.allows_commercial());
        assert!(CreativeCommonsLicense::BySa.allows_commercial());
        assert!(CreativeCommonsLicense::ByNd.allows_commercial());
        assert!(!CreativeCommonsLicense::ByNc.allows_commercial());
        assert!(!CreativeCommonsLicense::ByNcSa.allows_commercial());
        assert!(!CreativeCommonsLicense::ByNcNd.allows_commercial());
    }

    #[test]
    fn test_cc_license_allows_derivatives() {
        assert!(CreativeCommonsLicense::By.allows_derivatives());
        assert!(CreativeCommonsLicense::BySa.allows_derivatives());
        assert!(CreativeCommonsLicense::ByNc.allows_derivatives());
        assert!(CreativeCommonsLicense::ByNcSa.allows_derivatives());
        assert!(!CreativeCommonsLicense::ByNd.allows_derivatives());
        assert!(!CreativeCommonsLicense::ByNcNd.allows_derivatives());
    }

    #[test]
    fn test_cc_license_spdx_id() {
        assert_eq!(CreativeCommonsLicense::By.spdx_id(), "CC-BY-4.0");
        assert_eq!(CreativeCommonsLicense::BySa.spdx_id(), "CC-BY-SA-4.0");
        assert_eq!(CreativeCommonsLicense::ByNc.spdx_id(), "CC-BY-NC-4.0");
        assert_eq!(CreativeCommonsLicense::ByNd.spdx_id(), "CC-BY-ND-4.0");
        assert_eq!(CreativeCommonsLicense::ByNcSa.spdx_id(), "CC-BY-NC-SA-4.0");
        assert_eq!(CreativeCommonsLicense::ByNcNd.spdx_id(), "CC-BY-NC-ND-4.0");
    }

    #[test]
    fn test_rights_metadata_holder_field() {
        let rights = RightsMetadata::new(
            "Jane Doe Productions".to_string(),
            Some(2025),
            CopyrightStatus::CopyrightHolder,
            None,
            "Contact for licensing".to_string(),
        );
        assert_eq!(rights.copyright_holder, "Jane Doe Productions");
    }
}
