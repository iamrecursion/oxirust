//! Archive packaging formats for preservation
//!
//! Supports multiple archival package formats:
//! - `BagIt`: Standard digital preservation package format
//! - OAIS: Open Archival Information System (SIP/AIP/DIP)
//! - TAR: Tape Archive with checksums
//! - ZIP: ZIP archive with checksums

pub mod bagit;
pub mod oais;
pub mod tar;
pub mod zip;

pub use bagit::{
    verify_bagit_v1_compliance, BagItBuilder, BagItPackage, BagItValidator, BagitComplianceReport,
    ComplianceFinding, ComplianceLevel,
};
pub use oais::{DipGenerationConfig, DipGenerator, OaisBuilder, OaisPackage, OaisPackageType};
pub use tar::TarArchiver;
pub use zip::ZipArchiver;

use serde::{Deserialize, Serialize};

/// Package format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageFormat {
    /// `BagIt` format
    BagIt,
    /// OAIS Submission Information Package
    OaisSip,
    /// OAIS Archival Information Package
    OaisAip,
    /// OAIS Dissemination Information Package
    OaisDip,
    /// TAR archive
    Tar,
    /// ZIP archive
    Zip,
}

impl PackageFormat {
    /// Returns the format name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BagIt => "BagIt",
            Self::OaisSip => "OAIS-SIP",
            Self::OaisAip => "OAIS-AIP",
            Self::OaisDip => "OAIS-DIP",
            Self::Tar => "TAR",
            Self::Zip => "ZIP",
        }
    }

    /// Returns a description of the format
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::BagIt => "BagIt archival package with manifest",
            Self::OaisSip => "OAIS Submission Information Package",
            Self::OaisAip => "OAIS Archival Information Package (long-term)",
            Self::OaisDip => "OAIS Dissemination Information Package (access)",
            Self::Tar => "TAR archive with checksums",
            Self::Zip => "ZIP archive with checksums",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_format_names() {
        assert_eq!(PackageFormat::BagIt.name(), "BagIt");
        assert_eq!(PackageFormat::OaisAip.name(), "OAIS-AIP");
        assert_eq!(PackageFormat::Tar.name(), "TAR");
    }
}
