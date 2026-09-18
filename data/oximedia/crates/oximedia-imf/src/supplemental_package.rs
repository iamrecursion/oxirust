//! IMF Supplemental Package (S-IMF) support
//!
//! Supplemental IMF packages (S-IMFs) contain only the assets that differ from
//! one or more base packages, enabling efficient versioning and localisation.
//! This module provides a high-level, self-contained API for working with
//! supplemental packages.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// The nature / purpose of a supplemental IMF package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupplementalPackageType {
    /// New-language audio, subtitles, or dubbed content.
    Localization,
    /// Accessibility variants (e.g. audio description, open captions).
    Accessibility,
    /// Editorial decisions applied on top of a base package.
    EditDecision,
    /// Visual-effects replacement layers.
    Vfx,
}

impl SupplementalPackageType {
    /// Returns `true` for every variant, because all supplemental packages
    /// require at least one base package to be meaningful.
    pub fn requires_base(&self) -> bool {
        true
    }
}

/// A reference to a base package (identified by its PKL UUID).
#[derive(Debug, Clone)]
pub struct BasePackageRef {
    /// UUID of the base package's PKL (as a plain or `urn:uuid:` string).
    pub pkl_uuid: String,
    /// A human-readable package-type label (e.g. `"Original"`, `"Base EN"`).
    pub package_type: String,
}

impl BasePackageRef {
    /// Create a new `BasePackageRef`.
    pub fn new(pkl_uuid: impl Into<String>, package_type: impl Into<String>) -> Self {
        Self {
            pkl_uuid: pkl_uuid.into(),
            package_type: package_type.into(),
        }
    }

    /// Returns `true` when `pkl_uuid` is non-empty and at least 8 characters
    /// long (a minimal sanity check — not full UUID validation).
    pub fn is_valid_uuid(&self) -> bool {
        !self.pkl_uuid.is_empty() && self.pkl_uuid.len() >= 8
    }
}

/// A supplemental IMF package descriptor.
#[derive(Debug, Clone)]
pub struct ImfSupplementalPackage {
    /// UUID of this supplemental package.
    pub id: String,
    /// The kind of supplemental content this package provides.
    pub package_type: SupplementalPackageType,
    /// The base packages this package extends.
    pub base_packages: Vec<BasePackageRef>,
    /// Free-text description.
    pub description: String,
}

impl ImfSupplementalPackage {
    /// Create a new `ImfSupplementalPackage`.
    pub fn new(
        id: impl Into<String>,
        package_type: SupplementalPackageType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            package_type,
            base_packages: Vec::new(),
            description: description.into(),
        }
    }

    /// Returns `true` when the package type is `Localization`.
    pub fn is_localization(&self) -> bool {
        self.package_type == SupplementalPackageType::Localization
    }

    /// Returns the number of base packages this package references.
    pub fn base_package_count(&self) -> usize {
        self.base_packages.len()
    }

    /// Add a base package reference.
    pub fn add_base_package(&mut self, base: BasePackageRef) {
        self.base_packages.push(base);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkg(pkg_type: SupplementalPackageType) -> ImfSupplementalPackage {
        ImfSupplementalPackage::new("test-uuid-1234", pkg_type, "Test supplemental package")
    }

    // --- SupplementalPackageType ---

    #[test]
    fn test_requires_base_localization() {
        assert!(SupplementalPackageType::Localization.requires_base());
    }

    #[test]
    fn test_requires_base_accessibility() {
        assert!(SupplementalPackageType::Accessibility.requires_base());
    }

    #[test]
    fn test_requires_base_edit_decision() {
        assert!(SupplementalPackageType::EditDecision.requires_base());
    }

    #[test]
    fn test_requires_base_vfx() {
        assert!(SupplementalPackageType::Vfx.requires_base());
    }

    // --- BasePackageRef ---

    #[test]
    fn test_base_package_ref_is_valid_uuid_ok() {
        let r = BasePackageRef::new("urn:uuid:550e8400-e29b-41d4-a716-446655440000", "Base");
        assert!(r.is_valid_uuid());
    }

    #[test]
    fn test_base_package_ref_is_valid_uuid_empty() {
        let r = BasePackageRef::new("", "Base");
        assert!(!r.is_valid_uuid());
    }

    #[test]
    fn test_base_package_ref_is_valid_uuid_short() {
        let r = BasePackageRef::new("abc", "Base");
        assert!(!r.is_valid_uuid());
    }

    #[test]
    fn test_base_package_ref_package_type_stored() {
        let r = BasePackageRef::new("some-long-uuid-12", "OriginalEN");
        assert_eq!(r.package_type, "OriginalEN");
    }

    // --- ImfSupplementalPackage ---

    #[test]
    fn test_is_localization_true() {
        let pkg = make_pkg(SupplementalPackageType::Localization);
        assert!(pkg.is_localization());
    }

    #[test]
    fn test_is_localization_false_for_vfx() {
        let pkg = make_pkg(SupplementalPackageType::Vfx);
        assert!(!pkg.is_localization());
    }

    #[test]
    fn test_is_localization_false_for_accessibility() {
        let pkg = make_pkg(SupplementalPackageType::Accessibility);
        assert!(!pkg.is_localization());
    }

    #[test]
    fn test_base_package_count_empty() {
        let pkg = make_pkg(SupplementalPackageType::Localization);
        assert_eq!(pkg.base_package_count(), 0);
    }

    #[test]
    fn test_base_package_count_after_add() {
        let mut pkg = make_pkg(SupplementalPackageType::Localization);
        pkg.add_base_package(BasePackageRef::new("uuid-base-001234567", "Original"));
        pkg.add_base_package(BasePackageRef::new("uuid-base-009876543", "Dub"));
        assert_eq!(pkg.base_package_count(), 2);
    }

    #[test]
    fn test_description_stored() {
        let pkg = ImfSupplementalPackage::new(
            "id-abc",
            SupplementalPackageType::EditDecision,
            "Director's Cut edits",
        );
        assert_eq!(pkg.description, "Director's Cut edits");
    }

    #[test]
    fn test_id_stored() {
        let pkg =
            ImfSupplementalPackage::new("pkg-id-xyz", SupplementalPackageType::Vfx, "VFX layer");
        assert_eq!(pkg.id, "pkg-id-xyz");
    }

    #[test]
    fn test_vfx_not_localization() {
        let pkg = make_pkg(SupplementalPackageType::Vfx);
        assert!(!pkg.is_localization());
        assert!(pkg.package_type.requires_base());
    }
}
