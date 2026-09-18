//! Supplemental-package chaining tests.
//!
//! Supplemental IMF packages reference one or more base packages by PKL UUID.
//! There is no auto-resolver, so these tests assert directly on the
//! `base_packages` reference vector to confirm multi-level chains
//! (L0 ← L1 ← L2) and fan-in (one supplemental over two bases) are recorded
//! exactly as declared, each with a structurally valid UUID.

use oximedia_imf::supplemental_package::{
    BasePackageRef, ImfSupplementalPackage, SupplementalPackageType,
};

/// A two-link chain: L1 extends an original base (L0), and L2 extends L1.
/// Each level records exactly its declared base reference.
#[test]
fn multi_level_chain_resolves() {
    let mut l1 = ImfSupplementalPackage::new(
        "urn:uuid:L1",
        SupplementalPackageType::Localization,
        "FR audio",
    );
    l1.add_base_package(BasePackageRef::new("urn:uuid:L0-base-0001", "Original"));

    let mut l2 = ImfSupplementalPackage::new(
        "urn:uuid:L2",
        SupplementalPackageType::Accessibility,
        "FR audio-desc",
    );
    l2.add_base_package(BasePackageRef::new("urn:uuid:L1", "FR"));

    assert_eq!(l1.base_package_count(), 1, "L1 references one base (L0)");
    assert_eq!(l2.base_package_count(), 1, "L2 references one base (L1)");

    assert_eq!(
        l1.base_packages[0].pkl_uuid, "urn:uuid:L0-base-0001",
        "L1's base is the original package"
    );
    assert_eq!(
        l2.base_packages[0].pkl_uuid, "urn:uuid:L1",
        "L2's base is L1, forming the chain"
    );

    assert!(
        l1.base_packages[0].is_valid_uuid(),
        "L1's base reference has a structurally valid UUID"
    );
    assert!(
        l2.base_packages[0].is_valid_uuid(),
        "L2's base reference has a structurally valid UUID"
    );
}

/// A single supplemental package may fan in over two distinct base packages.
#[test]
fn two_bases_fan_in() {
    let mut supp = ImfSupplementalPackage::new(
        "urn:uuid:fan-in",
        SupplementalPackageType::EditDecision,
        "Conform over two reels",
    );
    supp.add_base_package(BasePackageRef::new("urn:uuid:base-reel-0001", "ReelA"));
    supp.add_base_package(BasePackageRef::new("urn:uuid:base-reel-0002", "ReelB"));

    assert_eq!(
        supp.base_package_count(),
        2,
        "the supplemental fans in over both base packages"
    );
}
