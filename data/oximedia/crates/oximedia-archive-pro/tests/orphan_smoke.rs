//! Smoke tests verifying newly wired archive-pro orphan modules compile and expose
//! at least one public item from each module.

use oximedia_archive_pro::{
    archive_extensions::{BagItPackager, PremisRecord},
    collection_manifest::{CollectionManifest, EntryStatus},
    cost_estimator::{CollectionDescriptor, CostEstimatorConfig, PreservationCostEstimator},
    iiif_manifest::IiifManifestBuilder,
    integrity_dashboard::{IntegrityDashboard, ObsolescenceRisk},
    lockss_package::{LockssAuMetadata, LockssNetwork, LockssPackageBuilder},
    preservation_level::{PreservationLevel, PreservationPolicy},
    secure_delete::{build_passes, verify_absent, DeletionStandard},
    sip_builder::{SipBuilder, TransferAgreementStatus},
};
use std::path::Path;

#[test]
fn test_premis_record_new() {
    let record = PremisRecord::new("test-object-001", "2026-05-30");
    assert_eq!(record.object_id, "test-object-001");
}

#[test]
fn test_bagit_packager_exists() {
    let _packager = BagItPackager;
}

#[test]
fn test_collection_manifest_new() {
    let manifest = CollectionManifest::new("test-collection", "Test Collection");
    assert_eq!(manifest.entry_count(), 0);
}

#[test]
fn test_collection_manifest_entry_status() {
    let status = EntryStatus::Ingested;
    assert_eq!(status, EntryStatus::Ingested);
}

#[test]
fn test_cost_estimator_config_new() {
    let config = CostEstimatorConfig::default();
    let estimator = PreservationCostEstimator::new(config);
    let _ = estimator;
}

#[test]
fn test_collection_descriptor_new() {
    let desc = CollectionDescriptor::new("test-col", 100.0, 1000);
    let _ = desc;
}

#[test]
fn test_iiif_manifest_builder_empty() {
    let manifest =
        IiifManifestBuilder::new("https://example.org/manifest/1", "Test Manifest").build();
    assert_eq!(manifest.items.len(), 0);
}

#[test]
fn test_iiif_manifest_id_field() {
    let manifest = IiifManifestBuilder::new("https://example.org/manifest/2", "Another").build();
    assert!(!manifest.id.is_empty());
}

#[test]
fn test_integrity_dashboard_new() {
    let dashboard = IntegrityDashboard::new();
    let report = dashboard.generate_report();
    // Fresh dashboard has no critical objects.
    assert!(report.critical_objects.is_empty());
}

#[test]
fn test_obsolescence_risk_ordering() {
    assert!(ObsolescenceRisk::Low < ObsolescenceRisk::High);
}

#[test]
fn test_lockss_package_builder_new() {
    let meta = LockssAuMetadata::new(
        "test-au-001",
        "Test AU",
        "COOLJAPAN",
        "https://example.org/",
    );
    let tmp = std::env::temp_dir().join("oximedia_lockss_smoke_test");
    let builder = LockssPackageBuilder::new(&tmp, meta);
    // Builder created without errors.
    let _ = builder;
}

#[test]
fn test_lockss_network_variants() {
    let net = LockssNetwork::Lockss;
    assert_eq!(net, LockssNetwork::Lockss);
    let net2 = LockssNetwork::Clockss;
    assert_eq!(net2, LockssNetwork::Clockss);
}

#[test]
fn test_preservation_level_ordering() {
    assert!(PreservationLevel::BitLevel < PreservationLevel::Full);
}

#[test]
fn test_preservation_policy_new() {
    let policy = PreservationPolicy::new(PreservationLevel::Logical);
    // default_level returns the default for unregistered content types.
    // required_level returns the registered level (or default) for a content type.
    let level = policy.required_level("video/x-matroska");
    assert_eq!(level, PreservationLevel::Logical);
}

#[test]
fn test_secure_delete_build_passes_dod() {
    let passes = build_passes(DeletionStandard::Dod522022M);
    // DoD 5220.22-M requires exactly 3 passes.
    assert_eq!(passes.len(), 3);
}

#[test]
fn test_verify_absent_nonexistent_path() {
    let path = Path::new("/nonexistent/path/abc123xyz_oximedia");
    assert!(verify_absent(path));
}

#[test]
fn test_sip_builder_validation_requires_title() {
    let builder = SipBuilder::new("SIP-SMOKE-001");
    let issues = builder.validate();
    // No title/producer set → validation should report issues.
    assert!(
        !issues.is_empty(),
        "SIP without title/producer should fail validation"
    );
}

#[test]
fn test_transfer_agreement_status_approved() {
    let status = TransferAgreementStatus::Approved;
    assert_eq!(status, TransferAgreementStatus::Approved);
}
