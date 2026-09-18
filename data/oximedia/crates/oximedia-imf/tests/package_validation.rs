//! Hash-integrity and package-validation report tests.
//!
//! Covers [`oximedia_imf::package_validator::verify_hashes_parallel`] (parallel
//! SHA verification of in-memory essence bodies) and the
//! [`oximedia_imf::package_validator::PackageValidator`] builder, which records
//! issues by severity and reports which are blocking.

use oximedia_imf::essence_hash::{compute_hash_hex, HashAlgo};
use oximedia_imf::package_validator::{
    verify_hashes_parallel, HashableAsset, PackageValidator, ValidationSeverity,
};

/// A correct stored hash over the exact bytes verifies as a match.
#[test]
fn valid_asset_hash_passes() {
    let data = b"IMF MXF essence body";
    let h = compute_hash_hex(data, HashAlgo::Sha256);
    let res = verify_hashes_parallel(&[HashableAsset::new(
        "video-001",
        data.to_vec(),
        h,
        HashAlgo::Sha256,
    )]);

    assert_eq!(res.len(), 1, "one result per asset");
    assert_eq!(
        res[0].as_ref().copied().unwrap_or(false),
        true,
        "matching hash must verify true"
    );
}

/// Corrupting one byte of the essence (while the stored hash stays correct for
/// the original) must be detected as a mismatch.
#[test]
fn corrupted_essence_flags_mismatch() {
    let data = b"IMF MXF essence body";
    let h = compute_hash_hex(data, HashAlgo::Sha256);

    let mut bad = data.to_vec();
    bad[0] ^= 0xFF;

    let res = verify_hashes_parallel(&[HashableAsset::new("video-001", bad, h, HashAlgo::Sha256)]);

    assert_eq!(res.len(), 1);
    assert_eq!(
        res[0],
        Ok(false),
        "a single flipped byte must fail verification"
    );
}

/// Correct data but a corrupted (all-zero) stored hash must fail.
#[test]
fn corrupted_stored_hash_flags_mismatch() {
    let data = b"IMF MXF essence body";
    let res = verify_hashes_parallel(&[HashableAsset::new(
        "video-001",
        data.to_vec(),
        "0".repeat(64),
        HashAlgo::Sha256,
    )]);

    assert_eq!(res.len(), 1);
    assert_eq!(
        res[0],
        Ok(false),
        "a garbage stored hash must fail verification"
    );
}

/// A failed equality check recorded via `check` becomes a blocking Error in the
/// finished report, retrievable through `blocking_issues`.
#[test]
fn report_records_blocking_error() {
    let actual = compute_hash_hex(b"x", HashAlgo::Sha256);
    let stored = "deadbeef".to_string();

    let mut v = PackageValidator::new();
    v.check(
        actual == stored,
        "PKL_HASH_MISMATCH",
        format!("expected {stored}, got {actual}"),
    );
    let r = v.finish();

    assert!(!r.is_ok(), "a recorded mismatch makes the report not-ok");
    assert_eq!(
        r.count_severity(ValidationSeverity::Error),
        1,
        "exactly one Error-severity issue"
    );
    assert_eq!(
        r.blocking_issues()[0].code,
        "PKL_HASH_MISMATCH",
        "the blocking issue carries the supplied code"
    );
}

/// A validator with no recorded issues yields a clean, empty report.
#[test]
fn clean_report_is_ok() {
    let r = PackageValidator::new().finish();
    assert!(r.is_ok());
    assert_eq!(r.total(), 0);
}
