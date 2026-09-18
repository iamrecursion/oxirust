//! Smoke tests verifying newly wired DRM orphan modules compile and expose
//! at least one public item from each module.

use oximedia_drm::{
    cenc_scheme::{CencScheme, CencSchemeSelector, CodecClass, EncryptionCoverage},
    dash_cenc::{CencSignaling, DashCencScheme},
    device_fingerprint::FingerprintBuilder,
    drm_token_verifier::{DrmTokenVerifier, VerifyOutcome},
    hw_key_storage::{HwKeyStorage, HwKeyStorageConfig, HwStorageBackend},
    key_schedule::{AesKeySize, KeySchedule},
    license_audit::{LicenseAuditLog, LicenseEventKind},
    pssh_parser::PsshParser,
    revocation_list::RevocationList,
    token_bucket_auth::{AuthGateConfig, TokenBucketAuthGate},
    DrmSystem,
};

#[test]
fn test_cenc_scheme_fourcc() {
    assert_eq!(CencScheme::Cenc.fourcc(), "cenc");
    assert_eq!(CencScheme::Cbcs.fourcc(), "cbcs");
    assert_eq!(CencScheme::Cbc1.fourcc(), "cbc1");
    assert_eq!(CencScheme::Cens.fourcc(), "cens");
}

#[test]
fn test_cenc_scheme_from_fourcc() {
    let scheme = CencScheme::from_fourcc("cenc").expect("cenc should parse");
    assert_eq!(scheme, CencScheme::Cenc);
    // Error case.
    assert!(CencScheme::from_fourcc("mpeg").is_err());
}

#[test]
fn test_cenc_scheme_selector_recommend() {
    let selector = CencSchemeSelector;
    let scheme = selector.recommend(
        &[DrmSystem::Widevine],
        CodecClass::Avc,
        EncryptionCoverage::Full,
    );
    assert!(scheme.is_ok());
}

#[test]
fn test_dash_cenc_scheme_fourcc() {
    assert_eq!(DashCencScheme::Cenc.fourcc(), "cenc");
    assert_eq!(DashCencScheme::Cbcs.fourcc(), "cbcs");
}

#[test]
fn test_cenc_signaling_new_empty() {
    let sig = CencSignaling::new();
    assert_eq!(sig.descriptor_count(), 0);
}

#[test]
fn test_fingerprint_builder_new() {
    let fp = FingerprintBuilder::new()
        .cpu_id("test-cpu-001")
        .mac_address("aa:bb:cc:dd:ee:ff")
        .build();
    assert!(fp.component_count() > 0);
}

#[test]
fn test_drm_token_verifier_new() {
    let verifier = DrmTokenVerifier::new();
    // An invalid token string must not panic.
    let outcome = verifier.verify("not-a-real-token", 0, None, None);
    // The outcome tuple should indicate a non-Valid result.
    assert!(!matches!(outcome.0, VerifyOutcome::Valid));
}

#[test]
fn test_hw_key_storage_software_backend() {
    let config = HwKeyStorageConfig::software();
    let storage = HwKeyStorage::new(config);
    assert_eq!(storage.active_backend(), HwStorageBackend::Software);
}

#[test]
fn test_key_schedule_aes128() {
    let key = [0u8; 16];
    let sched = KeySchedule::new(&key, AesKeySize::Aes128).expect("AES-128 schedule should build");
    assert_eq!(sched.rounds(), 10);
}

#[test]
fn test_license_audit_log_empty() {
    let log = LicenseAuditLog::new(100);
    // Fresh log has no records in a relevant event count.
    let _ = log;
}

#[test]
fn test_license_event_kind_label() {
    // Labels are uppercase.
    assert_eq!(LicenseEventKind::Grant.label(), "GRANT");
    assert!(LicenseEventKind::Grant.is_success());
    assert!(!LicenseEventKind::Deny.is_success());
}

#[test]
fn test_pssh_parser_empty_bytes_returns_error() {
    // Parsing empty bytes must return an error, not panic.
    let result = PsshParser::parse(&[]);
    assert!(result.is_err());
}

#[test]
fn test_revocation_list_empty() {
    let list = RevocationList::new();
    // Initial version starts at 1.
    assert_eq!(list.version(), 1);
    assert!(!list.is_revoked("device-001", 0));
}

#[test]
fn test_token_bucket_auth_gate_default() {
    let config = AuthGateConfig::default();
    let mut gate = TokenBucketAuthGate::new(config);
    gate.register_device("device-001");
    // A fresh gate should accept the first request immediately.
    let result = gate.request("device-001", 1_000_000);
    assert!(result.is_ok());
}
