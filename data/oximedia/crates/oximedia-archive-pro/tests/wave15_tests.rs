//! Wave 15 tests: concurrent multi-algorithm checksum verification

use oximedia_archive_pro::checksum::{
    ChecksumAlgorithm, ChecksumGenerator, ChecksumVerifier, FileChecksum,
};
use std::io::Write;

/// Build a `FileChecksum` for a temp file written with `content`, using the given algorithms.
fn make_expected(
    content: &[u8],
    algorithms: Vec<ChecksumAlgorithm>,
) -> (tempfile::NamedTempFile, FileChecksum) {
    let mut tmp = tempfile::NamedTempFile::new().expect("temp file creation should succeed");
    tmp.write_all(content)
        .expect("write to temp file should succeed");
    tmp.flush().expect("flush should succeed");

    let generator = ChecksumGenerator::new().with_algorithms(algorithms);
    let checksum = generator
        .generate_file(tmp.path())
        .expect("checksum generation should succeed");

    (tmp, checksum)
}

/// Wave 15 / Slice I — test 1
///
/// Write a temp file with known content; compute MD5 + SHA-256 via the generator
/// (reference path); then run both `verify_file_concurrent` and `verify_file`
/// (the sequential-delegating wrapper) and assert that every per-algorithm result
/// agrees.
#[test]
fn test_concurrent_checksum_verify_matches_sequential() {
    let content = b"OxiMedia Wave 15 concurrent verify test content";

    let (tmp, expected) = make_expected(
        content,
        vec![ChecksumAlgorithm::Md5, ChecksumAlgorithm::Sha256],
    );

    let verifier = ChecksumVerifier::new();

    // Concurrent path
    let concurrent_report = verifier
        .verify_file_concurrent(&expected)
        .expect("concurrent verify should succeed");

    // Sequential (delegating) path
    let sequential_report = verifier
        .verify_file(&expected)
        .expect("sequential verify should succeed");

    // Both should report full success
    assert!(
        concurrent_report.all_passed,
        "concurrent report must be all-passed for correct checksums"
    );
    assert!(
        sequential_report.is_success(),
        "sequential report must succeed for correct checksums"
    );

    // Per-algorithm results must agree between concurrent and sequential
    for ar in &concurrent_report.algo_results {
        let seq_result = sequential_report
            .results
            .get(&ar.algo)
            .expect("sequential report must contain the same algorithms");

        assert!(ar.passed, "concurrent result for {:?} must pass", ar.algo);
        assert!(
            seq_result.is_success(),
            "sequential result for {:?} must succeed",
            ar.algo
        );
    }

    // Both algorithms must be present in the concurrent report
    let algos_found: std::collections::HashSet<ChecksumAlgorithm> = concurrent_report
        .algo_results
        .iter()
        .map(|r| r.algo)
        .collect();
    assert!(
        algos_found.contains(&ChecksumAlgorithm::Md5),
        "MD5 must be in concurrent results"
    );
    assert!(
        algos_found.contains(&ChecksumAlgorithm::Sha256),
        "SHA-256 must be in concurrent results"
    );

    // Keep temp file alive until end of test
    drop(tmp);
}

/// Wave 15 / Slice I — test 2
///
/// Write a temp file; build a `FileChecksum` with the CORRECT SHA-256 but a
/// deliberately WRONG MD5 hex string; run `verify_file_concurrent` and assert:
/// - MD5 result: `passed == false`
/// - SHA-256 result: `passed == true`
/// - `all_passed == false`
#[test]
fn test_concurrent_checksum_verify_detects_mismatch() {
    let content = b"OxiMedia Wave 15 mismatch detection test";

    let (tmp, mut expected) = make_expected(
        content,
        vec![ChecksumAlgorithm::Md5, ChecksumAlgorithm::Sha256],
    );

    // Corrupt the MD5 entry with a clearly wrong value while preserving SHA-256
    expected.checksums.insert(
        ChecksumAlgorithm::Md5,
        "deadbeefdeadbeefdeadbeefdeadbeef".to_string(),
    );

    let verifier = ChecksumVerifier::new();
    let report = verifier
        .verify_file_concurrent(&expected)
        .expect("concurrent verify should not error on hash mismatch");

    // Combined flag must be false because MD5 is wrong
    assert!(
        !report.all_passed,
        "all_passed must be false when one algorithm fails"
    );

    // Find individual results
    let md5_result = report
        .algo_results
        .iter()
        .find(|r| r.algo == ChecksumAlgorithm::Md5)
        .expect("MD5 result must be present");

    let sha256_result = report
        .algo_results
        .iter()
        .find(|r| r.algo == ChecksumAlgorithm::Sha256)
        .expect("SHA-256 result must be present");

    assert!(
        !md5_result.passed,
        "MD5 must fail — we injected a wrong hash"
    );
    assert_eq!(
        md5_result.expected, "deadbeefdeadbeefdeadbeefdeadbeef",
        "expected field must reflect the injected bad hash"
    );
    assert!(
        !md5_result.computed.is_empty(),
        "computed MD5 must not be empty"
    );

    assert!(
        sha256_result.passed,
        "SHA-256 must still pass — only MD5 was corrupted"
    );

    // Keep temp file alive
    drop(tmp);
}
