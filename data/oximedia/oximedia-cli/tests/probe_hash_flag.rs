//! Regression test for `oximedia probe --hash`.
//!
//! Prior to this fix, `--hash` was parsed by clap (`commands/mod.rs`) but
//! discarded in `main.rs` (`hash: _hash`), and
//! `handlers/inspect.rs::probe_file` had no parameter for it at all — the
//! flag was a complete no-op regardless of what the caller passed.
//!
//! This test proves `--hash` now computes a real SHA-256 content hash via
//! `oximedia_archive_pro::checksum::ChecksumGenerator` (the same real,
//! streaming, multi-algorithm hasher already used by `oximedia-archive-pro`
//! elsewhere in the workspace — NOT the fake 64-bit rolling hash in
//! `dedup_cmd::compute_file_hash`) and threads it into all three probe
//! output formats the flag spec covers (text/json/csv). Each test
//! cross-checks the CLI's reported hash against an *independent* in-process
//! `ChecksumGenerator` call over the same fixture bytes, so a fake or
//! placeholder value would be caught immediately.
//!
//! The fixture must be a container `oximedia_container::probe_format` can
//! actually recognise (probe fails — independent of `--hash` — on
//! unrecognised magic bytes), so this reuses `common::write_wav_fixture`
//! (a real, valid, small WAV file) rather than arbitrary bytes.

mod common;

use assert_cmd::Command;
use tempfile::TempDir;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

/// Build a small, deterministic, real WAV fixture (via `common::write_wav_fixture`
/// so `oximedia_container::probe_format` recognises it) and independently
/// compute its real SHA-256 hex digest via
/// `oximedia_archive_pro::checksum::ChecksumGenerator` — the exact same
/// production hasher `probe_file` calls internally, invoked here directly
/// (not through the CLI subprocess) so the two computations are genuinely
/// independent.
fn write_fixture_with_expected_hash() -> (TempDir, std::path::PathBuf, String) {
    use oximedia_archive_pro::checksum::{ChecksumAlgorithm, ChecksumGenerator};

    // Small (a few hundred bytes), deterministic, real WAV content.
    let (dir, path) = common::write_wav_fixture(440.0, 8_000, 1, 0.05);

    let checksum = ChecksumGenerator::new()
        .with_algorithms(vec![ChecksumAlgorithm::Sha256])
        .generate_file(&path)
        .expect("independent SHA-256 generation should succeed");
    let expected_hash = checksum
        .checksums
        .get(&ChecksumAlgorithm::Sha256)
        .cloned()
        .expect("SHA-256 checksum must be present in generator output");

    // Sanity on the oracle itself: a real SHA-256 hex digest is exactly 64
    // lowercase hex characters.
    assert_eq!(
        expected_hash.len(),
        64,
        "SHA-256 hex digest must be 64 chars"
    );
    assert!(
        expected_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA-256 hex digest must be pure hex: {expected_hash}"
    );

    (dir, path, expected_hash)
}

#[test]
fn probe_hash_text_output_matches_independent_sha256() {
    let (_dir, path, expected_hash) = write_fixture_with_expected_hash();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--hash",
        ])
        .output()
        .expect("failed to run oximedia probe --hash");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "probe --hash must not panic: {stderr}"
    );
    assert!(
        output.status.success(),
        "probe --hash should succeed on a real WAV file; stderr: {stderr}"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&expected_hash),
        "text output must contain the real SHA-256 hex digest {expected_hash}; stdout:\n{stdout}"
    );
}

#[test]
fn probe_hash_json_output_field_matches_independent_sha256() {
    let (_dir, path, expected_hash) = write_fixture_with_expected_hash();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--hash",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run oximedia probe --hash --format json");

    assert!(
        output.status.success(),
        "probe --hash --format json should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("probe --format json output should be valid JSON");

    let hash_obj = json
        .get("hash")
        .unwrap_or_else(|| panic!("JSON output must contain a top-level \"hash\" field: {json}"));
    assert_eq!(
        hash_obj
            .get("algorithm")
            .and_then(serde_json::Value::as_str),
        Some("sha256"),
        "hash.algorithm must be \"sha256\": {hash_obj}"
    );
    assert_eq!(
        hash_obj.get("value").and_then(serde_json::Value::as_str),
        Some(expected_hash.as_str()),
        "hash.value must match the independently-computed SHA-256: {hash_obj}"
    );
}

#[test]
fn probe_hash_csv_output_row_ends_with_independent_sha256() {
    let (_dir, path, expected_hash) = write_fixture_with_expected_hash();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--hash",
            "--format",
            "csv",
        ])
        .output()
        .expect("failed to run oximedia probe --hash --format csv");

    assert!(
        output.status.success(),
        "probe --hash --format csv should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let mut lines = stdout.lines();
    let header = lines.next().expect("csv output must have a header line");
    assert!(
        header.contains("hash_sha256"),
        "csv header must include a hash_sha256 column when --hash is set: {header}"
    );
    let row = lines.next().expect("csv output must have a data row");
    assert!(
        row.ends_with(&expected_hash),
        "csv data row must end with the real SHA-256 hash {expected_hash}: {row}"
    );
}

#[test]
fn probe_without_hash_flag_omits_hash_field() {
    let (_dir, path, _expected_hash) = write_fixture_with_expected_hash();

    let output = oximedia()
        .args([
            "--quiet",
            "probe",
            "-i",
            path.to_str().expect("fixture path is UTF-8"),
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run oximedia probe --format json");

    assert!(
        output.status.success(),
        "probe --format json (no --hash) should succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("probe stdout should be UTF-8");
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("probe --format json output should be valid JSON");
    assert!(
        json.get("hash").is_none(),
        "hash field must be absent when --hash is not passed: {json}"
    );
}
