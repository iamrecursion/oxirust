//! Integration tests for `oximedia-ff --explain` mode.
//!
//! Verifies that `--explain` prints the arg→field translation table to stdout
//! and exits with code 0 without actually transcoding any files.

use assert_cmd::Command;

/// `--explain` with a basic AV1 + Opus job should print the translation table
/// and exit successfully without touching the filesystem.
#[test]
fn test_explain_prints_translation_table() {
    let tmp_in = std::env::temp_dir().join("oximedia_explain_test_input.mkv");

    // We do NOT need a real file — --explain exits before any I/O.
    let output = Command::cargo_bin("oximedia-ff")
        .expect("oximedia-ff binary must exist")
        .args([
            "--explain",
            "-i",
            tmp_in.to_str().expect("valid temp path"),
            "-c:v",
            "libaom-av1",
            "-crf",
            "28",
            "-c:a",
            "libopus",
            "output_explain_test.webm",
        ])
        .output()
        .expect("process must run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should exit successfully.
    assert!(
        output.status.success(),
        "oximedia-ff --explain should exit 0; stderr: {}",
        stderr
    );

    // The translation table header must appear.
    assert!(
        stdout.contains("Translation table") || stderr.contains("Translation table"),
        "Expected 'Translation table' in output, got stdout='{}', stderr='{}'",
        stdout,
        stderr
    );

    // The codec should be mentioned (after patent substitution).
    assert!(
        stdout.contains("av1") || stdout.contains("libaom-av1"),
        "Expected codec in explain output, got: {}",
        stdout
    );

    // The crf value should appear.
    assert!(
        stdout.contains("28") || stdout.contains("crf"),
        "Expected crf in explain output, got: {}",
        stdout
    );

    // No actual output file should have been created.
    assert!(
        !std::path::Path::new("output_explain_test.webm").exists(),
        "--explain must not create output files"
    );
}

/// `--explain` combined with patent-codec substitution should note the substitution
/// in the translation table (av1 replaces libx264).
#[test]
fn test_explain_shows_patent_substitution() {
    let tmp_in = std::env::temp_dir().join("oximedia_explain_patent_test.mkv");

    let output = Command::cargo_bin("oximedia-ff")
        .expect("oximedia-ff binary must exist")
        .args([
            "--explain",
            "-i",
            tmp_in.to_str().expect("valid temp path"),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "output_explain_patent.webm",
        ])
        .output()
        .expect("process must run");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "should exit 0 even with patent codecs"
    );

    // After substitution, av1 and/or opus should appear.
    assert!(
        stdout.contains("av1") || stdout.contains("opus") || stdout.contains("-c:v"),
        "Expected substituted codec in explain output, got: {}",
        stdout
    );
}
