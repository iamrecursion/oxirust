//! Smoke tests verifying the scopes subcommand handles non-Y4M containers
//! gracefully — either with a clean "unsupported extension" error or a
//! "file not found" error, never a panic.

use assert_cmd::Command;
use predicates::prelude::*;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

#[test]
fn scopes_waveform_avi_extension_clean_error() {
    // .avi is not a supported container — must produce a clean, non-panic error.
    let input_path = std::env::temp_dir()
        .join("test_oximedia_scopes_smoke.avi")
        .display()
        .to_string();
    let output_path = std::env::temp_dir().join("oximedia_smoke_waveform_avi.raw");
    let output_str = output_path.display().to_string();
    oximedia()
        .args([
            "scopes",
            "waveform",
            "--input",
            &input_path,
            "--output",
            &output_str,
        ])
        .assert()
        .failure()
        // Must NOT panic
        .stderr(predicate::str::contains("panicked").not())
        // Should mention the unsupported format in some way
        .stderr(
            predicate::str::contains("unsupported")
                .or(predicate::str::contains("Unsupported"))
                .or(predicate::str::contains("not found"))
                .or(predicate::str::contains("extension"))
                .or(predicate::str::contains("container")),
        );

    std::fs::remove_file(&output_path).ok();
}

#[test]
fn scopes_waveform_mp4_nonexistent_clean_error() {
    // Missing .mp4 file should produce a clean error — not "only Y4M input supported".
    let output_path = std::env::temp_dir().join("oximedia_smoke_waveform_mp4.raw");
    let output_path_str = output_path.display().to_string();
    let result = oximedia()
        .args([
            "scopes",
            "waveform",
            "--input",
            "/nonexistent_path_oximedia_test_12345.mp4",
            "--output",
            &output_path_str,
        ])
        .output()
        .expect("failed to run oximedia");

    let stderr = String::from_utf8_lossy(&result.stderr);

    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "scopes waveform must not panic on missing MP4: {stderr}"
    );

    // Should fail (file not found)
    assert!(
        !result.status.success(),
        "scopes waveform should fail on nonexistent MP4"
    );

    // Should NOT say "only Y4M input supported" — that old message is gone
    assert!(
        !stderr.contains("only Y4M input supported"),
        "old Y4M-only error should not appear for .mp4 extension: {stderr}"
    );

    std::fs::remove_file(&output_path).ok();
}
