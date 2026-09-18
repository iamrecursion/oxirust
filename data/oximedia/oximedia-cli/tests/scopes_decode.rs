//! Smoke tests for the `scopes` subcommand.
//!
//! These tests verify the command exits cleanly and does not panic even on
//! missing or unsupported input files.

use assert_cmd::Command;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

#[test]
fn scopes_help_includes_subcommands() {
    oximedia().args(["scopes", "--help"]).assert().success();
}

#[test]
fn scopes_waveform_help_exits_zero() {
    oximedia()
        .args(["scopes", "waveform", "--help"])
        .assert()
        .success();
}

#[test]
fn scopes_vectorscope_help_exits_zero() {
    oximedia()
        .args(["scopes", "vectorscope", "--help"])
        .assert()
        .success();
}

#[test]
fn scopes_histogram_help_exits_zero() {
    oximedia()
        .args(["scopes", "histogram", "--help"])
        .assert()
        .success();
}

#[test]
fn scopes_waveform_on_missing_file_exits_cleanly() {
    // scopes waveform on a nonexistent file must not panic.
    // It requires --output too.
    let input_path = std::env::temp_dir()
        .join("nonexistent_video_for_scopes_test_oximedia.mp4")
        .display()
        .to_string();
    let output_path = std::env::temp_dir().join("oximedia_test_waveform_out.raw");
    let output_path_str = output_path.display().to_string();
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("oximedia"))
        .args([
            "scopes",
            "waveform",
            "--input",
            &input_path,
            "--output",
            &output_path_str,
        ])
        .output()
        .expect("failed to run oximedia scopes waveform");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "scopes waveform must not panic on missing input: {stderr}"
    );

    // Should exit non-zero since the file does not exist
    assert!(
        !output.status.success(),
        "scopes waveform should fail on missing input"
    );

    // Clean up if accidentally created
    std::fs::remove_file(&output_path).ok();
}

#[test]
fn scopes_false_color_help_exits_zero() {
    oximedia()
        .args(["scopes", "false-color", "--help"])
        .assert()
        .success();
}

#[test]
fn scopes_stats_help_exits_zero() {
    oximedia()
        .args(["scopes", "stats", "--help"])
        .assert()
        .success();
}
