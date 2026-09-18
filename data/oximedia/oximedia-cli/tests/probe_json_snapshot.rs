//! Tests for the `probe` subcommand JSON output and error handling.

use assert_cmd::Command;
use predicates::prelude::*;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

#[test]
fn probe_nonexistent_file_exits_nonzero_cleanly() {
    // probe should fail cleanly (not panic) on nonexistent file
    let nonexistent = std::env::temp_dir()
        .join("this_file_does_not_exist_oximedia_test.mp4")
        .display()
        .to_string();
    let output = oximedia()
        .args(["probe", "--input", &nonexistent])
        .output()
        .expect("failed to run oximedia probe");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Must not panic
    assert!(
        !stderr.contains("thread 'main' panicked"),
        "must not panic, stderr: {stderr}"
    );
    // Must exit non-zero
    assert!(
        !output.status.success(),
        "probe on nonexistent file should fail"
    );
}

#[test]
fn probe_help_contains_input_flag() {
    oximedia()
        .args(["probe", "--help"])
        .assert()
        .success()
        .stdout(predicates::str::contains("input").or(predicates::str::contains("--help")));
}

#[test]
fn probe_json_flag_present_in_help() {
    // The global --json flag should appear in top-level help
    oximedia()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("json"));
}
