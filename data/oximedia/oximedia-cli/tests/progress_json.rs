//! Integration tests for --progress json flag.
//!
//! Verifies the flag is accepted by the CLI and doesn't cause an
//! "unexpected argument" error.

use assert_cmd::Command;

#[test]
fn progress_json_flag_accepted_by_help() {
    // `--help` is always available; if `--progress json` is an unknown flag
    // clap would reject it before printing help.
    Command::cargo_bin("oximedia")
        .expect("oximedia binary not found")
        .args(["--progress", "json", "--help"])
        .assert()
        .success();
}

#[test]
fn progress_plain_flag_accepted() {
    Command::cargo_bin("oximedia")
        .expect("oximedia binary not found")
        .args(["--progress", "plain", "--help"])
        .assert()
        .success();
}

#[test]
fn progress_json_flag_shows_in_help_output() {
    let output = Command::cargo_bin("oximedia")
        .expect("oximedia binary not found")
        .args(["--help"])
        .output()
        .expect("failed to run oximedia --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The flag must appear in the help text.
    assert!(
        stdout.contains("--progress"),
        "--progress flag should appear in --help output; got:\n{stdout}"
    );
}
