//! Root-level `--help` and `--version` smoke tests.
//!
//! Per-subcommand help coverage lives in `cli_help_per_command.rs`. This file
//! only exercises the bare `oximedia` binary surface (no subcommand).

use assert_cmd::Command;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary should exist")
}

#[test]
fn root_help_succeeds() {
    let out = oximedia()
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8_lossy(&out);
    assert!(
        s.contains("oximedia"),
        "root --help output should contain binary name; got:\n{s}"
    );
    assert!(
        s.contains("Usage") || s.contains("USAGE"),
        "root --help output should contain a Usage line; got:\n{s}"
    );
    assert!(
        s.len() > 200,
        "root --help unexpectedly short ({} bytes); got:\n{s}",
        s.len()
    );
}

#[test]
fn root_version_succeeds() {
    let out = oximedia()
        .arg("--version")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8_lossy(&out);
    assert!(
        s.contains("oximedia") || s.contains("OxiMedia"),
        "--version output should contain a binary identifier; got:\n{s}"
    );
    assert!(
        s.chars().any(|c| c.is_ascii_digit()),
        "--version output should contain at least one digit (semver); got:\n{s}"
    );
}

#[test]
fn root_invocation_with_no_args_exits_nonzero() {
    // `oximedia` with no subcommand should exit non-zero (clap prints help to
    // stderr and bails). This protects against accidental "default subcommand"
    // wiring regressions.
    oximedia().assert().failure();
}

// ---------------------------------------------------------------------------
// Sub-subcommand sanity (a couple of representative deep paths).
// Per-top-level-subcommand help coverage is in cli_help_per_command.rs.
// ---------------------------------------------------------------------------

#[test]
fn normalize_analyze_help_succeeds() {
    oximedia()
        .args(["normalize", "analyze", "--help"])
        .assert()
        .success();
}

#[test]
fn scopes_waveform_help_succeeds() {
    oximedia()
        .args(["scopes", "waveform", "--help"])
        .assert()
        .success();
}
