//! Integration tests for --no-color and environment-variable colour suppression.
//!
//! Each test runs the real `oximedia` binary via `assert_cmd` and checks that
//! ANSI escape codes (\x1b[) are absent from stderr when colour should be
//! disabled.

use assert_cmd::Command;
use predicates::prelude::*;

/// Helper: build a command that will fail (probe a nonexistent file) so the
/// binary exits with an error, which is the path most likely to emit coloured
/// text via `eprintln!("{} …", "Error:".red().bold(), …)`.
fn probe_nonexistent() -> Command {
    let mut cmd = Command::cargo_bin("oximedia").expect("oximedia binary not found");
    cmd.args(["probe", "/oximedia_integration_test_nonexistent_file.mp4"]);
    cmd
}

#[test]
fn no_color_env_disables_ansi() {
    probe_nonexistent()
        .env("NO_COLOR", "1")
        .env_remove("CLICOLOR")
        .env_remove("TERM")
        .assert()
        .failure()
        .stderr(predicates::str::contains("\x1b[").not());
}

#[test]
fn no_color_flag_disables_ansi() {
    probe_nonexistent()
        .arg("--no-color")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .assert()
        .failure()
        .stderr(predicates::str::contains("\x1b[").not());
}

#[test]
fn clicolor_0_disables_ansi() {
    probe_nonexistent()
        .env("CLICOLOR", "0")
        .env_remove("NO_COLOR")
        .env_remove("TERM")
        .assert()
        .failure()
        .stderr(predicates::str::contains("\x1b[").not());
}

#[test]
fn term_dumb_disables_ansi() {
    probe_nonexistent()
        .env("TERM", "dumb")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .assert()
        .failure()
        .stderr(predicates::str::contains("\x1b[").not());
}
