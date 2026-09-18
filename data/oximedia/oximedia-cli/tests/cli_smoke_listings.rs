//! Smoke tests for "no-input" listing-style subcommands.
//!
//! Each test invokes a deterministic list/info verb that requires no media
//! input, no sockets, no GPU — these should always exit 0 on any host with
//! the binary built. They complement the `cli_help_per_command.rs` `--help`
//! coverage by also exercising the actual handler dispatch path.

use assert_cmd::Command;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary should exist")
}

/// `oximedia preset list` — enumerates built-in encoding presets.
#[test]
fn preset_list_succeeds() {
    oximedia().args(["preset", "list"]).assert().success();
}

/// `oximedia plugin list` — lists registered codec plugins (zero on default
/// build, but the handler must run cleanly).
#[test]
fn plugin_list_succeeds() {
    oximedia().args(["plugin", "list"]).assert().success();
}

/// `oximedia loudness standards` — lists supported broadcast loudness targets.
#[test]
fn loudness_standards_succeeds() {
    oximedia()
        .args(["loudness", "standards"])
        .assert()
        .success();
}

/// `oximedia quality list` — lists available quality assessment metrics.
#[test]
fn quality_list_succeeds() {
    oximedia().args(["quality", "list"]).assert().success();
}

/// `oximedia info` — prints the supported codec/format matrix.
#[test]
fn info_succeeds() {
    oximedia().arg("info").assert().success();
}

/// `oximedia version` — prints version, build info, and feature set.
#[test]
fn version_subcommand_succeeds() {
    oximedia().arg("version").assert().success();
}

/// `oximedia doctor` — runs environment diagnostics (no input required).
#[test]
fn doctor_succeeds() {
    oximedia().arg("doctor").assert().success();
}

/// `oximedia doctor --json` — diagnostics as machine-readable JSON.
#[test]
fn doctor_json_succeeds() {
    oximedia().args(["doctor", "--json"]).assert().success();
}

/// `oximedia man-page` — emits a Unix man page on stdout.
#[test]
fn man_page_succeeds() {
    oximedia().arg("man-page").assert().success();
}

/// `oximedia completions bash` — emits bash shell completions on stdout.
#[test]
fn completions_bash_succeeds() {
    oximedia().args(["completions", "bash"]).assert().success();
}
