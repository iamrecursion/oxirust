//! Smoke tests for the TUI subcommand.

use assert_cmd::Command;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary not found")
}

#[test]
fn tui_help_exits_zero() {
    oximedia().args(["tui", "--help"]).assert().success();
}
