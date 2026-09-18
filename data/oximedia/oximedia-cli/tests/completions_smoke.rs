//! Smoke tests for the `completions` subcommand.

use assert_cmd::Command;
use predicates::prelude::*;

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary should exist")
}

#[test]
fn completions_bash_nonempty() {
    oximedia()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("oximedia").or(predicate::str::contains("complete")));
}

#[test]
fn completions_zsh_nonempty() {
    oximedia()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_fish_nonempty() {
    oximedia()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}
