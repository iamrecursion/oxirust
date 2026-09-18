//! Smoke tests for the `man-page` subcommand.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn man_page_contains_th_header() {
    // clap_mangen emits preamble lines before .TH on some versions.
    // We check that the output contains both .TH and the binary name.
    Command::cargo_bin("oximedia")
        .expect("oximedia binary should exist")
        .args(["man-page"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains(".TH oximedia").or(predicate::str::contains(".TH OXIMEDIA")),
        );
}

#[test]
fn man_page_contains_name_section() {
    Command::cargo_bin("oximedia")
        .expect("oximedia binary should exist")
        .args(["man-page"])
        .assert()
        .success()
        .stdout(predicate::str::contains(".SH NAME"));
}
