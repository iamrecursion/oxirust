//! Integration tests for the top-level `oxixml` entry point: help, version, and subcommand
//! dispatch/rejection — everything in [`oxixml_cli::run_status`] itself, above any one
//! subcommand.

mod support;

use support::run_no_stdin;

#[test]
fn no_arguments_prints_the_top_level_help_and_succeeds() {
    let (status, stdout, stderr) = run_no_stdin(&[]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml"));
    assert!(stdout.contains("SUBCOMMANDS:"));
    assert!(stderr.is_empty());
}

#[test]
fn top_level_help_flag_matches_no_arguments() {
    let (status, stdout, _stderr) = run_no_stdin(&["--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("SUBCOMMANDS:"));

    let (status, stdout, _stderr) = run_no_stdin(&["-h"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("SUBCOMMANDS:"));
}

#[test]
fn version_flag_prints_the_crate_version_and_succeeds() {
    let (status, stdout, stderr) = run_no_stdin(&["--version"]);
    assert_eq!(status, 0);
    assert!(stdout.starts_with("oxixml "), "{stdout}");
    assert!(stderr.is_empty());
}

#[test]
fn an_unknown_subcommand_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["frobnicate"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("frobnicate"));
}

#[test]
fn every_documented_subcommand_is_recognized() {
    // None of these are given enough of their own arguments to succeed; the point is only that
    // dispatch finds the subcommand at all (a usage error from *within* the subcommand, not the
    // "unknown subcommand" error from the top level).
    for subcommand in [
        "fmt",
        "lint",
        "validate",
        "transform",
        "query",
        "canon",
        "convert",
        "sparql",
    ] {
        let (status, _stdout, stderr) = run_no_stdin(&[subcommand]);
        assert_eq!(status, 1, "subcommand `{subcommand}`: {stderr}");
        assert!(
            stderr.starts_with(&format!("oxixml {subcommand}")),
            "subcommand `{subcommand}` was not recognized by dispatch: {stderr}"
        );
    }
    // `hdt` alone is not a usage error: like the top level, it shows its own help and succeeds
    // when no further subcommand is given (see `tests/hdt.rs` for its `build`/`dump`/`stats`
    // sub-subcommands).
    let (status, stdout, stderr) = run_no_stdin(&["hdt"]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("oxixml hdt"));
}
