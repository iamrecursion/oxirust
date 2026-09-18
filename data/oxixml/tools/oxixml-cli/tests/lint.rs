//! Integration tests for `oxixml lint`.

mod support;

use support::{TempFile, run, run_no_stdin};

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["lint", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml lint"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_a_well_formed_file_exits_zero() {
    let input = TempFile::new("lint-happy", "xml", "<a><b/></a>");
    let (status, stdout, stderr) = run_no_stdin(&["lint", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("well-formed"));
}

#[test]
fn stdin_input_is_read_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(&["lint", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<stdin>: well-formed"));
}

#[test]
fn a_not_well_formed_document_exits_with_the_validation_failure_status() {
    let (status, stdout, stderr) = run(&["lint", "-"], b"<a><b></a>");
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("not well-formed"));
    assert!(stdout.contains("<stdin>:1:"), "{stdout}");
}

#[test]
fn quiet_suppresses_the_summary_line_but_not_the_diagnostics() {
    let (status, stdout, _stderr) = run(&["lint", "--quiet", "-"], b"<a><b></a>");
    assert_eq!(status, 3);
    assert!(!stdout.contains("not well-formed"));
    assert!(stdout.contains("<stdin>:1:"));
}

#[test]
fn missing_input_path_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["lint"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("input path"));
}

#[test]
fn too_many_input_paths_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["lint", "a.xml", "b.xml"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("exactly one"));
}

#[test]
fn quiet_and_verbose_together_is_a_usage_error() {
    let (status, _stdout, stderr) = run(&["lint", "--quiet", "--verbose", "-"], b"<a/>");
    assert_eq!(status, 1);
    assert!(stderr.contains("--quiet"));
    assert!(stderr.contains("--verbose"));
}

#[test]
fn a_missing_file_is_a_processing_error() {
    let (status, stdout, stderr) =
        run_no_stdin(&["lint", "/no/such/directory/oxixml-cli-lint-test.xml"]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
