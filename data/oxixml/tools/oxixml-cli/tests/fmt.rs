//! Integration tests for `oxixml fmt`.

mod support;

use support::{TempFile, run, run_no_stdin};

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["fmt", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml fmt"));
    assert!(stdout.contains("USAGE:"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_pretty_prints_from_a_file() {
    let input = TempFile::new("fmt-happy", "xml", "<a><b/><c>text</c></a>");
    let (status, stdout, stderr) = run_no_stdin(&["fmt", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a>\n  <b/>\n  <c>text</c>\n</a>\n");
}

#[test]
fn stdin_input_is_read_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(&["fmt", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a>\n  <b/>\n</a>\n");
}

#[test]
fn custom_indent_is_honored() {
    let (status, stdout, stderr) = run(&["fmt", "--indent", "4", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a>\n    <b/>\n</a>\n");
}

#[test]
fn preserve_space_leaves_the_document_compact() {
    let (status, stdout, stderr) = run(&["fmt", "--preserve-space", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a><b/></a>\n");
}

#[test]
fn in_place_rewrites_the_input_file() {
    let input = TempFile::new("fmt-in-place", "xml", "<a><b/></a>");
    let (status, stdout, stderr) = run_no_stdin(&["fmt", "--in-place", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(
        stdout.is_empty(),
        "--in-place must not also write to stdout"
    );
    let rewritten = std::fs::read_to_string(input.path()).expect("read back the rewritten file");
    assert_eq!(rewritten, "<a>\n  <b/>\n</a>\n");
}

#[test]
fn missing_input_path_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["fmt"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("input path"));
}

#[test]
fn indent_combined_with_preserve_space_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run(&["fmt", "--indent", "4", "--preserve-space", "-"], b"<a/>");
    assert_eq!(status, 1);
    assert!(stderr.contains("--indent"));
    assert!(stderr.contains("--preserve-space"));
}

#[test]
fn in_place_with_stdin_is_a_usage_error() {
    let (status, _stdout, stderr) = run(&["fmt", "--in-place", "-"], b"<a/>");
    assert_eq!(status, 1);
    assert!(stderr.contains("--in-place"));
}

#[test]
fn an_unrecognized_flag_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["fmt", "--bogus"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--bogus"));
}

#[test]
fn a_not_well_formed_input_is_a_processing_error() {
    let (status, stdout, stderr) = run(&["fmt", "-"], b"<a><b></a>");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn a_missing_file_is_a_processing_error() {
    let (status, stdout, stderr) =
        run_no_stdin(&["fmt", "/no/such/directory/oxixml-cli-fmt-test.xml"]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
