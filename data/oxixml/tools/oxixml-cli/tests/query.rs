//! Integration tests for `oxixml query`.

mod support;

use support::{TempFile, run, run_no_stdin};

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["query", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml query"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_xpath_with_no_context_item() {
    let (status, stdout, stderr) = run_no_stdin(&["query", "--xpath", "1 + 1"]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "2");
}

#[test]
fn xpath_evaluates_against_a_context_document() {
    let input = TempFile::new("query-xpath-doc", "xml", "<a><b>hi</b><b>there</b></a>");
    let (status, stdout, stderr) = run_no_stdin(&["query", "--xpath", "count(//b)", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "2");
}

#[test]
fn stdin_supplies_the_context_document_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(
        &["query", "--xpath", "/a/b/text()", "-"],
        b"<a><b>hi</b></a>",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "hi");
}

#[test]
fn xquery_is_read_from_a_file() {
    let query = TempFile::new("query-xquery", "xq", "1 + 1");
    let (status, stdout, stderr) = run_no_stdin(&["query", "--xquery", query.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "2");
}

#[test]
fn namespace_bindings_are_honored() {
    let input = TempFile::new(
        "query-ns",
        "xml",
        r#"<r:root xmlns:r="urn:example:r"><r:child/></r:root>"#,
    );
    let (status, stdout, stderr) = run_no_stdin(&[
        "query",
        "--xpath",
        "count(//x:child)",
        "--ns",
        "x=urn:example:r",
        input.path(),
    ]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "1");
}

#[test]
fn xpath_10_can_be_selected() {
    let (status, stdout, stderr) = run_no_stdin(&["query", "--xpath", "1 + 1", "--version", "1.0"]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.trim(), "2");
}

#[test]
fn neither_xpath_nor_xquery_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["query"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--xpath"));
    assert!(stderr.contains("--xquery"));
}

#[test]
fn both_xpath_and_xquery_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["query", "--xpath", "1", "--xquery", "q.xq"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("not both"));
}

#[test]
fn version_with_xquery_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["query", "--xquery", "q.xq", "--version", "2.0"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--version"));
}

#[test]
fn an_invalid_version_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["query", "--xpath", "1", "--version", "9.9"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("9.9"));
}

#[test]
fn a_syntax_error_is_a_processing_error() {
    let (status, stdout, stderr) = run_no_stdin(&["query", "--xpath", "1 +"]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn too_many_input_paths_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["query", "--xpath", "1", "a.xml", "b.xml"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("at most one"));
}
