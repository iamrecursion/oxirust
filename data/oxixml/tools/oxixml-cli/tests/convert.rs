//! Integration tests for `oxixml convert`.

mod support;

use support::{TempFile, run, run_no_stdin};

const TURTLE_FIXTURE: &str = "@prefix ex: <http://example.com/> .\nex:s ex:p ex:o .\n";

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["convert", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml convert"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_rdf_format_conversion() {
    let input = TempFile::new("convert-happy", "ttl", TURTLE_FIXTURE);
    let (status, stdout, stderr) = run_no_stdin(&[
        "convert",
        "--from",
        "turtle",
        "--to",
        "ntriples",
        input.path(),
    ]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<http://example.com/s>"), "{stdout}");
    assert!(stdout.contains("<http://example.com/p>"), "{stdout}");
    assert!(stdout.contains("<http://example.com/o>"), "{stdout}");
}

#[test]
fn stdin_input_is_read_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(
        &["convert", "--from", "turtle", "--to", "ntriples", "-"],
        TURTLE_FIXTURE.as_bytes(),
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<http://example.com/o>"), "{stdout}");
}

#[test]
fn sparql_results_conversion() {
    let xml_results = r#"<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="s"/></head>
  <results>
    <result><binding name="s"><uri>http://example.com/s</uri></binding></result>
  </results>
</sparql>
"#;
    let input = TempFile::new("convert-results", "srx", xml_results);
    let (status, stdout, stderr) = run_no_stdin(&["convert", "--results", "csv", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains('s'));
    assert!(stdout.contains("http://example.com/s"), "{stdout}");
}

#[test]
fn missing_from_and_to_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["convert", "a.ttl"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--from"));
    assert!(stderr.contains("--to"));
}

#[test]
fn missing_input_path_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["convert", "--from", "turtle", "--to", "ntriples"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("input path"));
}

#[test]
fn an_unknown_format_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["convert", "--from", "bogus", "--to", "ntriples", "a.ttl"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("bogus"));
}

#[test]
fn to_together_with_results_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["convert", "--results", "csv", "--to", "ntriples", "a.srx"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--results"));
}

#[test]
fn csv_as_results_from_format_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["convert", "--results", "json", "--from", "csv", "a.csv"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("csv"));
}

#[test]
fn a_syntax_error_in_the_rdf_input_is_a_processing_error() {
    let (status, stdout, stderr) = run(
        &["convert", "--from", "turtle", "--to", "ntriples", "-"],
        b"not turtle {{{",
    );
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
