//! Integration tests for `oxixml sparql`.

mod support;

use support::{run, run_no_stdin};

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["sparql", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml sparql"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_prints_the_algebra_as_sse() {
    let (status, stdout, stderr) =
        run_no_stdin(&["sparql", "--parse", "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))\n");
}

#[test]
fn stdin_input_is_read_when_the_query_is_a_dash() {
    let (status, stdout, stderr) = run(&["sparql", "--parse", "-"], b"ASK { ?s ?p ?o }");
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("bgp"), "{stdout}");
}

#[test]
fn missing_parse_flag_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["sparql"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--parse"));
}

#[test]
fn a_positional_argument_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["sparql", "--parse", "ASK { ?s ?p ?o }", "extra"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("positional"));
}

#[test]
fn a_syntax_error_is_a_processing_error() {
    let (status, stdout, stderr) = run_no_stdin(&["sparql", "--parse", "SELECT ?s WHERE"]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn base_is_not_applicable_to_sparql() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["sparql", "--base", "urn:x", "--parse", "ASK{}"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--base"));
}
