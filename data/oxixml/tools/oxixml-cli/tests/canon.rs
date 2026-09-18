//! Integration tests for `oxixml canon`.

mod support;

use support::{TempFile, run, run_no_stdin};

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["canon", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml canon"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_c14n_of_an_xml_file() {
    let input = TempFile::new("canon-c14n", "xml", "<a  ><b/></a>");
    let (status, stdout, stderr) = run_no_stdin(&["canon", "--algorithm", "c14n", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    // Canonical XML normalizes attribute whitespace and expands an empty element to a matched
    // start/end tag pair.
    assert_eq!(stdout, "<a><b></b></a>");
}

#[test]
fn stdin_input_is_read_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(&["canon", "--algorithm", "c14n", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a><b></b></a>");
}

#[test]
fn exclusive_c14n_is_selectable() {
    let (status, stdout, stderr) = run(&["canon", "--algorithm", "exc-c14n", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a><b></b></a>");
}

#[test]
fn c14n_1_1_is_selectable() {
    let (status, stdout, stderr) = run(&["canon", "--algorithm", "c14n11", "-"], b"<a><b/></a>");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "<a><b></b></a>");
}

#[test]
fn with_comments_keeps_comments_in_the_output() {
    let (status, stdout, stderr) = run(
        &["canon", "--algorithm", "c14n", "--with-comments", "-"],
        b"<a><!--hi--></a>",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<!--hi-->"), "{stdout}");
}

#[test]
fn happy_path_rdfc10_canonicalizes_a_dataset() {
    let (status, stdout, stderr) = run(
        &["canon", "--algorithm", "rdfc10", "--from", "ntriples", "-"],
        b"<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("http://example.com/s"), "{stdout}");
    assert!(stdout.contains("http://example.com/p"), "{stdout}");
    assert!(stdout.contains("http://example.com/o"), "{stdout}");
}

#[test]
fn rdfc10_relabels_blank_nodes_deterministically() {
    let (status, stdout, stderr) = run(
        &["canon", "--algorithm", "rdfc10", "--from", "ntriples", "-"],
        b"_:b1 <http://example.com/p> <http://example.com/o> .\n",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("_:c14n0"), "{stdout}");
}

#[test]
fn missing_algorithm_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["canon", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--algorithm"));
}

#[test]
fn an_unknown_algorithm_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["canon", "--algorithm", "bogus", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("bogus"));
}

#[test]
fn with_comments_together_with_rdfc10_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["canon", "--algorithm", "rdfc10", "--with-comments", "a.nt"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--with-comments"));
}

#[test]
fn from_together_with_an_xml_algorithm_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&[
        "canon",
        "--algorithm",
        "c14n",
        "--from",
        "ntriples",
        "a.xml",
    ]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--from"));
}

#[test]
fn xinclude_together_with_rdfc10_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["canon", "--algorithm", "rdfc10", "--xinclude", "a.nt"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--xinclude"));
}

#[test]
fn a_not_well_formed_input_is_a_processing_error() {
    let (status, stdout, stderr) = run(&["canon", "--algorithm", "c14n", "-"], b"<a><b></a>");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
