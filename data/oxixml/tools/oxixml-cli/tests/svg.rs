//! Integration tests for `oxixml svg`.

mod support;

use support::{TempFile, run, run_no_stdin};

const CLEAN_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"
       viewBox="0 0 100 50">
  <rect id="r1" width="10" height="10"/>
  <circle id="c1" cx="50" cy="25" r="20"/>
  <path id="p1" d="M0 0 L10 10 L20 0 Z"/>
</svg>"##;

const DUPLICATE_ID_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg">
  <rect id="dup" width="1" height="1"/>
  <circle id="dup" r="1"/>
</svg>"##;

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["svg", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml svg"));
    assert!(stdout.contains("USAGE:"));
    assert!(stderr.is_empty());
}

#[test]
fn lint_clean_file_exits_zero() {
    let input = TempFile::new("svg-clean", "svg", CLEAN_SVG);
    let (status, stdout, stderr) = run_no_stdin(&["svg", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(
        stdout.contains("0 error(s), 0 warning(s), 0 info(s)"),
        "{stdout}"
    );
}

#[test]
fn a_duplicate_id_exits_validation_failure_and_prints_the_code() {
    let input = TempFile::new("svg-dup-id", "svg", DUPLICATE_ID_SVG);
    let (status, stdout, stderr) = run_no_stdin(&["svg", input.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("error:"), "{stdout}");
    assert!(stdout.contains("[duplicate-id]"), "{stdout}");
    assert!(stdout.contains("1 error(s)"), "{stdout}");
}

#[test]
fn stats_output_has_the_expected_shape() {
    let input = TempFile::new("svg-stats", "svg", CLEAN_SVG);
    let (status, stdout, stderr) = run_no_stdin(&["svg", "--stats", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("width: 100"), "{stdout}");
    assert!(stdout.contains("height: 50"), "{stdout}");
    assert!(stdout.contains("viewBox: 0 0 100 50"), "{stdout}");
    assert!(stdout.contains("elements: 4 total"), "{stdout}");
    assert!(stdout.contains("  svg: 1"), "{stdout}");
    assert!(stdout.contains("  rect: 1"), "{stdout}");
    assert!(stdout.contains("  circle: 1"), "{stdout}");
    assert!(stdout.contains("  path: 1"), "{stdout}");
    assert!(stdout.contains("ids: 3"), "{stdout}");
    assert!(stdout.contains("url() references: 0"), "{stdout}");
    assert!(stdout.contains("paths: 1"), "{stdout}");
    assert!(stdout.contains("path segments: 4"), "{stdout}");
}

#[test]
fn pretty_output_round_trips_through_a_parse() {
    let input = TempFile::new("svg-pretty", "svg", CLEAN_SVG);
    let (status, stdout, stderr) = run_no_stdin(&["svg", "--pretty", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<rect"));
    assert!(stdout.contains('\n'), "pretty output should be indented");

    // Feeding the pretty-printed output back through `svg` (default lint mode) proves it
    // re-parses as a well-formed, still-`svg`-rooted document and stays lint-clean — a
    // structural round trip, without assuming pretty-printing is byte-idempotent (re-parsing
    // introduces real text nodes for the inserted indentation, which a second reformat is not
    // obliged to reproduce identically).
    let (status2, stdout2, stderr2) = run(&["svg", "-"], stdout.as_bytes());
    assert_eq!(status2, 0, "{stderr2}");
    assert!(
        stdout2.contains("0 error(s), 0 warning(s), 0 info(s)"),
        "{stdout2}"
    );
}

#[test]
fn min_output_is_more_compact_than_pretty_output() {
    // Unlike `CLEAN_SVG` (deliberately pre-indented, so `--pretty`'s round-trip test above
    // exercises real whitespace-bearing text nodes), this fixture has no white space between
    // its elements at all, so `--pretty` must *insert* fresh indentation that `--min` omits —
    // otherwise both modes would just reproduce the same (zero) inter-element white space and
    // this comparison would prove nothing.
    let single_line = r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/><circle r="1"/></svg>"##;
    let input = TempFile::new("svg-min", "svg", single_line);
    let (pretty_status, pretty_stdout, _) = run_no_stdin(&["svg", "--pretty", input.path()]);
    let (min_status, min_stdout, min_stderr) = run_no_stdin(&["svg", "--min", input.path()]);
    assert_eq!(pretty_status, 0);
    assert_eq!(min_status, 0, "{min_stderr}");
    assert!(
        min_stdout.len() < pretty_stdout.len(),
        "pretty={pretty_stdout:?} min={min_stdout:?}"
    );
    assert!(min_stdout.contains("<rect"));
}

#[test]
fn pretty_takes_precedence_over_stats_and_min() {
    let input = TempFile::new("svg-precedence", "svg", CLEAN_SVG);
    let (status, stdout, stderr) =
        run_no_stdin(&["svg", "--stats", "--min", "--pretty", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("<rect"));
    assert!(!stdout.contains("elements:"), "{stdout}");
}

#[test]
fn quiet_suppresses_info_findings_and_the_summary_line() {
    // `xlink:href` alongside a bare `href` is an info-severity deprecation note (`xlink:href`
    // family), not an error — a fixture that trips it but nothing more severe.
    let deprecated_only = r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <linearGradient id="g"><stop offset="0" stop-color="red"/></linearGradient>
  <rect id="r1" width="1" height="1" fill="url(#g)"/>
  <use id="u1" href="#r1" xlink:href="#r1"/>
</svg>"##;
    let input = TempFile::new("svg-quiet", "svg", deprecated_only);
    let (status, stdout, stderr) = run_no_stdin(&["svg", "--quiet", input.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(!stdout.contains("info:"), "{stdout}");
    assert!(!stdout.contains("error(s)"), "{stdout}");
}

#[test]
fn missing_input_is_a_processing_error() {
    let (status, stdout, stderr) =
        run_no_stdin(&["svg", "/no/such/directory/oxixml-cli-svg-test.svg"]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn a_non_svg_document_is_a_processing_error_not_a_lint_finding() {
    let (status, stdout, stderr) = run(&["svg", "-"], b"<catalog/>");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}

#[test]
fn stdin_input_is_read_when_the_path_is_a_dash() {
    let (status, stdout, stderr) = run(&["svg", "-"], CLEAN_SVG.as_bytes());
    assert_eq!(status, 0, "{stderr}");
    assert!(
        stdout.contains("0 error(s), 0 warning(s), 0 info(s)"),
        "{stdout}"
    );
}

#[test]
fn omitting_the_input_path_also_reads_from_stdin() {
    let (status, stdout, stderr) = run(&["svg"], CLEAN_SVG.as_bytes());
    assert_eq!(status, 0, "{stderr}");
    assert!(
        stdout.contains("0 error(s), 0 warning(s), 0 info(s)"),
        "{stdout}"
    );
}

#[test]
fn too_many_input_paths_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["svg", "a.svg", "b.svg"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("at most one"));
}

#[test]
fn quiet_and_verbose_together_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run(&["svg", "--quiet", "--verbose", "-"], CLEAN_SVG.as_bytes());
    assert_eq!(status, 1);
    assert!(stderr.contains("--quiet"));
    assert!(stderr.contains("--verbose"));
}
