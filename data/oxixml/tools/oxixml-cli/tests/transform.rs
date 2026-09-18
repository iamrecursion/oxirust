//! Integration tests for `oxixml transform`.

mod support;

use support::{TempFile, run, run_no_stdin};

const IDENTITY_LIKE_XSL: &str = r#"<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="text"/>
  <xsl:template match="/">out=<xsl:value-of select="/a/b"/></xsl:template>
</xsl:stylesheet>
"#;

const PARAMETERIZED_XSL: &str = r#"<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="text"/>
  <xsl:param name="who" select="'world'"/>
  <xsl:template match="/">hello <xsl:value-of select="$who"/></xsl:template>
</xsl:stylesheet>
"#;

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["transform", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml transform"));
    assert!(stderr.is_empty());
}

#[test]
fn happy_path_applies_a_stylesheet_to_a_file() {
    let xsl = TempFile::new("transform-happy", "xsl", IDENTITY_LIKE_XSL);
    let source = TempFile::new("transform-happy", "xml", "<a><b>hello</b></a>");
    let (status, stdout, stderr) = run_no_stdin(&["transform", "--xsl", xsl.path(), source.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "out=hello");
}

#[test]
fn stdin_supplies_the_source_document_when_the_path_is_a_dash() {
    let xsl = TempFile::new("transform-stdin", "xsl", IDENTITY_LIKE_XSL);
    let (status, stdout, stderr) = run(
        &["transform", "--xsl", xsl.path(), "-"],
        b"<a><b>from stdin</b></a>",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "out=from stdin");
}

#[test]
fn the_stylesheet_itself_can_be_read_from_stdin() {
    let source = TempFile::new("transform-xsl-stdin", "xml", "<a><b>hi</b></a>");
    let (status, stdout, stderr) = run(
        &["transform", "--xsl", "-", source.path()],
        IDENTITY_LIKE_XSL.as_bytes(),
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "out=hi");
}

#[test]
fn a_stylesheet_parameter_is_bound() {
    let xsl = TempFile::new("transform-param", "xsl", PARAMETERIZED_XSL);
    let source = TempFile::new("transform-param", "xml", "<a/>");
    let (status, stdout, stderr) = run_no_stdin(&[
        "transform",
        "--xsl",
        xsl.path(),
        "--param",
        "who=oxixml",
        source.path(),
    ]);
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "hello oxixml");
}

#[test]
fn output_can_be_written_to_a_file() {
    let xsl = TempFile::new("transform-output", "xsl", IDENTITY_LIKE_XSL);
    let source = TempFile::new("transform-output", "xml", "<a><b>to a file</b></a>");
    let output_path = format!("{}.out", source.path());
    let (status, stdout, stderr) = run_no_stdin(&[
        "transform",
        "--xsl",
        xsl.path(),
        "--output",
        &output_path,
        source.path(),
    ]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.is_empty());
    let written = std::fs::read_to_string(&output_path).expect("read the --output file");
    assert_eq!(written, "out=to a file");
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn missing_xsl_flag_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["transform", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--xsl"));
}

#[test]
fn missing_input_path_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["transform", "--xsl", "a.xsl"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("input path"));
}

#[test]
fn a_malformed_param_is_a_usage_error() {
    let xsl = TempFile::new("transform-bad-param", "xsl", IDENTITY_LIKE_XSL);
    let (status, _stdout, stderr) = run_no_stdin(&[
        "transform",
        "--xsl",
        xsl.path(),
        "--param",
        "no-equals-sign",
        "a.xml",
    ]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--param"));
}

#[test]
fn a_missing_stylesheet_file_is_a_processing_error() {
    let source = TempFile::new("transform-missing-xsl", "xml", "<a/>");
    let (status, stdout, stderr) = run_no_stdin(&[
        "transform",
        "--xsl",
        "/no/such/oxixml-cli-transform.xsl",
        source.path(),
    ]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
