//! Integration tests for `oxixml validate`, covering all five schema kinds it supports.

mod support;

use support::{TempFile, run, run_no_stdin};

const XSD_FIXTURE: &str = r#"<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="root" type="xs:string"/>
</xs:schema>
"#;

const DTD_FIXTURE: &str = "<!ELEMENT root (#PCDATA)>\n";

const RNG_FIXTURE: &str = r#"<element name="root" xmlns="http://relaxng.org/ns/structure/1.0">
  <text/>
</element>
"#;

const RNC_FIXTURE: &str = "element root { text }\n";

const SCHEMATRON_FIXTURE: &str = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
  <pattern id="p1">
    <rule context="root">
      <assert test="item">a root must have an item</assert>
    </rule>
  </pattern>
</schema>
"#;

#[test]
fn help_is_rendered() {
    let (status, stdout, stderr) = run_no_stdin(&["validate", "--help"]);
    assert_eq!(status, 0);
    assert!(stdout.contains("oxixml validate"));
    assert!(stderr.is_empty());
}

#[test]
fn xsd_happy_path_and_validation_failure() {
    let schema = TempFile::new("validate-xsd", "xsd", XSD_FIXTURE);
    let valid = TempFile::new("validate-xsd-valid", "xml", "<root>hello</root>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--xsd", schema.path(), valid.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));

    let invalid = TempFile::new("validate-xsd-invalid", "xml", "<other/>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--xsd", schema.path(), invalid.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("error"));
}

#[test]
fn xsd_instance_can_be_read_from_stdin() {
    let schema = TempFile::new("validate-xsd-stdin", "xsd", XSD_FIXTURE);
    let (status, stdout, stderr) = run(
        &["validate", "--xsd", schema.path(), "-"],
        b"<root>hi</root>",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));
}

#[test]
fn dtd_happy_path_and_validation_failure() {
    let dtd = TempFile::new("validate-dtd", "dtd", DTD_FIXTURE);
    let valid = TempFile::new("validate-dtd-valid", "xml", "<root>hello</root>");
    let (status, stdout, stderr) = run_no_stdin(&["validate", "--dtd", dtd.path(), valid.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));

    let invalid = TempFile::new("validate-dtd-invalid", "xml", "<root><child/></root>");
    let (status, stdout, stderr) = run_no_stdin(&["validate", "--dtd", dtd.path(), invalid.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("error"));
}

#[test]
fn rng_happy_path_and_validation_failure() {
    let schema = TempFile::new("validate-rng", "rng", RNG_FIXTURE);
    let valid = TempFile::new("validate-rng-valid", "xml", "<root>hello</root>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--rng", schema.path(), valid.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));

    let invalid = TempFile::new("validate-rng-invalid", "xml", "<root><child/></root>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--rng", schema.path(), invalid.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("error"));
}

#[test]
fn rnc_happy_path_and_validation_failure() {
    let schema = TempFile::new("validate-rnc", "rnc", RNC_FIXTURE);
    let valid = TempFile::new("validate-rnc-valid", "xml", "<root>hello</root>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--rnc", schema.path(), valid.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));

    let invalid = TempFile::new("validate-rnc-invalid", "xml", "<other/>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--rnc", schema.path(), invalid.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("error"));
}

#[test]
fn schematron_happy_path_and_validation_failure() {
    let schema = TempFile::new("validate-sch", "sch", SCHEMATRON_FIXTURE);
    let valid = TempFile::new("validate-sch-valid", "xml", "<root><item/></root>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--schematron", schema.path(), valid.path()]);
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.contains("valid"));

    let invalid = TempFile::new("validate-sch-invalid", "xml", "<root/>");
    let (status, stdout, stderr) =
        run_no_stdin(&["validate", "--schematron", schema.path(), invalid.path()]);
    assert_eq!(status, 3, "{stderr}");
    assert!(stdout.contains("a root must have an item"), "{stdout}");
}

#[test]
fn no_schema_kind_flag_is_a_usage_error() {
    let (status, stdout, stderr) = run_no_stdin(&["validate", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("--xsd"));
}

#[test]
fn two_schema_kind_flags_together_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["validate", "--xsd", "a.xsd", "--dtd", "a.dtd", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("only one"));
}

#[test]
fn phase_without_schematron_is_a_usage_error() {
    let (status, _stdout, stderr) =
        run_no_stdin(&["validate", "--xsd", "a.xsd", "--phase", "p1", "a.xml"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("--phase"));
}

#[test]
fn missing_input_path_is_a_usage_error() {
    let (status, _stdout, stderr) = run_no_stdin(&["validate", "--xsd", "a.xsd"]);
    assert_eq!(status, 1);
    assert!(stderr.contains("input path"));
}

#[test]
fn a_missing_schema_file_is_a_processing_error() {
    let instance = TempFile::new("validate-missing-schema", "xml", "<root/>");
    let (status, stdout, stderr) = run_no_stdin(&[
        "validate",
        "--xsd",
        "/no/such/oxixml-cli-validate.xsd",
        instance.path(),
    ]);
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(!stderr.is_empty());
}
