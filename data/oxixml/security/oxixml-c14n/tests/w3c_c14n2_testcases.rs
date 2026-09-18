//! The W3C [Canonical XML 2.0 test cases][tc].
//!
//! The suite pairs 21 input documents with 8 parameter documents and records the
//! expected octets of every combination in a file named
//! `out_<input>_<parameters>.xml`. This test discovers the suite through the
//! `OXIXML_XML_SUITES_DIR` environment variable and skips silently when it is
//! not set, so that a checkout without the suites still builds green.
//!
//! # Reading the parameter documents
//!
//! The `Algorithm`, `PrefixRewrite` and `QNameAware` parameters are read
//! straight out of the `<dsig:CanonicalizationMethod>` documents with
//! [`C14n2Parameters::from_method_element`]. `IgnoreComments` and
//! `TrimTextNodes` are not, for two reasons, both of which are properties of the
//! published suite rather than of this implementation:
//!
//! * the suite has no `c14nTrim.xml` at all, even though it records
//!   `out_*_c14nTrim.xml` outputs; and
//! * `c14nComment.xml` sets `IgnoreComments` to `true` while its outputs are the
//!   ones *with* comments — the file predates the parameter being renamed from
//!   an "include" flag to an "ignore" flag.
//!
//! The mapping in [`comment_and_trim`] therefore restates what each parameter
//! document is documented to mean in [the suite's own prose][tc], and the
//! outputs are then reproduced byte for byte.
//!
//! [tc]: https://www.w3.org/TR/xml-c14n2-testcases/

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use common::{apply_attribute_default, apply_tokenized_type, expand_entity_reference};
use oxixml_c14n::{C14n2Parameters, C14n2Subset, c14n_2_0};
use oxixml_dom::{Document, build};

/// Locates `xml-c14n2-testcases/files` under the configured suite root.
fn suite_directory() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let files = root.join("xml-c14n2-testcases").join("files");
    files.is_dir().then_some(files)
}

/// The `IgnoreComments` and `TrimTextNodes` values each parameter document
/// stands for, per the suite's own description of it.
fn comment_and_trim(parameters: &str) -> (bool, bool) {
    match parameters {
        // "Canonicalization with comment removal" — the outputs are the ones
        // labelled "After c14n with Comments", so comments are kept.
        "c14nComment" => (false, false),
        "c14nTrim" => (true, true),
        // Every other document leaves both at the draft defaults the outputs
        // were generated with.
        _ => (true, false),
    }
}

/// Applies the declarations the DTD of an input document makes, which a
/// validating processor would have applied before canonicalization.
fn apply_dtd(document: &mut Document, input: &str) {
    match input {
        "inC14N3" => apply_attribute_default(document, "e9", "attr", "default"),
        "inC14N4" => {
            apply_tokenized_type(document, "normNames", "attr");
            apply_tokenized_type(document, "normId", "id");
        }
        "inC14N5" => expand_entity_reference(document, "ent2", "world"),
        _ => {}
    }
}

/// Splits `out_<input>_<parameters>.xml` into its two halves.
fn split_output_name(file_name: &str) -> Option<(String, String)> {
    let stem = file_name.strip_prefix("out_")?.strip_suffix(".xml")?;
    let separator = stem.find("_c14n")?;
    Some((
        stem[..separator].to_owned(),
        stem[separator + 1..].to_owned(),
    ))
}

fn read(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

#[test]
fn w3c_canonical_xml_2_0_test_cases() {
    let Some(directory) = suite_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xml-c14n2-testcases/files");
        return;
    };

    let mut entries: Vec<String> = fs::read_dir(&directory)
        .expect("the test case directory can be read")
        .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().into_owned()))
        .filter(|name| name.starts_with("out_"))
        .collect();
    entries.sort();

    let mut passed = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    // Parameter documents are shared by many cases; parse each of them once.
    let mut parameter_cache: BTreeMap<String, C14n2Parameters> = BTreeMap::new();

    for entry in &entries {
        let Some((input_name, parameter_name)) = split_output_name(entry) else {
            skipped.push(format!("{entry}: unrecognised file name"));
            continue;
        };
        let Some(input_text) = read(&directory.join(format!("{input_name}.xml"))) else {
            skipped.push(format!("{entry}: missing input {input_name}.xml"));
            continue;
        };
        let Some(expected) = fs::read(directory.join(entry)).ok() else {
            skipped.push(format!("{entry}: unreadable"));
            continue;
        };

        let parameters = parameter_cache
            .entry(parameter_name.clone())
            .or_insert_with(|| {
                let mut parameters = match read(&directory.join(format!("{parameter_name}.xml"))) {
                    Some(text) => {
                        let method = build::parse_str(&text)
                            .expect("a parameter document is well-formed XML");
                        let element = method
                            .document_element()
                            .expect("a parameter document has a document element");
                        C14n2Parameters::from_method_element(&method, element)
                            .expect("the parameter document is understood")
                    }
                    // `c14nTrim.xml` is missing from the published suite.
                    None => C14n2Parameters::new(),
                };
                let (ignore_comments, trim) = comment_and_trim(&parameter_name);
                parameters = parameters
                    .with_ignore_comments(ignore_comments)
                    .with_trim_text_nodes(trim);
                parameters
            })
            .clone();

        let mut document = match build::parse_str(&input_text) {
            Ok(document) => document,
            Err(error) => {
                failures.push(format!("{entry}: input did not parse: {error}"));
                continue;
            }
        };
        apply_dtd(&mut document, &input_name);

        match c14n_2_0(&document, &C14n2Subset::document(), &parameters) {
            Ok(actual) if actual == expected => passed += 1,
            Ok(actual) => failures.push(format!(
                "{entry}:\n  expected: {:?}\n  actual:   {:?}",
                String::from_utf8_lossy(&expected),
                String::from_utf8_lossy(&actual),
            )),
            Err(error) => failures.push(format!("{entry}: {error}")),
        }
    }

    println!(
        "W3C Canonical XML 2.0 test cases: {passed} passed, {} failed, {} skipped (of {} vectors)",
        failures.len(),
        skipped.len(),
        entries.len(),
    );
    for note in &skipped {
        println!("  skipped {note}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} vectors failed:\n{}",
        failures.len(),
        entries.len(),
        failures.join("\n")
    );
    assert!(passed > 0, "the suite was found but produced no results");
}
