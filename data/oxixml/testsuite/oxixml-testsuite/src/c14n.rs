//! W3C [Canonical XML 2.0 test cases][tc] runner.
//!
//! The suite pairs 21 input documents with 8 parameter documents and records the expected
//! octets of every combination in a file named `out_<input>_<parameters>.xml`. Each combination
//! is one case: canonicalize `<input>.xml` under the parameters `<parameters>.xml` describes
//! (via [`C14n2Parameters::from_method_element`]) and compare the result byte for byte against
//! `out_<input>_<parameters>.xml`.
//!
//! [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) canonicalizes the tree an XML
//! processor produces *after* applying the document type declaration's defaults — but
//! `oxixml-dom` is a non-validating builder, so three of the twenty-one inputs (`inC14N3`,
//! `inC14N4`, `inC14N5`) need the effect of a declaration applied by hand before
//! canonicalization sees the tree; `apply_dtd` stands in for the validating processor those
//! three examples assume, applying exactly the declaration each one's prose describes.
//!
//! [tc]: https://www.w3.org/TR/xml-c14n2-testcases/

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use oxixml_c14n::{C14n2Parameters, C14n2Subset, c14n_2_0};
use oxixml_dom::{Document, NodeId, NodeKind, build};

use crate::env;
use crate::report::SuiteReport;

/// The suite name reported for this runner.
pub const SUITE: &str = "c14n";

/// Runs the W3C Canonical XML 2.0 test cases.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(directory) = suite_directory() else {
        return vec![SuiteReport::unavailable(
            SUITE,
            match env::xml_suites_dir() {
                None => env::missing(env::XML_SUITES_DIR_VAR),
                Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "xml-c14n2-testcases/files"),
            },
        )];
    };

    let mut report = SuiteReport::new(SUITE);
    let Ok(read_dir) = fs::read_dir(&directory) else {
        report.record_skip("the test case directory could not be read");
        return vec![report];
    };
    let mut entries: Vec<String> = read_dir
        .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().into_owned()))
        .filter(|name| name.starts_with("out_"))
        .collect();
    entries.sort();

    // Parameter documents are shared by many cases; parse each of them once.
    let mut parameter_cache: BTreeMap<String, C14n2Parameters> = BTreeMap::new();

    for entry in &entries {
        run_case(&directory, entry, &mut parameter_cache, &mut report);
    }
    vec![report]
}

fn run_case(
    directory: &Path,
    entry: &str,
    parameter_cache: &mut BTreeMap<String, C14n2Parameters>,
    report: &mut SuiteReport,
) {
    let Some((input_name, parameter_name)) = split_output_name(entry) else {
        report.record_skip("unrecognised output file name");
        return;
    };
    let Some(input_text) = fs::read_to_string(directory.join(format!("{input_name}.xml"))).ok()
    else {
        report.record_skip(format!("missing input {input_name}.xml"));
        return;
    };
    let Ok(expected) = fs::read(directory.join(entry)) else {
        report.record_skip(format!("{entry} unreadable"));
        return;
    };

    let parameters = match parameter_cache.get(&parameter_name) {
        Some(parameters) => parameters.clone(),
        None => {
            let parameters = match build_parameters(directory, &parameter_name) {
                Ok(parameters) => parameters,
                Err(message) => {
                    report.record_fail(entry, message);
                    return;
                }
            };
            parameter_cache.insert(parameter_name.clone(), parameters.clone());
            parameters
        }
    };

    let mut document = match build::parse_str(&input_text) {
        Ok(document) => document,
        Err(error) => {
            report.record_fail(entry, format!("input did not parse: {error}"));
            return;
        }
    };
    apply_dtd(&mut document, &input_name);

    match c14n_2_0(&document, &C14n2Subset::document(), &parameters) {
        Ok(actual) if actual == expected => report.record_pass(),
        Ok(actual) => report.record_fail(
            entry,
            format!(
                "expected: {:?}\n  actual: {:?}",
                String::from_utf8_lossy(&expected),
                String::from_utf8_lossy(&actual),
            ),
        ),
        Err(error) => report.record_fail(entry, error.to_string()),
    }
}

/// Builds the [`C14n2Parameters`] a `<parameters>.xml` document describes, or the defaults for
/// `c14nTrim.xml`, which is documented by the suite but missing from the published distribution.
fn build_parameters(directory: &Path, parameter_name: &str) -> Result<C14n2Parameters, String> {
    let mut parameters = match fs::read_to_string(directory.join(format!("{parameter_name}.xml"))) {
        Ok(text) => {
            let method = build::parse_str(&text)
                .map_err(|error| format!("{parameter_name}.xml: {error}"))?;
            let element = method
                .document_element()
                .ok_or_else(|| format!("{parameter_name}.xml has no document element"))?;
            C14n2Parameters::from_method_element(&method, element)
                .map_err(|error| format!("{parameter_name}.xml: {error}"))?
        }
        Err(_) => C14n2Parameters::new(),
    };
    let (ignore_comments, trim) = comment_and_trim(parameter_name);
    parameters = parameters
        .with_ignore_comments(ignore_comments)
        .with_trim_text_nodes(trim);
    Ok(parameters)
}

/// Locates `xml-c14n2-testcases/files` under the configured suite root.
fn suite_directory() -> Option<PathBuf> {
    let files = env::xml_suites_dir()?
        .join("xml-c14n2-testcases")
        .join("files");
    files.is_dir().then_some(files)
}

/// The `IgnoreComments` and `TrimTextNodes` values each parameter document stands for, per the
/// suite's own description of it (see the crate-level doc comment on the divergent naming of
/// `c14nComment`).
fn comment_and_trim(parameters: &str) -> (bool, bool) {
    match parameters {
        "c14nComment" => (false, false),
        "c14nTrim" => (true, true),
        _ => (true, false),
    }
}

/// Applies the declarations the DTD of an input document makes, which a validating processor
/// would have applied before canonicalization; see the module documentation.
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

fn elements_named(document: &Document, name: &str) -> Vec<NodeId> {
    document
        .get_elements_by_tag_name(document.root(), name)
        .collect()
}

/// Applies an `<!ATTLIST element attribute … "value">` default declaration: every matching
/// element that does not specify the attribute gets it.
fn apply_attribute_default(
    document: &mut Document,
    element_name: &str,
    attribute_name: &str,
    value: &str,
) {
    for element in elements_named(document, element_name) {
        if document
            .get_attribute_node(element, attribute_name)
            .is_none()
        {
            let _ = document.set_attribute_ns(element, None, attribute_name, value);
        }
    }
}

/// Applies the tokenized-type attribute value normalization of [XML 1.0
/// §3.3.3](https://www.w3.org/TR/xml/#AVNormalize): leading/trailing `#x20` is discarded and
/// runs of `#x20` collapse to one. Only literal spaces count — a tab, CR or LF that survived as
/// a character reference is left alone.
fn apply_tokenized_type(document: &mut Document, element_name: &str, attribute_name: &str) {
    for element in elements_named(document, element_name) {
        let Some(attribute) = document.get_attribute_node(element, attribute_name) else {
            continue;
        };
        let Some(value) = document.attr(attribute).map(|data| data.value().to_owned()) else {
            continue;
        };
        let normalized = value
            .split(' ')
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let _ = document.set_node_value(attribute, normalized);
    }
}

/// Replaces every unexpanded `&name;` reference with the replacement text an external entity
/// would have supplied.
fn expand_entity_reference(document: &mut Document, name: &str, replacement: &str) {
    let references: Vec<NodeId> = (0..document.len())
        .map(|index| NodeId::from_index(u32::try_from(index).unwrap_or(u32::MAX)))
        .filter(|node| {
            document.kind(*node) == Some(NodeKind::EntityReference)
                && document.entity_reference(*node) == Some(name)
        })
        .collect();
    for reference in references {
        let Some(parent) = document.parent(reference) else {
            continue;
        };
        let Ok(text) = document.create_text_node(replacement) else {
            continue;
        };
        let _ = document.replace_child(parent, text, reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_output_name_separates_input_and_parameters() {
        assert_eq!(
            split_output_name("out_inC14N1_c14nInclusiveWithComments.xml"),
            Some(("inC14N1".to_owned(), "c14nInclusiveWithComments".to_owned()))
        );
        assert_eq!(split_output_name("not-an-output.xml"), None);
        assert_eq!(split_output_name("out_missing_suffix"), None);
    }

    #[test]
    fn comment_and_trim_matches_the_suites_documented_meaning() {
        assert_eq!(comment_and_trim("c14nComment"), (false, false));
        assert_eq!(comment_and_trim("c14nTrim"), (true, true));
        assert_eq!(comment_and_trim("c14nInclusiveWithComments"), (true, false));
    }

    #[test]
    fn apply_attribute_default_only_touches_elements_missing_the_attribute() {
        let mut document =
            oxixml_dom::build::parse_str("<root><e9/><e9 attr=\"explicit\"/></root>")
                .expect("well-formed");
        apply_attribute_default(&mut document, "e9", "attr", "default");
        let mut values = Vec::new();
        for element in elements_named(&document, "e9") {
            values.push(document.get_attribute(element, "attr").map(str::to_owned));
        }
        assert_eq!(
            values,
            vec![Some("default".to_owned()), Some("explicit".to_owned())]
        );
    }

    #[test]
    fn apply_tokenized_type_collapses_runs_of_literal_spaces() {
        let mut document =
            oxixml_dom::build::parse_str("<root><normNames attr=\"a   b  c\"/></root>")
                .expect("well-formed");
        apply_tokenized_type(&mut document, "normNames", "attr");
        let element = elements_named(&document, "normNames")[0];
        assert_eq!(document.get_attribute(element, "attr"), Some("a b c"));
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].is_available());
    }
}
