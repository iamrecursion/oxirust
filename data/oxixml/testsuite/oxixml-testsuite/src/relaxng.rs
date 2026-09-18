//! RELAX NG conformance runner, over the two suites [jing-trang] ships.
//!
//! * **`relaxng/spectest`** — `mod/rng-validate/test/spectest.xml`, James Clark's own
//!   specification conformance suite: schemas marked `<correct>`/`<incorrect>`, and `<valid>`/
//!   `<invalid>` instance documents checked against every schema that compiles. Cases carrying
//!   `<resource>`/`<dir>` auxiliary files are materialized into a scratch directory so `include`
//!   and `externalRef` exercise the real filesystem resolver.
//! * **`relaxng/compacttest`** — `mod/rng-schema/test/compacttest.xml`: each case pairs a
//!   compact-syntax schema with the XML syntax it should translate to, both marked `<correct>`/
//!   `<incorrect>`. An `<incorrect>` half must be rejected by its own parser; a case where both
//!   halves are `<correct>` must have the compact and XML parsers agree on whether the *whole
//!   schema* (§7 constraints, name resolution) compiles, not just the grammar.
//!
//! Both are discovered under `$OXIXML_XML_SUITES_DIR/jing-trang` and skip (as one combined
//! "unavailable" pair of reports) when that layout is not found.
//!
//! [jing-trang]: https://github.com/relaxng/jing-trang

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oxixml_dom::{Document, NodeId, NodeKind, SerializeOptions, build};
use oxixml_relaxng::{RelaxNg, RelaxNgBuilder, SchemaError};

use crate::env;
use crate::report::SuiteReport;

/// Runs both RELAX NG suites.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    vec![run_spectest(), run_compacttest()]
}

fn jing_trang_root() -> Option<PathBuf> {
    let root = env::xml_suites_dir()?.join("jing-trang");
    root.is_dir().then_some(root)
}

fn unavailable(suite: &str, what: &str) -> SuiteReport {
    SuiteReport::unavailable(
        suite,
        match env::xml_suites_dir() {
            None => env::missing(env::XML_SUITES_DIR_VAR),
            Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, what),
        },
    )
}

// --- spectest --------------------------------------------------------------------------------

fn run_spectest() -> SuiteReport {
    const SUITE: &str = "relaxng/spectest";
    let Some(path) = jing_trang_root()
        .map(|root| {
            root.join("mod")
                .join("rng-validate")
                .join("test")
                .join("spectest.xml")
        })
        .filter(|p| p.is_file())
    else {
        return unavailable(SUITE, "jing-trang/mod/rng-validate/test/spectest.xml");
    };

    let mut report = SuiteReport::new(SUITE);
    let Ok(text) = std::fs::read_to_string(&path) else {
        report.record_skip("spectest.xml unreadable");
        return report;
    };
    let text = expand_internal_entities(&text);
    let Ok(document) = build::parse_str(&text) else {
        report.record_skip("spectest.xml did not parse as XML");
        return report;
    };
    let Some(root) = document.document_element() else {
        report.record_skip("spectest.xml has no document element");
        return report;
    };

    let scratch = std::env::temp_dir().join(format!(
        "oxixml-testsuite-relaxng-spectest-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    if std::fs::create_dir_all(&scratch).is_err() {
        report.record_skip("could not create a scratch directory");
        return report;
    }

    let mut case_index = 0usize;
    walk_spectest(&document, root, &scratch, &mut case_index, &mut report);
    let _ = std::fs::remove_dir_all(&scratch);
    report
}

fn walk_spectest(
    document: &Document,
    node: NodeId,
    scratch: &Path,
    case_index: &mut usize,
    report: &mut SuiteReport,
) {
    for child in document.children(node) {
        let Some(data) = document.element(child) else {
            continue;
        };
        match data.local_name() {
            "testSuite" => walk_spectest(document, child, scratch, case_index, report),
            "testCase" => {
                *case_index += 1;
                run_spectest_case(document, child, scratch, *case_index, report);
            }
            _ => {}
        }
    }
}

fn run_spectest_case(
    document: &Document,
    case: NodeId,
    scratch: &Path,
    index: usize,
    report: &mut SuiteReport,
) {
    let mut resources = BTreeMap::new();
    collect_resources(document, case, "", &mut resources);

    let directory = scratch.join(format!("case{index:04}"));
    if std::fs::create_dir_all(&directory).is_err() {
        report.record_skip(format!(
            "case {index}: could not create its scratch directory"
        ));
        return;
    }
    for (name, contents) in &resources {
        let target = directory.join(name);
        if let Some(parent) = target.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&target, contents);
    }

    let mut expect_correct = None;
    let mut schema_text = None;
    for child in document.children(case) {
        let Some(data) = document.element(child) else {
            continue;
        };
        if matches!(data.local_name(), "correct" | "incorrect") {
            expect_correct = Some(data.local_name() == "correct");
            schema_text = first_element_source(document, child);
        }
    }
    let (Some(expect_correct), Some(schema_text)) = (expect_correct, schema_text) else {
        report.record_skip(format!("case {index}: no <correct>/<incorrect> schema"));
        return;
    };

    let base_uri = file_uri(&directory.join("schema.rng"));
    let compiled = RelaxNgBuilder::new()
        .base_uri(base_uri)
        .compile_xml(&schema_text);
    let id = format!("spec-{index:04}");
    if (compiled.is_ok()) != expect_correct {
        let outcome = match &compiled {
            Ok(_) => "compiled".to_owned(),
            Err(error) => format!("rejected: {error}"),
        };
        let expectation = if expect_correct {
            "correct"
        } else {
            "incorrect"
        };
        report.record_fail(
            format!("{id}/schema"),
            format!("expected {expectation}, {outcome}"),
        );
        return;
    }
    report.record_pass();

    let Ok(schema) = compiled else {
        return; // correctly rejected, and there is nothing further to check for an <incorrect> case
    };
    if !expect_correct {
        return;
    }

    for (position, child) in document.children(case).enumerate() {
        let Some(data) = document.element(child) else {
            continue;
        };
        let expect_valid = match data.local_name() {
            "valid" => true,
            "invalid" => false,
            _ => continue,
        };
        let Some(instance) = first_element_source(document, child) else {
            continue;
        };
        let result = schema.validate(&instance);
        if result.is_ok() == expect_valid {
            report.record_pass();
        } else {
            let outcome = match &result {
                Ok(()) => "accepted".to_owned(),
                Err(errors) => format!(
                    "rejected: {}",
                    errors
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            };
            let expectation = if expect_valid { "valid" } else { "invalid" };
            report.record_fail(
                format!("{id}/instance-{position}"),
                format!("expected {expectation}, {outcome}"),
            );
        }
    }
}

/// Collects `<resource name="…">` and `<dir name="…">` into a path/contents map.
fn collect_resources(
    document: &Document,
    node: NodeId,
    prefix: &str,
    out: &mut BTreeMap<String, String>,
) {
    for child in document.children(node) {
        let Some(data) = document.element(child) else {
            continue;
        };
        match data.local_name() {
            "resource" => {
                let Some(name) = document.get_attribute(child, "name") else {
                    continue;
                };
                let path = if prefix.is_empty() {
                    name.to_owned()
                } else {
                    format!("{prefix}/{name}")
                };
                if let Some(source) = first_element_source(document, child) {
                    out.insert(path, source);
                }
            }
            "dir" => {
                let Some(name) = document.get_attribute(child, "name") else {
                    continue;
                };
                let nested = if prefix.is_empty() {
                    name.to_owned()
                } else {
                    format!("{prefix}/{name}")
                };
                collect_resources(document, child, &nested, out);
            }
            _ => {}
        }
    }
}

/// Substitutes the general entities the internal DTD subset declares and drops the subset —
/// `spectest.xml` declares `dii` as a *markup* replacement text, which only a validating
/// processor would otherwise expand.
fn expand_internal_entities(text: &str) -> String {
    let Some(open) = text.find('[') else {
        return text.to_owned();
    };
    let Some(close) = text[open..].find("]>").map(|offset| open + offset) else {
        return text.to_owned();
    };
    let subset = &text[open + 1..close];
    let mut body = text[close + 2..].to_owned();
    let mut rest = subset;
    while let Some(start) = rest.find("<!ENTITY") {
        let declaration = rest[start + "<!ENTITY".len()..].trim_start();
        let name_end = declaration
            .find(|c: char| c.is_whitespace())
            .unwrap_or(declaration.len());
        let (name, after_name) = declaration.split_at(name_end);
        let after_name = after_name.trim_start();
        let Some(quote) = after_name
            .chars()
            .next()
            .filter(|c| *c == '"' || *c == '\'')
        else {
            rest = declaration;
            continue;
        };
        let literal = &after_name[quote.len_utf8()..];
        let Some(end) = literal.find(quote) else {
            break;
        };
        let (value, tail) = literal.split_at(end);
        rest = tail;
        if !name.is_empty() {
            body = body.replace(&format!("&{name};"), &expand_character_references(value));
        }
    }
    body
}

fn expand_character_references(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("&#") {
        out.push_str(&rest[..start]);
        let body = &rest[start + 2..];
        let Some(end) = body.find(';') else {
            out.push_str(&rest[start..]);
            return out;
        };
        let (digits, tail) = body.split_at(end);
        let code = match digits.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => digits.parse::<u32>().ok(),
        };
        match code.and_then(char::from_u32) {
            Some(c) => out.push(c),
            None => out.push_str(&rest[start..start + 2 + end + 1]),
        }
        rest = &tail[1..];
    }
    out.push_str(rest);
    out
}

fn file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' | b':' => {
                uri.push(char::from(byte))
            }
            other => {
                uri.push('%');
                uri.push(char::from(b"0123456789ABCDEF"[usize::from(other >> 4)]));
                uri.push(char::from(b"0123456789ABCDEF"[usize::from(other & 0x0f)]));
            }
        }
    }
    uri
}

fn first_element_source(document: &Document, node: NodeId) -> Option<String> {
    let element = document
        .children(node)
        .find(|child| document.kind(*child) == Some(NodeKind::Element))?;
    Some(document.node_to_xml(element, &SerializeOptions::new()))
}

// --- compacttest -----------------------------------------------------------------------------

fn run_compacttest() -> SuiteReport {
    const SUITE: &str = "relaxng/compacttest";
    let Some(path) = jing_trang_root()
        .map(|root| {
            root.join("mod")
                .join("rng-schema")
                .join("test")
                .join("compacttest.xml")
        })
        .filter(|p| p.is_file())
    else {
        return unavailable(SUITE, "jing-trang/mod/rng-schema/test/compacttest.xml");
    };

    let mut report = SuiteReport::new(SUITE);
    let Ok(text) = std::fs::read_to_string(&path) else {
        report.record_skip("compacttest.xml unreadable");
        return report;
    };
    let options = oxixml_dom::BuilderOptions::new().with_cdata_as_text(true);
    let Ok(document) = build::parse_str_with(&text, &options) else {
        report.record_skip("compacttest.xml did not parse as XML");
        return report;
    };
    let Some(root) = document.document_element() else {
        report.record_skip("compacttest.xml has no document element");
        return report;
    };

    for (index, case) in document
        .children(root)
        .filter(|child| {
            document
                .element(*child)
                .is_some_and(|data| data.local_name() == "testCase")
        })
        .enumerate()
    {
        run_compacttest_case(&document, case, index, &mut report);
    }
    report
}

fn run_compacttest_case(document: &Document, case: NodeId, index: usize, report: &mut SuiteReport) {
    let compact = extract(document, case, "compact");
    let xml = extract(document, case, "xml");

    for (label, entry) in [("compact", &compact), ("xml", &xml)] {
        let Some((expect_correct, source)) = entry else {
            continue;
        };
        if *expect_correct {
            continue;
        }
        let id = format!("compact-{index:04}/{label}");
        if compile_half(label, source).is_err() {
            report.record_pass();
        } else {
            report.record_fail(&id, "an incorrect schema compiled");
        }
    }

    if let (Some((true, compact_source)), Some((true, xml_source))) = (&compact, &xml) {
        let from_compact = compile_half("compact", compact_source);
        let from_xml = compile_half("xml", xml_source);
        if from_compact.is_ok() == from_xml.is_ok() {
            report.record_pass();
        } else {
            report.record_fail(
                format!("compact-{index:04}/agreement"),
                format!(
                    "the two syntaxes disagree: compact={:?} xml={:?}",
                    from_compact.err().map(|e| e.to_string()),
                    from_xml.err().map(|e| e.to_string())
                ),
            );
        }
    }
}

fn compile_half(label: &str, source: &str) -> Result<RelaxNg, SchemaError> {
    if label == "compact" {
        RelaxNg::from_compact(source)
    } else {
        RelaxNg::from_xml(source)
    }
}

/// Returns the `<correct>`/`<incorrect>` verdict and source of one half (`compact` or `xml`) of
/// a `compacttest.xml` case.
fn extract(document: &Document, case: NodeId, wrapper: &str) -> Option<(bool, String)> {
    let wrapper = document.children(case).find(|child| {
        document
            .element(*child)
            .is_some_and(|data| data.local_name() == wrapper)
    })?;
    let is_compact = document
        .element(wrapper)
        .is_some_and(|data| data.local_name() == "compact");
    for verdict in document.children(wrapper) {
        let Some(data) = document.element(verdict) else {
            continue;
        };
        let expect_correct = match data.local_name() {
            "correct" => true,
            "incorrect" => false,
            _ => continue,
        };
        let source = if is_compact {
            document.text_content(verdict).unwrap_or_default()
        } else {
            first_element_source(document, verdict)?
        };
        return Some((expect_correct, source));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_internal_entities_substitutes_markup_replacement_text() {
        let text = "<!DOCTYPE r [<!ENTITY dii \"<span/>\">]><r>&dii;</r>";
        let expanded = expand_internal_entities(text);
        assert_eq!(expanded, "<r><span/></r>");
    }

    #[test]
    fn expand_internal_entities_is_a_no_op_without_a_doctype() {
        assert_eq!(expand_internal_entities("<r/>"), "<r/>");
    }

    #[test]
    fn expand_character_references_handles_decimal_and_hex() {
        assert_eq!(expand_character_references("&#65;&#x42;"), "AB");
        assert_eq!(expand_character_references("plain"), "plain");
    }

    #[test]
    fn file_uri_percent_encodes_reserved_bytes() {
        let uri = file_uri(Path::new("/tmp/a b.rng"));
        assert!(uri.starts_with("file:///tmp/a%20b.rng"));
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| !report.is_available()));
    }
}
