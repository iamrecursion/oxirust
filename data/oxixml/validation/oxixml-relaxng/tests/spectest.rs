//! The RELAX NG specification test suite, as shipped with [jing-trang][jt].
//!
//! `spectest.xml` is James Clark's own conformance suite for the RELAX NG
//! specification: 385 test cases, each holding either an `<incorrect>` schema
//! that a conforming validator must reject, or a `<correct>` schema together with
//! `<valid>` and `<invalid>` instance documents. Several cases also carry
//! `<resource>` and `<dir>` elements describing auxiliary files that `include`
//! and `externalRef` must be able to retrieve; those are materialized into a
//! temporary directory so the real resolver is exercised.
//!
//! The suite is discovered through the `OXIXML_XML_SUITES_DIR` environment
//! variable and the test **skips silently** when it is not set.
//!
//! [jt]: https://github.com/relaxng/jing-trang

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oxixml_dom::{Document, NodeId, NodeKind, SerializeOptions, build};
use oxixml_relaxng::RelaxNgBuilder;

/// Locates `spectest.xml` under the suite root, if the environment allows it.
fn spectest_path() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let candidate = root
        .join("jing-trang")
        .join("mod")
        .join("rng-validate")
        .join("test")
        .join("spectest.xml");
    candidate.is_file().then_some(candidate)
}

#[derive(Debug, Default)]
struct Tally {
    passed: usize,
    failed: usize,
    failures: Vec<String>,
}

impl Tally {
    fn record(&mut self, ok: bool, describe: impl FnOnce() -> String) {
        if ok {
            self.passed += 1;
        } else {
            self.failed += 1;
            if self.failures.len() < 40 {
                self.failures.push(describe());
            }
        }
    }

    fn total(&self) -> usize {
        self.passed + self.failed
    }
}

#[derive(Debug, Default)]
struct Report {
    schemas: Tally,
    valid: Tally,
    invalid: Tally,
}

#[test]
fn relax_ng_specification_test_suite() {
    let Some(path) = spectest_path() else {
        eprintln!("skipping: OXIXML_XML_SUITES_DIR is not set or has no jing-trang suite");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("the spectest suite is readable");
    let text = expand_internal_entities(&text);
    let document = build::parse_str(&text).expect("the spectest suite is well-formed XML");
    let root = document
        .document_element()
        .expect("the spectest suite has a document element");

    let scratch =
        std::env::temp_dir().join(format!("oxixml-relaxng-spectest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).expect("a scratch directory can be created");

    let mut report = Report::default();
    let mut case_index = 0_usize;
    walk(&document, root, &scratch, &mut case_index, &mut report);
    let _ = std::fs::remove_dir_all(&scratch);

    eprintln!(
        "RELAX NG spectest: schemas {}/{}, valid instances {}/{}, invalid instances {}/{}",
        report.schemas.passed,
        report.schemas.total(),
        report.valid.passed,
        report.valid.total(),
        report.invalid.passed,
        report.invalid.total(),
    );
    for tally in [&report.schemas, &report.valid, &report.invalid] {
        for failure in &tally.failures {
            eprintln!("  {failure}");
        }
    }

    assert!(report.schemas.total() > 300, "the suite should be complete");

    // Every case must pass: the suite is the normative conformance test for the
    // specification this crate implements.
    assert_eq!(report.schemas.failed, 0, "incorrectly classified schemas");
    assert_eq!(report.valid.failed, 0, "valid instances rejected");
    assert_eq!(report.invalid.failed, 0, "invalid instances accepted");
}

fn walk(
    document: &Document,
    node: NodeId,
    scratch: &Path,
    case_index: &mut usize,
    report: &mut Report,
) {
    for child in document.children(node) {
        let Some(data) = document.element(child) else {
            continue;
        };
        match data.local_name() {
            "testSuite" => walk(document, child, scratch, case_index, report),
            "testCase" => {
                *case_index += 1;
                run_case(document, child, scratch, *case_index, report);
            }
            _ => {}
        }
    }
}

fn run_case(document: &Document, case: NodeId, scratch: &Path, index: usize, report: &mut Report) {
    let mut resources = BTreeMap::new();
    collect_resources(document, case, "", &mut resources);

    let directory = scratch.join(format!("case{index:04}"));
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    for (name, contents) in &resources {
        let target = directory.join(name);
        if let Some(parent) = target.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&target, contents);
    }

    let schema_path = directory.join("schema.rng");
    let base_uri = file_uri(&schema_path);

    let mut expect_correct = None;
    let mut schema_text = None;
    for child in document.children(case) {
        let Some(data) = document.element(child) else {
            continue;
        };
        match data.local_name() {
            "correct" | "incorrect" => {
                expect_correct = Some(data.local_name() == "correct");
                schema_text = first_element_source(document, child);
            }
            _ => {}
        }
    }
    let (Some(expect_correct), Some(schema_text)) = (expect_correct, schema_text) else {
        return;
    };

    let compiled = RelaxNgBuilder::new()
        .base_uri(base_uri)
        .compile_xml(&schema_text);

    report
        .schemas
        .record(compiled.is_ok() == expect_correct, || {
            let outcome = match &compiled {
                Ok(_) => "compiled".to_owned(),
                Err(error) => format!("rejected: {error}"),
            };
            let expectation = if expect_correct {
                "correct"
            } else {
                "incorrect"
            };
            format!("case {index}: schema is {expectation} but {outcome}\n{schema_text}")
        });

    let Ok(schema) = compiled else {
        return;
    };
    if !expect_correct {
        return;
    }

    for child in document.children(case) {
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
        let tally = if expect_valid {
            &mut report.valid
        } else {
            &mut report.invalid
        };
        tally.record(result.is_ok() == expect_valid, || {
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
            format!(
                "case {index}: instance is {expectation} but {outcome}\nschema:\n{schema_text}\ninstance:\n{instance}"
            )
        });
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

/// Serializes the first child element of `node` back to XML text.
fn first_element_source(document: &Document, node: NodeId) -> Option<String> {
    let element = document
        .children(node)
        .find(|child| document.kind(*child) == Some(NodeKind::Element))?;
    Some(document.node_to_xml(element, &SerializeOptions::new()))
}

/// Substitutes the general entities declared in the internal DTD subset and then
/// drops the subset.
///
/// `spectest.xml` declares `dii` as a *markup* replacement text
/// (`<!ENTITY dii "&lt;…/&gt;">`), which only a validating processor expands into
/// an element. Doing the substitution textually here keeps the harness
/// independent of how much of the DTD the tree builder processes.
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
        // The replacement text may itself contain `>`, so the declaration has to
        // be scanned rather than split on the first `>`.
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

/// Expands `&#…;` and `&#x…;` references, which XML resolves while the entity
/// value is being parsed rather than where the entity is referenced.
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
                uri.push(char::from(byte));
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
