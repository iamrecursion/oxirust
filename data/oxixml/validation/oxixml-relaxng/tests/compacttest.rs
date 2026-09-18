//! The compact-syntax test suite shipped with [jing-trang][jt]
//! (`mod/rng-schema/test/compacttest.xml`).
//!
//! Each test case pairs a compact-syntax schema with the XML syntax it is
//! supposed to translate to, and marks each of the two as `<correct>` or
//! `<incorrect>`. Both halves are compiled here, so the compact parser and the
//! XML parser are held to the same standard, and the cases that are deliberately
//! malformed have to be rejected by both.
//!
//! The suite is discovered through the `OXIXML_XML_SUITES_DIR` environment
//! variable and the test **skips silently** when it is not set.
//!
//! [jt]: https://github.com/relaxng/jing-trang

use std::path::PathBuf;

use oxixml_dom::{Document, NodeId, NodeKind, SerializeOptions, build};
use oxixml_relaxng::RelaxNg;

fn compacttest_path() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let candidate = root
        .join("jing-trang")
        .join("mod")
        .join("rng-schema")
        .join("test")
        .join("compacttest.xml");
    candidate.is_file().then_some(candidate)
}

#[test]
fn relax_ng_compact_syntax_test_suite() {
    let Some(path) = compacttest_path() else {
        eprintln!("skipping: OXIXML_XML_SUITES_DIR is not set or has no jing-trang suite");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("the compact suite is readable");
    let options = oxixml_dom::BuilderOptions::new().with_cdata_as_text(true);
    let document =
        build::parse_str_with(&text, &options).expect("the compact suite is well-formed XML");
    let root = document
        .document_element()
        .expect("the compact suite has a document element");

    let mut rejected_passed = 0_usize;
    let mut rejected_failed = 0_usize;
    let mut agreed = 0_usize;
    let mut disagreed = 0_usize;
    let mut failures: Vec<String> = Vec::new();

    for (index, case) in document
        .children(root)
        .filter(|child| {
            document
                .element(*child)
                .is_some_and(|data| data.local_name() == "testCase")
        })
        .enumerate()
    {
        let compact = extract(&document, case, "compact");
        let xml = extract(&document, case, "xml");

        // `compacttest.xml` belongs to the *syntax* module of jing-trang, so its
        // `<correct>` means "matches the grammar", not "is a correct RELAX NG
        // schema": several cases are syntactically fine but violate §7 or refer
        // to definitions and resources that do not exist. An `<incorrect>` case
        // must therefore be rejected outright, while a `<correct>` case is held
        // to the property that both syntaxes reach the same verdict.
        for (label, entry) in [("compact", &compact), ("xml", &xml)] {
            let Some((expect_correct, source)) = entry else {
                continue;
            };
            if *expect_correct {
                continue;
            }
            let result = compile(label, source);
            if result.is_err() {
                rejected_passed += 1;
            } else {
                rejected_failed += 1;
                if failures.len() < 30 {
                    failures.push(format!(
                        "case {}: {label} schema is incorrect but compiled\n{source}",
                        index + 1
                    ));
                }
            }
        }

        if let (Some((true, compact_source)), Some((true, xml_source))) = (&compact, &xml) {
            let from_compact = compile("compact", compact_source);
            let from_xml = compile("xml", xml_source);
            if from_compact.is_ok() == from_xml.is_ok() {
                agreed += 1;
            } else {
                disagreed += 1;
                if failures.len() < 30 {
                    failures.push(format!(
                        "case {}: the two syntaxes disagree\ncompact: {:?}\nxml: {:?}\n{compact_source}\n{xml_source}",
                        index + 1,
                        from_compact.err().map(|error| error.to_string()),
                        from_xml.err().map(|error| error.to_string()),
                    ));
                }
            }
        }
    }

    eprintln!(
        "RELAX NG compacttest: rejected-as-expected {}/{}, syntaxes agree {}/{}",
        rejected_passed,
        rejected_passed + rejected_failed,
        agreed,
        agreed + disagreed,
    );
    for failure in &failures {
        eprintln!("  {failure}");
    }

    assert!(agreed + disagreed > 40, "the suite is complete");
    assert_eq!(rejected_failed, 0, "malformed schemas were accepted");
    assert_eq!(disagreed, 0, "the two syntaxes disagree");
}

fn compile(label: &str, source: &str) -> Result<RelaxNg, oxixml_relaxng::SchemaError> {
    if label == "compact" {
        RelaxNg::from_compact(source)
    } else {
        RelaxNg::from_xml(source)
    }
}

/// Returns the `<correct>`/`<incorrect>` verdict and source of one half of a case.
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

fn first_element_source(document: &Document, node: NodeId) -> Option<String> {
    let element = document
        .children(node)
        .find(|child| document.kind(*child) == Some(NodeKind::Element))?;
    Some(document.node_to_xml(element, &SerializeOptions::new()))
}
