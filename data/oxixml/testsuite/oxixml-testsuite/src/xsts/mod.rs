//! [W3C XML Schema Test Suite][xsts] (`xsdtests`) runner.
//!
//! `suite.xml` is read, every `ts:testSetRef` it names is followed, and each `testGroup` runs
//!
//! * the **schema test** — assembling the listed schema documents with [`SchemaSet`] and
//!   comparing success against `<expected validity="…"/>`; and
//! * every **instance test** — validating the instance document against that schema, once it
//!   assembled.
//!
//! The suite is walked twice: once obeying `version="1.0"` selectors under [`XsdVersion::V1_0`]
//! (reported as `xsts/1.0`), and once obeying `version="1.1"` under [`XsdVersion::V1_1`]
//! (`xsts/1.1`). A `version` selector is written at three levels — on the `testSet` element, on a
//! `testGroup`, and on an individual `schemaTest`/`instanceTest` — and the innermost one in force
//! governs, so a group carrying none applies to whichever versions its test set
//! declares, and to both when its test set declares none either. A selector naming neither `"1.0"`
//! nor `"1.1"` (an optional-feature token such as `full-xpath-in-CTA`) applies to neither run.
//! Groups whose expected validity is `indeterminate` are skipped, exactly as the suite intends.
//!
//! [xsts]: https://www.w3.org/XML/2004/xml-schema-test-suite/

pub mod ledger;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_schema::{SchemaSet, XsdVersion};

use crate::env;
use crate::report::SuiteReport;

const TS_NS: &str = "http://www.w3.org/XML/2004/xml-schema-test-suite/";
const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

/// Runs both the XSD 1.0 and XSD 1.1 passes over the whole suite.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(root) = suites_root() else {
        let reason = match env::xml_suites_dir() {
            None => env::missing(env::XML_SUITES_DIR_VAR),
            Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "xsdtests*/suite.xml"),
        };
        return vec![
            SuiteReport::unavailable("xsts/1.0", reason.clone()),
            SuiteReport::unavailable("xsts/1.1", reason),
        ];
    };
    let suite_path = root.join("suite.xml");
    let Some(suite) = parse(&suite_path) else {
        let mut report = SuiteReport::new("xsts/1.0");
        report.record_skip(format!("could not read {}", suite_path.display()));
        return vec![
            report,
            SuiteReport::unavailable("xsts/1.1", "suite.xml could not be read"),
        ];
    };
    let Some(suite_root) = suite.document_element() else {
        return vec![
            SuiteReport::unavailable("xsts/1.0", "suite.xml has no document element"),
            SuiteReport::unavailable("xsts/1.1", "suite.xml has no document element"),
        ];
    };

    let mut report_1_0 = SuiteReport::new("xsts/1.0");
    let mut report_1_1 = SuiteReport::new("xsts/1.1");
    for reference in elements(&suite, suite_root) {
        if local_name(&suite, reference) != "testSetRef" {
            continue;
        }
        let Some(location) = href(&suite, reference) else {
            continue;
        };
        let path = resolve(&suite_path, &location);
        run_test_set(&path, RUN_1_0, &mut report_1_0);
        run_test_set(&path, RUN_1_1, &mut report_1_1);
    }
    vec![report_1_0, report_1_1]
}

fn suites_root() -> Option<PathBuf> {
    env::xml_suite_subdir("xsdtests", "suite.xml")
}

/// One pass over the suite: which `version` selector to obey, and which [`XsdVersion`] to
/// assemble schemas under.
#[derive(Debug, Clone, Copy)]
struct Run {
    selector: &'static str,
    version: XsdVersion,
}

const RUN_1_0: Run = Run {
    selector: "1.0",
    version: XsdVersion::V1_0,
};
const RUN_1_1: Run = Run {
    selector: "1.1",
    version: XsdVersion::V1_1,
};

/// Reads an XML document, honouring the encoding it announces — the suite ships a good number of
/// UTF-16 documents, so reading them as UTF-8 would report them as not well-formed and count a
/// harness failure that is not a conformance gap.
fn parse(path: &Path) -> Option<Document> {
    let bytes = std::fs::read(path).ok()?;
    let sniffed = oxixml_encoding::sniff_xml_encoding(&bytes);
    let mut text = String::with_capacity(bytes.len());
    let mut decoder = sniffed.best_effort_encoding().new_decoder();
    decoder
        .decode_to_string(bytes.get(sniffed.bom_len..)?, &mut text, true)
        .ok()?;
    oxixml_dom::build::parse_str(&text).ok()
}

fn elements(doc: &Document, node: NodeId) -> Vec<NodeId> {
    doc.children(node)
        .filter(|child| doc.kind(*child) == Some(NodeKind::Element))
        .collect()
}

fn local_name(doc: &Document, node: NodeId) -> String {
    doc.element(node)
        .map(|element| element.local_name().to_owned())
        .unwrap_or_default()
}

fn is_ts(doc: &Document, node: NodeId, name: &str) -> bool {
    match doc.element(node) {
        Some(element) => {
            element.local_name() == name && matches!(element.namespace_uri(), None | Some(TS_NS))
        }
        None => false,
    }
}

fn href(doc: &Document, node: NodeId) -> Option<String> {
    doc.get_attribute_ns(node, Some(XLINK_NS), "href")
        .or_else(|| doc.get_attribute_ns(node, None, "href"))
        .map(str::to_owned)
}

fn resolve(base: &Path, location: &str) -> PathBuf {
    let directory = base.parent().unwrap_or(Path::new(""));
    let mut out = PathBuf::new();
    for component in directory.join(location).components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The `validity` a group expects for `version`, or `None` when the group does not apply.
fn expectation(doc: &Document, node: NodeId, version: &str) -> Option<String> {
    let mut fallback = None;
    for child in elements(doc, node) {
        if !is_ts(doc, child, "expected") {
            continue;
        }
        let validity = doc.get_attribute_ns(child, None, "validity")?.to_owned();
        match doc.get_attribute_ns(child, None, "version") {
            Some(selector) if selector.split_whitespace().any(|token| token == version) => {
                return Some(validity);
            }
            Some(_) => {}
            None => fallback = Some(validity),
        }
    }
    fallback
}

/// The version selector in force at `node`: its own, or the one it inherits.
///
/// The suite writes `version` at three levels — on the `testSet` element, on a `testGroup`, and on
/// an individual `schemaTest`/`instanceTest` — and the inner one overrides rather than adds to the
/// outer. Reading only the innermost one is the difference between running a set and grading it
/// against the wrong version of the specification: the three XSD 1.1 contributions
/// (`saxonMeta`, `ibmMeta`, `oracleMeta`) declare `version="1.1"` on the `testSet` element and
/// leave most of their groups unmarked, so a runner that does not inherit assembles `xs:alternative`
/// and `notQName` under XSD 1.0 and records the refusals it asked for as failures.
fn selector<'a>(doc: &'a Document, node: NodeId, inherited: Option<&'a str>) -> Option<&'a str> {
    doc.get_attribute_ns(node, None, "version").or(inherited)
}

/// Whether a selector applies to `version`.
///
/// The suite writes the selector as a whitespace-separated list, and adds tokens such as
/// `full-xpath-in-CTA` that name an optional feature rather than a version; a group carrying only
/// such a token applies to neither run.
fn applies_to(selector: Option<&str>, version: &str) -> bool {
    match selector {
        None => true,
        Some(selector) => selector.split_whitespace().any(|token| token == version),
    }
}

/// Runs every `testGroup` of one test set (`path`) under `run`, folding the outcomes into
/// `report`. Groups are sharded across the available cores — the regular-expression test sets
/// alone hold several thousand — but each worker re-parses the (small) catalogue independently,
/// so nothing schema-shaped crosses a thread boundary.
fn run_test_set(path: &Path, run: Run, report: &mut SuiteReport) {
    let Some(total) = group_count(path) else {
        report.record_skip(format!("could not read test set {}", path.display()));
        return;
    };
    if total == 0 {
        return;
    }
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(total);
    let stripes: Vec<Stripe> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| scope.spawn(move || run_stripe(path, worker, workers, run)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_default())
            .collect()
    });
    for stripe in stripes {
        for _ in 0..stripe.passed {
            report.record_pass();
        }
        for (id, message) in stripe.failures {
            report.record_fail(id, message);
        }
        for _ in 0..stripe.skipped {
            report.record_skip("group or feature out of scope for this version");
        }
    }
}

fn group_count(path: &Path) -> Option<usize> {
    let doc = parse(path)?;
    let root = doc.document_element()?;
    Some(
        elements(&doc, root)
            .into_iter()
            .filter(|group| is_ts(&doc, *group, "testGroup"))
            .count(),
    )
}

/// One worker's partial tally over `groups[worker..].step_by(workers)`.
#[derive(Default)]
struct Stripe {
    passed: usize,
    skipped: usize,
    failures: Vec<(String, String)>,
}

fn run_stripe(path: &Path, offset: usize, stride: usize, run: Run) -> Stripe {
    let mut stripe = Stripe::default();
    let Some(doc) = parse(path) else {
        return stripe;
    };
    let Some(root) = doc.document_element() else {
        return stripe;
    };
    let set_name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("?")
        .to_owned();

    for (index, group) in elements(&doc, root)
        .into_iter()
        .filter(|group| is_ts(&doc, *group, "testGroup"))
        .enumerate()
    {
        if index % stride != offset {
            continue;
        }
        run_group(&doc, path, group, run, &set_name, &mut stripe);
    }
    stripe
}

fn run_group(
    doc: &Document,
    set_path: &Path,
    group: NodeId,
    run: Run,
    set_name: &str,
    stripe: &mut Stripe,
) {
    let group_name = doc
        .get_attribute_ns(group, None, "name")
        .unwrap_or("?")
        .to_owned();
    let id = format!("{set_name}/{group_name}");
    let inherited = doc
        .document_element()
        .and_then(|root| doc.get_attribute_ns(root, None, "version"));
    let group_selector = selector(doc, group, inherited);
    if !applies_to(group_selector, run.selector) {
        return;
    }

    let mut compiled = None;
    for child in elements(doc, group) {
        if !is_ts(doc, child, "schemaTest") {
            continue;
        }
        if !applies_to(selector(doc, child, group_selector), run.selector) {
            return;
        }
        let Some(expected) = expectation(doc, child, run.selector) else {
            return;
        };
        if expected == "indeterminate" {
            return;
        }
        let documents: Vec<PathBuf> = elements(doc, child)
            .into_iter()
            .filter(|member| is_ts(doc, *member, "schemaDocument"))
            .filter_map(|member| href(doc, member))
            .map(|location| resolve(set_path, &location))
            .collect();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut set = SchemaSet::new().with_version(run.version);
            for document in &documents {
                set.add_file(document)?;
            }
            set.compile()
        }));
        let (schema, reason) = match outcome {
            Ok(Ok(schema)) => (Some(schema), String::new()),
            Ok(Err(error)) => (None, error.to_string()),
            Err(_) => (None, "the schema compiler panicked".to_owned()),
        };
        if (expected == "valid") == schema.is_some() {
            stripe.passed += 1;
        } else {
            let outcome = if reason.is_empty() {
                "assembled without error".to_owned()
            } else {
                reason
            };
            stripe.failures.push((
                format!("{id}/schema"),
                format!("expected {expected}: {outcome}"),
            ));
        }
        compiled = schema;
        break;
    }

    let Some(schema) = compiled else { return };
    for (position, child) in elements(doc, group).into_iter().enumerate() {
        if !is_ts(doc, child, "instanceTest") {
            continue;
        }
        if !applies_to(selector(doc, child, group_selector), run.selector) {
            stripe.skipped += 1;
            continue;
        }
        let Some(expected) = expectation(doc, child, run.selector) else {
            stripe.skipped += 1;
            continue;
        };
        if expected == "indeterminate" {
            stripe.skipped += 1;
            continue;
        }
        let instance = elements(doc, child)
            .into_iter()
            .find(|member| is_ts(doc, *member, "instanceDocument"))
            .and_then(|member| href(doc, member))
            .map(|location| resolve(set_path, &location));
        let Some(instance) = instance else {
            stripe.skipped += 1;
            continue;
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| match parse(&instance) {
            Some(document) => {
                // The suite's instances carry `xsi:schemaLocation` and
                // `xsi:noNamespaceSchemaLocation` and expect them to be followed (XSD 1.1 Part 1
                // §4.3.2): several groups declare the document element — or the type an `xsi:type`
                // names — only in a document the `schemaTest` does not list. Supplying the
                // instance's own path is what those locations resolve against.
                let assessment =
                    schema.validate_document_at(&document, &instance.to_string_lossy());
                (
                    assessment.valid,
                    assessment
                        .first_error()
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                )
            }
            None => (false, "not well-formed".to_owned()),
        }));
        let (valid, reason) = outcome.unwrap_or((false, "the validator panicked".to_owned()));
        if (expected == "valid") == valid {
            stripe.passed += 1;
        } else {
            let outcome = if reason.is_empty() {
                "accepted".to_owned()
            } else {
                reason
            };
            stripe.failures.push((
                format!("{id}/instance-{position}"),
                format!("expected {expected}: {outcome}"),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_normalizes_parent_directory_segments() {
        let base = Path::new("/suite/dir/suite.xml");
        assert_eq!(
            resolve(base, "sets/a.xml"),
            PathBuf::from("/suite/dir/sets/a.xml")
        );
        assert_eq!(
            resolve(base, "../shared/a.xml"),
            PathBuf::from("/suite/shared/a.xml")
        );
    }

    /// The three XSD 1.1 contributions declare `version="1.1"` on the `testSet` element and leave
    /// most of their groups unmarked. A runner that reads only a group's own selector grades those
    /// groups under XSD 1.0, assembles `xs:alternative` and `notQName` as 1.0 syntax, and records
    /// the refusals the suite asked for as conformance failures — so the inheritance is the
    /// difference between measuring this processor and measuring the wrong specification.
    #[test]
    fn a_version_selector_is_inherited_until_something_overrides_it() {
        let doc = oxixml_dom::build::parse_str(
            r#"<testSet version="1.1">
                 <testGroup name="inherits"/>
                 <testGroup name="overrides" version="1.0"/>
                 <testGroup name="widens" version="1.0 1.1"/>
                 <testGroup name="neither" version="full-xpath-in-CTA"/>
               </testSet>"#,
        )
        .expect("well-formed fixture");
        let root = doc.document_element().expect("a document element");
        let inherited = doc.get_attribute_ns(root, None, "version");
        assert_eq!(inherited, Some("1.1"));
        let seen: Vec<(String, bool, bool)> = elements(&doc, root)
            .into_iter()
            .map(|group| {
                let group_selector = selector(&doc, group, inherited);
                (
                    doc.get_attribute_ns(group, None, "name")
                        .unwrap_or_default()
                        .to_owned(),
                    applies_to(group_selector, "1.0"),
                    applies_to(group_selector, "1.1"),
                )
            })
            .collect();
        assert_eq!(
            seen,
            vec![
                // The set is contributed for 1.1, so an unmarked group is 1.1 too…
                ("inherits".to_owned(), false, true),
                // …unless the group says otherwise, which replaces rather than adds…
                ("overrides".to_owned(), true, false),
                // …or names both.
                ("widens".to_owned(), true, true),
                // A selector naming an optional feature rather than a version applies to neither.
                ("neither".to_owned(), false, false),
            ]
        );
    }

    /// A test set that declares no version of its own leaves its unmarked groups in both passes,
    /// which is the majority of the suite and must not move.
    #[test]
    fn an_unversioned_set_leaves_its_groups_in_both_passes() {
        let doc = oxixml_dom::build::parse_str(r#"<testSet><testGroup name="both"/></testSet>"#)
            .expect("well-formed fixture");
        let root = doc.document_element().expect("a document element");
        let inherited = doc.get_attribute_ns(root, None, "version");
        let group = elements(&doc, root)[0];
        let group_selector = selector(&doc, group, inherited);
        assert!(applies_to(group_selector, "1.0"));
        assert!(applies_to(group_selector, "1.1"));
    }

    #[test]
    fn run_1_0_and_run_1_1_select_the_documented_versions() {
        assert_eq!(RUN_1_0.selector, "1.0");
        assert!(!RUN_1_0.version.is_1_1());
        assert_eq!(RUN_1_1.selector, "1.1");
        assert!(RUN_1_1.version.is_1_1());
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
