//! Checks against the W3C *XQuery/XPath Functions and Operators Test Suite* (QT3/FOTS), when it
//! is available.
//!
//! The suite is discovered through the environment variable `OXIXML_XML_SUITES_DIR`, which names
//! the directory the W3C suites were unpacked into; the QT3 directory beneath it is located by
//! searching rather than by a hard-coded name. When the variable is unset, or the suite is not
//! present, every test in this file **skips silently** so that the crate still builds and tests
//! green on a machine without the suites.
//!
//! Running the suite itself needs an XQuery processor, which this crate deliberately is not.
//! What *can* be checked here, and is:
//!
//! * that every function name this crate registers matches the name W3C uses for the
//!   corresponding test set — a real check, because the suite is the authority on spelling
//!   (`round-half-to-even`, `adjust-dateTime-to-timezone`, `local-name-from-QName`, …);
//! * that `fn:parse-xml` and the navigation functions cope with the suite's own catalogue, a
//!   real, namespace-using XML document of a few thousand nodes.

use oxixml_fn::{FnContext, nodes, registry, sequences, xmlio};
use oxixml_infoset::{Axis, NodeKind, Sequence, XdmNode, axis_nodes};
use std::path::{Path, PathBuf};

/// The root of the unpacked W3C suites, or `None` when the environment does not name one.
fn suites_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    root.is_dir().then_some(root)
}

/// The QT3/FOTS directory beneath `root`, found by searching so that a versioned directory name
/// is not hard-coded.
fn qt3_directory() -> Option<PathBuf> {
    let root = suites_root()?;
    let direct = root.join("qt3tests");
    if direct.join("catalog.xml").is_file() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(&root).ok()?;
    entries.flatten().map(|entry| entry.path()).find(|path| {
        path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("qt3"))
            && path.join("catalog.xml").is_file()
    })
}

/// Whether the suite names a test set `<area>/<local-name>.xml`.
fn has_test_set(qt3: &Path, area: &str, local_name: &str) -> bool {
    qt3.join(area).join(format!("{local_name}.xml")).is_file()
}

/// `fn:` names this crate legitimately registers that QT3/FOTS does not test.
///
/// QT3/FOTS is the *XPath and XQuery* Functions and Operators suite: it is the spelling
/// authority for the F&O function library proper, but `fn:copy-of` and `fn:snapshot` are not
/// F&O functions at all — [XSLT 3.0 §18.9](https://www.w3.org/TR/xslt-30/#func-copy-of) adds
/// them to the `fn:` namespace *for XSLT stylesheets specifically*, so no QT3 test set names
/// them. They are registered here anyway (rather than in `oxixml-xslt`) because the copying
/// itself is a pure function of the node it is given, needing nothing a transformation
/// supplies — see `oxixml_fn::nodes::copy_of`/`snapshot` — even though the zero-argument,
/// context-item-defaulting form has to complete the registration inside `oxixml-xslt`, the only
/// host this processor gives these two functions to.
const XSLT_ONLY_FN_NAMES: &[&str] = &["copy-of", "snapshot"];

#[test]
fn registered_function_names_match_the_test_suite() {
    let Some(qt3) = qt3_directory() else {
        eprintln!("skipping: OXIXML_XML_SUITES_DIR does not name a QT3 test suite");
        return;
    };

    let catalogue = registry::builtins();
    let mut unknown = Vec::new();
    for (name, _arity) in catalogue.names() {
        let Some(namespace) = name.namespace() else {
            continue;
        };
        let area = match namespace {
            registry::FN_NAMESPACE => "fn",
            oxixml_fn::maps::MAP_NAMESPACE => "map",
            oxixml_fn::arrays::ARRAY_NAMESPACE => "array",
            _ => continue,
        };
        if area == "fn" && XSLT_ONLY_FN_NAMES.contains(&name.local_name()) {
            continue;
        }
        if !has_test_set(&qt3, area, name.local_name()) {
            unknown.push(format!("{area}:{}", name.local_name()));
        }
    }
    unknown.sort();
    unknown.dedup();
    assert!(
        unknown.is_empty(),
        "these registered names have no matching QT3 test set, which means the spelling is \
         wrong: {unknown:?}"
    );
}

#[test]
fn the_test_suite_catalogue_parses_and_navigates() -> Result<(), Box<dyn std::error::Error>> {
    let Some(qt3) = qt3_directory() else {
        eprintln!("skipping: OXIXML_XML_SUITES_DIR does not name a QT3 test suite");
        return Ok(());
    };

    let xml = std::fs::read_to_string(qt3.join("catalog.xml"))?;
    let document = xmlio::parse_xml(&xml, Some("http://www.w3.org/2010/09/qt-fots-catalog"))?;
    let catalog = document
        .children()
        .into_iter()
        .find(|node| node.node_kind() == NodeKind::Element)
        .ok_or("no document element")?;

    assert_eq!(nodes::local_name(&catalog), "catalog");
    assert_eq!(
        nodes::namespace_uri(&catalog),
        "http://www.w3.org/2010/09/qt-fots-catalog"
    );

    let test_sets: Vec<_> = axis_nodes(&catalog, Axis::Child)
        .into_iter()
        .filter(|node| node.node_kind() == NodeKind::Element)
        .filter(|node| nodes::local_name(node) == "test-set")
        .collect();
    assert!(
        test_sets.len() > 100,
        "expected the FOTS catalogue to list many test sets, found {}",
        test_sets.len()
    );

    // Every test-set element carries a name, which exercises attribute access.
    let mut names = Sequence::empty();
    for test_set in &test_sets {
        let name = test_set
            .attributes()
            .into_iter()
            .find(|attribute| nodes::local_name(attribute) == "name")
            .ok_or("a test-set has no name attribute")?;
        names.push(name);
    }

    // The sequence functions work over a real document of a few thousand nodes, and the test-set
    // names really are all distinct.
    let context = FnContext::new();
    let everything = Sequence::from_iter(axis_nodes(&catalog, Axis::DescendantOrSelf));
    assert!(sequences::count(&everything) > 500);
    assert_eq!(
        sequences::count(&sequences::distinct_values(&context, &names, None)?),
        i64::try_from(test_sets.len()).unwrap_or(i64::MAX)
    );
    Ok(())
}

#[test]
fn the_oxigraph_testsuite_directory_is_optional() {
    // The RDF test suite is not used by this crate; the check exists so that a misconfigured
    // environment variable is visible rather than silently ignored.
    match std::env::var_os("OXIXML_TESTSUITE_DIR") {
        None => eprintln!("skipping: OXIXML_TESTSUITE_DIR is not set"),
        Some(path) => {
            let path = PathBuf::from(path);
            assert!(
                path.is_dir(),
                "OXIXML_TESTSUITE_DIR names `{}`, which is not a directory",
                path.display()
            );
        }
    }
}
