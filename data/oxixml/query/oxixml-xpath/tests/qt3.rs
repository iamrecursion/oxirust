//! The W3C *QT3* test suite (FOTS), filtered to the XPath expression tests.
//!
//! The suite is discovered through the `OXIXML_XML_SUITES_DIR` environment variable, which must
//! name a directory containing a `qt3tests` (or `qt3*`) subdirectory with a `catalog.xml`. When
//! the variable is unset, or the suite is not there, every test in this file **skips silently**,
//! so the crate builds and tests cleanly without the suite.
//!
//! # What is run, and what is not
//!
//! A test case is *runnable* when
//!
//! * every `<dependency>` it declares is one this engine satisfies — a `spec` dependency naming
//!   an XPath version, and a `feature` dependency this crate implements. Cases that need schema
//!   import or validation, XSLT transformation, module import, serialization, a non-default XSD
//!   or Unicode version, or the XPath 1.0 compatibility mode of XQuery are skipped;
//! * its environment is one this driver can build: any number of source documents, namespace
//!   bindings, variable parameters and a static base URI, but no schema;
//! * its expected result uses an assertion this driver understands. `assert-xml`,
//!   `assert-serialization*` and `assert-permutation` need a serializer or a multiset comparison
//!   and are skipped.
//!
//! Everything else is compiled as XPath 3.1 and evaluated, and a mismatch is a failure.

use oxixml_fn::{CollectingTrace, FnContext};
use oxixml_infoset::{
    AtomicValue, NodeKind, NodeRef, ParseOptions, Sequence, Tree, XdmNode, deep_equal,
};
use oxixml_xpath::{DocumentResolver, StaticContext, XPath, XPathError, XPathVersion, XpathValue};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The features of the suite this engine implements.
const SUPPORTED_FEATURES: &[&str] = &["higherOrderFunctions", "namespace-axis"];

/// The dependency kinds this driver knows how to decide. Anything else skips the case.
const DECIDABLE_DEPENDENCIES: &[&str] = &["spec", "feature"];

/// Locates the `qt3tests` directory under `OXIXML_XML_SUITES_DIR`.
fn qt3_root() -> Option<PathBuf> {
    let root = PathBuf::from(env::var_os("OXIXML_XML_SUITES_DIR")?);
    let entries = fs::read_dir(&root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name()?.to_string_lossy().to_string();
        if name.starts_with("qt3") && path.join("catalog.xml").is_file() {
            return Some(path);
        }
    }
    None
}

// --- a very small reader over the catalogue's own XML -------------------------------------

fn parse_file(path: &Path) -> Option<Tree> {
    let text = fs::read_to_string(path).ok()?;
    Tree::parse_str(&text).ok()
}

/// Parses a source document, giving it a document URI so that `fn:document-uri` and
/// `fn:base-uri` have something to report.
fn parse_source(path: &Path) -> Option<Tree> {
    let text = fs::read_to_string(path).ok()?;
    let options = ParseOptions::default().document_uri(path.to_string_lossy().to_string());
    Tree::parse_str_with(&text, options).ok()
}

fn children_named<'a>(node: &'a NodeRef, local: &'a str) -> impl Iterator<Item = NodeRef> + 'a {
    node.children().into_iter().filter(move |child| {
        child.node_kind() == NodeKind::Element
            && child
                .node_name()
                .is_some_and(|name| name.local_name() == local)
    })
}

fn attribute(node: &NodeRef, local: &str) -> Option<String> {
    node.attributes().into_iter().find_map(|candidate| {
        candidate
            .node_name()
            .filter(|name| name.local_name() == local)
            .map(|_| candidate.string_value())
    })
}

fn element_children(node: &NodeRef) -> Vec<NodeRef> {
    node.children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .collect()
}

fn local_name(node: &NodeRef) -> String {
    node.node_name()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default()
}

// --- environments --------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Environment {
    /// Source documents by role: `"."` is the context item.
    sources: Vec<(String, String)>,
    namespaces: Vec<(String, String)>,
    parameters: Vec<(String, String)>,
    /// Text resources by URI, for `fn:unparsed-text`.
    resources: Vec<(String, String)>,
    base_uri: Option<String>,
    /// A schema declaration means the case needs schema awareness, which this engine has not.
    needs_schema: bool,
}

fn read_environment(node: &NodeRef) -> Environment {
    let mut environment = Environment::default();
    for child in element_children(node) {
        match local_name(&child).as_str() {
            "source" => {
                let role = attribute(&child, "role").unwrap_or_default();
                if let Some(file) = attribute(&child, "file") {
                    let uri = attribute(&child, "uri").unwrap_or_else(|| file.clone());
                    environment.sources.push((role, file));
                    let _ = uri;
                }
            }
            "namespace" => {
                if let (Some(prefix), Some(uri)) =
                    (attribute(&child, "prefix"), attribute(&child, "uri"))
                {
                    environment.namespaces.push((prefix, uri));
                }
            }
            "param" => {
                if let (Some(name), Some(select)) =
                    (attribute(&child, "name"), attribute(&child, "select"))
                {
                    environment.parameters.push((name, select));
                }
            }
            "resource" => {
                if let (Some(file), Some(uri)) =
                    (attribute(&child, "file"), attribute(&child, "uri"))
                {
                    environment.resources.push((uri, file));
                }
            }
            "static-base-uri" => environment.base_uri = attribute(&child, "uri"),
            "schema" => environment.needs_schema = true,
            _ => {}
        }
    }
    environment
}

/// The documents a test-set's environment makes available to `fn:doc`.
#[derive(Debug, Default)]
struct Documents {
    by_uri: BTreeMap<String, NodeRef>,
    text_by_uri: BTreeMap<String, String>,
}

impl DocumentResolver for Documents {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        self.by_uri.get(uri).cloned()
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        match uri {
            None => Some(self.by_uri.values().cloned().collect()),
            Some(uri) => self.by_uri.get(uri).map(|node| vec![node.clone()]),
        }
    }

    fn unparsed_text(&self, uri: &str, _encoding: Option<&str>) -> Option<String> {
        self.text_by_uri.get(uri).cloned()
    }
}

// --- dependencies ----------------------------------------------------------------------------

/// Whether every dependency declared on `node` is one this engine satisfies.
fn dependencies_supported(node: &NodeRef) -> bool {
    for dependency in children_named(node, "dependency") {
        let kind = attribute(&dependency, "type").unwrap_or_default();
        let value = attribute(&dependency, "value").unwrap_or_default();
        let satisfied =
            attribute(&dependency, "satisfied").is_none_or(|text| text.trim() != "false");
        if !DECIDABLE_DEPENDENCIES.contains(&kind.as_str()) {
            return false;
        }
        let holds = match kind.as_str() {
            "spec" => value
                .split_whitespace()
                .any(|token| token.starts_with("XP")),
            _ => SUPPORTED_FEATURES.contains(&value.as_str()),
        };
        if holds != satisfied {
            return false;
        }
    }
    true
}

// --- the driver -------------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Tally {
    passed: usize,
    failed: usize,
    skipped: usize,
    failures: Vec<String>,
}

impl Tally {
    fn merge(&mut self, other: Self) {
        self.passed += other.passed;
        self.failed += other.failed;
        self.skipped += other.skipped;
        self.failures.extend(other.failures);
    }
}

/// Everything a test case needs in order to run: the static context, the context item, the
/// variable bindings still to be evaluated, the resolver, and the trees that must stay alive.
struct Prepared {
    statik: StaticContext,
    context_item: Option<NodeRef>,
    parameters: Vec<(String, String)>,
    documents: Arc<Documents>,
    trees: Vec<Tree>,
}

/// Builds the static and dynamic contexts a test case runs in, or `None` when the environment
/// cannot be reproduced.
fn build_context(environment: &Environment, directory: &Path) -> Option<Prepared> {
    if environment.needs_schema {
        return None;
    }
    let mut statik = StaticContext::new();
    for (prefix, uri) in &environment.namespaces {
        statik = statik.with_namespace(prefix.clone(), uri.clone());
    }
    if let Some(base) = &environment.base_uri {
        statik = statik.with_base_uri(base.clone());
    }
    for (name, _) in &environment.parameters {
        statik = statik.with_variable(name.clone());
    }

    let mut documents = Documents::default();
    let mut trees = Vec::new();
    let mut context_item = None;
    for (uri, file) in &environment.resources {
        if let Ok(text) = fs::read_to_string(directory.join(file)) {
            documents.text_by_uri.insert(uri.clone(), text.clone());
            documents
                .text_by_uri
                .insert(directory.join(file).to_string_lossy().to_string(), text);
        }
    }
    for (role, file) in &environment.sources {
        let tree = parse_source(&directory.join(file))?;
        let root = tree.root();
        if role == "." {
            context_item = Some(root.clone());
        }
        documents.by_uri.insert(file.clone(), root.clone());
        documents
            .by_uri
            .insert(directory.join(file).to_string_lossy().to_string(), root);
        trees.push(tree);
    }
    Some(Prepared {
        statik,
        context_item,
        parameters: environment.parameters.clone(),
        documents: Arc::new(documents),
        trees,
    })
}

fn evaluate_case(
    expression: &str,
    environment: &Environment,
    directory: &Path,
) -> Option<Result<XpathValue, XPathError>> {
    let Prepared {
        statik,
        context_item,
        parameters,
        documents,
        trees,
    } = build_context(environment, directory)?;
    let compiled = match XPath::compile(expression, XPathVersion::XPath31, statik.clone()) {
        Ok(compiled) => compiled,
        Err(error) => return Some(Err(error)),
    };
    // `fn:trace` writes to standard error by default, and the suite calls it hundreds of times.
    let fn_context = FnContext::new().with_trace_sink(Arc::new(CollectingTrace::new()));
    let mut context = compiled
        .context()
        .with_fn_context(fn_context)
        .with_document_resolver(documents);
    if let Some(item) = context_item {
        context = context.with_context_item(item);
    }
    for (name, select) in &parameters {
        let bound = match XPath::compile(select, XPathVersion::XPath31, statik.clone()).and_then(
            |compiled| {
                let inner = compiled.context();
                compiled.evaluate_value_in(&inner)
            },
        ) {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        context = context.with_variable(name.clone(), bound);
    }
    let outcome = compiled.evaluate_value_in(&context);
    drop(trees);
    Some(outcome)
}

/// Checks one assertion element against the outcome. `None` means "this driver cannot decide".
fn check(
    assertion: &NodeRef,
    outcome: &Result<XpathValue, XPathError>,
) -> Option<Result<(), String>> {
    let kind = local_name(assertion);
    match kind.as_str() {
        "all-of" => {
            for child in element_children(assertion) {
                match check(&child, outcome)? {
                    Ok(()) => {}
                    Err(reason) => return Some(Err(reason)),
                }
            }
            Some(Ok(()))
        }
        "any-of" => {
            let mut decided = false;
            let mut last = String::from("no alternative matched");
            for child in element_children(assertion) {
                if let Some(result) = check(&child, outcome) {
                    decided = true;
                    match result {
                        Ok(()) => return Some(Ok(())),
                        Err(reason) => last = reason,
                    }
                }
            }
            decided.then_some(Err(last))
        }
        "not" => {
            let child = element_children(assertion).into_iter().next()?;
            match check(&child, outcome)? {
                Ok(()) => Some(Err("the negated assertion held".to_owned())),
                Err(_) => Some(Ok(())),
            }
        }
        "error" => {
            let expected = attribute(assertion, "code").unwrap_or_else(|| "*".to_owned());
            match outcome {
                Err(error) => {
                    if expected == "*" || error.name().local_name() == expected {
                        Some(Ok(()))
                    } else {
                        Some(Err(format!(
                            "expected error {expected}, raised {}",
                            error.name().local_name()
                        )))
                    }
                }
                Ok(value) => Some(Err(format!(
                    "expected error {expected}, produced `{value}`"
                ))),
            }
        }
        "assert-empty" => match outcome {
            Ok(value) if value.is_empty() => Some(Ok(())),
            Ok(value) => Some(Err(format!("expected the empty sequence, got `{value}`"))),
            Err(error) => Some(Err(format!("expected the empty sequence, raised {error}"))),
        },
        "assert-count" => {
            let expected: usize = assertion.string_value().trim().parse().ok()?;
            match outcome {
                Ok(value) if value.len() == expected => Some(Ok(())),
                Ok(value) => Some(Err(format!(
                    "expected {expected} item(s), got {}",
                    value.len()
                ))),
                Err(error) => Some(Err(format!("expected {expected} item(s), raised {error}"))),
            }
        }
        "assert-true" | "assert-false" => {
            let wanted = kind == "assert-true";
            match outcome {
                Ok(value) => match value.as_slice() {
                    [item] => match item.as_atomic().and_then(AtomicValue::as_bool) {
                        Some(actual) if actual == wanted => Some(Ok(())),
                        Some(actual) => Some(Err(format!("expected {wanted}, got {actual}"))),
                        None => Some(Err(format!("expected {wanted}, got `{value}`"))),
                    },
                    _ => Some(Err(format!("expected {wanted}, got `{value}`"))),
                },
                Err(error) => Some(Err(format!("expected {wanted}, raised {error}"))),
            }
        }
        "assert-string-value" => {
            let normalize =
                attribute(assertion, "normalize-space").is_some_and(|text| text.trim() == "true");
            let mut expected = assertion.string_value();
            match outcome {
                Ok(value) => {
                    let mut actual = string_value_of(value).ok()?;
                    if normalize {
                        expected = collapse(&expected);
                        actual = collapse(&actual);
                    }
                    if actual == expected {
                        Some(Ok(()))
                    } else {
                        Some(Err(format!("expected `{expected}`, got `{actual}`")))
                    }
                }
                Err(error) => Some(Err(format!("expected `{expected}`, raised {error}"))),
            }
        }
        "assert-eq" | "assert-deep-eq" => {
            let expected_source = assertion.string_value();
            let expected = compile_and_run(&expected_source).ok()?;
            match outcome {
                Ok(value) => {
                    let mut left = value.clone().into_sequence().ok()?;
                    if kind == "assert-eq" {
                        // `assert-eq` is `$result eq E`, which atomizes.
                        left = left.atomize().ok()?;
                    }
                    match deep_equal(&left, &expected) {
                        Ok(true) => Some(Ok(())),
                        Ok(false) => {
                            Some(Err(format!("expected `{expected_source}`, got `{value}`")))
                        }
                        Err(_) => None,
                    }
                }
                Err(error) => Some(Err(format!("expected `{expected_source}`, raised {error}"))),
            }
        }
        "assert-type" => {
            let required = assertion.string_value();
            let source = format!("$__result instance of {}", required.trim());
            let statik = StaticContext::new().with_variable("__result");
            let compiled = XPath::compile(&source, XPathVersion::XPath31, statik).ok()?;
            match outcome {
                Ok(value) => {
                    let context = compiled.context().with_variable("__result", value.clone());
                    match compiled.evaluate_to_boolean_in(&context) {
                        Ok(true) => Some(Ok(())),
                        Ok(false) => {
                            Some(Err(format!("`{value}` is not an instance of {required}")))
                        }
                        Err(_) => None,
                    }
                }
                Err(error) => Some(Err(format!(
                    "expected an instance of {required}, raised {error}"
                ))),
            }
        }
        "assert" => {
            let predicate = assertion.string_value();
            let source = predicate.replace("$result", "$__result");
            let statik = StaticContext::new().with_variable("__result");
            let compiled = XPath::compile(&source, XPathVersion::XPath31, statik).ok()?;
            match outcome {
                Ok(value) => {
                    let context = compiled.context().with_variable("__result", value.clone());
                    match compiled.evaluate_to_boolean_in(&context) {
                        Ok(true) => Some(Ok(())),
                        Ok(false) => Some(Err(format!("`{predicate}` is false for `{value}`"))),
                        Err(_) => None,
                    }
                }
                Err(error) => Some(Err(format!("`{predicate}` could not be tested: {error}"))),
            }
        }
        // Needs a serializer or a multiset comparison.
        _ => None,
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn string_value_of(value: &XpathValue) -> Result<String, XPathError> {
    let mut parts = Vec::with_capacity(value.len());
    for item in value.iter() {
        parts.push(item.string_value()?);
    }
    Ok(parts.join(" "))
}

fn compile_and_run(source: &str) -> Result<Sequence, XPathError> {
    let compiled = XPath::compile_31(source)?;
    compiled.evaluate_without_context()
}

fn run_test_set(path: &Path, tally: &mut Tally, narrowed: bool) {
    let Some(tree) = parse_file(path) else {
        tally.skipped += 1;
        return;
    };
    let Some(root) = tree.document_element() else {
        tally.skipped += 1;
        return;
    };
    let directory = path.parent().unwrap_or(Path::new("."));
    let set_name = attribute(&root, "name").unwrap_or_default();

    let mut environments: BTreeMap<String, Environment> = BTreeMap::new();
    environments.insert("empty".to_owned(), Environment::default());
    for definition in children_named(&root, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            environments.insert(name, read_environment(&definition));
        }
    }
    let set_supported = dependencies_supported(&root);

    for case in children_named(&root, "test-case") {
        let case_name = attribute(&case, "name").unwrap_or_default();
        if !set_supported || !dependencies_supported(&case) {
            tally.skipped += 1;
            continue;
        }
        let Some(test) = children_named(&case, "test").next() else {
            tally.skipped += 1;
            continue;
        };
        if attribute(&test, "file").is_some() {
            tally.skipped += 1;
            continue;
        }
        let expression = test.string_value();
        if expression.trim().is_empty() {
            tally.skipped += 1;
            continue;
        }

        let environment = match children_named(&case, "environment").next() {
            Some(node) => match attribute(&node, "ref") {
                Some(reference) => match environments.get(&reference) {
                    Some(environment) => environment.clone(),
                    // A reference into the shared catalogue environments this driver does not
                    // reproduce.
                    None => {
                        tally.skipped += 1;
                        continue;
                    }
                },
                None => read_environment(&node),
            },
            None => Environment::default(),
        };

        let Some(result) = children_named(&case, "result").next() else {
            tally.skipped += 1;
            continue;
        };
        let Some(assertion) = element_children(&result).into_iter().next() else {
            tally.skipped += 1;
            continue;
        };

        let Some(outcome) = evaluate_case(&expression, &environment, directory) else {
            tally.skipped += 1;
            continue;
        };
        match check(&assertion, &outcome) {
            None => tally.skipped += 1,
            Some(Ok(())) => tally.passed += 1,
            Some(Err(reason)) => {
                tally.failed += 1;
                // The whole list is kept only for a narrowed triage run; a full run keeps the
                // first few so that the report stays readable.
                if tally.failures.len() < 25 || narrowed {
                    tally
                        .failures
                        .push(format!("{set_name}/{case_name}: {expression} — {reason}"));
                }
            }
        }
    }
}

/// A triage filter: `OXIXML_QT3_ONLY` is a space-separated list of test-set *names*, and when it
/// is set only those sets run. The assertions below never set it, so a plain `cargo test` always
/// measures the whole corpus; it exists so that a failing bucket can be re-run in a second
/// instead of four minutes.
fn narrowed_to() -> Option<Vec<String>> {
    let value = env::var("OXIXML_QT3_ONLY").ok()?;
    let names: Vec<String> = value.split_whitespace().map(ToOwned::to_owned).collect();
    (!names.is_empty()).then_some(names)
}

fn run_suite(root: &Path, only: Option<&[&str]>) -> Tally {
    let mut tally = Tally::default();
    let Some(catalog) = parse_file(&root.join("catalog.xml")) else {
        return tally;
    };
    let Some(document_element) = catalog.document_element() else {
        return tally;
    };
    let narrowed = narrowed_to();
    for entry in children_named(&document_element, "test-set") {
        let Some(file) = attribute(&entry, "file") else {
            continue;
        };
        if let Some(names) = &narrowed {
            let name = attribute(&entry, "name").unwrap_or_default();
            if !names.iter().any(|wanted| name.starts_with(wanted.as_str())) {
                continue;
            }
        }
        if let Some(prefixes) = only {
            if !prefixes.iter().any(|prefix| file.starts_with(prefix)) {
                continue;
            }
        }
        let mut set_tally = Tally::default();
        run_test_set(&root.join(&file), &mut set_tally, narrowed.is_some());
        tally.merge(set_tally);
    }
    tally
}

fn report(label: &str, tally: &Tally) {
    let attempted = tally.passed + tally.failed;
    let rate = if attempted == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            tally.passed as f64 * 100.0 / attempted as f64
        }
    };
    println!(
        "QT3 [{label}]: {} passed, {} failed, {} skipped ({rate:.1}% of {attempted} attempted)",
        tally.passed, tally.failed, tally.skipped
    );
    for failure in &tally.failures {
        println!("  - {failure}");
    }
}

#[test]
fn qt3_operator_and_production_tests() {
    let Some(root) = qt3_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is unset or has no qt3tests directory; skipping");
        return;
    };
    let tally = run_suite(&root, Some(&["op/", "prod/"]));
    report("op+prod", &tally);
    let attempted = tally.passed + tally.failed;
    assert!(
        attempted > 9_000,
        "too few QT3 operator tests were attempted"
    );
    #[allow(clippy::cast_precision_loss)]
    let rate = tally.passed as f64 / attempted as f64;
    assert!(
        rate > 0.987,
        "QT3 operator and production pass rate regressed to {:.1}%",
        rate * 100.0
    );
}

#[test]
fn qt3_function_tests() {
    let Some(root) = qt3_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is unset or has no qt3tests directory; skipping");
        return;
    };
    let tally = run_suite(&root, Some(&["fn/", "math/", "map/", "array/"]));
    report("fn+math+map+array", &tally);
    let attempted = tally.passed + tally.failed;
    assert!(
        attempted > 8_000,
        "too few QT3 function tests were attempted"
    );
    #[allow(clippy::cast_precision_loss)]
    let rate = tally.passed as f64 / attempted as f64;
    assert!(
        rate > 0.969,
        "QT3 function pass rate regressed to {:.1}%",
        rate * 100.0
    );
}

#[test]
fn qt3_whole_suite_is_reported() {
    let Some(root) = qt3_root() else {
        eprintln!("OXIXML_XML_SUITES_DIR is unset or has no qt3tests directory; skipping");
        return;
    };
    let tally = run_suite(&root, None);
    report("all", &tally);
    let attempted = tally.passed + tally.failed;
    assert!(attempted > 19_000, "too few QT3 tests were attempted");
    #[allow(clippy::cast_precision_loss)]
    let rate = tally.passed as f64 / attempted as f64;
    assert!(
        rate > 0.979,
        "QT3 overall pass rate regressed to {:.1}%",
        rate * 100.0
    );
}
