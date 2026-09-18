//! The QT3 catalogue driven against [`oxixml_xpath`].
//!
//! Every `<test-case>` whose dependencies name only an XPath (`XP…`) specification version and a
//! feature this engine implements is compiled and evaluated; the outcome is checked with
//! [`crate::assertions::check_with_serialization`]. The compile target is not always XPath 3.1:
//! a case whose (or whose test-set's) `<dependency type="spec">` pins an *exact* version —
//! `"XP20"`, never `"XP20+"` — is compiled at that version instead, so a construct 3.1 accepts
//! but 2.0/3.0 reject (or vice versa) is judged the way the catalogue actually wrote the case.
//! See `most_restrictive_version`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use oxixml_fn::xmlio::SerializationParameters;
use oxixml_fn::{CollectingTrace, FnContext};
use oxixml_infoset::{NodeRef, Tree, XdmNode};
use oxixml_xpath::{StaticContext, XPath, XPathError, XPathVersion, XpathValue};

use super::budget::{CaseOutcome, CaseRunner, progress, timeout_message};
use super::{
    Environment, apply_default_base_uri, dependencies_supported, parameter_name, parse_file,
    qt3_root, read_environment, resolve_environment, shared_environments, unavailable_reason,
};
use crate::assertions::{ErrorInfo, Verdict, check_with_serialization};
use crate::report::SuiteReport;
use crate::xdm_util::{attribute, children_named, element_children, find_child};

/// The features of the suite this engine implements.
const SUPPORTED_FEATURES: &[&str] = &[
    "higherOrderFunctions",
    // A document parsed by `oxixml_infoset::Tree` carries the infoset properties its DTD
    // declares: an `<!ATTLIST>` of type `ID`, `IDREF` or `IDREFS` reaches the XDM attribute
    // node's `is-id`/`is-idrefs` properties, which is exactly what `fn:id` and `fn:idref` read.
    // The 31 cases of `fn/id.xml` and `fn/idref.xml` that gate on this all run against
    // fixtures with an *internal* subset (`fn/id/iddtd.xml`, `app/FunctxFn/functx_book.xml`).
    "infoset-dtd",
    "namespace-axis",
    // `fn:format-dateTime`'s `$place` argument accepts an Olson timezone name, which this
    // workspace resolves — offset and daylight-saving rules alike — from its own zone data.
    "olson-timezone",
    // `fn:serialize` and the `serialization-matches` assertion are decided against
    // `oxixml_fn::xmlio`, which implements the xml, xhtml, html, text, json and adaptive output
    // methods of Serialization 3.1. The six `ser/method-*` test sets gate on this feature at
    // test-set or case level, so leaving it undeclared hid every one of them.
    "serialization",
];

/// Runs the whole catalogue against the XPath engine.
#[must_use]
pub fn run() -> SuiteReport {
    let name = super::suite_name("qt3-xpath");
    let Some(root) = qt3_root() else {
        return SuiteReport::unavailable(name, unavailable_reason());
    };
    let mut report = SuiteReport::new(name);
    let Some(catalog) = parse_file(&root.join("catalog.xml")) else {
        report.record_skip("catalog.xml did not parse");
        return report;
    };
    let Some(catalog_root) = catalog.document_element() else {
        return report;
    };
    let shared = shared_environments(&catalog_root, &root);
    let requested = super::requested_test_sets();
    let mut runner = CaseRunner::new();
    for entry in children_named(&catalog_root, "test-set") {
        let Some(file) = attribute(&entry, "file") else {
            continue;
        };
        if !super::test_set_selected(&requested, &file) {
            continue;
        }
        run_test_set(&root.join(&file), &shared, &mut report, &mut runner);
    }
    report
}

fn run_test_set(
    path: &Path,
    shared: &BTreeMap<String, Environment>,
    report: &mut SuiteReport,
    runner: &mut CaseRunner<CaseOutcome>,
) {
    let Some(tree) = parse_file(path) else {
        report.record_skip(format!("{}: did not parse", path.display()));
        return;
    };
    let Some(root) = tree.document_element() else {
        return;
    };
    let directory = path.parent().unwrap_or(Path::new("."));
    let set_name = attribute(&root, "name").unwrap_or_default();
    progress("qt3-xpath", &set_name);

    let mut environments = shared.clone();
    environments.entry("empty".to_owned()).or_default();
    for definition in children_named(&root, "environment") {
        if let Some(name) = attribute(&definition, "name") {
            environments.insert(name, read_environment(&definition));
        }
    }
    let set_supported = dependencies_supported(&root, "XP", SUPPORTED_FEATURES);
    let root_versions = exact_versions(&root);

    for case in children_named(&root, "test-case") {
        let case_name = attribute(&case, "name").unwrap_or_default();
        let id = format!("{set_name}/{case_name}");
        if !set_supported || !dependencies_supported(&case, "XP", SUPPORTED_FEATURES) {
            report.record_skip("unsupported dependency");
            continue;
        }
        let Some(test) = find_child(&case, "test") else {
            report.record_skip("no <test>");
            continue;
        };
        if attribute(&test, "file").is_some() {
            report.record_skip("external module file");
            continue;
        }
        let expression = test.string_value();
        if expression.trim().is_empty() {
            report.record_skip("empty <test>");
            continue;
        }

        let mut environment = match find_child(&case, "environment") {
            Some(node) => match attribute(&node, "ref") {
                Some(reference) => match environments.get(&reference) {
                    Some(environment) => environment.clone(),
                    None => {
                        report.record_skip("environment reference outside this test-set");
                        continue;
                    }
                },
                None => read_environment(&node),
            },
            None => Environment::default(),
        };
        apply_default_base_uri(&mut environment, path);

        let Some(result) = find_child(&case, "result") else {
            report.record_skip("no <result>");
            continue;
        };
        let Some(assertion) = element_children(&result).into_iter().next() else {
            report.record_skip("empty <result>");
            continue;
        };

        let version = most_restrictive_version(&case, &root_versions);
        // The whole case — evaluation *and* assertion checking — runs on the runner's worker
        // thread under a wall-clock budget; nothing but the verdict crosses back.
        let directory = directory.to_path_buf();
        let body = expression.clone();
        // An XPath expression has no prolog, so there is no output declaration to honour and a
        // `serialization-matches` is decided against the serializer's defaults — the same
        // "serialization options specified within the query (if any)" rule the XQuery driver
        // applies, with nothing on the "if any" side.
        let serialization = SerializationParameters::default();
        let job = Box::new(move || {
            evaluate_case(&body, &environment, &directory, version)
                .map(|outcome| check_with_serialization(&assertion, &outcome, Some(&serialization)))
        });
        // Read before the run; see `CaseRunner::still_running`.
        let contention = runner.still_running();
        let abandoned = runner.abandoned();
        let Some(verdict) = runner.run(job) else {
            let message = timeout_message(runner.budget(), contention, abandoned, &expression);
            report.record_fail(&id, message);
            continue;
        };
        match verdict {
            None => report.record_skip("environment could not be reproduced"),
            Some(Verdict::Pass) => report.record_pass(),
            Some(Verdict::Fail(reason)) => {
                report.record_fail(&id, format!("{expression} — {reason}"));
            }
            Some(Verdict::Skip) => report.record_skip("unsupported assertion kind"),
        }
    }
}

/// Parses one `XP`-prefixed spec token, e.g. `"XP20"`, into the exact [`XPathVersion`] it pins —
/// `None` for anything else, including a `"+"`-suffixed (open-ended) token, which pins nothing.
fn exact_xpath_version(token: &str) -> Option<XPathVersion> {
    match token.strip_suffix('+') {
        Some(_) => None,
        None => match token.strip_prefix("XP")? {
            "10" => Some(XPathVersion::XPath10),
            "20" => Some(XPathVersion::XPath20),
            "30" => Some(XPathVersion::XPath30),
            "31" => Some(XPathVersion::XPath31),
            _ => None,
        },
    }
}

/// Every exact `XP`-prefixed version `node`'s own `<dependency type="spec">` children pin, across
/// however many such elements it has (space-separated tokens within one are alternatives; an
/// unsatisfied — `satisfied="false"` — dependency contributes nothing, since it is a *negative*
/// constraint the coarse [`dependencies_supported`] gate has already acted on).
fn exact_versions(node: &NodeRef) -> Vec<XPathVersion> {
    let mut versions = Vec::new();
    for dependency in children_named(node, "dependency") {
        if attribute(&dependency, "type").as_deref() != Some("spec") {
            continue;
        }
        if attribute(&dependency, "satisfied").as_deref() == Some("false") {
            continue;
        }
        let value = attribute(&dependency, "value").unwrap_or_default();
        versions.extend(value.split_whitespace().filter_map(exact_xpath_version));
    }
    versions
}

/// The version to compile `case` at: the minimum of whatever exact versions it pins itself, or —
/// when the case pins none of its own — the minimum of whatever the test-set root pins; XPath 3.1
/// (this engine's fullest/default mode) when neither pins anything, matching the previous
/// unconditional behaviour for every case that does not name an exact version at all.
fn most_restrictive_version(case: &NodeRef, root_versions: &[XPathVersion]) -> XPathVersion {
    let case_versions = exact_versions(case);
    let versions: &[XPathVersion] = if case_versions.is_empty() {
        root_versions
    } else {
        &case_versions
    };
    versions.iter().copied().min().unwrap_or_default()
}

/// Everything a case needs in order to run: the static context, the context item, the variable
/// bindings still to be evaluated, the resolver, and the trees that must stay alive for the
/// duration of the evaluation.
struct Prepared {
    statik: StaticContext,
    context_item: Option<NodeRef>,
    parameters: Vec<(String, String)>,
    source_variables: Vec<(String, NodeRef)>,
    documents: Arc<super::Documents>,
    trees: Vec<Tree>,
}

fn build_context(environment: &Environment, directory: &Path) -> Option<Prepared> {
    let resolved = resolve_environment(environment, directory)?;
    let mut statik = StaticContext::new();
    for (prefix, uri) in &environment.namespaces {
        statik = statik.with_namespace(prefix.clone(), uri.clone());
    }
    if let Some(base) = &environment.base_uri {
        statik = statik.with_base_uri(base.clone());
    }
    if let Some(collation) = &environment.default_collation {
        statik = statik.with_default_collation(collation.clone());
    }
    for (name, _) in &environment.parameters {
        statik = statik.with_variable_name(parameter_name(name, &environment.namespaces));
    }
    for (name, _) in &resolved.source_variables {
        statik = statik.with_variable(name.clone());
    }
    Some(Prepared {
        statik,
        context_item: resolved.context_item,
        parameters: environment.parameters.clone(),
        source_variables: resolved.source_variables,
        documents: resolved.documents,
        trees: resolved.trees,
    })
}

fn evaluate_case(
    expression: &str,
    environment: &Environment,
    directory: &Path,
    version: XPathVersion,
) -> Option<Result<XpathValue, ErrorInfo>> {
    let Prepared {
        statik,
        context_item,
        parameters,
        source_variables,
        documents,
        trees,
    } = build_context(environment, directory)?;
    let compiled = match XPath::compile(expression, version, statik.clone()) {
        Ok(compiled) => compiled,
        Err(error) => return Some(Err(normalize_error(&error))),
    };
    // `fn:trace` writes to standard error by default, and the suite calls it hundreds of times.
    // A `<collation default="true">` environment is stated once, on the static context
    // (`build_context`); `DynamicContext::with_fn_context` carries it into a supplied function
    // context that still has the codepoint default, so `fn:compare($a, $b)` uses it here without
    // the environment being restated.
    let mut fn_context = FnContext::new().with_trace_sink(Arc::new(CollectingTrace::new()));
    for (name, format) in &environment.decimal_formats {
        fn_context = match name {
            Some(name) => fn_context.with_decimal_format(name.clone(), format.clone()),
            None => fn_context.with_default_decimal_format(format.clone()),
        };
    }
    let mut context = compiled
        .context()
        .with_fn_context(fn_context)
        .with_document_resolver(documents);
    if let Some(item) = context_item {
        context = context.with_context_item(item);
    }
    // An explicit `<context-item select="…">` supersedes a `<source role=".">`: it is the general
    // form, able to name an atomic value or an array where a source can only be a document.
    if let Some(select) = &environment.context_item_select {
        match evaluate_selector(select, version, &statik) {
            Ok(value) => {
                if let [item] = value.iter().as_slice() {
                    context = context.with_context_item(item.clone());
                }
            }
            Err(error) => return Some(Err(error)),
        }
    }
    for (name, node) in &source_variables {
        context = context.with_variable(name.clone(), node.clone());
    }
    for (name, select) in &parameters {
        let bound = match evaluate_selector(select, version, &statik) {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        };
        context = context.with_variable_name(parameter_name(name, &environment.namespaces), bound);
    }
    let outcome = compiled
        .evaluate_value_in(&context)
        .map_err(|error| normalize_error(&error));
    drop(trees);
    Some(outcome)
}

/// Evaluates one `<param select="…">` or `<context-item select="…">` expression: plain XPath in
/// the case's own namespaces, with no focus and no variables of its own (the catalogue's selectors
/// are literals and simple constructor calls, never references to the case's other bindings).
fn evaluate_selector(
    select: &str,
    version: XPathVersion,
    statik: &StaticContext,
) -> Result<XpathValue, ErrorInfo> {
    XPath::compile(select, version, statik.clone())
        .and_then(|compiled| {
            let inner = compiled.context();
            compiled.evaluate_value_in(&inner)
        })
        .map_err(|error| normalize_error(&error))
}

fn normalize_error(error: &XPathError) -> ErrorInfo {
    ErrorInfo::qualified(
        error.name().namespace(),
        error.name().local_name(),
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(
        expression: &str,
        environment: &Environment,
    ) -> Option<Result<XpathValue, ErrorInfo>> {
        evaluate_case(
            expression,
            environment,
            Path::new("."),
            XPathVersion::XPath31,
        )
    }

    #[test]
    fn evaluate_case_runs_a_simple_expression_without_an_environment() {
        let environment = Environment::default();
        let outcome = evaluate("1 + 1", &environment);
        let value = outcome
            .expect("no environment reproduction is needed")
            .expect("no compile/eval error");
        assert_eq!(value.to_string(), "2");
    }

    #[test]
    fn evaluate_case_reports_a_static_error() {
        let environment = Environment::default();
        let outcome = evaluate("1 +", &environment).expect("no environment reproduction is needed");
        assert!(outcome.is_err());
    }

    /// `fn:error` takes an arbitrary `xs:QName`, and the catalogue writes the expected code as
    /// an `EQName`, so a raised code's namespace has to survive normalization — a bare local
    /// name would neither match the case that wants it nor be distinguishable from a standard
    /// code that happened to spell the same.
    #[test]
    fn a_user_error_keeps_the_namespace_it_was_raised_in() {
        let environment = Environment::default();
        let error = evaluate(
            r#"fn:error(fn:QName("http://www.example.com/", "example:EXER3141"))"#,
            &environment,
        )
        .expect("no environment reproduction is needed")
        .expect_err("fn:error always raises");
        assert_eq!(error.code, "Q{http://www.example.com/}EXER3141");

        let standard = evaluate("1 idiv 0", &environment)
            .expect("no environment reproduction is needed")
            .expect_err("division by zero");
        assert_eq!(standard.code, "FOAR0001");
    }

    #[test]
    fn a_schema_environment_is_not_reproduced() {
        let environment = Environment {
            needs_schema: true,
            ..Environment::default()
        };
        assert!(evaluate("1", &environment).is_none());
    }

    /// Thirty-one cases of `fn/id.xml` and `fn/idref.xml` gate on the catalogue's `infoset-dtd`
    /// feature, which asks whether an `<!ATTLIST>`-declared `ID`/`IDREF`/`IDREFS` attribute
    /// reaches the XDM attribute node's `is-id`/`is-idrefs` properties — the properties
    /// `fn:id` and `fn:idref` read. Declaring a feature the engine does not have would turn
    /// thirty-one honest skips into thirty-one failures, so this pins the declaration to the
    /// *behaviour* and not to the constant: it reproduces the shape of `fn/id/iddtd.xml` and
    /// asks it what `fn-id-dtd-5` and `fn-idref-dtd-5` ask.
    #[test]
    fn a_dtd_declared_id_is_found_by_fn_id() {
        let case = Tree::parse_str(
            r#"<test-case><dependency type="feature" value="infoset-dtd"/></test-case>"#,
        )
        .expect("well-formed fixture");
        let case = case.document_element().expect("a document element");
        assert!(dependencies_supported(&case, "XP", SUPPORTED_FEATURES));

        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xpath-infoset-dtd-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("ids.xml"),
            concat!(
                "<!DOCTYPE IDS [\n",
                "<!ELEMENT IDS (elementwithid-1, elementwithidrefattr-1)>\n",
                "<!ELEMENT elementwithid-1 EMPTY>\n",
                "<!ELEMENT elementwithidrefattr-1 EMPTY>\n",
                "<!ATTLIST elementwithid-1 anId ID #REQUIRED>\n",
                "<!ATTLIST elementwithidrefattr-1 anIdRef IDREF #REQUIRED>\n",
                "]>\n",
                "<IDS><elementwithid-1 anId=\"id1\"/>",
                "<elementwithidrefattr-1 anIdRef=\"id1\"/></IDS>",
            ),
        )
        .expect("fixture file");
        let environment = Environment {
            sources: vec![crate::qt3::Source {
                role: ".".to_owned(),
                file: "ids.xml".to_owned(),
                uri: None,
            }],
            ..Environment::default()
        };

        for (expression, expected) in [
            (r#"fn:id("id1", /IDS[1])/string(@anId)"#, "id1"),
            (
                r#"fn:idref("id1", /IDS[1])/name(..)"#,
                "elementwithidrefattr-1",
            ),
        ] {
            let value = evaluate_case(expression, &environment, &directory, XPathVersion::XPath31)
                .expect("the environment reproduces")
                .expect("no compile/eval error");
            assert_eq!(value.to_string(), expected, "{expression}");
        }
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(crate::env::XML_SUITES_DIR_VAR).is_some() {
            return;
        }
        assert!(!run().is_available());
    }

    #[test]
    fn exact_xpath_version_pins_only_a_plain_version_token() {
        assert_eq!(exact_xpath_version("XP20"), Some(XPathVersion::XPath20));
        assert_eq!(exact_xpath_version("XP31"), Some(XPathVersion::XPath31));
        assert_eq!(
            exact_xpath_version("XP20+"),
            None,
            "open-ended pins nothing"
        );
        assert_eq!(exact_xpath_version("XQ20"), None, "wrong family");
    }

    #[test]
    fn most_restrictive_version_prefers_the_case_over_the_root() {
        let tree =
            Tree::parse_str(r#"<test-case><dependency type="spec" value="XP20"/></test-case>"#)
                .expect("well-formed fixture");
        let case = tree.document_element().expect("a document element");
        let root_versions = [XPathVersion::XPath31];
        assert_eq!(
            most_restrictive_version(&case, &root_versions),
            XPathVersion::XPath20
        );
    }

    #[test]
    fn most_restrictive_version_takes_the_minimum_of_several_alternatives() {
        let tree = Tree::parse_str(
            r#"<test-case><dependency type="spec" value="XP30 XP20"/></test-case>"#,
        )
        .expect("well-formed fixture");
        let case = tree.document_element().expect("a document element");
        assert_eq!(most_restrictive_version(&case, &[]), XPathVersion::XPath20);
    }

    #[test]
    fn most_restrictive_version_falls_back_to_31_when_nothing_pins_a_version() {
        let tree = Tree::parse_str(r#"<test-case/>"#).expect("well-formed fixture");
        let case = tree.document_element().expect("a document element");
        assert_eq!(most_restrictive_version(&case, &[]), XPathVersion::XPath31);
    }

    #[test]
    fn a_case_compiled_at_xpath20_rejects_a_30_only_construct() {
        // `let` bindings do not exist before XPath 3.0.
        let environment = Environment::default();
        let outcome = evaluate_case(
            "let $x := 1 return $x",
            &environment,
            Path::new("."),
            XPathVersion::XPath20,
        )
        .expect("no environment reproduction is needed");
        assert!(outcome.is_err(), "`let` must not parse under XPath 2.0");
    }

    #[test]
    fn a_dollar_role_source_is_reachable_as_a_bound_variable() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xpath-role-var-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(directory.join("in.xml"), "<a><b/></a>").expect("fixture file");
        let mut environment = Environment::default();
        environment.sources.push(super::super::Source {
            role: "$doc".to_owned(),
            file: "in.xml".to_owned(),
            uri: None,
        });
        let outcome = evaluate_case(
            "count($doc/a/b)",
            &environment,
            &directory,
            XPathVersion::XPath31,
        )
        .expect("the source reproduces")
        .expect("no evaluation error");
        assert_eq!(outcome.to_string(), "1");
        std::fs::remove_dir_all(&directory).ok();
    }

    /// fn-unparsed-text-039/fn-unparsed-text-lines-039: F&O 3.1 §14.8.1's rule that a text
    /// resource's decoded characters must be permitted XML characters, reproducing the QT3
    /// fixture `fn/unparsed-text/non-xml-character.txt` (a UTF-8 BOM followed by five NUL
    /// bytes) exactly through the driver's own environment machinery
    /// ([`Environment::resources`] → [`super::resolve_environment`] → [`super::Documents`]),
    /// so the whole `oxixml_xpath::DocumentResolver` wiring is exercised, not just the check
    /// itself.
    #[test]
    fn unparsed_text_of_a_resource_with_a_prohibited_character_is_fout1190() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xpath-unparsed-text-char-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        // UTF-8 BOM + five NUL bytes, byte-for-byte what the real QT3 fixture contains.
        std::fs::write(
            directory.join("non-xml-character.txt"),
            b"\xEF\xBB\xBF\0\0\0\0\0",
        )
        .expect("fixture file");
        let mut environment = Environment::default();
        environment.resources.push((
            "non-xml-character.txt".to_owned(),
            "non-xml-character.txt".to_owned(),
        ));
        let outcome = evaluate_case(
            "unparsed-text('non-xml-character.txt')",
            &environment,
            &directory,
            XPathVersion::XPath31,
        )
        .expect("the resource reproduces")
        .expect_err("a NUL character is not a permitted XML character");
        assert_eq!(outcome.code, "FOUT1190", "{outcome:?}");
        std::fs::remove_dir_all(&directory).ok();
    }

    /// The same resource, retrieved the way `fn:json-doc` retrieves it: F&O 3.1 §14.8.1's
    /// permitted-XML-character rule is `fn:unparsed-text`'s alone
    /// ([`oxixml_xpath`]'s `resources.rs` says so explicitly), so raw JSON-flavoured retrieval
    /// of the very same bytes must not raise it — only fail on whatever `fn:parse-json` itself
    /// makes of a string full of NUL characters (not valid JSON, `FOJS0001`).
    #[test]
    fn json_doc_of_the_same_resource_is_not_subject_to_the_xml_character_rule() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xpath-json-doc-char-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("non-xml-character.txt"),
            b"\xEF\xBB\xBF\0\0\0\0\0",
        )
        .expect("fixture file");
        let mut environment = Environment::default();
        environment.resources.push((
            "non-xml-character.txt".to_owned(),
            "non-xml-character.txt".to_owned(),
        ));
        let outcome = evaluate_case(
            "json-doc('non-xml-character.txt')",
            &environment,
            &directory,
            XPathVersion::XPath31,
        )
        .expect("the resource reproduces")
        .expect_err("five NUL bytes is not valid JSON");
        assert_eq!(
            outcome.code, "FOJS0001",
            "raw retrieval must reach the JSON parser, not FOUT1190: {outcome:?}"
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    /// fn-unparsed-text-available-038: `fn:unparsed-text-available` folds the same `FOUT1190`
    /// into `false` rather than raising it — it is a predicate, never an error.
    #[test]
    fn unparsed_text_available_is_false_for_a_resource_with_a_prohibited_character() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-qt3-xpath-unparsed-text-avail-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("non-xml-character.txt"),
            b"\xEF\xBB\xBF\0\0\0\0\0",
        )
        .expect("fixture file");
        let mut environment = Environment::default();
        environment.resources.push((
            "non-xml-character.txt".to_owned(),
            "non-xml-character.txt".to_owned(),
        ));
        let outcome = evaluate_case(
            "unparsed-text-available('non-xml-character.txt')",
            &environment,
            &directory,
            XPathVersion::XPath31,
        )
        .expect("the resource reproduces")
        .expect("unparsed-text-available never raises");
        assert_eq!(outcome.to_string(), "false", "{outcome}");
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn a_default_decimal_format_from_the_environment_is_used() {
        let mut environment = Environment::default();
        environment.decimal_formats.push((
            None,
            oxixml_fn::numeric::DecimalFormat::default()
                .with_decimal_separator(',')
                .with_grouping_separator('.'),
        ));
        let outcome = evaluate("format-number(1.5, '0,0')", &environment)
            .expect("no environment reproduction is needed")
            .expect("no evaluation error");
        assert_eq!(outcome.to_string(), "1,5");
    }
}
